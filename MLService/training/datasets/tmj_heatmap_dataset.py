#!/usr/bin/env python3
"""
TMJ Heatmap Detection Dataset.

Each item:
  volume: (1, D, H, W) float32  — normalized downsampled CBCT
  target: (2, D, H, W) float32  — ch0=left heatmap, ch1=right heatmap
"""
from __future__ import annotations
import json
import random
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from training.utils.heatmap import make_heatmap

logger = logging.getLogger(__name__)

DOWNSAMPLE_SHAPE = (96, 128, 128)


def _load_dicom_volume(study_dir: Path) -> np.ndarray:
    files = sorted(study_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm in {study_dir}")
    slices = [pydicom.dcmread(str(f)) for f in files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(s.InstanceNumber))
    planes = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        planes.append(arr)
    return np.stack(planes, axis=0)


def _normalize(volume: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(volume, [2, 98])
    volume = np.clip(volume, p2, p98)
    denom = p98 - p2
    return ((volume - p2) / denom if denom > 0 else np.zeros_like(volume)).astype(np.float32)


def _downsample(volume: np.ndarray, factor: int = 6) -> np.ndarray:
    zoom = [1.0 / factor] * 3
    return ndimage.zoom(volume, zoom, order=1).astype(np.float32)


def _augment(
    volume: np.ndarray,
    left_orig: List[int],
    right_orig: List[int],
) -> Tuple[np.ndarray, List[int], List[int]]:
    # 1. Intensity jitter
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)
        beta  = random.uniform(-0.05, 0.05)
        volume = np.clip(alpha * volume + beta, 0.0, 1.0)

    # 2. Gaussian noise
    if random.random() < 0.4:
        volume = np.clip(volume + np.random.normal(0, 0.01, volume.shape), 0.0, 1.0).astype(np.float32)

    # 3. X-axis flip → swap left/right + flip x coordinate
    if random.random() < 0.5:
        volume = np.flip(volume, axis=2).copy()
        ds_W = volume.shape[2]
        left_new  = [right_orig[0], right_orig[1], (ds_W - 1 - right_orig[2] // 6) * 6]
        right_new = [left_orig[0],  left_orig[1],  (ds_W - 1 - left_orig[2] // 6) * 6]
        left_orig, right_orig = left_new, right_new

    # 4. Rotation ±10°
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        axes  = random.choice([(0, 1), (0, 2), (1, 2)])
        volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1, mode="nearest")
        D, H, W = volume.shape
        centers = [D / 2, H / 2, W / 2]
        a, b = axes
        c = np.cos(np.radians(angle)); s = np.sin(np.radians(angle))
        dim_sizes = [D * 6, H * 6, W * 6]  # back to original space
        for orig in (left_orig, right_orig):
            da = (orig[a] / 6 - centers[a])
            db = (orig[b] / 6 - centers[b])
            orig[a] = int((da * c - db * s + centers[a]) * 6)
            orig[b] = int((da * s + db * c + centers[b]) * 6)
            orig[a] = min(max(0, orig[a]), dim_sizes[a] - 1)
            orig[b] = min(max(0, orig[b]), dim_sizes[b] - 1)

    return volume.astype(np.float32), left_orig, right_orig


class TMJHeatmapDataset(Dataset):
    """
    Dataset for 3D heatmap-based TMJ detection.

    Parameters
    ----------
    study_ids : list of str
        Study IDs to include (from detector_split.json).
    annotations_dir : str
        Directory with *_rois.json files.
    volume_loader : callable, optional
        Function (study_id: str) -> np.ndarray (D,H,W) normalized float32.
        Defaults to DICOM loader.
    dataset_dir : str, optional
        Root of DICOM dataset (used by default loader).
    sigma : float
        Gaussian sigma in downsampled voxels.
    downsample_factor : int
        Spatial downsampling factor.
    is_train : bool
        Enable augmentation when True.
    """

    def __init__(
        self,
        study_ids: List[str],
        annotations_dir: str,
        volume_loader: Optional[Callable] = None,
        dataset_dir: str = "data/dataset_cbct_public",
        sigma: float = 3.0,
        downsample_factor: int = 6,
        is_train: bool = True,
    ):
        self.sigma             = sigma
        self.downsample_factor = downsample_factor
        self.is_train          = is_train
        self.ann_dir           = Path(annotations_dir)
        self.dataset_dir       = Path(dataset_dir)

        if volume_loader is not None:
            self._load_volume = volume_loader
        else:
            def _loader(study_id: str) -> np.ndarray:
                raw = _load_dicom_volume(self.dataset_dir / study_id)
                normalized = _normalize(raw)
                return _downsample(normalized, downsample_factor)
            self._load_volume = _loader

        self.records: List[Dict] = []
        for sid in study_ids:
            ann_path = self.ann_dir / f"{sid}_rois.json"
            if not ann_path.exists():
                logger.warning("Annotation missing: %s — skipped", ann_path)
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            self.records.append({
                "study_id":     ann["scan_id"],
                "left_center":  list(ann["left_tmj"]["center"]),
                "right_center": list(ann["right_tmj"]["center"]),
                "original_shape": ann["original_shape"],
            })

        logger.info("TMJHeatmapDataset: %d samples (%s)", len(self.records),
                    "train" if is_train else "val/test")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vol = self._load_volume(rec["study_id"])

        left  = list(rec["left_center"])
        right = list(rec["right_center"])

        if self.is_train:
            vol, left, right = _augment(vol, left, right)

        ds_shape = vol.shape

        hm_left  = make_heatmap(ds_shape, left,  self.sigma, self.downsample_factor)
        hm_right = make_heatmap(ds_shape, right, self.sigma, self.downsample_factor)

        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0)
        hm_tensor  = torch.from_numpy(np.stack([hm_left, hm_right])).float()
        return vol_tensor, hm_tensor


def get_heatmap_dataloaders(
    split_json: str = "data/detector_split.json",
    annotations_dir: str = "data/roi_annotations",
    dataset_dir: str = "data/dataset_cbct_public",
    sigma: float = 3.0,
    downsample_factor: int = 6,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from split JSON."""
    with open(split_json) as f:
        split = json.load(f)

    train_ds = TMJHeatmapDataset(
        split["train"], annotations_dir, dataset_dir=dataset_dir,
        sigma=sigma, downsample_factor=downsample_factor, is_train=True,
    )
    val_ds = TMJHeatmapDataset(
        split["val"], annotations_dir, dataset_dir=dataset_dir,
        sigma=sigma, downsample_factor=downsample_factor, is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader
