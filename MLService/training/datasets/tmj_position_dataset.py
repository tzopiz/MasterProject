#!/usr/bin/env python3
"""
TMJ Position Classification Dataset

Loads CBCT DICOM series, normalises intensities, downsamples the volume,
then extracts a central crop of fixed size (D×H×W).

MVP limitation: central crop is taken from the whole volume without any
ROI detector. A note about transitioning to detector-based crops is in
MLService/docs/README.md (issue #67).

Each __getitem__ returns:
    volume_tensor : (1, D, H, W) float32, values in [0, 1]
    labels_tensor : (4,) int64   — [sag_right, sag_left, fr_right, fr_left], each 0..2
"""

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from training.tmj_position_label_table import (
    binarize_labels,
    build_index,
    split_by_patient,
)
from training.utils.volume_aug_3d import augment_binary_volume_train

logger = logging.getLogger(__name__)

# Default crop size (in downsampled voxels)
DEFAULT_CROP = (96, 128, 128)  # (D, H, W)
# Max random shift during training augmentation (voxels, in downsampled space)
DEFAULT_SHIFT = 5


# ------------------------------------------------------------------
# Shared normalization utility
# ------------------------------------------------------------------


def _normalize_volume_percentile(volume: np.ndarray) -> np.ndarray:
    """Clip to [2nd, 98th] percentile and scale to [0, 1]."""
    p2, p98 = np.percentile(volume, [2, 98])
    volume = np.clip(volume, p2, p98)
    denom = p98 - p2
    if denom > 0:
        volume = (volume - p2) / denom
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32)


class TMJPositionClassificationDataset(Dataset):
    """
    Dataset for TMJ position classification (4-head multi-class).

    Parameters
    ----------
    records : list of dict
        Output of training.tmj_position_label_table.build_index().
    dataset_root : str | Path
        Root directory of the CBCT dataset (study_* folders must be here).
    downsample_factor : int
        Factor applied uniformly to all three spatial axes via scipy.ndimage.zoom.
    crop_size : (D, H, W)
        Central crop size in downsampled voxel coordinates.
    is_train : bool
        Enables random crop shift augmentation when True.
    shift_voxels : int
        Maximum random offset (±) in each axis during training.
    """

    def __init__(
        self,
        records: List[Dict],
        dataset_root,
        downsample_factor: int = 6,
        crop_size: Tuple[int, int, int] = DEFAULT_CROP,
        is_train: bool = True,
        shift_voxels: int = DEFAULT_SHIFT,
    ):
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.downsample_factor = downsample_factor
        self.crop_size = crop_size
        self.is_train = is_train
        self.shift_voxels = shift_voxels

        logger.info(
            "TMJPositionClassificationDataset: %d samples (%s) crop=%s ds=%d",
            len(records),
            "train" if is_train else "val",
            crop_size,
            downsample_factor,
        )

    # ------------------------------------------------------------------
    # DICOM loading
    # ------------------------------------------------------------------

    def _load_dicom_volume(self, dicom_dir: Path) -> np.ndarray:
        """Load all .dcm files, sort by Z (ImagePositionPatient[2]), apply HU rescale."""
        dcm_files = sorted(dicom_dir.glob("*.dcm"))
        if not dcm_files:
            raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

        slices = [pydicom.dcmread(str(f)) for f in dcm_files]

        # Sort by physical Z position (same as DICOMProcessor.load_series)
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        except (AttributeError, IndexError, TypeError):
            slices.sort(key=lambda s: int(s.InstanceNumber))

        # Stack slices with HU conversion
        planes = []
        for s in slices:
            arr = s.pixel_array.astype(np.float32)
            slope = float(getattr(s, "RescaleSlope", 1.0))
            intercept = float(getattr(s, "RescaleIntercept", 0.0))
            planes.append(arr * slope + intercept)

        return np.stack(planes, axis=0)  # (D, H, W) float32, HU

    # ------------------------------------------------------------------
    # Intensity normalisation (matches TMJDetectorDataset)
    # ------------------------------------------------------------------

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Clip to [2nd, 98th] percentile and scale to [0, 1]."""
        return _normalize_volume_percentile(volume)

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    def _downsample_volume(self, volume: np.ndarray) -> np.ndarray:
        factor = 1.0 / self.downsample_factor
        return ndimage.zoom(volume, [factor, factor, factor], order=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Central crop with optional random shift (training augmentation)
    # ------------------------------------------------------------------

    def _central_crop(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract a central crop of self.crop_size from volume.

        If the volume is smaller than the crop along any axis, pad with zeros
        before cropping. During training a random shift of ±shift_voxels is
        applied; the shift does not affect the class labels.
        """
        D, H, W = volume.shape
        cD, cH, cW = self.crop_size

        # Pad if volume is smaller than crop
        pad_d = max(0, cD - D)
        pad_h = max(0, cH - H)
        pad_w = max(0, cW - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(
                volume,
                [
                    (pad_d // 2, pad_d - pad_d // 2),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ],
                mode="constant",
                constant_values=0,
            )
            D, H, W = volume.shape

        # Central start indices
        sd = (D - cD) // 2
        sh = (H - cH) // 2
        sw = (W - cW) // 2

        # Random shift augmentation (train only)
        if self.is_train and self.shift_voxels > 0:
            sv = self.shift_voxels
            sd += random.randint(-sv, sv)
            sh += random.randint(-sv, sv)
            sw += random.randint(-sv, sv)

        # Clamp to valid range
        sd = max(0, min(sd, D - cD))
        sh = max(0, min(sh, H - cH))
        sw = max(0, min(sw, W - cW))

        return volume[sd : sd + cD, sh : sh + cH, sw : sw + cW]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        dicom_dir = Path(rec["dicom_dir"])

        volume = self._load_dicom_volume(dicom_dir)
        volume = self._normalize_volume(volume)
        volume = self._downsample_volume(volume)
        volume = self._central_crop(volume)

        # (1, D, H, W) float32
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)

        # (4,) int64  — [sag_right, sag_left, fr_right, fr_left]
        labels_tensor = torch.tensor(
            [rec["sag_right"], rec["sag_left"], rec["fr_right"], rec["fr_left"]],
            dtype=torch.long,
        )

        return volume_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_position_dataloaders(
    manifest_path: str = "data/dataset_cbct_public/manifest_private.json",
    labels_path: str = "data/tmj_position_labels.json",
    dataset_root: str = "data/dataset_cbct_public",
    batch_size: int = 2,
    num_workers: int = 0,
    downsample_factor: int = 6,
    crop_size: Tuple[int, int, int] = DEFAULT_CROP,
    split_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training index, split by patient, return (train_loader, val_loader).
    """
    all_records = build_index(
        manifest_path=manifest_path,
        labels_path=labels_path,
        dataset_root=dataset_root,
    )
    train_records, val_records = split_by_patient(all_records, split_ratio=split_ratio)

    train_ds = TMJPositionClassificationDataset(
        train_records,
        dataset_root,
        downsample_factor=downsample_factor,
        crop_size=crop_size,
        is_train=True,
    )
    val_ds = TMJPositionClassificationDataset(
        val_records,
        dataset_root,
        downsample_factor=downsample_factor,
        crop_size=crop_size,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Binary classification dataset (detector-based NIfTI crops)
# ---------------------------------------------------------------------------


class TMJBinaryPositionDataset(Dataset):
    """
    Dataset for binary TMJ position classification (central vs non-central).

    Loads pre-generated 128×128×128 NIfTI crops produced by
    tools/auto_crop_from_detector.py. Each sample represents one condyle
    side (left or right) with two binary labels: sagittal and frontal.

    Parameters
    ----------
    records : list of dict
        Output of training.tmj_position_label_table.binarize_labels().
        Each record must have keys: crop_path, sag, fr.
    is_train : bool
        When True, applies ``train_augment_mode`` (validation always has no aug).
    sagittal_only : bool
        If True, returns only the sagittal binary label as float32 scalar
        (for BCE / focal on a single head).
    train_augment_mode : str
        ``none`` | ``flip_only`` | ``strong`` — see ``training.utils.volume_aug_3d``.
        Ignored when ``is_train`` is False.
    """

    def __init__(
        self,
        records: List[Dict],
        is_train: bool = False,
        sagittal_only: bool = False,
        train_augment_mode: str = "flip_only",
    ) -> None:
        self.records = records
        self.is_train = is_train
        self.sagittal_only = sagittal_only
        self.train_augment_mode = train_augment_mode
        logger.info(
            "TMJBinaryPositionDataset: %d samples (%s)%s aug=%s",
            len(records),
            "train" if is_train else "val",
            " sagittal_only" if sagittal_only else "",
            train_augment_mode if is_train else "none",
        )

    def _load_nifti(self, path: str) -> np.ndarray:
        import nibabel as nib

        img = nib.load(path)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        return arr  # (D, H, W)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Clip to [2nd, 98th] percentile, scale to [0, 1]."""
        return _normalize_volume_percentile(volume)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        volume = self._load_nifti(rec["crop_path"])
        volume = self._normalize(volume)

        if self.is_train:
            volume = augment_binary_volume_train(volume, self.train_augment_mode)

        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
        if self.sagittal_only:
            labels_tensor = torch.tensor(float(rec["sag"]), dtype=torch.float32)
        else:
            labels_tensor = torch.tensor([rec["sag"], rec["fr"]], dtype=torch.long)  # (2,)
        return volume_tensor, labels_tensor


def make_binary_position_loaders(
    train_records: List[Dict],
    val_records: List[Dict],
    batch_size: int = 8,
    num_workers: int = 0,
    sagittal_only: bool = False,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    train_augment_mode: str = "flip_only",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from pre-split binary record lists.

    Used by K-fold CV and notebooks; keeps ``get_binary_position_dataloaders``
    as the thin wrapper over ``build_index`` → ``binarize_labels`` → split.

    There is **no separate test split** here — only train and validation loaders.
    """
    train_ds = TMJBinaryPositionDataset(
        train_records,
        is_train=True,
        sagittal_only=sagittal_only,
        train_augment_mode=train_augment_mode,
    )
    val_ds = TMJBinaryPositionDataset(val_records, is_train=False, sagittal_only=sagittal_only)

    loader_kw: Dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    if worker_init_fn is not None:
        loader_kw["worker_init_fn"] = worker_init_fn

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader


def get_binary_position_dataloaders(
    crop_dir: str,
    manifest_path: str = "data/dataset_cbct_public/manifest_private.json",
    labels_path: str = "data/tmj_position_labels.json",
    dataset_root: str = "data/dataset_cbct_public",
    batch_size: int = 4,
    num_workers: int = 0,
    split_ratio: float = 0.8,
    sagittal_only: bool = False,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    train_augment_mode: str = "flip_only",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build binary classification dataloaders from detector-generated crops.

    Workflow:
        build_index() → binarize_labels() → split_by_patient()
        → TMJBinaryPositionDataset (train/val)

    Args:
        crop_dir: Root of detector crops, e.g. "data/detector_crops"
        manifest_path, labels_path, dataset_root: Same as get_position_dataloaders.
        batch_size: Samples per batch (each sample = one condyle side).
        num_workers: DataLoader workers.
        split_ratio: Fraction of patients for training.
        sagittal_only: If True, only sagittal binary label is returned per sample.
        worker_init_fn: Optional ``worker_init_fn`` for reproducible augmentations.
        train_augment_mode: Augmentations on **train** only (``none`` / ``flip_only`` / ``strong``).

    Returns:
        ``(train_loader, val_loader)`` — **no test loader**; use nested CV or a
        frozen holdout workflow elsewhere if you need a final test set.
    """
    all_records = build_index(
        manifest_path=manifest_path,
        labels_path=labels_path,
        dataset_root=dataset_root,
    )
    binary_records = binarize_labels(all_records, crop_dir)
    train_records, val_records = split_by_patient(binary_records, split_ratio=split_ratio)

    return make_binary_position_loaders(
        train_records,
        val_records,
        batch_size=batch_size,
        num_workers=num_workers,
        sagittal_only=sagittal_only,
        worker_init_fn=worker_init_fn,
        train_augment_mode=train_augment_mode,
    )
