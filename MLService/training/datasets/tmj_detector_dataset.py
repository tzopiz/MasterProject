#!/usr/bin/env python3
"""
TMJ Detector Dataset

Dataset for training TMJ coordinate regression model.
Loads full CBCT scans and ROI annotations, applies augmentations.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pydicom
import torch
from scipy import ndimage
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TMJDetectorDataset(Dataset):
    """
    Dataset for TMJ coordinate regression

    Returns:
        volume: (C, D, H, W) - downsampled 3D volume
        target: (6,) - normalized coordinates [left_z, left_y, left_x, right_z, right_y, right_x]
    """

    def __init__(
        self,
        annotations_dir: str,
        dataset_dir: str,
        downsample_factor: int = 6,
        is_train: bool = True,
        split_ratio: float = 0.8,
        augment: bool = True,
        cache_volumes: bool = False,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.dataset_dir = Path(dataset_dir)
        self.downsample_factor = downsample_factor
        self.is_train = is_train
        self.augment = augment and is_train
        self.cache_volumes = cache_volumes

        # Load annotations
        self.annotations = self._load_annotations()

        # Split train/val
        self._split_data(split_ratio)

        # Cache
        self.volume_cache = {}

        logger.info(
            f"TMJDetectorDataset: {len(self.annotations)} samples ({'train' if is_train else 'val'})"
        )

    def _load_annotations(self) -> List[Dict]:
        """Load all annotation files"""
        annotation_files = sorted(list(self.annotations_dir.glob("*_rois.json")))

        annotations = []
        for ann_file in annotation_files:
            with open(ann_file, "r") as f:
                ann = json.load(f)

            study_dir = self.dataset_dir / ann["scan_id"]
            if study_dir.exists():
                annotations.append(
                    {
                        "study_id": ann["scan_id"],
                        "study_dir": study_dir,
                        "left_center": ann["left_tmj"]["center"],
                        "right_center": ann["right_tmj"]["center"],
                        "original_shape": ann["original_shape"],
                    }
                )

        return annotations

    def _split_data(self, split_ratio: float):
        """Split into train/val"""
        random.seed(42)
        random.shuffle(self.annotations)

        split_idx = int(len(self.annotations) * split_ratio)

        if self.is_train:
            self.annotations = self.annotations[:split_idx]
        else:
            self.annotations = self.annotations[split_idx:]

    def _load_dicom_volume(self, dicom_dir: Path) -> np.ndarray:
        """Load DICOM series"""
        # Check cache
        study_id = dicom_dir.name
        if self.cache_volumes and study_id in self.volume_cache:
            return self.volume_cache[study_id]

        # Load DICOM files
        dicom_files = list(dicom_dir.glob("*.dcm"))
        slices = [pydicom.dcmread(f) for f in dicom_files]
        slices.sort(key=lambda x: int(x.InstanceNumber))

        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        # Cache if needed
        if self.cache_volumes:
            self.volume_cache[study_id] = volume

        return volume

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1]"""
        # Clip extreme values
        p2, p98 = np.percentile(volume, [2, 98])
        volume = np.clip(volume, p2, p98)

        # Normalize
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())

        return volume

    def _downsample_volume(self, volume: np.ndarray) -> np.ndarray:
        """Downsample volume for faster training"""
        # Using zoom for downsampling
        zoom_factors = [1 / self.downsample_factor] * 3
        downsampled = ndimage.zoom(volume, zoom_factors, order=1)
        return downsampled

    def _normalize_coordinates(self, coords: List[int], original_shape: List[int]) -> np.ndarray:
        """Normalize coordinates to [0, 1] relative to volume shape"""
        coords = np.array(coords, dtype=np.float32)
        original_shape = np.array(original_shape, dtype=np.float32)

        # Normalize to [0, 1]
        normalized = coords / original_shape

        # Adjust for downsampling (coordinates are in original space)
        # After downsampling, coordinates need to be scaled
        normalized = normalized  # Already normalized to [0,1]

        return normalized

    def _denormalize_coordinates(
        self, normalized_coords: np.ndarray, downsampled_shape: np.ndarray
    ) -> np.ndarray:
        """Convert normalized coordinates back to downsampled volume space"""
        return normalized_coords * downsampled_shape

    def _augment_volume(
        self, volume: np.ndarray, left_coords: np.ndarray, right_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations to volume and adjust coordinates"""

        # 1. Random intensity shift
        if random.random() < 0.5:
            shift = random.uniform(-0.1, 0.1)
            volume = np.clip(volume + shift, 0, 1)

        # 2. Random intensity scale
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)

        # 3. Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, volume.shape).astype(np.float32)
            volume = np.clip(volume + noise, 0, 1)

        # 4. Random flip (left-right)
        # NOTE: This also swaps left/right TMJ!
        if random.random() < 0.5:
            volume = np.flip(volume, axis=2).copy()  # Flip X axis

            # Swap and flip X coordinates
            left_coords_new = right_coords.copy()
            right_coords_new = left_coords.copy()

            # Flip X coordinate (axis 2)
            left_coords_new[2] = 1.0 - left_coords_new[2]
            right_coords_new[2] = 1.0 - right_coords_new[2]

            left_coords = left_coords_new
            right_coords = right_coords_new

        # 5. Random shift (spatial translation)
        if random.random() < 0.5:
            # Small spatial shift
            max_shift = 0.05  # 5% of volume size
            shift_z = random.uniform(-max_shift, max_shift)
            shift_y = random.uniform(-max_shift, max_shift)
            shift_x = random.uniform(-max_shift, max_shift)

            # Apply shift to volume (using affine transform)
            shift_pixels = np.array(
                [
                    shift_z * volume.shape[0],
                    shift_y * volume.shape[1],
                    shift_x * volume.shape[2],
                ]
            )

            # Shift volume
            volume = ndimage.shift(volume, shift_pixels, order=1, mode="constant", cval=0)

            # Adjust coordinates
            left_coords += np.array([shift_z, shift_y, shift_x])
            right_coords += np.array([shift_z, shift_y, shift_x])

            # Clip to valid range
            left_coords = np.clip(left_coords, 0, 1)
            right_coords = np.clip(right_coords, 0, 1)

        return volume, left_coords, right_coords

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load volume
        volume = self._load_dicom_volume(ann["study_dir"])

        # Normalize intensity
        volume = self._normalize_volume(volume)

        # Downsample
        volume = self._downsample_volume(volume)
        np.array(volume.shape)

        # Normalize coordinates
        left_coords = self._normalize_coordinates(ann["left_center"], ann["original_shape"])
        right_coords = self._normalize_coordinates(ann["right_center"], ann["original_shape"])

        # Augmentation
        if self.augment:
            volume, left_coords, right_coords = self._augment_volume(
                volume, left_coords, right_coords
            )

        # Combine coordinates into single target
        target = np.concatenate([left_coords, right_coords])  # (6,)

        # Convert to tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W) float32
        target = torch.from_numpy(target).float()  # (6,) float32

        return volume, target


def get_dataloaders(
    annotations_dir: str,
    dataset_dir: str,
    batch_size: int = 2,
    num_workers: int = 0,
    downsample_factor: int = 6,
    split_ratio: float = 0.8,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders"""

    train_dataset = TMJDetectorDataset(
        annotations_dir=annotations_dir,
        dataset_dir=dataset_dir,
        downsample_factor=downsample_factor,
        is_train=True,
        split_ratio=split_ratio,
        augment=True,
    )

    val_dataset = TMJDetectorDataset(
        annotations_dir=annotations_dir,
        dataset_dir=dataset_dir,
        downsample_factor=downsample_factor,
        is_train=False,
        split_ratio=split_ratio,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
