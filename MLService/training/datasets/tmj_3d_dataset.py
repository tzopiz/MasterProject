"""
3D TMJ Dataset for volumetric segmentation

Loads 3D NIfTI volumes and their corresponding masks for training 3D U-Net.
Supports both full volumes and slice-by-slice processing.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TMJ3DDataset(Dataset):
    """
    Dataset for 3D TMJ segmentation.

    Expects pairs of files:
    - volume: {name}.nii.gz
    - mask: {name}_mask.nii.gz

    Example:
        data/processed_crops/
            1_left.nii.gz
            1_left_mask.nii.gz
            1_right.nii.gz
            1_right_mask.nii.gz
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        target_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Args:
            data_dir: Directory with .nii.gz files
            transform: Optional transforms (augmentation)
            normalize: Normalize volumes to [0, 1]
            target_shape: Resize to this shape (D, H, W), e.g. (128, 128, 128)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.normalize = normalize
        self.target_shape = target_shape

        # Find all volume files (not masks)
        self.volume_files = sorted(
            [f for f in self.data_dir.glob("*.nii.gz") if "_mask" not in f.name]
        )

        if not self.volume_files:
            raise ValueError(f"No .nii.gz files found in {data_dir}")

        logger.info(f"Found {len(self.volume_files)} volumes in {data_dir}")

        # Verify each volume has a corresponding mask
        self.valid_pairs = []
        for vol_path in self.volume_files:
            mask_path = vol_path.parent / f"{vol_path.stem.replace('.nii', '')}_mask.nii.gz"
            if mask_path.exists():
                self.valid_pairs.append((vol_path, mask_path))
            else:
                logger.warning(f"No mask found for {vol_path.name}, skipping")

        if not self.valid_pairs:
            raise ValueError(f"No valid volume-mask pairs found in {data_dir}")

        logger.info(f"Loaded {len(self.valid_pairs)} valid volume-mask pairs")

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            volume: [1, D, H, W] tensor
            mask: [1, D, H, W] tensor
        """
        vol_path, mask_path = self.valid_pairs[idx]

        # Load NIfTI files
        volume = self._load_nifti(vol_path)
        mask = self._load_nifti(mask_path)

        # Normalize volume
        if self.normalize:
            volume = self._normalize(volume)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # Resize if needed
        if self.target_shape is not None:
            volume = self._resize(volume, self.target_shape)
            mask = self._resize(mask, self.target_shape)

        # Apply transforms (augmentation)
        if self.transform:
            volume, mask = self.transform(volume, mask)

        # Convert to torch tensors [1, D, H, W]
        volume = torch.from_numpy(volume).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return volume, mask

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file and return as numpy array"""
        try:
            nii = nib.load(str(path))
            data = nii.get_fdata()
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        return volume

    def _resize(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target shape"""
        from scipy.ndimage import zoom

        current_shape = volume.shape
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]

        # Use order=1 for masks (nearest neighbor-like), order=3 for volumes
        order = 0 if volume.max() <= 1.0 and len(np.unique(volume)) < 10 else 1

        resized = zoom(volume, zoom_factors, order=order)
        return resized

    def get_sample_info(self, idx: int) -> dict:
        """Get information about a sample"""
        vol_path, mask_path = self.valid_pairs[idx]

        # Load to get shapes
        volume = self._load_nifti(vol_path)
        mask = self._load_nifti(mask_path)

        return {
            "volume_file": vol_path.name,
            "mask_file": mask_path.name,
            "original_shape": volume.shape,
            "volume_range": (volume.min(), volume.max()),
            "mask_unique_values": np.unique(mask),
            "mask_coverage": (mask > 0).sum() / mask.size,
        }


class TMJ3DSliceDataset(Dataset):
    """
    2D slice dataset from 3D volumes.

    Extracts 2D slices from 3D volumes for slice-by-slice training.
    Useful for 2D models or when GPU memory is limited.
    """

    def __init__(
        self,
        data_dir: str,
        axis: int = 2,  # 0=sagittal, 1=coronal, 2=axial
        transform: Optional[Callable] = None,
        normalize: bool = True,
        min_mask_ratio: float = 0.01,  # Skip slices with < 1% mask
    ):
        """
        Args:
            data_dir: Directory with .nii.gz files
            axis: Which axis to slice along (0, 1, or 2)
            transform: Optional transforms
            normalize: Normalize to [0, 1]
            min_mask_ratio: Minimum ratio of mask pixels to include slice
        """
        self.data_dir = Path(data_dir)
        self.axis = axis
        self.transform = transform
        self.normalize = normalize
        self.min_mask_ratio = min_mask_ratio

        # Load all volumes first
        volume_files = sorted([f for f in self.data_dir.glob("*.nii.gz") if "_mask" not in f.name])

        # Build slice index: [(vol_path, mask_path, slice_idx), ...]
        self.slice_index = []

        for vol_path in volume_files:
            mask_path = vol_path.parent / f"{vol_path.stem.replace('.nii', '')}_mask.nii.gz"

            if not mask_path.exists():
                continue

            # Load volume to get number of slices
            volume = nib.load(str(vol_path)).get_fdata()
            mask = nib.load(str(mask_path)).get_fdata()

            n_slices = volume.shape[self.axis]

            # Add each slice if it has enough mask
            for slice_idx in range(n_slices):
                # Get mask slice
                if self.axis == 0:
                    mask_slice = mask[slice_idx, :, :]
                elif self.axis == 1:
                    mask_slice = mask[:, slice_idx, :]
                else:  # axis == 2
                    mask_slice = mask[:, :, slice_idx]

                # Check if slice has enough mask
                mask_ratio = (mask_slice > 0).sum() / mask_slice.size

                if mask_ratio >= self.min_mask_ratio:
                    self.slice_index.append((vol_path, mask_path, slice_idx))

        logger.info(
            f"Created slice dataset with {len(self.slice_index)} slices "
            f"from {len(volume_files)} volumes (axis={axis})"
        )

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: [1, H, W] tensor
            mask: [1, H, W] tensor
        """
        vol_path, mask_path, slice_idx = self.slice_index[idx]

        # Load volumes
        volume = nib.load(str(vol_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()

        # Extract slice
        if self.axis == 0:
            image_slice = volume[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :]
        elif self.axis == 1:
            image_slice = volume[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :]
        else:  # axis == 2
            image_slice = volume[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]

        # Normalize
        if self.normalize:
            vmin, vmax = image_slice.min(), image_slice.max()
            if vmax > vmin:
                image_slice = (image_slice - vmin) / (vmax - vmin)

        # Binarize mask
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Apply transforms
        if self.transform:
            image_slice, mask_slice = self.transform(image_slice, mask_slice)

        # Convert to tensors [1, H, W]
        image = torch.from_numpy(image_slice.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask_slice).unsqueeze(0)

        return image, mask


if __name__ == "__main__":
    # Test dataset
    print("Testing TMJ3DDataset...")

    dataset = TMJ3DDataset(data_dir="data/processed_crops", target_shape=(128, 128, 128))

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Get first sample
        volume, mask = dataset[0]

        print(f"Volume shape: {volume.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"Mask unique: {torch.unique(mask)}")

        # Get info
        info = dataset.get_sample_info(0)
        print("\nSample info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\n✅ 3D Dataset test passed!")

    # Test slice dataset
    print("\n\nTesting TMJ3DSliceDataset...")

    slice_dataset = TMJ3DSliceDataset(
        data_dir="data/processed_crops",
        axis=2,  # axial slices
        min_mask_ratio=0.01,
    )

    print(f"Slice dataset size: {len(slice_dataset)}")

    if len(slice_dataset) > 0:
        image, mask = slice_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")

        print("\n✅ Slice Dataset test passed!")
