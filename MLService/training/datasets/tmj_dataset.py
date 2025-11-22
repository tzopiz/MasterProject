import logging
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib

from services.dicom_processor import DICOMProcessor

logger = logging.getLogger(__name__)

class TMJDataset(Dataset):
    """
    Dataset for TMJ segmentation.
    Supports:
    1. Directory of 2D Images (.png/.jpg) + Masks
    2. Directory of 3D NIfTI volumes (.nii.gz) + Masks (sliced on-the-fly)
    3. Specific list of files (via file_list param)
    """
    def __init__(self, images_dir, masks_dir, transform=None, file_list=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.dicom_processor = DICOMProcessor()
        self.is_volume = False
        self.samples = [] # List of (image_data, mask_data) or (vol_idx, slice_idx)
        self.volumes = [] # Cache for 3D volumes
        
        if file_list is not None:
            # Use provided files
            # Determine if they are NIfTI or Images based on first file
            if str(file_list[0]).endswith('.nii.gz'):
                self.is_volume = True
                self._prepare_volumes(file_list)
            else:
                # Assume standard images
                self._prepare_images_from_list(file_list)
        else:
            # Auto-detect from directory
            nifti_files = sorted(list(self.images_dir.glob('*.nii.gz')))
            
            if nifti_files:
                self.is_volume = True
                self._prepare_volumes(nifti_files)
            else:
                self._prepare_images()
            
        logger.info(f"Dataset loaded: {len(self)} samples (Volume Mode: {self.is_volume})")

    def _prepare_volumes(self, nifti_files):
        """Pre-loads NIfTI volumes and indexes slices"""
        for f in nifti_files:
            f = Path(f)
            # Skip mask files if they are in the same directory/list
            if '_mask' in f.name: continue
            
            # Construct mask path
            mask_name = f.name.replace('.nii.gz', '_mask.nii.gz')
            mask_path = self.masks_dir / mask_name
            
            # Fallback: check images_dir if mask not found in masks_dir
            if not mask_path.exists():
                mask_path = self.images_dir / mask_name
            
            # Fallback: check if file path itself has a mask sibling (if file_list passed from arbitrary locations)
            if not mask_path.exists():
                mask_path = f.parent / mask_name

            if not mask_path.exists():
                logger.warning(f"Mask not found for {f.name}, skipping.")
                continue
                
            try:
                # Load volume and mask
                vol = nib.load(f).get_fdata()
                mask = nib.load(mask_path).get_fdata()
                
                if vol.shape != mask.shape:
                    logger.warning(f"Shape mismatch for {f.name}: {vol.shape} vs {mask.shape}")
                    continue
                    
                # Normalize volume
                vol = np.clip(vol, -1000, 2000)
                vol = (vol + 1000) / 3000.0
                
                self.volumes.append((vol.astype(np.float32), mask.astype(np.float32)))
                vol_idx = len(self.volumes) - 1
                
                # Index slices (Dim 0 is Depth/Axial)
                for i in range(vol.shape[0]):
                    self.samples.append((vol_idx, i))
                    
            except Exception as e:
                logger.error(f"Failed to load volume {f}: {e}")

    def _prepare_images(self):
        """Scan directory for images"""
        images = sorted([
            f for f in self.images_dir.glob('*') 
            if f.suffix.lower() in ['.dcm', '.png', '.jpg', '.jpeg']
        ])
        self._prepare_images_from_list(images)

    def _prepare_images_from_list(self, images):
        """Prepare samples from list of image paths"""
        masks = sorted([
            f for f in self.masks_dir.glob('*') 
            if f.suffix.lower() in ['.png', '.jpg']
        ])
        
        # Map filenames for robustness
        mask_map = {m.stem: m for m in masks}
        
        for img_path in images:
            img_path = Path(img_path)
            stem = img_path.stem
            if stem in mask_map:
                self.samples.append((str(img_path), str(mask_map[stem])))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.is_volume:
            vol_idx, slice_idx = self.samples[idx]
            vol, mask_vol = self.volumes[vol_idx]
            
            image = vol[slice_idx]
            mask = mask_vol[slice_idx]
            
            mask = (mask > 0.5).astype(np.float32)
            
        else:
            img_path, mask_path = self.samples[idx]
            
            if img_path.lower().endswith('.dcm'):
                dicom_data = self.dicom_processor.load_dicom(img_path)
                if dicom_data is None:
                    raise ValueError(f"Failed to load DICOM: {img_path}")
                image = dicom_data['pixel_array']
                if len(image.shape) == 3:
                    image = image[image.shape[0]//2] 
                image = np.clip(image, -1000, 2000)
                image = (image + 1000) / 3000.0
                image = image.astype(np.float32)
            else:
                image = np.array(Image.open(img_path).convert("L"))
                image = image.astype(np.float32) / 255.0
                
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 127).astype(np.float32)

        # Ensure dimensions are correct for Albumentations (H, W)
        # If image is (H, W), albumentations returns (H, W) or (H, W, C) depending on config?
        # ToTensorV2 will convert to (C, H, W).
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            default_transform = A.Compose([
                A.Resize(256, 256),
                ToTensorV2()
            ])
            augmented = default_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask
