import logging
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from services.dicom_processor import DICOMProcessor

logger = logging.getLogger(__name__)

class TMJDataset(Dataset):
    """
    Dataset for TMJ segmentation.
    Expects a directory structure with 'images' and 'masks' subdirectories.
    Images can be DICOM (.dcm) or standard images (.png, .jpg).
    Masks should be binary images (.png).
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.dicom_processor = DICOMProcessor()
        
        # Find all images
        self.images = sorted([
            f for f in self.images_dir.glob('*') 
            if f.suffix.lower() in ['.dcm', '.png', '.jpg', '.jpeg']
        ])
        
        # Find corresponding masks
        self.masks = sorted([
            f for f in self.masks_dir.glob('*') 
            if f.suffix.lower() in ['.png', '.jpg']
        ])
        
        # Basic validation
        if len(self.images) != len(self.masks):
            logger.warning(f"Mismatch in number of images ({len(self.images)}) and masks ({len(self.masks)})")
    
    def __len__(self):
        return min(len(self.images), len(self.masks))
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Ideally masks should be matched by filename, but for now we assume sorted order
        # Improving robustness: try to find mask with same stem
        mask_path = self.masks[idx] 
        
        # Load Image
        if img_path.suffix.lower() == '.dcm':
            dicom_data = self.dicom_processor.load_dicom(str(img_path))
            if dicom_data is None:
                raise ValueError(f"Failed to load DICOM: {img_path}")
            
            image = dicom_data['pixel_array']
            if len(image.shape) == 3:
                # Take middle slice for 3D volumes
                image = image[image.shape[0]//2] 
        else:
            image = np.array(Image.open(img_path).convert("L"))
            
        # Load Mask
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Binary mask (0 or 255) -> (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default transform
            default_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
            augmented = default_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask

