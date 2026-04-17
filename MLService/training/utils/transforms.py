import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms(height=256, width=256):
    """
    Returns augmentation pipeline for training
    """
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


def get_validation_transforms(height=256, width=256):
    """
    Returns pipeline for validation (resize + normalize only)
    """
    return A.Compose([A.Resize(height, width), A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
