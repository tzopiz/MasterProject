import torch
from torch.utils.data import DataLoader
from training.datasets.tmj_dataset import TMJDataset
from training.utils.transforms import get_training_transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def debug_data():
    data_dir = Path("MLService/data/processed_crops")
    
    print(f"📂 Checking data in {data_dir}")
    
    # Load dataset directly with same params as train.py
    # Note: we pass data_dir as both images/masks dir because of how we wrote the fallback logic
    ds = TMJDataset(
        data_dir, 
        data_dir, 
        transform=get_training_transforms()
    )
    
    print(f"✅ Dataset length: {len(ds)}")
    
    if len(ds) == 0:
        print("❌ No samples found!")
        return

    # Get one sample
    image, mask = ds[0]
    
    print(f"Tensor Shapes -> Image: {image.shape}, Mask: {mask.shape}")
    print(f"Image Range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Mask Unique Values: {torch.unique(mask)}")
    
    # Convert back to numpy for plot
    # Un-normalize image for viewing: x * 0.5 + 0.5
    img_np = image[0].numpy() * 0.5 + 0.5
    mask_np = mask.numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_np, cmap='gray')
    axs[0].set_title(f"Input Image\nMean: {img_np.mean():.2f}")
    
    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title(f"Target Mask\nSum: {mask_np.sum()}")
    
    plt.show()

if __name__ == "__main__":
    debug_data()

