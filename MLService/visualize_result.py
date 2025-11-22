import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_model import UNet

def predict_and_show(model_path, nifti_path, device='cpu'):
    print(f"🔍 Loading model from {model_path}...")
    
    # Load Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Data
    print(f"📂 Loading volume from {nifti_path}...")
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    
    # Normalize like in training
    # 1. Clip and Scale to [0, 1]
    vol_norm = np.clip(volume, -1000, 2000)
    vol_norm = (vol_norm + 1000) / 3000.0
    
    # 2. Normalize (mean=0.5, std=0.5) -> [-1, 1]
    # This matches Albumentations Normalize(mean=0.5, std=0.5)
    vol_norm = (vol_norm - 0.5) / 0.5
    
    # Predict slice by slice
    print("🧠 Running inference...")
    preds = []
    
    # Iterate Depth (Dim 0)
    with torch.no_grad():
        for i in range(vol_norm.shape[0]):
            slice_img = vol_norm[i] # (H, W)
            
            # Preprocess for model: (1, 1, H, W)
            # Resize to 256x256 is mandatory if model trained on 256
            # But our UNet handles any size multiple of 16. 
            # Training used Resize(256, 256).
            # Let's resize input to 256, predict, resize back.
            
            from skimage.transform import resize
            h, w = slice_img.shape
            inp = resize(slice_img, (256, 256), preserve_range=True)
            
            tensor = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)
            
            out = model(tensor)
            
            mask = out.squeeze().cpu().numpy() # (256, 256)
            
            # Resize back to original
            mask = resize(mask, (h, w), preserve_range=True)
            
            preds.append(mask)
            
    pred_volume = np.stack(preds)
    
    # Find slice with max activation
    max_slice_idx = np.unravel_index(np.argmax(pred_volume), pred_volume.shape)[0]
    print(f"✨ Max activation found at slice {max_slice_idx}")
    
    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axs[0].imshow(volume[max_slice_idx], cmap='gray', vmin=-400, vmax=1000)
    axs[0].set_title(f"Original Slice {max_slice_idx}")
    
    # Prediction (Probability)
    im = axs[1].imshow(pred_volume[max_slice_idx], cmap='jet', vmin=0, vmax=1)
    axs[1].set_title("Model Prediction (Prob)")
    plt.colorbar(im, ax=axs[1])
    
    # Overlay
    axs[2].imshow(volume[max_slice_idx], cmap='gray', vmin=-400, vmax=1000)
    axs[2].imshow(pred_volume[max_slice_idx] > 0.5, cmap='jet', alpha=0.4) # Threshold 0.5
    axs[2].set_title("Overlay (Threshold > 0.5)")
    
    plt.show()
    
    # Check if ground truth mask exists
    mask_path = Path(str(nifti_path).replace('.nii.gz', '_mask.nii.gz'))
    if mask_path.exists():
        print("✅ Ground truth mask found, comparing...")
        mask_gt = nib.load(mask_path).get_fdata()
        
        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
        axs2[0].imshow(mask_gt[max_slice_idx], cmap='gray')
        axs2[0].set_title("Ground Truth")
        
        axs2[1].imshow(pred_volume[max_slice_idx] > 0.5, cmap='gray')
        axs2[1].set_title("Prediction")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nifti", help="Path to NIfTI volume")
    parser.add_argument("--model", default="MLService/models/segmentation_model_best.pth", help="Path to model")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    predict_and_show(args.model, args.nifti, device)

