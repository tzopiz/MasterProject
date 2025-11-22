import argparse
import logging
import sys
import os
import random
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.segmentation_model import UNet
from training.datasets.tmj_dataset import TMJDataset
from training.losses.dice_loss import DiceLoss
from training.utils.transforms import get_training_transforms, get_validation_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_model(data_dir, epochs=50, batch_size=8, learning_rate=1e-4, device='cpu'):
    """
    Main training loop
    """
    logger.info(f"Starting training on device: {device}")
    
    # 1. Setup Data
    root_dir = Path(data_dir)
    train_images_dir = root_dir / 'train' / 'images'
    
    train_files = []
    val_files = []
    
    # Scenario A: Structured data (train/val folders)
    if train_images_dir.exists():
        logger.info("Found structured dataset (train/val folders)")
        train_dataset = TMJDataset(
            root_dir / 'train' / 'images', 
            root_dir / 'train' / 'masks', 
            transform=get_training_transforms()
        )
        val_dataset = TMJDataset(
            root_dir / 'val' / 'images', 
            root_dir / 'val' / 'masks', 
            transform=get_validation_transforms()
        )
    else:
        # Scenario B: Flat structure (e.g. NIfTI files in one folder)
        logger.info("Looking for flat dataset structure (NIfTI/Images in data_dir)")
        
        # Try to find NIfTI files
        all_files = sorted(list(root_dir.glob('*.nii.gz')))
        # Filter out masks
        all_volumes = [f for f in all_files if '_mask' not in f.name]
        
        if all_volumes:
            logger.info(f"Found {len(all_volumes)} volumes. Splitting into Train/Val.")
            
            # Shuffle and split by volume (to avoid slice leakage)
            random.shuffle(all_volumes)
            split_idx = int(len(all_volumes) * 0.8) # 80% train
            
            train_files = all_volumes[:split_idx]
            val_files = all_volumes[split_idx:]
            
            # If dataset is too small (e.g. 1 volume), put it in both for testing?
            # Or just train on it.
            if not train_files and val_files:
                train_files = val_files
            if not val_files and train_files:
                val_files = train_files # Validate on train set if only 1 volume
                
            logger.info(f"Train volumes: {len(train_files)}")
            logger.info(f"Val volumes: {len(val_files)}")
            
            train_dataset = TMJDataset(
                root_dir, root_dir, 
                transform=get_training_transforms(),
                file_list=train_files
            )
            val_dataset = TMJDataset(
                root_dir, root_dir, 
                transform=get_validation_transforms(),
                file_list=val_files
            )
        else:
            logger.error("No valid dataset found in structure or flat format.")
            return

    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if device != 'cpu' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)} slices")
        logger.info(f"Val dataset size: {len(val_dataset)} slices")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    # 2. Setup Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # 3. Loss and Optimizer
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1) # Add channel dim
                
                optimizer.zero_grad()
                
                outputs = model(images)
                
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                # Combined loss: BCE for pixel-wise accuracy, Dice for overlap
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                
                outputs = model(images)
                
                loss_bce = criterion_bce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = models_dir / 'segmentation_model_best.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ Saved new best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TMJ Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Device selection
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr, device)
