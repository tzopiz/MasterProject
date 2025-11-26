"""
Training script for 3D TMJ segmentation

Usage:
    python train_3d.py --data_dir data/processed_crops --epochs 100 --batch_size 2
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from training.datasets.tmj_3d_dataset import TMJ3DDataset
from training.losses.dice_loss import DiceLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion_bce, criterion_dice, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    
    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for volumes, masks in pbar:
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(volumes)
            
            # Compute losses
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            epoch_dice += (1.0 - loss_dice.item())  # Dice score = 1 - Dice loss
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{(1.0 - loss_dice.item()):.4f}'
            })
    
    avg_loss = epoch_loss / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion_bce, criterion_dice, device):
    """Validate model"""
    model.eval()
    val_loss = 0
    val_dice = 0
    
    with torch.no_grad():
        for volumes, masks in dataloader:
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            outputs = model(volumes)
            
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            
            val_loss += loss.item()
            val_dice += (1.0 - loss_dice.item())
    
    avg_loss = val_loss / len(dataloader)
    avg_dice = val_dice / len(dataloader)
    
    return avg_loss, avg_dice


def main(args):
    logger.info("="*70)
    logger.info("3D TMJ Segmentation Training")
    logger.info("="*70)
    
    # Setup device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.info("Using CPU")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_3d_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['experiment_name'] = exp_name
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    
    dataset = TMJ3DDataset(
        data_dir=args.data_dir,
        normalize=True,
        target_shape=(args.crop_size, args.crop_size, args.crop_size)
    )
    
    logger.info(f"Total samples: {len(dataset)}")
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    if train_size == 0 or val_size == 0:
        logger.warning("Dataset too small for split, using all for training")
        train_dataset = dataset
        val_dataset = dataset  # Use same for validation (not ideal but for testing)
    else:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for 3D to avoid memory issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info(f"Creating 3D U-Net (features={args.init_features})")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        init_features=args.init_features
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")
    
    # Loss functions
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'lr': []
    }
    
    logger.info("\nStarting training...")
    logger.info("="*70)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion_bce, criterion_dice, optimizer, device
        )
        
        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion_bce, criterion_dice, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        # Save checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint, exp_dir / 'model_best.pth')
            torch.save(model.state_dict(), 'models/unet_3d_best.pth')  # Also save to models/
            
            logger.info(f"✅ Saved new best model (Dice: {val_dice:.4f})")
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if current_lr < 1e-7:
            logger.info("Learning rate too small, stopping training")
            break
    
    # Save final history
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D TMJ Segmentation")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed_crops',
                       help='Path to dataset directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model
    parser.add_argument('--init_features', type=int, default=16,
                       help='Initial number of features (16 or 32)')
    parser.add_argument('--crop_size', type=int, default=128,
                       help='Crop size (128 or 96 for less memory)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (2 for 3D, may need 1 if OOM)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    main(args)

