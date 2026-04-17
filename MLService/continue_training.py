#!/usr/bin/env python3
"""
Continue training from checkpoint
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tmj_detector import TMJDetectorLarge
from training.datasets.tmj_detector_dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def compute_metrics(pred_coords, target_coords, original_shape=(576, 768, 768)):
    original_shape = torch.tensor(original_shape, device=pred_coords.device).float()
    
    pred_left = pred_coords[:, :3] * original_shape
    pred_right = pred_coords[:, 3:] * original_shape
    target_left = target_coords[:, :3] * original_shape
    target_right = target_coords[:, 3:] * original_shape
    
    dist_left = torch.norm(pred_left - target_left, dim=1)
    dist_right = torch.norm(pred_right - target_right, dim=1)
    
    error_left = torch.abs(pred_left - target_left)
    error_right = torch.abs(pred_right - target_right)
    
    metrics = {
        'mae_left': dist_left.mean().item(),
        'mae_right': dist_right.mean().item(),
        'mae_overall': (dist_left.mean() + dist_right.mean()).item() / 2,
        'mae_z': (error_left[:, 0].mean() + error_right[:, 0].mean()).item() / 2,
        'mae_y': (error_left[:, 1].mean() + error_right[:, 1].mean()).item() / 2,
        'mae_x': (error_left[:, 2].mean() + error_right[:, 2].mean()).item() / 2,
    }
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for volumes, targets in pbar:
        volumes = volumes.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(volumes)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            metrics = compute_metrics(predictions, targets)
            all_metrics.append(metrics)
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'mae': metrics['mae_overall']})
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = running_loss / len(train_loader)
    
    return avg_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for volumes, targets in pbar:
            volumes = volumes.to(device)
            targets = targets.to(device)
            
            predictions = model(volumes)
            loss = criterion(predictions, targets)
            
            metrics = compute_metrics(predictions, targets)
            all_metrics.append(metrics)
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'mae': metrics['mae_overall']})
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = running_loss / len(val_loader)
    
    return avg_metrics


def main(args):
    logger.info("="*70)
    logger.info("CONTINUING TMJ DETECTOR TRAINING")
    logger.info("="*70)
    
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.info("Using CPU")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_mae = checkpoint.get('best_val_mae', float('inf'))
    
    logger.info(f"Resuming from epoch {start_epoch}")
    logger.info(f"Previous best MAE: {best_val_mae:.2f} px")
    
    # Use same experiment directory
    exp_dir = Path(args.checkpoint).parent
    logger.info(f"Continuing in: {exp_dir}")
    
    # Load config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Original config: {config}")
    
    # Data
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(
        annotations_dir=config['annotations'],
        dataset_dir=config['dataset'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        downsample_factor=config['downsample_factor'],
        split_ratio=config['split_ratio']
    )
    
    # Model
    logger.info(f"Creating model: {config['model_type']}")
    model = TMJDetectorLarge().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params/1e6:.2f}M")
    
    # Optimizer & Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=config['weight_decay']
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        except Exception:
            logger.warning("Could not load optimizer state, using fresh optimizer")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience
    )
    
    # Training loop
    epochs_no_improve = 0
    
    logger.info("\n" + "="*70)
    logger.info("Continuing training...")
    logger.info("="*70 + "\n")
    
    for epoch in range(start_epoch, args.total_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Scheduler step
        scheduler.step(val_metrics['mae_overall'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(f"\nEpoch {epoch}/{args.total_epochs}")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae_overall']:.2f} px")
        logger.info(f"  Val   Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae_overall']:.2f} px")
        logger.info(f"  Val MAE - Left: {val_metrics['mae_left']:.2f}, Right: {val_metrics['mae_right']:.2f}")
        logger.info(f"  Val MAE - Z: {val_metrics['mae_z']:.2f}, Y: {val_metrics['mae_y']:.2f}, X: {val_metrics['mae_x']:.2f}")
        logger.info(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['mae_overall'] < best_val_mae:
            best_val_mae = val_metrics['mae_overall']
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'metrics': val_metrics
            }, exp_dir / 'best_model.pth')
            
            logger.info(f"  ✅ Saved best model (MAE: {best_val_mae:.2f} px)")
        else:
            epochs_no_improve += 1
            logger.info(f"  No improvement ({epochs_no_improve}/{args.early_stopping})")
        
        # Early stopping
        if epochs_no_improve >= args.early_stopping:
            logger.info(f"\nEarly stopping after {epoch} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, exp_dir / f'checkpoint_epoch{epoch}.pth')
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best validation MAE: {best_val_mae:.2f} pixels")
    logger.info(f"Model saved to: {exp_dir / 'best_model.pth'}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continue Training TMJ Detector")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--total_epochs', type=int, default=150,
                        help='Total number of epochs to train to')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for continued training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for LR scheduler')
    parser.add_argument('--early_stopping', type=int, default=50,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)

