#!/usr/bin/env python3
"""
Train TMJ Detector

Train a 3D CNN to regress TMJ coordinates from full CBCT scans.

Usage:
    python train_detector.py \\
        --annotations data/roi_annotations \\
        --dataset data/dataset \\
        --epochs 200 \\
        --batch_size 2
"""

import argparse
import datetime
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

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tmj_detector import TMJDetector, TMJDetectorLarge
from training.datasets.tmj_detector_dataset import get_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_metrics(pred_coords, target_coords, original_shape=(576, 768, 768)):
    """
    Compute evaluation metrics

    Args:
        pred_coords: (B, 6) normalized predictions
        target_coords: (B, 6) normalized targets
        original_shape: tuple of original volume shape

    Returns:
        dict with metrics
    """
    # Denormalize to pixel space
    original_shape = torch.tensor(original_shape, device=pred_coords.device).float()

    pred_left = pred_coords[:, :3] * original_shape
    pred_right = pred_coords[:, 3:] * original_shape
    target_left = target_coords[:, :3] * original_shape
    target_right = target_coords[:, 3:] * original_shape

    # Euclidean distance
    dist_left = torch.norm(pred_left - target_left, dim=1)
    dist_right = torch.norm(pred_right - target_right, dim=1)

    # Per-axis errors
    error_left = torch.abs(pred_left - target_left)
    error_right = torch.abs(pred_right - target_right)

    metrics = {
        "mae_left": dist_left.mean().item(),
        "mae_right": dist_right.mean().item(),
        "mae_overall": (dist_left.mean() + dist_right.mean()).item() / 2,
        "mae_z": (error_left[:, 0].mean() + error_right[:, 0].mean()).item() / 2,
        "mae_y": (error_left[:, 1].mean() + error_right[:, 1].mean()).item() / 2,
        "mae_x": (error_left[:, 2].mean() + error_right[:, 2].mean()).item() / 2,
    }

    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for volumes, targets in pbar:
        volumes = volumes.to(device)
        targets = targets.to(device)

        # Forward
        optimizer.zero_grad()
        predictions = model(volumes)
        loss = criterion(predictions, targets)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(predictions, targets)
            all_metrics.append(metrics)

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "mae": metrics["mae_overall"]})

    # Average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    avg_metrics["loss"] = running_loss / len(train_loader)

    return avg_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()

    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for volumes, targets in pbar:
            volumes = volumes.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(volumes)
            loss = criterion(predictions, targets)

            # Metrics
            metrics = compute_metrics(predictions, targets)
            all_metrics.append(metrics)

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "mae": metrics["mae_overall"]})

    # Average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    avg_metrics["loss"] = running_loss / len(val_loader)

    return avg_metrics


def main(args):
    logger.info("=" * 70)
    logger.info("TMJ DETECTOR TRAINING")
    logger.info("=" * 70)

    # Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.info("Using CPU")

    # Experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"detector_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Experiment directory: {exp_dir}")

    # Data
    logger.info(f"Loading data from {args.annotations}")
    train_loader, val_loader = get_dataloaders(
        annotations_dir=args.annotations,
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_factor=args.downsample_factor,
        split_ratio=args.split_ratio,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Model
    logger.info(f"Creating model: {args.model_type}")
    if args.model_type == "small":
        model = TMJDetector().to(device)
    elif args.model_type == "large":
        model = TMJDetectorLarge().to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    # Training loop
    best_val_mae = float("inf")
    epochs_no_improve = 0

    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        # Scheduler step
        scheduler.step(val_metrics["mae_overall"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae_overall']:.2f} px"
        )
        logger.info(
            f"  Val   Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae_overall']:.2f} px"
        )
        logger.info(
            f"  Val MAE - Left: {val_metrics['mae_left']:.2f}, Right: {val_metrics['mae_right']:.2f}"
        )
        logger.info(
            f"  Val MAE - Z: {val_metrics['mae_z']:.2f}, Y: {val_metrics['mae_y']:.2f}, X: {val_metrics['mae_x']:.2f}"
        )
        logger.info(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_metrics["mae_overall"] < best_val_mae:
            best_val_mae = val_metrics["mae_overall"]
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "metrics": val_metrics,
                },
                exp_dir / "best_model.pth",
            )

            logger.info(f"  ✅ Saved best model (MAE: {best_val_mae:.2f} px)")
        else:
            epochs_no_improve += 1
            logger.info(f"  No improvement ({epochs_no_improve}/{args.early_stopping})")

        # Early stopping
        if epochs_no_improve >= args.early_stopping:
            logger.info(f"\nEarly stopping after {epoch} epochs")
            break

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                exp_dir / f"checkpoint_epoch{epoch}.pth",
            )

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best validation MAE: {best_val_mae:.2f} pixels")
    logger.info(f"Model saved to: {exp_dir / 'best_model.pth'}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TMJ Detector")

    # Data
    parser.add_argument(
        "--annotations", type=str, default="data/roi_annotations", help="Path to ROI annotations"
    )
    parser.add_argument("--dataset", type=str, default="data/dataset", help="Path to DICOM dataset")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/val split ratio")

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Model architecture",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=6,
        help="Downsample factor for input volumes (6 = 576→96)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")

    # Regularization
    parser.add_argument("--patience", type=int, default=10, help="Patience for LR scheduler")
    parser.add_argument("--early_stopping", type=int, default=30, help="Early stopping patience")

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="Output directory for experiments"
    )

    args = parser.parse_args()
    main(args)
