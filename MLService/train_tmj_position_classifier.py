#!/usr/bin/env python3
"""
Train TMJ Position Classifier

Trains a 3D CNN to classify TMJ condyle positions (sagittal + frontal,
left + right) from CBCT volumes.

Loss: sum of four CrossEntropyLoss terms (one per head).
Best model is saved based on mean accuracy across all four heads on val set.

Usage example:
    cd MLService
    ./venv/bin/python train_tmj_position_classifier.py \\
        --dataset-root  data/dataset_cbct_public \\
        --labels-json   data/tmj_position_labels.json \\
        --manifest-private data/dataset_cbct_public/manifest_private.json \\
        --epochs 50 \\
        --batch-size 2 \\
        --output-dir experiments/position_run1

See MLService/docs/README.md for full documentation.
Refs #67
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

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tmj_position_classifier import TMJPositionClassifier
from training.datasets.tmj_position_dataset import get_position_dataloaders

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Fixed order of the four heads
HEAD_NAMES = ["sag_right", "sag_left", "fr_right", "fr_left"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(logits_tuple, labels: torch.Tensor):
    """
    Compute per-head and mean accuracy.

    Args:
        logits_tuple: (sag_right, sag_left, fr_right, fr_left) — each (B, 3)
        labels: (B, 4) int64

    Returns:
        dict: per-head accuracy and mean_accuracy
    """
    metrics = {}
    accs = []
    for i, (name, logits) in enumerate(zip(HEAD_NAMES, logits_tuple)):
        preds = logits.argmax(dim=1)  # (B,)
        acc = (preds == labels[:, i]).float().mean().item()
        metrics[f"acc_{name}"] = acc
        accs.append(acc)
    metrics["mean_accuracy"] = float(np.mean(accs))
    return metrics


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for volumes, labels in pbar:
        volumes = volumes.to(device)
        labels = labels.to(device)  # (B, 4)

        optimizer.zero_grad()
        outputs = model(volumes)  # tuple of 4 × (B, 3)

        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            m = compute_accuracy(outputs, labels)
        m["loss"] = loss.item()
        all_metrics.append(m)
        running_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{m['mean_accuracy']:.3f}"})

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg["loss"] = running_loss / len(loader)
    return avg


def validate_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for volumes, labels in pbar:
            volumes = volumes.to(device)
            labels = labels.to(device)

            outputs = model(volumes)
            loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))

            m = compute_accuracy(outputs, labels)
            m["loss"] = loss.item()
            all_metrics.append(m)
            running_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{m['mean_accuracy']:.3f}"})

    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg["loss"] = running_loss / len(loader)
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logger.info("=" * 70)
    logger.info("TMJ POSITION CLASSIFIER TRAINING")
    logger.info("=" * 70)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # --- Experiment directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"position_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # File logger
    fh = logging.FileHandler(exp_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    # Save config
    config = vars(args)
    config["experiment_dir"] = str(exp_dir)
    config["timestamp"] = timestamp
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Experiment dir: %s", exp_dir)

    # --- Data ---
    logger.info("Building dataloaders …")
    train_loader, val_loader = get_position_dataloaders(
        manifest_path=args.manifest_private,
        labels_path=args.labels_json,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        downsample_factor=args.downsample_factor,
        split_ratio=args.split_ratio,
    )
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    if len(train_loader) == 0:
        logger.error("Train loader is empty — check that manifest and labels have matching patients.")
        sys.exit(1)
    if len(val_loader) == 0:
        logger.error(
            "Val loader is empty — need at least 2 distinct patients after join, or adjust --split-ratio."
        )
        sys.exit(1)

    # --- Model ---
    model = TMJPositionClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)

    # --- Loss & optimiser ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    # --- Training loop ---
    best_val_acc = -1.0
    epochs_no_improve = 0
    metrics_log = []

    logger.info("\n" + "=" * 70)
    logger.info("Starting training …")
    logger.info("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_m = validate_epoch(model, val_loader, criterion, device, epoch)

        scheduler.step(val_m["mean_accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info("Epoch %d/%d", epoch, args.epochs)
        logger.info(
            "  Train  loss=%.4f  acc=%.3f  [sr=%.3f sl=%.3f fr=%.3f fl=%.3f]",
            train_m["loss"], train_m["mean_accuracy"],
            train_m["acc_sag_right"], train_m["acc_sag_left"],
            train_m["acc_fr_right"],  train_m["acc_fr_left"],
        )
        logger.info(
            "  Val    loss=%.4f  acc=%.3f  [sr=%.3f sl=%.3f fr=%.3f fl=%.3f]",
            val_m["loss"], val_m["mean_accuracy"],
            val_m["acc_sag_right"], val_m["acc_sag_left"],
            val_m["acc_fr_right"],  val_m["acc_fr_left"],
        )
        logger.info("  LR: %.6f", current_lr)

        # Persist metrics
        row = {"epoch": epoch, "lr": current_lr}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        metrics_log.append(row)

        with open(exp_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        # Best model (by mean val accuracy)
        if val_m["mean_accuracy"] > best_val_acc:
            best_val_acc = val_m["mean_accuracy"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "val_metrics": val_m,
                },
                exp_dir / "best_model.pth",
            )
            logger.info("  [saved best model  acc=%.3f]", best_val_acc)
        else:
            epochs_no_improve += 1
            logger.info("  No improvement (%d/%d)", epochs_no_improve, args.early_stopping)

        # Early stopping
        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            logger.info("Early stopping after epoch %d", epoch)
            break

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                exp_dir / f"checkpoint_epoch{epoch:04d}.pth",
            )

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("Best val mean accuracy: %.3f", best_val_acc)
    logger.info("Artefacts saved to: %s", exp_dir)
    logger.info("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TMJ Position Classifier (issue #67)"
    )

    # Data
    parser.add_argument(
        "--dataset-root",
        default="data/dataset_cbct_public",
        help="Root directory with study_* DICOM folders",
    )
    parser.add_argument(
        "--labels-json",
        default="data/tmj_position_labels.json",
        help="Path to tmj_position_labels.json",
    )
    parser.add_argument(
        "--manifest-private",
        default="data/dataset_cbct_public/manifest_private.json",
        help="Path to manifest_private.json (contains patient names)",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8)

    # Volume preprocessing
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=6,
        help="Uniform downsampling factor (6 → 576 slices become 96)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=30,
        help="Early stopping patience (0 = disabled)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu', 'cuda', 'mps' (auto-detected if omitted)",
    )

    # Output
    parser.add_argument("--output-dir", default="experiments", help="Base experiment directory")

    args = parser.parse_args()
    main(args)
