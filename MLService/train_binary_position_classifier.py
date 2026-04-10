#!/usr/bin/env python3
"""
Train Binary TMJ Position Classifier (central vs non-central)

Loads detector-generated NIfTI crops, trains a 2-head 3D CNN with
BinaryFocalLoss, then calibrates prediction thresholds per head via
Youden's J on the validation ROC curve.

Usage:
    cd MLService
    ./venv/bin/python train_binary_position_classifier.py \\
        --crop-dir          data/detector_crops \\
        --labels-json       data/tmj_position_labels.json \\
        --manifest-private  data/dataset_cbct_public/manifest_private.json \\
        --dataset-root      data/dataset_cbct_public \\
        --epochs            100 \\
        --batch-size        4 \\
        --gamma             2.0 \\
        --output-dir        experiments

See MLService/docs/superpowers/specs/2026-04-10-binary-classifier-improvement-design.md
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
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier
from training.datasets.tmj_position_dataset import get_binary_position_dataloaders
from training.losses.focal_loss import BinaryFocalLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

HEAD_NAMES = ["sag", "fr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(loader) -> dict:
    """Count class distribution across the training set for alpha estimation."""
    counts = {name: {0: 0, 1: 0} for name in HEAD_NAMES}
    for _, labels in loader:
        for i, name in enumerate(HEAD_NAMES):
            for cls in (0, 1):
                counts[name][cls] += (labels[:, i] == cls).sum().item()
    alphas = {}
    for name in HEAD_NAMES:
        total = counts[name][0] + counts[name][1]
        # alpha = fraction of negative class (weight for positive)
        alphas[name] = counts[name][0] / total if total > 0 else 0.5
        logger.info(
            "Class distribution [%s]: 0=%d (%.1f%%)  1=%d (%.1f%%)  → alpha=%.3f",
            name,
            counts[name][0], 100 * counts[name][0] / max(total, 1),
            counts[name][1], 100 * counts[name][1] / max(total, 1),
            alphas[name],
        )
    return alphas


def compute_metrics(sag_logits, fr_logits, labels, thresholds=(0.5, 0.5)):
    """Binary accuracy per head using provided thresholds."""
    metrics = {}
    thresh_sag, thresh_fr = thresholds
    for i, (name, logits, thresh) in enumerate(
        zip(HEAD_NAMES, [sag_logits, fr_logits], [thresh_sag, thresh_fr])
    ):
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= thresh).long()
        true = labels[:, i]
        acc = (preds == true).float().mean().item()
        metrics[f"acc_{name}"] = acc
    metrics["mean_accuracy"] = float(np.mean([metrics[f"acc_{n}"] for n in HEAD_NAMES]))
    return metrics


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criteria, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for volumes, labels in pbar:
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        sag_logit, fr_logit = model(volumes)

        loss = (
            criteria["sag"](sag_logit, labels[:, 0].float())
            + criteria["fr"](fr_logit, labels[:, 1].float())
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            m = compute_metrics(sag_logit, fr_logit, labels)
        m["loss"] = loss.item()
        all_metrics.append(m)
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{m['mean_accuracy']:.3f}"})

    if not all_metrics:
        raise RuntimeError(f"No batches processed in train epoch {epoch}. Check your dataloader.")
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg["loss"] = running_loss / len(loader)
    return avg


def validate_epoch(model, loader, criteria, device, epoch):
    model.eval()
    running_loss = 0.0
    all_metrics = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]  ")
    with torch.no_grad():
        for volumes, labels in pbar:
            volumes = volumes.to(device)
            labels = labels.to(device)
            sag_logit, fr_logit = model(volumes)
            loss = (
                criteria["sag"](sag_logit, labels[:, 0].float())
                + criteria["fr"](fr_logit, labels[:, 1].float())
            )
            m = compute_metrics(sag_logit, fr_logit, labels)
            m["loss"] = loss.item()
            all_metrics.append(m)
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{m['mean_accuracy']:.3f}"})

    if not all_metrics:
        raise RuntimeError(f"No batches processed in val epoch {epoch}. Check your dataloader.")
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg["loss"] = running_loss / len(loader)
    return avg


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_thresholds(model, val_loader, device) -> dict:
    """
    Collect val predictions and calibrate thresholds via Youden's J.

    Returns dict with keys: optimal_thresholds, auc_roc, accuracy_at_threshold
    """
    model.eval()
    all_probs = {name: [] for name in HEAD_NAMES}
    all_labels = {name: [] for name in HEAD_NAMES}

    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(device)
            sag_logit, fr_logit = model(volumes)
            for i, name in enumerate(HEAD_NAMES):
                logit = [sag_logit, fr_logit][i]
                all_probs[name].extend(torch.sigmoid(logit.squeeze(1)).cpu().tolist())
                all_labels[name].extend(labels[:, i].tolist())

    results = {"optimal_thresholds": {}, "auc_roc": {}, "accuracy_at_threshold": {}}

    for name in HEAD_NAMES:
        probs = np.array(all_probs[name])
        labels = np.array(all_labels[name])

        if len(np.unique(labels)) < 2:
            logger.warning("[%s] Only one class in val — skipping ROC, threshold=0.5", name)
            results["optimal_thresholds"][name] = 0.5
            results["auc_roc"][name] = float("nan")
            results["accuracy_at_threshold"][name] = float(np.mean((probs >= 0.5) == labels))
            continue

        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        # Youden's J = sensitivity + specificity - 1
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        acc = float(np.mean((probs >= best_thresh) == labels))

        results["optimal_thresholds"][name] = round(best_thresh, 4)
        results["auc_roc"][name] = round(auc, 4)
        results["accuracy_at_threshold"][name] = round(acc, 4)

        logger.info(
            "[%s] AUC=%.3f  Youden J best thresh=%.3f  acc@thresh=%.3f",
            name, auc, best_thresh, acc,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    logger.info("=" * 70)
    logger.info("BINARY TMJ POSITION CLASSIFIER TRAINING (Approach A+B)")
    logger.info("=" * 70)

    # Device
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

    # Experiment dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"binary_position_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(exp_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    config = vars(args)
    config["experiment_dir"] = str(exp_dir)
    config["timestamp"] = timestamp
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Experiment dir: %s", exp_dir)

    # Data
    logger.info("Building dataloaders …")
    train_loader, val_loader = get_binary_position_dataloaders(
        crop_dir=args.crop_dir,
        manifest_path=args.manifest_private,
        labels_path=args.labels_json,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
    )
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    if len(train_loader) == 0:
        logger.error("Train loader is empty.")
        sys.exit(1)
    if len(val_loader) == 0:
        logger.error("Val loader is empty.")
        sys.exit(1)

    # Auto-compute alpha from training class distribution
    logger.info("Computing class weights for alpha …")
    alphas = compute_class_weights(train_loader)

    # Model
    model = TMJBinaryPositionClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)

    # Loss (one per head, with head-specific alpha)
    criteria = {
        name: BinaryFocalLoss(gamma=args.gamma, alpha=alphas[name])
        for name in HEAD_NAMES
    }

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    # Training loop
    best_val_acc = -1.0
    epochs_no_improve = 0
    metrics_log = []

    logger.info("\n" + "=" * 70)
    logger.info("Starting training …")
    logger.info("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, criteria, optimizer, device, epoch)
        val_m = validate_epoch(model, val_loader, criteria, device, epoch)

        scheduler.step(val_m["mean_accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info("Epoch %d/%d", epoch, args.epochs)
        logger.info(
            "  Train  loss=%.4f  acc=%.3f  [sag=%.3f  fr=%.3f]",
            train_m["loss"], train_m["mean_accuracy"],
            train_m["acc_sag"], train_m["acc_fr"],
        )
        logger.info(
            "  Val    loss=%.4f  acc=%.3f  [sag=%.3f  fr=%.3f]",
            val_m["loss"], val_m["mean_accuracy"],
            val_m["acc_sag"], val_m["acc_fr"],
        )
        logger.info("  LR: %.6f", current_lr)

        row = {"epoch": epoch, "lr": current_lr}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        metrics_log.append(row)
        with open(exp_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

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
            logger.info("  [saved best  acc=%.3f]", best_val_acc)
        else:
            epochs_no_improve += 1
            logger.info("  No improvement (%d/%d)", epochs_no_improve, args.early_stopping)

        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            logger.info("Early stopping after epoch %d", epoch)
            break

    # Load best model for threshold calibration
    logger.info("\n" + "=" * 70)
    logger.info("Calibrating thresholds on val set …")
    ckpt = torch.load(exp_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    calibration = calibrate_thresholds(model, val_loader, device)

    # Persist thresholds into config.json
    with open(exp_dir / "config.json", "r") as f:
        saved_config = json.load(f)
    saved_config.update(calibration)
    saved_config["best_val_accuracy"] = best_val_acc
    with open(exp_dir / "config.json", "w") as f:
        json.dump(saved_config, f, indent=2)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("Best val mean accuracy: %.3f", best_val_acc)
    logger.info("Optimal thresholds: %s", calibration["optimal_thresholds"])
    logger.info("AUC-ROC: %s", calibration["auc_roc"])
    logger.info("Artefacts saved to: %s", exp_dir)
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Binary TMJ Position Classifier (Approach A+B)"
    )

    # Data
    parser.add_argument("--crop-dir", required=True,
                        help="Root of detector NIfTI crops (data/detector_crops)")
    parser.add_argument("--labels-json", default="data/tmj_position_labels.json")
    parser.add_argument("--manifest-private",
                        default="data/dataset_cbct_public/manifest_private.json")
    parser.add_argument("--dataset-root", default="data/dataset_cbct_public")
    parser.add_argument("--split-ratio", type=float, default=0.8)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (default 2.0)")

    # Hardware
    parser.add_argument("--device", default=None,
                        help="Force device: cpu / cuda / mps (auto-detected if omitted)")

    # Output
    parser.add_argument("--output-dir", default="experiments")

    args = parser.parse_args()
    main(args)
