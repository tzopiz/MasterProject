#!/usr/bin/env python3
"""
Train TMJ Heatmap Detector (3D U-Net).

Usage:
    ./venv/bin/python train_heatmap_detector.py \
        --split-json    data/detector_split.json \
        --annotations   data/roi_annotations \
        --dataset       data/dataset_cbct_public \
        --epochs        200 \
        --batch-size    2 \
        --output-dir    experiments
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.datasets.tmj_heatmap_dataset import get_heatmap_dataloaders
from training.losses.heatmap_loss import weighted_mse_loss
from training.utils.heatmap import coords_from_heatmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_mae(pred_hm: torch.Tensor, target_hm: torch.Tensor, ds_factor: int = 6) -> dict:
    """
    MAE in downsampled pixels for a batch.
    pred_hm, target_hm: (B, 2, D, H, W)
    """
    B = pred_hm.shape[0]
    errors_left, errors_right = [], []

    for b in range(B):
        for ch, err_list in enumerate([errors_left, errors_right]):
            pred_coords, _ = coords_from_heatmap(torch.sigmoid(pred_hm[b, ch]), ds_factor)
            true_coords, _ = coords_from_heatmap(target_hm[b, ch], ds_factor)
            err = torch.sqrt(((pred_coords - true_coords) ** 2).sum()).item()
            err_list.append(err)

    return {
        "mae_left": float(np.mean(errors_left)),
        "mae_right": float(np.mean(errors_right)),
        "mae_overall": float(np.mean(errors_left + errors_right)),
    }


def run_epoch(model, loader, optimizer, device, scaler, is_train, epoch, ds_factor):
    model.train() if is_train else model.eval()
    tag = "Train" if is_train else "Val  "
    running_loss, all_mae_l, all_mae_r = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for vols, targets in tqdm(loader, desc=f"[{epoch}] {tag}", leave=False):
            vols, targets = vols.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(vols)
                        loss = weighted_mse_loss(torch.sigmoid(pred), targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(vols)
                    loss = weighted_mse_loss(torch.sigmoid(pred), targets)
                    loss.backward()
                    optimizer.step()
            else:
                pred = model(vols)
                loss = weighted_mse_loss(torch.sigmoid(pred), targets)

            running_loss += loss.item()
            m = compute_mae(pred.detach().cpu(), targets.cpu(), ds_factor)
            all_mae_l.append(m["mae_left"])
            all_mae_r.append(m["mae_right"])

    if not all_mae_l:
        raise RuntimeError(f"No batches processed in epoch {epoch}")

    n = len(loader)
    return {
        "loss": running_loss / n,
        "mae_left": float(np.mean(all_mae_l)),
        "mae_right": float(np.mean(all_mae_r)),
        "mae_overall": float(np.mean(all_mae_l + all_mae_r)),
    }


def main(args):
    logger.info("=" * 70)
    logger.info("TMJ HEATMAP DETECTOR TRAINING")
    logger.info("=" * 70)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS")
    else:
        device = torch.device("cpu")
        logger.info("CPU")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Experiment dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"heatmap_detector_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(exp_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    config = vars(args)
    config["timestamp"] = ts
    config["heatmap"] = True
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Experiment: %s", exp_dir)

    # Data
    train_loader, val_loader = get_heatmap_dataloaders(
        split_json=args.split_json,
        annotations_dir=args.annotations,
        dataset_dir=args.dataset,
        sigma=args.sigma,
        downsample_factor=args.downsample_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Train: %d batches  Val: %d batches", len(train_loader), len(val_loader))

    if len(train_loader) == 0:
        logger.error("Train loader empty.")
        sys.exit(1)
    if len(val_loader) == 0:
        logger.error("Val loader empty.")
        sys.exit(1)

    # Model
    model = TMJHeatmapDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.2fM", n_params / 1e6)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    best_mae = float("inf")
    no_imp = 0
    history = []

    logger.info("Starting training...")
    print(
        f"\n{'Ep':>4}  {'tr_loss':>8}  {'tr_mae':>7}  │  {'va_loss':>8}  {'va_mae':>7}  {'va_L':>6}  {'va_R':>6}"
    )
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, device, scaler, True, epoch, args.downsample_factor
        )
        val = run_epoch(
            model, val_loader, optimizer, device, scaler, False, epoch, args.downsample_factor
        )
        scheduler.step(val["mae_overall"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:4d}  {tr['loss']:8.4f}  {tr['mae_overall']:7.2f}  │  "
            f"{val['loss']:8.4f}  {val['mae_overall']:7.2f}  "
            f"{val['mae_left']:6.2f}  {val['mae_right']:6.2f}  lr={lr_now:.1e}"
        )

        row = {"epoch": epoch, "lr": lr_now}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in val.items()})
        history.append(row)
        with open(exp_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        if val["mae_overall"] < best_mae:
            best_mae = val["mae_overall"]
            no_imp = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_mae": best_mae,
                    "val_metrics": val,
                },
                exp_dir / "best_model.pth",
            )
            print(
                f"     ✓ best (MAE_ds={best_mae:.2f}px ≈ {best_mae * args.downsample_factor:.0f}px orig)"
            )
        else:
            no_imp += 1
            if args.early_stopping > 0 and no_imp >= args.early_stopping:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info(
        "Done. Best val MAE: %.2f downsampled px (~%.0f original px)",
        best_mae,
        best_mae * args.downsample_factor,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TMJ Heatmap Detector")
    p.add_argument("--split-json", default="data/detector_split.json")
    p.add_argument("--annotations", default="data/roi_annotations")
    p.add_argument("--dataset", default="data/dataset_cbct_public")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--downsample-factor", dest="downsample_factor", type=int, default=6)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=15)
    p.add_argument("--early-stopping", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="experiments")
    main(p.parse_args())
