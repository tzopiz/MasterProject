"""
5-fold StratifiedGroupKFold CV for sagittal binary position (central vs non-central).

Implements the evaluation protocol from
``docs/superpowers/prompts/improve-sag-classifier-metrics.md``:
patient-level groups, stratification from max sagittal label per patient,
Youden threshold fit on **train** predictions only, then metrics on val.

**Splits:** each fold is **train + validation only** — there is no separate test
set inside this routine; add a locked-off test cohort only if you need a final
report that must not be tuned.

Training uses ``TMJBinaryPositionClassifier`` with loss only on the sagittal head
(shared backbone; frontal head receives no gradient).

**Logging:** when ``output_json`` is set, the report includes per-fold
``epoch_history`` (train loss, val AUC, val acc/balanced acc/F1 @ 0.5, LR each
epoch) plus final metrics at the Youden threshold fit on train. The same path
is **rewritten after each completed fold** (atomic write) with ``status`` /
``completed_folds`` / ``n_splits`` so another process or notebook can read
partial CV while training continues; the current fold appears only after it
finishes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _reconcile_cv_paths_if_missing(cfg: "SagittalBinaryCVConfig") -> None:
    """
    If ``cfg`` still points to removed default paths (e.g. old notebook / stale
    kernel), re-resolve manifest and labels from the crop directory parent
    (usually ``.../datasets/tmj``) plus filestore / nested dataset dirs.
    """
    from training.utils.datasphere_env import (
        default_tmj_dataset_dir,
        resolve_detector_crop_dir,
        resolve_labels_path,
        resolve_manifest_path,
    )

    crop = Path(cfg.crop_dir).resolve()
    if not crop.is_dir():
        fixed = resolve_detector_crop_dir(default_tmj_dataset_dir())
        logger.warning("crop_dir missing (%s) — using %s", crop, fixed)
        cfg.crop_dir = str(fixed)
        cfg.dataset_root = str(fixed)
        crop = Path(cfg.crop_dir).resolve()

    mp = Path(cfg.manifest_path)
    lp = Path(cfg.labels_path)
    hub = crop.parent if crop.parent.is_dir() else default_tmj_dataset_dir()

    if not mp.is_file():
        fixed = resolve_manifest_path(hub)
        logger.warning("Manifest not found at %s — using %s", mp, fixed)
        cfg.manifest_path = str(fixed)
    if not lp.is_file():
        fixed = resolve_labels_path(hub)
        logger.warning("Labels not found at %s — using %s", lp, fixed)
        cfg.labels_path = str(fixed)


def _json_sanitize(obj: Any) -> Any:
    """Tuples → lists, NaN/inf → null for strict JSON export."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _build_cv_report_dict(
    cfg: "SagittalBinaryCVConfig",
    fold_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Full JSON payload: config, folds so far, summary, progress fields."""
    from training.utils.binary_metrics import aggregate_fold_metrics

    keys = (
        "val_auc",
        "val_balanced_accuracy",
        "val_f1_minority",
        "val_accuracy_at_threshold",
    )
    summary = aggregate_fold_metrics(fold_rows, keys)
    aucs = [r["val_auc"] for r in fold_rows if not np.isnan(r["val_auc"])]
    n_done = len(fold_rows)
    return {
        "config": asdict(cfg),
        "folds": list(fold_rows),
        "summary": summary,
        "best_fold_auc": float(max(aucs)) if aucs else float("nan"),
        "worst_fold_auc": float(min(aucs)) if aucs else float("nan"),
        "status": "complete" if n_done >= cfg.n_splits else "in_progress",
        "completed_folds": n_done,
        "n_splits": cfg.n_splits,
    }


def _write_cv_report_json_atomic(path_str: str, report: Dict[str, Any]) -> None:
    """Write JSON via a temp file + replace so readers never see a half file."""
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(report), f, indent=2)
    tmp.replace(p)


@dataclass
class SagittalBinaryCVConfig:
    """Hyperparameters and paths for :func:`run_sagittal_binary_cv`."""

    crop_dir: str = "data/detector_crops"
    manifest_path: str = "data/dataset_cbct_public/manifest_private.json"
    labels_path: str = "data/tmj_position_labels.json"
    dataset_root: str = "data/dataset_cbct_public"
    n_splits: int = 5
    seed: int = 42
    epochs: int = 80
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 1e-4
    early_stopping_patience: int = 25
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    max_grad_norm: float = 1.0
    num_workers: int = 0
    gamma: float = 2.0
    # Smaller backbone by default (see improve-sag-classifier-metrics.md)
    features: Tuple[int, ...] = (8, 16, 32, 64)
    fc_hidden: int = 128
    dropout: float = 0.5
    # Train-only augmentations: flips + small rotations + intensity jitter
    train_augment_mode: str = "strong"
    device: Optional[str] = None
    output_json: Optional[str] = None
    tqdm_disable: bool = False


def _ensure_mlservice_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def _focal_alpha_sagittal(train_loader: DataLoader) -> float:
    """``alpha`` for :class:`training.losses.focal_loss.BinaryFocalLoss` (positive weight)."""
    n0 = n1 = 0
    for _, labels in train_loader:
        y = labels.view(-1).long()
        n0 += int((y == 0).sum().item())
        n1 += int((y == 1).sum().item())
    total = n0 + n1
    return n0 / total if total > 0 else 0.5


def _collect_sag_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_chunks: List[torch.Tensor] = []
    labels_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device)
            sag_logit, _ = model(volumes)
            logits_chunks.append(sag_logit.detach().cpu().squeeze(1))
            labels_chunks.append(labels.detach().cpu().float().view(-1))
    return torch.cat(logits_chunks), torch.cat(labels_chunks)


def _train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: SagittalBinaryCVConfig,
) -> Dict[str, Any]:
    from training.utils.binary_metrics import (
        binary_metrics_at_threshold,
        binary_roc_auc,
        youden_optimal_threshold,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_plateau_factor,
        patience=cfg.lr_plateau_patience,
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_score = float("-inf")
    epochs_no_improve = 0
    epoch_history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Fold train ep {epoch}",
            leave=False,
            disable=cfg.tqdm_disable,
        )
        for volumes, labels in pbar:
            volumes = volumes.to(device)
            labels_f = labels.to(device).float().view(-1, 1)
            sag_logit, _ = model(volumes)
            loss = criterion(sag_logit, labels_f)
            optimizer.zero_grad()
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_logits, val_labels = _collect_sag_logits_labels(model, val_loader, device)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_y = val_labels.numpy().astype(int)
        val_auc = binary_roc_auc(val_y, val_probs)

        train_loss_avg = running_loss / max(n_batches, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        m05 = binary_metrics_at_threshold(val_y, val_probs, 0.5)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_avg),
                "val_auc": float(val_auc) if not np.isnan(val_auc) else None,
                "val_accuracy_at_0.5": float(m05["accuracy"]),
                "val_balanced_accuracy_at_0.5": float(m05["balanced_accuracy"]),
                "val_f1_minority_at_0.5": float(m05["f1_minority"]),
                "lr": lr,
            }
        )

        if not np.isnan(val_auc):
            scheduler.step(val_auc)

        last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if np.isnan(val_auc):
            continue

        improved = val_auc > best_score + 1e-8
        if improved:
            best_score = val_auc
            best_epoch = epoch
            best_state = last_state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.early_stopping_patience > 0 and epochs_no_improve >= cfg.early_stopping_patience:
            break

    if best_state is None:
        best_state = last_state or {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = cfg.epochs

    model.load_state_dict(best_state)
    model.to(device)

    # Youden threshold on **train** (prompt: not on val/test)
    tr_logits, tr_labels = _collect_sag_logits_labels(model, train_loader, device)
    tr_probs = torch.sigmoid(tr_logits).numpy()
    tr_y = tr_labels.numpy().astype(int)
    threshold = youden_optimal_threshold(tr_y, tr_probs)

    val_logits, val_labels = _collect_sag_logits_labels(model, val_loader, device)
    val_probs = torch.sigmoid(val_logits).numpy()
    val_y = val_labels.numpy().astype(int)

    val_auc_final = binary_roc_auc(val_y, val_probs)
    val_m = binary_metrics_at_threshold(val_y, val_probs, threshold)

    return {
        "best_epoch": best_epoch,
        "val_auc": val_auc_final,
        "threshold_from_train_youden": threshold,
        "val_balanced_accuracy": val_m["balanced_accuracy"],
        "val_f1_minority": val_m["f1_minority"],
        "val_accuracy_at_threshold": val_m["accuracy"],
        "val_confusion_matrix_at_threshold": val_m["confusion_matrix"],
        "epoch_history": epoch_history,
    }


def run_sagittal_binary_cv(cfg: SagittalBinaryCVConfig) -> Dict[str, Any]:
    _ensure_mlservice_on_path()

    from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier
    from training.datasets.tmj_position_dataset import make_binary_position_loaders
    from training.losses.focal_loss import BinaryFocalLoss
    from training.tmj_position_label_table import (
        binarize_labels,
        build_index,
        iter_stratified_group_kfold_indices,
    )
    from training.utils.seed import make_worker_init_fn, set_seed

    set_seed(cfg.seed)
    worker_init = make_worker_init_fn(cfg.seed) if cfg.num_workers > 0 else None

    if cfg.device:
        device = torch.device(cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _reconcile_cv_paths_if_missing(cfg)

    all_records = build_index(
        manifest_path=cfg.manifest_path,
        labels_path=cfg.labels_path,
        dataset_root=cfg.dataset_root,
    )
    binary_records = binarize_labels(all_records, cfg.crop_dir)

    fold_rows: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        iter_stratified_group_kfold_indices(
            binary_records,
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.seed,
        )
    ):
        logger.info("--- Fold %d / %d ---", fold_idx + 1, cfg.n_splits)
        train_recs = [binary_records[i] for i in tr_idx]
        val_recs = [binary_records[i] for i in va_idx]

        train_loader, val_loader = make_binary_position_loaders(
            train_recs,
            val_recs,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            sagittal_only=True,
            worker_init_fn=worker_init,
            train_augment_mode=cfg.train_augment_mode,
        )

        alpha = _focal_alpha_sagittal(train_loader)
        criterion = BinaryFocalLoss(gamma=cfg.gamma, alpha=alpha)

        model = TMJBinaryPositionClassifier(
            features=list(cfg.features),
            fc_hidden=cfg.fc_hidden,
            dropout=cfg.dropout,
        ).to(device)

        set_seed(cfg.seed + fold_idx)
        fold_out = _train_one_fold(
            model, train_loader, val_loader, criterion, device, cfg
        )
        fold_out["fold"] = fold_idx
        fold_out["n_train_samples"] = len(train_recs)
        fold_out["n_val_samples"] = len(val_recs)
        fold_rows.append(fold_out)

        if cfg.output_json:
            snap = _build_cv_report_dict(cfg, fold_rows)
            _write_cv_report_json_atomic(cfg.output_json, snap)
            logger.info(
                "Wrote CV snapshot (%s, %d/%d folds) → %s",
                snap["status"],
                snap["completed_folds"],
                snap["n_splits"],
                cfg.output_json,
            )

    out = _build_cv_report_dict(cfg, fold_rows)
    if cfg.output_json:
        logger.info("CV finished → %s", cfg.output_json)

    return out


def _print_cv_table(result: Dict[str, Any]) -> None:
    s = result["summary"]
    print("\n=== Sagittal binary CV (5-fold StratifiedGroupKFold) ===")
    if result.get("status") == "in_progress":
        print(
            f"(partial snapshot: {result.get('completed_folds', 0)}/"
            f"{result.get('n_splits', '?')} folds)\n"
        )
    print(f"mean val AUC: {s.get('mean_val_auc', float('nan')):.4f} ± {s.get('std_val_auc', 0):.4f}")
    print(
        f"mean balanced acc: {s.get('mean_val_balanced_accuracy', float('nan')):.4f} "
        f"± {s.get('std_val_balanced_accuracy', 0):.4f}"
    )
    print(
        f"mean F1 (minority): {s.get('mean_val_f1_minority', float('nan')):.4f} "
        f"± {s.get('std_val_f1_minority', 0):.4f}"
    )
    print(f"best fold AUC: {result['best_fold_auc']:.4f}  worst: {result['worst_fold_auc']:.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sagittal binary position — StratifiedGroupKFold CV")
    parser.add_argument("--crop-dir", default="data/detector_crops")
    parser.add_argument("--manifest-private", default="data/dataset_cbct_public/manifest_private.json")
    parser.add_argument("--labels-json", default="data/tmj_position_labels.json")
    parser.add_argument("--dataset-root", default="data/dataset_cbct_public")
    parser.add_argument("--output-json", default="", help="Write full JSON report to this path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--train-augment-mode",
        default="strong",
        choices=("none", "flip_only", "strong"),
        help="Train-time 3D augmentations (val is always unaugmented)",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated backbone widths, e.g. 8,16,32,64 (empty = config default)",
    )
    parser.add_argument("--fc-hidden", type=int, default=0, help="0 = use config default (128)")
    args = parser.parse_args()

    feat: Optional[Tuple[int, ...]] = None
    if args.features.strip():
        feat = tuple(int(x.strip()) for x in args.features.split(",") if x.strip())

    cfg = SagittalBinaryCVConfig(
        crop_dir=args.crop_dir,
        manifest_path=args.manifest_private,
        labels_path=args.labels_json,
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        output_json=args.output_json or None,
        tqdm_disable=False,
        train_augment_mode=args.train_augment_mode,
        features=feat if feat is not None else SagittalBinaryCVConfig.features,
        fc_hidden=args.fc_hidden if args.fc_hidden > 0 else SagittalBinaryCVConfig.fc_hidden,
    )
    result = run_sagittal_binary_cv(cfg)
    _print_cv_table(result)


if __name__ == "__main__":
    main()
