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
``epoch_history`` plus final metrics at the Youden threshold fit on train. The
JSON path is **rewritten after each completed fold** (atomic) with ``status`` /
``completed_folds`` / ``n_splits``. If ``log_epochs_jsonl`` is true, every epoch
appends one line to ``<stem>_epochs.jsonl`` next to ``output_json`` so crashes
still leave a **metric trail** (not weights — use checkpoints for resume).
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
from tqdm.auto import tqdm

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


def _epoch_jsonl_path(output_json: str) -> Path:
    p = Path(output_json)
    return p.with_name(p.stem + "_epochs.jsonl")


def _append_epoch_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_sanitize(row), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    # One summary line per epoch in the notebook (tqdm.write); survives leave=False on batch bar.
    log_each_epoch: bool = True
    # If True and output_json is set, append one JSON object per epoch to ``<stem>_epochs.jsonl``.
    log_epochs_jsonl: bool = True


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
    fold_log_prefix: str = "",
    fold_idx: int = 0,
    epoch_jsonl_path: Optional[Path] = None,
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
        fold_tag = fold_log_prefix if fold_log_prefix else f"fold {fold_idx + 1}/{cfg.n_splits}"
        pbar = tqdm(
            train_loader,
            desc=f"{fold_tag} | ep {epoch} train",
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

        if epoch_jsonl_path is not None:
            row = dict(epoch_history[-1])
            row["fold"] = fold_idx
            _append_epoch_jsonl(epoch_jsonl_path, row)

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

        if cfg.log_each_epoch and not cfg.tqdm_disable:
            auc_s = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
            tag = f"{fold_log_prefix} " if fold_log_prefix else ""
            best_s = f"{best_score:.4f}@{best_epoch}" if best_score > float("-inf") else "—"
            tqdm.write(
                f"{tag}ep {epoch}/{cfg.epochs}  train_loss={train_loss_avg:.4f}  val_auc={auc_s}  "
                f"val_bal_acc@0.5={m05['balanced_accuracy']:.4f}  best_val_auc={best_s}  "
                f"no_improve={epochs_no_improve}/{cfg.early_stopping_patience}  lr={lr:.2e}"
            )

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

    epoch_jsonl_path: Optional[Path] = None
    if cfg.output_json and cfg.log_epochs_jsonl:
        epoch_jsonl_path = _epoch_jsonl_path(cfg.output_json)
        epoch_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        epoch_jsonl_path.write_text("", encoding="utf-8")

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
            model,
            train_loader,
            val_loader,
            criterion,
            device,
            cfg,
            fold_log_prefix=f"fold {fold_idx + 1}/{cfg.n_splits}",
            fold_idx=fold_idx,
            epoch_jsonl_path=epoch_jsonl_path,
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


def analyze_sagittal_cv_result(
    result: Optional[Dict[str, Any]] = None,
    *,
    json_path: Optional[str | Path] = None,
    show_plots: bool = True,
    report_path: Optional[str | Path] = None,
    export_json: bool = True,
    export_csv: bool = True,
    save_curves: bool = True,
) -> Dict[str, Any]:
    """
    Pretty-print and return structured views of :func:`run_sagittal_binary_cv` output.

    Pass either ``result`` (in-memory dict) or ``json_path`` (same schema as written
    to ``output_json``). Optional ``matplotlib`` figures: val AUC and train loss
    vs epoch per fold.

    If ``report_path`` is set, writes next to that path's parent:

    * ``<stem>.txt`` (or the path you pass if it already ends in ``.txt``) — full text report
    * ``<stem>_export.json`` — ``config``, ``summary``, ``fold_summaries``, ``epoch_rows``
    * ``<stem>_folds.csv`` / ``<stem>_epochs.csv`` when pandas is available and ``export_csv``
    * ``<stem>_curves.png`` when matplotlib is available, ``epoch_rows`` non-empty, and ``save_curves``

    Returns a dict with ``fold_summaries``, ``epoch_rows``, ``summary``, ``config``,
    optional ``epochs_df`` / ``folds_df``, and ``report_files_written`` (paths created).
    """
    if result is None:
        if json_path is None:
            raise ValueError("Pass result=... or json_path=...")
        raw = Path(json_path).expanduser().read_text(encoding="utf-8")
        result = json.loads(raw)

    lines: List[str] = []
    written: List[str] = []

    def ln(*parts: Any) -> None:
        if parts:
            lines.append(" ".join(str(p) for p in parts))
        else:
            lines.append("")

    out: Dict[str, Any] = {
        "raw_keys": list(result.keys()),
        "config": result.get("config"),
        "summary": result.get("summary"),
        "status": result.get("status"),
        "completed_folds": result.get("completed_folds"),
        "n_splits": result.get("n_splits"),
        "best_fold_auc": result.get("best_fold_auc"),
        "worst_fold_auc": result.get("worst_fold_auc"),
        "fold_summaries": [],
        "epoch_rows": [],
        "report_files_written": written,
    }

    cfg = result.get("config") or {}
    ln()
    ln("=== Sagittal CV — config (main fields) ===")
    for k in (
        "crop_dir",
        "manifest_path",
        "labels_path",
        "dataset_root",
        "n_splits",
        "seed",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "early_stopping_patience",
        "gamma",
        "features",
        "fc_hidden",
        "dropout",
        "train_augment_mode",
        "num_workers",
        "output_json",
    ):
        if k in cfg:
            ln(f"  {k}: {cfg[k]}")

    if result.get("status"):
        ln()
        ln(
            "status=",
            result["status"],
            "  completed_folds=",
            result.get("completed_folds", "?"),
            "/",
            result.get("n_splits", "?"),
        )

    s = result.get("summary") or {}
    ln()
    ln("=== Cross-fold summary (val, threshold from train Youden) ===")
    for key in sorted(s.keys()):
        v = s[key]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        ln(f"  {key}: {v}")

    ln()
    ln("  best_fold_auc:", result.get("best_fold_auc"), "  worst_fold_auc:", result.get("worst_fold_auc"))

    folds = result.get("folds") or []
    epoch_rows: List[Dict[str, Any]] = []

    ln()
    ln("=== Per-fold (final val metrics @ train-Youden threshold) ===")
    for i, f in enumerate(folds):
        fi = f.get("fold", i)
        cm = f.get("val_confusion_matrix_at_threshold")
        summ = {
            "fold": fi,
            "n_train_samples": f.get("n_train_samples"),
            "n_val_samples": f.get("n_val_samples"),
            "best_epoch": f.get("best_epoch"),
            "val_auc": f.get("val_auc"),
            "threshold_from_train_youden": f.get("threshold_from_train_youden"),
            "val_balanced_accuracy": f.get("val_balanced_accuracy"),
            "val_f1_minority": f.get("val_f1_minority"),
            "val_accuracy_at_threshold": f.get("val_accuracy_at_threshold"),
            "n_epochs_logged": len(f.get("epoch_history") or []),
        }
        out["fold_summaries"].append(summ)
        ln()
        ln(f"--- fold {fi} ---")
        for k, v in summ.items():
            if k == "fold":
                continue
            ln(f"  {k}: {v}")
        ln(f"  val_confusion_matrix_at_threshold [[TN, FP],[FN, TP]]: {cm}")

        for row in f.get("epoch_history") or []:
            er = dict(row)
            er["fold"] = fi
            epoch_rows.append(er)

    out["epoch_rows"] = epoch_rows
    ln()
    ln("=== Epoch history === total rows:", len(epoch_rows), "(all folds)")

    try:
        import pandas as pd

        out["folds_df"] = pd.DataFrame(out["fold_summaries"])
        out["epochs_df"] = pd.DataFrame(epoch_rows)
        ln()
        ln("folds_df:")
        ln(out["folds_df"].to_string(index=False))
        ln()
        ln("epochs_df (head):")
        ln(out["epochs_df"].head(12).to_string(index=False))
        if len(epoch_rows) > 12:
            ln("  ...")
    except ImportError:
        out["folds_df"] = None
        out["epochs_df"] = None
        ln()
        ln("(install pandas for DataFrame tables)")

    report_text = "\n".join(lines)
    print(report_text)

    if report_path is not None:
        rp = Path(report_path).expanduser()
        if rp.suffix == "":
            rp = rp.with_suffix(".txt")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report_text + "\n", encoding="utf-8")
        written.append(str(rp.resolve()))

        stem = rp.stem

        if export_json:
            export_payload = {
                "config": result.get("config"),
                "summary": result.get("summary"),
                "status": result.get("status"),
                "completed_folds": result.get("completed_folds"),
                "n_splits": result.get("n_splits"),
                "best_fold_auc": result.get("best_fold_auc"),
                "worst_fold_auc": result.get("worst_fold_auc"),
                "fold_summaries": out["fold_summaries"],
                "epoch_rows": out["epoch_rows"],
            }
            jp = rp.parent / f"{stem}_export.json"
            with open(jp, "w", encoding="utf-8") as jf:
                json.dump(_json_sanitize(export_payload), jf, indent=2)
            written.append(str(jp.resolve()))

        if export_csv and out.get("folds_df") is not None and out.get("epochs_df") is not None:
            folds_csv = rp.parent / f"{stem}_folds.csv"
            epochs_csv = rp.parent / f"{stem}_epochs.csv"
            out["folds_df"].to_csv(folds_csv, index=False)
            out["epochs_df"].to_csv(epochs_csv, index=False)
            written.append(str(folds_csv.resolve()))
            written.append(str(epochs_csv.resolve()))

    if epoch_rows:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(install matplotlib for curve plots / PNG export)")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
            for f in folds:
                fi = f.get("fold", 0)
                h = f.get("epoch_history") or []
                if not h:
                    continue
                xs = [r["epoch"] for r in h]
                val_aucs = [r["val_auc"] for r in h]
                losses = [r["train_loss"] for r in h]
                axes[0].plot(xs, val_aucs, marker="o", ms=2, lw=1, label=f"fold {fi}")
                axes[1].plot(xs, losses, marker="o", ms=2, lw=1, label=f"fold {fi}")

            axes[0].set_ylabel("val ROC-AUC")
            axes[0].set_title("Validation ROC-AUC per epoch")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="lower right", fontsize=8)

            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("train loss (mean batch)")
            axes[1].set_title("Train loss @ epoch")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right", fontsize=8)

            plt.tight_layout()

            if report_path is not None and save_curves:
                rpp = Path(report_path).expanduser()
                if rpp.suffix == "":
                    rpp = rpp.with_suffix(".txt")
                curves_path = rpp.parent / f"{rpp.stem}_curves.png"
                fig.savefig(curves_path, dpi=150, bbox_inches="tight")
                written.append(str(curves_path.resolve()))

            if show_plots:
                plt.show()
            plt.close(fig)

    if written:
        print("\n--- report files ---")
        for w in written:
            print(" ", w)

    return out


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
