"""
Shared StratifiedGroupKFold CV for per-side sagittal binary or3-class ROI position.

Patient groups come from a strict source-qualified index (explicit legacy name
join remains binary-only). Stratification uses maximum sagittal class perpatient;
allclasses must occur in everytrain/val fold. Binary selects valAUC and calibrates
Youden on unaugmented ordered training rows. Multiclass selects valmacroF1 and
uses argmax without calibration. Both are developmentCV estimates.

**Splits:** each fold is **train + validation only** — there is no separate test
set inside this routine; add a locked-off test cohort only if you need a final
report that must not be tuned.

Training reuses ``TMJBinaryPositionClassifier`` with sagittal-only loss. Binary
frontal head is untrained; multiclass omits it. Multiclass spatialaugments refuse.

**Artifacts:** every run gets a new protected directory. Public report and
metric trail contain aggregates; patient membership/predictions and replay
configuration remain private. Selected fold checkpoints carry the sagittal-only
contract. Failure preserves completed folds; this is not a resume mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _validate_legacy_cv_paths(cfg: "SagittalBinaryCVConfig") -> None:
    """Explicit legacy paths are required; discovery belongs to the path helper."""
    from training.utils.datasphere_env import validate_explicit_environment

    validate_explicit_environment()
    try:
        if not all(Path(value).is_dir() for value in (cfg.crop_dir, cfg.dataset_root)) or not all(
            Path(value).is_file() for value in (cfg.manifest_path, cfg.labels_path)
        ):
            raise ValueError("invalid_explicit_path")
    except (OSError, TypeError, RuntimeError):
        raise ValueError("invalid_explicit_path") from None


def _json_sanitize(obj: Any) -> Any:
    """Tuples → lists, NaN/inf → null for strict JSON export."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _publishable_cv_config(config: Any) -> Dict[str, Any]:
    """One narrow public allowlist, including types, for current/legacy configs."""
    if not isinstance(config, dict):
        return {}
    public = {}
    for key in (
        "n_splits",
        "seed",
        "epochs",
        "batch_size",
        "early_stopping_patience",
        "lr_plateau_patience",
        "num_workers",
        "fc_hidden",
    ):
        value = config.get(key)
        if type(value) is int and value >= 0:
            public[key] = value
    for key in ("lr", "weight_decay", "lr_plateau_factor", "max_grad_norm", "gamma", "dropout"):
        value = config.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            public[key] = value
    features = config.get("features")
    if (
        isinstance(features, (list, tuple))
        and features
        and all(type(v) is int and v > 0 for v in features)
    ):
        public["features"] = list(features)
    mode = config.get("train_augment_mode")
    if isinstance(mode, str) and mode in ("none", "flip_only", "strong"):
        public["train_augment_mode"] = mode
    if type(config.get("legacy_name_join")) is bool:
        public["legacy_name_join"] = config["legacy_name_join"]
    if config.get("mode") in ("binary", "multiclass"):
        public["mode"] = config["mode"]
    return public


def _publishable_analyzer_report(result: Any) -> Dict[str, Any]:
    """Sanitize known aggregate fields before text/JSON/CSV/plot consumers.

    Unknown fields are dropped. Invalid known metric/label types refuse safely;
    historical valid aggregate reports and their legacy positive-F1 alias remain.
    """

    def fail():
        raise ValueError("invalid_cv_report")

    def number(value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            fail()
        return float(value) if math.isfinite(value) else None

    def integer(value):
        if value is None:
            return None
        if type(value) is not int or value < 0:
            fail()
        return value

    def enum(value, allowed, default=None):
        if value is None:
            return default
        if not isinstance(value, str) or value not in allowed:
            fail()
        return value

    def mapping(value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            fail()
        return value

    def numeric_fields(value, keys):
        obj = mapping(value)
        return {key: number(obj[key]) for key in keys if key in obj}

    def support(value):
        obj = mapping(value)
        return {key: integer(obj[key]) for key in map(str, range(classes)) if key in obj}

    def confusion(value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != classes:
            fail()
        if any(not isinstance(row, (list, tuple)) or len(row) != classes for row in value):
            fail()
        return [[integer(v) for v in row] for row in value]

    def per_class(value):
        obj = mapping(value)
        result = {}
        for label in map(str, range(classes)):
            if label in obj:
                row = mapping(obj[label])
                result[label] = numeric_fields(row, ("precision", "recall", "f1"))
                result[label]["support"] = integer(row.get("support"))
        return result

    def baseline(value):
        obj = mapping(value)
        clean = numeric_fields(
            obj,
            (
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1_positive",
                "f1_minority",
                "auc",
                "macro_f1",
            ),
        )
        for key in ("predicted_class", "positive_class"):
            if key in obj:
                clean[key] = integer(obj[key])
                if clean[key] not in range(classes):
                    fail()
        if "source" in obj:
            clean["source"] = enum(obj["source"], ("training",))
        if "support" in obj:
            clean["support"] = support(obj["support"])
        if "confusion_matrix" in obj:
            clean["confusion_matrix"] = confusion(obj["confusion_matrix"])
        if "per_class" in obj:
            clean["per_class"] = per_class(obj["per_class"])
        return clean

    result = mapping(result)
    mode = enum(result.get("mode"), ("binary", "multiclass"), "binary")
    classes = 3 if mode == "multiclass" else 2
    score_key = "val_macro_f1" if mode == "multiclass" else "val_auc"
    from models.tmj_binary_position_classifier import (
        BINARY_CLASS_SEMANTICS,
        MULTICLASS_CLASS_SEMANTICS,
    )

    semantics = MULTICLASS_CLASS_SEMANTICS if mode == "multiclass" else BINARY_CLASS_SEMANTICS
    if result.get("class_semantics", semantics) != semantics:
        fail()
    if (
        type(result.get("num_classes", classes)) is not int
        or result.get("num_classes", classes) != classes
    ):
        fail()
    clean = {
        "mode": mode,
        "num_classes": classes,
        "class_semantics": semantics,
        "model_selection_metric": enum(
            result.get("model_selection_metric"),
            ("macro_f1",) if mode == "multiclass" else ("auc",),
            "macro_f1" if mode == "multiclass" else "auc",
        ),
        "config": _publishable_cv_config(result.get("config")),
        "assessment": enum(result.get("assessment"), ("development_cv",), "development_cv"),
        "model_selection_source": enum(
            result.get("model_selection_source"), ("validation",), "validation"
        ),
        "calibration_source": enum(
            result.get("calibration_source"),
            ("not_applicable",)
            if mode == "multiclass"
            else ("training_unaugmented", "unknown_legacy"),
            "not_applicable" if mode == "multiclass" else "unknown_legacy",
        ),
        "status": enum(result.get("status"), ("complete", "in_progress", "failed", "interrupted")),
    }
    for key in ("completed_folds", "n_splits"):
        clean[key] = integer(result.get(key))
    for key in (
        ("best_fold_macro_f1", "worst_fold_macro_f1")
        if mode == "multiclass"
        else ("best_fold_auc", "worst_fold_auc")
    ):
        clean[key] = number(result.get(key))
    fold_metrics = (
        "val_auc",
        "val_balanced_accuracy",
        "val_f1_positive",
        "val_f1_minority",
        "val_sensitivity",
        "val_specificity",
        "val_accuracy_at_threshold",
    )
    if mode == "multiclass":
        fold_metrics = ("val_macro_f1", "val_accuracy", "val_balanced_accuracy")
    clean["summary"] = numeric_fields(
        result.get("summary"),
        tuple(f"{prefix}_{key}" for key in fold_metrics for prefix in ("mean", "std")),
    )
    folds = result.get("folds", [])
    if not isinstance(folds, list):
        fail()
    clean["folds"] = []
    for index, value in enumerate(folds):
        obj = mapping(value)
        fold = numeric_fields(obj, (*fold_metrics, "threshold_from_train_youden"))
        fold["fold"] = integer(obj.get("fold", index))
        for key in ("n_train_samples", "n_val_samples", "best_epoch"):
            if key in obj:
                fold[key] = integer(obj[key])
        for key in ("val_support", "train_support"):
            if key in obj:
                fold[key] = support(obj[key])
        if "val_confusion_matrix_at_threshold" in obj:
            fold["val_confusion_matrix_at_threshold"] = confusion(
                obj["val_confusion_matrix_at_threshold"]
            )
        if "val_confusion_matrix" in obj:
            fold["val_confusion_matrix"] = confusion(obj["val_confusion_matrix"])
        if "val_per_class" in obj:
            fold["val_per_class"] = per_class(obj["val_per_class"])
        if "train_majority_baseline" in obj:
            fold["train_majority_baseline"] = baseline(obj["train_majority_baseline"])
        history = obj.get("epoch_history", [])
        if not isinstance(history, list):
            fail()
        fold["epoch_history"] = []
        for value in history:
            epoch = numeric_fields(
                value,
                (
                    "train_loss",
                    "val_auc",
                    "val_accuracy_at_0.5",
                    "val_balanced_accuracy_at_0.5",
                    "val_f1_positive_at_0.5",
                    "val_f1_minority_at_0.5",
                    "lr",
                    "val_macro_f1",
                    "val_accuracy",
                    "val_balanced_accuracy",
                ),
            )
            row = mapping(value)
            epoch["epoch"] = integer(row.get("epoch"))
            if epoch["epoch"] is None or not all(key in epoch for key in ("train_loss", score_key)):
                fail()
            fold["epoch_history"].append(epoch)
        clean["folds"].append(fold)
    return clean


def _build_cv_report_dict(
    cfg: "SagittalBinaryCVConfig",
    fold_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Full JSON payload: config, folds so far, summary, progress fields."""
    from training.utils.binary_metrics import aggregate_fold_metrics

    if cfg.mode == "multiclass":
        keys = ("val_macro_f1", "val_accuracy", "val_balanced_accuracy")
        selection = "macro_f1"
    else:
        keys = (
            "val_auc",
            "val_balanced_accuracy",
            "val_f1_positive",
            "val_sensitivity",
            "val_specificity",
            "val_accuracy_at_threshold",
        )
        selection = "auc"
    summary = aggregate_fold_metrics(fold_rows, keys)
    scores = [r[f"val_{selection}"] for r in fold_rows if not np.isnan(r[f"val_{selection}"])]
    n_done = len(fold_rows)
    report = {
        "mode": cfg.mode,
        "num_classes": _num_classes(cfg),
        "config": _publishable_cv_config(asdict(cfg)),
        "assessment": "development_cv",
        "model_selection_source": "validation",
        "model_selection_metric": selection,
        "selection_tie_rule": "earliest-epoch-within-1e-8",
        "calibration_source": "not_applicable"
        if cfg.mode == "multiclass"
        else "training_unaugmented",
        "loss": "cross_entropy" if cfg.mode == "multiclass" else "binary_focal",
        "independent_test": False,
        "folds": list(fold_rows),
        "summary": summary,
        f"best_fold_{selection}": float(max(scores)) if scores else float("nan"),
        f"worst_fold_{selection}": float(min(scores)) if scores else float("nan"),
        "status": "complete" if n_done >= cfg.n_splits else "in_progress",
        "completed_folds": n_done,
        "n_splits": cfg.n_splits,
    }
    if cfg.mode == "binary":
        report["positive_class"] = 1
    from models.tmj_binary_position_classifier import (
        BINARY_CLASS_SEMANTICS,
        MULTICLASS_CLASS_SEMANTICS,
    )

    report["class_semantics"] = (
        MULTICLASS_CLASS_SEMANTICS if cfg.mode == "multiclass" else BINARY_CLASS_SEMANTICS
    )
    return report


def _write_cv_report_json_atomic(path_str: str, report: Dict[str, Any]) -> None:
    """Write JSON via a temp file + replace so readers never see a half file."""
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with os.fdopen(
        os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8"
    ) as f:
        json.dump(_json_sanitize(report), f, indent=2, allow_nan=False)
    tmp.replace(p)


def _append_epoch_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_sanitize(row), ensure_ascii=False, allow_nan=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class SagittalBinaryCVConfig:
    """Hyperparameters and paths for :func:`run_sagittal_binary_cv`."""

    mode: str = "binary"
    input_path: Optional[str] = None
    legacy_name_join: bool = False
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
    train_augment_mode: Optional[str] = None
    device: Optional[str] = None
    output_json: Optional[str] = None
    output_dir: str = "experiments"
    run_id: Optional[str] = None
    tqdm_disable: bool = False
    # One summary line per epoch in the notebook (tqdm.write); survives leave=False on batch bar.
    log_each_epoch: bool = True
    # Append aggregated per-epoch metrics to the protected run directory.
    log_epochs_jsonl: bool = True

    def __post_init__(self):
        if self.train_augment_mode is None:
            self.train_augment_mode = "none" if self.mode == "multiclass" else "strong"


def _num_classes(cfg):
    return 3 if cfg.mode == "multiclass" else 2


def _side_records(records, cfg):
    """Reuse side/crop/identity expansion; strict intake already mapped0/1/2."""
    from training.tmj_position_label_table import binarize_labels

    rows = binarize_labels(records, cfg.crop_dir)
    if cfg.mode == "multiclass":
        for index, study in enumerate(records):
            for offset, side in enumerate(("left", "right")):
                row = rows[2 * index + offset]
                row["sag"] = study[f"sag_{side}"]
                if f"fr_{side}" in study:
                    row["fr"] = study[f"fr_{side}"]
    return rows


def _validate_cv_config(cfg: SagittalBinaryCVConfig) -> None:
    """Fail on unusable configuration before touching data or constructing a model."""
    _ensure_mlservice_on_path()
    if cfg.mode not in ("binary", "multiclass"):
        raise ValueError("invalid_research_mode")
    if cfg.mode == "multiclass" and (cfg.legacy_name_join or cfg.train_augment_mode != "none"):
        raise ValueError("incompatible_multiclass_inputs_or_augmentation")
    for name, minimum in (
        ("n_splits", 2),
        ("epochs", 1),
        ("batch_size", 1),
        ("num_workers", 0),
        ("fc_hidden", 1),
        ("early_stopping_patience", 0),
        ("lr_plateau_patience", 0),
    ):
        value = getattr(cfg, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (
        not isinstance(cfg.features, (list, tuple))
        or not cfg.features
        or any(type(v) is not int or v <= 0 for v in cfg.features)
    ):
        raise ValueError("features must contain positive integers")
    from models.blocks import validate_research_architecture

    validate_research_architecture(cfg.features, fc_hidden=cfg.fc_hidden)
    for name in ("lr", "weight_decay", "gamma", "max_grad_norm", "dropout", "lr_plateau_factor"):
        value = getattr(cfg, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite numeric")
    if cfg.lr <= 0 or min(cfg.weight_decay, cfg.gamma, cfg.max_grad_norm) < 0:
        raise ValueError("lr must be positive; regularization parameters must be nonnegative")
    if not 0 <= cfg.dropout < 1 or not 0 < cfg.lr_plateau_factor < 1:
        raise ValueError("dropout must be in [0, 1); lr_plateau_factor in (0, 1)")
    if cfg.train_augment_mode not in ("none", "flip_only", "strong"):
        raise ValueError("Unknown train_augment_mode")
    if type(cfg.legacy_name_join) is not bool:
        raise ValueError("legacy_name_join must be boolean")
    if cfg.input_path is not None and (
        not isinstance(cfg.input_path, str) or not cfg.input_path.strip()
    ):
        raise ValueError("input_path must be a nonempty path string")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir.strip():
        raise ValueError("output_dir must be a nonempty path string")
    if cfg.run_id is not None and (
        not isinstance(cfg.run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cfg.run_id)
    ):
        raise ValueError("invalid_run_id")
    if cfg.output_json and cfg.run_id:
        raise ValueError("Choose output_json alias or run_id")
    if cfg.input_path and cfg.legacy_name_join:
        raise ValueError("Choose canonical input_path or explicit legacy_name_join")


def _validated_cv_folds(records: List[Dict[str, Any]], cfg: SagittalBinaryCVConfig):
    from training.tmj_position_label_table import (
        iter_stratified_group_kfold_indices,
        patient_group_key,
    )

    if not records:
        raise ValueError("CV requires nonempty records")
    groups = []
    for record in records:
        if not cfg.legacy_name_join and not all(record.get(k) for k in ("source_id", "patient_id")):
            raise ValueError("CV requires verified patient groups")
        groups.append(patient_group_key(record))
        if type(record.get("sag")) is not int or record["sag"] not in range(_num_classes(cfg)):
            raise ValueError("CV requires mode-compatible sagittal labels")
    if len(set(groups)) < cfg.n_splits:
        raise ValueError("n_splits exceeds the number of patient groups")
    for label in range(_num_classes(cfg)):
        if (
            len({group for group, rec in zip(groups, records) if rec["sag"] == label})
            < cfg.n_splits
        ):
            raise ValueError("Each class needs enough patient groups for every validation fold")
    folds = list(
        iter_stratified_group_kfold_indices(
            records, n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
        )
    )
    if len(folds) != cfg.n_splits:
        raise ValueError("CV did not produce all planned folds")
    validation_indices = []
    for train_indices, val_indices in folds:
        if not len(train_indices) or not len(val_indices):
            raise ValueError("CV folds must have nonempty train and validation")
        if {groups[i] for i in train_indices} & {groups[i] for i in val_indices}:
            raise ValueError("CV patient groups overlap")
        if set(train_indices) | set(val_indices) != set(range(len(records))):
            raise ValueError("CV fold does not cover all records")
        for indices in (train_indices, val_indices):
            if {records[i]["sag"] for i in indices} != set(range(_num_classes(cfg))):
                raise ValueError("Every CV train/validation fold must contain all classes")
        validation_indices.extend(val_indices)
    if sorted(validation_indices) != list(range(len(records))):
        raise ValueError("CV validation folds must cover every record exactly once")
    return folds


def _make_calibration_loader(
    records: List[Dict[str, Any]], cfg: SagittalBinaryCVConfig
) -> DataLoader:
    from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
    from training.utils.seed import make_worker_init_fn

    return DataLoader(
        TMJBinaryPositionDataset(
            records, is_train=False, sagittal_only=True, num_classes=_num_classes(cfg)
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=make_worker_init_fn(cfg.seed) if cfg.num_workers else None,
    )


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
    calibration_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    from training.utils.binary_metrics import (
        binary_metrics_at_threshold,
        binary_roc_auc,
        youden_optimal_threshold,
    )

    if calibration_loader is None:
        raise ValueError("An unaugmented training calibration loader is required")

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
            labels_f = (
                labels.to(device).long().view(-1)
                if cfg.mode == "multiclass"
                else labels.to(device).float().view(-1, 1)
            )
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
        val_y = val_labels.numpy().astype(int)
        train_loss_avg = running_loss / max(n_batches, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        if cfg.mode == "multiclass":
            from training.utils.binary_metrics import multiclass_metrics

            val_probs = torch.softmax(val_logits, dim=1).numpy()
            validation_metrics = multiclass_metrics(val_y, val_probs)
            val_score = validation_metrics["macro_f1"]
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss_avg),
                "val_macro_f1": val_score,
                "val_accuracy": validation_metrics["accuracy"],
                "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "lr": lr,
            }
        else:
            val_probs = torch.sigmoid(val_logits).numpy()
            val_score = binary_roc_auc(val_y, val_probs)
            m05 = binary_metrics_at_threshold(val_y, val_probs, 0.5)
            row = {
                "epoch": epoch,
                "train_loss": float(train_loss_avg),
                "val_auc": float(val_score) if not np.isnan(val_score) else None,
                "val_accuracy_at_0.5": float(m05["accuracy"]),
                "val_balanced_accuracy_at_0.5": float(m05["balanced_accuracy"]),
                "val_f1_positive_at_0.5": float(m05["f1_positive"]),
                "lr": lr,
            }
        epoch_history.append(row)

        if epoch_jsonl_path is not None:
            row = dict(epoch_history[-1])
            row["fold"] = fold_idx
            _append_epoch_jsonl(epoch_jsonl_path, row)

        if not np.isnan(val_score):
            scheduler.step(val_score)
        last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if np.isnan(val_score):
            continue
        # Scores within1e-8 retain earliest epoch, for both modes.
        improved = val_score > best_score + 1e-8
        if improved:
            best_score, best_epoch, best_state, epochs_no_improve = val_score, epoch, last_state, 0
        else:
            epochs_no_improve += 1
        if cfg.log_each_epoch and not cfg.tqdm_disable:
            metric = "macro_f1" if cfg.mode == "multiclass" else "auc"
            tqdm.write(
                f"{fold_log_prefix} ep {epoch}/{cfg.epochs} train_loss={train_loss_avg:.4f} val_{metric}={val_score:.4f} best={best_score:.4f}@{best_epoch} no_improve={epochs_no_improve}/{cfg.early_stopping_patience} lr={lr:.2e}"
            )

        if cfg.early_stopping_patience > 0 and epochs_no_improve >= cfg.early_stopping_patience:
            break

    if best_state is None:
        best_state = last_state or {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = cfg.epochs

    model.load_state_dict(best_state)
    model.to(device)

    if cfg.mode == "multiclass":
        tr_logits, tr_labels = _collect_sag_logits_labels(model, calibration_loader, device)
        val_logits, val_labels = _collect_sag_logits_labels(model, val_loader, device)
        tr_probs, val_probs = (
            torch.softmax(tr_logits, dim=1).numpy(),
            torch.softmax(val_logits, dim=1).numpy(),
        )
        tr_y, val_y = tr_labels.numpy().astype(int), val_labels.numpy().astype(int)
        return {
            "best_epoch": best_epoch,
            **_multiclass_fold_metrics(tr_y, val_y, val_probs),
            "epoch_history": epoch_history,
            "_predictions": {
                "training": {
                    "labels": tr_y.tolist(),
                    "logits": tr_logits.tolist(),
                    "probabilities": tr_probs.tolist(),
                },
                "validation": {
                    "labels": val_y.tolist(),
                    "logits": val_logits.tolist(),
                    "probabilities": val_probs.tolist(),
                },
            },
        }

    # Selection used validation; calibration uses deterministic training inputs.
    tr_logits, tr_labels = _collect_sag_logits_labels(model, calibration_loader, device)
    tr_probs = torch.sigmoid(tr_logits).numpy()
    tr_y = tr_labels.numpy().astype(int)
    threshold = youden_optimal_threshold(tr_y, tr_probs)

    val_logits, val_labels = _collect_sag_logits_labels(model, val_loader, device)
    val_probs = torch.sigmoid(val_logits).numpy()
    val_y = val_labels.numpy().astype(int)

    val_auc_final = binary_roc_auc(val_y, val_probs)
    val_m = binary_metrics_at_threshold(val_y, val_probs, threshold)

    majority_class = int(np.sum(tr_y == 1) > np.sum(tr_y == 0))
    baseline_probs = np.full(val_y.shape, majority_class, dtype=float)
    baseline = binary_metrics_at_threshold(val_y, baseline_probs, 0.5)
    baseline.update(
        {
            "source": "training",
            "predicted_class": majority_class,
            "auc": binary_roc_auc(val_y, baseline_probs),
        }
    )

    return {
        "best_epoch": best_epoch,
        "val_auc": val_auc_final,
        "threshold_from_train_youden": threshold,
        "val_balanced_accuracy": val_m["balanced_accuracy"],
        "val_f1_positive": val_m["f1_positive"],
        "val_sensitivity": val_m["sensitivity"],
        "val_specificity": val_m["specificity"],
        "val_support": val_m["support"],
        "train_support": {"0": int(np.sum(tr_y == 0)), "1": int(np.sum(tr_y == 1))},
        "train_majority_baseline": baseline,
        "val_accuracy_at_threshold": val_m["accuracy"],
        "val_confusion_matrix_at_threshold": val_m["confusion_matrix"],
        "epoch_history": epoch_history,
        "_predictions": {
            "training": {
                "labels": tr_y.tolist(),
                "logits": tr_logits.tolist(),
                "probabilities": tr_probs.tolist(),
            },
            "validation": {
                "labels": val_y.tolist(),
                "logits": val_logits.tolist(),
                "probabilities": val_probs.tolist(),
            },
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _record_keys(record: Dict[str, Any]) -> Dict[str, str]:
    from training.tmj_position_label_table import patient_group_key

    group = patient_group_key(record)
    return {
        "group_key": _digest_json(["tmj-private-group-v1", group]),
        "sample_key": _digest_json(
            ["tmj-private-sample-v1", group, record["study_id"], record["side"]]
        ),
    }


def _private_record(record: Dict[str, Any]) -> Dict[str, Any]:
    passport = Path(str(record["crop_path"]) + ".passport.json")
    return {
        **record,
        **_record_keys(record),
        "crop_sha256": _file_sha256(Path(record["crop_path"])),
        "passport_sha256": _file_sha256(passport) if passport.is_file() else None,
    }


def _artifact_ref(path: Path) -> Dict[str, str]:
    return {"file": path.name, "sha256": _file_sha256(path)}


def _create_run_directory(cfg: SagittalBinaryCVConfig) -> Path:
    if cfg.output_json:
        alias = Path(cfg.output_json)
        if alias.exists():
            raise ValueError("run_directory_exists")
        directory = alias.parent / alias.stem
    else:
        directory = Path(cfg.output_dir) / (cfg.run_id or uuid.uuid4().hex)
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError("run_directory_exists") from None
    except (OSError, ValueError):
        raise ValueError("invalid_run_directory") from None
    return directory


def _code_fingerprint() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "training/sagittal_binary_cv.py",
        "tools/run_research.py",
        "training/utils/datasphere_env.py",
        "models/tmj_binary_position_classifier.py",
        "models/blocks.py",
        "training/tmj_position_label_table.py",
        "training/datasets/tmj_position_dataset.py",
        "training/utils/seed.py",
        "training/utils/binary_metrics.py",
        "training/utils/volume_aug_3d.py",
        "training/losses/focal_loss.py",
    )
    files = {path: _file_sha256(root / path) for path in paths}
    if (root / "training/roi_provenance.py").exists():
        files["training/roi_provenance.py"] = _file_sha256(root / "training/roi_provenance.py")
    return {"files": files, "sha256": _digest_json(files)}


def _runtime_fingerprint(device: torch.device) -> Dict[str, Any]:
    versions = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "nibabel"):
        versions[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "packages": versions,
        "device_type": device.type,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def _aligned_prediction_rows(
    predictions, train_records, val_records, fold, threshold, mode="binary"
):
    rows = []
    for role, records in (("training", train_records), ("validation", val_records)):
        arrays = predictions[role]
        if any(len(values) != len(records) for values in arrays.values()):
            raise ValueError("prediction_alignment_failed")
        for index, record in enumerate(records):
            label, logit, probability = (
                arrays[key][index] for key in ("labels", "logits", "probabilities")
            )
            if label != record["sag"]:
                raise ValueError("prediction_alignment_mismatch")
            shared = {
                **_record_keys(record),
                "crop_sha256": _file_sha256(Path(record["crop_path"])),
                "fold": fold,
                "role": role,
                "side": record["side"],
                "label": int(label),
                "mode": mode,
            }
            if mode == "multiclass":
                if (
                    len(logit) != 3
                    or len(probability) != 3
                    or not np.isfinite(logit).all()
                    or not np.isfinite(probability).all()
                ):
                    raise ValueError("prediction_alignment_mismatch")
                rows.append(
                    {
                        **shared,
                        "logits": list(logit),
                        "probabilities": list(probability),
                        "decision": int(np.argmax(probability)),
                    }
                )
            else:
                if not math.isfinite(logit) or not math.isfinite(probability):
                    raise ValueError("prediction_alignment_mismatch")
                rows.append(
                    {
                        **shared,
                        "logit": float(logit),
                        "probability": float(probability),
                        "threshold": float(threshold),
                        "decision": int(probability >= threshold),
                    }
                )
    return rows


def _multiclass_fold_metrics(train_y, val_y, val_probabilities):
    from training.utils.binary_metrics import multiclass_metrics

    metrics = multiclass_metrics(val_y, val_probabilities)
    counts = np.bincount(np.asarray(train_y, dtype=int), minlength=3)
    majority = int(np.argmax(counts))
    baseline = multiclass_metrics(val_y, np.eye(3)[np.full(len(val_y), majority, dtype=int)])
    baseline.update(source="training", predicted_class=majority)
    return {
        "val_accuracy": metrics["accuracy"],
        "val_balanced_accuracy": metrics["balanced_accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_per_class": metrics["per_class"],
        "val_support": metrics["support"],
        "train_support": {str(i): int(counts[i]) for i in range(3)},
        "val_confusion_matrix": metrics["confusion_matrix"],
        "train_majority_baseline": baseline,
    }


def recompute_fold_metrics(rows: List[Dict[str, Any]], mode=None) -> Dict[str, Any]:
    """Recompute selected-checkpoint metrics solely from PRIVATE prediction rows."""
    from training.utils.binary_metrics import (
        binary_metrics_at_threshold,
        binary_roc_auc,
        youden_optimal_threshold,
    )

    mode = mode or (rows[0].get("mode", "binary") if rows else "binary")
    if mode not in ("binary", "multiclass") or any(
        row.get("mode", "binary") != mode for row in rows
    ):
        raise ValueError("prediction_mode_mismatch")
    if mode == "multiclass":
        from training.utils.binary_metrics import multiclass_metrics

        train = [row for row in rows if row["role"] == "training"]
        val = [row for row in rows if row["role"] == "validation"]
        if not train or not val or len(train) + len(val) != len(rows):
            raise ValueError("invalid_multiclass_predictions")
        labels = np.asarray([row["label"] for row in rows])
        probabilities = np.asarray([row["probabilities"] for row in rows])
        multiclass_metrics(labels, probabilities)
        if any(row["decision"] != int(np.argmax(row["probabilities"])) for row in rows):
            raise ValueError("prediction_decision_mismatch")
        return _multiclass_fold_metrics(
            np.asarray([row["label"] for row in train]),
            np.asarray([row["label"] for row in val]),
            np.asarray([row["probabilities"] for row in val]),
        )
    train = [row for row in rows if row["role"] == "training"]
    val = [row for row in rows if row["role"] == "validation"]
    tr_y = np.asarray([row["label"] for row in train])
    va_y = np.asarray([row["label"] for row in val])
    tr_p = np.asarray([row["probability"] for row in train])
    va_p = np.asarray([row["probability"] for row in val])
    threshold = youden_optimal_threshold(tr_y, tr_p)
    if any(
        row["threshold"] != threshold or row["decision"] != int(row["probability"] >= threshold)
        for row in rows
    ):
        raise ValueError("prediction_threshold_mismatch")
    metrics = binary_metrics_at_threshold(va_y, va_p, threshold)
    majority = int(np.sum(tr_y == 1) > np.sum(tr_y == 0))
    baseline_p = np.full(va_y.shape, majority, dtype=float)
    baseline = binary_metrics_at_threshold(va_y, baseline_p, 0.5)
    baseline.update(
        {"source": "training", "predicted_class": majority, "auc": binary_roc_auc(va_y, baseline_p)}
    )
    return {
        "threshold_from_train_youden": threshold,
        "val_auc": binary_roc_auc(va_y, va_p),
        "val_balanced_accuracy": metrics["balanced_accuracy"],
        "val_f1_positive": metrics["f1_positive"],
        "val_sensitivity": metrics["sensitivity"],
        "val_specificity": metrics["specificity"],
        "val_support": metrics["support"],
        "train_support": {"0": int(np.sum(tr_y == 0)), "1": int(np.sum(tr_y == 1))},
        "train_majority_baseline": baseline,
        "val_accuracy_at_threshold": metrics["accuracy"],
        "val_confusion_matrix_at_threshold": metrics["confusion_matrix"],
    }


def _save_fold_artifacts(run_dir, model, cfg, fold, fold_out, rows):
    from models.tmj_binary_position_classifier import (
        BINARY_CHECKPOINT_FAMILY,
        BINARY_CLASS_SEMANTICS,
        BINARY_PREPROCESSING,
        MULTICLASS_CHECKPOINT_FAMILY,
        MULTICLASS_CLASS_SEMANTICS,
        MULTICLASS_DECISION_RULE,
    )

    checkpoint = run_dir / f"fold_{fold:02d}_checkpoint.pth"
    temporary = checkpoint.with_suffix(".tmp")
    payload = {
        "schema_version": 1,
        "family": BINARY_CHECKPOINT_FAMILY,
        "model_kwargs": {
            "in_channels": 1,
            "features": list(cfg.features),
            "fc_hidden": cfg.fc_hidden,
            "dropout": cfg.dropout,
        },
        "trained_tasks": ["sagittal"],
        "preprocessing": BINARY_PREPROCESSING,
        "class_semantics": BINARY_CLASS_SEMANTICS,
        "threshold": float(fold_out["threshold_from_train_youden"])
        if cfg.mode == "binary"
        else None,
        "decision_rule": ">=",
        "fold": fold,
        "best_epoch": fold_out["best_epoch"],
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
    }
    if cfg.mode == "multiclass":
        payload.update(
            family=MULTICLASS_CHECKPOINT_FAMILY,
            mode="multiclass",
            num_classes=3,
            class_semantics=MULTICLASS_CLASS_SEMANTICS,
            threshold=None,
            decision_rule=MULTICLASS_DECISION_RULE,
        )
        payload["model_kwargs"]["num_classes"] = 3
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        torch.save(payload, stream)
    temporary.replace(checkpoint)
    predictions = run_dir / f"fold_{fold:02d}_predictions.private.jsonl"
    temporary = predictions.with_suffix(".tmp")
    with os.fdopen(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(predictions)
    return {"checkpoint": _artifact_ref(checkpoint), "predictions": _artifact_ref(predictions)}


def run_sagittal_binary_cv(cfg: SagittalBinaryCVConfig) -> Dict[str, Any]:
    _ensure_mlservice_on_path()
    _validate_cv_config(cfg)

    from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier
    from training.datasets.tmj_position_dataset import make_binary_position_loaders
    from training.losses.focal_loss import BinaryFocalLoss
    from training.tmj_position_label_table import (
        build_index,
    )
    from training.utils.seed import make_worker_init_fn, set_seed

    set_seed(cfg.seed)

    if cfg.device:
        device = torch.device(cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if cfg.input_path:
        from training.tmj_position_label_table import build_canonical_index

        all_records = build_canonical_index(
            cfg.input_path, cfg.dataset_root, sagittal_only=True, require_crops=True
        )
    elif cfg.legacy_name_join:
        _validate_legacy_cv_paths(cfg)
        all_records = build_index(
            manifest_path=cfg.manifest_path,
            labels_path=cfg.labels_path,
            dataset_root=cfg.dataset_root,
        )
    else:
        raise ValueError(
            "input_path is required; name-based inputs require explicit legacy_name_join"
        )
    roi_results = []
    roi_contract = None
    if cfg.input_path:
        from training.roi_provenance import require_shared_roi_contract, validate_roi_pair

        roi_results = [validate_roi_pair(record) for record in all_records]
        roi_contract = require_shared_roi_contract(
            [result["roi_contract"] for result in roi_results]
        )
        from models.blocks import validate_research_architecture

        validate_research_architecture(
            cfg.features,
            fc_hidden=cfg.fc_hidden,
            crop_size=roi_contract["preprocessing"]["crop_size"],
        )
    binary_records = _side_records(all_records, cfg)
    folds = _validated_cv_folds(binary_records, cfg)

    run_dir = _create_run_directory(cfg)
    fold_rows: List[Dict[str, Any]] = []
    run_metadata = {
        "schema_version": 1,
        "run_key": run_dir.name if not cfg.run_id and not cfg.output_json else uuid.uuid4().hex,
        "source_namespace": "source-qualified" if cfg.input_path else "legacy-unverified",
        "provenance": {
            "detector_training_identity": "unknown",
            "detector_independence": "unknown",
            "roi_validation": "verified" if cfg.input_path else "unverified_legacy",
            "roi_contract": roi_contract,
            "source_rechecked_studies": sum(row["source_rechecked"] for row in roi_results),
            "coordinate_spaces": {
                space: sum(row["coordinate_space"] == space for row in roi_results)
                for space in sorted({row["coordinate_space"] for row in roi_results})
            },
        },
        "counts": {
            "studies": len(all_records),
            "samples": len(binary_records),
            "patient_groups": len({_record_keys(record)["group_key"] for record in binary_records}),
        },
    }
    report_path = str(run_dir / "report.json")

    def snapshot(status=None):
        report = _build_cv_report_dict(cfg, fold_rows)
        report.update(run_metadata)
        if status is not None:
            report["status"] = status
            report["failure_code"] = (
                "cv_run_interrupted" if status == "interrupted" else "cv_run_failed"
            )
        report["partial"] = len(fold_rows) != cfg.n_splits
        _write_cv_report_json_atomic(report_path, report)
        return report

    snapshot()
    try:
        private_records = [_private_record(record) for record in binary_records]
        input_paths = [cfg.input_path] if cfg.input_path else [cfg.manifest_path, cfg.labels_path]
        input_fingerprints = [
            {"path": path, "sha256": _file_sha256(Path(path))} for path in input_paths
        ]
        run_metadata["fingerprints"] = {
            "input_sha256": input_fingerprints[0]["sha256"]
            if len(input_fingerprints) == 1
            else _digest_json([entry["sha256"] for entry in input_fingerprints]),
            "dataset_sha256": _digest_json(
                [
                    {
                        k: record[k]
                        for k in (
                            "sample_key",
                            "group_key",
                            "sag",
                            "crop_sha256",
                            "passport_sha256",
                        )
                    }
                    for record in private_records
                ]
            ),
            "code": _code_fingerprint(),
            "runtime": _runtime_fingerprint(device),
        }
        replay = {
            "schema_version": 1,
            "config": asdict(cfg),
            "roi_contract": roi_contract,
            "inputs": input_fingerprints,
            "records": private_records,
        }
        original_crop_hashes = {
            record["sample_key"]: record["crop_sha256"] for record in private_records
        }
        _write_cv_report_json_atomic(str(run_dir / "replay.private.json"), replay)
        run_metadata["artifacts"] = {"replay": _artifact_ref(run_dir / "replay.private.json")}
        snapshot()
        epoch_jsonl_path = run_dir / "epochs.jsonl" if cfg.log_epochs_jsonl else None
        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            fold_seed = (cfg.seed + fold_idx) % (2**32)
            set_seed(fold_seed)
            worker_init = make_worker_init_fn(fold_seed) if cfg.num_workers > 0 else None
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
                num_classes=_num_classes(cfg),
            )

            train_loader.pin_memory = val_loader.pin_memory = device.type == "cuda"
            calibration_loader = _make_calibration_loader(train_recs, cfg)
            criterion = (
                nn.CrossEntropyLoss()
                if cfg.mode == "multiclass"
                else BinaryFocalLoss(
                    gamma=cfg.gamma, alpha=_focal_alpha_sagittal(calibration_loader)
                )
            )

            model = TMJBinaryPositionClassifier(
                features=list(cfg.features),
                fc_hidden=cfg.fc_hidden,
                dropout=cfg.dropout,
                num_classes=_num_classes(cfg),
            ).to(device)

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
                calibration_loader=calibration_loader,
            )
            predictions = fold_out.pop("_predictions")
            private_rows = _aligned_prediction_rows(
                predictions,
                train_recs,
                val_recs,
                fold_idx,
                fold_out.get("threshold_from_train_youden"),
                mode=cfg.mode,
            )
            if any(
                row["crop_sha256"] != original_crop_hashes[row["sample_key"]]
                for row in private_rows
            ):
                raise ValueError("input_changed_during_run")
            fold_out["artifacts"] = _save_fold_artifacts(
                run_dir, model, cfg, fold_idx, fold_out, private_rows
            )
            fold_out["fold"] = fold_idx
            fold_out["n_train_samples"] = len(train_recs)
            fold_out["n_val_samples"] = len(val_recs)
            fold_rows.append(fold_out)

            snapshot()
        return snapshot()
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        snapshot(status)
        raise RuntimeError(
            "cv_run_interrupted" if status == "interrupted" else "cv_run_failed"
        ) from None


def _print_cv_table(result: Dict[str, Any]) -> None:
    if result.get("mode") == "multiclass":
        print(
            "Sagittal multiclass development CV",
            json.dumps(_json_sanitize(result["summary"]), allow_nan=False),
        )
        return
    s = result["summary"]
    print("\n=== Sagittal binary development CV (StratifiedGroupKFold) ===")
    if result.get("status") == "in_progress":
        print(
            f"(partial snapshot: {result.get('completed_folds', 0)}/"
            f"{result.get('n_splits', '?')} folds)\n"
        )
    print(
        f"mean val AUC: {s.get('mean_val_auc', float('nan')):.4f} ± {s.get('std_val_auc', 0):.4f}"
    )
    print(
        f"mean balanced acc: {s.get('mean_val_balanced_accuracy', float('nan')):.4f} "
        f"± {s.get('std_val_balanced_accuracy', 0):.4f}"
    )
    print(
        f"mean F1 (positive=1): {s.get('mean_val_f1_positive', float('nan')):.4f} "
        f"± {s.get('std_val_f1_positive', 0):.4f}"
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
        try:
            raw = Path(json_path).expanduser().read_text(encoding="utf-8")
            result = json.loads(raw)
        except (OSError, ValueError, TypeError, RecursionError):
            raise ValueError("invalid_cv_report") from None
    result = _publishable_analyzer_report(result)

    lines: List[str] = []
    written: List[str] = []

    def ln(*parts: Any) -> None:
        if parts:
            lines.append(" ".join(str(p) for p in parts))
        else:
            lines.append("")

    out: Dict[str, Any] = {
        "raw_keys": list(result.keys()),
        "mode": result["mode"],
        "num_classes": result["num_classes"],
        "class_semantics": result["class_semantics"],
        "model_selection_metric": result["model_selection_metric"],
        "assessment": result.get("assessment", "development_cv"),
        "model_selection_source": result.get("model_selection_source", "validation"),
        "calibration_source": result.get("calibration_source", "unknown_legacy"),
        "config": result.get("config"),
        "summary": result.get("summary"),
        "status": result.get("status"),
        "completed_folds": result.get("completed_folds"),
        "n_splits": result.get("n_splits"),
        "fold_summaries": [],
        "epoch_rows": [],
        "report_files_written": written,
    }

    selection = result["model_selection_metric"]
    out.update(
        {
            f"{prefix}_fold_{selection}": result.get(f"{prefix}_fold_{selection}")
            for prefix in ("best", "worst")
        }
    )
    multiclass = result["mode"] == "multiclass"
    cfg = result.get("config") or {}
    ln()
    ln("=== Sagittal CV — config (main fields) ===")
    ln(
        "mode:",
        result["mode"],
        "class_semantics:",
        result["class_semantics"],
        "selection_metric:",
        selection,
    )
    for k in (
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
    ln("=== Development CV summary ===")
    ln(
        "model_selection_source:",
        out["model_selection_source"],
        "calibration_source:",
        out["calibration_source"],
    )
    for key in sorted(s.keys()):
        v = s[key]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = None
        ln(f"  {key}: {v}")

    ln()
    ln(
        f"  best_fold_{selection}:",
        result.get(f"best_fold_{selection}"),
        f"  worst_fold_{selection}:",
        result.get(f"worst_fold_{selection}"),
    )

    folds = result.get("folds") or []
    epoch_rows: List[Dict[str, Any]] = []

    ln()
    ln(
        "=== Per-fold development multiclass argmax metrics ==="
        if multiclass
        else "=== Per-fold (development val metrics @ training-Youden threshold) ==="
    )
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
            "val_f1_positive": f.get("val_f1_positive", f.get("val_f1_minority")),
            "val_sensitivity": f.get("val_sensitivity"),
            "val_specificity": f.get("val_specificity"),
            "val_support": f.get("val_support"),
            "train_majority_baseline": f.get("train_majority_baseline"),
            "val_accuracy_at_threshold": f.get("val_accuracy_at_threshold"),
            "n_epochs_logged": len(f.get("epoch_history") or []),
        }
        if multiclass:
            summ = {
                key: f.get(key)
                for key in (
                    "fold",
                    "n_train_samples",
                    "n_val_samples",
                    "best_epoch",
                    "val_accuracy",
                    "val_balanced_accuracy",
                    "val_macro_f1",
                    "val_per_class",
                    "val_confusion_matrix",
                    "val_support",
                    "train_support",
                    "train_majority_baseline",
                )
            }
            summ["n_epochs_logged"] = len(f.get("epoch_history") or [])
        out["fold_summaries"].append(summ)
        ln()
        ln(f"--- fold {fi} ---")
        for k, v in summ.items():
            if k == "fold":
                continue
            ln(f"  {k}: {v}")
        if multiclass:
            ln(
                "  val_confusion_matrix [true rows/predicted columns;0/1/2]:",
                f.get("val_confusion_matrix"),
            )
        else:
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
                "mode": out["mode"],
                "num_classes": out["num_classes"],
                "class_semantics": out["class_semantics"],
                "model_selection_metric": out["model_selection_metric"],
                "assessment": out["assessment"],
                "model_selection_source": out["model_selection_source"],
                "calibration_source": out["calibration_source"],
                "config": result.get("config"),
                "summary": result.get("summary"),
                "status": result.get("status"),
                "completed_folds": result.get("completed_folds"),
                "n_splits": result.get("n_splits"),
                **{
                    f"{prefix}_fold_{selection}": result.get(f"{prefix}_fold_{selection}")
                    for prefix in ("best", "worst")
                },
                "fold_summaries": out["fold_summaries"],
                "epoch_rows": out["epoch_rows"],
            }
            jp = rp.parent / f"{stem}_export.json"
            with open(jp, "w", encoding="utf-8") as jf:
                json.dump(_json_sanitize(export_payload), jf, indent=2, allow_nan=False)
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
                val_aucs = [r["val_macro_f1" if multiclass else "val_auc"] for r in h]
                losses = [r["train_loss"] for r in h]
                axes[0].plot(xs, val_aucs, marker="o", ms=2, lw=1, label=f"fold {fi}")
                axes[1].plot(xs, losses, marker="o", ms=2, lw=1, label=f"fold {fi}")

            axes[0].set_ylabel("val macro F1" if multiclass else "val ROC-AUC")
            axes[0].set_title(
                "Validation macro F1 per epoch" if multiclass else "Validation ROC-AUC per epoch"
            )
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
        print(f"\nWrote {len(written)} report files")

    return out


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"ready": False, "code": "invalid_arguments"}))
        self.exit(2)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _SafeArgumentParser(description="Sagittal ROI position — StratifiedGroupKFold CV")
    parser.add_argument("--mode", default="binary", choices=("binary", "multiclass"))
    parser.add_argument("--input-path", default=None, help="Canonical schema-v1 research index")
    parser.add_argument(
        "--legacy-name-join", action="store_true", help="Explicit legacy name-based inputs"
    )
    parser.add_argument("--crop-dir", default="data/detector_crops")
    parser.add_argument(
        "--manifest-private", default="data/dataset_cbct_public/manifest_private.json"
    )
    parser.add_argument("--labels-json", default="data/tmj_position_labels.json")
    parser.add_argument("--dataset-root", default="data/dataset_cbct_public")
    parser.add_argument(
        "--output-dir", default="experiments", help="Root for protected per-run artifacts"
    )
    parser.add_argument(
        "--run-id", default=None, help="Optional local directory token; reuse refused"
    )
    parser.add_argument(
        "--output-json", default="", help="Legacy alias: parent/stem selects run directory"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--train-augment-mode",
        default=None,
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

    try:
        feat: Optional[Tuple[int, ...]] = None
        if args.features.strip():
            feat = tuple(int(x.strip()) for x in args.features.split(",") if x.strip())

        cfg = SagittalBinaryCVConfig(
            mode=args.mode,
            input_path=args.input_path,
            legacy_name_join=args.legacy_name_join,
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
            output_dir=args.output_dir,
            run_id=args.run_id,
            tqdm_disable=False,
            train_augment_mode=args.train_augment_mode,
            features=feat if feat is not None else SagittalBinaryCVConfig.features,
            fc_hidden=args.fc_hidden if args.fc_hidden > 0 else SagittalBinaryCVConfig.fc_hidden,
        )
        _validate_cv_config(cfg)
        if cfg.device:
            torch.device(cfg.device)
    except (ValueError, TypeError, RuntimeError):
        parser.error("invalid_arguments")
    from training.tmj_position_label_table import InputValidationError

    try:
        result = run_sagittal_binary_cv(cfg)
        _print_cv_table(result)
        return 0
    except InputValidationError as error:
        print(json.dumps(error.report, allow_nan=False))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"schema_version": 1, "ready": False, "code": "research_run_interrupted"}))
        return 130
    except Exception as error:
        # CV preserves partial private artifacts and wraps interrupts in a fixed
        # RuntimeError for notebook callers. Preserve shell interrupt semantics.
        interrupted = isinstance(error, RuntimeError) and error.args == ("cv_run_interrupted",)
        code = "research_run_interrupted" if interrupted else "research_run_failed"
        print(json.dumps({"schema_version": 1, "ready": False, "code": code}))
        return 130 if interrupted else 2


if __name__ == "__main__":
    raise SystemExit(main())
