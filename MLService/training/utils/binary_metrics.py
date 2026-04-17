"""
Binary classification metrics (numpy / sklearn) — ROC-AUC, Youden threshold, etc.

Used by position-classifier training and CV; keep segmentation metrics in
`validation/metrics.py` separate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC; returns nan if fewer than two classes present."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def youden_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Threshold maximizing Youden's J (TPR - FPR) on the ROC curve.

    If only one class is present, returns 0.5.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    # sklearn: thresholds[i] corresponds to score >= thresholds[i] for point i
    return float(thresholds[idx])


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Accuracy, balanced accuracy, F1 for positive class (1), confusion @ threshold.

    y_score are probabilities (or any monotone scores; threshold applied as >=).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_minority": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def calibration_report_binary_head(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """
    One binary head: AUC, Youden threshold, accuracy at that threshold.

    Matches the spirit of ``calibrate_thresholds`` in ``train_binary_position_classifier.py``.
    """
    auc = binary_roc_auc(y_true, y_score)
    thresh = youden_optimal_threshold(y_true, y_score)
    if np.isnan(auc):
        m = binary_metrics_at_threshold(y_true, y_score, 0.5)
        return {
            "auc_roc": float("nan"),
            "optimal_threshold": 0.5,
            "accuracy_at_threshold": m["accuracy"],
            "balanced_accuracy_at_threshold": m["balanced_accuracy"],
            "f1_minority_at_threshold": m["f1_minority"],
            "confusion_matrix_at_threshold": m["confusion_matrix"],
        }
    m = binary_metrics_at_threshold(y_true, y_score, thresh)
    return {
        "auc_roc": auc,
        "optimal_threshold": thresh,
        "accuracy_at_threshold": m["accuracy"],
        "balanced_accuracy_at_threshold": m["balanced_accuracy"],
        "f1_minority_at_threshold": m["f1_minority"],
        "confusion_matrix_at_threshold": m["confusion_matrix"],
    }


def aggregate_fold_metrics(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Mean and std over folds for numeric keys (skips nan when computing mean)."""
    out: Dict[str, Any] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and not np.isnan(r[k])], dtype=np.float64)
        if vals.size == 0:
            out[f"mean_{k}"] = float("nan")
            out[f"std_{k}"] = float("nan")
        else:
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals)) if vals.size > 1 else 0.0
    return out
