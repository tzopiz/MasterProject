"""
Classification metrics (numpy/sklearn): binary ROC/Youden and fixed3-class reports.

Used by position-classifier training and CV; keep segmentation metrics in
`validation/metrics.py` separate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _binary_inputs(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("Binary labels and scores must be nonempty and equally sized")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Binary labels must be 0/1 and scores must be finite")
    return labels.astype(int), scores


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC; undefined when either class is absent."""
    y_true, y_score = _binary_inputs(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def youden_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Maximize Youden J among finite ROC thresholds; ties use the highest.

    Single-class input has no ROC and falls back to 0.5. The nonfinite
    all-negative ROC sentinel is excluded so the decision survives JSON export.
    """
    y_true, y_score = _binary_inputs(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Explicit positive=1 metrics, applying score >= threshold.

    Undefined class recalls and balanced accuracy are NaN. ``f1_minority``
    remains a legacy alias of ``f1_positive``; it does not identify a minority.
    """
    y_true, y_score = _binary_inputs(y_true, y_score)
    if not np.isfinite(threshold):
        raise ValueError("Binary decision threshold must be finite")
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return {
        "positive_class": 1,
        "support": {"0": tn + fp, "1": tp + fn},
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_positive": f1,
        "f1_minority": f1,
        "confusion_matrix": cm.tolist(),
    }


def calibration_report_binary_head(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """One head's ROC and calibrated decision metrics (positive=1)."""
    auc = binary_roc_auc(y_true, y_score)
    threshold = youden_optimal_threshold(y_true, y_score)
    metrics = binary_metrics_at_threshold(y_true, y_score, threshold)
    return {
        "auc_roc": auc,
        "optimal_threshold": threshold,
        "positive_class": metrics["positive_class"],
        "support": metrics["support"],
        "accuracy_at_threshold": metrics["accuracy"],
        "balanced_accuracy_at_threshold": metrics["balanced_accuracy"],
        "sensitivity_at_threshold": metrics["sensitivity"],
        "specificity_at_threshold": metrics["specificity"],
        "f1_positive_at_threshold": metrics["f1_positive"],
        "f1_minority_at_threshold": metrics["f1_positive"],
        "confusion_matrix_at_threshold": metrics["confusion_matrix"],
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


def multiclass_metrics(y_true, probabilities, num_classes=3):
    """Fixed class order, argmax with lowest-index ties; zero_division=0.

    Macro F1 includes all declared classes; balanced accuracy averages recalls
    of supported classes. CV independently requires every class in each fold.
    """
    from sklearn.metrics import precision_recall_fscore_support

    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=np.float64)
    if type(num_classes) is not int or num_classes != 3 or labels.ndim != 1 or not len(labels):
        raise ValueError("invalid_multiclass_inputs")
    if labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(num_classes)).all():
        raise ValueError("invalid_multiclass_inputs")
    if (
        scores.shape != (len(labels), num_classes)
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("invalid_multiclass_inputs")
    predictions = np.argmax(scores, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(num_classes), zero_division=0
    )
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_f1": float(np.mean(f1)),
        "support": {str(i): int(support[i]) for i in range(num_classes)},
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(num_classes)
        ).tolist(),
        "per_class": {
            str(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        },
    }
