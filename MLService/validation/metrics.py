"""
Metrics for segmentation model evaluation

Implements common metrics for binary and multi-class segmentation:
- Dice Coefficient (F1 Score)
- IoU (Jaccard Index)
- Precision, Recall, Specificity
- Hausdorff Distance
- Average Surface Distance
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

logger = logging.getLogger(__name__)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice Coefficient (F1 Score) for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: Predicted binary mask (0 or 1)
        target: Ground truth binary mask (0 or 1)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient in range [0, 1] where 1 is perfect overlap
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred & target)
    union = np.sum(pred) + np.sum(target)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor

    Returns:
        IoU score in range [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    intersection = np.sum(pred & target)
    union = np.sum(pred | target)

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def precision_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate precision (positive predictive value).

    Precision = TP / (TP + FP)

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor

    Returns:
        Precision in range [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    true_positive = np.sum(pred & target)
    false_positive = np.sum(pred & ~target)

    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    return float(precision)


def recall_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate recall (sensitivity, true positive rate).

    Recall = TP / (TP + FN)

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor

    Returns:
        Recall in range [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    true_positive = np.sum(pred & target)
    false_negative = np.sum(~pred & target)

    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    return float(recall)


def specificity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate specificity (true negative rate).

    Specificity = TN / (TN + FP)

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor

    Returns:
        Specificity in range [0, 1]
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    true_negative = np.sum(~pred & ~target)
    false_positive = np.sum(pred & ~target)

    specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
    return float(specificity)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: int = 95) -> float:
    """
    Calculate Hausdorff Distance between two binary masks.

    Measures the maximum distance from a point in one set to the closest point in the other.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        percentile: Use percentile HD for robustness (default 95)

    Returns:
        Hausdorff distance in pixels
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Get edge points
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        logger.warning("Empty mask in Hausdorff distance calculation")
        return float("inf")

    # Calculate directed Hausdorff distances
    hd_pred_to_target = directed_hausdorff(pred_points, target_points)[0]
    hd_target_to_pred = directed_hausdorff(target_points, pred_points)[0]

    # Return maximum (or percentile)
    if percentile < 100:
        # For 95th percentile HD
        distances_pred = np.min(
            np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=2), axis=1
        )
        distances_target = np.min(
            np.linalg.norm(target_points[:, None] - pred_points[None, :], axis=2), axis=1
        )

        hd_pred_to_target = np.percentile(distances_pred, percentile)
        hd_target_to_pred = np.percentile(distances_target, percentile)

    hd = max(hd_pred_to_target, hd_target_to_pred)
    return float(hd)


def average_surface_distance(
    pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...] = None
) -> float:
    """
    Calculate Average Symmetric Surface Distance (ASSD).

    Measures average distance between surface points of two masks.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Physical spacing between pixels (e.g., from DICOM)

    Returns:
        Average surface distance
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if spacing is None:
        spacing = tuple([1.0] * pred.ndim)

    # Compute distance transforms
    pred_dt = distance_transform_edt(~pred, sampling=spacing)
    target_dt = distance_transform_edt(~target, sampling=spacing)

    # Get surface points
    pred_surface = pred & (pred_dt <= 1)
    target_surface = target & (target_dt <= 1)

    if np.sum(pred_surface) == 0 or np.sum(target_surface) == 0:
        logger.warning("Empty surface in ASD calculation")
        return float("inf")

    # Average distances
    pred_to_target_dist = np.mean(target_dt[pred_surface])
    target_to_pred_dist = np.mean(pred_dt[target_surface])

    asd = (pred_to_target_dist + target_to_pred_dist) / 2.0
    return float(asd)


def confusion_matrix(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Calculate confusion matrix components.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()

    tp = int(np.sum(pred & target))
    tn = int(np.sum(~pred & ~target))
    fp = int(np.sum(pred & ~target))
    fn = int(np.sum(~pred & target))

    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}


def compute_all_metrics(
    pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, ...] = None
) -> dict:
    """
    Compute all segmentation metrics at once.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        spacing: Physical spacing for distance metrics

    Returns:
        Dictionary with all metrics
    """
    # Binarize if needed
    if pred.dtype != bool:
        pred = (pred > 0.5).astype(np.uint8)
    if target.dtype != bool:
        target = (target > 0.5).astype(np.uint8)

    metrics = {}

    # Overlap metrics
    metrics["dice"] = dice_coefficient(pred, target)
    metrics["iou"] = iou_score(pred, target)

    # Classification metrics
    metrics["precision"] = precision_score(pred, target)
    metrics["recall"] = recall_score(pred, target)
    metrics["specificity"] = specificity_score(pred, target)

    # F1 score (same as Dice for binary)
    metrics["f1_score"] = metrics["dice"]

    # Distance metrics (can be slow for large volumes)
    try:
        metrics["hausdorff_95"] = hausdorff_distance(pred, target, percentile=95)
        metrics["hausdorff_100"] = hausdorff_distance(pred, target, percentile=100)
    except Exception as e:
        logger.warning(f"Error computing Hausdorff distance: {e}")
        metrics["hausdorff_95"] = None
        metrics["hausdorff_100"] = None

    try:
        metrics["avg_surface_distance"] = average_surface_distance(pred, target, spacing)
    except Exception as e:
        logger.warning(f"Error computing ASD: {e}")
        metrics["avg_surface_distance"] = None

    # Confusion matrix
    cm = confusion_matrix(pred, target)
    metrics.update(cm)

    return metrics


def print_metrics(metrics: dict, title: str = "Segmentation Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    # Overlap metrics
    print("\n📊 Overlap Metrics:")
    print(f"  Dice Coefficient:  {metrics.get('dice', 0):.4f}")
    print(f"  IoU (Jaccard):     {metrics.get('iou', 0):.4f}")

    # Classification metrics
    print("\n📈 Classification Metrics:")
    print(f"  Precision:         {metrics.get('precision', 0):.4f}")
    print(f"  Recall:            {metrics.get('recall', 0):.4f}")
    print(f"  Specificity:       {metrics.get('specificity', 0):.4f}")

    # Distance metrics
    if metrics.get("hausdorff_95") is not None:
        print("\n📏 Distance Metrics:")
        print(f"  Hausdorff (95%):   {metrics.get('hausdorff_95', 0):.2f} pixels")
        print(f"  Hausdorff (100%):  {metrics.get('hausdorff_100', 0):.2f} pixels")
        if metrics.get("avg_surface_distance") is not None:
            print(f"  Avg Surface Dist:  {metrics.get('avg_surface_distance', 0):.2f}")

    # Confusion matrix
    if "true_positive" in metrics:
        print("\n🔢 Confusion Matrix:")
        print(f"  True Positive:     {metrics['true_positive']}")
        print(f"  True Negative:     {metrics['true_negative']}")
        print(f"  False Positive:    {metrics['false_positive']}")
        print(f"  False Negative:    {metrics['false_negative']}")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create dummy masks for testing
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1  # Square in center

    # Slightly offset prediction
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[32:72, 32:72] = 1

    print("Testing metrics on dummy data...")
    metrics = compute_all_metrics(pred, target)
    print_metrics(metrics, "Test Results")
