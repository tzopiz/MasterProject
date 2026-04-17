"""
Visualization utilities for segmentation results

Functions to visualize predictions, ground truth, and errors.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_segmentation_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    title: str = "Segmentation Result",
    save_path: Optional[Path] = None,
    alpha: float = 0.4
):
    """
    Plot image with segmentation overlay.
    
    Args:
        image: Input grayscale image (2D)
        pred_mask: Predicted mask
        gt_mask: Ground truth mask (optional)
        title: Plot title
        save_path: Path to save figure
        alpha: Transparency of overlay
    """
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    
    # Normalize image to [0, 1]
    if image.max() > 1:
        image = image.astype(np.float32) / image.max()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction overlay
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(pred_mask, cmap='Reds', alpha=alpha * (pred_mask > 0))
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth overlay (if provided)
    if gt_mask is not None:
        axes[2].imshow(image, cmap='gray')
        # Show differences: TP=green, FP=red, FN=blue
        tp = (pred_mask > 0) & (gt_mask > 0)
        fp = (pred_mask > 0) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask > 0)
        
        overlay = np.zeros((*image.shape, 3))
        overlay[tp] = [0, 1, 0]  # Green: correct
        overlay[fp] = [1, 0, 0]  # Red: false positive
        overlay[fn] = [0, 0, 1]  # Blue: false negative
        
        axes[2].imshow(overlay, alpha=alpha)
        axes[2].set_title('Comparison (TP=green, FP=red, FN=blue)')
        axes[2].axis('off')
        
        # Add legend
        green_patch = mpatches.Patch(color='green', label='True Positive')
        red_patch = mpatches.Patch(color='red', label='False Positive')
        blue_patch = mpatches.Patch(color='blue', label='False Negative')
        axes[2].legend(handles=[green_patch, red_patch, blue_patch], 
                      loc='upper right', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


def plot_side_by_side(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    metrics: dict,
    title: str = "Segmentation Comparison",
    save_path: Optional[Path] = None
):
    """
    Side-by-side comparison of prediction and ground truth.
    
    Args:
        image: Input image
        pred_mask: Predicted mask
        gt_mask: Ground truth mask
        metrics: Dictionary of computed metrics
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Normalize image
    if image.max() > 1:
        image = image.astype(np.float32) / image.max()
    
    # Top-left: Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # Top-right: Ground Truth
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].imshow(gt_mask, cmap='Greens', alpha=0.4 * (gt_mask > 0))
    axes[0, 1].set_title('Ground Truth', fontsize=12)
    axes[0, 1].axis('off')
    
    # Bottom-left: Prediction
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].imshow(pred_mask, cmap='Reds', alpha=0.4 * (pred_mask > 0))
    axes[1, 0].set_title('Prediction', fontsize=12)
    axes[1, 0].axis('off')
    
    # Bottom-right: Error map
    tp = (pred_mask > 0) & (gt_mask > 0)
    fp = (pred_mask > 0) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask > 0)
    
    error_map = np.zeros((*image.shape, 3))
    error_map[tp] = [0, 1, 0]
    error_map[fp] = [1, 0, 0]
    error_map[fn] = [0, 0, 1]
    
    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].imshow(error_map, alpha=0.6)
    axes[1, 1].set_title('Error Map', fontsize=12)
    axes[1, 1].axis('off')
    
    # Add legend
    green_patch = mpatches.Patch(color='green', label='TP')
    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color='blue', label='FN')
    axes[1, 1].legend(handles=[green_patch, red_patch, blue_patch], 
                     loc='upper right', fontsize=10)
    
    # Add metrics text
    metrics_text = f"Dice: {metrics.get('dice', 0):.3f} | "
    metrics_text += f"IoU: {metrics.get('iou', 0):.3f} | "
    metrics_text += f"Precision: {metrics.get('precision', 0):.3f} | "
    metrics_text += f"Recall: {metrics.get('recall', 0):.3f}"
    
    plt.suptitle(f"{title}\n{metrics_text}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison to {save_path}")
    
    plt.close()


def plot_slice_sequence(
    volume: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: Optional[np.ndarray] = None,
    slice_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot sequence of slices with segmentation overlays.
    
    Args:
        volume: 3D volume (Z, H, W)
        pred_masks: 3D predicted masks
        gt_masks: 3D ground truth masks (optional)
        slice_indices: Which slices to show (if None, show evenly spaced)
        save_path: Path to save figure
    """
    if slice_indices is None:
        # Show 6 evenly spaced slices
        n_slices = 6
        slice_indices = np.linspace(0, volume.shape[0] - 1, n_slices, dtype=int)
    
    n_cols = len(slice_indices)
    n_rows = 2 if gt_masks is not None else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slice_indices):
        img_slice = volume[slice_idx]
        pred_slice = pred_masks[slice_idx]
        
        # Normalize
        if img_slice.max() > 1:
            img_slice = img_slice.astype(np.float32) / img_slice.max()
        
        # Prediction
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].imshow(pred_slice, cmap='Reds', alpha=0.4 * (pred_slice > 0))
        axes[0, i].set_title(f'Slice {slice_idx}\nPrediction', fontsize=10)
        axes[0, i].axis('off')
        
        # Ground truth (if provided)
        if gt_masks is not None:
            gt_slice = gt_masks[slice_idx]
            axes[1, i].imshow(img_slice, cmap='gray')
            axes[1, i].imshow(gt_slice, cmap='Greens', alpha=0.4 * (gt_slice > 0))
            axes[1, i].set_title('Ground Truth', fontsize=10)
            axes[1, i].axis('off')
    
    plt.suptitle('Segmentation Sequence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sequence to {save_path}")
    
    plt.close()


def plot_metrics_history(
    train_metrics: dict,
    val_metrics: dict,
    save_path: Optional[Path] = None
):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_metrics: Dictionary with lists of training metrics
        val_metrics: Dictionary with lists of validation metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_metrics.get('loss', [])) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, train_metrics.get('loss', []), 'b-', label='Train')
    axes[0, 0].plot(epochs, val_metrics.get('loss', []), 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss History')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs, train_metrics.get('dice', []), 'b-', label='Train')
    axes[0, 1].plot(epochs, val_metrics.get('dice', []), 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].set_title('Dice Score History')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    if 'iou' in train_metrics:
        axes[1, 0].plot(epochs, train_metrics.get('iou', []), 'b-', label='Train')
        axes[1, 0].plot(epochs, val_metrics.get('iou', []), 'r-', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('IoU History')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'precision' in val_metrics and 'recall' in val_metrics:
        axes[1, 1].plot(epochs, val_metrics.get('precision', []), 'g-', label='Precision')
        axes[1, 1].plot(epochs, val_metrics.get('recall', []), 'm-', label='Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved metrics history to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: dict,
    save_path: Optional[Path] = None
):
    """
    Plot confusion matrix visualization.
    
    Args:
        confusion_matrix: Dict with TP, TN, FP, FN
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create matrix
    cm = np.array([
        [confusion_matrix['true_negative'], confusion_matrix['false_positive']],
        [confusion_matrix['false_negative'], confusion_matrix['true_positive']]
    ])
    
    # Plot
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}',
                         ha="center", va="center", color="black", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Testing visualization functions...")
    
    # Create dummy data
    np.random.seed(42)
    image = np.random.rand(256, 256)
    
    gt_mask = np.zeros((256, 256), dtype=np.uint8)
    gt_mask[80:180, 80:180] = 1
    
    pred_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask[85:185, 85:185] = 1
    
    # Test overlay
    plot_segmentation_overlay(image, pred_mask, gt_mask, 
                             title="Test Overlay",
                             save_path="test_overlay.png")
    
    print("Visualization test complete!")

