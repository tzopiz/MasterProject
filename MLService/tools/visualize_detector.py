#!/usr/bin/env python3
"""
Visualize TMJ Detector Results.

Generates an image showing the full CBCT slices with bounding boxes
indicating where the model detected the TMJ and extracted the crop.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# from training.datasets.tmj_detector_dataset import load_dicom_volume, preprocess_volume


def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    """Load DICOM series from directory."""
    dicom_files = list(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    return volume


def load_metadata(crop_dir: Path, study_id: str) -> dict:
    """Load metadata with predicted coordinates."""
    json_path = crop_dir / study_id / f"{study_id}_metadata.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def load_manual_annotation(dataset_dir: Path, study_id: str) -> dict:
    """Load manual annotation if available."""
    # Try finding in roi_annotations dir relative to dataset parent
    roi_dir = dataset_dir.parent / "roi_annotations"
    json_path = roi_dir / f"{study_id}_rois.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def plot_prediction(
    volume, coords_left, coords_right, crop_size, output_path, manual_left=None, manual_right=None
):
    """Plot 3 orthogonal views with bounding boxes for both TMJs."""

    # Get centers
    zl, yl, xl = coords_left
    zr, yr, xr = coords_right

    # Half size for bbox
    h = crop_size // 2

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title = (
        f"TMJ Detection (Red=Model, Green=Manual)\nCrop Size: {crop_size}x{crop_size}x{crop_size}"
    )
    fig.suptitle(title, fontsize=16)

    # Helper to draw box
    def draw_box(ax, center_r, center_c, color, label=None):
        r, c = center_r, center_c
        rect = patches.Rectangle(
            (c - h, r - h),
            crop_size,
            crop_size,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=label,
        )
        ax.add_patch(rect)
        ax.plot(c, r, color + "+")

    # --- LEFT TMJ (Top Row) ---

    # Axial (XY plane) - take slice at Z center
    ax = axes[0, 0]
    ax.imshow(volume[int(zl), :, :], cmap="gray")
    ax.set_title(f"Left TMJ - Axial (Z={int(zl)})")
    draw_box(ax, yl, xl, "r", "Model")
    if manual_left is not None:
        draw_box(ax, manual_left[1], manual_left[2], "g", "Manual")
        # Show offset
        dist = np.sqrt(np.sum((np.array([zl, yl, xl]) - np.array(manual_left)) ** 2))
        ax.set_xlabel(f"Err: {dist:.1f} px")

    # Coronal (XZ plane) - take slice at Y center
    ax = axes[0, 1]
    ax.imshow(volume[:, int(yl), :], cmap="gray", aspect="auto")
    ax.set_title(f"Left TMJ - Coronal (Y={int(yl)})")
    draw_box(ax, zl, xl, "r")
    if manual_left is not None:
        draw_box(ax, manual_left[0], manual_left[2], "g")

    # Sagittal (YZ plane) - take slice at X center
    ax = axes[0, 2]
    ax.imshow(volume[:, :, int(xl)], cmap="gray", aspect="auto")
    ax.set_title(f"Left TMJ - Sagittal (X={int(xl)})")
    draw_box(ax, zl, yl, "r")
    if manual_left is not None:
        draw_box(ax, manual_left[0], manual_left[1], "g")

    # --- RIGHT TMJ (Bottom Row) ---

    # Axial
    ax = axes[1, 0]
    ax.imshow(volume[int(zr), :, :], cmap="gray")
    ax.set_title(f"Right TMJ - Axial (Z={int(zr)})")
    draw_box(ax, yr, xr, "r")
    if manual_right is not None:
        draw_box(ax, manual_right[1], manual_right[2], "g")
        dist = np.sqrt(np.sum((np.array([zr, yr, xr]) - np.array(manual_right)) ** 2))
        ax.set_xlabel(f"Err: {dist:.1f} px")

    # Coronal
    ax = axes[1, 1]
    ax.imshow(volume[:, int(yr), :], cmap="gray", aspect="auto")
    ax.set_title(f"Right TMJ - Coronal (Y={int(yr)})")
    draw_box(ax, zr, xr, "r")
    if manual_right is not None:
        draw_box(ax, manual_right[0], manual_right[2], "g")

    # Sagittal
    ax = axes[1, 2]
    ax.imshow(volume[:, :, int(xr)], cmap="gray", aspect="auto")
    ax.set_title(f"Right TMJ - Sagittal (X={int(xr)})")
    draw_box(ax, zr, yr, "r")
    if manual_right is not None:
        draw_box(ax, manual_right[0], manual_right[1], "g")

    # Add legend to first plot
    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize detector results")
    parser.add_argument("--study", type=str, default="study_0001", help="Study ID to visualize")
    parser.add_argument("--dataset", type=str, default="data/dataset", help="Path to dataset")
    parser.add_argument(
        "--crops", type=str, default="data/auto_crops", help="Path to crop metadata"
    )

    args = parser.parse_args()

    study_id = args.study
    dataset_dir = Path(args.dataset)
    crop_dir = Path(args.crops)

    # 1. Load Metadata (Predicted Coords)
    try:
        meta = load_metadata(crop_dir, study_id)
    except Exception as e:
        print(f"Error: {e}")
        return

    coords_left = meta["predicted_coords"]["left"]
    coords_right = meta["predicted_coords"]["right"]
    crop_size = meta["crop_size"]

    print(f"Visualizing {study_id}")
    print(f"Left predicted (Red): {coords_left}")
    print(f"Right predicted (Red): {coords_right}")

    # 2. Load Manual Annotation
    manual_ann = load_manual_annotation(dataset_dir, study_id)
    manual_left = None
    manual_right = None
    if manual_ann:
        manual_left = manual_ann["left_tmj"]["center"]
        manual_right = manual_ann["right_tmj"]["center"]
        print(f"Left manual (Green):  {manual_left}")
        print(f"Right manual (Green): {manual_right}")
    else:
        print("No manual annotation found.")

    # 3. Load Volume
    study_path = dataset_dir / study_id
    print(f"Loading volume from {study_path}...")
    volume = load_dicom_volume(study_path)

    # 4. Plot
    output_file = crop_dir / f"{study_id}_visualization_comparison.png"
    plot_prediction(
        volume, coords_left, coords_right, crop_size, output_file, manual_left, manual_right
    )


if __name__ == "__main__":
    main()
