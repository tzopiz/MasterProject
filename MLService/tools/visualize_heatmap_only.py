#!/usr/bin/env python3
"""
Visualize heatmap detector output in three projections (Sagittal, Coronal, Axial).
Shows the raw heatmap probability values with color mapping (dark=low, bright=high).
Visualizes slices centered on the peak (detected joint center).

Usage:
    ./venv/bin/python tools/visualize_heatmap_only.py \
        --dicom-dir /path/to/study_with_dcm \
        --left-model models/checkpoints/left_detector.pth \
        --right-model models/checkpoints/right_detector.pth \
        --output heatmap_viz.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_MLSERVICE = Path(__file__).resolve().parent.parent
if str(_MLSERVICE) not in sys.path:
    sys.path.insert(0, str(_MLSERVICE))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_peak(heatmap_3d: np.ndarray) -> tuple[int, int, int]:
    """Find coordinates of maximum value in 3D heatmap."""
    idx = np.unravel_index(heatmap_3d.argmax(), heatmap_3d.shape)
    return tuple(map(int, idx))


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize heatmap detector output in 3 projections")
    p.add_argument("--dicom-dir", required=True, type=Path, help="Folder with *.dcm slices")
    p.add_argument("--left-model", required=True, type=Path)
    p.add_argument("--right-model", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("heatmap_visualization.png"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ac = _load_module("auto_crop", _MLSERVICE / "tools" / "auto_crop_from_detector.py")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def load_heatmap_ckpt(path: Path) -> torch.nn.Module:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ac.TMJHeatmapDetector()
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        return m

    left_m = load_heatmap_ckpt(args.left_model)
    right_m = load_heatmap_ckpt(args.right_model)

    raw_vol = ac.load_dicom_volume(args.dicom_dir)
    inp, orig_shape = ac.prepare_input(raw_vol)
    inp = inp.to(device)

    with torch.no_grad():
        left_hm = torch.sigmoid(left_m(inp)).squeeze().cpu().numpy()
        right_hm = torch.sigmoid(right_m(inp)).squeeze().cpu().numpy()

    # Find peaks (detected joint centers)
    left_peak = find_peak(left_hm)
    right_peak = find_peak(right_hm)

    peak_z_l, peak_y_l, peak_x_l = left_peak
    peak_z_r, peak_y_r, peak_x_r = right_peak

    print(f"Left peak (z,y,x): {left_peak}")
    print(f"Right peak (z,y,x): {right_peak}")

    # Create figure with 2 rows × 3 columns (2 joints × 3 projections)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # --- Left Joint (Row 0) ---
    # Axial (y-x plane at peak_z_l)
    im = axes[0, 0].imshow(
        left_hm[peak_z_l, :, :], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[0, 0].plot(peak_x_l, peak_y_l, "r+", markersize=15, markeredgewidth=2)
    axes[0, 0].set_title("Left Axial", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], label="Prob")

    # Coronal (z-x plane at peak_y_l)
    im = axes[0, 1].imshow(
        left_hm[:, peak_y_l, :], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[0, 1].plot(peak_x_l, peak_z_l, "r+", markersize=15, markeredgewidth=2)
    axes[0, 1].set_title("Left Coronal", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], label="Prob")

    # Sagittal (z-y plane at peak_x_l)
    im = axes[0, 2].imshow(
        left_hm[:, :, peak_x_l], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[0, 2].plot(peak_y_l, peak_z_l, "r+", markersize=15, markeredgewidth=2)
    axes[0, 2].set_title("Left Sagittal", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], label="Prob")

    # --- Right Joint (Row 1) ---
    # Axial (y-x plane at peak_z_r)
    im = axes[1, 0].imshow(
        right_hm[peak_z_r, :, :], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[1, 0].plot(peak_x_r, peak_y_r, "r+", markersize=15, markeredgewidth=2)
    axes[1, 0].set_title("Right Axial", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], label="Prob")

    # Coronal (z-x plane at peak_y_r)
    im = axes[1, 1].imshow(
        right_hm[:, peak_y_r, :], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[1, 1].plot(peak_x_r, peak_z_r, "r+", markersize=15, markeredgewidth=2)
    axes[1, 1].set_title("Right Coronal", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], label="Prob")

    # Sagittal (z-y plane at peak_x_r)
    im = axes[1, 2].imshow(
        right_hm[:, :, peak_x_r], cmap="gray", vmin=0, vmax=1, origin="upper", aspect="auto"
    )
    axes[1, 2].plot(peak_y_r, peak_z_r, "r+", markersize=15, markeredgewidth=2)
    axes[1, 2].set_title("Right Sagittal", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], label="Prob")

    fig.suptitle(
        f"Heatmap Detector Output (Peak-centered) — {args.dicom_dir.name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Heatmap visualization saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
