#!/usr/bin/env python3
"""
Visualize TMJ detection pipeline: full CBCT → detector → crops.

Shows for a given study:
  - Full CBCT orthogonal slices with detected joint centers and crop boxes
  - Resulting NIfTI crops (what actually enters the dataset)

Usage:
    # One random study (with existing crops)
    ./venv/bin/python tools/visualize_detection.py

    # Specific study
    ./venv/bin/python tools/visualize_detection.py --study study_0001

    # Save to file instead of display
    ./venv/bin/python tools/visualize_detection.py --study study_0001 --save /tmp/detection.png

    # From metadata JSON (no need to rerun detector)
    ./venv/bin/python tools/visualize_detection.py --study study_0001 --crops-dir data/detector_crops
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    files = sorted(dicom_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm in {dicom_dir}")
    slices = [pydicom.dcmread(str(f)) for f in files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(s.InstanceNumber))
    planes = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        planes.append(arr)
    return np.stack(planes, axis=0)   # (D, H, W), HU


def normalize_display(vol: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(vol, [p_lo, p_hi])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    p2, p98 = np.percentile(vol, [2, 98])
    vol = np.clip(vol, p2, p98)
    denom = p98 - p2
    return ((vol - p2) / denom if denom > 0 else np.zeros_like(vol))


def draw_box_on_slice(ax, center_2d, half=64, color="lime", lw=1.5):
    """Draw crop bounding box on a 2D slice."""
    cy, cx = center_2d
    rect = mpatches.Rectangle(
        (cx - half, cy - half), 2 * half, 2 * half,
        linewidth=lw, edgecolor=color, facecolor="none", linestyle="--"
    )
    ax.add_patch(rect)
    ax.plot(cx, cy, "+", color=color, markersize=10, markeredgewidth=1.5)


def show_slices_with_box(
    axes_row,
    vol_norm,
    center_zyx,
    half=64,
    color="lime",
    title_prefix="",
    *,
    imshow_origin: str = "upper",
):
    """Plot axial / coronal / sagittal slices centred on detected joint.

    ``imshow_origin``: default ``\"upper\"`` so the first row of each 2D slice
    is at the **top** of the figure — same convention as ``roi_annotation_tool``
    (OpenCV) and typical axial viewing. Use ``\"lower\"`` only if you need old
    matplotlib-style plots.
    """
    z, y, x = [int(c) for c in center_zyx]
    D, H, W = vol_norm.shape

    slices = [
        (vol_norm[z, :, :],   (y, x),    f"{title_prefix} Axial z={z}"),
        (vol_norm[:, y, :],   (z, x),    f"{title_prefix} Coronal y={y}"),
        (vol_norm[:, :, x],   (z, y),    f"{title_prefix} Sagittal x={x}"),
    ]

    for ax, (sl, c2d, title) in zip(axes_row, slices):
        ax.imshow(sl, cmap="gray", origin=imshow_origin, aspect="auto")
        draw_box_on_slice(ax, c2d, half=half, color=color)
        ax.set_title(title, fontsize=8)
        ax.axis("off")


def show_crop_slices(axes_row, crop, title_prefix=""):
    """Plot axial/coronal/sagittal mid-slices of a 128³ NIfTI crop."""
    D, H, W = crop.shape
    mid = (D // 2, H // 2, W // 2)
    slices = [
        (crop[mid[0], :, :], f"{title_prefix} Axial"),
        (crop[:, mid[1], :], f"{title_prefix} Coronal"),
        (crop[:, :, mid[2]], f"{title_prefix} Sagittal"),
    ]
    for ax, (sl, title) in zip(axes_row, slices):
        ax.imshow(sl, cmap="gray", origin="lower", aspect="auto")
        # crosshair
        cx, cy = sl.shape[1] // 2, sl.shape[0] // 2
        ax.axhline(cy, color="lime", alpha=0.5, linewidth=0.8)
        ax.axvline(cx, color="lime", alpha=0.5, linewidth=0.8)
        ax.set_title(title, fontsize=8)
        ax.axis("off")


# ── main ─────────────────────────────────────────────────────────────────────

def visualize_study(
    study_id: str,
    dataset_root: Path,
    crops_dir: Path,
    save_path: Path | None = None,
    crop_size: int = 128,
):
    study_dir  = dataset_root / study_id
    meta_path  = crops_dir / study_id / f"{study_id}_metadata.json"
    left_path  = crops_dir / study_id / f"{study_id}_left.nii.gz"
    right_path = crops_dir / study_id / f"{study_id}_right.nii.gz"

    # Load metadata (detector predictions)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}\nRun auto_crop_from_detector.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    left_zyx  = np.array(meta["predicted_coords"]["left"])
    right_zyx = np.array(meta["predicted_coords"]["right"])

    print(f"Study: {study_id}")
    print(f"Volume: {meta['volume_shape']}")
    print(f"Left  TMJ: z={left_zyx[0]}  y={left_zyx[1]}  x={left_zyx[2]}")
    print(f"Right TMJ: z={right_zyx[0]}  y={right_zyx[1]}  x={right_zyx[2]}")

    # Load full CBCT
    print("Loading DICOM volume...")
    volume    = load_dicom_volume(study_dir)
    vol_norm  = normalize_display(volume)
    print(f"Shape: {volume.shape}  HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    # Load crops
    left_crop  = load_nifti(left_path)
    right_crop = load_nifti(right_path)

    half = crop_size // 2

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")

    fig.suptitle(
        f"TMJ Detection Pipeline — {study_id}",
        fontsize=13, fontweight="bold", color="white", y=0.995
    )

    # Row 0: full volume with LEFT joint
    show_slices_with_box(axes[0], vol_norm, left_zyx,  half=half, color="#22c55e", title_prefix="Full → LEFT")

    # Row 1: full volume with RIGHT joint
    show_slices_with_box(axes[1], vol_norm, right_zyx, half=half, color="#f97316", title_prefix="Full → RIGHT")

    # Row 2: LEFT crop
    show_crop_slices(axes[2], left_crop,  title_prefix="Crop LEFT")

    # Row 3: RIGHT crop
    show_crop_slices(axes[3], right_crop, title_prefix="Crop RIGHT")

    # Style all titles white
    for ax in axes.flat:
        ax.title.set_color("white")

    # Legend
    leg = [
        mpatches.Patch(color="#22c55e", label="Left TMJ (detector)"),
        mpatches.Patch(color="#f97316", label="Right TMJ (detector)"),
        mpatches.Patch(color="lime",    label="Crosshair (crop centre)"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9,
               facecolor="#1e293b", edgecolor="#334155", labelcolor="white",
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.995])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize TMJ detection pipeline")
    parser.add_argument("--study",       default=None,
                        help="study_id (e.g. study_0001). Random if omitted.")
    parser.add_argument("--dataset-root", default="data/dataset_cbct_public",
                        help="Root with study_* DICOM folders")
    parser.add_argument("--crops-dir",   default="data/detector_crops",
                        help="Root with NIfTI crops + metadata JSON")
    parser.add_argument("--crop-size",   type=int, default=128)
    parser.add_argument("--save",        default=None,
                        help="Save PNG to this path instead of showing")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    crops_dir    = Path(args.crops_dir)

    # Pick study
    if args.study:
        study_id = args.study
    else:
        candidates = [
            d.name for d in crops_dir.iterdir()
            if d.is_dir()
            and (d / f"{d.name}_left.nii.gz").exists()
            and (d / f"{d.name}_metadata.json").exists()
            and (dataset_root / d.name).exists()
        ]
        if not candidates:
            print(f"No studies found in {crops_dir}")
            sys.exit(1)
        study_id = random.choice(candidates)
        print(f"Picked random study: {study_id}")

    visualize_study(
        study_id    = study_id,
        dataset_root = dataset_root,
        crops_dir   = crops_dir,
        save_path   = args.save,
        crop_size   = args.crop_size,
    )


if __name__ == "__main__":
    main()
