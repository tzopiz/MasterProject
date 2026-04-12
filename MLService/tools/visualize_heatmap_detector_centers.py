#!/usr/bin/env python3
"""
Run dual single-joint heatmap detectors (left + right) on one DICOM folder,
print predicted centers (z, y, x) in original voxel space, save a PNG with
orthogonal views through each predicted center (same pipeline as auto_crop_from_detector).

Usage:
    ./venv/bin/python tools/visualize_heatmap_detector_centers.py \\
        --dicom-dir /path/to/study_with_dcm \\
        --left-model  models/checkpoints/left_detector.pth \\
        --right-model models/checkpoints/right_detector.pth \\
        --output      centers_preview.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_MLSERVICE = Path(__file__).resolve().parent.parent
if str(_MLSERVICE) not in sys.path:
    sys.path.insert(0, str(_MLSERVICE))


def _load_auto_crop_module():
    path = _MLSERVICE / "tools" / "auto_crop_from_detector.py"
    spec = importlib.util.spec_from_file_location("auto_crop_from_detector", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _normalize_slice_for_display(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.float32)
    lo, hi = np.percentile(s, [1, 99])
    if hi <= lo:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0, 1)


def _plot_crosshair(ax, py: float, px: float, color: str, label: str) -> None:
    ax.scatter([px], [py], s=120, facecolors="none", edgecolors=color, linewidths=2, label=label)
    ax.axvline(px, color=color, alpha=0.35, linewidth=1)
    ax.axhline(py, color=color, alpha=0.35, linewidth=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Dual heatmap TMJ centers + PNG preview")
    p.add_argument("--dicom-dir", required=True, type=Path, help="Folder with *.dcm slices")
    p.add_argument("--left-model", required=True, type=Path)
    p.add_argument("--right-model", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("heatmap_detector_centers.png"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ac = _load_auto_crop_module()

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

    left_zyx = ac.argmax_to_orig(left_hm, orig_shape)
    right_zyx = ac.argmax_to_orig(right_hm, orig_shape)

    lz, ly, lx = (int(left_zyx[0]), int(left_zyx[1]), int(left_zyx[2]))
    rz, ry, rx = (int(right_zyx[0]), int(right_zyx[1]), int(right_zyx[2]))

    D, H, W = raw_vol.shape
    print(f"volume_shape (Z,Y,X): {(D, H, W)}")
    print(f"left_center  (z,y,x): [{lz}, {ly}, {lx}]")
    print(f"right_center (z,y,x): [{rz}, {ry}, {rx}]")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install matplotlib: pip install matplotlib") from e

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def row_for_joint(row: int, z: int, y: int, x: int, name: str, color: str) -> None:
        # Axial: plane z, show (x horizontal, y vertical) like imshow on vol[z]
        ax0 = axes[row, 0]
        sl = _normalize_slice_for_display(raw_vol[z])
        ax0.imshow(sl, cmap="gray", origin="lower")
        _plot_crosshair(ax0, y, x, color, name)
        ax0.set_title(f"{name} — axial (z={z})")
        ax0.set_xlabel("x"); ax0.set_ylabel("y")
        ax0.legend(loc="upper right", fontsize=8)

        # Sagittal: fix x, image is (z, y)
        ax1 = axes[row, 1]
        sl1 = _normalize_slice_for_display(raw_vol[:, :, x])
        ax1.imshow(sl1, cmap="gray", origin="lower", aspect="auto")
        _plot_crosshair(ax1, z, y, color, name)
        ax1.set_title(f"{name} — sagittal (x={x})")
        ax1.set_xlabel("y"); ax1.set_ylabel("z")

        # Coronal: fix y, image is (z, x)
        ax2 = axes[row, 2]
        sl2 = _normalize_slice_for_display(raw_vol[:, y, :])
        ax2.imshow(sl2, cmap="gray", origin="lower", aspect="auto")
        _plot_crosshair(ax2, z, x, color, name)
        ax2.set_title(f"{name} — coronal (y={y})")
        ax2.set_xlabel("x"); ax2.set_ylabel("z")

    row_for_joint(0, lz, ly, lx, "left", "#22c55e")
    row_for_joint(1, rz, ry, rx, "right", "#f97316")

    fig.suptitle(f"Heatmap detector centers — {args.dicom_dir.name}", fontsize=11)
    plt.tight_layout()
    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
