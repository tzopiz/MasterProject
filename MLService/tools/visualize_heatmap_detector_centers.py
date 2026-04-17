#!/usr/bin/env python3
"""
Run dual single-joint heatmap detectors (left + right) on one DICOM folder,
print predicted centers (z, y, x) in original voxel space, save a PNG.

Uses the same inference as ``auto_crop_from_detector.py``. Orthogonal views
reuse ``visualize_detection.show_slices_with_box`` so layout matches
``tools/visualize_detection.py`` (Axial → Coronal → Sagittal).

Default display matches ``roi_annotation_tool`` (first array row at top).
Use ``--imshow-origin lower`` for the old matplotlib convention.

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


def main() -> None:
    p = argparse.ArgumentParser(description="Dual heatmap TMJ centers + PNG preview")
    p.add_argument("--dicom-dir", required=True, type=Path, help="Folder with *.dcm slices")
    p.add_argument("--left-model", required=True, type=Path)
    p.add_argument("--right-model", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("heatmap_detector_centers.png"))
    p.add_argument("--device", default=None)
    p.add_argument(
        "--imshow-origin",
        choices=("lower", "upper"),
        default="upper",
        help="Matplotlib imshow origin; default upper = same as roi_annotation_tool (OpenCV).",
    )
    p.add_argument(
        "--box-half",
        type=int,
        default=48,
        help="Half-size in pixels of the dashed ROI box (same helper as visualize_detection).",
    )
    args = p.parse_args()

    ac = _load_module("auto_crop", _MLSERVICE / "tools" / "auto_crop_from_detector.py")
    vd = _load_module("visualize_detection", _MLSERVICE / "tools" / "visualize_detection.py")

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

    print(f"volume_shape (Z,Y,X): {tuple(raw_vol.shape)}")
    print(f"left_center  (z,y,x): [{lz}, {ly}, {lx}]")
    print(f"right_center (z,y,x): [{rz}, {ry}, {rx}]")

    vol_norm = vd.normalize_display(raw_vol.astype(np.float32))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    vd.show_slices_with_box(
        axes[0],
        vol_norm,
        left_zyx,
        half=args.box_half,
        color="#22c55e",
        title_prefix="left ",
        imshow_origin=args.imshow_origin,
    )
    vd.show_slices_with_box(
        axes[1],
        vol_norm,
        right_zyx,
        half=args.box_half,
        color="#f97316",
        title_prefix="right ",
        imshow_origin=args.imshow_origin,
    )

    fig.suptitle(f"Heatmap detector centers — {args.dicom_dir.name}", fontsize=11)
    plt.tight_layout()
    args.output = args.output.expanduser()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
