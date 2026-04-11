#!/usr/bin/env python3
"""
Automatic ROI cropping using trained TMJ heatmap detectors.

Two separate single-joint detectors (left + right), each TMJHeatmapDetector(out_channels=1).

Usage:
    ./venv/bin/python tools/auto_crop_from_detector.py \
        --left-model  models/checkpoints/left_detector.pth  \
        --right-model models/checkpoints/right_detector.pth \
        --dataset     data/dataset_cbct_public              \
        --output      data/detector_crops_v2               \
        --crop-size   128
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SHAPE = (96, 128, 128)


# ── Model (inline, no import dependency) ──────────────────────────────────

def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
    )


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[4]-x.shape[4],
                           0, skip.shape[3]-x.shape[3],
                           0, skip.shape[2]-x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))


class TMJHeatmapDetector(nn.Module):
    def __init__(self, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.encoders = nn.ModuleList()
        prev = 1
        for f in features:
            self.encoders.append(_EncoderBlock(prev, f)); prev = f
        self.bottleneck = _double_conv(features[-1], features[-1] * 2)
        prev = features[-1] * 2
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(_DecoderBlock(prev, f, f)); prev = f
        self.head = nn.Conv3d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x); skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


# ── Volume I/O ────────────────────────────────────────────────────────────

def load_dicom_volume(dicom_dir: Path) -> np.ndarray:
    files = sorted(dicom_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm in {dicom_dir}")
    slices = [pydicom.dcmread(str(f)) for f in files]
    slices.sort(key=lambda s: float(s.InstanceNumber))
    planes = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        planes.append(arr)
    return np.stack(planes, axis=0)


def normalize(vol: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(vol, [2, 98])
    vol = np.clip(vol, p2, p98)
    denom = p98 - p2
    return ((vol - p2) / denom if denom > 0 else np.zeros_like(vol)).astype(np.float32)


def prepare_input(vol: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """Normalize + resize to TARGET_SHAPE. Returns (tensor, orig_shape)."""
    orig_shape = np.array(vol.shape, dtype=float)
    vol = normalize(vol)
    if tuple(vol.shape) != TARGET_SHAPE:
        zoom = [t / s for t, s in zip(TARGET_SHAPE, vol.shape)]
        vol = ndimage.zoom(vol, zoom, order=1).astype(np.float32)
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    return t, orig_shape


def argmax_to_orig(hm: np.ndarray, orig_shape: np.ndarray) -> np.ndarray:
    """Find peak in heatmap and scale back to original voxel space."""
    idx = np.unravel_index(hm.argmax(), hm.shape)
    ds_coords = np.array(idx, dtype=float)
    scale = orig_shape / np.array(TARGET_SHAPE, dtype=float)
    orig_coords = (ds_coords * scale).astype(int)
    orig_coords = np.clip(orig_coords, 0, orig_shape.astype(int) - 1)
    return orig_coords


# ── Crop extraction ───────────────────────────────────────────────────────

def extract_crop(vol: np.ndarray, center: np.ndarray, crop_size: int = 128) -> np.ndarray:
    D, H, W = vol.shape
    half = crop_size // 2
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    zs, ze = max(0, z-half), min(D, z+half)
    ys, ye = max(0, y-half), min(H, y+half)
    xs, xe = max(0, x-half), min(W, x+half)
    crop = vol[zs:ze, ys:ye, xs:xe]
    if crop.shape != (crop_size, crop_size, crop_size):
        pad = np.zeros((crop_size, crop_size, crop_size), dtype=crop.dtype)
        pz = (crop_size - crop.shape[0]) // 2
        py = (crop_size - crop.shape[1]) // 2
        px_ = (crop_size - crop.shape[2]) // 2
        pad[pz:pz+crop.shape[0], py:py+crop.shape[1], px_:px_+crop.shape[2]] = crop
        crop = pad
    return crop


def save_nifti(arr: np.ndarray, path: Path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left-model",  required=True)
    p.add_argument("--right-model", required=True)
    p.add_argument("--dataset",     default="data/dataset_cbct_public")
    p.add_argument("--split-json",  default="data/detector_split.json",
                   help="Use all studies from split JSON (train+val+test). "
                        "Ignored if --studies is given.")
    p.add_argument("--studies",     nargs="*",
                   help="Explicit list of study IDs to process")
    p.add_argument("--output",      default="data/detector_crops_v2")
    p.add_argument("--crop-size",   type=int, default=128)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device",      default=None)
    args = p.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load models
    def load_model(path):
        ckpt = torch.load(path, map_location="cpu")
        m = TMJHeatmapDetector()
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval().to(device)
        logger.info("Loaded %s (ep %s, MAE %.2f ds_px)",
                    Path(path).name, ckpt.get("epoch", "?"),
                    ckpt.get("best_val_mae", float("nan")))
        return m

    left_model  = load_model(args.left_model)
    right_model = load_model(args.right_model)

    # Studies
    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.studies:
        study_ids = args.studies
    else:
        with open(args.split_json) as f:
            split = json.load(f)
        study_ids = split["train"] + split["val"] + split["test"]

    logger.info("Studies to process: %d", len(study_ids))

    done, skipped, failed = 0, 0, 0
    all_meta = []

    for sid in tqdm(study_ids, desc="Cropping"):
        out_study = output_dir / sid
        left_nii  = out_study / f"{sid}_left.nii.gz"
        right_nii = out_study / f"{sid}_right.nii.gz"

        if args.skip_existing and left_nii.exists() and right_nii.exists():
            skipped += 1
            continue

        study_dir = dataset_dir / sid
        if not study_dir.exists():
            logger.warning("SKIP %s: directory not found", sid)
            failed += 1
            continue

        try:
            raw_vol = load_dicom_volume(study_dir)
            inp, orig_shape = prepare_input(raw_vol)
            inp = inp.to(device)

            with torch.no_grad():
                left_hm  = torch.sigmoid(left_model(inp)).squeeze().cpu().numpy()
                right_hm = torch.sigmoid(right_model(inp)).squeeze().cpu().numpy()

            left_orig  = argmax_to_orig(left_hm,  orig_shape)
            right_orig = argmax_to_orig(right_hm, orig_shape)

            # Extract crops from raw (non-normalized) volume for classifier
            left_crop  = extract_crop(raw_vol, left_orig,  args.crop_size)
            right_crop = extract_crop(raw_vol, right_orig, args.crop_size)

            out_study.mkdir(exist_ok=True)
            save_nifti(left_crop,  left_nii)
            save_nifti(right_crop, right_nii)

            meta = {
                "study": sid,
                "volume_shape": list(raw_vol.shape),
                "predicted_coords": {
                    "left":  left_orig.tolist(),
                    "right": right_orig.tolist(),
                },
                "crop_size": args.crop_size,
                "crop_paths": {
                    "left":  str(left_nii.relative_to(output_dir.parent)),
                    "right": str(right_nii.relative_to(output_dir.parent)),
                },
            }
            with open(out_study / f"{sid}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            all_meta.append(meta)
            done += 1

        except Exception as e:
            logger.error("FAIL %s: %s", sid, e)
            failed += 1

    summary = {
        "left_model":  args.left_model,
        "right_model": args.right_model,
        "crop_size":   args.crop_size,
        "total": len(study_ids), "done": done,
        "skipped": skipped, "failed": failed,
        "studies": all_meta,
    }
    with open(output_dir / "crop_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done: %d  Skipped: %d  Failed: %d", done, skipped, failed)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
