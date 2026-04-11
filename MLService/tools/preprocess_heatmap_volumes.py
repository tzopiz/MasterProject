#!/usr/bin/env python3
"""
Pre-process DICOM studies for heatmap detector training.

Loads each study, applies HU normalization and 6× downsampling,
saves as uint8 .npy (96×128×128, ~1.2 MB each → ~130 MB total).

Output directory structure:
    data/heatmap_volumes/
        study_0001.npy   # uint8, shape (96, 128, 128)
        study_0002.npy
        ...

Usage:
    ./venv/bin/python tools/preprocess_heatmap_volumes.py
    ./venv/bin/python tools/preprocess_heatmap_volumes.py --skip-existing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage
from tqdm import tqdm

DATASET_DIR  = Path("data/dataset_cbct_public")
SPLIT_JSON   = Path("data/detector_split.json")
OUTPUT_DIR   = Path("data/heatmap_volumes")
DS_FACTOR    = 6


def load_and_preprocess(study_dir: Path) -> np.ndarray:
    """Load DICOM → normalize [0,255] → downsample 6× → uint8 (96,128,128)."""
    files = sorted(study_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm in {study_dir}")

    slices = [pydicom.dcmread(str(f)) for f in files]
    slices.sort(key=lambda s: float(s.InstanceNumber))

    planes = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        planes.append(arr)
    volume = np.stack(planes, axis=0)   # (D, H, W) float32

    # Normalize to [0, 255]
    p2, p98 = np.percentile(volume, [2, 98])
    volume = np.clip(volume, p2, p98)
    volume = ((volume - p2) / max(p98 - p2, 1.0) * 255).astype(np.float32)

    # Downsample 6×
    zoom = [1.0 / DS_FACTOR] * 3
    volume = ndimage.zoom(volume, zoom, order=1)

    return volume.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--split-json",  default=str(SPLIT_JSON))
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip studies already processed")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_json) as f:
        split = json.load(f)

    all_studies = split["train"] + split["val"] + split["test"]
    print(f"Studies to process: {len(all_studies)}")

    done, skipped, failed = 0, 0, 0
    for study_id in tqdm(all_studies, desc="Preprocessing"):
        out_path = output_dir / f"{study_id}.npy"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        study_dir = dataset_dir / study_id
        if not study_dir.exists():
            tqdm.write(f"  SKIP {study_id}: directory not found")
            failed += 1
            continue

        try:
            vol = load_and_preprocess(study_dir)
            np.save(str(out_path), vol)
            done += 1
        except Exception as e:
            tqdm.write(f"  FAIL {study_id}: {e}")
            failed += 1

    total_mb = sum(p.stat().st_size for p in output_dir.glob("*.npy")) / 1e6
    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"Output: {output_dir}  ({total_mb:.0f} MB)")
    print(f"Shape: {vol.shape} uint8 per study")


if __name__ == "__main__":
    main()
