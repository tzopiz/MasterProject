#!/usr/bin/env python3
"""
Generate deterministic train/val/test split from all available ROI annotations.

Run once after annotating new studies:
    ./venv/bin/python tools/create_detector_split.py

Output: data/detector_split.json
"""
import json, random
from pathlib import Path

ANNOTATIONS_DIR = Path("data/roi_annotations")
OUTPUT = Path("data/detector_split.json")
SEED = 42
VAL_FRAC  = 0.14
TEST_FRAC = 0.06

def main():
    studies = sorted(p.stem.replace("_rois", "") for p in ANNOTATIONS_DIR.glob("*_rois.json"))
    print(f"Annotated studies: {len(studies)}")

    rng = random.Random(SEED)
    shuffled = studies[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * TEST_FRAC))
    n_val  = max(1, round(n * VAL_FRAC))
    n_train = n - n_test - n_val

    split = {
        "seed": SEED,
        "total": n,
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == n

    OUTPUT.write_text(json.dumps(split, indent=2))
    print(f"Train: {len(split['train'])}  Val: {len(split['val'])}  Test: {len(split['test'])}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
