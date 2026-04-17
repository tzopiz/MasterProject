#!/usr/bin/env python3
"""
Batch annotation tool for TMJ ROI detection.

Opens all studies one by one in the annotation tool.
Skips already annotated studies (use --redo to re-annotate all).

Usage:
    # Annotate only missing studies
    ./venv/bin/python tools/batch_annotate.py

    # Re-annotate ALL studies from scratch
    ./venv/bin/python tools/batch_annotate.py --redo

    # Start from a specific study
    ./venv/bin/python tools/batch_annotate.py --from study_0038
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("data/dataset_cbct_public")
ANNOTATIONS_DIR = Path("data/roi_annotations")
TOOL = Path("tools/roi_annotation_tool.py")


def main():
    parser = argparse.ArgumentParser(description="Batch TMJ annotation")
    parser.add_argument(
        "--redo", action="store_true", help="Re-annotate all studies, overwriting existing"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        default=None,
        help="Start from this study_id (skip earlier ones)",
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--output", default=str(ANNOTATIONS_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    ann_dir = Path(args.output)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # All studies, sorted
    all_studies = sorted(
        d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("study_")
    )
    if not all_studies:
        print(f"No studies found in {dataset_dir}")
        sys.exit(1)

    # Already annotated
    annotated = {p.stem.replace("_rois", "") for p in ann_dir.glob("*_rois.json")}

    # Decide which to process
    if args.redo:
        to_annotate = all_studies
        print(f"Re-annotating ALL {len(to_annotate)} studies (--redo)")
    else:
        to_annotate = [s for s in all_studies if s not in annotated]
        print(f"Already annotated: {len(annotated)}")
        print(f"To annotate: {len(to_annotate)}")

    if args.start_from:
        if args.start_from not in to_annotate:
            print(f"'{args.start_from}' not in queue (already done or not found)")
        else:
            idx = to_annotate.index(args.start_from)
            to_annotate = to_annotate[idx:]
            print(f"Starting from {args.start_from} ({len(to_annotate)} remaining)")

    if not to_annotate:
        print("Nothing to annotate!")
        sys.exit(0)

    print("\nControls: Left Click = place point, L/R = switch side, S = save, U = undo, Q = quit\n")
    print("─" * 60)

    done = 0
    for i, study_id in enumerate(to_annotate, 1):
        dicom_dir = dataset_dir / study_id
        if not dicom_dir.exists():
            print(f"[{i}/{len(to_annotate)}] SKIP {study_id} (directory not found)")
            continue

        print(f"\n[{i}/{len(to_annotate)}] {study_id}  (total done this session: {done})")

        result = subprocess.run(
            [sys.executable, str(TOOL), str(dicom_dir), "--output", str(ann_dir)],
        )

        # Check if annotation was saved
        ann_file = ann_dir / f"{study_id}_rois.json"
        if ann_file.exists():
            done += 1
            print(f"  ✓ Saved ({done} done, {len(to_annotate) - i} remaining)")
        else:
            print("  ✗ Not saved (skipped?)")

        if result.returncode != 0:
            ans = input("\nAnnotation tool exited with error. Continue? [y/N]: ")
            if ans.lower() != "y":
                break

    print("\n" + "─" * 60)
    total_annotated = len(list(ann_dir.glob("*_rois.json")))
    print(f"Session complete: {done} annotated this run")
    print(f"Total in {ann_dir}: {total_annotated} / {len(all_studies)} studies")


if __name__ == "__main__":
    main()
