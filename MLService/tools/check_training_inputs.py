#!/usr/bin/env python3
"""Validate canonical study/label inputs locally, without models or training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.roi_provenance import SUPPORTED_PROFILE_HELP, ROIValidationError, validate_roi_pair
from training.tmj_position_label_table import InputValidationError, build_canonical_index


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        report = InputValidationError([{"code": "invalid_arguments", "collection": "input"}]).report
        print(json.dumps(report, ensure_ascii=False))
        self.exit(2)


def main():
    parser = _SafeArgumentParser(description=__doc__, epilog=SUPPORTED_PROFILE_HELP)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--all-planes", action="store_true", help="Require sagittal and frontal labels"
    )
    parser.add_argument(
        "--allow-uncropped",
        action="store_true",
        help="Validate source-series intake only; not training-ready",
    )
    args = parser.parse_args()
    try:
        records = build_canonical_index(
            args.input_json,
            args.dataset_root,
            sagittal_only=not args.all_planes,
            require_crops=not args.allow_uncropped,
        )
    except InputValidationError as error:
        print(json.dumps(error.report, ensure_ascii=False))
        return 2
    provenance = []
    if not args.allow_uncropped:
        for number, record in enumerate(records, 1):
            try:
                provenance.append(validate_roi_pair(record))
            except ROIValidationError as error:
                report = InputValidationError(
                    [{"code": error.code, "collection": "studies", "row": number}]
                ).report
                report["stage"] = "roi"
                print(json.dumps(report, ensure_ascii=False))
                return 2
    report = {
        "schema_version": 1,
        "stage": "intake-only" if args.allow_uncropped else "preflight",
        "ready": not args.allow_uncropped,
        "intake_ready": True,
        "training_ready": not args.allow_uncropped,
        "pending_checks": ["roi_generation", "roi_provenance"] if args.allow_uncropped else [],
        "study_count": len(records),
        "patient_count": len({row["patient_key"] for row in records}),
        "diagnostics": [],
    }
    if provenance:
        rechecked = sum(item["source_rechecked"] for item in provenance)
        report["roi_provenance"] = {
            "source_rechecked_study_count": rechecked,
            "cached_only_study_count": len(provenance) - rechecked,
            "coordinate_space_counts": {
                space: sum(item["coordinate_space"] == space for item in provenance)
                for space in ("physical-ras", "voxel")
            },
            "detector_training_identity": "unknown",
        }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
