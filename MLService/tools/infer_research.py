#!/usr/bin/env python3
"""Private sagittal research inference from one selected DICOM series and all CV folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_JSON_BYTES = 16 * 1024**2
MAX_CHECKPOINT_BYTES = 256 * 1024**2
MAX_RUN_ARTIFACT_BYTES = 1024**3
MAX_FOLDS = 64


class InferenceError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError

    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
        if len(data) > MAX_JSON_BYTES:
            raise ValueError
        return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (OSError, ValueError, TypeError, RecursionError):
        raise InferenceError("invalid_run_metadata") from None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(run, ref):
    if not isinstance(ref, dict) or set(ref) != {"file", "sha256"}:
        raise InferenceError("invalid_run_artifact")
    name, digest = ref["file"], ref["sha256"]
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise InferenceError("invalid_run_artifact")
    path = run / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != run:
        raise InferenceError("missing_run_artifact")
    if path.stat().st_size > MAX_CHECKPOINT_BYTES or _sha256(path) != digest:
        raise InferenceError("run_artifact_mismatch")
    return path


def load_run_ensemble(run_dir, expected_mode, device="cpu"):
    """Portable run contract; training source paths/identities are never consumed."""
    from models.tmj_binary_position_classifier import (
        BINARY_CLASS_SEMANTICS,
        MULTICLASS_CLASS_SEMANTICS,
        load_position_checkpoint,
    )
    from training.roi_provenance import require_shared_roi_contract

    try:
        if expected_mode not in ("binary", "multiclass"):
            raise InferenceError("invalid_mode")
        run = Path(run_dir).resolve()
        report_path = run / "report.json"
        if report_path.is_symlink():
            raise InferenceError("invalid_run_metadata")
        report = _read_json(report_path)
        count = report["n_splits"]
        classes = (
            BINARY_CLASS_SEMANTICS if expected_mode == "binary" else MULTICLASS_CLASS_SEMANTICS
        )
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or report.get("status") != "complete"
            or report.get("partial") is not False
            or report.get("mode", "binary") != expected_mode
            or report.get("class_semantics") != classes
            or type(count) is not int
            or not 2 <= count <= MAX_FOLDS
            or type(report["completed_folds"]) is not int
            or report["completed_folds"] != count
            or not isinstance(report["folds"], list)
            or len(report["folds"]) != count
        ):
            raise InferenceError("incomplete_or_incompatible_run")
        folds = sorted(report["folds"], key=lambda fold: fold["fold"])
        if any(
            type(row["fold"]) is not int or row["fold"] != index for index, row in enumerate(folds)
        ):
            raise InferenceError("invalid_run_folds")
        provenance = report["provenance"]
        if provenance.get("roi_validation") != "verified":
            raise InferenceError("missing_roi_contract")
        contract = require_shared_roi_contract([provenance.get("roi_contract")])
        replay = _artifact(run, report["artifacts"]["replay"])
        private = _read_json(replay)
        if (
            type(private.get("schema_version")) is not int
            or private["schema_version"] != 1
            or require_shared_roi_contract([private.get("roi_contract")]) != contract
            or private["config"].get("mode", "binary") != expected_mode
            or private["config"]["n_splits"] != count
        ):
            raise InferenceError("incompatible_roi_contract")
        paths = [replay]
        checkpoints = []
        for row in folds:
            checkpoint = _artifact(run, row["artifacts"]["checkpoint"])
            paths.extend((checkpoint, _artifact(run, row["artifacts"]["predictions"])))
            checkpoints.append(checkpoint)
        if sum(path.stat().st_size for path in paths) > MAX_RUN_ARTIFACT_BYTES:
            raise InferenceError("run_artifact_size_limit")
        from models.blocks import validate_research_architecture

        validate_research_architecture(
            private["config"]["features"],
            fc_hidden=private["config"]["fc_hidden"],
            crop_size=contract["preprocessing"]["crop_size"],
        )
        models, metadata = [], []
        for row, checkpoint in zip(folds, checkpoints):
            model, meta = load_position_checkpoint(
                checkpoint,
                device=device,
                expected_sha256=row["artifacts"]["checkpoint"]["sha256"],
                expected_mode=expected_mode,
            )
            if (
                meta["fold"] != row["fold"]
                or meta["best_epoch"] != row["best_epoch"]
                or meta["class_semantics"] != classes
                or any(
                    meta["model_kwargs"][key] != private["config"][key]
                    for key in ("features", "fc_hidden", "dropout")
                )
                or (
                    expected_mode == "binary"
                    and meta["threshold"] != row["threshold_from_train_youden"]
                )
                or (
                    metadata
                    and any(
                        meta[key] != metadata[0][key]
                        for key in ("model_kwargs", "preprocessing", "family", "num_classes")
                    )
                )
            ):
                raise InferenceError("incompatible_fold_metadata")
            models.append(model)
            metadata.append(meta)
        return {
            "models": models,
            "metadata": metadata,
            "roi_contract": contract,
            "classes": classes,
            "report_sha256": _sha256(report_path),
            "checkpoint_sha256": [row["artifacts"]["checkpoint"]["sha256"] for row in folds],
        }
    except InferenceError:
        raise
    except Exception:
        raise InferenceError("invalid_saved_run") from None


def aggregate_predictions(mode, predictions, thresholds):
    try:
        values = np.asarray(predictions, dtype=float)
        if values.size == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError
        if mode == "binary":
            cutoffs = np.asarray(thresholds, dtype=float)
            if (
                values.ndim != 1
                or cutoffs.shape != values.shape
                or not np.isfinite(cutoffs).all()
                or np.any((cutoffs < 0) | (cutoffs > 1))
            ):
                raise ValueError
            votes = (values >= cutoffs).astype(int)
            fraction = float(votes.mean())
            return {
                "decision": int(fraction > 0.5),
                "vote_fraction": fraction,
                "fold_decisions": votes.tolist(),
            }
        if (
            mode != "multiclass"
            or values.ndim != 2
            or values.shape[1] != 3
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError
        mean = values.mean(axis=0)
        return {"decision": int(np.argmax(mean)), "class_probabilities": mean.tolist()}
    except (TypeError, ValueError):
        raise InferenceError("invalid_model_output") from None


def _write_private(path, payload):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(payload, stream, allow_nan=False, separators=(",", ":"))


def infer_research(
    dicom_dir, left_weights, right_weights, run_dir, output_dir, *, expected_mode, device="cpu"
):
    from tools.auto_crop_from_detector import generate_roi_pair, load_paired_detector
    from training.datasets.tmj_position_dataset import _normalize_volume_percentile
    from training.roi_provenance import passport_path, validate_roi_pair

    output = Path(output_dir).resolve()
    if output.exists():
        raise InferenceError("output_exists")
    created = False
    try:
        for source in (Path(dicom_dir).resolve(), Path(run_dir).resolve()):
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise InferenceError("output_input_overlap")
        torch.device(device)
        ensemble = load_run_ensemble(run_dir, expected_mode, device=device)
        contract = ensemble["roi_contract"]
        for side, path in (("left", Path(left_weights)), ("right", Path(right_weights))):
            if (
                not path.is_file()
                or path.stat().st_size > MAX_CHECKPOINT_BYTES
                or _sha256(path) != contract["detectors"][f"{side}_sha256"]
            ):
                raise InferenceError("detector_checkpoint_mismatch")
            if output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output):
                raise InferenceError("output_input_overlap")
        left = load_paired_detector(left_weights, device)
        right = load_paired_detector(right_weights, device)
        # Keep recorded hashes bound to the checkpoint versions just loaded.
        for side, path in (("left", Path(left_weights)), ("right", Path(right_weights))):
            if (
                path.stat().st_size > MAX_CHECKPOINT_BYTES
                or _sha256(path) != contract["detectors"][f"{side}_sha256"]
            ):
                raise InferenceError("detector_checkpoint_mismatch")
        record = {
            "source_id": "inference",
            "study_id": uuid.uuid4().hex,
            "dicom_dir": str(Path(dicom_dir).resolve()),
        }
        crop_paths = {side: output / f"{side}.nii.gz" for side in ("left", "right")}
        output.mkdir(mode=0o700, parents=True)
        created = True
        generate_roi_pair(
            record,
            left,
            right,
            contract["detectors"],
            crop_paths,
            crop_size=contract["preprocessing"]["crop_size"],
            device=device,
        )
        record["crop_paths"] = {side: str(path) for side, path in crop_paths.items()}
        validated = validate_roi_pair(
            record,
            expected_detectors=contract["detectors"],
            expected_preprocessing=contract["preprocessing"],
        )
        sides = {}
        for side, path in crop_paths.items():
            volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
            tensor = (
                torch.from_numpy(_normalize_volume_percentile(volume))
                .float()[None, None]
                .to(device)
            )
            values = []
            with torch.no_grad():
                for model in ensemble["models"]:
                    logits, _ = model(tensor)
                    if not torch.isfinite(logits).all():
                        raise InferenceError("invalid_model_output")
                    if expected_mode == "binary":
                        if tuple(logits.shape) != (1, 1):
                            raise InferenceError("invalid_model_output")
                        values.append(float(torch.sigmoid(logits)[0, 0].cpu()))
                    else:
                        if tuple(logits.shape) != (1, 3):
                            raise InferenceError("invalid_model_output")
                        values.append(torch.softmax(logits, dim=1)[0].cpu().tolist())
            decision = aggregate_predictions(
                expected_mode, values, [meta["threshold"] for meta in ensemble["metadata"]]
            )
            decision["class"] = ensemble["classes"][str(decision["decision"])]
            sides[side] = decision
        passport = _read_json(passport_path(crop_paths["left"]))
        result = {
            "schema_version": 1,
            "task": "sagittal_position",
            "mode": expected_mode,
            "classes": ensemble["classes"],
            "sides": sides,
            "ensemble": {
                "fold_count": len(ensemble["models"]),
                "rule": "majority;ties-central;saved-threshold->="
                if expected_mode == "binary"
                else "mean-softmax;argmax;ties-lowest-class-index",
            },
            "provenance": {
                "roi_contract": contract,
                "classifier_preprocessing": ensemble["metadata"][0]["preprocessing"],
                "source_sha256": passport["source"]["sha256"],
                "coordinate_space": validated["coordinate_space"],
                "run_report_sha256": ensemble["report_sha256"],
                "checkpoint_sha256": ensemble["checkpoint_sha256"],
                "detector_training_independence": "unknown",
                "independent_test_assessment": "not_established",
                "calibrated_confidence": False,
            },
        }
        _write_private(output / "prediction.private.json", result)
        return {
            "schema_version": 1,
            "status": "complete",
            "mode": expected_mode,
            "side_count": 2,
            "fold_count": len(ensemble["models"]),
        }
    except InferenceError:
        if created:
            _write_private(
                output / "failure.private.json",
                {"schema_version": 1, "status": "failed", "code": "inference_failed"},
            )
        raise
    except Exception:
        if created:
            _write_private(
                output / "failure.private.json",
                {"schema_version": 1, "status": "failed", "code": "inference_failed"},
            )
        raise InferenceError("inference_failed") from None


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"schema_version": 1, "status": "failed", "code": "invalid_arguments"}))
        self.exit(2)


def main():
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--dicom-series", required=True)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("binary", "multiclass"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = infer_research(
            args.dicom_series,
            args.left_model,
            args.right_model,
            args.run,
            args.output,
            expected_mode=args.mode,
            device=args.device,
        )
        print(json.dumps(result, allow_nan=False))
        return 0
    except InferenceError as error:
        print(json.dumps({"schema_version": 1, "status": "failed", "code": error.code}))
        return 2
    except Exception:
        print(json.dumps({"schema_version": 1, "status": "failed", "code": "inference_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
