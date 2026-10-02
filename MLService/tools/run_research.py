#!/usr/bin/env python3
"""One private schema-v1 config owner for local/notebook research runs; no cloud IO."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.sagittal_binary_cv import (
    SagittalBinaryCVConfig,
    _side_records,
    _validate_cv_config,
    _validated_cv_folds,
    run_sagittal_binary_cv,
)
from training.utils.datasphere_env import PathResolutionError, validate_explicit_environment

CV_CONFIG_FIELDS = (
    "input_path",
    "dataset_root",
    "output_dir",
    "run_id",
    "n_splits",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "early_stopping_patience",
    "lr_plateau_patience",
    "lr_plateau_factor",
    "max_grad_norm",
    "num_workers",
    "gamma",
    "features",
    "fc_hidden",
    "dropout",
    "train_augment_mode",
    "device",
    "tqdm_disable",
    "log_each_epoch",
    "log_epochs_jsonl",
)


class ResearchConfigError(ValueError):
    def __init__(self, code, report=None):
        self.code = code
        self.report = report or {"schema_version": 1, "ready": False, "code": code}
        super().__init__(code)


@dataclass(frozen=True)
class ResearchConfig:
    schema_version: int
    mode: str
    cv: SagittalBinaryCVConfig


def _resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ResearchConfigError("invalid_config_path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _check_output_paths(cfg):
    output, dataset, index = map(Path, (cfg.output_dir, cfg.dataset_root, cfg.input_path))
    if (
        output.is_relative_to(dataset)
        or dataset.is_relative_to(output)
        or index.is_relative_to(output)
    ):
        raise ResearchConfigError("output_input_overlap")
    if output.exists() and not output.is_dir():
        raise ResearchConfigError("invalid_output_path")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise ResearchConfigError("output_not_writable")


def load_research_config(path) -> ResearchConfig:
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ResearchConfigError("invalid_config")
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise ResearchConfigError("invalid_config")

    try:
        validate_explicit_environment()
        path = Path(path).expanduser().resolve()
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ResearchConfigError("config_too_large")
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant
        )
        if not isinstance(payload, dict) or set(payload) - {
            "schema_version",
            "mode",
            *CV_CONFIG_FIELDS,
        }:
            raise ResearchConfigError("invalid_config")
        if not {"schema_version", "mode", "input_path", "dataset_root", "output_dir"} <= set(
            payload
        ):
            raise ResearchConfigError("invalid_config")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["mode"] not in ("binary", "multiclass")
        ):
            raise ResearchConfigError("unsupported_config_version_or_mode")
        values = {key: payload[key] for key in CV_CONFIG_FIELDS if key in payload}
        for key in ("input_path", "dataset_root", "output_dir"):
            values[key] = str(_resolve_path(values[key], path.parent))
        if not Path(values["input_path"]).is_file() or not Path(values["dataset_root"]).is_dir():
            raise ResearchConfigError("missing_config_input")
        for key in ("tqdm_disable", "log_each_epoch", "log_epochs_jsonl"):
            if key in values and type(values[key]) is not bool:
                raise ResearchConfigError("invalid_config")
        device = values.get("device")
        if device is not None:
            import torch

            if not isinstance(device, str) or torch.device(device).type not in (
                "cpu",
                "cuda",
                "mps",
            ):
                raise ResearchConfigError("invalid_config_device")
        cfg = SagittalBinaryCVConfig(
            **values, mode=payload["mode"], crop_dir=values["dataset_root"]
        )
        _validate_cv_config(cfg)
        _check_output_paths(cfg)
        return ResearchConfig(1, payload["mode"], cfg)
    except ResearchConfigError:
        raise
    except PathResolutionError as error:
        raise ResearchConfigError(error.code) from None
    except (OSError, ValueError, RuntimeError, TypeError, RecursionError):
        raise ResearchConfigError("invalid_config") from None


def preflight_research(config: ResearchConfig) -> dict:
    from training.roi_provenance import (
        ROIValidationError,
        require_shared_roi_contract,
        validate_roi_pair,
    )
    from training.tmj_position_label_table import (
        InputValidationError,
        build_canonical_index,
        patient_group_key,
    )

    try:
        validate_explicit_environment()
        if (
            type(config.schema_version) is not int
            or config.schema_version != 1
            or config.mode not in ("binary", "multiclass")
            or config.mode != config.cv.mode
            or config.cv.legacy_name_join
            or not config.cv.input_path
        ):
            raise ResearchConfigError("invalid_config")
        _validate_cv_config(config.cv)
        _check_output_paths(config.cv)
        records = build_canonical_index(
            config.cv.input_path, config.cv.dataset_root, sagittal_only=True, require_crops=True
        )
        provenance = [validate_roi_pair(record) for record in records]
        contract = require_shared_roi_contract([result["roi_contract"] for result in provenance])
        from models.blocks import validate_research_architecture

        validate_research_architecture(
            config.cv.features,
            fc_hidden=config.cv.fc_hidden,
            crop_size=contract["preprocessing"]["crop_size"],
        )
        binary = _side_records(records, config.cv)
        try:
            folds = _validated_cv_folds(binary, config.cv)
        except ValueError:
            raise ResearchConfigError("unsuitable_cv_groups") from None
        return {
            "schema_version": 1,
            "mode": config.mode,
            "ready": True,
            "stage": "preflight",
            "study_count": len(records),
            "sample_count": len(binary),
            "patient_count": len({patient_group_key(row) for row in binary}),
            "fold_count": len(folds),
            "num_classes": 3 if config.mode == "multiclass" else 2,
            "class_support": {
                str(label): sum(row["sag"] == label for row in binary)
                for label in range(3 if config.mode == "multiclass" else 2)
            },
            "assessment": "development_cv",
            "detector_training_identity": "unknown",
            "device_availability": "not_checked",
            "source_rechecked_study_count": sum(row["source_rechecked"] for row in provenance),
            "coordinate_space_counts": {
                space: sum(row["coordinate_space"] == space for row in provenance)
                for space in ("physical-ras", "voxel")
            },
        }
    except ResearchConfigError:
        raise
    except PathResolutionError as error:
        raise ResearchConfigError(error.code) from None
    except InputValidationError as error:
        raise ResearchConfigError("invalid_training_inputs", error.report) from None
    except ROIValidationError as error:
        raise ResearchConfigError(error.code) from None
    except (OSError, ValueError, RuntimeError, TypeError):
        raise ResearchConfigError("invalid_preflight") from None


def run_research(config: ResearchConfig) -> dict:
    preflight_research(config)
    return run_sagittal_binary_cv(config.cv)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"schema_version": 1, "ready": False, "code": "invalid_arguments"}))
        self.exit(2)


def main() -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Private schema-v1 JSON config")
    parser.add_argument(
        "--preflight", action="store_true", help="Validate without model construction or training"
    )
    args = parser.parse_args()
    try:
        config = load_research_config(args.config)
        if args.preflight:
            result = preflight_research(config)
        else:
            with contextlib.redirect_stdout(sys.stderr):
                result = run_research(config)
        print(json.dumps(result, allow_nan=False))
        return 0
    except ResearchConfigError as error:
        print(json.dumps(error.report, allow_nan=False))
        return 2
    except Exception:
        print(json.dumps({"schema_version": 1, "ready": False, "code": "research_run_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
