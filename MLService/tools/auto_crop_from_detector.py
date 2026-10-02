#!/usr/bin/env python3
"""
Automatic ROI cropping using trained TMJ heatmap detectors.

Two separate single-joint detectors (left + right), each TMJHeatmapDetector(out_channels=1).

Usage (historical selection is explicit legacy):
    ./venv/bin/python tools/auto_crop_from_detector.py \
        --legacy-input \
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.blocks import validate_research_architecture
from training.roi_provenance import (
    DETECTOR_FAMILY,
    PREPROCESSING,
    SUPPORTED_PROFILE_HELP,
    ROIValidationError,
    extract_fixed_crop,
    load_validated_series,
    preprocessing_options,
    sha256_file,
    study_key,
    validate_roi_pair,
    write_roi_crop,
)
from training.tmj_position_label_table import InputValidationError, build_canonical_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SHAPE = (96, 128, 128)


# ── Model (inline, no import dependency) ──────────────────────────────────


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
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
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.pad(
                x,
                [
                    0,
                    skip.shape[4] - x.shape[4],
                    0,
                    skip.shape[3] - x.shape[3],
                    0,
                    skip.shape[2] - x.shape[2],
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class TMJHeatmapDetector(nn.Module):
    def __init__(self, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        validate_research_architecture(features)
        self.encoders = nn.ModuleList()
        prev = 1
        for f in features:
            self.encoders.append(_EncoderBlock(prev, f))
            prev = f
        self.bottleneck = _double_conv(features[-1], features[-1] * 2)
        prev = features[-1] * 2
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(_DecoderBlock(prev, f, f))
            prev = f
        self.head = nn.Conv3d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.head(x)


# ── Shared bounded volume / crop path ──────────────────────────────────────


def load_dicom_volume(dicom_dir):
    """Compatibility array return; new writers retain load_validated_series metadata."""
    return load_validated_series(dicom_dir)[0]


def normalize(vol):
    if vol.ndim != 3 or not np.isfinite(vol).all():
        raise ROIValidationError("invalid_source_pixels")
    p2, p98 = np.percentile(vol, PREPROCESSING["percentiles"])
    vol = np.clip(vol, p2, p98)
    denom = p98 - p2
    return ((vol - p2) / denom if denom > 0 else np.zeros_like(vol)).astype(np.float32)


def prepare_input(vol):
    """Historical percentile + scipy zoom convention, unchanged target coordinates."""
    orig_shape = np.array(vol.shape, dtype=float)
    vol = normalize(vol)
    if tuple(vol.shape) != TARGET_SHAPE:
        vol = ndimage.zoom(
            vol,
            [t / s for t, s in zip(TARGET_SHAPE, vol.shape)],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=True,
            grid_mode=False,
        ).astype(np.float32)
    return torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0), orig_shape


def argmax_to_orig(hm, orig_shape):
    if hm.shape != TARGET_SHAPE or not np.isfinite(hm).all():
        raise ROIValidationError("invalid_detector_output")
    idx = np.unravel_index(hm.argmax(), hm.shape)
    coords = (np.array(idx, dtype=float) * orig_shape / np.array(TARGET_SHAPE)).astype(int)
    return np.clip(coords, 0, orig_shape.astype(int) - 1)


def extract_crop(vol, center, crop_size=128):
    return extract_fixed_crop(vol, center, crop_size)[0]


def load_paired_detector(path, device):
    """Historical architecture; optional explicit features support bounded CPU checks."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("detector_family", DETECTOR_FAMILY) != DETECTOR_FAMILY:
            raise ROIValidationError("unsupported_detector_family")
        config = checkpoint.get("model_config", {})
        features = config.get("features", [32, 64, 128, 256])
        if (
            not isinstance(features, list)
            or not 1 <= len(features) <= 4
            or any(type(f) is not int or f <= 0 for f in features)
        ):
            raise ROIValidationError("unsupported_detector_configuration")
        try:
            validate_research_architecture(features)
        except ValueError:
            raise ROIValidationError("unsupported_detector_configuration") from None
        model = TMJHeatmapDetector(features=features)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.eval().to(device)
    except ROIValidationError:
        raise
    except Exception:
        raise ROIValidationError("incompatible_detector_checkpoint") from None


def generate_roi_pair(
    record,
    left_model,
    right_model,
    detectors,
    crop_paths,
    *,
    crop_size=128,
    device="cpu",
    skip_existing=False,
):
    """Reusable source→paired ROI seam; cache mismatch regenerates BOTH sides."""
    output_record = {
        **record,
        "crop_paths": {side: str(crop_paths[side]) for side in ("left", "right")},
    }
    options = preprocessing_options(crop_size)
    if skip_existing:
        try:
            validate_roi_pair(
                output_record, expected_detectors=detectors, expected_preprocessing=options
            )
            return "skipped"
        except ROIValidationError:
            logger.info("ROI cache invalid; regenerating pair")
    raw_vol, source = load_validated_series(record["dicom_dir"])
    tensor, original_shape = prepare_input(raw_vol)
    with torch.no_grad():
        tensor = tensor.to(device)
        for side, model in (("left", left_model), ("right", right_model)):
            heatmap = torch.sigmoid(model(tensor)).squeeze(0).squeeze(0).cpu().numpy()
            center = argmax_to_orig(heatmap, original_shape)
            write_roi_crop(
                raw_vol,
                center,
                crop_size,
                crop_paths[side],
                side=side,
                record=record,
                source=source,
                detectors=detectors,
            )
    validate_roi_pair(output_record, expected_detectors=detectors, expected_preprocessing=options)
    return "generated"


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps({"ready": False, "code": "invalid_arguments"}))
        self.exit(2)


def main():
    parser = _SafeParser(description=__doc__, epilog=SUPPORTED_PROFILE_HELP)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--canonical-input", help="Strict schema-v1 private study/label JSON")
    parser.add_argument("--dataset-root", help="Root for canonical relative paths and output")
    parser.add_argument("--index-output", help="Generated strict private index (*.private.json)")
    parser.add_argument(
        "--legacy-input",
        action="store_true",
        help="Explicit historical split/study-ID path; not a canonical label join",
    )
    parser.add_argument("--dataset", default="data/dataset_cbct_public")
    parser.add_argument("--split-json", default="data/detector_split.json")
    parser.add_argument("--studies", nargs="*")
    parser.add_argument("--output", default="data/detector_crops_v2")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    try:
        preprocessing_options(args.crop_size)
        output = Path(args.output).resolve()
        canonical = None
        if args.canonical_input:
            if args.legacy_input or args.studies or not args.dataset_root:
                raise ROIValidationError("invalid_arguments")
            root = Path(args.dataset_root).resolve()
            if not output.is_relative_to(root):
                raise ROIValidationError("output_outside_dataset_root")
            index_output = (
                Path(args.index_output) if args.index_output else output / "inputs.private.json"
            )
            if not index_output.name.endswith(".private.json"):
                raise ROIValidationError("private_index_suffix_required")
            records = build_canonical_index(args.canonical_input, root, require_crops=False)
            canonical = json.loads(Path(args.canonical_input).read_text(encoding="utf-8"))
        else:
            if not args.legacy_input or args.index_output:
                raise ROIValidationError("canonical_input_required")
            root = Path(args.dataset).resolve()
            if args.studies:
                study_ids = args.studies
            else:
                split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
                study_ids = split["train"] + split["val"] + split["test"]
            records = []
            for sid in study_ids:
                if not isinstance(sid, str) or not sid:
                    raise ROIValidationError("invalid_legacy_study")
                series = (root / sid).resolve()
                if (
                    Path(sid).is_absolute()
                    or not series.is_relative_to(root)
                    or not series.is_dir()
                ):
                    raise ROIValidationError("invalid_legacy_study")
                records.append(
                    {"source_id": "legacy-unverified", "study_id": sid, "dicom_dir": str(series)}
                )
        device = torch.device(
            args.device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        detectors = {
            "family": DETECTOR_FAMILY,
            "left_sha256": sha256_file(args.left_model),
            "right_sha256": sha256_file(args.right_model),
        }
        left = load_paired_detector(args.left_model, device)
        right = load_paired_detector(args.right_model, device)
        # Detect a changing checkpoint instead of attaching its old hash to new weights.
        if detectors["left_sha256"] != sha256_file(args.left_model) or detectors[
            "right_sha256"
        ] != sha256_file(args.right_model):
            raise ROIValidationError("changing_detector_checkpoint")
        done = skipped = failed = 0
        generated = {}
        for record in records:
            paths = {
                side: output / study_key(record) / f"{side}.nii.gz" for side in ("left", "right")
            }
            try:
                if any(not path.resolve().is_relative_to(output) for path in paths.values()):
                    raise ROIValidationError("output_path_escape")
                status = generate_roi_pair(
                    record,
                    left,
                    right,
                    detectors,
                    paths,
                    crop_size=args.crop_size,
                    device=device,
                    skip_existing=args.skip_existing,
                )
                if status == "skipped":
                    skipped += 1
                else:
                    done += 1
                generated[(record["source_id"], record["study_id"])] = paths
            except ROIValidationError as error:
                logger.error("ROI generation failed: %s", error.code)
                failed += 1
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": 1,
            "total": len(records),
            "generated": done,
            "skipped": skipped,
            "failed": failed,
            "detectors": detectors,
            "preprocessing": preprocessing_options(args.crop_size),
            "detector_training_identity": "unknown",
            "canonical_index_written": False,
        }
        if canonical is not None and failed == 0:
            for row in canonical["studies"]:
                row["crops"] = {
                    side: str(path.relative_to(root))
                    for side, path in generated[(row["source_id"], row["study_id"])].items()
                }
            index_output.parent.mkdir(parents=True, exist_ok=True)
            index_output.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
            summary["canonical_index_written"] = True
        (output / "crop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
        return 2 if failed else 0
    except InputValidationError as error:
        print(json.dumps(error.report))
        return 2
    except ROIValidationError as error:
        print(json.dumps({"ready": False, "code": error.code}))
        return 2
    except Exception:
        # Arg paths, checkpoint metadata and decoder errors must stay private.
        print(json.dumps({"ready": False, "code": "unreadable_crop_input"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
