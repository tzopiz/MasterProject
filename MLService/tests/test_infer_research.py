"""Actual synthetic DICOM/heatmap/saved classifier connection, not clinical quality."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from .test_roi_provenance import write_series
from .test_sagittal_binary_cv import _canonical_synthetic_config


def inference():
    from tools import infer_research

    return infer_research


@pytest.fixture(autouse=True)
def small_cpu_threads(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    yield
    torch.set_num_threads(previous)


def make_saved_run(tmp_path, mode="binary"):
    from tools.auto_crop_from_detector import TMJHeatmapDetector
    from training import roi_provenance as roi
    from training import sagittal_binary_cv as cv

    cfg = _canonical_synthetic_config(tmp_path)
    weights = {}
    for side in ("left", "right"):
        torch.manual_seed(19 if side == "left" else 20)
        detector = TMJHeatmapDetector(features=[1])
        path = tmp_path / f"{side}.pth"
        torch.save(
            {
                "detector_family": roi.DETECTOR_FAMILY,
                "model_config": {"features": [1]},
                "model_state_dict": detector.state_dict(),
            },
            path,
        )
        weights[side] = path
    detectors = {
        "family": roi.DETECTOR_FAMILY,
        **{f"{side}_sha256": roi.sha256_file(path) for side, path in weights.items()},
    }
    index = json.loads(Path(cfg.input_path).read_text())
    for number, row in enumerate(index["studies"]):
        for side in ("left", "right"):
            path = Path(cfg.dataset_root) / row["crops"][side]
            passport = json.loads(roi.passport_path(path).read_text())
            volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)
            roi.write_roi_crop(
                volume,
                (4, 4, 4),
                8,
                path,
                side=side,
                record=row,
                source=passport["source"],
                detectors=detectors,
            )
        if mode == "multiclass":
            index["labels"][number]["labels"]["sagittal"] = {
                "left": 1 + number % 3,
                "right": 1 + number % 3,
            }
    Path(cfg.input_path).write_text(json.dumps(index))
    cfg.mode, cfg.train_augment_mode = mode, "none"
    cfg.output_dir, cfg.run_id = str(tmp_path / "runs"), "SyntheticPrivateRun"
    report = cv.run_sagittal_binary_cv(cfg)
    run = Path(cfg.output_dir) / cfg.run_id
    return run, weights, report, cfg


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_real_dicom_to_all_trained_folds_and_two_private_sides(tmp_path, mode):
    module = inference()
    run, weights, trained_report, cfg = make_saved_run(tmp_path, mode)
    # Portable saved run: neither raw training index nor training crop paths remain.
    shutil.rmtree(cfg.dataset_root)
    Path(cfg.input_path).unlink()
    source = write_series(tmp_path / "SyntheticPrivateSource")
    destination = tmp_path / "predictions"
    summary = module.infer_research(
        source,
        weights["left"],
        weights["right"],
        run,
        destination,
        expected_mode=mode,
        device="cpu",
    )
    assert summary == {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "side_count": 2,
        "fold_count": 2,
    }
    private = json.loads((destination / "prediction.private.json").read_text())
    assert private["task"] == "sagittal_position" and private["mode"] == mode
    assert set(private["sides"]) == {"left", "right"}
    assert private["provenance"]["detector_training_independence"] == "unknown"
    assert "SyntheticPrivate" not in json.dumps(private)
    assert str(tmp_path) not in json.dumps(private)
    assert "frontal" not in json.dumps(private)
    from models.tmj_binary_position_classifier import load_position_checkpoint
    from training.datasets.tmj_position_dataset import _normalize_volume_percentile

    for side in ("left", "right"):
        image = nib.load(destination / f"{side}.nii.gz")
        assert image.shape == (8, 8, 8)
        values = _normalize_volume_percentile(np.asarray(image.dataobj, dtype=np.float32))
        tensor = torch.from_numpy(values).float()[None, None]
        outputs, thresholds = [], []
        for fold in trained_report["folds"]:
            ref = fold["artifacts"]["checkpoint"]
            model, metadata = load_position_checkpoint(
                run / ref["file"], expected_sha256=ref["sha256"], expected_mode=mode
            )
            with torch.no_grad():
                logits, _ = model(tensor)
            outputs.append(
                float(torch.sigmoid(logits)[0, 0])
                if mode == "binary"
                else torch.softmax(logits, dim=1)[0].tolist()
            )
            thresholds.append(metadata["threshold"])
        result = private["sides"][side]
        if mode == "binary":
            votes = [int(p >= t) for p, t in zip(outputs, thresholds)]
            assert result["decision"] == int(sum(votes) > len(votes) / 2)
            assert result["vote_fraction"] == sum(votes) / len(votes)
            assert "probability" not in result
        else:
            mean = np.mean(outputs, axis=0)
            assert result["class_probabilities"] == pytest.approx(mean.tolist())
            assert result["decision"] == int(np.argmax(mean))
        assert result["class"] == private["classes"][str(result["decision"])]
    before = (destination / "prediction.private.json").read_bytes()
    with pytest.raises(module.InferenceError, match="output_exists"):
        module.infer_research(
            source, weights["left"], weights["right"], run, destination, expected_mode=mode
        )
    assert (destination / "prediction.private.json").read_bytes() == before


def test_ensemble_ties_and_saved_binary_threshold_boundary():
    module = inference()
    result = module.aggregate_predictions("binary", [0.4, 0.9], [0.4, 0.95])
    assert result["decision"] == 0 and result["vote_fraction"] == 0.5
    assert "probability" not in result
    result = module.aggregate_predictions(
        "multiclass", [[0.5, 0.5, 0], [0.5, 0.5, 0]], [None, None]
    )
    assert result["decision"] == 0 and result["class_probabilities"] == [0.5, 0.5, 0.0]


@pytest.mark.parametrize(
    "damage",
    [
        "partial",
        "missing",
        "corrupt",
        "folds",
        "mode",
        "no_contract",
        "escape",
        "symlink",
        "detector",
        "generation_options",
        "family",
        "preprocessing",
        "checkpoint_fold",
        "replay",
    ],
)
def test_saved_run_failures_are_safe_before_outputs(tmp_path, damage):
    module = inference()
    run, weights, report, cfg = make_saved_run(tmp_path)
    if damage == "partial":
        report["status"] = "failed"
    elif damage == "missing":
        (run / report["folds"][0]["artifacts"]["checkpoint"]["file"]).unlink()
    elif damage == "corrupt":
        (run / report["folds"][0]["artifacts"]["checkpoint"]["file"]).write_bytes(
            b"SyntheticPrivate"
        )
    elif damage == "folds":
        report["folds"][1]["fold"] = 0
    elif damage == "mode":
        report["mode"] = "multiclass"
    elif damage == "no_contract":
        report["provenance"].pop("roi_contract", None)
    elif damage == "escape":
        report["folds"][0]["artifacts"]["checkpoint"]["file"] = "../SyntheticPrivate.pth"
    elif damage == "symlink":
        path = run / report["folds"][0]["artifacts"]["checkpoint"]["file"]
        outside = tmp_path / "SyntheticPrivate.pth"
        path.replace(outside)
        path.symlink_to(outside)
    elif damage == "detector":
        weights["left"].write_bytes(b"changed")
    elif damage == "generation_options":
        report["provenance"]["roi_contract"]["preprocessing"]["crop_size"] = 16
    elif damage == "replay":
        (run / "replay.private.json").write_text("{}")
    else:
        ref = report["folds"][0]["artifacts"]["checkpoint"]
        path = run / ref["file"]
        payload = torch.load(path, weights_only=True)
        if damage == "family":
            payload["family"] = "SyntheticPrivateUnknownFamily"
        elif damage == "preprocessing":
            payload["preprocessing"]["clip_percentiles"] = [1, 99]
        else:
            payload["fold"] = 3
        torch.save(payload, path)
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))
    destination = tmp_path / "prediction"
    with pytest.raises(module.InferenceError) as error:
        module.infer_research(
            tmp_path / "absent-source",
            weights["left"],
            weights["right"],
            run,
            destination,
            expected_mode="binary",
        )
    assert "SyntheticPrivate" not in str(error.value) and str(tmp_path) not in str(error.value)
    assert not destination.exists()


def test_invalid_dicom_and_cli_argument_privacy(tmp_path):
    module = inference()
    run, weights, report, cfg = make_saved_run(tmp_path)
    source = write_series(tmp_path / "SyntheticPrivateSource")
    import pydicom

    path = next(source.glob("*.dcm"))
    ds = pydicom.dcmread(path)
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(path, enforce_file_format=True)
    destination = tmp_path / "prediction"
    with pytest.raises(module.InferenceError) as error:
        module.infer_research(
            source, weights["left"], weights["right"], run, destination, expected_mode="binary"
        )
    assert "SyntheticPrivate" not in str(error.value)
    result = subprocess.run(
        [sys.executable, module.__file__, "--mode=SyntheticPrivate"], capture_output=True, text=True
    )
    assert result.returncode == 2 and json.loads(result.stdout)["code"] == "invalid_arguments"
    assert "SyntheticPrivate" not in result.stdout + result.stderr


@pytest.mark.parametrize("entrypoint", ["preflight", "cv"])
def test_mixed_roi_contract_refuses_before_model_or_outputs(tmp_path, entrypoint):
    from tools.run_research import ResearchConfig, preflight_research
    from training import roi_provenance as roi
    from training import sagittal_binary_cv as cv

    cfg = _canonical_synthetic_config(tmp_path)
    cfg.output_dir, cfg.run_id = str(tmp_path / "runs"), "refused"
    index = json.loads(Path(cfg.input_path).read_text())
    for side in ("left", "right"):
        path = Path(cfg.dataset_root) / index["studies"][0]["crops"][side]
        passport = json.loads(roi.passport_path(path).read_text())
        passport["detectors"]["left_sha256"] = "d" * 64
        common = {
            key: passport[key] for key in ("study_key", "source", "detectors", "preprocessing")
        }
        passport["input_sha256"] = hashlib.sha256(
            json.dumps(common, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        roi.passport_path(path).write_text(json.dumps(passport))
    with pytest.raises(ValueError, match="inconsistent_roi_contract"):
        if entrypoint == "preflight":
            preflight_research(ResearchConfig(1, "binary", cfg))
        else:
            cv.run_sagittal_binary_cv(cfg)
    assert not (tmp_path / "runs").exists()


def test_real_cli_success_outputs_only_safe_aggregate(tmp_path):
    module = inference()
    run, weights, report, cfg = make_saved_run(tmp_path)
    source = write_series(tmp_path / "SyntheticPrivateSource")
    result = subprocess.run(
        [
            sys.executable,
            module.__file__,
            "--dicom-series",
            str(source),
            "--left-model",
            str(weights["left"]),
            "--right-model",
            str(weights["right"]),
            "--run",
            str(run),
            "--output",
            str(tmp_path / "prediction"),
            "--mode",
            "binary",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "status": "complete",
        "mode": "binary",
        "side_count": 2,
        "fold_count": 2,
    }
    assert "SyntheticPrivate" not in result.stdout + result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
    assert "central" not in result.stdout


def test_detector_changed_during_real_load_refuses_before_outputs(tmp_path, monkeypatch):
    module = inference()
    run, weights, report, cfg = make_saved_run(tmp_path)
    from tools import auto_crop_from_detector as crop

    original = crop.load_paired_detector

    def changing_load(path, device):
        model = original(path, device)
        if Path(path) == weights["left"]:
            Path(path).write_bytes(b"SyntheticPrivateChangedWeights")
        return model

    monkeypatch.setattr(crop, "load_paired_detector", changing_load)
    output = tmp_path / "prediction"
    with pytest.raises(module.InferenceError, match="detector_checkpoint_mismatch"):
        module.infer_research(
            tmp_path / "absent-series",
            weights["left"],
            weights["right"],
            run,
            output,
            expected_mode="binary",
        )
    assert not output.exists()


def test_nonfinite_logits_do_not_become_saturated_binary_vote(tmp_path):
    module = inference()
    run, weights, report, cfg = make_saved_run(tmp_path)
    ref = report["folds"][0]["artifacts"]["checkpoint"]
    checkpoint = run / ref["file"]
    payload = torch.load(checkpoint, weights_only=True)
    for name in ("head_sag.0.weight", "head_sag.0.bias", "head_sag.3.weight"):
        payload["model_state_dict"][name].fill_(1e38)
    torch.save(payload, checkpoint)
    ref["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))
    source = write_series(tmp_path / "series")
    output = tmp_path / "prediction"
    with pytest.raises(module.InferenceError, match="invalid_model_output"):
        module.infer_research(
            source, weights["left"], weights["right"], run, output, expected_mode="binary"
        )
    assert not (output / "prediction.private.json").exists()


def test_digest_consistent_small_checkpoint_refuses_before_constructor(tmp_path, monkeypatch):
    from models import tmj_binary_position_classifier as classifier

    run, _, report, _ = make_saved_run(tmp_path)
    ref = report["folds"][0]["artifacts"]["checkpoint"]
    path = run / ref["file"]
    payload = torch.load(path, weights_only=True)
    payload["model_kwargs"]["features"] = [65536]
    torch.save(payload, path)
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))
    calls = []

    def allocation_tripwire(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("allocation tripwire")

    monkeypatch.setattr(classifier, "TMJBinaryPositionClassifier", allocation_tripwire)
    with pytest.raises(inference().InferenceError):
        inference().load_run_ensemble(run, "binary")
    assert not calls


def test_replay_crop_cannot_support_backbone_refuses_before_ensemble(tmp_path, monkeypatch):
    from models import tmj_binary_position_classifier as classifier

    run, _, report, _ = make_saved_run(tmp_path)
    ref = report["artifacts"]["replay"]
    path = run / ref["file"]
    replay = json.loads(path.read_text())
    replay["config"]["features"] = [2, 2, 2, 2]  # crop8 cannot survive four pools.
    path.write_text(json.dumps(replay))
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))
    calls = []

    def allocation_tripwire(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("allocation tripwire")

    monkeypatch.setattr(classifier, "TMJBinaryPositionClassifier", allocation_tripwire)
    with pytest.raises(inference().InferenceError):
        inference().load_run_ensemble(run, "binary")
    assert not calls
