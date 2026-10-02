"""Focused CPU regressions for CV correctness, not trained quality."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier
from training import sagittal_binary_cv as cv


@pytest.mark.parametrize(
    "changes",
    [
        {"epochs": 0},
        {"n_splits": 1},
        {"batch_size": 0},
        {"num_workers": -1},
        {"features": ()},
        {"features": 2},
        {"legacy_name_join": "yes"},
        {"dropout": 1},
        {"lr": float("nan")},
        {"lr_plateau_factor": 1},
    ],
)
def test_bad_config_fails_before_loading_paths(changes):
    cfg = cv.SagittalBinaryCVConfig(**changes)
    with pytest.raises(ValueError):
        cv.run_sagittal_binary_cv(cfg)


def _records():
    return [
        {
            "patient_id": str(i),
            "source_id": "synthetic",
            "patient_name": str(i),
            "sag": i % 2,
            "side": "left",
        }
        for i in range(8)
    ]


def test_fold_validation_rejects_empty_and_single_class():
    cfg = cv.SagittalBinaryCVConfig(n_splits=2)
    for records in ([], [{**r, "sag": 0} for r in _records()]):
        with pytest.raises(ValueError):
            cv._validated_cv_folds(records, cfg)


def test_fold_validation_rejects_group_overlap(monkeypatch):
    from training import tmj_position_label_table as table

    monkeypatch.setattr(
        table,
        "iter_stratified_group_kfold_indices",
        lambda *a, **k: iter([(np.arange(8), np.arange(8))] * 2),
    )
    with pytest.raises(ValueError, match="overlap"):
        cv._validated_cv_folds(_records(), cv.SagittalBinaryCVConfig(n_splits=2))


def test_calibration_loader_is_unaugmented_ordered_and_repeatable(tmp_path):
    records = []
    for i in range(4):
        path = tmp_path / f"crop{i}.nii.gz"
        nib.save(
            nib.Nifti1Image(np.arange(512, dtype=np.float32).reshape(8, 8, 8) + i, np.eye(4)), path
        )
        records.append({"crop_path": str(path), "sag": i % 2})
    cfg = cv.SagittalBinaryCVConfig(batch_size=2, features=(2,), fc_hidden=4)
    loader = cv._make_calibration_loader(records, cfg)
    assert not loader.dataset.is_train
    assert isinstance(loader.sampler, SequentialSampler)
    model = TMJBinaryPositionClassifier(features=[2], fc_hidden=4)
    first = cv._collect_sag_logits_labels(model, loader, torch.device("cpu"))
    second = cv._collect_sag_logits_labels(model, loader, torch.device("cpu"))
    assert torch.equal(first[0], second[0])
    assert first[1].tolist() == [0, 1, 0, 1]
    assert torch.equal(first[1], second[1])


def test_report_declares_development_and_strict_json(tmp_path):
    cfg = cv.SagittalBinaryCVConfig(n_splits=2)
    result = cv._build_cv_report_dict(cfg, [])
    assert result["assessment"] == "development_cv"
    assert result["model_selection_source"] == "validation"
    assert result["calibration_source"] == "training_unaugmented"
    path = tmp_path / "report.json"
    cv._write_cv_report_json_atomic(str(path), result)
    result = json.loads(path.read_text(), parse_constant=lambda s: pytest.fail(s))
    assert result["status"] == "in_progress"
    assert result["summary"]["mean_val_auc"] is None


@pytest.fixture()
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fold_baseline_uses_train_majority(small_cpu_threads):
    from training.losses.focal_loss import BinaryFocalLoss

    volumes = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 1, 8, 8, 8) / 2048
    train = DataLoader(TensorDataset(volumes, torch.tensor([0.0, 0.0, 0.0, 1.0])), batch_size=2)
    val = DataLoader(TensorDataset(volumes, torch.tensor([1.0, 1.0, 1.0, 0.0])), batch_size=2)
    model = TMJBinaryPositionClassifier(features=[2], fc_hidden=4, dropout=0)
    cfg = cv.SagittalBinaryCVConfig(epochs=1, features=(2,), fc_hidden=4, tqdm_disable=True)
    result = cv._train_one_fold(
        model, train, val, BinaryFocalLoss(), torch.device("cpu"), cfg, calibration_loader=train
    )
    baseline = result["train_majority_baseline"]
    assert baseline["source"] == "training"
    assert baseline["predicted_class"] == 0
    assert baseline["confusion_matrix"] == [[1, 0], [3, 0]]
    assert baseline["accuracy"] == 0.25
    assert baseline["balanced_accuracy"] == 0.5
    assert result["train_support"] == {"0": 3, "1": 1}
    assert result["val_support"] == {"0": 1, "1": 3}


def _legacy_synthetic_config(tmp_path):
    crops = tmp_path / "crops"
    studies, patients = [], []
    for i in range(8):
        study = f"study_{i}"
        (crops / study).mkdir(parents=True)
        for side in ("left", "right"):
            nib.save(
                nib.Nifti1Image(np.arange(512, dtype=np.float32).reshape(8, 8, 8), np.eye(4)),
                crops / study / f"{study}_{side}.nii.gz",
            )
        studies.append({"study_id": study, "patient_name": f"synthetic_{i}"})
        patients.append(
            {
                "name_raw": f"synthetic_{i}",
                "labels": {
                    "sagittal": {"left": 1 + i % 2, "right": 1 + i % 2},
                    "frontal": {"left": 4, "right": 4},
                },
            }
        )
    manifest, labels = tmp_path / "manifest.json", tmp_path / "labels.json"
    manifest.write_text(json.dumps({"studies": studies}))
    labels.write_text(json.dumps({"patients": patients}))
    return cv.SagittalBinaryCVConfig(
        legacy_name_join=True,
        output_dir=str(tmp_path / "runs"),
        crop_dir=str(crops),
        dataset_root=str(crops),
        manifest_path=str(manifest),
        labels_path=str(labels),
        n_splits=2,
        epochs=1,
        batch_size=4,
        features=(2,),
        fc_hidden=4,
        dropout=0,
        device="cpu",
        tqdm_disable=True,
        train_augment_mode="none",
    )


def test_fold_initialization_does_not_depend_on_previous_epochs(
    tmp_path, monkeypatch, small_cpu_threads
):
    from models import tmj_binary_position_classifier as models

    initializations = []
    real_model = models.TMJBinaryPositionClassifier

    def capture_model(*args, **kwargs):
        model = real_model(*args, **kwargs)
        initializations.append(model.backbone[0][0].weight.detach().clone())
        return model

    monkeypatch.setattr(models, "TMJBinaryPositionClassifier", capture_model)
    cfg = _legacy_synthetic_config(tmp_path)
    cv.run_sagittal_binary_cv(cfg)
    cfg.epochs = 2
    cv.run_sagittal_binary_cv(cfg)
    assert torch.equal(initializations[0], initializations[2])
    assert torch.equal(initializations[1], initializations[3])


def test_canonical_input_required_by_default():
    with pytest.raises(ValueError, match="input_path"):
        cv.run_sagittal_binary_cv(cv.SagittalBinaryCVConfig())


def _canonical_synthetic_config(tmp_path):
    cfg = _legacy_synthetic_config(tmp_path)
    from training import roi_provenance as roi

    studies, labels = [], []
    detectors = {"family": roi.DETECTOR_FAMILY, "left_sha256": "a" * 64, "right_sha256": "b" * 64}
    source = {"sha256": "c" * 64, "geometry": roi.voxel_geometry((8, 8, 8))}
    for i in range(8):
        study = f"study_{i}"
        shared = {"source_id": "synthetic", "patient_id": f"p{i}", "label_record_id": f"l{i}"}
        studies.append(
            {
                **shared,
                "study_id": study,
                "label_applicability": "confirmed",
                "crops": {side: f"{study}/{study}_{side}.nii.gz" for side in ("left", "right")},
            }
        )
        for side in ("left", "right"):
            roi.write_roi_crop(
                np.random.default_rng(i).normal(size=(8, 8, 8)).astype(np.float32),
                (4, 4, 4),
                8,
                Path(cfg.dataset_root) / studies[-1]["crops"][side],
                side=side,
                record=studies[-1],
                source=source,
                detectors=detectors,
            )
        labels.append({**shared, "labels": {"sagittal": {"left": 1 + i % 2, "right": 1 + i % 2}}})
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps({"schema_version": 1, "studies": studies, "labels": labels}))
    cfg.input_path, cfg.legacy_name_join = str(path), False
    return cfg


def test_canonical_sagittal_without_frontal_runs(tmp_path, small_cpu_threads):
    cfg = _canonical_synthetic_config(tmp_path)
    path = Path(cfg.input_path)
    index = json.loads(path.read_text())
    studies, labels = index["studies"], index["labels"]
    result = cv.run_sagittal_binary_cv(cfg)
    assert result["completed_folds"] == 2
    assert result["assessment"] == "development_cv"
    assert all(fold["val_support"][label] > 0 for fold in result["folds"] for label in ("0", "1"))
    assert {
        label: sum(fold["val_support"][label] for fold in result["folds"]) for label in ("0", "1")
    } == {"0": 8, "1": 8}
    sentinel = tmp_path / "preserve.json"
    sentinel.write_text("preserve")
    cfg.output_json = str(sentinel)
    for label in labels:
        label["labels"]["sagittal"] = {"left": 1, "right": 1}
    path.write_text(json.dumps({"schema_version": 1, "studies": studies, "labels": labels}))
    with pytest.raises(ValueError, match="class"):
        cv.run_sagittal_binary_cv(cfg)
    assert sentinel.read_text() == "preserve"
    assert not (tmp_path / "preserve_epochs.jsonl").exists()


def test_analyzer_does_not_print_private_paths(capsys):
    result = cv._build_cv_report_dict(
        cv.SagittalBinaryCVConfig(
            crop_dir="PRIVATE_INPUT_TOKEN", output_json="PRIVATE_OUTPUT_TOKEN"
        ),
        [],
    )
    cv.analyze_sagittal_cv_result(result, show_plots=False)
    output = capsys.readouterr().out
    assert "PRIVATE_INPUT_TOKEN" not in output
    assert "PRIVATE_OUTPUT_TOKEN" not in output


def test_cli_accepts_canonical_input_without_echoing_bad_path(tmp_path):
    import subprocess

    bad_path = tmp_path / "PRIVATE_INPUT_TOKEN.json"
    command = [
        sys.executable,
        str(Path(cv.__file__)),
        "--input-path",
        str(bad_path),
        "--dataset-root",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["ready"] is False
    assert report["stage"] == "intake"
    assert report["diagnostics"][0]["code"] == "unreadable_input"
    assert not completed.stderr
    assert "PRIVATE_INPUT_TOKEN" not in completed.stdout + completed.stderr
    assert str(tmp_path) not in completed.stdout + completed.stderr
    assert "Traceback" not in completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "flag", ["--lr", "--features", "--device", "--seed", "--train-augment-mode"]
)
def test_cli_argument_errors_do_not_echo_values(flag):
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(Path(cv.__file__)), f"{flag}=SyntheticSensitiveToken"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "SyntheticSensitiveToken" not in completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["code"] == "invalid_arguments"


def test_cli_help_is_preserved():
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(Path(cv.__file__)), "--help"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert "--input-path" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_analyzer_sanitizes_historical_private_payload(tmp_path, capsys):
    token = "SyntheticSensitiveToken"
    result = {
        token: token,
        "config": {
            "epochs": 2,
            "lr": 0.01,
            "features": [2],
            "crop_dir": token,
            "manifest_path": token,
            "labels_path": token,
            "dataset_root": token,
            "patient_id": token,
            "train_augment_mode": token,
        },
        "summary": {"mean_val_auc": 0.75, "private_mapping": token},
        "status": "complete",
        "n_splits": 1,
        "completed_folds": 1,
        "best_fold_auc": 0.75,
        "worst_fold_auc": 0.75,
        "folds": [
            {
                "fold": 0,
                "val_auc": 0.75,
                "val_f1_minority": 0.5,
                "private_id": token,
                "train_majority_baseline": {
                    "accuracy": 0.5,
                    "source": "training",
                    "private_path": token,
                },
                "epoch_history": [
                    {
                        "epoch": 1,
                        "train_loss": 0.25,
                        "val_auc": 0.75,
                        "lr": 0.01,
                        "private_mapping": token,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "historical.private.json"
    path.write_text(json.dumps(result))
    returned = cv.analyze_sagittal_cv_result(
        json_path=path, report_path=tmp_path / "public.txt", show_plots=False, save_curves=False
    )
    assert token not in capsys.readouterr().out
    assert token not in json.dumps({k: v for k, v in returned.items() if not k.endswith("_df")})
    assert returned["config"] == {"epochs": 2, "lr": 0.01, "features": [2]}
    assert returned["fold_summaries"][0]["val_f1_positive"] == 0.5
    for output in tmp_path.glob("public*"):
        assert token not in output.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("config", {"epochs": "SyntheticSensitiveToken", "features": ["SyntheticSensitiveToken"]}),
        ("summary", {"mean_val_auc": "SyntheticSensitiveToken"}),
        ("status", "SyntheticSensitiveToken"),
        ("assessment", "SyntheticSensitiveToken"),
        ("calibration_source", "SyntheticSensitiveToken"),
        ("model_selection_source", "SyntheticSensitiveToken"),
        ("folds", [{"fold": "SyntheticSensitiveToken"}]),
        ("folds", [{"epoch_history": [{"epoch": 1, "train_loss": "SyntheticSensitiveToken"}]}]),
    ],
)
def test_analyzer_untrusted_known_values_never_reach_outputs(tmp_path, capsys, field, value):
    result = {"summary": {}, "folds": [], field: value}
    try:
        returned = cv.analyze_sagittal_cv_result(
            result, report_path=tmp_path / "public.txt", show_plots=False
        )
    except ValueError as error:
        assert str(error) == "invalid_cv_report"
    else:
        assert "SyntheticSensitiveToken" not in json.dumps(
            {k: v for k, v in returned.items() if not k.endswith("_df")}
        )
    assert "SyntheticSensitiveToken" not in capsys.readouterr().out
    for path in tmp_path.glob("public*"):
        assert "SyntheticSensitiveToken" not in path.read_text()


@pytest.mark.parametrize(
    "error,code,exitcode",
    [
        (RuntimeError("SyntheticSensitiveToken/private/path"), "research_run_failed", 2),
        (KeyboardInterrupt(), "research_run_interrupted", 130),
        (RuntimeError("cv_run_interrupted"), "research_run_interrupted", 130),
    ],
)
def test_direct_cli_runtime_errors_are_safe_json(monkeypatch, capsys, error, code, exitcode):
    monkeypatch.setattr(sys, "argv", [str(Path(cv.__file__)), "--device", "cpu"])

    def fail_run(config):
        raise error

    monkeypatch.setattr(cv, "run_sagittal_binary_cv", fail_run)
    assert cv.main() == exitcode
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"schema_version": 1, "ready": False, "code": code}
    assert not captured.err
    assert "SyntheticSensitiveToken" not in captured.out
    assert "Traceback" not in captured.out
