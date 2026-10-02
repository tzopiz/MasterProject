"""Shared ROI multiclass protocol; genuine tinyCPU checks establish no quality."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import run_research as runner
from training import sagittal_binary_cv as cv
from training.utils import binary_metrics as metrics

from .test_sagittal_binary_cv import _canonical_synthetic_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _multiclass_config(tmp_path):
    cfg = dataclasses.replace(
        _canonical_synthetic_config(tmp_path),
        mode="multiclass",
        train_augment_mode=None,
        run_id="SYNTHETIC_PRIVATE_RUN",
        output_dir=str(tmp_path / "runs"),
    )
    path = Path(cfg.input_path)
    payload = json.loads(path.read_text())
    for i, row in enumerate(payload["labels"]):
        row["labels"]["sagittal"] = {"left": 1 + i % 3, "right": 1 + i % 3}
    payload["labels"][0]["labels"]["frontal"] = {"left": 6, "right": 4}
    path.write_text(json.dumps(payload))
    return cfg


def test_multiclass_mode_defaults_and_spatial_augmentation_refusal():
    cfg = cv.SagittalBinaryCVConfig(mode="multiclass")
    assert cfg.train_augment_mode == "none"
    assert cv.SagittalBinaryCVConfig().train_augment_mode == "strong"
    for mode in ("flip_only", "strong"):
        with pytest.raises(ValueError):
            cv._validate_cv_config(dataclasses.replace(cfg, train_augment_mode=mode))


def test_multiclass_model_only_predicts_sagittal():
    from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier

    model = TMJBinaryPositionClassifier(features=[2], fc_hidden=4, dropout=0, num_classes=3)
    sagittal, frontal = model(torch.randn(2, 1, 8, 8, 8))
    assert sagittal.shape == (2, 3)
    assert frontal is None
    assert not hasattr(model, "head_fr")


def test_multiclass_metrics_keep_all_class_support():
    assert hasattr(metrics, "multiclass_metrics")
    result = metrics.multiclass_metrics(np.array([0, 0, 1, 1, 2, 2]), np.eye(3)[[0, 1, 1, 1, 0, 2]])
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["support"] == {"0": 2, "1": 2, "2": 2}
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert result["per_class"]["1"]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"]["2"]["recall"] == 0.5
    for bad in ([[1, 1, 1]], [[float("nan"), 0, 1]]):
        with pytest.raises(ValueError):
            metrics.multiclass_metrics(np.array([0]), np.array(bad))


def test_real_multiclass_save_reload_recompute_and_public_analyzer(tmp_path, capsys):
    from models import tmj_binary_position_classifier as models

    assert hasattr(models, "load_position_checkpoint")
    cfg = _multiclass_config(tmp_path)
    report = cv.run_sagittal_binary_cv(cfg)
    directory = Path(cfg.output_dir) / cfg.run_id
    assert report["mode"] == "multiclass"
    assert report["status"] == "complete"
    assert report["calibration_source"] == "not_applicable"
    assert report["model_selection_metric"] == "macro_f1"
    replay = json.loads((directory / "replay.private.json").read_text())
    lookup = {row["sample_key"]: row for row in replay["records"]}
    folds = []
    seen = []
    from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset

    for fold in report["folds"]:
        reference = fold["artifacts"]["checkpoint"]
        checkpoint = directory / reference["file"]
        model, metadata = models.load_position_checkpoint(
            checkpoint, expected_sha256=reference["sha256"], expected_mode="multiclass"
        )
        assert metadata["trained_tasks"] == ["sagittal"]
        assert metadata["num_classes"] == 3
        assert metadata["class_semantics"] == {"0": "central", "1": "anterior", "2": "posterior"}
        assert metadata["threshold"] is None
        with pytest.raises(ValueError):
            models.load_position_checkpoint(checkpoint, expected_mode="binary")
        with pytest.raises(ValueError):
            models.load_binary_position_checkpoint(checkpoint)
        rows = [
            json.loads(line)
            for line in (directory / fold["artifacts"]["predictions"]["file"])
            .read_text()
            .splitlines()
        ]
        reconstructed = cv.recompute_fold_metrics(rows, mode="multiclass")
        folds.append(reconstructed)
        assert reconstructed == {key: fold[key] for key in reconstructed}
        assert set(fold["val_support"]) == {"0", "1", "2"}
        assert all(fold["val_support"].values())
        assert {row["group_key"] for row in rows if row["role"] == "training"}.isdisjoint(
            {row["group_key"] for row in rows if row["role"] == "validation"}
        )
        for row in rows:
            tensor, label = TMJBinaryPositionDataset(
                [lookup[row["sample_key"]]], sagittal_only=True, num_classes=3
            )[0]
            with torch.no_grad():
                logits, frontal = model(tensor.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)[0].tolist()
            assert frontal is None
            assert logits[0].tolist() == pytest.approx(row["logits"], abs=1e-6)
            assert probabilities == pytest.approx(row["probabilities"], abs=1e-6)
            assert row["decision"] == int(np.argmax(probabilities))
            assert label.item() == row["label"]
            if row["role"] == "validation":
                seen.append(row["sample_key"])
    assert len(seen) == len(set(seen)) == 16
    assert cv._build_cv_report_dict(cfg, folds)["summary"] == report["summary"]
    from training.utils.seed import set_seed

    set_seed(cfg.seed)
    initial = models.TMJBinaryPositionClassifier(
        features=list(cfg.features), fc_hidden=cfg.fc_hidden, dropout=cfg.dropout, num_classes=3
    )
    state = torch.load(
        directory / report["folds"][0]["artifacts"]["checkpoint"]["file"], weights_only=True
    )["model_state_dict"]
    assert not torch.equal(initial.backbone[0][0].weight, state["backbone.0.0.weight"])
    analyzed = cv.analyze_sagittal_cv_result(report, show_plots=False, save_curves=False)
    assert analyzed["summary"] == report["summary"]
    assert analyzed["fold_summaries"][0]["val_per_class"] == report["folds"][0]["val_per_class"]
    public = (directory / "report.json").read_text() + capsys.readouterr().out
    assert str(tmp_path) not in public
    assert "SYNTHETIC_PRIVATE_RUN" not in public
    assert '"patient_id"' not in public


def _write_research_config(cfg, path, **changes):
    values = dataclasses.asdict(cfg)
    payload = {key: values[key] for key in runner.CV_CONFIG_FIELDS}
    payload.update(schema_version=1, mode=cfg.mode, **changes)
    path.write_text(json.dumps(payload))
    return payload


def test_multiclass_runner_default_none_and_allclass_preflight(tmp_path):
    cfg = _multiclass_config(tmp_path)
    path = tmp_path / "research.private.json"
    payload = _write_research_config(cfg, path)
    del payload["train_augment_mode"]
    path.write_text(json.dumps(payload))
    config = runner.load_research_config(path)
    assert config.mode == config.cv.mode == "multiclass"
    assert config.cv.train_augment_mode == "none"
    report = runner.preflight_research(config)
    assert report["mode"] == "multiclass"
    assert report["class_support"] == {"0": 6, "1": 6, "2": 4}
    assert not Path(cfg.output_dir).exists()
    for bad in ("strong", "flip_only"):
        payload["train_augment_mode"] = bad
        path.write_text(json.dumps(payload))
        with pytest.raises(runner.ResearchConfigError):
            runner.load_research_config(path)


def test_missing_multiclass_group_support_fails_before_model(tmp_path, monkeypatch):
    cfg = _multiclass_config(tmp_path)
    path = tmp_path / "research.private.json"
    _write_research_config(cfg, path)
    config = runner.load_research_config(path)
    index = json.loads(Path(cfg.input_path).read_text())
    for row in index["labels"]:
        if row["labels"]["sagittal"]["left"] == 3:
            row["labels"]["sagittal"] = {"left": 2, "right": 2}
    Path(cfg.input_path).write_text(json.dumps(index))
    from models import tmj_binary_position_classifier as models

    monkeypatch.setattr(
        models,
        "TMJBinaryPositionClassifier",
        lambda *args, **kwargs: pytest.fail("unsuitable preflight constructed model"),
    )
    with pytest.raises(runner.ResearchConfigError, match="unsuitable_cv_groups"):
        runner.run_research(config)
    assert not Path(cfg.output_dir).exists()


def test_generic_loader_multiclass_metadata_refusal(tmp_path):
    from models import tmj_binary_position_classifier as models

    cfg = _multiclass_config(tmp_path)
    report = cv.run_sagittal_binary_cv(cfg)
    checkpoint = (
        Path(cfg.output_dir) / cfg.run_id / report["folds"][0]["artifacts"]["checkpoint"]["file"]
    )
    payload = torch.load(checkpoint, weights_only=True)
    wrong = tmp_path / "SyntheticSensitiveToken.pth"
    for field, value in (
        ("mode", "binary"),
        ("num_classes", 2),
        ("num_classes", True),
        ("threshold", 0.5),
        ("decision_rule", ">="),
        ("class_semantics", {"0": "SyntheticSensitiveToken"}),
        ("trained_tasks", ["frontal"]),
        ("model_kwargs", {**payload["model_kwargs"], "num_classes": 2}),
        ("preprocessing", {}),
    ):
        torch.save({**payload, field: value}, wrong)
        with pytest.raises(ValueError, match="invalid_position_checkpoint") as caught:
            models.load_position_checkpoint(wrong)
        assert "SyntheticSensitiveToken" not in str(caught.value)
    with pytest.raises(ValueError):
        models.load_position_checkpoint(checkpoint, expected_sha256="0" * 64)
    wrong.write_bytes(b"SyntheticSensitiveToken")
    with pytest.raises(ValueError, match="invalid_position_checkpoint"):
        models.load_position_checkpoint(wrong)


def test_generic_loader_accepts_existing_binary_contract(tmp_path):
    from models import tmj_binary_position_classifier as models

    cfg = _canonical_synthetic_config(tmp_path)
    cfg.output_dir = str(tmp_path / "runs")
    cfg.run_id = "binary"
    report = cv.run_sagittal_binary_cv(cfg)
    checkpoint = (
        Path(cfg.output_dir) / cfg.run_id / report["folds"][0]["artifacts"]["checkpoint"]["file"]
    )
    model, metadata = models.load_position_checkpoint(checkpoint, expected_mode="binary")
    assert metadata["mode"] == "binary"
    assert metadata["num_classes"] == 2
    assert model(torch.ones(1, 1, 8, 8, 8))[0].shape == (1, 1)
    with pytest.raises(ValueError):
        models.load_position_checkpoint(checkpoint, expected_mode="multiclass")


def test_macro_f1_exact_ties_select_earliest_real_epoch(tmp_path):
    cfg = _multiclass_config(tmp_path)
    cfg.epochs = 3
    cfg.lr = 1e-30
    report = cv.run_sagittal_binary_cv(cfg)
    for fold in report["folds"]:
        scores = [row["val_macro_f1"] for row in fold["epoch_history"]]
        assert len(scores) == 3 and len(set(scores)) == 1
        assert fold["best_epoch"] == 1


def test_multiclass_analyzer_private_extras_are_not_exported(tmp_path, capsys):
    cfg = cv.SagittalBinaryCVConfig(mode="multiclass")
    result = cv._build_cv_report_dict(cfg, [])
    token = "SyntheticSensitiveToken"
    result["config"]["private_config"] = {"path": token}
    result["summary"][token] = token
    result["folds"] = [
        {
            "fold": 0,
            "val_macro_f1": 0.25,
            "val_per_class": {
                "0": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2, "private": token}
            },
            "epoch_history": [
                {"epoch": 1, "train_loss": 1.0, "val_macro_f1": 0.25, "private": token}
            ],
            "private": token,
        }
    ]
    output = tmp_path / "aggregate"
    analyzed = cv.analyze_sagittal_cv_result(
        result, show_plots=False, save_curves=False, report_path=output
    )
    assert analyzed["mode"] == "multiclass"
    assert token not in capsys.readouterr().out
    for artifact in tmp_path.glob("aggregate*"):
        assert token not in artifact.read_text()
    result["folds"][0]["val_per_class"]["0"]["precision"] = token
    with pytest.raises(ValueError, match="invalid_cv_report"):
        cv.analyze_sagittal_cv_result(result, show_plots=False)
    assert token not in capsys.readouterr().out


def test_multiclass_cli_and_notebook_use_same_config(tmp_path, monkeypatch, capsys):
    import os
    import subprocess
    import sys

    cfg = _multiclass_config(tmp_path)
    path = tmp_path / "research.private.json"
    _write_research_config(cfg, path)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        [sys.executable, str(Path(runner.__file__)), "--config", str(path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "multiclass" and report["status"] == "complete"
    assert "val_macro_f1" in report["folds"][0]
    assert "threshold_from_train_youden" not in report["folds"][0]
    assert str(tmp_path) not in result.stdout + result.stderr
    notebook = json.loads(
        (
            Path(runner.__file__).parents[1] / "google_colab/train_sagittal_binary_cv.ipynb"
        ).read_text()
    )
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    calls = []

    def captured(config):
        calls.append(config)
        return report

    monkeypatch.setattr(runner, "run_research", captured)
    namespace = {}
    exec(
        cells[0].replace('CONFIG_PATH = "research.private.json"', f"CONFIG_PATH = {str(path)!r}"),
        namespace,
    )
    exec(cells[1], namespace)
    exec(cells[2], namespace)
    assert calls == [runner.load_research_config(path)]
    assert str(tmp_path) not in capsys.readouterr().out
