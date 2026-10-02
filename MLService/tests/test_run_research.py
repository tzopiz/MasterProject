"""Versioned configuration and model-free readiness, with synthetic private data."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_research as runner

from .test_sagittal_binary_cv import _canonical_synthetic_config


@pytest.fixture()
def config_file(tmp_path):
    cfg = _canonical_synthetic_config(tmp_path)
    values = dataclasses.asdict(cfg)
    allowed = set(runner.CV_CONFIG_FIELDS)
    payload = {key: value for key, value in values.items() if key in allowed}
    payload.update(
        schema_version=1,
        mode="binary",
        input_path="canonical.json",
        dataset_root="crops",
        output_dir="outputs",
    )
    path = tmp_path / "research.private.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_config_relative_paths_and_model_free_preflight(config_file, monkeypatch, tmp_path):
    path, _ = config_file
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    config = runner.load_research_config(path)
    assert config.cv.input_path == str(path.parent / "canonical.json")
    assert config.cv.dataset_root == str(path.parent / "crops")
    assert config.cv.output_dir == str(path.parent / "outputs")
    import models.tmj_binary_position_classifier as model

    monkeypatch.setattr(
        model,
        "TMJBinaryPositionClassifier",
        lambda *a, **k: pytest.fail("preflight constructed model"),
    )
    report = runner.preflight_research(config)
    assert report["ready"]
    assert report["patient_count"] == 8
    assert report["fold_count"] == 2
    assert not Path(config.cv.output_dir).exists()
    assert str(path.parent) not in json.dumps(report)
    assert "synthetic" not in json.dumps(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("mode", "unknown"),
        ("unknown", "SyntheticSensitiveToken"),
        ("epochs", True),
        ("features", ["SyntheticSensitiveToken"]),
        ("num_workers", -1),
        ("tqdm_disable", "yes"),
        ("log_epochs_jsonl", 1),
        ("device", "SyntheticSensitiveToken"),
        ("input_path", "SyntheticSensitiveToken"),
        ("output_dir", "crops/results"),
        ("output_dir", "."),
        ("dataset_root", "SyntheticSensitiveToken"),
    ],
)
def test_config_refuses_bad_contract_safely(config_file, field, value):
    path, payload = config_file
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(runner.ResearchConfigError) as caught:
        runner.load_research_config(path)
    assert "SyntheticSensitiveToken" not in str(caught.value)


def test_duplicate_and_missing_fields_refused(config_file):
    path, payload = config_file
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(runner.ResearchConfigError):
        runner.load_research_config(path)
    del payload["output_dir"]
    path.write_text(json.dumps(payload))
    with pytest.raises(runner.ResearchConfigError):
        runner.load_research_config(path)


def test_fold_unsuitability_fails_preflight_without_output(config_file):
    path, _ = config_file
    config = runner.load_research_config(path)
    index_path = Path(config.cv.input_path)
    index = json.loads(index_path.read_text())
    for label in index["labels"]:
        label["labels"]["sagittal"] = {"left": 1, "right": 1}
    index_path.write_text(json.dumps(index))
    with pytest.raises(runner.ResearchConfigError, match="unsuitable_cv_groups"):
        runner.preflight_research(config)
    assert not Path(config.cv.output_dir).exists()


def test_cli_preflight_parity_and_argument_privacy(config_file):
    path, _ = config_file
    command = [sys.executable, str(Path(runner.__file__)), "--config", str(path), "--preflight"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == runner.preflight_research(runner.load_research_config(path))
    assert str(path.parent) not in result.stdout + result.stderr
    invalid = subprocess.run(
        [sys.executable, str(Path(runner.__file__)), "--SyntheticSensitiveToken"],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert "SyntheticSensitiveToken" not in invalid.stdout + invalid.stderr


def test_invalid_explicit_environment_never_falls_back(config_file, monkeypatch):
    path, _ = config_file
    for key in (
        "TMJ_DATASET_DIR",
        "TMJ_CROP_DIR",
        "TMJ_MANIFEST_PATH",
        "TMJ_LABELS_PATH",
        "ML_SERVICE_ROOT",
    ):
        with monkeypatch.context() as context:
            context.setenv(key, "SyntheticSensitiveToken")
            with pytest.raises(runner.ResearchConfigError):
                runner.load_research_config(path)


def test_read_only_inputs_and_separate_output(config_file):
    path, _ = config_file
    root = path.parent / "crops"
    root.chmod(0o500)
    try:
        config = runner.load_research_config(path)
        assert runner.preflight_research(config)["ready"]
        assert not Path(config.cv.output_dir).exists()
    finally:
        root.chmod(0o700)


def test_thin_notebook_uses_same_config_preflight_and_run(config_file, monkeypatch, capsys):
    path, _ = config_file
    notebook = json.loads(
        (
            Path(runner.__file__).parents[1] / "google_colab/train_sagittal_binary_cv.ipynb"
        ).read_text()
    )
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    calls = []

    def captured_run(config):
        calls.append(config)
        return {
            "status": "complete",
            "assessment": "development_cv",
            "completed_folds": 2,
            "summary": {},
        }

    monkeypatch.setattr(runner, "run_research", captured_run)
    namespace = {}
    exec(
        cells[0].replace('CONFIG_PATH = "research.private.json"', f"CONFIG_PATH = {str(path)!r}"),
        namespace,
    )
    exec(cells[1], namespace)
    assert namespace["readiness"] == runner.preflight_research(runner.load_research_config(path))
    exec(cells[2], namespace)
    assert calls == [runner.load_research_config(path)]
    printed = capsys.readouterr().out
    assert str(path.parent) not in printed
    assert "synthetic" not in printed
    assert all(not cell.get("outputs") for cell in notebook["cells"])


def test_cli_actual_tiny_cpu_run(config_file):
    path, _ = config_file
    import os

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
    assert report["status"] == "complete"
    assert report["completed_folds"] == 2
    assert str(path.parent) not in result.stdout + result.stderr
    runs = list((path.parent / "outputs").iterdir())
    assert len(runs) == 1
    assert json.loads((runs[0] / "report.json").read_text())["summary"] == report["summary"]
    assert len(list(runs[0].glob("*_checkpoint.pth"))) == 2
    assert (runs[0] / "replay.private.json").is_file()


def test_explicit_bad_legacy_cv_paths_do_not_reconcile(tmp_path, monkeypatch):
    from training import sagittal_binary_cv as cv
    from training.utils import datasphere_env as dse

    cfg = cv.SagittalBinaryCVConfig(
        legacy_name_join=True, crop_dir=str(tmp_path / "SyntheticSensitiveToken")
    )
    monkeypatch.setattr(
        dse,
        "resolve_detector_crop_dir",
        lambda *a, **k: pytest.fail("explicit path attempted discovery"),
    )
    with pytest.raises(ValueError, match="invalid_explicit_path"):
        cv._validate_legacy_cv_paths(cfg)
    assert cfg.crop_dir == str(tmp_path / "SyntheticSensitiveToken")


def test_config_output_symlink_cannot_enter_read_only_input(config_file):
    path, payload = config_file
    (path.parent / "output-link").symlink_to(path.parent / "crops", target_is_directory=True)
    payload["output_dir"] = "output-link/runs"
    path.write_text(json.dumps(payload))
    with pytest.raises(runner.ResearchConfigError, match="output_input_overlap"):
        runner.load_research_config(path)


def test_valid_legacy_environment_does_not_override_canonical_config(config_file, monkeypatch):
    path, _ = config_file
    alternate = path.parent / "alternate"
    alternate.mkdir()
    monkeypatch.setenv("TMJ_DATASET_DIR", str(alternate))
    monkeypatch.setenv("TMJ_CROP_DIR", str(alternate))
    config = runner.load_research_config(path)
    assert config.cv.dataset_root == str(path.parent / "crops")
    assert runner.preflight_research(config)["ready"]


def test_preflight_rechecks_changed_environment(config_file, monkeypatch):
    path, _ = config_file
    config = runner.load_research_config(path)
    monkeypatch.setenv("ML_SERVICE_ROOT", "SyntheticSensitiveToken")
    with pytest.raises(runner.ResearchConfigError, match="invalid_explicit_path"):
        runner.preflight_research(config)


def test_notebook_invalid_explicit_root_refuses_without_value(config_file, monkeypatch):
    notebook = json.loads(
        (
            Path(runner.__file__).parents[1] / "google_colab/train_sagittal_binary_cv.ipynb"
        ).read_text()
    )
    setup = "".join(
        next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    )
    monkeypatch.setenv("ML_SERVICE_ROOT", "SyntheticSensitiveToken")
    with pytest.raises(ValueError, match="research_module_unavailable") as caught:
        exec(setup, {})
    assert "SyntheticSensitiveToken" not in str(caught.value)


def test_cli_malformed_private_json_has_only_fixed_diagnostic(config_file):
    path, _ = config_file
    path.write_text('{"SyntheticSensitiveToken":')
    result = subprocess.run(
        [sys.executable, str(Path(runner.__file__)), "--config", str(path), "--preflight"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "ready": False,
        "code": "invalid_config",
    }
    assert str(path.parent) not in result.stdout + result.stderr
    assert "SyntheticSensitiveToken" not in result.stdout + result.stderr
