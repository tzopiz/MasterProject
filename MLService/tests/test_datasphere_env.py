"""Tests for training.utils.datasphere_env (no real /home/jupyter required)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from training.utils import datasphere_env as dse


def test_infer_mlservice_root_walks_up(tmp_path):
    mls = tmp_path / "repo" / "MLService"
    (mls / "training").mkdir(parents=True)
    (mls / "training" / "sagittal_binary_cv.py").write_text("#", encoding="utf-8")
    nested = mls / "google_colab" / "nb"
    nested.mkdir(parents=True)
    found = dse.infer_mlservice_root(nested)
    assert found == mls.resolve()


def test_path_kwargs_relative_to_dataset_dir(tmp_path):
    d = tmp_path / "tmj"
    crops = d / "detector_crops_v2"
    (crops / "study_0001").mkdir(parents=True)
    (d / "manifest_private.json").write_text("{}", encoding="utf-8")
    (d / "tmj_position_labels.json").write_text('{"patients":[]}', encoding="utf-8")
    kw = dse.sagittal_binary_cv_path_kwargs(dataset_dir=d)
    assert kw["crop_dir"] == str(crops.resolve())
    assert kw["dataset_root"] == str(crops.resolve())


def test_resolve_ambiguous_crops_refuses(tmp_path):
    d = tmp_path / "tmj"
    (d / "detector_crops_v2" / "study_1").mkdir(parents=True)
    (d / "detector_crops" / "study_2").mkdir(parents=True)
    with pytest.raises(dse.PathResolutionError, match="ambiguous_discovery_path"):
        dse.resolve_detector_crop_dir(d)


def test_manifest_resolves_manifest_json(tmp_path):
    """DataSphere / binary notebook sometimes uses manifest.json (not *_private)."""
    d = tmp_path / "tmj"
    (d / "detector_crops_v2").mkdir(parents=True)
    (d / "manifest.json").write_text('{"studies":[]}', encoding="utf-8")
    (d / "tmj_position_labels.json").write_text('{"patients":[]}', encoding="utf-8")
    assert dse.resolve_manifest_path(d).name == "manifest.json"
    kw = dse.sagittal_binary_cv_path_kwargs(dataset_dir=d)
    assert Path(kw["manifest_path"]).name == "manifest.json"


def test_resolve_crop_uses_filestore_only_when_dataset_unset(monkeypatch, tmp_path):
    """Binary notebook layout: crops only under filestore, no datasets/tmj mount."""
    monkeypatch.setattr(dse, "is_datasphere", lambda: True)
    fs = tmp_path / "filestore"
    monkeypatch.setattr(dse, "FILESTORE", fs)
    crops = fs / "detector_crops_v2"
    (crops / "study_0001").mkdir(parents=True)
    missing = tmp_path / "datasets" / "tmj"
    assert not missing.is_dir()
    monkeypatch.setattr(dse, "DATASPHERE_DATASET_TMJ", missing)
    assert dse.resolve_detector_crop_dir() == crops.resolve()
    with pytest.raises(dse.PathResolutionError, match="invalid_explicit_path"):
        dse.resolve_detector_crop_dir(missing)


def test_resolve_crop_tmj_crop_dir_env(tmp_path, monkeypatch):
    custom = tmp_path / "my_v2"
    custom.mkdir()
    monkeypatch.setenv("TMJ_CROP_DIR", str(custom))
    assert dse.resolve_detector_crop_dir(tmp_path) == custom.resolve()
    with pytest.raises(dse.PathResolutionError, match="invalid_explicit_path"):
        dse.resolve_detector_crop_dir(tmp_path / "nowhere")
    monkeypatch.delenv("TMJ_CROP_DIR", raising=False)


def test_manifest_env_override(tmp_path, monkeypatch):
    d = tmp_path / "tmj"
    (d / "detector_crops_v2").mkdir(parents=True)
    custom = tmp_path / "custom_manifest.json"
    custom.write_text('{"studies":[]}', encoding="utf-8")
    monkeypatch.setenv("TMJ_MANIFEST_PATH", str(custom))
    assert dse.resolve_manifest_path(d) == custom.resolve()
    monkeypatch.delenv("TMJ_MANIFEST_PATH", raising=False)


@pytest.mark.parametrize(
    "name,directory",
    [
        ("TMJ_DATASET_DIR", True),
        ("TMJ_CROP_DIR", True),
        ("TMJ_MANIFEST_PATH", False),
        ("TMJ_LABELS_PATH", False),
        ("ML_SERVICE_ROOT", True),
    ],
)
@pytest.mark.parametrize("value", ["", "SyntheticSensitiveToken"])
def test_bad_explicit_environment_is_not_discovery(name, directory, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(dse.PathResolutionError) as caught:
        dse.validate_explicit_environment()
    assert "SyntheticSensitiveToken" not in str(caught.value)


def test_ambiguous_manifest_and_root_refuse(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "manifest_private.json").write_text("{}")
    with pytest.raises(dse.PathResolutionError, match="ambiguous_discovery_path"):
        dse.resolve_manifest_path(tmp_path)
    for base in (tmp_path, tmp_path / "MLService"):
        (base / "training").mkdir(parents=True)
        (base / "training/sagittal_binary_cv.py").write_text("#")
    with pytest.raises(dse.PathResolutionError, match="ambiguous_discovery_path"):
        dse.infer_mlservice_root(tmp_path)


def test_output_path_helper_does_not_create_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(dse, "is_datasphere", lambda: False)
    assert dse.default_cv_output_json(tmp_path) == tmp_path / "experiments/sagittal_cv_last.json"
    assert not (tmp_path / "experiments").exists()
