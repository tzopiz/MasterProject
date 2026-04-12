"""Tests for training.utils.datasphere_env (no real /home/jupyter required)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def test_resolve_prefers_v2(tmp_path):
    d = tmp_path / "tmj"
    (d / "detector_crops_v2" / "study_1").mkdir(parents=True)
    (d / "detector_crops" / "study_2").mkdir(parents=True)
    assert dse.resolve_detector_crop_dir(d).name == "detector_crops_v2"


def test_manifest_resolves_manifest_json(tmp_path):
    """DataSphere / binary notebook sometimes uses manifest.json (not *_private)."""
    d = tmp_path / "tmj"
    (d / "detector_crops_v2").mkdir(parents=True)
    (d / "manifest.json").write_text('{"studies":[]}', encoding="utf-8")
    (d / "tmj_position_labels.json").write_text('{"patients":[]}', encoding="utf-8")
    assert dse.resolve_manifest_path(d).name == "manifest.json"
    kw = dse.sagittal_binary_cv_path_kwargs(dataset_dir=d)
    assert Path(kw["manifest_path"]).name == "manifest.json"


def test_resolve_crop_prefers_filestore_when_dataset_dir_missing(monkeypatch, tmp_path):
    """Binary notebook layout: crops only under filestore, no datasets/tmj mount."""
    monkeypatch.setattr(dse, "is_datasphere", lambda: True)
    fs = tmp_path / "filestore"
    monkeypatch.setattr(dse, "FILESTORE", fs)
    crops = fs / "detector_crops_v2"
    (crops / "study_0001").mkdir(parents=True)
    missing = tmp_path / "datasets" / "tmj"
    assert not missing.is_dir()
    assert dse.resolve_detector_crop_dir(missing) == crops.resolve()


def test_resolve_crop_tmj_crop_dir_env(tmp_path, monkeypatch):
    custom = tmp_path / "my_v2"
    custom.mkdir()
    monkeypatch.setenv("TMJ_CROP_DIR", str(custom))
    assert dse.resolve_detector_crop_dir(tmp_path / "nowhere") == custom.resolve()
    monkeypatch.delenv("TMJ_CROP_DIR", raising=False)


def test_manifest_env_override(tmp_path, monkeypatch):
    d = tmp_path / "tmj"
    (d / "detector_crops_v2").mkdir(parents=True)
    custom = tmp_path / "custom_manifest.json"
    custom.write_text('{"studies":[]}', encoding="utf-8")
    monkeypatch.setenv("TMJ_MANIFEST_PATH", str(custom))
    assert dse.resolve_manifest_path(d) == custom.resolve()
    monkeypatch.delenv("TMJ_MANIFEST_PATH", raising=False)
