"""Reject metadata-driven allocation before constructing research models."""

import dataclasses

import pytest
import torch

from models import tmj_binary_position_classifier as classifier
from tools import auto_crop_from_detector as crop
from training import sagittal_binary_cv as cv


def classifier_payload():
    model = classifier.TMJBinaryPositionClassifier(features=[2], fc_hidden=4)
    return {
        "schema_version": 1,
        "family": classifier.BINARY_CHECKPOINT_FAMILY,
        "model_kwargs": {"in_channels": 1, "features": [2], "fc_hidden": 4, "dropout": 0.5},
        "trained_tasks": ["sagittal"],
        "preprocessing": classifier.BINARY_PREPROCESSING,
        "class_semantics": classifier.BINARY_CLASS_SEMANTICS,
        "threshold": 0.5,
        "decision_rule": ">=",
        "model_state_dict": model.state_dict(),
        "fold": 0,
        "best_epoch": 1,
    }


def constructor_tripwire(monkeypatch, module, name):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("allocation tripwire")

    monkeypatch.setattr(module, name, forbidden)
    return calls


@pytest.mark.parametrize(
    "field,value", [("features", [65536]), ("features", [2] * 5), ("fc_hidden", 2**32)]
)
def test_classifier_checkpoint_refuses_before_constructor(tmp_path, monkeypatch, field, value):
    payload = classifier_payload()
    payload["model_kwargs"][field] = value
    path = tmp_path / "invalid.pth"
    torch.save(payload, path)
    calls = constructor_tripwire(monkeypatch, classifier, "TMJBinaryPositionClassifier")
    with pytest.raises(ValueError, match="^invalid_position_checkpoint$"):
        classifier.load_position_checkpoint(path)
    assert not calls


@pytest.mark.parametrize("features", [[65536], [2] * 5])
def test_detector_checkpoint_refuses_before_constructor(tmp_path, monkeypatch, features):
    model = crop.TMJHeatmapDetector(features=[1])
    path = tmp_path / "invalid.pth"
    torch.save(
        {"model_config": {"features": features}, "model_state_dict": model.state_dict()}, path
    )
    calls = constructor_tripwire(monkeypatch, crop, "TMJHeatmapDetector")
    with pytest.raises(crop.ROIValidationError):
        crop.load_paired_detector(path, "cpu")
    assert not calls


@pytest.mark.parametrize(
    "kwargs", [{"features": (65536,)}, {"features": (2,) * 5}, {"fc_hidden": 2**32}]
)
def test_training_config_rejects_unloadable_architecture(kwargs):
    cfg = dataclasses.replace(cv.SagittalBinaryCVConfig(), **kwargs)
    with pytest.raises(ValueError, match="^unsupported_research_architecture$"):
        cv._validate_cv_config(cfg)


@pytest.mark.parametrize("model", [classifier.TMJBinaryPositionClassifier, crop.TMJHeatmapDetector])
def test_direct_constructor_refuses_before_conv_allocation(monkeypatch, model):
    calls = constructor_tripwire(monkeypatch, torch.nn, "Conv3d")
    with pytest.raises(ValueError, match="^unsupported_research_architecture$"):
        model(features=[65536])
    assert not calls


def test_default_architectures_remain_supported():
    # Construct real defaults, without the much larger detector input activations.
    assert classifier.TMJBinaryPositionClassifier().backbone[-1][0].out_channels == 128
    assert crop.TMJHeatmapDetector().bottleneck[0].out_channels == 512


def test_profile_edges_without_allocating_boundary_models():
    from models.blocks import validate_research_architecture

    validate_research_architecture([256] * 4, fc_hidden=2048)
    for features, hidden, channels in (
        ([257], 4, 1),
        ([2], 2049, 1),
        ([True], 4, 1),
        ([2], True, 1),
        ([2], 4, 65536),
    ):
        with pytest.raises(ValueError, match="^unsupported_research_architecture$"):
            validate_research_architecture(features, fc_hidden=hidden, in_channels=channels)


def test_cloud_preflight_refuses_before_reading_data_or_allocating(monkeypatch):
    from tools import run_research as runner
    from training import tmj_position_label_table as labels

    calls = constructor_tripwire(monkeypatch, labels, "build_canonical_index")
    cfg = cv.SagittalBinaryCVConfig(input_path="SyntheticPrivate", features=(65536,))
    with pytest.raises(runner.ResearchConfigError, match="^invalid_preflight$"):
        runner.preflight_research(runner.ResearchConfig(1, "binary", cfg))
    assert not calls


@pytest.mark.parametrize("operation", ["preflight", "cv"])
def test_crop_smaller_than_pooling_backbone_refuses_before_output(tmp_path, monkeypatch, operation):
    from tools import run_research as runner

    from .test_sagittal_binary_cv import _canonical_synthetic_config

    cfg = _canonical_synthetic_config(tmp_path)  # Genuine8-cube passports.
    cfg.features = (2, 2, 2, 2)  # Four pools require at least16voxels per axis.
    cfg.output_dir = str(tmp_path / "outputs")
    calls = constructor_tripwire(monkeypatch, classifier, "TMJBinaryPositionClassifier")
    if operation == "preflight":
        with pytest.raises(runner.ResearchConfigError, match="^invalid_preflight$"):
            runner.preflight_research(runner.ResearchConfig(1, "binary", cfg))
    else:
        with pytest.raises(ValueError, match="^incompatible_research_crop_shape$"):
            cv.run_sagittal_binary_cv(cfg)
    assert not calls
    assert not (tmp_path / "outputs").exists()


def test_exact_pooling_crop_boundary_has_real_forward():
    model = classifier.TMJBinaryPositionClassifier(features=[2, 2, 2], fc_hidden=4).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 8, 8, 8))[0].shape == (2, 1)
