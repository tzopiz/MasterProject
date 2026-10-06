"""Offline behavioral checks for frozen ROI features and fitted heads."""
import hashlib
import importlib
import importlib.util
import io
import sys
import types
import warnings

import joblib
import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning
from torch import nn


@pytest.fixture
def features():
    assert importlib.util.find_spec("training.tmj_osseous_features") is not None
    return importlib.import_module("training.tmj_osseous_features")


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.inputs = None

    def forward(self, images):
        assert not torch.is_grad_enabled()
        self.inputs = images.clone()
        rgb = self.bn(images).mean((2, 3)) * self.scale
        return rgb.repeat(1, 171)[:, :512]


def matrix():
    result = np.zeros((12, 512), dtype=np.float32)
    result[:, 0] = np.arange(12) - 5.5
    return result


def test_frozen_feature_library_exists():
    assert importlib.util.find_spec("training.tmj_osseous_features") is not None


def test_extractor_whole_roi_rgb_normalization_and_frozen_pooling(features):
    backbone = Probe().train()
    bag = np.zeros((16, 1, 96, 96), dtype=np.float16)
    bag[8:] = 1
    before = backbone.bn.running_mean.clone()
    result = features.extract_features(backbone, bag)
    assert result.shape == (512,) and result.dtype == np.float32
    expected = np.array([(.5 - .485) / .229, (.5 - .456) / .224, (.5 - .406) / .225])
    np.testing.assert_allclose(result[:3], expected, atol=2e-5)
    assert backbone.inputs.shape == (16, 3, 224, 224)
    assert not backbone.training and all(not p.requires_grad for p in backbone.parameters())
    assert all(p.grad is None for p in backbone.parameters())
    torch.testing.assert_close(backbone.bn.running_mean, before)
    np.testing.assert_allclose(features.extract_features(backbone, bag[::-1].copy()), result, atol=2e-6)
    np.testing.assert_allclose(features.extract_features(backbone, bag.astype(np.float32)), result, atol=2e-6)
    bag[:] = 0
    bag[:, :, :, 48:] = 1
    features.extract_features(backbone, bag)
    # Whole-ROI bilinear interpolation crosses the edge with 2/7 and 5/7 weights.
    edge = backbone.inputs[0, 0, 111, 111:113].numpy() * .229 + .485
    np.testing.assert_allclose(edge, [2 / 7, 5 / 7], atol=5e-6)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "low", "high"])
def test_bad_crops_fail_without_backbone_execution(features, mutation):
    bag = np.zeros((16, 1, 96, 96), dtype=np.float16)
    if mutation == "shape":
        bag = bag[:15]
    elif mutation == "dtype":
        bag = bag.astype(np.uint8)
    else:
        bag[0, 0, 0, 0] = {"nan": np.nan, "low": -.1, "high": 1.1}[mutation]
    with pytest.raises(features.ResearchError, match="^invalid_feature_crop$"):
        features.extract_features(None, bag)


def test_binary_positive_probability_scaler_is_training_only_and_bundle_serializes(features):
    x = matrix()
    bundle = features.fit_head(x, (x[:, :1] > 0).astype(np.int64), mode="binary")
    before = bundle.scaler.mean_.copy()
    probabilities = features.predict_head(bundle, x[[0, -1]])
    assert probabilities.shape == (2, 1) and probabilities[0, 0] < .5 < probabilities[1, 0]
    features.predict_head(bundle, x + 1000)
    np.testing.assert_array_equal(bundle.scaler.mean_, before)
    assert before[0] == 0 and bundle.train_support.tolist() == [True]
    assert bundle.mode == "binary" and bundle.C == .01
    assert bundle.weights_sha256 == features.WEIGHTS_SHA256 and "224" in bundle.transform_recipe
    stream = io.BytesIO()
    joblib.dump(bundle, stream)
    stream.seek(0)
    np.testing.assert_array_equal(features.predict_head(joblib.load(stream), x[[0, -1]]), probabilities)


@pytest.mark.parametrize("mode, outputs", [("binary", 1), ("multilabel", 6)])
def test_constant_outputs_keep_training_prevalence_and_label_order(features, mode, outputs):
    x = matrix()
    targets = np.zeros((12, outputs), dtype=int)
    targets[:, -1] = 1
    if outputs == 6:
        targets[:, 2] = x[:, 0] > 0
    bundle = features.fit_head(x, targets, mode=mode)
    probabilities = features.predict_head(bundle, x[[0, -1]])
    assert probabilities.shape == (2, outputs)
    np.testing.assert_array_equal(probabilities[:, -1], [1, 1])
    if outputs == 6:
        np.testing.assert_array_equal(probabilities[:, [0, 1, 3, 4]], 0)
        assert probabilities[0, 2] < .5 < probabilities[1, 2]
        assert bundle.train_support.tolist() == [False, False, True, False, False, False]
        np.testing.assert_array_equal(bundle.train_prevalence, [0, 0, .5, 0, 0, 1])
    else:
        assert bundle.train_support.tolist() == [False]
    assert np.isfinite(probabilities).all()


def test_balanced_heads_remain_independent_on_imbalanced_training(features):
    x = np.zeros((12, 512), dtype=np.float32)
    targets = np.zeros((12, 6), dtype=np.int64)
    targets[:3, 0] = 1
    targets[3:, 1] = 1
    bundle = features.fit_head(x, targets, mode="multilabel")
    # With no signal each balanced head predicts .5, regardless of prevalence.
    probabilities = features.predict_head(bundle, x[:1])
    np.testing.assert_allclose(probabilities[0, :2], [.5, .5], atol=1e-8)
    np.testing.assert_array_equal(bundle.train_prevalence[:2], [.25, .75])
    informative = matrix()
    targets[:, 0] = informative[:, 0] > 0
    targets[:, 1] = 1 - targets[:, 0]
    bundle = features.fit_head(informative, targets, mode="multilabel")
    probabilities = features.predict_head(bundle, informative[[0, -1]])
    assert probabilities[0, 0] < .5 < probabilities[0, 1]
    assert probabilities[1, 1] < .5 < probabilities[1, 0]


@pytest.mark.parametrize("mutation", ["shape", "empty", "nan", "string", "mode", "target_shape", "target_value", "target_nan", "C", "C_nan", "C_bool"])
def test_head_rejects_invalid_inputs_with_safe_errors(features, mutation):
    x, y, mode, c = matrix(), np.zeros((12, 1)), "binary", .01
    if mutation == "shape":
        x = x[:, :511]
    elif mutation == "empty":
        x, y = x[:0], y[:0]
    elif mutation == "nan":
        x[0, 0] = np.nan
    elif mutation == "string":
        x = x.astype(str)
    elif mutation == "mode":
        mode = "position"
    elif mutation == "target_shape":
        y = y[:, 0]
    elif mutation.startswith("target_"):
        y[0, 0] = np.nan if mutation == "target_nan" else 2
    else:
        c = {"C": 0, "C_nan": np.nan, "C_bool": True}[mutation]
    with pytest.raises(features.ResearchError) as caught:
        features.fit_head(x, y, mode=mode, C=c)
    assert str(caught.value) in {"invalid_feature_matrix", "invalid_feature_targets", "invalid_head_mode", "invalid_head_C"}


def test_convergence_warning_cannot_produce_a_silent_bundle(features, monkeypatch):
    def warn(*args, **kwargs):
        warnings.warn("private sample path", ConvergenceWarning)
    monkeypatch.setattr(features.LogisticRegression, "fit", warn)
    with pytest.raises(features.ResearchError, match="^head_fit_failed$"):
        features.fit_head(matrix(), (matrix()[:, :1] > 0).astype(int), mode="binary")


def test_predict_rejects_nonfinite_or_wrong_width(features):
    bundle = features.fit_head(matrix(), np.ones((12, 1)), mode="binary")
    for x in (np.full((1, 512), np.inf), np.zeros((1, 511))):
        with pytest.raises(features.ResearchError, match="^invalid_feature_matrix$"):
            features.predict_head(bundle, x)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_head_preserves_feature_precision_used_by_scaler(features, monkeypatch, dtype):
    original = features.LogisticRegression.fit
    def fit(estimator, x, y):
        assert x.dtype == dtype
        return original(estimator, x, y)
    monkeypatch.setattr(features.LogisticRegression, "fit", fit)
    features.fit_head(matrix().astype(dtype), (matrix()[:, :1] > 0).astype(int), mode="binary")


def test_finite_features_with_overflowing_logits_are_rejected(features):
    bundle = features.fit_head(matrix(), (matrix()[:, :1] > 0).astype(int), mode="binary")
    bundle.estimators[0].coef_[0, 0] = np.finfo(np.float64).max
    with pytest.raises(features.ResearchError, match="^invalid_head_probabilities$"):
        features.predict_head(bundle, matrix()[-1:].astype(np.float64))


@pytest.mark.parametrize("c", [10 ** 400, np.longdouble(np.inf)])
def test_unrepresentable_C_is_safe_even_without_train_support(features, c):
    with pytest.raises(features.ResearchError, match="^invalid_head_C$"):
        features.fit_head(matrix(), np.ones((12, 1)), mode="binary", C=c)


def test_loader_rejects_missing_or_wrong_hash_before_deserialization(features, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unverified weights must not be deserialized")
    monkeypatch.setattr(torch, "load", forbidden)
    bad = tmp_path / "private-name.pt"
    bad.write_bytes(b"untrusted")
    for path, code in ((bad, "feature_weights_checksum_mismatch"), (bad.with_suffix(".missing"), "unreadable_feature_weights")):
        with pytest.raises(features.ResearchError, match=f"^{code}$"):
            features.load_backbone(path)


def test_loader_uses_only_verified_offline_weights_and_freezes(features, tmp_path, monkeypatch):
    model = Probe()
    model.fc = nn.Linear(512, 1000)
    path = tmp_path / "synthetic.pt"
    torch.save(model.state_dict(), path)
    monkeypatch.setattr(features, "WEIGHTS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    def factory(*, weights):
        assert weights is None  # Factory may never select a downloading weights enum.
        return model
    monkeypatch.setitem(sys.modules, "torchvision.models", types.SimpleNamespace(resnet18=factory))
    real_load = torch.load
    def safe_load(*args, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", safe_load)
    loaded = features.load_backbone(path)
    assert isinstance(loaded.fc, nn.Identity)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
