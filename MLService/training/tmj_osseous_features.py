"""Offline frozen ImageNet ROI features and train-only osseous logistic heads.

Callers own splitting, model selection, serialization and private artifact IO.
This library makes no localization, position or clinical-quality claim.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from training.tmj_osseous_research import ResearchError

WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
TRANSFORM_RECIPE = (
    "ResNet18 ImageNet1K V1; float16/float32 [16,1,96,96] finite [0,1]; "
    "whole ROI bilinear antialias resize 224x224 align_corners=False; RGB repeat; "
    "mean=(.485,.456,.406), std=(.229,.224,.225); frozen eval; mean 16x512"
)


def _fail(code):
    raise ResearchError(code) from None


def load_backbone(weights_path):
    """Load only a caller-provided, exact-digest public checkpoint; never download."""
    try:
        stream = Path(weights_path).open("rb")
    except Exception:
        _fail("unreadable_feature_weights")
    with stream:
        try:
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > 64 * 1024 * 1024:
                    _fail("feature_weights_checksum_mismatch")
                digest.update(block)
            if digest.hexdigest() != WEIGHTS_SHA256:
                _fail("feature_weights_checksum_mismatch")
            stream.seek(0)
        except ResearchError:
            raise
        except Exception:
            _fail("unreadable_feature_weights")
        try:
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            model.load_state_dict(
                torch.load(stream, map_location="cpu", weights_only=True), strict=True
            )
            model.fc = nn.Identity()
            return model.requires_grad_(False).eval()
        except Exception:
            _fail("feature_weights_load_failed")


def extract_features(backbone, crop):
    """Extract one fixed-size bag on CPU, without batching cohorts or opening files."""
    if (
        not isinstance(crop, np.ndarray)
        or crop.shape != (16, 1, 96, 96)
        or crop.dtype not in (np.dtype("float16"), np.dtype("float32"))
        or not np.isfinite(crop).all()
        or crop.min() < 0
        or crop.max() > 1
    ):
        _fail("invalid_feature_crop")
    if not isinstance(backbone, nn.Module):
        _fail("invalid_feature_backbone")
    try:
        backbone.requires_grad_(False).eval()
        with torch.inference_mode():
            images = torch.from_numpy(crop.astype(np.float32, copy=True))
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
            ).repeat(1, 3, 1, 1)
            mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            encoded = backbone((images - mean) / std)
            if (
                not isinstance(encoded, torch.Tensor)
                or encoded.shape != (16, 512)
                or not torch.isfinite(encoded).all()
            ):
                _fail("invalid_feature_output")
            result = encoded.float().mean(0).cpu().numpy().copy()
            if not np.isfinite(result).all():
                _fail("invalid_feature_output")
            return result
    except ResearchError:
        raise
    except Exception:
        _fail("feature_extraction_failed")


@dataclass
class HeadBundle:
    """Joblib-serializable fitted state; absent class support remains explicit."""

    mode: str
    C: float
    scaler: StandardScaler
    estimators: list
    train_prevalence: np.ndarray
    train_support: np.ndarray
    weights_sha256: str = WEIGHTS_SHA256
    transform_recipe: str = TRANSFORM_RECIPE


def _matrix(features):
    if (
        not isinstance(features, np.ndarray)
        or features.ndim != 2
        or features.shape[1] != 512
        or not features.shape[0]
        or features.dtype.kind not in "fiu"
        or not np.isfinite(features).all()
    ):
        _fail("invalid_feature_matrix")
    return (
        features
        if features.dtype in (np.dtype("float32"), np.dtype("float64"))
        else features.astype(np.float64)
    )


def fit_head(features, targets, *, mode, C=0.01):
    """Fit a fresh scaler and balanced positive-class estimator for supported outputs."""
    x = _matrix(features)
    if not isinstance(mode, str) or mode not in ("binary", "multilabel"):
        _fail("invalid_head_mode")
    if isinstance(C, (bool, np.bool_)) or not isinstance(C, (int, float, np.integer, np.floating)):
        _fail("invalid_head_C")
    try:
        C = float(C)
    except (OverflowError, TypeError, ValueError):
        _fail("invalid_head_C")
    if not np.isfinite(C) or C <= 0:
        _fail("invalid_head_C")
    outputs = 1 if mode == "binary" else 6
    if (
        not isinstance(targets, np.ndarray)
        or targets.shape != (len(x), outputs)
        or targets.dtype.kind not in "biuf"
        or not np.isin(targets, (0, 1)).all()
    ):
        _fail("invalid_feature_targets")
    prevalence = targets.mean(0, dtype=np.float64)
    support = (prevalence > 0) & (prevalence < 1)
    scaler, estimators = StandardScaler(), []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            scaled = scaler.fit_transform(x)
            for column in range(outputs):
                estimator = None
                if support[column]:
                    estimator = LogisticRegression(
                        C=float(C),
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=42,
                    )
                    estimator.fit(scaled, targets[:, column].astype(np.int64))
                estimators.append(estimator)
    except Exception:
        _fail("head_fit_failed")
    return HeadBundle(mode, float(C), scaler, estimators, prevalence, support)


def predict_head(bundle, features):
    """Return probabilities in binary/osseous-code 1..6 order, without refitting."""
    x = _matrix(features)
    if not isinstance(bundle, HeadBundle):
        _fail("invalid_head_bundle")
    try:
        with warnings.catch_warnings():
            # BLAS may raise spurious FP flags; validate actual scaled/logit/probability values.
            warnings.filterwarnings(
                "ignore",
                r"(divide by zero|overflow|invalid value) encountered in matmul",
                RuntimeWarning,
            )
            scaled = bundle.scaler.transform(x)
            if not np.isfinite(scaled).all():
                _fail("invalid_head_probabilities")
            outputs = 1 if bundle.mode == "binary" else 6
            result = np.tile(bundle.train_prevalence, (len(x), 1))
            for column, estimator in enumerate(bundle.estimators):
                if estimator is not None:
                    if not np.isfinite(estimator.decision_function(scaled)).all():
                        _fail("invalid_head_probabilities")
                    positive = np.flatnonzero(estimator.classes_ == 1)
                    if len(positive) != 1:
                        _fail("invalid_head_probabilities")
                    result[:, column] = estimator.predict_proba(scaled)[:, positive[0]]
            if (
                result.shape != (len(x), outputs)
                or not np.isfinite(result).all()
                or (result < 0).any()
                or (result > 1).any()
            ):
                _fail("invalid_head_probabilities")
            return result
    except ResearchError:
        raise
    except Exception:
        _fail("head_prediction_failed")
