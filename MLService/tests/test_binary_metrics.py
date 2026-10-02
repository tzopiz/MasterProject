"""Tests for training.utils.binary_metrics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.utils.binary_metrics import (
    binary_metrics_at_threshold,
    binary_roc_auc,
    calibration_report_binary_head,
    youden_optimal_threshold,
)


def test_youden_extremes():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.8])
    t = youden_optimal_threshold(y, s)
    assert 0.0 <= t <= 1.0


def test_auc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.01, 0.99, 1.0])
    assert binary_roc_auc(y, s) == pytest.approx(1.0)


def test_auc_single_class_returns_nan():
    y = np.zeros(4, dtype=int)
    s = np.random.rand(4)
    assert np.isnan(binary_roc_auc(y, s))


def test_metrics_at_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_metrics_at_threshold(y, s, 0.5)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_calibration_report_matches_components():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.85, 0.9, 0.15, 0.75])
    rep = calibration_report_binary_head(y, p)
    assert "optimal_threshold" in rep
    assert not np.isnan(rep["auc_roc"])


@pytest.mark.parametrize("scores", [[0.5, 0.5, 0.5, 0.5], [0.9, 0.8, 0.2, 0.1]])
def test_youden_tied_or_reversed_scores_have_finite_threshold(scores):
    threshold = youden_optimal_threshold(np.array([0, 0, 1, 1]), np.array(scores))
    assert np.isfinite(threshold)


def test_report_names_positive_class_and_undefined_recalls():
    m = binary_metrics_at_threshold([0, 0, 1, 1, 1], [0.1, 0.9, 0.2, 0.8, 0.9], 0.5)
    assert m["support"] == {"0": 2, "1": 3}
    assert m["positive_class"] == 1
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == 0.5
    assert m["f1_positive"] == pytest.approx(2 / 3)
    one_class = binary_metrics_at_threshold([0, 0], [0.1, 0.2], 0.5)
    assert np.isnan(one_class["sensitivity"])
    assert np.isnan(one_class["balanced_accuracy"])


@pytest.mark.parametrize(
    "labels,scores", [([], []), ([0, 2], [0.2, 0.9]), ([0, 1], [0.2]), ([0, 1], [0.2, np.nan])]
)
def test_metrics_reject_invalid_input(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics_at_threshold(labels, scores, 0.5)
