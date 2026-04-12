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
