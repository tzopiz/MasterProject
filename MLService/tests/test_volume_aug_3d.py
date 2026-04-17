"""Tests for training.utils.volume_aug_3d."""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.utils.volume_aug_3d import augment_binary_volume_train


@pytest.fixture
def small_volume():
    rng = np.random.default_rng(0)
    v = rng.random((16, 24, 32)).astype(np.float32)
    return np.clip(v, 0.0, 1.0)


def test_none_returns_same_shape(small_volume):
    out = augment_binary_volume_train(small_volume, "none")
    assert out.shape == small_volume.shape
    assert np.allclose(out, small_volume)


def test_strong_preserves_shape_and_range(small_volume):
    random.seed(123)
    out = augment_binary_volume_train(small_volume, "strong")
    assert out.shape == small_volume.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= -1e-5
    assert float(out.max()) <= 1.0 + 1e-5


def test_invalid_mode_raises(small_volume):
    with pytest.raises(ValueError):
        augment_binary_volume_train(small_volume, "typo")
