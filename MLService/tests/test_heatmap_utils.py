"""Tests for training.utils.heatmap"""
import sys, os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.utils.heatmap import make_heatmap, soft_argmax_3d, coords_from_heatmap


class TestMakeHeatmap:
    def test_output_shape(self):
        hm = make_heatmap(shape=(96, 128, 128), center_zyx=(288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.shape == (96, 128, 128)

    def test_output_dtype(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.dtype == np.float32

    def test_peak_at_center(self):
        # center_zyx in original space → divided by 6 → (48, 40, 30) in downsampled
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
        assert abs(peak_idx[0] - 48) <= 1
        assert abs(peak_idx[1] - 40) <= 1
        assert abs(peak_idx[2] - 30) <= 1

    def test_values_in_01(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0 + 1e-6

    def test_peak_equals_one(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert abs(hm.max() - 1.0) < 1e-5

    def test_sigma_controls_spread(self):
        hm_narrow = make_heatmap((96, 128, 128), (288, 240, 180), sigma=1.0, downsample_factor=6)
        hm_wide   = make_heatmap((96, 128, 128), (288, 240, 180), sigma=5.0, downsample_factor=6)
        assert hm_narrow.sum() < hm_wide.sum()


class TestSoftArgmax3d:
    def test_output_shape(self):
        hm = torch.zeros(96, 128, 128)
        hm[10, 20, 30] = 1.0
        coords = soft_argmax_3d(hm)
        assert coords.shape == (3,)

    def test_recovers_peak_location(self):
        hm = torch.zeros(96, 128, 128)
        z0, y0, x0 = 10, 20, 30
        for dz in range(-3, 4):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    z, y, x = z0+dz, y0+dy, x0+dx
                    if 0 <= z < 96 and 0 <= y < 128 and 0 <= x < 128:
                        hm[z, y, x] = float(np.exp(-(dz**2+dy**2+dx**2)/2.0))
        coords = soft_argmax_3d(hm)
        assert abs(coords[0].item() - z0) < 1.0
        assert abs(coords[1].item() - y0) < 1.0
        assert abs(coords[2].item() - x0) < 1.0

    def test_uniform_heatmap_returns_center(self):
        hm = torch.ones(10, 10, 10)
        coords = soft_argmax_3d(hm)
        assert abs(coords[0].item() - 4.5) < 0.1
        assert abs(coords[1].item() - 4.5) < 0.1
        assert abs(coords[2].item() - 4.5) < 0.1


class TestCoordsFromHeatmap:
    def test_returns_downsampled_and_original(self):
        hm = torch.zeros(96, 128, 128)
        hm[48, 64, 64] = 1.0
        ds_coords, orig_coords = coords_from_heatmap(hm, downsample_factor=6)
        assert ds_coords.shape == (3,)
        assert orig_coords.shape == (3,)
        # original = downsampled * 6
        assert abs(orig_coords[0].item() - ds_coords[0].item() * 6) < 1.0
