"""Tests for models.tmj_heatmap_detector.TMJHeatmapDetector"""
import sys, os, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.tmj_heatmap_detector import TMJHeatmapDetector


class TestTMJHeatmapDetectorForward:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_output_shape(self, batch_size):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(batch_size, 1, 32, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 2, 32, 64, 64), f"Got {out.shape}"

    def test_output_matches_input_spatial(self):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(1, 1, 48, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape[2:] == x.shape[2:]

    def test_no_nan_output(self):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(1, 1, 32, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_backward_runs(self):
        from training.losses.heatmap_loss import weighted_mse_loss
        model = TMJHeatmapDetector()
        x      = torch.randn(1, 1, 32, 64, 64)
        target = torch.rand(1, 2, 32, 64, 64)
        out  = model(x)
        loss = weighted_mse_loss(out, target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_custom_features(self):
        model = TMJHeatmapDetector(features=[8, 16])
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 32, 32, 32))
        assert out.shape == (1, 2, 32, 32, 32)
