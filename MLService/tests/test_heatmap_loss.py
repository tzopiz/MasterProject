"""Tests for training.losses.heatmap_loss"""
import sys, os, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.losses.heatmap_loss import weighted_mse_loss


class TestWeightedMseLoss:
    def test_output_is_scalar(self):
        pred   = torch.rand(2, 2, 16, 16, 16)
        target = torch.rand(2, 2, 16, 16, 16)
        loss = weighted_mse_loss(pred, target)
        assert loss.ndim == 0

    def test_zero_loss_when_perfect(self):
        t = torch.rand(1, 2, 16, 16, 16)
        assert weighted_mse_loss(t, t).item() < 1e-6

    def test_loss_positive(self):
        pred   = torch.zeros(1, 2, 16, 16, 16)
        target = torch.ones(1, 2, 16, 16, 16) * 0.5
        assert weighted_mse_loss(pred, target).item() > 0

    def test_pos_weight_increases_peak_loss(self):
        pred   = torch.zeros(1, 2, 4, 4, 4)
        target = torch.zeros(1, 2, 4, 4, 4)
        target[0, 0, 2, 2, 2] = 1.0
        loss_low  = weighted_mse_loss(pred, target, pos_weight=1.0).item()
        loss_high = weighted_mse_loss(pred, target, pos_weight=20.0).item()
        assert loss_high > loss_low

    def test_backward_runs(self):
        pred   = torch.rand(1, 2, 16, 16, 16, requires_grad=True)
        target = torch.rand(1, 2, 16, 16, 16)
        weighted_mse_loss(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
