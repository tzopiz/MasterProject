"""
Tests for training.losses.focal_loss.BinaryFocalLoss
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.losses.focal_loss import BinaryFocalLoss


class TestBinaryFocalLossDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_targets_cpu_logits_cuda_runs_and_backward(self):
        """CPU targets with CUDA logits: targets must be moved to logits.device."""
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(4, device="cuda", requires_grad=True)
        targets = torch.randint(0, 2, (4,)).float()
        assert targets.device.type == "cpu"
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestBinaryFocalLossOutput:
    def test_output_is_scalar(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(4)
        targets = torch.randint(0, 2, (4,)).float()
        assert loss_fn(logits, targets).ndim == 0

    def test_accepts_2d_logits(self):
        """(B, 1) logits should be squeezed automatically."""
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_loss_is_positive(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        assert loss_fn(logits, targets).item() > 0

    def test_gamma_zero_matches_bce(self):
        """With gamma=0 and no alpha, focal loss == standard BCE."""
        loss_fn = BinaryFocalLoss(gamma=0.0)
        logits = torch.tensor([0.5, -0.5, 1.0, -1.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        fl = loss_fn(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        assert torch.isclose(fl, bce, atol=1e-5), f"focal={fl.item()}, bce={bce.item()}"

    def test_focal_weight_reduces_easy_examples(self):
        """A confidently-correct prediction contributes less with gamma>0."""
        loss_focal = BinaryFocalLoss(gamma=2.0)
        loss_bce = BinaryFocalLoss(gamma=0.0)
        logits = torch.tensor([5.0])  # highly confident
        targets = torch.tensor([1.0])  # correct label
        assert loss_focal(logits, targets).item() < loss_bce(logits, targets).item()

    def test_backward_runs_and_gradients_finite(self):
        loss_fn = BinaryFocalLoss()
        logits = torch.randn(4, requires_grad=True)
        targets = torch.randint(0, 2, (4,)).float()
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestBinaryFocalLossAlpha:
    def test_alpha_changes_loss(self):
        """Loss with alpha should differ from loss without alpha."""
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_no_alpha = BinaryFocalLoss(gamma=2.0, alpha=None)(logits, targets)
        loss_with_alpha = BinaryFocalLoss(gamma=2.0, alpha=0.25)(logits, targets)
        assert not torch.isclose(loss_no_alpha, loss_with_alpha)

    def test_loss_finite_with_extreme_logits(self):
        """Focal loss should not produce NaN/Inf for extreme logits."""
        loss_fn = BinaryFocalLoss(gamma=2.0)
        logits = torch.tensor([20.0, -20.0, 15.0, -15.0])
        targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)


class TestBinaryFocalLossReduction:
    def test_reduction_none_returns_per_sample_tensor(self):
        loss_fn = BinaryFocalLoss(reduction="none")
        logits = torch.randn(6)
        targets = torch.randint(0, 2, (6,)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == (6,), f"Expected (6,), got {loss.shape}"

    def test_reduction_sum_equals_none_sum(self):
        loss_fn_sum = BinaryFocalLoss(reduction="sum")
        loss_fn_none = BinaryFocalLoss(reduction="none")
        logits = torch.tensor([0.5, -0.5, 1.0, -1.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.isclose(
            loss_fn_sum(logits, targets), loss_fn_none(logits, targets).sum(), atol=1e-5
        )

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            BinaryFocalLoss(reduction="average")
