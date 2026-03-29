"""
Unit tests for models.tmj_position_classifier

All tests use synthetic in-memory tensors — no DICOM files needed.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tmj_position_classifier import TMJPositionClassifier, get_position_classifier


# ---------------------------------------------------------------------------
# Architecture / forward pass
# ---------------------------------------------------------------------------

class TestTMJPositionClassifierForward:
    """Verify output shapes for various input sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("input_shape", [
        (32, 48, 48),
        (16, 24, 24),
    ])
    def test_output_count_and_shape(self, batch_size, input_shape):
        """Forward pass must return exactly 4 tensors, each (B, 3)."""
        model = TMJPositionClassifier()
        model.eval()
        D, H, W = input_shape
        x = torch.randn(batch_size, 1, D, H, W)
        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == 4, "Expected 4 output heads"
        for out in outputs:
            assert out.shape == (batch_size, 3), (
                f"Expected ({batch_size}, 3), got {out.shape}"
            )

    def test_output_order(self):
        """Outputs come in the order (sag_right, sag_left, fr_right, fr_left)."""
        model = TMJPositionClassifier()
        model.eval()
        x = torch.randn(1, 1, 32, 48, 48)
        with torch.no_grad():
            sag_right, sag_left, fr_right, fr_left = model(x)
        # Shapes must all be (1, 3)
        for t in (sag_right, sag_left, fr_right, fr_left):
            assert t.shape == (1, 3)

    def test_no_nan_in_output(self):
        """Logits should not contain NaN for normal inputs."""
        model = TMJPositionClassifier()
        model.eval()
        x = torch.randn(2, 1, 32, 48, 48)
        with torch.no_grad():
            outputs = model(x)
        for out in outputs:
            assert not torch.isnan(out).any(), "NaN detected in model output"

    def test_custom_features(self):
        """Model should work with a non-default feature list."""
        model = TMJPositionClassifier(features=[8, 16])
        model.eval()
        x = torch.randn(1, 1, 32, 48, 48)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 4
        for out in outputs:
            assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TestLossComputation:
    """Verify that the four CrossEntropyLoss terms can be summed."""

    def test_total_loss_is_scalar(self):
        model = TMJPositionClassifier()
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(2, 1, 32, 48, 48)
        labels = torch.randint(0, 3, (2, 4))  # (B, 4)

        outputs = model(x)
        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))

        assert loss.ndim == 0, "Total loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive for random inputs"

    def test_backward_runs(self):
        model = TMJPositionClassifier()
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(2, 1, 32, 48, 48)
        labels = torch.randint(0, 3, (2, 4))

        outputs = model(x)
        loss = sum(criterion(outputs[i], labels[:, i]) for i in range(4))
        loss.backward()

        # Gradients should exist and be finite for all parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name}"
                )


# ---------------------------------------------------------------------------
# Parameter count sanity check
# ---------------------------------------------------------------------------

class TestModelParameters:
    def test_has_parameters(self):
        model = TMJPositionClassifier()
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_four_separate_heads(self):
        """All four heads must have disjoint parameter sets."""
        model = TMJPositionClassifier()
        head_params = {
            "sag_right": set(id(p) for p in model.head_sag_right.parameters()),
            "sag_left":  set(id(p) for p in model.head_sag_left.parameters()),
            "fr_right":  set(id(p) for p in model.head_fr_right.parameters()),
            "fr_left":   set(id(p) for p in model.head_fr_left.parameters()),
        }
        names = list(head_params.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = head_params[names[i]] & head_params[names[j]]
                assert not overlap, (
                    f"Heads {names[i]} and {names[j]} share parameters"
                )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestGetPositionClassifier:
    def test_factory_returns_model(self):
        model = get_position_classifier()
        assert isinstance(model, TMJPositionClassifier)

    def test_factory_no_pretrained(self):
        model = get_position_classifier(pretrained=None)
        assert model is not None
