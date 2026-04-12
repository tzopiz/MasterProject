"""
Tests for models.tmj_binary_position_classifier.TMJBinaryPositionClassifier
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier


class TestTMJBinaryPositionClassifierForward:
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("input_shape", [(32, 48, 48), (16, 24, 24)])
    def test_output_count_and_shape(self, batch_size, input_shape):
        """Forward must return 2 tensors, each (B, 1)."""
        model = TMJBinaryPositionClassifier()
        model.eval()
        D, H, W = input_shape
        x = torch.randn(batch_size, 1, D, H, W)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 2, "Expected (sag_logit, fr_logit)"
        for out in outputs:
            assert out.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {out.shape}"

    def test_output_order_sag_then_fr(self):
        model = TMJBinaryPositionClassifier()
        model.eval()
        x = torch.randn(1, 1, 32, 48, 48)
        with torch.no_grad():
            sag_logit, fr_logit = model(x)
        assert sag_logit.shape == (1, 1)
        assert fr_logit.shape == (1, 1)

    def test_no_nan_in_output(self):
        model = TMJBinaryPositionClassifier()
        model.eval()
        x = torch.randn(2, 1, 32, 48, 48)
        with torch.no_grad():
            sag, fr = model(x)
        assert not torch.isnan(sag).any()
        assert not torch.isnan(fr).any()

    def test_custom_features(self):
        model = TMJBinaryPositionClassifier(features=[8, 16])
        model.eval()
        x = torch.randn(1, 1, 32, 48, 48)
        with torch.no_grad():
            sag, fr = model(x)
        assert sag.shape == (1, 1)
        assert fr.shape == (1, 1)


class TestTMJBinaryPositionClassifierLoss:
    def test_focal_loss_backward_runs(self):
        from training.losses.focal_loss import BinaryFocalLoss

        model = TMJBinaryPositionClassifier()
        loss_fn = BinaryFocalLoss(gamma=2.0)

        x = torch.randn(2, 1, 32, 48, 48)
        sag_labels = torch.randint(0, 2, (2,)).float()
        fr_labels = torch.randint(0, 2, (2,)).float()

        sag_logit, fr_logit = model(x)
        loss = loss_fn(sag_logit, sag_labels) + loss_fn(fr_logit, fr_labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"


class TestTMJBinaryPositionClassifierParameters:
    def test_has_parameters(self):
        model = TMJBinaryPositionClassifier()
        n = sum(p.numel() for p in model.parameters())
        # Default config: features=[16,32,64,128], fc_hidden=256 → ~946k params
        assert n > 900_000, f"Expected >900k params for default config, got {n}"

    def test_two_separate_heads(self):
        model = TMJBinaryPositionClassifier()
        sag_ids = set(id(p) for p in model.head_sag.parameters())
        fr_ids = set(id(p) for p in model.head_fr.parameters())
        assert sag_ids.isdisjoint(fr_ids), "sag and fr heads share parameters"

    def test_fewer_params_than_4head_model(self):
        """Binary model (2 heads, 1 logit each) should have fewer params than 4-head (3 logit) model."""
        from models.tmj_position_classifier import TMJPositionClassifier

        binary = TMJBinaryPositionClassifier()
        multiclass = TMJPositionClassifier()
        n_binary = sum(p.numel() for p in binary.parameters())
        n_multi = sum(p.numel() for p in multiclass.parameters())
        assert n_binary < n_multi
