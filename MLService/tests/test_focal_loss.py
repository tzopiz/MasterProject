"""BinaryFocalLoss shape handling."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.losses.focal_loss import BinaryFocalLoss


def test_focal_accepts_b1_targets():
    crit = BinaryFocalLoss(gamma=2.0, alpha=0.5)
    logits = torch.randn(8, 1)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    loss = crit(logits, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
