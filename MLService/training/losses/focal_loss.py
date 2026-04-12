"""
Binary Focal Loss

Focal Loss for binary classification. Downweights easy examples so training
focuses on hard (minority-class) predictions.

Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 recovers standard BCE. gamma=2 is the
        value used in the original paper for dense object detection.
    alpha : float or None
        Weight for the positive class (class 1). When None, no alpha weighting
        is applied. Typical values: 0.25–0.75 depending on class imbalance.
    reduction : str
        "mean" (default) or "sum" or "none".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) or (B, 1) raw (pre-sigmoid) logits.
            targets: (B,) binary labels 0/1, dtype float32.

        Returns:
            Scalar loss (or per-sample tensor when reduction="none").
        """
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # (B, 1) → (B,)

        targets = targets.float().to(logits.device)

        # Standard per-sample BCE (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability assigned to the correct class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Focal weight suppresses easy examples
        focal_weight = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            # alpha for positive class, (1-alpha) for negative
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
