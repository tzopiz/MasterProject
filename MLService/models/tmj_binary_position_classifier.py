#!/usr/bin/env python3
"""
TMJ Binary Position Classifier

3D CNN with a shared backbone and two independent binary classification heads:
one for the sagittal plane and one for the frontal plane.

Input:  (B, 1, D, H, W) — 128×128×128 detector-based crop for ONE side (left or right)
Output: (sag_logit, fr_logit) — each (B, 1), raw logits (no sigmoid applied).

Decision at inference: sigmoid(logit) > threshold → 1 (non-central), else 0 (central).
Threshold is calibrated post-training via ROC / Youden's J and stored in config.json.

Loss: BinaryFocalLoss applied independently to each head, then summed.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.blocks import _conv_block


class TMJBinaryPositionClassifier(nn.Module):
    """
    Binary TMJ position classifier (central vs non-central), per side.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale CBCT crop).
    features : list of int
        Output channels for each encoder block.
    fc_hidden : int
        Width of the hidden layer in each head.
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Optional[List[int]] = None,
        fc_hidden: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if features is None:
            features = [16, 32, 64, 128]

        blocks: List[nn.Module] = []
        prev = in_channels
        for out_ch in features:
            blocks.append(_conv_block(prev, out_ch))
            prev = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        feat_dim = features[-1]

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feat_dim, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_hidden, 1),  # 1 logit → binary decision
            )

        self.head_sag = _head()
        self.head_fr = _head()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) — normalised 128³ crop for one condyle side.

        Returns:
            (sag_logit, fr_logit) — each (B, 1), raw logits.
        """
        feat = self.backbone(x)           # (B, C, d, h, w)
        feat = self.global_pool(feat)     # (B, C, 1, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, C)
        return self.head_sag(feat), self.head_fr(feat)
