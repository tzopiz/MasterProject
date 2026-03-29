#!/usr/bin/env python3
"""
TMJ Position Classifier

3D CNN with a shared backbone and four independent classification heads.

Input:  (B, 1, D, H, W) — downsampled + cropped CBCT volume
Output: tuple of four (B, 3) logit tensors, in fixed order:
            (sag_right, sag_left, fr_right, fr_left)

Each head predicts one of three classes (0 = central, 1 = anterior/medial,
2 = posterior/lateral) for a single condyle and projection plane.

Loss: sum of four CrossEntropyLoss terms (one per head).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv3d → BN → ReLU layers followed by MaxPool3d(2)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=2, stride=2),
    )


class TMJPositionClassifier(nn.Module):
    """
    3D CNN classifier for TMJ condyle position.

    Shared encoder → global average pooling → four independent FC heads.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale CBCT).
    features : list of int
        Number of channels for each encoder block.
    fc_hidden : int
        Width of the hidden layer in each classification head.
    dropout : float
        Dropout probability applied before the final linear layer.
    """

    NUM_CLASSES: int = 3  # classes per head (0, 1, 2)

    def __init__(
        self,
        in_channels: int = 1,
        features: Optional[List[int]] = None,
        fc_hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        if features is None:
            features = [16, 32, 64, 128]

        # --- Shared backbone ---
        blocks: List[nn.Module] = []
        prev = in_channels
        for out_ch in features:
            blocks.append(_conv_block(prev, out_ch))
            prev = out_ch
        self.backbone = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        feat_dim = features[-1]

        # --- Four independent classification heads ---
        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feat_dim, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_hidden, self.NUM_CLASSES),
            )

        self.head_sag_right = _head()
        self.head_sag_left  = _head()
        self.head_fr_right  = _head()
        self.head_fr_left   = _head()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (B, 1, D, H, W) — normalised, downsampled, cropped volume

        Returns:
            (sag_right, sag_left, fr_right, fr_left)
            Each tensor is (B, 3) logits (no softmax applied).
        """
        feat = self.backbone(x)          # (B, C, d, h, w)
        feat = self.global_pool(feat)    # (B, C, 1, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, C)

        sag_right = self.head_sag_right(feat)
        sag_left  = self.head_sag_left(feat)
        fr_right  = self.head_fr_right(feat)
        fr_left   = self.head_fr_left(feat)

        return sag_right, sag_left, fr_right, fr_left


def get_position_classifier(pretrained: Optional[str] = None) -> TMJPositionClassifier:
    """
    Factory: create TMJPositionClassifier, optionally loading saved weights.

    Args:
        pretrained: path to a .pth checkpoint (saved with torch.save on a dict
                    containing 'model_state_dict', or directly as a state dict).

    Returns:
        TMJPositionClassifier instance.
    """
    model = TMJPositionClassifier()

    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

    return model


if __name__ == "__main__":
    import torch

    model = TMJPositionClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    x = torch.randn(2, 1, 32, 48, 48)
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shapes: {[t.shape for t in outputs]}")
