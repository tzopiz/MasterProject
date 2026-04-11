#!/usr/bin/env python3
"""
TMJ Heatmap Detector — 3D U-Net.

Input:  (B, 1, D, H, W)  — normalized downsampled CBCT
Output: (B, 2, D, H, W)  — raw heatmap logits, ch0=left, ch1=right

Apply sigmoid at inference to get probabilities in [0,1].
Extract coordinates with soft_argmax_3d from training.utils.heatmap.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv3d → BN → ReLU (no pooling)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
    )


class _EncoderBlock(nn.Module):
    """Double conv → store skip → MaxPool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    """Upsample → concat skip → double conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims don't match (odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[4]-x.shape[4],
                           0, skip.shape[3]-x.shape[3],
                           0, skip.shape[2]-x.shape[2]])
        return self.conv(torch.cat([skip, x], dim=1))


class TMJHeatmapDetector(nn.Module):
    """
    3D U-Net for TMJ joint heatmap prediction.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for grayscale CBCT).
    features : list of int
        Channel sizes for each encoder level.
    out_channels : int
        Output channels (2: left + right heatmap).
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Optional[List[int]] = None,
        out_channels: int = 2,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(_EncoderBlock(prev, f))
            prev = f

        # Bottleneck
        self.bottleneck = _double_conv(features[-1], features[-1] * 2)
        prev = features[-1] * 2

        # Decoder (reverse features)
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(_DecoderBlock(prev, f, f))
            prev = f

        # Output
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)   # raw logits, no sigmoid
