"""
Shared 3D CNN building blocks for TMJ position classifiers.
"""

import torch.nn as nn


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
