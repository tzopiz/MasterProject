"""
Shared 3D CNN building blocks for TMJ position classifiers.
"""

import torch.nn as nn


def validate_research_architecture(features, *, fc_hidden=None, in_channels=1, crop_size=None):
    """Bound model allocation before construction, in preflight and loaders.

    Supported research profile: one input channel, 1..4 encoder stages, each
    1..256 channels; classifier hidden head 1..2048. The detector's internal
    bottleneck may double the final width to512. This bounds parameter shapes,
    not training activation memory or CUDA availability. A known cubic crop
    must also survive every MaxPool3d(2). Generic blocks below
    remain unrestricted for other historical models.
    """
    if (
        type(in_channels) is not int
        or in_channels != 1
        or not isinstance(features, (list, tuple))
        or not 1 <= len(features) <= 4
        or any(type(width) is not int or not 1 <= width <= 256 for width in features)
        or (fc_hidden is not None and (type(fc_hidden) is not int or not 1 <= fc_hidden <= 2048))
    ):
        raise ValueError("unsupported_research_architecture")
    if crop_size is not None and (type(crop_size) is not int or crop_size < 2 ** len(features)):
        raise ValueError("incompatible_research_crop_shape")


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
