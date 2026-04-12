"""
Weighted MSE loss for 3D heatmap detection.

Background voxels (target≈0) vastly outnumber positive peaks.
pos_weight amplifies gradients near GT peaks so the network
focuses on getting the peak location right.
"""

import torch


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """
    Weighted MSE: loss = mean( (1 + pos_weight * target) * (pred - target)^2 )

    Args:
        pred:       (B, C, D, H, W) raw model output.
        target:     (B, C, D, H, W) Gaussian heatmaps in [0, 1].
        pos_weight: Scalar multiplier for weight at positive peaks.

    Returns:
        Scalar loss.
    """
    weight = 1.0 + pos_weight * target
    return (weight * (pred - target) ** 2).mean()
