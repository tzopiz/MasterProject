"""
3D Gaussian heatmap utilities for TMJ keypoint detection.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple


def make_heatmap(
    shape: Tuple[int, int, int],
    center_zyx: Tuple[int, int, int],
    sigma: float = 3.0,
    downsample_factor: int = 6,
) -> np.ndarray:
    """
    Generate a 3D Gaussian heatmap.

    Args:
        shape: Output shape (D, H, W) in downsampled voxel space.
        center_zyx: Joint center (z, y, x) in ORIGINAL voxel space.
        sigma: Gaussian sigma in downsampled voxels.
        downsample_factor: Factor applied to volume before training.

    Returns:
        float32 array of shape `shape`, values in [0, 1], peak=1.
    """
    D, H, W = shape
    cz = center_zyx[0] / downsample_factor
    cy = center_zyx[1] / downsample_factor
    cx = center_zyx[2] / downsample_factor

    z = np.arange(D, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    x = np.arange(W, dtype=np.float32)

    dist_sq = (
        (z[:, None, None] - cz) ** 2
        + (y[None, :, None] - cy) ** 2
        + (x[None, None, :] - cx) ** 2
    )
    heatmap = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)
    return heatmap  # peak = 1.0 at center


def soft_argmax_3d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable coordinate extraction from a 3D heatmap.

    Args:
        heatmap: (D, H, W) tensor (raw logits or probabilities).

    Returns:
        (3,) tensor [z, y, x] in voxel coordinates (downsampled space).
    """
    D, H, W = heatmap.shape
    heatmap = heatmap / (heatmap.sum() + 1e-8)

    z_grid = torch.arange(D, dtype=heatmap.dtype, device=heatmap.device)
    y_grid = torch.arange(H, dtype=heatmap.dtype, device=heatmap.device)
    x_grid = torch.arange(W, dtype=heatmap.dtype, device=heatmap.device)

    z = (heatmap.sum(dim=[1, 2]) * z_grid).sum()
    y = (heatmap.sum(dim=[0, 2]) * y_grid).sum()
    x = (heatmap.sum(dim=[0, 1]) * x_grid).sum()

    return torch.stack([z, y, x])


def coords_from_heatmap(
    heatmap: torch.Tensor,
    downsample_factor: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract joint coordinates from a single-channel predicted heatmap.

    Args:
        heatmap: (D, H, W) tensor.
        downsample_factor: Multiply to recover original-space coordinates.

    Returns:
        (ds_coords, orig_coords): both (3,) tensors [z, y, x].
    """
    ds_coords = soft_argmax_3d(heatmap)
    orig_coords = ds_coords * float(downsample_factor)
    return ds_coords, orig_coords
