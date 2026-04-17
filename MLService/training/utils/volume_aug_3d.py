"""
Lightweight 3D augmentations for CBCT-like volumes (numpy, float32).

Applied **after** percentile normalization to approximately [0, 1].

**Safety note:** samples are already **per-side** condyle crops; axis flips and
small in-plane rotations act as geometric / appearance regularization. They do
not swap patient left/right labels (those are fixed per record). If a future
task is sensitive to absolute anatomical chirality within the crop, restrict
``mode`` to ``flip_only`` or ``none``.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from scipy import ndimage


def _random_flips(volume: np.ndarray) -> np.ndarray:
    out = volume
    for axis in range(3):
        if random.random() < 0.5:
            out = np.flip(out, axis=axis).copy()
    return out.astype(np.float32, copy=False)


def _random_small_rotations(volume: np.ndarray, max_degrees: float, plane_p: float) -> np.ndarray:
    """Up to three successive small rotations in orthogonal planes (order=1)."""
    out = volume
    axis_pairs: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
    for ax in axis_pairs:
        if random.random() < plane_p:
            angle = random.uniform(-max_degrees, max_degrees)
            out = ndimage.rotate(
                out,
                angle,
                axes=ax,
                reshape=False,
                order=1,
                mode="nearest",
                prefilter=False,
            ).astype(np.float32, copy=False)
    return out


def _random_intensity(
    volume: np.ndarray, scale_range: Tuple[float, float], shift_range: Tuple[float, float]
) -> np.ndarray:
    scale = random.uniform(scale_range[0], scale_range[1])
    shift = random.uniform(shift_range[0], shift_range[1])
    return np.clip(volume * scale + shift, 0.0, 1.0).astype(np.float32, copy=False)


def augment_binary_volume_train(
    volume: np.ndarray,
    mode: str,
    *,
    rot_max_degrees: float = 10.0,
    rot_plane_p: float = 0.5,
    intensity_scale: Tuple[float, float] = (0.95, 1.05),
    intensity_shift: Tuple[float, float] = (-0.05, 0.05),
    intensity_p: float = 0.8,
) -> np.ndarray:
    """
    Apply train-time augmentations to a single volume (D, H, W).

    Parameters
    ----------
    volume
        Normalized float32 array (typically in [0, 1]).
    mode
        ``\"none\"`` — return unchanged.
        ``\"flip_only\"`` — independent axis flips (p=0.5 each).
        ``\"strong\"`` — flips + small rotations + random intensity jitter.
    """
    if mode == "none":
        return volume.astype(np.float32, copy=False)
    if mode == "flip_only":
        return _random_flips(volume)
    if mode == "strong":
        out = _random_flips(volume)
        out = _random_small_rotations(out, rot_max_degrees, rot_plane_p)
        if random.random() < intensity_p:
            out = _random_intensity(out, intensity_scale, intensity_shift)
        return out
    raise ValueError(f"Unknown augment mode: {mode!r} (expected none|flip_only|strong)")
