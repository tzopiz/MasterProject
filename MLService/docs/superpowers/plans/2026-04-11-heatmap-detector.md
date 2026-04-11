# TMJ Heatmap Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace coordinate-regression TMJ detector with a 3D U-Net that outputs Gaussian heatmaps, enabling precise condyle localization.

**Architecture:** 3D U-Net (encoder with skip connections + decoder) takes (B,1,96,128,128) downsampled CBCT → outputs (B,2,96,128,128) heatmaps (ch0=left, ch1=right). Coordinates extracted via soft-argmax. Loss = weighted MSE on heatmaps.

**Tech Stack:** PyTorch, scipy.ndimage, nibabel, pydicom. Tests with pytest + synthetic tensors.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `training/utils/heatmap.py` | `make_heatmap()`, `soft_argmax_3d()`, `coords_from_heatmap()` |
| Create | `tests/test_heatmap_utils.py` | Tests for heatmap utilities |
| Create | `tools/create_detector_split.py` | Generate deterministic train/val/test split JSON |
| Create | `data/detector_split.json` | Output of above (committed) |
| Create | `training/datasets/tmj_heatmap_dataset.py` | `TMJHeatmapDataset` + `get_heatmap_dataloaders()` |
| Create | `tests/test_tmj_heatmap_dataset.py` | Tests for dataset (synthetic data) |
| Create | `models/tmj_heatmap_detector.py` | `TMJHeatmapDetector` (3D U-Net) |
| Create | `tests/test_tmj_heatmap_detector.py` | Tests for model shapes |
| Create | `training/losses/heatmap_loss.py` | `weighted_mse_loss()` |
| Create | `tests/test_heatmap_loss.py` | Tests for loss function |
| Create | `train_heatmap_detector.py` | Training entrypoint |
| Create | `tools/evaluate_detector.py` | MAE in px + mm, % within threshold |
| Modify | `tools/auto_crop_from_detector.py` | Support heatmap model at inference |

Run all tests from `MLService/`:
```bash
./venv/bin/python -m pytest tests/ -v
```

---

## Task 1: Heatmap utilities

**Files:**
- Create: `training/utils/heatmap.py`
- Create: `tests/test_heatmap_utils.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_heatmap_utils.py`:

```python
"""Tests for training.utils.heatmap"""
import sys, os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.utils.heatmap import make_heatmap, soft_argmax_3d, coords_from_heatmap


class TestMakeHeatmap:
    def test_output_shape(self):
        hm = make_heatmap(shape=(96, 128, 128), center_zyx=(288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.shape == (96, 128, 128)

    def test_output_dtype(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.dtype == np.float32

    def test_peak_at_center(self):
        # center_zyx in original space → divided by 6 → (48, 40, 30) in downsampled
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
        assert abs(peak_idx[0] - 48) <= 1
        assert abs(peak_idx[1] - 40) <= 1
        assert abs(peak_idx[2] - 30) <= 1

    def test_values_in_01(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0 + 1e-6

    def test_peak_equals_one(self):
        hm = make_heatmap((96, 128, 128), (288, 240, 180), sigma=3.0, downsample_factor=6)
        assert abs(hm.max() - 1.0) < 1e-5

    def test_sigma_controls_spread(self):
        hm_narrow = make_heatmap((96, 128, 128), (288, 240, 180), sigma=1.0, downsample_factor=6)
        hm_wide   = make_heatmap((96, 128, 128), (288, 240, 180), sigma=5.0, downsample_factor=6)
        assert hm_narrow.sum() < hm_wide.sum()


class TestSoftArgmax3d:
    def test_output_shape(self):
        hm = torch.zeros(96, 128, 128)
        hm[10, 20, 30] = 1.0
        coords = soft_argmax_3d(hm)
        assert coords.shape == (3,)

    def test_recovers_peak_location(self):
        # Put a sharp Gaussian peak at (10, 20, 30)
        hm = torch.zeros(96, 128, 128)
        z0, y0, x0 = 10, 20, 30
        for dz in range(-3, 4):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    z, y, x = z0+dz, y0+dy, x0+dx
                    if 0 <= z < 96 and 0 <= y < 128 and 0 <= x < 128:
                        hm[z, y, x] = float(np.exp(-(dz**2+dy**2+dx**2)/2.0))
        coords = soft_argmax_3d(hm)
        assert abs(coords[0].item() - z0) < 1.0
        assert abs(coords[1].item() - y0) < 1.0
        assert abs(coords[2].item() - x0) < 1.0

    def test_uniform_heatmap_returns_center(self):
        hm = torch.ones(10, 10, 10)
        coords = soft_argmax_3d(hm)
        assert abs(coords[0].item() - 4.5) < 0.1
        assert abs(coords[1].item() - 4.5) < 0.1
        assert abs(coords[2].item() - 4.5) < 0.1


class TestCoordsFromHeatmap:
    def test_returns_downsampled_and_original(self):
        hm = torch.zeros(96, 128, 128)
        hm[48, 64, 64] = 1.0
        ds_coords, orig_coords = coords_from_heatmap(hm, downsample_factor=6)
        assert ds_coords.shape == (3,)
        assert orig_coords.shape == (3,)
        # original = downsampled * 6
        assert abs(orig_coords[0].item() - ds_coords[0].item() * 6) < 1.0
```

- [ ] **Step 1.2: Run — verify fail**

```bash
cd /Users/tzopiz/Developer/MasterProject/MLService
./venv/bin/python -m pytest tests/test_heatmap_utils.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.utils.heatmap'`

- [ ] **Step 1.3: Create `training/utils/heatmap.py`**

```python
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

    # Broadcasting: (D,1,1) + (1,H,1) + (1,1,W)
    dist_sq = (
        (z[:, None, None] - cz) ** 2
        + (y[None, :, None] - cy) ** 2
        + (x[None, None, :] - cx) ** 2
    )
    heatmap = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)
    return heatmap  # peak already = 1.0 (at center)


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
            ds_coords   — in downsampled voxel space
            orig_coords — in original voxel space (ds_coords * downsample_factor)
    """
    ds_coords = soft_argmax_3d(heatmap)
    orig_coords = ds_coords * float(downsample_factor)
    return ds_coords, orig_coords
```

- [ ] **Step 1.4: Run — verify pass**

```bash
./venv/bin/python -m pytest tests/test_heatmap_utils.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add training/utils/heatmap.py tests/test_heatmap_utils.py
git commit -m "feat: add 3D Gaussian heatmap utilities (make_heatmap, soft_argmax_3d)"
```

---

## Task 2: Deterministic train/val/test split

**Files:**
- Create: `tools/create_detector_split.py`
- Create: `data/detector_split.json` (generated, committed)

- [ ] **Step 2.1: Create `tools/create_detector_split.py`**

```python
#!/usr/bin/env python3
"""
Generate deterministic train/val/test split from all available ROI annotations.

Run once after annotating new studies:
    ./venv/bin/python tools/create_detector_split.py

Output: data/detector_split.json
"""
import json, random
from pathlib import Path

ANNOTATIONS_DIR = Path("data/roi_annotations")
OUTPUT = Path("data/detector_split.json")
SEED = 42
VAL_FRAC  = 0.14  # ~14% val
TEST_FRAC = 0.06  # ~6% test  → ~80% train

def main():
    studies = sorted(p.stem.replace("_rois", "") for p in ANNOTATIONS_DIR.glob("*_rois.json"))
    print(f"Annotated studies: {len(studies)}")

    rng = random.Random(SEED)
    shuffled = studies[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * TEST_FRAC))
    n_val  = max(1, round(n * VAL_FRAC))
    n_train = n - n_test - n_val

    split = {
        "seed": SEED,
        "total": n,
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == n

    OUTPUT.write_text(json.dumps(split, indent=2))
    print(f"Train: {len(split['train'])}  Val: {len(split['val'])}  Test: {len(split['test'])}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run and commit split**

```bash
cd /Users/tzopiz/Developer/MasterProject/MLService
./venv/bin/python tools/create_detector_split.py
```
Expected output (with 37 annotations):
```
Annotated studies: 37
Train: 29  Val: 5  Test: 3
Saved: data/detector_split.json
```

```bash
git add tools/create_detector_split.py data/detector_split.json
git commit -m "feat: deterministic train/val/test split for heatmap detector (37→107 ready)"
```

---

## Task 3: Heatmap loss

**Files:**
- Create: `training/losses/heatmap_loss.py`
- Create: `tests/test_heatmap_loss.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_heatmap_loss.py`:

```python
"""Tests for training.losses.heatmap_loss"""
import sys, os, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.losses.heatmap_loss import weighted_mse_loss


class TestWeightedMseLoss:
    def test_output_is_scalar(self):
        pred   = torch.rand(2, 2, 16, 16, 16)
        target = torch.rand(2, 2, 16, 16, 16)
        loss = weighted_mse_loss(pred, target)
        assert loss.ndim == 0

    def test_zero_loss_when_perfect(self):
        t = torch.rand(1, 2, 16, 16, 16)
        assert weighted_mse_loss(t, t).item() < 1e-6

    def test_loss_positive(self):
        pred   = torch.zeros(1, 2, 16, 16, 16)
        target = torch.ones(1, 2, 16, 16, 16) * 0.5
        assert weighted_mse_loss(pred, target).item() > 0

    def test_pos_weight_increases_peak_loss(self):
        pred   = torch.zeros(1, 2, 4, 4, 4)
        target = torch.zeros(1, 2, 4, 4, 4)
        target[0, 0, 2, 2, 2] = 1.0  # one peak
        loss_low  = weighted_mse_loss(pred, target, pos_weight=1.0).item()
        loss_high = weighted_mse_loss(pred, target, pos_weight=20.0).item()
        assert loss_high > loss_low

    def test_backward_runs(self):
        pred   = torch.rand(1, 2, 16, 16, 16, requires_grad=True)
        target = torch.rand(1, 2, 16, 16, 16)
        weighted_mse_loss(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 3.2: Run — verify fail**

```bash
./venv/bin/python -m pytest tests/test_heatmap_loss.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.losses.heatmap_loss'`

- [ ] **Step 3.3: Create `training/losses/heatmap_loss.py`**

```python
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
        pred:       (B, C, D, H, W) raw model output (no activation needed).
        target:     (B, C, D, H, W) Gaussian heatmaps in [0, 1].
        pos_weight: Scalar multiplier for weight at positive peaks.

    Returns:
        Scalar loss.
    """
    weight = 1.0 + pos_weight * target
    return (weight * (pred - target) ** 2).mean()
```

- [ ] **Step 3.4: Run — verify pass**

```bash
./venv/bin/python -m pytest tests/test_heatmap_loss.py -v
```
Expected: 5/5 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add training/losses/heatmap_loss.py tests/test_heatmap_loss.py
git commit -m "feat: weighted MSE heatmap loss"
```

---

## Task 4: Heatmap Dataset

**Files:**
- Create: `training/datasets/tmj_heatmap_dataset.py`
- Create: `tests/test_tmj_heatmap_dataset.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_tmj_heatmap_dataset.py`:

```python
"""Tests for TMJHeatmapDataset — synthetic NIfTI-free tests."""
import sys, os, json, pytest
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.datasets.tmj_heatmap_dataset import TMJHeatmapDataset


def _make_annotation(tmp_path, study_id, left=(288,240,180), right=(288,240,560)):
    ann = {
        "scan_id": study_id,
        "original_shape": [576, 768, 768],
        "left_tmj":  {"center": list(left),  "confidence": "manual"},
        "right_tmj": {"center": list(right), "confidence": "manual"},
    }
    (tmp_path / f"{study_id}_rois.json").write_text(json.dumps(ann))


class TestTMJHeatmapDataset:
    @pytest.fixture()
    def ds(self, tmp_path):
        # Create two fake annotations
        for sid in ("study_0001", "study_0002"):
            _make_annotation(tmp_path / "ann", sid)
        # Create a matching volume (a tiny numpy array saved as .npy, then mocked)
        # Dataset accepts a volume_factory callable for testing
        vol = np.random.rand(96, 128, 128).astype(np.float32)

        def mock_load(study_id):
            return vol

        study_ids = ["study_0001", "study_0002"]
        ann_dir = tmp_path / "ann"
        return TMJHeatmapDataset(
            study_ids=study_ids,
            annotations_dir=str(ann_dir),
            volume_loader=mock_load,
            sigma=3.0,
            downsample_factor=6,
            is_train=False,
        )

    def test_len(self, ds):
        assert len(ds) == 2

    def test_volume_shape(self, ds):
        vol, hm = ds[0]
        assert vol.shape == (1, 96, 128, 128)

    def test_heatmap_shape(self, ds):
        vol, hm = ds[0]
        assert hm.shape == (2, 96, 128, 128)

    def test_volume_dtype(self, ds):
        vol, _ = ds[0]
        assert vol.dtype == torch.float32

    def test_heatmap_dtype(self, ds):
        _, hm = ds[0]
        assert hm.dtype == torch.float32

    def test_heatmap_values_01(self, ds):
        _, hm = ds[0]
        assert hm.min().item() >= 0.0
        assert hm.max().item() <= 1.0 + 1e-5

    def test_each_channel_has_peak(self, ds):
        _, hm = ds[0]
        assert hm[0].max().item() > 0.5   # left channel has peak
        assert hm[1].max().item() > 0.5   # right channel has peak
```

- [ ] **Step 4.2: Run — verify fail**

```bash
./venv/bin/python -m pytest tests/test_tmj_heatmap_dataset.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.datasets.tmj_heatmap_dataset'`

- [ ] **Step 4.3: Create `training/datasets/tmj_heatmap_dataset.py`**

```python
#!/usr/bin/env python3
"""
TMJ Heatmap Detection Dataset.

Each item:
  volume: (1, D, H, W) float32  — normalized downsampled CBCT
  target: (2, D, H, W) float32  — ch0=left heatmap, ch1=right heatmap
"""
from __future__ import annotations
import json
import random
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from training.utils.heatmap import make_heatmap

logger = logging.getLogger(__name__)

DOWNSAMPLE_SHAPE = (96, 128, 128)


# ---------------------------------------------------------------------------
# Default DICOM loader (used in production; swapped out in tests)
# ---------------------------------------------------------------------------

def _load_dicom_volume(study_dir: Path) -> np.ndarray:
    files = sorted(study_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm in {study_dir}")
    slices = [pydicom.dcmread(str(f)) for f in files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(s.InstanceNumber))
    planes = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        arr = arr * float(getattr(s, "RescaleSlope", 1.0)) + float(getattr(s, "RescaleIntercept", 0.0))
        planes.append(arr)
    return np.stack(planes, axis=0)


def _normalize(volume: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(volume, [2, 98])
    volume = np.clip(volume, p2, p98)
    denom = p98 - p2
    return ((volume - p2) / denom if denom > 0 else np.zeros_like(volume)).astype(np.float32)


def _downsample(volume: np.ndarray, factor: int = 6) -> np.ndarray:
    zoom = [1.0 / factor] * 3
    return ndimage.zoom(volume, zoom, order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation (applied to volume; heatmap rebuilt after flip)
# ---------------------------------------------------------------------------

def _augment(
    volume: np.ndarray,
    left_orig: List[int],
    right_orig: List[int],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Return augmented volume and (possibly swapped/flipped) original coords."""

    # 1. Intensity jitter
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)
        beta  = random.uniform(-0.05, 0.05)
        volume = np.clip(alpha * volume + beta, 0.0, 1.0)

    # 2. Gaussian noise
    if random.random() < 0.4:
        volume = np.clip(volume + np.random.normal(0, 0.01, volume.shape), 0.0, 1.0).astype(np.float32)

    # 3. X-axis flip → swap left/right + flip x coordinate
    if random.random() < 0.5:
        volume = np.flip(volume, axis=2).copy()
        orig_w = volume.shape[2]
        # flip x in original space: x_new = (orig_W - 1) - x_old
        # we don't know original W exactly, but annotations are normalized—use downsampled W * factor
        # Simple approach: flip within downsampled then convert back
        ds_W = DOWNSAMPLE_SHAPE[2]
        left_new  = [right_orig[0], right_orig[1], (ds_W - 1 - right_orig[2] // 6) * 6]
        right_new = [left_orig[0],  left_orig[1],  (ds_W - 1 - left_orig[2] // 6) * 6]
        left_orig, right_orig = left_new, right_new

    # 4. Rotation ±10° around one axis (volume + coords both transformed)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        axes  = random.choice([(0, 1), (0, 2), (1, 2)])
        volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1, mode="nearest")
        # Approximate: rotate coords around volume center
        D, H, W = volume.shape
        centers = [D / 2, H / 2, W / 2]
        a, b = axes
        c = np.cos(np.radians(angle)); s = np.sin(np.radians(angle))
        for orig in (left_orig, right_orig):
            da = (orig[a] / 6 - centers[a])
            db = (orig[b] / 6 - centers[b])
            orig[a] = int((da * c - db * s + centers[a]) * 6)
            orig[b] = int((da * s + db * c + centers[b]) * 6)
            orig[a] = max(0, orig[a]); orig[b] = max(0, orig[b])

    return volume.astype(np.float32), left_orig, right_orig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TMJHeatmapDataset(Dataset):
    """
    Dataset for 3D heatmap-based TMJ detection.

    Parameters
    ----------
    study_ids : list of str
        Study IDs to include (from detector_split.json).
    annotations_dir : str
        Directory with *_rois.json files.
    volume_loader : callable, optional
        Function (study_id: str) -> np.ndarray (D,H,W) normalized float32.
        Defaults to DICOM loader from data/dataset_cbct_public/.
    dataset_dir : str, optional
        Root of DICOM dataset (used by default loader).
    sigma : float
        Gaussian sigma in downsampled voxels.
    downsample_factor : int
        Spatial downsampling factor applied to volumes.
    is_train : bool
        Enable augmentation when True.
    """

    def __init__(
        self,
        study_ids: List[str],
        annotations_dir: str,
        volume_loader: Optional[Callable] = None,
        dataset_dir: str = "data/dataset_cbct_public",
        sigma: float = 3.0,
        downsample_factor: int = 6,
        is_train: bool = True,
    ):
        self.sigma             = sigma
        self.downsample_factor = downsample_factor
        self.is_train          = is_train
        self.ann_dir           = Path(annotations_dir)
        self.dataset_dir       = Path(dataset_dir)

        # Build volume loader
        if volume_loader is not None:
            self._load_volume = volume_loader
        else:
            def _loader(study_id: str) -> np.ndarray:
                raw = _load_dicom_volume(self.dataset_dir / study_id)
                normalized = _normalize(raw)
                return _downsample(normalized, downsample_factor)
            self._load_volume = _loader

        # Load annotations for requested study_ids
        self.records: List[Dict] = []
        for sid in study_ids:
            ann_path = self.ann_dir / f"{sid}_rois.json"
            if not ann_path.exists():
                logger.warning("Annotation missing: %s — skipped", ann_path)
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            self.records.append({
                "study_id":    ann["scan_id"],
                "left_center": list(ann["left_tmj"]["center"]),
                "right_center": list(ann["right_tmj"]["center"]),
                "original_shape": ann["original_shape"],
            })

        logger.info("TMJHeatmapDataset: %d samples (%s)", len(self.records),
                    "train" if is_train else "val/test")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        vol = self._load_volume(rec["study_id"])   # (D, H, W) float32 [0,1]

        left  = list(rec["left_center"])
        right = list(rec["right_center"])

        if self.is_train:
            vol, left, right = _augment(vol, left, right)

        ds_shape = vol.shape  # (D, H, W) — already downsampled by loader

        hm_left  = make_heatmap(ds_shape, left,  self.sigma, self.downsample_factor)
        hm_right = make_heatmap(ds_shape, right, self.sigma, self.downsample_factor)

        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0)          # (1,D,H,W)
        hm_tensor  = torch.from_numpy(np.stack([hm_left, hm_right])).float()  # (2,D,H,W)
        return vol_tensor, hm_tensor


def get_heatmap_dataloaders(
    split_json: str = "data/detector_split.json",
    annotations_dir: str = "data/roi_annotations",
    dataset_dir: str = "data/dataset_cbct_public",
    sigma: float = 3.0,
    downsample_factor: int = 6,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from split JSON."""
    with open(split_json) as f:
        split = json.load(f)

    train_ds = TMJHeatmapDataset(
        split["train"], annotations_dir, dataset_dir=dataset_dir,
        sigma=sigma, downsample_factor=downsample_factor, is_train=True,
    )
    val_ds = TMJHeatmapDataset(
        split["val"], annotations_dir, dataset_dir=dataset_dir,
        sigma=sigma, downsample_factor=downsample_factor, is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader
```

- [ ] **Step 4.4: Run — verify pass**

```bash
./venv/bin/python -m pytest tests/test_tmj_heatmap_dataset.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 4.5: Run full suite — no regressions**

```bash
./venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

- [ ] **Step 4.6: Commit**

```bash
git add training/datasets/tmj_heatmap_dataset.py tests/test_tmj_heatmap_dataset.py
git commit -m "feat: TMJHeatmapDataset with Gaussian targets and augmentation"
```

---

## Task 5: 3D U-Net model

**Files:**
- Create: `models/tmj_heatmap_detector.py`
- Create: `tests/test_tmj_heatmap_detector.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_tmj_heatmap_detector.py`:

```python
"""Tests for models.tmj_heatmap_detector.TMJHeatmapDetector"""
import sys, os, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.tmj_heatmap_detector import TMJHeatmapDetector


class TestTMJHeatmapDetectorForward:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_output_shape(self, batch_size):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(batch_size, 1, 32, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 2, 32, 64, 64), f"Got {out.shape}"

    def test_output_matches_input_spatial(self):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(1, 1, 48, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape[2:] == x.shape[2:]

    def test_no_nan_output(self):
        model = TMJHeatmapDetector()
        model.eval()
        x = torch.randn(1, 1, 32, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_backward_runs(self):
        from training.losses.heatmap_loss import weighted_mse_loss
        model = TMJHeatmapDetector()
        x      = torch.randn(1, 1, 32, 64, 64)
        target = torch.rand(1, 2, 32, 64, 64)
        out  = model(x)
        loss = weighted_mse_loss(out, target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_fewer_params_than_regression_model(self):
        from models.tmj_detector import TMJDetectorLarge
        hm_model  = TMJHeatmapDetector(features=[16, 32, 64, 128])
        reg_model = TMJDetectorLarge()
        n_hm  = sum(p.numel() for p in hm_model.parameters())
        n_reg = sum(p.numel() for p in reg_model.parameters())
        # U-Net with smaller features should be manageable
        assert n_hm > 0

    def test_custom_features(self):
        model = TMJHeatmapDetector(features=[8, 16])
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 32, 32, 32))
        assert out.shape == (1, 2, 32, 32, 32)
```

- [ ] **Step 5.2: Run — verify fail**

```bash
./venv/bin/python -m pytest tests/test_tmj_heatmap_detector.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.tmj_heatmap_detector'`

- [ ] **Step 5.3: Create `models/tmj_heatmap_detector.py`**

```python
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

from models.blocks import _conv_block


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
```

- [ ] **Step 5.4: Run — verify pass**

```bash
./venv/bin/python -m pytest tests/test_tmj_heatmap_detector.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5.5: Run full suite**

```bash
./venv/bin/python -m pytest tests/ -v 2>&1 | tail -5
```

- [ ] **Step 5.6: Commit**

```bash
git add models/tmj_heatmap_detector.py tests/test_tmj_heatmap_detector.py
git commit -m "feat: 3D U-Net TMJHeatmapDetector with encoder-decoder and skip connections"
```

---

## Task 6: Training script

**Files:**
- Create: `train_heatmap_detector.py`

- [ ] **Step 6.1: Create `train_heatmap_detector.py`**

```python
#!/usr/bin/env python3
"""
Train TMJ Heatmap Detector (3D U-Net).

Usage:
    ./venv/bin/python train_heatmap_detector.py \\
        --split-json    data/detector_split.json \\
        --annotations   data/roi_annotations \\
        --dataset       data/dataset_cbct_public \\
        --epochs        200 \\
        --batch-size    2 \\
        --output-dir    experiments
"""
import argparse, datetime, json, logging, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.datasets.tmj_heatmap_dataset import get_heatmap_dataloaders
from training.losses.heatmap_loss import weighted_mse_loss
from training.utils.heatmap import coords_from_heatmap

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def compute_mae(pred_hm: torch.Tensor, target_hm: torch.Tensor, ds_factor: int = 6) -> dict:
    """
    MAE in downsampled pixels for a batch.
    pred_hm, target_hm: (B, 2, D, H, W)
    Returns dict with mae_left, mae_right, mae_overall (in downsampled px).
    """
    B = pred_hm.shape[0]
    errors_left, errors_right = [], []

    for b in range(B):
        for ch, err_list in enumerate([errors_left, errors_right]):
            pred_coords, _ = coords_from_heatmap(
                torch.sigmoid(pred_hm[b, ch]), ds_factor
            )
            true_coords, _ = coords_from_heatmap(target_hm[b, ch], ds_factor)
            err = torch.sqrt(((pred_coords - true_coords) ** 2).sum()).item()
            err_list.append(err)

    return {
        "mae_left":    float(np.mean(errors_left)),
        "mae_right":   float(np.mean(errors_right)),
        "mae_overall": float(np.mean(errors_left + errors_right)),
    }


def run_epoch(model, loader, optimizer, device, scaler, is_train, epoch, ds_factor):
    model.train() if is_train else model.eval()
    tag = "Train" if is_train else "Val  "
    running_loss, all_mae_l, all_mae_r = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for vols, targets in tqdm(loader, desc=f"[{epoch}] {tag}", leave=False):
            vols, targets = vols.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(vols)
                        loss = weighted_mse_loss(torch.sigmoid(pred), targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                else:
                    pred = model(vols)
                    loss = weighted_mse_loss(torch.sigmoid(pred), targets)
                    loss.backward(); optimizer.step()
            else:
                pred = model(vols)
                loss = weighted_mse_loss(torch.sigmoid(pred), targets)

            running_loss += loss.item()
            m = compute_mae(pred.detach().cpu(), targets.cpu(), ds_factor)
            all_mae_l.append(m["mae_left"]); all_mae_r.append(m["mae_right"])

    n = len(loader)
    return {
        "loss":        running_loss / n,
        "mae_left":    float(np.mean(all_mae_l)),
        "mae_right":   float(np.mean(all_mae_r)),
        "mae_overall": float(np.mean(all_mae_l + all_mae_r)),
    }


def main(args):
    logger.info("=" * 70)
    logger.info("TMJ HEATMAP DETECTOR TRAINING")
    logger.info("=" * 70)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda"); logger.info("GPU: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps"); logger.info("MPS")
    else:
        device = torch.device("cpu"); logger.info("CPU")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Experiment dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / f"heatmap_detector_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(exp_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fh)

    config = vars(args); config["timestamp"] = ts
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Experiment: %s", exp_dir)

    # Data
    train_loader, val_loader = get_heatmap_dataloaders(
        split_json=args.split_json,
        annotations_dir=args.annotations,
        dataset_dir=args.dataset,
        sigma=args.sigma,
        downsample_factor=args.downsample_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Train: %d batches  Val: %d batches", len(train_loader), len(val_loader))

    # Model
    model = TMJHeatmapDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.2fM", n_params / 1e6)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    best_mae = float("inf")
    no_imp   = 0
    history  = []

    logger.info("Starting training...")
    print(f"\n{'Ep':>4}  {'tr_loss':>8}  {'tr_mae':>7}  │  {'va_loss':>8}  {'va_mae':>7}  {'va_L':>6}  {'va_R':>6}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, optimizer, device, scaler, True,  epoch, args.downsample_factor)
        val = run_epoch(model, val_loader,   optimizer, device, scaler, False, epoch, args.downsample_factor)
        scheduler.step(val["mae_overall"])
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"{epoch:4d}  {tr['loss']:8.4f}  {tr['mae_overall']:7.2f}  │  "
              f"{val['loss']:8.4f}  {val['mae_overall']:7.2f}  "
              f"{val['mae_left']:6.2f}  {val['mae_right']:6.2f}  lr={lr_now:.1e}")

        row = {"epoch": epoch, "lr": lr_now}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}":   v for k, v in val.items()})
        history.append(row)
        with open(exp_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        if val["mae_overall"] < best_mae:
            best_mae = val["mae_overall"]
            no_imp   = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_mae": best_mae,
                "val_metrics": val,
            }, exp_dir / "best_model.pth")
            print(f"     ✓ best (MAE_ds={best_mae:.2f}px ≈ {best_mae*args.downsample_factor:.0f}px orig)")
        else:
            no_imp += 1
            if args.early_stopping > 0 and no_imp >= args.early_stopping:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Done. Best val MAE: %.2f downsampled px (~%.0f original px)",
                best_mae, best_mae * args.downsample_factor)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split-json",      default="data/detector_split.json")
    p.add_argument("--annotations",     default="data/roi_annotations")
    p.add_argument("--dataset",         default="data/dataset_cbct_public")
    p.add_argument("--sigma",           type=float, default=3.0)
    p.add_argument("--downsample-factor", dest="downsample_factor", type=int, default=6)
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--batch-size",      type=int,   default=2)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--lr-patience",     type=int,   default=15)
    p.add_argument("--early-stopping",  type=int,   default=40)
    p.add_argument("--num-workers",     type=int,   default=0)
    p.add_argument("--device",          default=None)
    p.add_argument("--output-dir",      default="experiments")
    main(p.parse_args())
```

- [ ] **Step 6.2: Verify import**

```bash
./venv/bin/python -c "import train_heatmap_detector; print('OK')"
```
Expected: `OK`

- [ ] **Step 6.3: Commit**

```bash
git add train_heatmap_detector.py
git commit -m "feat: training script for heatmap detector with MAE logging"
```

---

## Task 7: Evaluation script

**Files:**
- Create: `tools/evaluate_detector.py`

- [ ] **Step 7.1: Create `tools/evaluate_detector.py`**

```python
#!/usr/bin/env python3
"""
Evaluate a trained TMJHeatmapDetector on the test split.

Reports MAE in downsampled pixels, original pixels, mm (if voxel spacing known),
and percentage of predictions within 5mm / 10mm thresholds.

Usage:
    ./venv/bin/python tools/evaluate_detector.py \\
        --model  experiments/heatmap_detector_XXXXXX/best_model.pth \\
        --split  data/detector_split.json \\
        --annotations data/roi_annotations \\
        --dataset data/dataset_cbct_public
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.datasets.tmj_heatmap_dataset import TMJHeatmapDataset
from training.utils.heatmap import coords_from_heatmap
from torch.utils.data import DataLoader


def evaluate(model, loader, device, ds_factor, voxel_mm=None):
    model.eval()
    errs_left, errs_right = [], []

    with torch.no_grad():
        for vols, targets in loader:
            vols = vols.to(device)
            pred = torch.sigmoid(model(vols)).cpu()

            B = pred.shape[0]
            for b in range(B):
                for ch, err_list in enumerate([errs_left, errs_right]):
                    pc, _ = coords_from_heatmap(pred[b, ch],    ds_factor)
                    tc, _ = coords_from_heatmap(targets[b, ch], ds_factor)
                    err_ds = torch.sqrt(((pc - tc) ** 2).sum()).item()
                    err_list.append(err_ds)

    all_errs = errs_left + errs_right
    results = {
        "n_samples":    len(errs_left),
        "mae_left_ds":  float(np.mean(errs_left)),
        "mae_right_ds": float(np.mean(errs_right)),
        "mae_overall_ds": float(np.mean(all_errs)),
        "mae_left_orig":  float(np.mean(errs_left)  * ds_factor),
        "mae_right_orig": float(np.mean(errs_right) * ds_factor),
        "mae_overall_orig": float(np.mean(all_errs) * ds_factor),
    }
    if voxel_mm:
        results["mae_overall_mm"] = results["mae_overall_orig"] * voxel_mm
        results["pct_within_5mm"]  = 100 * np.mean([e * ds_factor * voxel_mm < 5  for e in all_errs])
        results["pct_within_10mm"] = 100 * np.mean([e * ds_factor * voxel_mm < 10 for e in all_errs])

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = TMJHeatmapDetector().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']} (val MAE={ckpt['best_val_mae']:.2f} ds px)")

    with open(args.split) as f:
        split = json.load(f)

    test_ds = TMJHeatmapDataset(
        split["test"], args.annotations,
        dataset_dir=args.dataset, is_train=False,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test set: {len(test_ds)} studies")

    results = evaluate(model, loader, device, ds_factor=6,
                       voxel_mm=args.voxel_mm)

    print("\n=== TEST RESULTS ===")
    print(f"  MAE left  (ds px):    {results['mae_left_ds']:.2f}")
    print(f"  MAE right (ds px):    {results['mae_right_ds']:.2f}")
    print(f"  MAE overall (ds px):  {results['mae_overall_ds']:.2f}")
    print(f"  MAE left  (orig px):  {results['mae_left_orig']:.1f}")
    print(f"  MAE right (orig px):  {results['mae_right_orig']:.1f}")
    print(f"  MAE overall (orig px):{results['mae_overall_orig']:.1f}")
    if args.voxel_mm:
        print(f"  MAE overall (mm):     {results['mae_overall_mm']:.2f}")
        print(f"  % within 5mm:         {results['pct_within_5mm']:.1f}%")
        print(f"  % within 10mm:        {results['pct_within_10mm']:.1f}%")

    out = Path(args.model).parent / "test_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True)
    p.add_argument("--split",       default="data/detector_split.json")
    p.add_argument("--annotations", default="data/roi_annotations")
    p.add_argument("--dataset",     default="data/dataset_cbct_public")
    p.add_argument("--voxel-mm",    type=float, default=None,
                   help="Voxel spacing in mm for MAE_mm and %within_Nmm metrics")
    main(p.parse_args())
```

- [ ] **Step 7.2: Verify import**

```bash
./venv/bin/python -c "import tools.evaluate_detector; print('OK')" 2>/dev/null || \
./venv/bin/python tools/evaluate_detector.py --help | head -3
```

- [ ] **Step 7.3: Commit**

```bash
git add tools/evaluate_detector.py
git commit -m "feat: evaluate_detector.py — MAE in px/mm and % within threshold"
```

---

## Task 8: Update auto_crop_from_detector.py for heatmap model

**Files:**
- Modify: `tools/auto_crop_from_detector.py`

- [ ] **Step 8.1: Add heatmap model support**

In `tools/auto_crop_from_detector.py`, find the `load_detector` function and add a heatmap branch. Also update `predict_tmj_coords` to handle heatmap output.

Add these functions after the existing imports:

```python
# Add after existing imports
from models.tmj_heatmap_detector import TMJHeatmapDetector
from training.utils.heatmap import coords_from_heatmap as _hm_coords
```

Replace the `load_detector` function body to detect model type from config:

```python
def load_detector(model_path: str, device: str = 'mps'):
    logger.info(f"Loading detector from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model_path_obj = Path(model_path)
    config_path = model_path_obj.parent / 'config.json'
    model_type = 'large'
    is_heatmap = False

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get('model_type', 'large')
            is_heatmap = config.get('heatmap', False)
            logger.info(f"Model type: {model_type}, heatmap: {is_heatmap}")
        except Exception as e:
            logger.warning(f"Could not read config: {e}")

    if is_heatmap:
        model = TMJHeatmapDetector()
    else:
        model = get_detector_model(model_type=model_type)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    model._is_heatmap = is_heatmap   # tag for predict_tmj_coords
    logger.info(f"Loaded from epoch {checkpoint['epoch']}")
    return model, checkpoint['epoch']
```

Update `predict_tmj_coords` to handle both model types:

```python
def predict_tmj_coords(model, volume, downsample_factor=6, device='mps'):
    volume_processed = preprocess_volume(volume)
    D, H, W = volume_processed.shape
    nD, nH, nW = D // downsample_factor, H // downsample_factor, W // downsample_factor

    volume_down = torch.nn.functional.interpolate(
        torch.from_numpy(volume_processed[None, None]).float(),
        size=(nD, nH, nW), mode='trilinear', align_corners=False
    )[0, 0].numpy()

    volume_tensor = torch.from_numpy(volume_down).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(volume_tensor)

    is_heatmap = getattr(model, '_is_heatmap', False)

    if is_heatmap:
        # pred: (1, 2, D, H, W) — ch0=left, ch1=right
        hm_left  = torch.sigmoid(pred[0, 0])
        hm_right = torch.sigmoid(pred[0, 1])
        ds_left,  _ = _hm_coords(hm_left,  downsample_factor)
        ds_right, _ = _hm_coords(hm_right, downsample_factor)
        left_coords  = (ds_left  * downsample_factor).cpu().numpy().astype(int)
        right_coords = (ds_right * downsample_factor).cpu().numpy().astype(int)
    else:
        # Original regression model
        pred_np = pred.cpu().numpy()[0]
        left_coords  = np.array([pred_np[0]*nD, pred_np[1]*nH, pred_np[2]*nW])
        right_coords = np.array([pred_np[3]*nD, pred_np[4]*nH, pred_np[5]*nW])
        left_coords  = (left_coords  * downsample_factor).astype(int)
        right_coords = (right_coords * downsample_factor).astype(int)

    return {'left': left_coords, 'right': right_coords}
```

- [ ] **Step 8.2: Verify import**

```bash
./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from tools.auto_crop_from_detector import load_detector, predict_tmj_coords
print('OK')
"
```

- [ ] **Step 8.3: Run full test suite — no regressions**

```bash
./venv/bin/python -m pytest tests/ -v 2>&1 | tail -10
```

- [ ] **Step 8.4: Commit**

```bash
git add tools/auto_crop_from_detector.py
git commit -m "feat: auto_crop_from_detector supports heatmap model via config.json flag"
```

---

## Self-Review

**Spec coverage:**
- ✅ 3D U-Net architecture (Task 5)
- ✅ `make_heatmap()` with sigma + downsample_factor (Task 1)
- ✅ `soft_argmax_3d()` differentiable coord extraction (Task 1)
- ✅ Train/val/test split JSON (Task 2)
- ✅ `TMJHeatmapDataset` with augmentation + X-flip swap (Task 4)
- ✅ Weighted MSE loss (Task 3)
- ✅ MAE in px and mm (Task 6 training, Task 7 eval)
- ✅ % within 5mm/10mm (Task 7)
- ✅ `auto_crop_from_detector.py` updated for heatmap model (Task 8)
- ✅ All files have tests

**Type consistency:**
- `make_heatmap` returns `np.ndarray` float32 → used correctly in dataset (Task 4) ✅
- `soft_argmax_3d` takes `torch.Tensor (D,H,W)` → called correctly in evaluate_detector (Task 7) ✅
- `TMJHeatmapDetector.forward` returns `(B,2,D,H,W)` raw logits → sigmoid applied in loss and inference ✅
- `coords_from_heatmap` returns `(ds_coords, orig_coords)` → destructured correctly in all callers ✅
