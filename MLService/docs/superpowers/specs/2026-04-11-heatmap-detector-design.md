# TMJ Heatmap Detector — Design Spec

## Goal

Replace coordinate regression detector (MAE ~24px downsampled = ~21-29mm) with heatmap-based 3D U-Net detector. Annotate all 107 scans → train on ~85, eval on ~15, test on ~7.

## Problem with current approach

- Only 37 annotated scans (31 train / 6 val)
- MSE on 6 scalar coordinates → single-point gradient signal per sample
- MAE=24px in 6× downsampled space ≈ 144px original ≈ 21-29mm — larger than the condyle itself (~10-15mm)
- No test hold-out set

## Architecture

**Model:** 3D U-Net with skip connections.

```
Input:  (B, 1, 96, 128, 128)  — 6× downsampled CBCT, normalized [0,1]
Output: (B, 2, 96, 128, 128)  — 2 heatmap channels: ch0=left, ch1=right
```

**Encoder** (same conv blocks as current detector):
```
(1→32) → MaxPool → (32→64) → MaxPool → (64→128) → MaxPool → (128→256) → MaxPool → (256→512)
```

**Decoder** (new — adds skip connections):
```
Upsample + skip → Conv(512+256→256) → Upsample + skip → Conv(256+128→128)
→ Upsample + skip → Conv(128+64→64)  → Upsample + skip → Conv(64+32→32)
→ Conv(32→2, kernel=1)  [output heatmaps]
```

Each conv block: Conv3d → BN → ReLU → Conv3d → BN → ReLU.

File: `models/tmj_heatmap_detector.py`

## Dataset

### Annotations

Format unchanged — same JSON files as existing 37 annotations:

```json
{
  "study_id": "study_0042",
  "original_shape": [576, 768, 768],
  "left_tmj":  {"center": [z, y, x], "confidence": "manual"},
  "right_tmj": {"center": [z, y, x], "confidence": "manual"}
}
```

Directory: `data/roi_annotations/` (37 exist → annotate remaining 70 → total 107).

Annotation tool: `tools/roi_annotation_tool.py` (unchanged).

### Split

Strict by `study_id` (no patient-level concern here — each study independent):

| Split | Count | % | Purpose |
|-------|-------|---|---------|
| Train | 85 | 80% | Gradient updates |
| Val   | 15 | 14% | Early stopping, LR schedule |
| Test  |  7 |  6% | Final evaluation only, never touched during training |

Deterministic seed=42. Split saved to `data/detector_split.json`.

### Heatmap generation

At training time, generate Gaussian 3D heatmap from annotation coordinates:

```python
def make_heatmap(shape, center_zyx, sigma=3.0):
    """
    shape: (D, H, W) in downsampled space
    center_zyx: (z, y, x) in ORIGINAL space → divide by downsample_factor=6
    sigma: Gaussian sigma in downsampled voxels
    Returns: float32 array in [0, 1], peak=1 at center
    """
    center = np.array(center_zyx) / 6.0  # convert to downsampled coords
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_sq = (z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2
    return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)
```

Target tensor: `(2, D, H, W)` — stack left and right heatmaps.

### Augmentation

All augmentations applied **simultaneously** to volume AND target heatmaps:

| Op | P | Notes |
|----|---|-------|
| X-axis flip | 0.5 | Swap ch0↔ch1 in target (left↔right) |
| Intensity jitter ±10% | 0.7 | Volume only (not heatmap) |
| Gaussian noise σ=0.01 | 0.4 | Volume only |
| Random rotation ±15° | 0.5 | Rotate both volume and heatmap with same angle |

File: `training/datasets/tmj_heatmap_dataset.py`

## Loss

**Weighted MSE** to balance sparse positive regions vs dense background:

```python
pos_weight = 10.0

def heatmap_loss(pred, target):
    weight = 1.0 + pos_weight * target  # higher weight near peaks
    return torch.mean(weight * (pred - target) ** 2)
```

Alternative if unstable: switch to BCE with sigmoid output.

## Coordinate extraction (inference)

**Soft-argmax** — differentiable, more precise than hard argmax:

```python
def soft_argmax_3d(heatmap):
    """heatmap: (D, H, W) → (z, y, x) in downsampled voxels"""
    heatmap = heatmap / (heatmap.sum() + 1e-8)
    z = (heatmap * z_grid).sum()
    y = (heatmap * y_grid).sum()
    x = (heatmap * x_grid).sum()
    return torch.stack([z, y, x])
```

At inference: multiply by `downsample_factor * voxel_spacing_mm` to get mm.

## Metrics

| Metric | Description |
|--------|-------------|
| MAE_px | Mean absolute error in downsampled pixels (compare with current 24px) |
| MAE_mm | MAE_px × 6 × voxel_spacing (target: < 5mm) |
| MAE_left / MAE_right | Per-joint breakdown |
| MAE_z / MAE_y / MAE_x | Per-axis breakdown |
| % within 5mm | Fraction of predictions within 5mm of ground truth |
| % within 10mm | Fraction within 10mm |

Voxel spacing: read from DICOM `PixelSpacing` and `SliceThickness` tags.

## Training config

| Param | Value |
|-------|-------|
| Optimizer | Adam |
| LR | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=15, factor=0.5) |
| Batch size | 2 |
| Epochs | 200 |
| Early stopping | 40 epochs no val improvement |
| Sigma (Gaussian) | 3.0 voxels |
| pos_weight | 10.0 |

## Files created / modified

| Action | File | What |
|--------|------|------|
| Create | `models/tmj_heatmap_detector.py` | 3D U-Net model |
| Create | `training/datasets/tmj_heatmap_dataset.py` | Dataset + heatmap generation + augmentation |
| Create | `train_heatmap_detector.py` | Training script |
| Create | `tools/evaluate_detector.py` | Evaluation script: MAE in px and mm, % within threshold |
| Create | `data/detector_split.json` | Deterministic train/val/test split |
| Modify | `tools/auto_crop_from_detector.py` | Load new heatmap model at inference |

Existing `tools/roi_annotation_tool.py` and annotation JSON format: **unchanged**.

## Success criteria

- MAE_mm < 10mm on val set (current: ~21-29mm)
- % within 10mm > 80% on val set
- No complete failures (> 30mm) on test set
- Both left and right condyles detected correctly on visualize_detection.py output
