"""Tests for TMJHeatmapDataset — synthetic tests, no DICOM needed."""
import sys, os, json, pytest
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.datasets.tmj_heatmap_dataset import TMJHeatmapDataset


def _make_annotation(ann_dir, study_id, left=(288,240,180), right=(288,240,560)):
    ann = {
        "scan_id": study_id,
        "original_shape": [576, 768, 768],
        "left_tmj":  {"center": list(left),  "confidence": "manual"},
        "right_tmj": {"center": list(right), "confidence": "manual"},
    }
    (ann_dir / f"{study_id}_rois.json").write_text(json.dumps(ann))


class TestTMJHeatmapDataset:
    @pytest.fixture()
    def ds(self, tmp_path):
        ann_dir = tmp_path / "ann"
        ann_dir.mkdir()
        for sid in ("study_0001", "study_0002"):
            _make_annotation(ann_dir, sid)
        vol = np.random.rand(96, 128, 128).astype(np.float32)
        def mock_load(study_id):
            return vol
        return TMJHeatmapDataset(
            study_ids=["study_0001", "study_0002"],
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
        assert hm[0].max().item() > 0.5
        assert hm[1].max().item() > 0.5
