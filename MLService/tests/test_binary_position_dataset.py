"""
Tests for TMJBinaryPositionDataset and get_binary_position_dataloaders.

All tests use synthetic NIfTI files — no real DICOM data needed.
"""

import json
import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_nifti(path, shape=(128, 128, 128)):
    """Write a synthetic NIfTI file with random float32 data."""
    import nibabel as nib
    arr = np.random.rand(*shape).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


@pytest.fixture()
def binary_records(tmp_path):
    """Two synthetic binary records (left crops) with real NIfTI files."""
    crop_dir = tmp_path / "crops"
    for study_id in ("study_0001", "study_0002"):
        study_dir = crop_dir / study_id
        study_dir.mkdir(parents=True)
        _make_nifti(study_dir / f"{study_id}_left.nii.gz")
        _make_nifti(study_dir / f"{study_id}_right.nii.gz")

    records = [
        {
            "study_id": "study_0001",
            "patient_name": "Patient A",
            "side": "left",
            "sag": 0,
            "fr": 1,
            "crop_path": str(crop_dir / "study_0001" / "study_0001_left.nii.gz"),
        },
        {
            "study_id": "study_0002",
            "patient_name": "Patient B",
            "side": "right",
            "sag": 1,
            "fr": 0,
            "crop_path": str(crop_dir / "study_0002" / "study_0002_right.nii.gz"),
        },
    ]
    return records


class TestTMJBinaryPositionDataset:
    def test_len(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        assert len(ds) == 2

    def test_volume_tensor_shape(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        volume, labels = ds[0]
        assert volume.shape == (1, 128, 128, 128), f"Got {volume.shape}"

    def test_volume_values_in_01(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        volume, _ = ds[0]
        assert volume.min().item() >= 0.0
        assert volume.max().item() <= 1.0

    def test_labels_shape_and_type(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        _, labels = ds[0]
        assert labels.shape == (2,), f"Expected (2,), got {labels.shape}"
        assert labels.dtype == torch.long

    def test_labels_values_binary(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        _, labels = ds[0]
        assert labels[0].item() in (0, 1), "sag label must be binary"
        assert labels[1].item() in (0, 1), "fr label must be binary"

    def test_correct_labels_returned(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        # First record: sag=0, fr=1
        _, labels = ds[0]
        assert labels[0].item() == 0
        assert labels[1].item() == 1

    def test_volume_is_float32(self, binary_records):
        from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset
        ds = TMJBinaryPositionDataset(binary_records)
        volume, _ = ds[0]
        assert volume.dtype == torch.float32
