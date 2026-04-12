"""
Tests for training.tmj_position_label_table.binarize_labels
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.tmj_position_label_table import binarize_labels, build_index

MANIFEST = {
    "studies": [
        {"study_id": "study_0001", "patient_name": "Patient A"},
        {"study_id": "study_0002", "patient_name": "Patient B"},
    ]
}
LABELS = {
    "schema_version": 1,
    "patients": [
        {
            "patient_number": 1,
            "name_raw": "Patient A",
            "labels": {
                "sagittal": {
                    "right": 1,
                    "left": 2,
                },  # right=central(0), left=non-central(1)
                "frontal": {
                    "right": 4,
                    "left": 6,
                },  # right=central(0), left=non-central(2→1)
            },
        },
        {
            "patient_number": 2,
            "name_raw": "Patient B",
            "labels": {
                "sagittal": {
                    "right": 3,
                    "left": 1,
                },  # right=non-central(2→1), left=central(0)
                "frontal": {
                    "right": 5,
                    "left": 4,
                },  # right=non-central(1→1), left=central(0)
            },
        },
    ],
}


@pytest.fixture()
def records(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "labels.json"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path.write_text(json.dumps(MANIFEST))
    labels_path.write_text(json.dumps(LABELS))
    return build_index(str(manifest_path), str(labels_path), str(dataset_root))


class TestBinarizeLabels:
    def test_output_count_is_double_input(self, records, tmp_path):
        binary = binarize_labels(records, str(tmp_path))
        assert len(binary) == len(records) * 2

    def test_each_record_has_both_sides(self, records, tmp_path):
        binary = binarize_labels(records, str(tmp_path))
        study_sides = [(r["study_id"], r["side"]) for r in binary]
        assert ("study_0001", "left") in study_sides
        assert ("study_0001", "right") in study_sides
        assert ("study_0002", "left") in study_sides
        assert ("study_0002", "right") in study_sides

    def test_central_class_maps_to_0(self, records, tmp_path):
        """sag_right=0 (central) → binary sag=0"""
        binary = binarize_labels(records, str(tmp_path))
        r = next(r for r in binary if r["study_id"] == "study_0001" and r["side"] == "right")
        assert r["sag"] == 0  # original sag_right=0 (central)
        assert r["fr"] == 0  # original fr_right=0 (central)

    def test_non_central_class_1_maps_to_1(self, records, tmp_path):
        """sag_left=1 (anterior) → binary sag=1"""
        binary = binarize_labels(records, str(tmp_path))
        r = next(r for r in binary if r["study_id"] == "study_0001" and r["side"] == "left")
        assert r["sag"] == 1  # original sag_left=1 (non-central)
        assert r["fr"] == 1  # original fr_left=2 (non-central)

    def test_non_central_class_2_maps_to_1(self, records, tmp_path):
        """sag_right=2 (posterior) → binary sag=1"""
        binary = binarize_labels(records, str(tmp_path))
        r = next(r for r in binary if r["study_id"] == "study_0002" and r["side"] == "right")
        assert r["sag"] == 1  # original sag_right=2
        assert r["fr"] == 1  # original fr_right=1 (non-central)

    def test_crop_path_contains_study_and_side(self, records, tmp_path):
        binary = binarize_labels(records, str(tmp_path))
        r = next(r for r in binary if r["study_id"] == "study_0001" and r["side"] == "left")
        p = Path(r["crop_path"])
        assert p.name == "study_0001_left.nii.gz"
        assert p.parent.name == "study_0001"

    def test_patient_name_preserved(self, records, tmp_path):
        binary = binarize_labels(records, str(tmp_path))
        names = {r["patient_name"] for r in binary}
        assert "Patient A" in names
        assert "Patient B" in names

    def test_required_keys_present(self, records, tmp_path):
        binary = binarize_labels(records, str(tmp_path))
        required = {"study_id", "patient_name", "side", "sag", "fr", "crop_path"}
        for r in binary:
            assert required <= set(r.keys())
