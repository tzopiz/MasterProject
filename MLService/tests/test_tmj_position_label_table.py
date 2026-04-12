"""
Unit tests for training.tmj_position_label_table

Tests are self-contained (no disk I/O) — all data is synthetic.
"""

import json
import os
import sys

import pytest

# Make sure the MLService root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.tmj_position_label_table import (
    build_index,
    map_frontal,
    map_sagittal,
    split_by_patient,
)

# ---------------------------------------------------------------------------
# Code mapping
# ---------------------------------------------------------------------------


class TestMapSagittal:
    def test_code_1_maps_to_0(self):
        assert map_sagittal(1) == 0

    def test_code_2_maps_to_1(self):
        assert map_sagittal(2) == 1

    def test_code_3_maps_to_2(self):
        assert map_sagittal(3) == 2

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            map_sagittal(4)

    def test_invalid_code_zero_raises(self):
        with pytest.raises(ValueError):
            map_sagittal(0)


class TestMapFrontal:
    def test_code_4_maps_to_0(self):
        assert map_frontal(4) == 0

    def test_code_5_maps_to_1(self):
        assert map_frontal(5) == 1

    def test_code_6_maps_to_2(self):
        assert map_frontal(6) == 2

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError):
            map_frontal(3)

    def test_invalid_code_7_raises(self):
        with pytest.raises(ValueError):
            map_frontal(7)


# ---------------------------------------------------------------------------
# build_index with synthetic JSON files
# ---------------------------------------------------------------------------

MANIFEST = {
    "warning": "test",
    "studies": [
        {
            "study_id": "study_0001",
            "patient_name": "Иванов Иван Иванович",
        },
        {
            "study_id": "study_0002",
            "patient_name": "Иванов Иван Иванович",  # second scan of same patient
        },
        {
            "study_id": "study_0003",
            "patient_name": "Петрова Анна Сергеевна",
        },
        {
            "study_id": "study_0004",
            "patient_name": "Без меток Пациент",  # not in labels → must be skipped
        },
    ],
}

LABELS = {
    "schema_version": 1,
    "patients": [
        {
            "patient_number": 1,
            "name_raw": "Иванов Иван Иванович",
            "labels": {
                "sagittal": {"right": 1, "left": 2},
                "frontal": {"right": 4, "left": 6},
            },
        },
        {
            "patient_number": 2,
            "name_raw": "Петрова Анна Сергеевна",
            "labels": {
                "sagittal": {"right": 3, "left": 1},
                "frontal": {"right": 5, "left": 4},
            },
        },
    ],
}


@pytest.fixture()
def synthetic_files(tmp_path):
    """Write synthetic manifest + labels JSON files; return their paths."""
    manifest_path = tmp_path / "manifest_private.json"
    labels_path = tmp_path / "tmj_position_labels.json"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    labels_path.write_text(json.dumps(LABELS), encoding="utf-8")

    return str(manifest_path), str(labels_path), str(dataset_root)


class TestBuildIndex:
    def test_matched_records_count(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        # study_0001, study_0002, study_0003 match; study_0004 is skipped
        assert len(records) == 3

    def test_skipped_patient_not_in_records(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        study_ids = [r["study_id"] for r in records]
        assert "study_0004" not in study_ids

    def test_label_mapping_ivanov(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        ivanov = [r for r in records if r["study_id"] == "study_0001"][0]
        assert ivanov["sag_right"] == 0  # code 1 → 0
        assert ivanov["sag_left"] == 1  # code 2 → 1
        assert ivanov["fr_right"] == 0  # code 4 → 0
        assert ivanov["fr_left"] == 2  # code 6 → 2

    def test_label_mapping_petrova(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        petrova = [r for r in records if r["study_id"] == "study_0003"][0]
        assert petrova["sag_right"] == 2  # code 3 → 2
        assert petrova["sag_left"] == 0  # code 1 → 0
        assert petrova["fr_right"] == 1  # code 5 → 1
        assert petrova["fr_left"] == 0  # code 4 → 0

    def test_record_has_required_keys(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        required = {
            "study_id",
            "dicom_dir",
            "patient_name",
            "sag_right",
            "sag_left",
            "fr_right",
            "fr_left",
        }
        for rec in records:
            assert required <= set(rec.keys()), f"Missing keys in {rec}"

    def test_cache_file_written(self, synthetic_files, tmp_path):
        manifest_path, labels_path, dataset_root = synthetic_files
        cache = str(tmp_path / "cache.json")
        records = build_index(manifest_path, labels_path, dataset_root, cache_path=cache)
        import os

        assert os.path.exists(cache)
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
        assert len(cached) == len(records)


# ---------------------------------------------------------------------------
# split_by_patient
# ---------------------------------------------------------------------------


class TestSplitByPatient:
    def test_no_patient_leakage(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        train, val = split_by_patient(records, split_ratio=0.5, seed=0)

        train_patients = {r["patient_name"] for r in train}
        val_patients = {r["patient_name"] for r in val}
        # No patient should appear in both splits
        assert train_patients.isdisjoint(val_patients)

    def test_all_records_covered(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        train, val = split_by_patient(records, split_ratio=0.5)
        assert len(train) + len(val) == len(records)

    def test_same_seed_reproducible(self, synthetic_files):
        manifest_path, labels_path, dataset_root = synthetic_files
        records = build_index(manifest_path, labels_path, dataset_root)
        t1, v1 = split_by_patient(records, seed=99)
        t2, v2 = split_by_patient(records, seed=99)
        assert [r["study_id"] for r in t1] == [r["study_id"] for r in t2]
        assert [r["study_id"] for r in v1] == [r["study_id"] for r in v2]
