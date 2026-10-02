"""Strict intake boundaries: supplied identity, labels, paths and safe diagnostics."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import tmj_position_label_table as table


@pytest.fixture
def intake(tmp_path):
    for name in ("series", "crops"):
        (tmp_path / name).mkdir()
    for side in ("left", "right"):
        (tmp_path / "crops" / f"{side}.nii.gz").write_bytes(b"synthetic")
    data = {
        "schema_version": 1,
        "studies": [
            {
                "source_id": "cohort-a",
                "study_id": "scan-a",
                "patient_id": "p-a",
                "label_record_id": "label-a",
                "label_applicability": "confirmed",
                "series_path": "series",
                "crops": {"left": "crops/left.nii.gz", "right": "crops/right.nii.gz"},
            }
        ],
        "labels": [
            {
                "source_id": "cohort-a",
                "label_record_id": "label-a",
                "patient_id": "p-a",
                "labels": {"sagittal": {"left": 2, "right": 1}},
            }
        ],
    }
    path = tmp_path / "inputs.json"

    def write():
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return data, tmp_path, write


def reject(intake, code, **kwargs):
    _, root, write = intake
    with pytest.raises(table.InputValidationError) as error:
        table.build_canonical_index(write(), root, **kwargs)
    report = error.value.report
    assert report["ready"] is False
    assert code in {item["code"] for item in report["diagnostics"]}
    return report


def test_canonical_sagittal_index_preserves_supplied_ids_and_missing_frontal(intake):
    _, root, write = intake
    assert hasattr(table, "build_canonical_index"), "strict intake API is missing"
    records = table.build_canonical_index(write(), root)
    assert len(records) == 1
    row = records[0]
    assert (row["source_id"], row["study_id"], row["patient_id"]) == ("cohort-a", "scan-a", "p-a")
    assert row["patient_key"] == '["cohort-a","p-a"]'
    assert (row["sag_left"], row["sag_right"]) == (1, 0)
    assert "fr_left" not in row
    assert "patient_name" not in row
    binary = table.binarize_labels(records, str(root / "unused"))
    assert [(r["side"], r["sag"]) for r in binary] == [("left", 1), ("right", 0)]
    assert all("fr" not in r and r["patient_key"] == row["patient_key"] for r in binary)
    assert binary[0]["crop_path"] == str(root / "crops/left.nii.gz")


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, 0, 4, -1, 7])
def test_invalid_sagittal_never_becomes_normal(intake, value):
    intake[0]["labels"][0]["labels"]["sagittal"]["left"] = value
    reject(intake, "invalid_label_code")


@pytest.mark.parametrize("field", ["patient_id", "source_id", "study_id", "label_record_id"])
def test_missing_identity_blocks_intake(intake, field):
    del intake[0]["studies"][0][field]
    reject(intake, "invalid_identity")


def test_unconfirmed_study_label_is_not_inferred(intake):
    intake[0]["studies"][0]["label_applicability"] = "unknown"
    reject(intake, "label_applicability_unconfirmed")


def test_patient_mismatch_blocks_join(intake):
    intake[0]["labels"][0]["patient_id"] = "another-patient"
    reject(intake, "label_patient_mismatch")


def test_missing_label_record_blocks_join(intake):
    intake[0]["studies"][0]["label_record_id"] = "missing"
    reject(intake, "label_not_found")


@pytest.mark.parametrize(
    "collection,code", [("studies", "duplicate_study"), ("labels", "duplicate_label_record")]
)
def test_duplicate_keys_never_overwrite(intake, collection, code):
    intake[0][collection].append(copy.deepcopy(intake[0][collection][0]))
    if collection == "labels":
        intake[0][collection][1]["labels"]["sagittal"]["left"] = 3
    reject(intake, code)


def test_missing_sagittal_side_blocks_intake(intake):
    del intake[0]["labels"][0]["labels"]["sagittal"]["left"]
    reject(intake, "invalid_label_code")


def test_all_planes_requires_frontal_and_validates_optional_frontal(intake):
    reject(intake, "missing_plane", sagittal_only=False)
    intake[0]["labels"][0]["labels"]["frontal"] = {"left": 1, "right": 4}
    reject(intake, "invalid_label_code")
    intake[0]["labels"][0]["labels"]["frontal"]["left"] = 6
    records = table.build_canonical_index(intake[2](), intake[1], sagittal_only=False)
    assert (records[0]["fr_left"], records[0]["fr_right"]) == (2, 0)


@pytest.mark.parametrize(
    "path,code",
    [
        ("../private-name.nii.gz", "path_escape"),
        ("/tmp/private-name.nii.gz", "path_escape"),
        ("crops/missing.nii.gz", "missing_file"),
    ],
)
def test_bad_crop_path_has_no_private_value_in_diagnostics(intake, path, code):
    intake[0]["studies"][0]["crops"]["left"] = path
    report = reject(intake, code)
    assert "private-name" not in json.dumps(report)
    assert str(intake[1]) not in json.dumps(report)


def test_symlink_escape_blocks_intake(intake):
    outside = intake[1].parent / "outside.nii.gz"
    outside.write_bytes(b"outside")
    (intake[1] / "crops/link.nii.gz").symlink_to(outside)
    intake[0]["studies"][0]["crops"]["left"] = "crops/link.nii.gz"
    reject(intake, "path_escape")


def test_crop_only_is_supported_but_uncropped_requires_source_series(intake):
    del intake[0]["studies"][0]["series_path"]
    assert table.build_canonical_index(intake[2](), intake[1])
    reject(intake, "missing_series", require_crops=False)


def test_explicit_missing_series_is_never_replaced(intake):
    intake[0]["studies"][0]["series_path"] = "missing"
    reject(intake, "missing_directory")


def test_absent_crop_side_and_reused_crop_are_rejected(intake):
    del intake[0]["studies"][0]["crops"]["left"]
    reject(intake, "missing_crop")
    intake[0]["studies"][0]["crops"]["left"] = "crops/right.nii.gz"
    reject(intake, "duplicate_input_path")


def test_duplicate_json_key_and_unsupported_schema_are_safe(intake):
    _, root, write = intake
    path = write()
    path.write_text('{"schema_version":1,"schema_version":2,"private-name":0}')
    with pytest.raises(table.InputValidationError) as error:
        table.build_canonical_index(path, root)
    assert error.value.report["diagnostics"][0]["code"] == "duplicate_json_key"
    assert "private-name" not in str(error.value)
    intake[0]["schema_version"] = True
    reject(intake, "unsupported_schema")


def test_unknown_fields_are_rejected_without_echoing_field_names(intake):
    intake[0]["studies"][0]["Patient Real Name"] = "private-name"
    report = reject(intake, "unknown_fields")
    assert "Patient Real Name" not in json.dumps(report)


def test_source_qualified_groups_and_input_order_are_stable(intake):
    data, root, write = intake
    for index in range(1, 12):
        source = "cohort-b" if index == 1 else "cohort-a"
        patient = "p-a" if index < 3 else f"p-{index}"
        study = copy.deepcopy(data["studies"][0])
        study.update(source_id=source, study_id=f"scan-{index}", patient_id=patient)
        for side in ("left", "right"):
            crop = f"crops/{index}-{side}.nii.gz"
            (root / crop).write_bytes(b"synthetic")
            study["crops"][side] = crop
        study["series_path"] = f"series-{index}"
        (root / study["series_path"]).mkdir()
        label = copy.deepcopy(data["labels"][0])
        label.update(source_id=source, patient_id=patient, label_record_id=f"label-{index}")
        label["labels"]["sagittal"] = {"left": 1 if index < 6 else 2, "right": 1}
        if index == 1:
            study["study_id"] = "scan-a"
            label["label_record_id"] = "label-a"
        study["label_record_id"] = label["label_record_id"]
        data["studies"].append(study)
        data["labels"].append(label)
    first = table.build_canonical_index(write(), root)
    data["studies"].reverse()
    data["labels"].reverse()
    assert table.build_canonical_index(write(), root) == first
    binary = table.binarize_labels(first, str(root))
    groups = table.patient_sagittal_strat_labels(binary)
    assert '["cohort-a","p-a"]' in groups and '["cohort-b","p-a"]' in groups
    train, val = table.split_by_patient(binary, split_ratio=0.5)
    assert {r["patient_key"] for r in train}.isdisjoint({r["patient_key"] for r in val})
    for tr, va in table.iter_stratified_group_kfold_indices(binary, n_splits=2):
        assert {binary[i]["patient_key"] for i in tr}.isdisjoint(
            {binary[i]["patient_key"] for i in va}
        )


def test_binarization_rejects_unvalidated_nonzero_and_missing_classes():
    for value in (None, True, -1, 3, "1"):
        row = {"study_id": "s", "patient_name": "legacy", "sag_left": value, "sag_right": 0}
        with pytest.raises(ValueError):
            table.binarize_labels([row], "/unused")


def test_cli_reports_intake_only_without_claiming_training_readiness(intake):
    script = Path(__file__).resolve().parents[1] / "tools/check_training_inputs.py"
    command = [
        sys.executable,
        str(script),
        "--input-json",
        str(intake[2]()),
        "--dataset-root",
        str(intake[1]),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert not report["ready"] and not report["training_ready"]
    assert report["diagnostics"][0]["code"] == "unreadable_roi_passport"
    result = subprocess.run(command + ["--allow-uncropped"], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 0 and report["stage"] == "intake-only"
    assert not report["ready"] and not report["training_ready"]
    intake[0]["labels"][0]["patient_id"] = "Private Person"
    intake[2]()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2 and "Private Person" not in result.stdout + result.stderr


def test_successful_cli_minimizes_supplied_identifiers(intake):
    data, root, write = intake
    for row in data["studies"] + data["labels"]:
        row["patient_id"] = "SensitivePatientToken"
        row["source_id"] = "SensitiveSourceToken"
    # Genuine cached synthetic ROIs for CLI; intake unit fixtures stay file-only.
    import numpy as np

    from training import roi_provenance as roi

    study = data["studies"][0]
    study.pop("series_path")
    volume = np.ones((3, 4, 5), dtype=np.float32)
    source = {"sha256": "c" * 64, "geometry": roi.voxel_geometry(volume.shape)}
    detectors = {"family": roi.DETECTOR_FAMILY, "left_sha256": "a" * 64, "right_sha256": "b" * 64}
    for side in ("left", "right"):
        roi.write_roi_crop(
            volume,
            (1, 2, 2),
            5,
            root / study["crops"][side],
            side=side,
            record=study,
            source=source,
            detectors=detectors,
        )
    script = Path(__file__).resolve().parents[1] / "tools/check_training_inputs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--input-json", str(write()), "--dataset-root", str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["patient_count"] == 1 and report["training_ready"]
    assert report["roi_provenance"]["cached_only_study_count"] == 1
    assert report["roi_provenance"]["source_rechecked_study_count"] == 0
    assert "Sensitive" not in result.stdout + result.stderr
    assert str(root) not in result.stdout + result.stderr


def test_legacy_logs_do_not_disclose_names_or_cache_paths(tmp_path, caplog):
    import logging

    manifest = tmp_path / "manifest.json"
    labels = tmp_path / "labels.json"
    manifest.write_text(
        json.dumps({"studies": [{"study_id": "scan", "patient_name": "SyntheticPrivatePerson"}]})
    )
    labels.write_text(json.dumps({"patients": []}))
    cache = tmp_path / "SyntheticPrivateFolder.json"
    with caplog.at_level(logging.DEBUG, logger="training.tmj_position_label_table"):
        assert table.build_index(manifest, labels, tmp_path, cache_path=cache) == []
    assert "SyntheticPrivate" not in caplog.text
    assert str(tmp_path) not in caplog.text


@pytest.mark.parametrize(
    "bad_argument", ["--allow-uncropped=SyntheticSensitiveToken", "--SyntheticSensitiveToken"]
)
def test_cli_argument_errors_are_safe_json_and_help_still_works(intake, bad_argument):
    script = Path(__file__).resolve().parents[1] / "tools/check_training_inputs.py"
    command = [
        sys.executable,
        str(script),
        "--input-json",
        str(intake[2]()),
        "--dataset-root",
        str(intake[1]),
    ]
    result = subprocess.run(command + [bad_argument], capture_output=True, text=True)
    assert result.returncode == 2
    assert "SyntheticSensitiveToken" not in result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert not report["ready"]
    assert report["diagnostics"] == [{"code": "invalid_arguments", "collection": "input"}]
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_result.returncode == 0 and "--input-json" in help_result.stdout


def test_excessive_json_nesting_is_structured_without_traceback(intake):
    _, root, write = intake
    path = write()
    path.write_text("[" * 10000 + "0" + "]" * 10000)
    with pytest.raises(table.InputValidationError) as error:
        table.build_canonical_index(path, root)
    assert error.value.report["diagnostics"] == [
        {"code": "unreadable_input", "collection": "input"}
    ]
    script = Path(__file__).resolve().parents[1] / "tools/check_training_inputs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--input-json", str(path), "--dataset-root", str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and not result.stderr
    assert json.loads(result.stdout)["diagnostics"] == error.value.report["diagnostics"]


def test_nul_dataset_root_is_structured_without_echo(intake):
    with pytest.raises(table.InputValidationError) as error:
        table.build_canonical_index(intake[2](), "SyntheticSensitiveRoot\0")
    assert error.value.report["diagnostics"] == [
        {"code": "invalid_dataset_root", "collection": "input"}
    ]
    assert "SyntheticSensitiveRoot" not in str(error.value) + json.dumps(error.value.report)
