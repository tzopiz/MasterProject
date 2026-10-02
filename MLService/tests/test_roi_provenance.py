"""Synthetic array/DICOM/ROI checks; no clinical data or model training."""

import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from training import roi_provenance as roi


def write_series(root, physical=True):
    root.mkdir(exist_ok=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    for z in range(3):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(
            str(root / f"SyntheticPrivateName-{2 - z}.dcm"),
            {},
            file_meta=meta,
            preamble=b"\0" * 128,
        )
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID, ds.StudyInstanceUID = series_uid, study_uid
        ds.PatientName, ds.PatientID = "SyntheticPrivateName", "SyntheticPrivateID"
        ds.InstanceNumber = z + 1
        ds.Rows, ds.Columns = 4, 5
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 2, -10
        if physical:
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [10, 20, 30 + z * 2]
            ds.PixelSpacing = [0.5, 0.75]
        ds.PixelData = (np.arange(20, dtype=np.int16).reshape(4, 5) + z * 100).tobytes()
        ds.save_as(ds.filename, enforce_file_format=True)
    return root


@pytest.mark.parametrize("center", [(0, 0, 0), (2, 3, 4), (0, 3, 0), (2, 0, 4)])
@pytest.mark.parametrize("size", [4, 5, 9])
def test_fixed_window_preserves_every_source_landmark(center, size):
    volume = np.arange(1, 61).reshape(3, 4, 5)
    crop, transform = roi.extract_fixed_crop(volume, center, size)
    start = np.array(center) - size // 2
    assert crop.shape == (size,) * 3
    assert transform["requested_start"] == start.tolist()
    for position in np.ndindex(volume.shape):
        target = np.array(position) - start
        if np.all(target >= 0) and np.all(target < size):
            assert crop[tuple(target)] == volume[position]
    assert crop[(size // 2,) * 3] == volume[center]


@pytest.mark.parametrize(
    "center,size",
    [
        ((-1, 0, 0), 4),
        ((3, 0, 0), 4),
        ((0.5, 0, 0), 4),
        ((True, 0, 0), 4),
        ((0, 0, 0), 0),
        ((0, 0, 0), True),
    ],
)
def test_invalid_crop_requests_fail(center, size):
    with pytest.raises(roi.ROIValidationError):
        roi.extract_fixed_crop(np.ones((3, 4, 5)), center, size)


def test_instance_order_and_verified_zyx_ras_geometry(tmp_path):
    volume, source = roi.load_validated_series(write_series(tmp_path / "series"))
    assert volume[0, 0, 0] == -10 and volume[2, 0, 0] == 390
    affine = np.array(source["geometry"]["affine"])
    assert np.allclose(affine @ [2, 3, 4, 1], [-13, -21.5, 34, 1])
    assert source["geometry"]["coordinate_space"] == "physical-ras"
    assert "SyntheticPrivate" not in json.dumps(source)
    assert str(tmp_path) not in json.dumps(source)


def make_pair(tmp_path, physical=True):
    source_dir = write_series(tmp_path / "series", physical=physical)
    volume, source = roi.load_validated_series(source_dir)
    record = {
        "source_id": "cohort",
        "study_id": "scan",
        "dicom_dir": str(source_dir),
        "crop_paths": {},
    }
    detectors = {"family": roi.DETECTOR_FAMILY, "left_sha256": "a" * 64, "right_sha256": "b" * 64}
    for side, center in (("left", (0, 0, 0)), ("right", (2, 3, 4))):
        path = tmp_path / f"{side}.nii.gz"
        roi.write_roi_crop(
            volume, center, 5, path, side=side, record=record, source=source, detectors=detectors
        )
        record["crop_paths"][side] = str(path)
    return record, detectors


def test_pair_roundtrip_source_recheck_and_explicit_crop_only(tmp_path):
    record, detectors = make_pair(tmp_path)
    report = roi.validate_roi_pair(record, expected_detectors=detectors)
    assert report == {
        "source_rechecked": True,
        "coordinate_space": "physical-ras",
        "detector_training_identity": "unknown",
        "roi_contract": {"detectors": detectors, "preprocessing": roi.preprocessing_options(5)},
    }
    for side in ("left", "right"):
        path = Path(record["crop_paths"][side])
        passport = json.loads(roi.passport_path(path).read_text())
        assert "SyntheticPrivate" not in json.dumps(passport)
        assert str(tmp_path) not in json.dumps(passport)
        assert np.allclose(nib.load(path).get_sform(), passport["crop"]["affine"])
    del record["dicom_dir"]
    assert not roi.validate_roi_pair(record)["source_rechecked"]


def test_voxel_space_never_claims_physical_affine(tmp_path):
    record, _ = make_pair(tmp_path, physical=False)
    report = roi.validate_roi_pair(record)
    assert report["coordinate_space"] == "voxel"
    image = nib.load(record["crop_paths"]["left"])
    assert int(image.header["sform_code"]) == 0
    assert image.header.get_xyzt_units()[0] == "unknown"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "corrupt",
        "checksum",
        "shape",
        "transform",
        "side",
        "study",
        "detector",
        "preprocessing",
        "extra",
        "source",
    ],
)
def test_pair_refuses_missing_stale_or_mismatched_passports(tmp_path, damage):
    record, detectors = make_pair(tmp_path)
    path = roi.passport_path(record["crop_paths"]["left"])
    passport = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{SyntheticPrivateName")
    elif damage == "checksum":
        Path(record["crop_paths"]["left"]).write_bytes(b"SyntheticPrivateCorruptCrop")
    elif damage == "source":
        import pydicom

        file = next(Path(record["dicom_dir"]).glob("*.dcm"))
        ds = pydicom.dcmread(file)
        ds.RescaleIntercept = -11
        ds.save_as(file, enforce_file_format=True)
    else:
        if damage == "shape":
            passport["crop"]["shape"][0] += 1
        if damage == "transform":
            passport["crop"]["requested_start"][0] += 1
        if damage == "side":
            passport["side"] = "right"
        if damage == "study":
            passport["study_key"] = "f" * 64
        if damage == "detector":
            passport["detectors"]["left_sha256"] = "c" * 64
        if damage == "preprocessing":
            passport["preprocessing"]["zoom_order"] = 3
        if damage == "extra":
            passport["source"]["PatientName"] = "SyntheticPrivateName"
        path.write_text(json.dumps(passport))
    with pytest.raises(roi.ROIValidationError) as error:
        roi.validate_roi_pair(record, expected_detectors=detectors)
    assert "SyntheticPrivate" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "mixed-series",
        "orientation",
        "spacing",
        "positions",
        "partial",
        "color",
        "multiframe",
        "tilt",
    ],
)
def test_unsupported_or_mixed_series_are_rejected_without_identifiers(tmp_path, damage):
    import pydicom

    root = write_series(tmp_path / "SyntheticPrivateSeries")
    file = next(p for p in root.glob("*.dcm") if pydicom.dcmread(p).InstanceNumber == 3)
    ds = pydicom.dcmread(file)
    if damage == "duplicate":
        ds.InstanceNumber = 2
    if damage == "mixed-series":
        ds.SeriesInstanceUID = generate_uid()
    if damage == "orientation":
        ds.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    if damage == "spacing":
        ds.PixelSpacing = [1, 1]
    if damage == "positions":
        ds.ImagePositionPatient = [10, 20, 35]
    if damage == "partial":
        del ds.PixelSpacing
    if damage == "color":
        ds.SamplesPerPixel = 3
    if damage == "multiframe":
        ds.NumberOfFrames = 2
    if damage == "tilt":
        ds.ImagePositionPatient = [11, 20, 34]
    ds.save_as(file, enforce_file_format=True)
    with pytest.raises(roi.ROIValidationError) as error:
        roi.load_validated_series(root)
    assert "SyntheticPrivate" not in str(error.value)


def test_cache_invalidates_requested_size_and_both_model_hashes(tmp_path):
    record, detectors = make_pair(tmp_path)
    for changed in (
        {**detectors, "left_sha256": "c" * 64},
        {**detectors, "right_sha256": "d" * 64},
    ):
        with pytest.raises(roi.ROIValidationError, match="stale_detector_checkpoint"):
            roi.validate_roi_pair(record, expected_detectors=changed)
    with pytest.raises(roi.ROIValidationError, match="stale_preprocessing"):
        roi.validate_roi_pair(record, expected_preprocessing=roi.preprocessing_options(7))


def test_existing_writer_uses_fixed_window(tmp_path):
    from tools import auto_crop_from_detector as writer

    volume = np.arange(1, 61).reshape(3, 4, 5)
    crop = writer.extract_crop(volume, np.array([2, 3, 4]), 5)
    assert crop[2, 2, 2] == volume[2, 3, 4]
    assert np.count_nonzero(crop[3:]) == 0


def test_small_paired_checkpoint_loads_only_matching_family(tmp_path):
    import torch

    from tools import auto_crop_from_detector as writer

    model = writer.TMJHeatmapDetector(features=[2, 4])
    path = tmp_path / "SyntheticPrivateCheckpoint.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "detector_family": roi.DETECTOR_FAMILY,
        "model_config": {"features": [2, 4]},
    }
    torch.save(checkpoint, path)
    loaded = writer.load_paired_detector(path, torch.device("cpu"))
    assert loaded.head.in_channels == 2 and not loaded.training
    checkpoint["detector_family"] = "regression"
    torch.save(checkpoint, path)
    with pytest.raises(roi.ROIValidationError, match="unsupported_detector_family"):
        writer.load_paired_detector(path, torch.device("cpu"))


def test_canonical_writer_cli_preflight_and_cache_invalidation(tmp_path):
    import os
    import subprocess
    import sys

    import torch

    from tools import auto_crop_from_detector as writer
    from training.tmj_position_label_table import build_canonical_index

    series = write_series(tmp_path / "SyntheticPrivateSeries")
    input_path = tmp_path / "input.private.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "studies": [
                    {
                        "source_id": "synthetic",
                        "study_id": "SyntheticPrivateStudy",
                        "patient_id": "SyntheticPrivatePatient",
                        "label_record_id": "labels-a",
                        "label_applicability": "confirmed",
                        "series_path": series.name,
                    }
                ],
                "labels": [
                    {
                        "source_id": "synthetic",
                        "patient_id": "SyntheticPrivatePatient",
                        "label_record_id": "labels-a",
                        "labels": {"sagittal": {"left": 2, "right": 1}},
                    }
                ],
            }
        )
    )
    models = []
    for side in ("left", "right"):
        path = tmp_path / f"SyntheticPrivate-{side}.pth"
        model = writer.TMJHeatmapDetector(features=[2, 4])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "detector_family": roi.DETECTOR_FAMILY,
                "model_config": {"features": [2, 4]},
            },
            path,
        )
        models.append(path)
    output = tmp_path / "roi"
    script = Path(writer.__file__)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        str(script),
        "--left-model",
        str(models[0]),
        "--right-model",
        str(models[1]),
        "--canonical-input",
        str(input_path),
        "--dataset-root",
        str(tmp_path),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--crop-size",
        "5",
        "--skip-existing",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["generated"] == 1
    assert "SyntheticPrivate" not in result.stdout + result.stderr
    index = output / "inputs.private.json"
    records = build_canonical_index(index, tmp_path)
    assert roi.validate_roi_pair(records[0])["source_rechecked"]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0 and json.loads(result.stdout)["skipped"] == 1
    # Changing requested size invalidates the entire cached pair.
    command[command.index("5")] = "7"
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0 and json.loads(result.stdout)["generated"] == 1
    assert nib.load(records[0]["crop_paths"]["left"]).shape == (7, 7, 7)
    preflight = Path(writer.__file__).parent / "check_training_inputs.py"
    result = subprocess.run(
        [
            sys.executable,
            str(preflight),
            "--input-json",
            str(index),
            "--dataset-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["ready"] and report["training_ready"]
    assert report["roi_provenance"]["source_rechecked_study_count"] == 1
    assert "SyntheticPrivate" not in result.stdout + result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
    roi.passport_path(records[0]["crop_paths"]["right"]).unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(preflight),
            "--input-json",
            str(index),
            "--dataset-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2 and not json.loads(result.stdout)["training_ready"]


def test_decoder_warnings_do_not_echo_header_values(tmp_path, caplog):
    import pydicom

    root = write_series(tmp_path / "series")
    file = next(root.glob("*.dcm"))
    # Alter bytes after writing to avoid test construction warnings.
    raw = file.read_bytes()
    ds = pydicom.dcmread(file)
    uid = str(ds.SeriesInstanceUID).encode()
    private = b"SyntheticPrivateIdentifier".ljust(len(uid), b" ")
    file.write_bytes(raw.replace(uid, private))
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(roi.ROIValidationError):
            roi.load_validated_series(root)
    assert not seen
    assert "SyntheticPrivateIdentifier" not in caplog.text


@pytest.mark.parametrize("container", ["geometry", "detectors", "preprocessing", "crop"])
def test_all_passport_containers_are_whitelisted(tmp_path, container):
    record, _ = make_pair(tmp_path)
    path = roi.passport_path(record["crop_paths"]["left"])
    passport = json.loads(path.read_text())
    target = passport["source"]["geometry"] if container == "geometry" else passport[container]
    target["PrivateField"] = "SyntheticPrivateHeader"
    path.write_text(json.dumps(passport))
    with pytest.raises(roi.ROIValidationError) as error:
        roi.validate_roi_pair(record)
    assert "SyntheticPrivate" not in str(error.value)


@pytest.mark.parametrize("damage", ["shape", "affine", "undecodable"])
def test_actual_nifti_is_validated_even_with_updated_checksum(tmp_path, damage):
    record, _ = make_pair(tmp_path)
    path = Path(record["crop_paths"]["left"])
    passport = json.loads(roi.passport_path(path).read_text())
    if damage == "undecodable":
        path.write_bytes(b"SyntheticPrivateNotNifti")
    else:
        shape = (4, 5, 5) if damage == "shape" else (5, 5, 5)
        affine = np.array(passport["crop"]["affine"])
        if damage == "affine":
            affine[0, 3] += 1
        nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), path)
    passport["crop"]["sha256"] = roi.sha256_file(path)
    roi.passport_path(path).write_text(json.dumps(passport))
    with pytest.raises(roi.ROIValidationError) as error:
        roi.validate_roi_pair(record)
    assert "SyntheticPrivate" not in str(error.value)


def test_fractional_physical_origin_roundtrips_nifti1_sform(tmp_path):
    import pydicom

    root = write_series(tmp_path / "series")
    for path in root.iterdir():
        ds = pydicom.dcmread(path)
        z = int(ds.InstanceNumber) - 1
        ds.ImagePositionPatient = [859.001234, 1234.001234, 765.001234 + z * 2]
        ds.save_as(path, enforce_file_format=True)
    volume, source = roi.load_validated_series(root)
    record = {
        "source_id": "cohort",
        "study_id": "fractional",
        "dicom_dir": str(root),
        "crop_paths": {},
    }
    detectors = {"family": roi.DETECTOR_FAMILY, "left_sha256": "a" * 64, "right_sha256": "b" * 64}
    for side in ("left", "right"):
        path = tmp_path / f"{side}.nii.gz"
        roi.write_roi_crop(
            volume, (1, 2, 2), 5, path, side=side, record=record, source=source, detectors=detectors
        )
        record["crop_paths"][side] = str(path)
    passport = json.loads(roi.passport_path(record["crop_paths"]["left"]).read_text())
    original = np.array(passport["crop"]["affine"])
    stored = nib.load(record["crop_paths"]["left"]).get_sform()
    assert np.max(np.abs(stored - original)) > 1e-5
    assert roi.validate_roi_pair(record)["source_rechecked"]


def test_extension_variants_are_complete_and_only_known_sidecars_ignored(tmp_path):
    root = write_series(tmp_path / "series")
    files = list(root.glob("*.dcm"))
    files[0].rename(files[0].with_suffix(".DCM"))
    assert roi.load_validated_series(root)[0].shape == (3, 4, 5)
    files[1].rename(files[1].with_suffix(""))
    (root / "notes.txt").write_text("SyntheticSidecar")
    volume, source = roi.load_validated_series(root)
    assert volume.shape == (3, 4, 5) and source["geometry"]["slice_count"] == 3
    (root / "unknown.bin").write_bytes(b"SyntheticUnknownFile")
    with pytest.raises(roi.ROIValidationError):
        roi.load_validated_series(root)


def test_missing_study_series_uid_refuses_before_pixel_decode(tmp_path, monkeypatch):
    import pydicom
    from pydicom.dataset import Dataset

    root = write_series(tmp_path / "series")
    for path in root.iterdir():
        ds = pydicom.dcmread(path)
        del ds.StudyInstanceUID
        del ds.SeriesInstanceUID
        ds.PatientID = "SyntheticDifferentPatient" + str(ds.InstanceNumber)
        ds.save_as(path, enforce_file_format=True)
    monkeypatch.setattr(
        Dataset, "pixel_array", property(lambda self: pytest.fail("decoded before UID validation"))
    )
    with pytest.raises(roi.ROIValidationError):
        roi.load_validated_series(root)


def test_deep_geometry_is_safe_roi_error_and_cli_json(tmp_path):
    import os
    import subprocess
    import sys

    record, _ = make_pair(tmp_path)
    path = roi.passport_path(record["crop_paths"]["left"])
    passport = json.loads(path.read_text())
    nested = 0
    for _ in range(500):
        nested = [nested]
    passport["source"]["geometry"]["affine"] = nested
    path.write_text(json.dumps(passport))
    with pytest.raises(roi.ROIValidationError):
        roi.validate_roi_pair(record)
    # Use the canonical CLI boundary, with synthetic labels/IDs only.
    input_path = tmp_path / "inputs.private.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "studies": [
                    {
                        "source_id": record["source_id"],
                        "study_id": record["study_id"],
                        "patient_id": "patient",
                        "label_record_id": "labels",
                        "label_applicability": "confirmed",
                        "crops": {
                            side: Path(path).name for side, path in record["crop_paths"].items()
                        },
                    }
                ],
                "labels": [
                    {
                        "source_id": record["source_id"],
                        "patient_id": "patient",
                        "label_record_id": "labels",
                        "labels": {"sagittal": {"left": 1, "right": 2}},
                    }
                ],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/check_training_inputs.py"),
            "--input-json",
            str(input_path),
            "--dataset-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 2 and not result.stderr
    assert not json.loads(result.stdout)["training_ready"]


def test_crop_cap_applies_before_np_pad_and_typical_512_cube_is_supported(monkeypatch):
    assert roi.voxel_geometry((512, 512, 512))["shape"] == [512, 512, 512]
    monkeypatch.setattr(
        np, "pad", lambda *args, **kwargs: pytest.fail("allocated padding before cap")
    )
    with pytest.raises(roi.ROIValidationError):
        roi.extract_fixed_crop(np.ones((3, 4, 5)), (1, 2, 2), 1_000_000)
    with pytest.raises(roi.ROIValidationError):
        roi.preprocessing_options(1_000_000)


@pytest.mark.parametrize(
    "limit", ["file_bytes", "source_bytes", "slices", "dimensions", "voxels", "compressed"]
)
def test_source_profile_refuses_before_pixel_decode(tmp_path, monkeypatch, limit):
    import pydicom
    from pydicom.dataset import Dataset

    root = write_series(tmp_path / "series")
    if limit == "file_bytes":
        monkeypatch.setattr(roi, "MAX_SOURCE_FILE_BYTES", 1)
    if limit == "source_bytes":
        monkeypatch.setattr(roi, "MAX_SOURCE_BYTES", 1)
    if limit == "slices":
        monkeypatch.setattr(roi, "MAX_SOURCE_SLICES", 2)
    if limit == "dimensions":
        for path in root.iterdir():
            ds = pydicom.dcmread(path)
            ds.Rows = 5000
            ds.save_as(path, enforce_file_format=True)
    if limit == "voxels":
        monkeypatch.setattr(roi, "MAX_SOURCE_VOXELS", 1)
    if limit == "compressed":
        from pydicom.encaps import encapsulate
        from pydicom.uid import RLELossless

        for path in root.iterdir():
            ds = pydicom.dcmread(path)
            ds.file_meta.TransferSyntaxUID = RLELossless
            ds.PixelData = encapsulate([b"synthetic-not-compressed"])
            ds["PixelData"].is_undefined_length = True
            ds.save_as(path, enforce_file_format=True)
    monkeypatch.setattr(
        Dataset,
        "pixel_array",
        property(lambda self: pytest.fail("decoded before profile validation")),
    )
    with pytest.raises(roi.ROIValidationError) as caught:
        roi.load_validated_series(root)
    if limit == "compressed":
        assert caught.value.code == "compressed_source_unsupported"


@pytest.mark.parametrize("field", ["geometry", "options", "crop"])
def test_deep_nested_fields_all_refuse_without_recursive_validation(tmp_path, field):
    record, _ = make_pair(tmp_path)
    path = roi.passport_path(record["crop_paths"]["left"])
    passport = json.loads(path.read_text())
    nested = 0
    for _ in range(500):
        nested = [nested]
    if field == "geometry":
        passport["source"]["geometry"]["affine"] = nested
    if field == "options":
        passport["preprocessing"]["zoom_mode"] = nested
    if field == "crop":
        passport["crop"]["source_bounds"] = nested
    path.write_text(json.dumps(passport))
    with pytest.raises(roi.ROIValidationError):
        roi.validate_roi_pair(record)


@pytest.mark.parametrize("budget", ["entries", "passport", "roi"])
def test_remaining_budgets_refuse_safely(tmp_path, monkeypatch, budget):
    if budget == "entries":
        root = write_series(tmp_path / "series")
        monkeypatch.setattr(roi, "MAX_SOURCE_ENTRIES", 2)
        with pytest.raises(roi.ROIValidationError, match="source_entry_limit"):
            roi.load_validated_series(root)
    else:
        record, _ = make_pair(tmp_path)
        constant = "MAX_PASSPORT_BYTES" if budget == "passport" else "MAX_ROI_FILE_BYTES"
        code = "passport_byte_limit" if budget == "passport" else "roi_file_byte_limit"
        monkeypatch.setattr(roi, constant, 1)
        with pytest.raises(roi.ROIValidationError, match=code):
            roi.validate_roi_pair(record)


def test_frame_uid_consistency_and_missing_crop_have_safe_errors(tmp_path):
    import pydicom

    root = write_series(tmp_path / "series")
    uid = generate_uid()
    for path in root.iterdir():
        ds = pydicom.dcmread(path)
        ds.FrameOfReferenceUID = uid
        ds.save_as(path, enforce_file_format=True)
    assert roi.load_validated_series(root)[0].shape == (3, 4, 5)
    path = next(root.iterdir())
    ds = pydicom.dcmread(path)
    del ds.FrameOfReferenceUID
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(roi.ROIValidationError, match="invalid_source_series_uid"):
        roi.load_validated_series(root)
    record, _ = make_pair(tmp_path, physical=False)
    Path(record["crop_paths"]["left"]).unlink()
    with pytest.raises(roi.ROIValidationError, match="unreadable_roi_input"):
        roi.validate_roi_pair(record, recheck_source=False)


def test_deflated_source_is_refused_before_dataset_inflation(tmp_path, monkeypatch):
    import pydicom
    from pydicom.uid import DeflatedExplicitVRLittleEndian

    root = write_series(tmp_path / "series")
    for path in root.iterdir():
        ds = pydicom.dcmread(path)
        ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
        ds.save_as(path, enforce_file_format=True)

    calls = []

    def inflate(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unsupported dataset must not be inflated")

    monkeypatch.setattr("pydicom.filereader.zlib.decompress", inflate)
    with pytest.raises(roi.ROIValidationError, match="compressed_source_unsupported"):
        roi.load_validated_series(root)
    assert not calls


@pytest.mark.parametrize(
    "damage", ["extension", "descrip", "aux_file", "intent_name", "db_name", "data_type", "offset"]
)
def test_roi_extensions_refused_before_loading_private_metadata(tmp_path, monkeypatch, damage):
    record, detectors = make_pair(tmp_path)
    path = Path(record["crop_paths"]["left"])
    image = nib.load(path)
    if damage == "extension":
        image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"SyntheticPrivateName"))
    elif damage == "offset":
        image.header["vox_offset"] = 4096
    else:
        image.header[damage] = b"Private"
    nib.save(image, path)
    passport = json.loads(roi.passport_path(path).read_text())
    passport["crop"]["sha256"] = roi.sha256_file(path)
    roi.passport_path(path).write_text(json.dumps(passport))
    calls = []
    original_load = nib.load

    def load(*args, **kwargs):
        calls.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(nib, "load", load)
    with pytest.raises(roi.ROIValidationError, match="unsupported_roi_header"):
        roi.validate_roi_pair(record, expected_detectors=detectors, recheck_source=False)
    assert not calls


@pytest.mark.parametrize(
    "wrapper", ["filename", "comment", "extra", "timestamp", "member", "tail", "payload"]
)
def test_private_gzip_metadata_refused_before_nifti_loading(tmp_path, monkeypatch, wrapper):
    record, detectors = make_pair(tmp_path)
    path = Path(record["crop_paths"]["left"])
    data = gzip.compress(gzip.decompress(path.read_bytes()), mtime=0)
    marker = b"SyntheticSensitiveToken"
    if wrapper in ("filename", "comment", "extra"):
        flag = {"filename": 8, "comment": 16, "extra": 4}[wrapper]
        metadata = (len(marker).to_bytes(2, "little") + marker) if flag == 4 else marker + b"\0"
        data = data[:3] + bytes([flag]) + data[4:10] + metadata + data[10:]
    elif wrapper == "timestamp":
        data = data[:4] + (1).to_bytes(4, "little") + data[8:]
    elif wrapper == "member":
        member = gzip.compress(b"", mtime=0)
        data += member[:3] + b"\x08" + member[4:10] + marker + b"\0" + member[10:]
    elif wrapper == "payload":
        data = gzip.compress(gzip.decompress(data) + marker * 65536, mtime=0)
    else:
        data += marker
    path.write_bytes(data)
    passport = json.loads(roi.passport_path(path).read_text())
    passport["crop"]["sha256"] = roi.sha256_file(path)
    roi.passport_path(path).write_text(json.dumps(passport))
    calls = []
    original_load = nib.load

    def load(*args, **kwargs):
        calls.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(nib, "load", load)
    with pytest.raises(roi.ROIValidationError, match="unsupported_roi_header"):
        roi.validate_roi_pair(record, expected_detectors=detectors, recheck_source=False)
    assert not calls
