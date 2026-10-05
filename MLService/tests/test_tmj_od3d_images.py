import json

import numpy as np
import pydicom
import pytest

from training.tmj_od3d_images import PreparationError, digest_file, prepare_patient

from .test_roi_provenance import write_series


def source(tmp_path):
    folder = write_series(tmp_path / "synthetic-private-patient")
    # Sparse InstanceNumber 10,20,30; no claim that these are adjacent anatomy.
    for path in folder.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.InstanceNumber = int(ds.InstanceNumber) * 10
        ds.save_as(path, enforce_file_format=True)
    files = {
        int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber): p.name
        for p in folder.glob("*.dcm")
    }
    for side, codes in [("L", "0"), ("R", "1,5")]:
        box = {
            "image_path": "C:\\private\\" + files[20],
            "frames": [
                {
                    "width": 5,
                    "height": 4,
                    "shapes": [
                        {
                            "label": f"{codes}-{side}-10-30",
                            "shape_type": "rectangle",
                            "points": [[1, 1], [4, 3]],
                        }
                    ],
                }
            ],
        }
        (folder / f"{side}-label.json").write_text(json.dumps(box))
    return folder, {
        "left": (0,),
        "right": (1, 5),
        "slice_count": 3,
        "slice_min": 0,
        "slice_max": 30,
    }


def test_sparse_bags_use_actual_instance_numbers_and_bind_hashes(tmp_path):
    folder, meta = source(tmp_path)
    out = tmp_path / "prepared"
    rows, excluded = prepare_patient(folder, meta, out, slice_count=4, image_size=16)
    assert excluded == {} and len(rows) == 2
    assert rows[0]["codes"] == [0] and rows[0]["binary_target"] == 0
    assert rows[1]["codes"] == [1, 5] and rows[1]["binary_target"] == 1
    assert rows[0]["instance_numbers"] == [10, 20, 20, 30]
    for row in rows:
        path = out / row["crop_path"]
        assert digest_file(path) == row["crop_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            image = archive["images"]
            assert image.shape == (4, 1, 16, 16)
            assert image.dtype == np.float16 and np.isfinite(image).all()
            assert 0 <= image.min() <= image.max() <= 1
        assert row["roi_source"] == "author_annotation" and row["assessment_input"] == "oracle_roi"
    assert "private-patient" not in json.dumps(rows)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("wrong_labels", "csv_annotation_label_mismatch"),
        ("range", "reference_outside_range"),
        ("frame", "annotation_frame_mismatch"),
        ("instance", "duplicate_dicom"),
        ("series", "mixed_series"),
    ],
)
def test_rejects_binding_and_dicom_errors_without_private_values(tmp_path, mutation, error):
    folder, meta = source(tmp_path)
    p = folder / "L-label.json"
    data = json.loads(p.read_text())
    if mutation == "wrong_labels":
        meta["left"] = (1,)
    elif mutation == "reference":
        data["image_path"] = "secret-name.dcm"
    elif mutation == "range":
        data["frames"][0]["shapes"][0]["label"] = "0-L-1-9"
    elif mutation == "frame":
        data["frames"][0]["width"] = 6
    elif mutation in ("instance", "series"):
        files = sorted(folder.glob("*.dcm"))
        ds = pydicom.dcmread(files[1])
        other = pydicom.dcmread(files[0])
        if mutation == "instance":
            ds.InstanceNumber = other.InstanceNumber
        else:
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(files[1], enforce_file_format=True)
    p.write_text(json.dumps(data))
    with pytest.raises(PreparationError) as caught:
        prepare_patient(folder, meta, tmp_path / "out", 4, 16)
    assert str(caught.value) == error
    assert "private" not in str(caught.value)


def test_unknown_side_excluded_without_inventing_negative(tmp_path):
    folder, meta = source(tmp_path)
    meta["left"] = None
    rows, reasons = prepare_patient(folder, meta, tmp_path / "out", 4, 16)
    assert len(rows) == 1 and rows[0]["side"] == "R"
    assert reasons == {"missing_label": 1}


def test_existing_artifact_never_overwritten(tmp_path):
    folder, meta = source(tmp_path)
    out = tmp_path / "out"
    rows, _ = prepare_patient(folder, meta, out, 4, 16)
    path = out / rows[0]["crop_path"]
    before = path.read_bytes()
    with pytest.raises(PreparationError, match="prepared_crop_exists"):
        prepare_patient(folder, meta, out, 4, 16)
    assert path.read_bytes() == before


def test_pixel_fingerprint_survives_changed_headers(tmp_path):
    import shutil

    folder, meta = source(tmp_path)
    other = tmp_path / "different-private-patient"
    shutil.copytree(folder, other)
    uid = pydicom.uid.generate_uid()
    for path in other.glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.SeriesInstanceUID = uid
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.save_as(path, enforce_file_format=True)
    first, _ = prepare_patient(folder, meta, tmp_path / "out-a", 4, 16)
    second, _ = prepare_patient(other, meta, tmp_path / "out-b", 4, 16)
    assert first[0]["source_sha256"] != second[0]["source_sha256"]
    assert first[0]["source_pixel_sha256"] == second[0]["source_pixel_sha256"]
    assert first[0]["pixel_sha256"] == second[0]["pixel_sha256"]


def test_missing_reference_keeps_only_published_slices_in_author_range(tmp_path):
    folder, meta = source(tmp_path)
    path = folder / "L-label.json"
    annotation = json.loads(path.read_text())
    annotation["image_path"] = "file_not_released.dcm"
    path.write_text(json.dumps(annotation))
    rows, _ = prepare_patient(folder, meta, tmp_path / "out", 4, 16)
    assert rows[0]["reference_images_not_released"] == 1
    assert rows[0]["instance_numbers"] == [10, 20, 20, 30]


@pytest.mark.parametrize("side", ["L", "R"])
def test_empty_annotation_excludes_only_that_side(tmp_path, side):
    folder, meta = source(tmp_path)
    (folder / f"{side}-label.json").write_text(json.dumps({"frames": []}))
    rows, reasons = prepare_patient(folder, meta, tmp_path / "out", 4, 16)
    assert [row["side"] for row in rows] == ["R" if side == "L" else "L"]
    assert reasons == {"empty_annotation": 1}
    assert len(list((tmp_path / "out/data").glob("*.npz"))) == 1
