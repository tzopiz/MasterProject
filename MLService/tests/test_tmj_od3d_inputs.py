"""Author OD3D ontology, CSV intake and rectangle binding; synthetic inputs only."""

import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    assert importlib.util.find_spec("training.tmj_od3d_inputs"), "OD3D intake is missing"
    return importlib.import_module("training.tmj_od3d_inputs")


@pytest.mark.parametrize(
    "value,separator,expected",
    [
        ("0", "|", (0,)),
        ("6|1|3", "|", (1, 3, 6)),
        (" 5,2 ", ",", (2, 5)),
        ("", "|", None),
        ("  ", ",", None),
    ],
)
def test_codes_preserve_ontology_combinations_and_missing(value, separator, expected):
    assert module().parse_codes(value, separator) == expected


@pytest.mark.parametrize(
    "value", ["0|1", "7", "-1", "1||2", "|1", "1|", "1|1", "1.0", "normal", "1,2", None, True]
)
def test_invalid_codes_are_refused_without_input_values(value):
    api = module()
    with pytest.raises(api.OD3DInputError) as error:
        api.parse_codes(value, "|")
    assert str(error.value) in {"invalid_codes", "normal_abnormal_conflict"}
    assert error.value.code == str(error.value)


HEADERS = [
    "anonymous_id",
    "patient_sex",
    "age_years",
    "age_group",
    "selected_slice_min",
    "selected_slice_max",
    "selected_slice_count",
    "label_L",
    "label_R",
]


def metadata(tmp_path, rows=None, headers=HEADERS):
    path = tmp_path / "private.csv"
    rows = rows or [
        ["patient-token", "F", "42", "adult", "24081200", "24081350", "151", "0", "1|5"]
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_metadata_preserves_supplied_id_and_inclusive_integer_ranges(tmp_path):
    assert module().load_metadata(metadata(tmp_path)) == {
        "patient-token": {
            "left": (0,),
            "right": (1, 5),
            "slice_min": 24081200,
            "slice_max": 24081350,
            "slice_count": 151,
        }
    }


@pytest.mark.parametrize(
    "minimum,maximum,count",
    [
        ("5", "4", "1"),
        ("-1", "4", "6"),
        ("1.0", "4", "4"),
        ("1", "4", "0"),
        ("1", "4", "5"),
        ("1", "4", "bad"),
    ],
)
def test_invalid_slice_metadata_is_refused(tmp_path, minimum, maximum, count):
    api = module()
    path = metadata(
        tmp_path, [["sensitive-id", "F", "42", "adult", minimum, maximum, count, "0", ""]]
    )
    with pytest.raises(api.OD3DInputError, match="^invalid_slice_info$"):
        api.load_metadata(path)


def test_metadata_missing_labels_remain_unknown(tmp_path):
    path = metadata(tmp_path, [["patient-token", "F", "42", "adult", "1", "4", "4", "", "2"]])
    assert module().load_metadata(path)["patient-token"]["left"] is None


def test_duplicate_ids_and_headers_and_wrong_row_width_are_refused(tmp_path):
    api = module()
    row = ["private-id", "F", "42", "adult", "1", "4", "4", "0", "5"]
    cases = [
        ([row, row], HEADERS, "duplicate_patient_id"),
        ([row[:-1]], HEADERS, "invalid_metadata_row"),
        ([row], HEADERS[:-1] + ["label_L"], "invalid_metadata_headers"),
    ]
    for rows, headers, code in cases:
        with pytest.raises(api.OD3DInputError, match=f"^{code}$"):
            api.load_metadata(metadata(tmp_path, rows, headers))


def shape(label="5,2-L-280-310", points=None, kind="rectangle"):
    return {
        "label": label,
        "points": points or [[485.5, 267.25], [390.5, 319.75]],
        "shape_type": kind,
    }


def annotation(tmp_path, payload):
    path = tmp_path / "private-patient.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frames_author_rectangle_coordinates_and_reference_basename(tmp_path):
    payload = {
        "image_path": r"C:\private\patient\scan.dcm",
        "frames": [{"width": 512, "height": 512, "shapes": [shape()]}],
    }
    assert module().parse_annotation(annotation(tmp_path, payload), "L") == [
        {
            "codes": (2, 5),
            "side": "L",
            "instance_start": 280,
            "instance_end": 310,
            "bbox_xyxy": (390.5, 267.25, 485.5, 319.75),
            "reference_basename": "scan.dcm",
            "width": 512,
            "height": 512,
        }
    ]


def test_root_and_frames_shapes_support_multiple_regions_and_dimension_fallback(tmp_path):
    payload = {
        "image_path": "/private/root.dcm",
        "imageWidth": 512,
        "imageHeight": 512,
        "shapes": [shape("0-L-1-1")],
        "frames": [
            {"image_path": "/another/frame.dcm", "shapes": [shape("1-L-2-3"), shape("6-L-4-5")]}
        ],
    }
    rows = module().parse_annotation(annotation(tmp_path, payload), "L")
    assert [r["codes"] for r in rows] == [(0,), (1,), (6,)]
    assert [r["reference_basename"] for r in rows] == ["root.dcm", "frame.dcm", "frame.dcm"]
    assert [(r["width"], r["height"]) for r in rows] == [(512, 512)] * 3


@pytest.mark.parametrize(
    "label",
    [
        "1-R-1-2",
        "0,5-L-1-2",
        "7-L-1-2",
        "1-L-2-1",
        "1-L--1-2",
        "1-L-1",
        "1-L-1.0-2",
        "1-X-1-2",
        "-L-1-2",
    ],
)
def test_bad_labels_and_conflicting_sides_are_refused(tmp_path, label):
    api = module()
    payload = {"image_path": "scan.dcm", "width": 512, "height": 512, "shapes": [shape(label)]}
    with pytest.raises(api.OD3DInputError):
        api.parse_annotation(annotation(tmp_path, payload), "L")


@pytest.mark.parametrize(
    "points,kind",
    [
        ([[1, 1], [1, 2]], "rectangle"),
        ([[1, 1], [2, 1]], "rectangle"),
        ([[-1, 1], [2, 2]], "rectangle"),
        ([[1, 1], [513, 2]], "rectangle"),
        ([[float("nan"), 1], [2, 2]], "rectangle"),
        ([[1, 1], [float("inf"), 2]], "rectangle"),
        ([[True, 1], [2, 2]], "rectangle"),
        ([[1, 1]], "rectangle"),
        ([[1, 1], [2, 2]], "polygon"),
    ],
)
def test_nonrectangle_nonfinite_and_out_of_bounds_geometry_is_refused(tmp_path, points, kind):
    api = module()
    payload = {
        "image_path": "scan.dcm",
        "width": 512,
        "height": 512,
        "shapes": [shape(points=points, kind=kind)],
    }
    with pytest.raises(api.OD3DInputError):
        api.parse_annotation(annotation(tmp_path, payload), "L")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"shapes": []},
        {"frames": []},
        {"frames": [None]},
        {"image_path": "scan.dcm", "width": 0, "height": 512, "shapes": [shape()]},
        {"image_path": "scan.dcm", "width": 512, "height": 512.0, "shapes": [shape()]},
        {"image_path": "/private/..", "width": 512, "height": 512, "shapes": [shape()]},
        {"width": 512, "height": 512, "shapes": [shape()]},
    ],
)
def test_empty_or_invalid_annotation_cannot_be_normal(tmp_path, payload):
    api = module()
    with pytest.raises(api.OD3DInputError):
        api.parse_annotation(annotation(tmp_path, payload), "L")


def test_expected_side_is_explicit_and_errors_are_safe(tmp_path, capsys):
    api = module()
    path = annotation(
        tmp_path, {"image_path": "scan.dcm", "width": 512, "height": 512, "shapes": [shape()]}
    )
    for side in ("left", "l", None):
        with pytest.raises(api.OD3DInputError, match="^invalid_expected_side$"):
            api.parse_annotation(path, side)
    missing = tmp_path / "sensitive-name.json"
    with pytest.raises(api.OD3DInputError, match="^unreadable_input$") as error:
        api.parse_annotation(missing, "L")
    assert error.value.__cause__ is None
    assert capsys.readouterr() == ("", "")


def test_duplicate_json_keys_and_bounded_input_bytes_and_rows(tmp_path, monkeypatch):
    api = module()
    path = tmp_path / "private.json"
    path.write_text('{"shapes":[],"shapes":[]}')
    with pytest.raises(api.OD3DInputError, match="^invalid_annotation_json$"):
        api.parse_annotation(path, "L")
    monkeypatch.setattr(api, "MAX_INPUT_BYTES", 8)
    with pytest.raises(api.OD3DInputError, match="^input_byte_limit$"):
        api.load_metadata(metadata(tmp_path))
    monkeypatch.setattr(api, "MAX_INPUT_BYTES", 1024)
    monkeypatch.setattr(api, "MAX_METADATA_ROWS", 1)
    row = ["p1", "F", "42", "adult", "1", "4", "4", "0", "5"]
    row2 = ["p2", "F", "42", "adult", "1", "4", "4", "0", "5"]
    with pytest.raises(api.OD3DInputError, match="^metadata_row_limit$"):
        api.load_metadata(metadata(tmp_path, [row, row2]))


def test_sparse_selected_filenames_do_not_require_contiguous_instance_numbers(tmp_path):
    path = metadata(tmp_path, [["p", "F", "42", "adult", "100", "200", "30", "0", "5"]])
    assert module().load_metadata(path)["p"]["slice_count"] == 30


def test_huge_integer_coordinates_and_annotation_limit_refuse_safely(tmp_path, monkeypatch):
    api = module()
    payload = {
        "image_path": "scan.dcm",
        "width": 512,
        "height": 512,
        "shapes": [shape(points=[[1, 1], [10**400, 2]])],
    }
    with pytest.raises(api.OD3DInputError):
        api.parse_annotation(annotation(tmp_path, payload), "L")
    payload["shapes"] = [shape(), shape()]
    monkeypatch.setattr(api, "MAX_ANNOTATION_SHAPES", 1)
    with pytest.raises(api.OD3DInputError, match="^annotation_shape_limit$"):
        api.parse_annotation(annotation(tmp_path, payload), "L")
