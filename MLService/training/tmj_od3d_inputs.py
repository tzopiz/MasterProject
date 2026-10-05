"""Bounded private TMJ-OD3D intake; author osseous labels are not position labels.

CSV selected_slice_* values remain source filename numbers; JSON ranges are
inclusive InstanceNumber values. This module does not equate these namespaces.
"""

import csv
import io
import json
import math
import re
from pathlib import Path

SOURCE_DOI = "10.57760/sciencedb.37727"
CODEBOOK_SOURCE = (
    "https://github.com/ZiTingW/TMJ-OD3D/blob/a4bce89c89b7175e330193f04a282887b5e3f8b0/README.md"
)
CODEBOOK = {
    0: "normal",
    1: "cortical erosion",
    2: "subchondral sclerosis",
    3: "subchondral cystic changes",
    4: "condylar flattening",
    5: "osteophyte",
    6: "other",
}
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_METADATA_ROWS = 100_000
MAX_ANNOTATION_SHAPES = 100_000
METADATA_HEADERS = {
    "anonymous_id",
    "patient_sex",
    "age_years",
    "age_group",
    "selected_slice_min",
    "selected_slice_max",
    "selected_slice_count",
    "label_L",
    "label_R",
}


class OD3DInputError(ValueError):
    """Only fixed codes, never values, paths or underlying exceptions."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise OD3DInputError(code)


def _read(path):
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            _fail("input_byte_limit")
        return data.decode("utf-8-sig")
    except OD3DInputError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError):
        raise OD3DInputError("unreadable_input") from None


def parse_codes(value, separator):
    """Known unique codes in sorted order; a blank string is unknown, not normal."""
    if not isinstance(value, str) or separator not in ("|", ","):
        _fail("invalid_codes")
    if not value.strip():
        return None
    tokens = [token.strip() for token in value.split(separator)]
    if any(re.fullmatch("[0-6]", token) is None for token in tokens):
        _fail("invalid_codes")
    codes = tuple(sorted(int(token) for token in tokens))
    if len(set(codes)) != len(codes):
        _fail("invalid_codes")
    if 0 in codes and len(codes) > 1:
        _fail("normal_abnormal_conflict")
    return codes


def _integer(value, code):
    if not isinstance(value, str) or re.fullmatch("[0-9]{1,20}", value.strip()) is None:
        _fail(code)
    return int(value)


def load_metadata(path):
    """Load source patient IDs and side codes; selected filename bounds stay intact."""
    try:
        reader = csv.DictReader(io.StringIO(_read(path), newline=""))
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or set(fields) != METADATA_HEADERS:
            _fail("invalid_metadata_headers")
        records = {}
        for number, row in enumerate(reader, 1):
            if number > MAX_METADATA_ROWS:
                _fail("metadata_row_limit")
            if None in row or any(value is None for value in row.values()):
                _fail("invalid_metadata_row")
            identity = row["anonymous_id"]
            if not identity or identity != identity.strip() or any(ord(c) < 32 for c in identity):
                _fail("invalid_patient_id")
            if identity in records:
                _fail("duplicate_patient_id")
            minimum, maximum, count = [
                _integer(row[key], "invalid_slice_info")
                for key in ("selected_slice_min", "selected_slice_max", "selected_slice_count")
            ]
            if maximum < minimum or not 1 <= count <= maximum - minimum + 1:
                _fail("invalid_slice_info")
            records[identity] = {
                "left": parse_codes(row["label_L"], "|"),
                "right": parse_codes(row["label_R"], "|"),
                "slice_min": minimum,
                "slice_max": maximum,
                "slice_count": count,
            }
        if not records:
            _fail("empty_metadata")
        return records
    except OD3DInputError:
        raise
    except (csv.Error, ValueError, TypeError, RecursionError):
        raise OD3DInputError("invalid_metadata") from None


def _unique_json(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("invalid_annotation_json")
        value[key] = item
    return value


def _dimension(container, root, name):
    alternate = "imageWidth" if name == "width" else "imageHeight"
    value = container.get(name, container.get(alternate, root.get(name, root.get(alternate))))
    if type(value) is not int or value <= 0:
        _fail("invalid_annotation_dimensions")
    return value


def _reference(container, root):
    value = container.get("image_path", root.get("image_path"))
    if not isinstance(value, str) or "\x00" in value:
        _fail("invalid_annotation_reference")
    # Discard both Windows and Unix directories; never resolve an author path.
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    if not basename or basename in (".", "..") or any(ord(c) < 32 for c in basename):
        _fail("invalid_annotation_reference")
    return basename


def _finite_coordinate(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def parse_annotation(path, expected_side):
    """Parse author root/frame rectangles with inclusive InstanceNumber ranges."""
    if expected_side not in ("L", "R"):
        _fail("invalid_expected_side")
    try:
        root = json.loads(
            _read(path),
            object_pairs_hook=_unique_json,
            parse_constant=lambda _: _fail("invalid_annotation_json"),
        )
    except OD3DInputError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise OD3DInputError("invalid_annotation_json") from None
    if not isinstance(root, dict):
        _fail("invalid_annotation_schema")
    frames = root.get("frames", [])
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        _fail("invalid_annotation_schema")
    records = []
    for container in [root, *frames]:
        shapes = container.get("shapes", [])
        if not isinstance(shapes, list):
            _fail("invalid_annotation_schema")
        if not shapes:
            continue
        width, height = [_dimension(container, root, name) for name in ("width", "height")]
        reference = _reference(container, root)
        for shape in shapes:
            if len(records) >= MAX_ANNOTATION_SHAPES:
                _fail("annotation_shape_limit")
            if not isinstance(shape, dict) or shape.get("shape_type") != "rectangle":
                _fail("invalid_annotation_rectangle")
            label = shape.get("label")
            match = (
                re.fullmatch(r"([^\-]+)-(L|R)-([0-9]{1,20})-([0-9]{1,20})", label)
                if isinstance(label, str)
                else None
            )
            if match is None:
                _fail("invalid_annotation_label")
            codes, side, start, end = match.groups()
            if side != expected_side:
                _fail("annotation_side_conflict")
            codes = parse_codes(codes, ",")
            start, end = int(start), int(end)
            if codes is None or start > end:
                _fail("invalid_annotation_range")
            points = shape.get("points")
            if (
                not isinstance(points, list)
                or len(points) != 2
                or any(not isinstance(point, list) or len(point) != 2 for point in points)
                or any(not _finite_coordinate(v) for point in points for v in point)
            ):
                _fail("invalid_annotation_coordinates")
            x1, x2 = sorted((float(points[0][0]), float(points[1][0])))
            y1, y2 = sorted((float(points[0][1]), float(points[1][1])))
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                _fail("invalid_annotation_bounds")
            records.append(
                {
                    "codes": codes,
                    "side": side,
                    "instance_start": start,
                    "instance_end": end,
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "reference_basename": reference,
                    "width": width,
                    "height": height,
                }
            )
    if not records:
        _fail("empty_annotation")
    return records
