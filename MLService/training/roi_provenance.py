"""Bounded paired-heatmap ROI geometry and privacy-minimized technical passports.

Passports describe technical provenance, not anonymity, anatomical quality, or
independence from detector training. Raw-source verification requires the source.
"""

import hashlib
import io
import json
import logging
import warnings
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

# Supported research profile, checked before native pixel decode / padding.
MAX_SOURCE_ENTRIES = 4096
MAX_SOURCE_SLICES = 1024
MAX_SOURCE_DIMENSION = 1024
MAX_SOURCE_VOXELS = 512**3
MAX_SOURCE_FILE_BYTES = 64 * 1024**2
MAX_SOURCE_BYTES = 1024**3
MAX_CROP_SIZE = 256
MAX_PASSPORT_BYTES = 64 * 1024
MAX_ROI_FILE_BYTES = 128 * 1024**2
NATIVE_TRANSFER_SYNTAXES = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"}
IGNORED_SOURCE_SIDECARS = {".json", ".txt", ".csv"}
SUPPORTED_PROFILE_HELP = (
    "Supported research ROI profile: a direct single-series directory of Part-10 "
    "native uncompressed 8/16/32-bit monochrome single-frame DICOM; compressed "
    "syntax fails with compressed_source_unsupported. Valid Study/Series UID required. "
    "Limits: 4096 direct entries, 1024 slices, 1024 rows/columns, 512^3 source voxels, "
    "64 MiB per source file, 1 GiB source bytes; crop edge <=256, float32 NIfTI-1 "
    "<=128 MiB and passport <=64 KiB. No recursive directory scan."
)

DETECTOR_FAMILY = "paired-single-channel-heatmap-v1"
PREPROCESSING = {
    "version": "paired-heatmap-crop-v1",
    "slice_order": "InstanceNumber-ascending",
    "rescale": "per-slice-slope-intercept-default-1-0",
    "percentiles": [2, 98],
    "target_shape": [96, 128, 128],
    "zoom_order": 1,
    "zoom_mode": "constant",
    "zoom_cval": 0.0,
    "zoom_prefilter": True,
    "zoom_grid_mode": False,
    "center_mapping": "floor(index*original_shape/target_shape)",
    "crop_values": "rescaled-source-float32",
    "padding_value": 0.0,
}


def preprocessing_options(crop_size=128):
    if type(crop_size) is not int or crop_size <= 0:
        _fail("invalid_crop_request")
    if crop_size > MAX_CROP_SIZE:
        _fail("crop_size_limit")
    return {**PREPROCESSING, "crop_size": crop_size}


class ROIValidationError(ValueError):
    """Only fixed codes may cross this boundary; never retain source exceptions."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise ROIValidationError(code)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ROIValidationError("invalid_passport_schema") from None


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail("invalid_passport_schema")


def _hash(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        _fail("invalid_fingerprint")


def sha256_file(path):
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        raise ROIValidationError("unreadable_roi_input") from None


def passport_path(path):
    return Path(str(path) + ".passport.json")


def study_key(record):
    try:
        values = [record["source_id"], record["study_id"]]
        if any(not isinstance(v, str) or not v for v in values):
            _fail("invalid_study_identity")
        return _digest(values)
    except (KeyError, TypeError):
        raise ROIValidationError("invalid_study_identity") from None


def _shape(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(v) is not int or v <= 0 or v > np.iinfo(np.int64).max for v in value)
    ):
        _fail("invalid_geometry")
    return np.array(value, dtype=int)


def _numbers(value, shape):
    """Only bounded vectors/matrices; never recurse into an arbitrary JSON tree."""
    try:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != shape[0]:
            _fail("invalid_geometry")
        if len(shape) == 1:
            flat = list(value)
        else:
            if any(
                not isinstance(row, (list, tuple, np.ndarray)) or len(row) != shape[1]
                for row in value
            ):
                _fail("invalid_geometry")
            flat = [item for row in value for item in row]
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, float, np.integer, np.floating))
            for item in flat
        ):
            _fail("invalid_geometry")
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            _fail("invalid_geometry")
        return array
    except (ValueError, TypeError, OverflowError):
        raise ROIValidationError("invalid_geometry") from None


def _check_source_shape(shape):
    shape = _shape(list(shape))
    if shape[0] > MAX_SOURCE_SLICES:
        _fail("source_slice_limit")
    if np.any(shape[1:] > MAX_SOURCE_DIMENSION):
        _fail("source_dimension_limit")
    if int(shape[0]) * int(shape[1]) * int(shape[2]) > MAX_SOURCE_VOXELS:
        _fail("source_voxel_limit")
    return shape


def voxel_geometry(shape):
    shape = _check_source_shape(shape)
    return {
        "shape": shape.tolist(),
        "slice_count": int(shape[0]),
        "slice_order": "InstanceNumber-ascending",
        "coordinate_space": "voxel",
        "affine": np.eye(4).tolist(),
        "orientation": None,
        "pixel_spacing": None,
        "slice_step": None,
    }


def _validate_geometry(geometry):
    _keys(
        geometry,
        {
            "shape",
            "slice_count",
            "slice_order",
            "coordinate_space",
            "affine",
            "orientation",
            "pixel_spacing",
            "slice_step",
        },
    )
    shape = _check_source_shape(geometry["shape"])
    if (
        type(geometry["slice_count"]) is not int
        or geometry["slice_count"] != shape[0]
        or geometry["slice_order"] != "InstanceNumber-ascending"
    ):
        _fail("invalid_geometry")
    affine = _numbers(geometry["affine"], (4, 4))
    if not np.array_equal(affine[3], [0, 0, 0, 1]):
        _fail("invalid_geometry")
    if geometry["coordinate_space"] == "voxel":
        if any(
            geometry[k] is not None for k in ("orientation", "pixel_spacing", "slice_step")
        ) or not np.array_equal(affine, np.eye(4)):
            _fail("invalid_geometry")
    elif geometry["coordinate_space"] == "physical-ras":
        orientation = _numbers(geometry["orientation"], (6,))
        spacing = _numbers(geometry["pixel_spacing"], (2,))
        step = _numbers(geometry["slice_step"], (3,))
        u, v = orientation[:3], orientation[3:]
        if (
            shape[0] < 2
            or np.any(spacing <= 0)
            or not np.allclose(
                [np.linalg.norm(u), np.linalg.norm(v), np.dot(u, v)], [1, 1, 0], atol=1e-5
            )
        ):
            _fail("unsupported_source_geometry")
        if np.linalg.norm(step) <= 1e-6 or not np.allclose(
            [np.dot(step, u), np.dot(step, v)], 0, atol=1e-4
        ):
            _fail("unsupported_source_geometry")
        ras = np.diag([-1, -1, 1])
        expected = ras @ np.column_stack((step, v * spacing[0], u * spacing[1]))
        if not np.allclose(affine[:3, :3], expected, atol=1e-5):
            _fail("invalid_geometry")
    else:
        _fail("invalid_geometry")


def _discover_source_files(dicom_dir):
    """Dedicated single-series directory; direct entries only, bounded inventory."""
    files = []
    total = 0
    for count, path in enumerate(Path(dicom_dir).iterdir(), 1):
        if count > MAX_SOURCE_ENTRIES:
            _fail("source_entry_limit")
        if path.is_symlink() or not path.is_file():
            _fail("unsupported_source_entry")
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(132)
        preamble = len(prefix) == 132 and prefix[128:] == b"DICM"
        if not preamble and path.suffix.lower() in IGNORED_SOURCE_SIDECARS:
            continue
        if not preamble and path.suffix.lower() != ".dcm" and path.suffix:
            _fail("unrecognized_source_file")
        if size > MAX_SOURCE_FILE_BYTES:
            _fail("source_file_byte_limit")
        total += size
        if total > MAX_SOURCE_BYTES:
            _fail("source_byte_limit")
        files.append(path)
        if len(files) > MAX_SOURCE_SLICES:
            _fail("source_slice_limit")
    if not files:
        _fail("missing_source_slices")
    return files


def _validated_series_uids(datasets):
    for field in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"):
        values = [str(getattr(ds, field, "")) for ds in datasets]
        if field == "FrameOfReferenceUID" and not any(values):
            continue
        if any(not value or not pydicom.uid.UID(value).is_valid for value in values):
            _fail("invalid_source_series_uid")
        if len(set(values)) != 1:
            _fail("mixed_source_series")


def _check_native_file_meta(data):
    # dcmread inflates Deflated Explicit VR even with stop_before_pixels=True.
    # Read only Part-10 group 0002 from these same bounded, immutable bytes.
    if data[128:132] != b"DICM":
        _fail("unsupported_source_encoding")
    stream = io.BytesIO(data)
    stream.seek(132)
    meta = pydicom.filereader.read_dataset(
        stream,
        is_implicit_VR=False,
        is_little_endian=True,
        stop_when=lambda tag, vr, length: tag.group != 0x0002,
    )
    syntax = pydicom.uid.UID(str(getattr(meta, "TransferSyntaxUID", "")))
    if str(syntax) not in NATIVE_TRANSFER_SYNTAXES:
        compressed = syntax.is_transfer_syntax and (
            syntax.is_compressed or syntax == pydicom.uid.DeflatedExplicitVRLittleEndian
        )
        _fail("compressed_source_unsupported" if compressed else "unsupported_source_encoding")


def _load_validated_series(dicom_dir):
    """Validate all metadata/budgets BEFORE any native pixel decode/allocation."""
    try:
        files = _discover_source_files(dicom_dir)
        slices = []
        total_bytes = 0
        for path in files:
            with path.open("rb") as stream:
                data = stream.read(min(MAX_SOURCE_FILE_BYTES, MAX_SOURCE_BYTES - total_bytes) + 1)
            if len(data) > MAX_SOURCE_FILE_BYTES:
                _fail("source_file_byte_limit")
            total_bytes += len(data)
            if total_bytes > MAX_SOURCE_BYTES:
                _fail("source_byte_limit")
            _check_native_file_meta(data)
            ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
            number = float(ds.InstanceNumber)
            if not np.isfinite(number) or not number.is_integer():
                _fail("invalid_slice_order")
            slices.append((int(number), ds, hashlib.sha256(data).hexdigest(), data))
        slices.sort(key=lambda item: item[0])
        if len({item[0] for item in slices}) != len(slices):
            _fail("invalid_slice_order")
        datasets = [item[1] for item in slices]
        _validated_series_uids(datasets)
        first = datasets[0]
        rows, cols = int(first.Rows), int(first.Columns)
        shape = _check_source_shape([len(slices), rows, cols]).tolist()
        for ds in datasets:
            syntax = str(ds.file_meta.TransferSyntaxUID)
            if pydicom.uid.UID(syntax).is_compressed:
                _fail("compressed_source_unsupported")
            if syntax not in NATIVE_TRANSFER_SYNTAXES:
                _fail("unsupported_source_encoding")
            if (
                int(getattr(ds, "NumberOfFrames", 1)) != 1
                or int(ds.SamplesPerPixel) != 1
                or ds.PhotometricInterpretation != first.PhotometricInterpretation
                or ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2")
                or (int(ds.Rows), int(ds.Columns)) != (rows, cols)
            ):
                _fail("unsupported_source_series")
            if (
                int(ds.BitsAllocated) not in (8, 16, 32)
                or not 1 <= int(ds.BitsStored) <= int(ds.BitsAllocated)
                or int(ds.HighBit) != int(ds.BitsStored) - 1
                or int(ds.PixelRepresentation) not in (0, 1)
            ):
                _fail("unsupported_source_encoding")
            slope, intercept = (
                float(getattr(ds, "RescaleSlope", 1)),
                float(getattr(ds, "RescaleIntercept", 0)),
            )
            if not np.isfinite([slope, intercept]).all() or slope == 0:
                _fail("invalid_source_rescale")
        geometry = voxel_geometry(shape)
        tags = ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing")
        presence = [[hasattr(ds, field) for field in tags] for ds in datasets]
        if any(any(flags) for flags in presence):
            if not all(all(flags) for flags in presence) or len(datasets) < 2:
                _fail("unsupported_source_geometry")
            orientation = _numbers(list(first.ImageOrientationPatient), (6,))
            spacing = _numbers(list(first.PixelSpacing), (2,))
            positions = np.stack([_numbers(list(ds.ImagePositionPatient), (3,)) for ds in datasets])
            step = positions[1] - positions[0]
            if not np.allclose(np.diff(positions, axis=0), step, atol=1e-3, rtol=0):
                _fail("irregular_slice_positions")
            for ds in datasets:
                if not np.allclose(
                    _numbers(list(ds.ImageOrientationPatient), (6,)), orientation, atol=1e-5, rtol=0
                ) or not np.allclose(
                    _numbers(list(ds.PixelSpacing), (2,)), spacing, atol=1e-5, rtol=0
                ):
                    _fail("mixed_source_geometry")
            lps = np.eye(4)
            lps[:3, :3] = np.column_stack(
                (step, orientation[3:] * spacing[0], orientation[:3] * spacing[1])
            )
            lps[:3, 3] = positions[0]
            affine = np.diag([-1, -1, 1, 1]) @ lps
            geometry.update(
                coordinate_space="physical-ras",
                affine=affine.tolist(),
                orientation=orientation.tolist(),
                pixel_spacing=spacing.tolist(),
                slice_step=step.tolist(),
            )
        _validate_geometry(geometry)
        volume = np.empty(shape, dtype=np.float32)
        for index, (_, header, _, data) in enumerate(slices):
            ds = pydicom.dcmread(io.BytesIO(data))
            expected_bytes = rows * cols * (int(header.BitsAllocated) // 8)
            if len(ds.PixelData) != expected_bytes + expected_bytes % 2:
                _fail("invalid_native_pixel_length")
            arr = ds.pixel_array.astype(np.float32)
            if arr.shape != (rows, cols):
                _fail("unsupported_source_series")
            arr = arr * float(getattr(header, "RescaleSlope", 1)) + float(
                getattr(header, "RescaleIntercept", 0)
            )
            if not np.isfinite(arr).all():
                _fail("invalid_source_pixels")
            volume[index] = arr
        fingerprint = _digest(
            {"version": "ordered-dicom-bytes-v1", "slices": [item[2] for item in slices]}
        )
        return volume, {"sha256": fingerprint, "geometry": geometry}
    except ROIValidationError:
        raise
    except Exception:
        raise ROIValidationError("unreadable_source_series") from None


def load_validated_series(dicom_dir):
    """Keep decoder diagnostics private; warn-only malformed data is unsupported."""
    decoder_logger = logging.getLogger("pydicom")
    old_level = decoder_logger.level
    try:
        # pydicom emits the same untrusted header values to logging and warnings.
        # This synchronous reader does not export either diagnostic stream.
        decoder_logger.setLevel(logging.CRITICAL + 1)
        with warnings.catch_warnings(record=True) as diagnostics:
            warnings.simplefilter("always")
            result = _load_validated_series(dicom_dir)
        if diagnostics:
            _fail("unsupported_source_encoding")
        return result
    finally:
        decoder_logger.setLevel(old_level)


def _crop_transform(shape, center, size, affine):
    shape = _check_source_shape(shape)
    preprocessing_options(size)
    if (
        type(size) is not int
        or size <= 0
        or not isinstance(center, (list, tuple, np.ndarray))
        or len(center) != 3
        or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in center
        )
    ):
        _fail("invalid_crop_request")
    if any(v < 0 or v >= limit for v, limit in zip(center, shape)):
        _fail("invalid_crop_center")
    center = np.array(center, dtype=int)
    if np.any(center < 0) or np.any(center >= shape):
        _fail("invalid_crop_center")
    start = center - size // 2
    low, high = np.maximum(start, 0), np.minimum(start + size, shape)
    padding = np.column_stack((low - start, start + size - high))
    shift = np.eye(4)
    shift[:3, 3] = start
    return {
        "shape": [size] * 3,
        "center": center.tolist(),
        "requested_start": start.tolist(),
        "source_bounds": np.column_stack((low, high)).tolist(),
        "padding": padding.tolist(),
        "affine": (np.asarray(affine) @ shift).tolist(),
    }


def extract_fixed_crop(volume, center, crop_size=128):
    if np.asarray(volume).ndim != 3:
        _fail("invalid_source_pixels")
    transform = _crop_transform(volume.shape, center, crop_size, np.eye(4))
    bounds, padding = np.array(transform["source_bounds"]), np.array(transform["padding"])
    crop = volume[tuple(slice(int(a), int(b)) for a, b in bounds)]
    return np.pad(crop, padding, mode="constant", constant_values=0), transform


def _validate_detectors(detectors):
    _keys(detectors, {"family", "left_sha256", "right_sha256"})
    if detectors["family"] != DETECTOR_FAMILY:
        _fail("unsupported_detector_family")
    _hash(detectors["left_sha256"])
    _hash(detectors["right_sha256"])


def _check_preprocessing(options):
    if not isinstance(options, dict) or "crop_size" not in options:
        _fail("stale_preprocessing")
    expected = preprocessing_options(options["crop_size"])
    if set(options) != set(expected):
        _fail("stale_preprocessing")
    for key, value in expected.items():
        actual = options[key]
        if isinstance(value, list):
            if (
                not isinstance(actual, list)
                or len(actual) != len(value)
                or any(type(a) is not type(b) or a != b for a, b in zip(actual, value))
            ):
                _fail("stale_preprocessing")
        elif type(actual) is not type(value) or actual != value:
            _fail("stale_preprocessing")


def _serialized_sform(affine):
    with np.errstate(over="ignore", invalid="ignore"):
        stored = np.array(affine, dtype=np.float32).astype(float)
    if not np.isfinite(stored).all():
        _fail("unsupported_nifti_geometry")
    stored[3] = [0, 0, 0, 1]
    return stored


def _validate_common(passport, record, expected_detectors, expected_preprocessing):
    _keys(
        passport,
        {
            "schema_version",
            "study_key",
            "source",
            "detectors",
            "preprocessing",
            "input_sha256",
            "side",
            "crop",
        },
    )
    if type(passport["schema_version"]) is not int or passport["schema_version"] != 1:
        _fail("unsupported_roi_schema")
    if passport["study_key"] != study_key(record):
        _fail("roi_study_mismatch")
    _keys(passport["source"], {"sha256", "geometry"})
    _hash(passport["source"]["sha256"])
    _validate_geometry(passport["source"]["geometry"])
    _validate_detectors(passport["detectors"])
    if expected_detectors is not None:
        _validate_detectors(expected_detectors)
    if expected_preprocessing is not None:
        _check_preprocessing(expected_preprocessing)
    if expected_detectors is not None and _json(passport["detectors"]) != _json(expected_detectors):
        _fail("stale_detector_checkpoint")
    options = passport["preprocessing"]
    if not isinstance(options, dict) or "crop_size" not in options:
        _fail("stale_preprocessing")
    _check_preprocessing(options)
    if _json(options) != _json(preprocessing_options(options["crop_size"])) or (
        expected_preprocessing is not None and _json(options) != _json(expected_preprocessing)
    ):
        _fail("stale_preprocessing")
    common = {key: passport[key] for key in ("study_key", "source", "detectors", "preprocessing")}
    if passport["input_sha256"] != _digest(common):
        _fail("stale_roi_input")
    return common


def write_roi_crop(volume, center, crop_size, path, *, side, record, source, detectors):
    """Write a float32 NIfTI and its whitelisted passport from validated source."""
    _keys(source, {"sha256", "geometry"})
    _hash(source["sha256"])
    _validate_geometry(source["geometry"])
    _validate_detectors(detectors)
    if (
        side not in ("left", "right")
        or list(volume.shape) != source["geometry"]["shape"]
        or not np.isfinite(volume).all()
    ):
        _fail("invalid_source_pixels")
    crop, _ = extract_fixed_crop(volume, center, crop_size)
    transform = _crop_transform(volume.shape, center, crop_size, source["geometry"]["affine"])
    common = {
        "study_key": study_key(record),
        "source": source,
        "detectors": detectors,
        "preprocessing": preprocessing_options(crop_size),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _serialized_sform(transform["affine"])
    image = nib.Nifti1Image(crop.astype(np.float32), np.array(transform["affine"]))
    physical = source["geometry"]["coordinate_space"] == "physical-ras"
    image.set_sform(np.array(transform["affine"]), code=1 if physical else 0)
    image.set_qform(None, code=0)
    image.header.set_xyzt_units("mm" if physical else "unknown")
    nib.save(image, path)
    passport = {
        "schema_version": 1,
        **common,
        "input_sha256": _digest(common),
        "side": side,
        "crop": {**transform, "sha256": sha256_file(path)},
    }
    passport_path(path).write_text(_json(passport), encoding="utf-8")
    return passport


def _load_passport(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("invalid_passport_schema")
            result[key] = value
        return result

    def invalid_constant(value):
        _fail("invalid_passport_schema")

    try:
        if passport_path(path).stat().st_size > MAX_PASSPORT_BYTES:
            _fail("passport_byte_limit")
        with passport_path(path).open("rb") as stream:
            encoded = stream.read(MAX_PASSPORT_BYTES + 1)
        if len(encoded) > MAX_PASSPORT_BYTES:
            _fail("passport_byte_limit")
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=invalid_constant,
        )
    except ROIValidationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise ROIValidationError("unreadable_roi_passport") from None


def _check_roi_header(path, expected_bytes):
    # Only the writer's single-file NIfTI-1 profile: blank text/extension fields,
    # one metadata-free gzip member, and exactly the declared float32 voxel bytes.
    # Bound decompression before nibabel can interpret arbitrary metadata/offsets.
    path = Path(path)
    decoder = None
    if str(path).endswith(".gz"):
        with path.open("rb") as stream:
            data = stream.read(MAX_ROI_FILE_BYTES + 1)
        if len(data) > MAX_ROI_FILE_BYTES:
            _fail("roi_file_byte_limit")
        if len(data) < 10 or data[:8] != b"\x1f\x8b\x08\0\0\0\0\0":
            _fail("unsupported_roi_header")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        encoded = decoder.decompress(data, 352)
    else:
        with path.open("rb") as stream:
            encoded = stream.read(352)
        if path.stat().st_size != expected_bytes:
            _fail("unsupported_roi_header")
    if len(encoded) != 352 or encoded[348:] != b"\0" * 4:
        _fail("unsupported_roi_header")
    header = nib.Nifti1Header(binaryblock=encoded[:348], check=False)
    if bytes(header["magic"]) != b"n+1\0" or float(header["vox_offset"]) != 352:
        _fail("unsupported_roi_header")
    if any(
        bytes(header[name]).strip(b"\0")
        for name in ("descrip", "aux_file", "intent_name", "db_name", "data_type")
    ):
        _fail("unsupported_roi_header")
    if decoder is not None:
        remainder = decoder.decompress(decoder.unconsumed_tail, expected_bytes - 352 + 1)
        if (
            len(remainder) != expected_bytes - 352
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            _fail("unsupported_roi_header")


def validate_roi_pair(
    record, *, expected_detectors=None, expected_preprocessing=None, recheck_source=True
):
    """Validate one canonical pair; absence of raw source is explicit in result."""
    shared = None
    sizes = []
    for side in ("left", "right"):
        try:
            path = record["crop_paths"][side]
        except (KeyError, TypeError):
            raise ROIValidationError("missing_roi_pair") from None
        passport = _load_passport(path)
        common = _validate_common(passport, record, expected_detectors, expected_preprocessing)
        if shared is not None and _json(common) != _json(shared):
            _fail("roi_pair_mismatch")
        shared = common
        if passport["side"] != side:
            _fail("roi_side_mismatch")
        crop = passport["crop"]
        _keys(
            crop,
            {"shape", "center", "requested_start", "source_bounds", "padding", "affine", "sha256"},
        )
        shape = _shape(crop["shape"])
        if len(set(shape)) != 1 or shape[0] != common["preprocessing"]["crop_size"]:
            _fail("invalid_crop_shape")
        transform = _crop_transform(
            common["source"]["geometry"]["shape"],
            crop["center"],
            int(shape[0]),
            common["source"]["geometry"]["affine"],
        )
        for field, dimensions in (
            ("requested_start", (3,)),
            ("source_bounds", (3, 2)),
            ("padding", (3, 2)),
            ("affine", (4, 4)),
        ):
            _numbers(crop[field], dimensions)
        if _json({key: crop[key] for key in transform}) != _json(transform):
            _fail("invalid_crop_transform")
        sizes.append(int(shape[0]))
        _hash(crop["sha256"])
        try:
            if Path(path).stat().st_size > MAX_ROI_FILE_BYTES:
                _fail("roi_file_byte_limit")
        except ROIValidationError:
            raise
        except (OSError, TypeError, ValueError):
            raise ROIValidationError("unreadable_roi_input") from None
        if sha256_file(path) != crop["sha256"]:
            _fail("roi_checksum_mismatch")
        try:
            _check_roi_header(path, 352 + 4 * int(np.prod(shape)))
            image = nib.load(path)
            if (
                list(image.shape) != crop["shape"]
                or type(image.header) is not nib.Nifti1Header
                or image.get_data_dtype() != np.dtype(np.float32)
                or not np.isfinite(np.asarray(image.dataobj)).all()
            ):
                _fail("invalid_crop_pixels")
            physical = common["source"]["geometry"]["coordinate_space"] == "physical-ras"
            if (
                not np.array_equal(image.get_sform(), _serialized_sform(crop["affine"]))
                or int(image.header["sform_code"]) != (1 if physical else 0)
                or int(image.header["qform_code"]) != 0
                or image.header.get_xyzt_units()[0] != ("mm" if physical else "unknown")
            ):
                _fail("roi_affine_mismatch")
        except ROIValidationError:
            raise
        except Exception:
            raise ROIValidationError("unreadable_roi_nifti") from None
    if sizes[0] != sizes[1]:
        _fail("roi_pair_mismatch")
    rechecked = False
    if recheck_source and record.get("dicom_dir"):
        _, actual = load_validated_series(record["dicom_dir"])
        if _json(actual) != _json(shared["source"]):
            _fail("stale_source_series")
        rechecked = True
    return {
        "source_rechecked": rechecked,
        "coordinate_space": shared["source"]["geometry"]["coordinate_space"],
        "detector_training_identity": "unknown",
        "roi_contract": {
            "detectors": shared["detectors"],
            "preprocessing": shared["preprocessing"],
        },
    }


def require_shared_roi_contract(contracts):
    """One portable generation contract, with no source geometry/IDs/paths.

    Used by preflight, training and inference; classifier preprocessing remains
    separately versioned in classifier checkpoints.
    """
    if (
        not isinstance(contracts, list)
        or not contracts
        or any(value is None for value in contracts)
    ):
        _fail("missing_roi_contract")
    expected = None
    for contract in contracts:
        _keys(contract, {"detectors", "preprocessing"})
        _validate_detectors(contract["detectors"])
        _check_preprocessing(contract["preprocessing"])
        encoded = _json(contract)
        if expected is not None and expected != encoded:
            _fail("inconsistent_roi_contract")
        expected = encoded
    return json.loads(expected)
