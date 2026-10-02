#!/usr/bin/env python3
"""Shared TMJ labels and patient groups.

Research intake uses schema-v1 build_canonical_index(): explicit source/patient/
study/label-record IDs and confirmed applicability, with no name inference.
build_index() is the historical, explicitly legacy patient-name join.
Sagittal codes 1–3 and frontal 4–6 map to class indices 0–2; missing frontal
is allowed for sagittal-only intake. Canonical groups qualify patient_id with
source_id; legacy groups use patient_name.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code mapping helpers
# ---------------------------------------------------------------------------


def map_sagittal(code: int) -> int:
    """Sagittal code 1-3 → class index 0-2."""
    if type(code) is not int or code not in (1, 2, 3):
        raise ValueError("Invalid sagittal code (expected integer 1-3)")
    return code - 1


def map_frontal(code: int) -> int:
    """Frontal code 4-6 → class index 0-2."""
    if type(code) is not int or code not in (4, 5, 6):
        raise ValueError("Invalid frontal code (expected integer 4-6)")
    return code - 4


# ---------------------------------------------------------------------------
# Canonical intake: supplied identity and study-specific label applicability
# ---------------------------------------------------------------------------


class InputValidationError(ValueError):
    """Safe report: never include untrusted IDs, names, paths or exception text."""

    def __init__(self, diagnostics: List[Dict]):
        self.report = {
            "schema_version": 1,
            "stage": "intake",
            "ready": False,
            "intake_ready": False,
            "training_ready": False,
            "diagnostics": diagnostics,
        }
        super().__init__("Canonical input validation failed")


def patient_group_key(record: Dict) -> str:
    """Source-qualified supplied identity; patient_name is legacy only."""
    if "patient_id" in record or "source_id" in record:
        if not all(
            isinstance(record.get(k), str) and record[k] for k in ("source_id", "patient_id")
        ):
            raise ValueError("Missing canonical patient identity")
        return json.dumps([record["source_id"], record["patient_id"]], separators=(",", ":"))
    return record["patient_name"]


def build_canonical_index(
    input_path: str,
    dataset_root: str,
    *,
    sagittal_only: bool = True,
    require_crops: bool = True,
) -> List[Dict]:
    """Validate schema v1 and return stable records, without models or geometry I/O.

    IDs are supplied opaque tokens. Joins use (source_id, label_record_id), not
    names, DICOM identifiers, series numbers or automatically generated IDs.
    Crop-only inputs may omit series_path. Uncropped preparation requires it.
    ROI provenance validation belongs to the subsequent preparation stage.
    """
    diagnostics: List[Dict] = []

    def issue(code, collection="input", row=None, field=None):
        item = {"code": code, "collection": collection}
        if row is not None:
            item["row"] = row
        if field is not None:
            item["field"] = field
        diagnostics.append(item)

    def unique_json(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputValidationError([{"code": "duplicate_json_key", "collection": "input"}])
            result[key] = value
        return result

    try:
        with open(input_path, encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_json)
    except InputValidationError:
        raise
    except (OSError, ValueError, TypeError, RecursionError):
        raise InputValidationError([{"code": "unreadable_input", "collection": "input"}]) from None
    try:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            raise OSError
    except (OSError, RuntimeError, TypeError, ValueError):
        raise InputValidationError(
            [{"code": "invalid_dataset_root", "collection": "input"}]
        ) from None

    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise InputValidationError([{"code": "unsupported_schema", "collection": "input"}])
    if set(data) - {"schema_version", "studies", "labels"}:
        issue("unknown_fields")
    for name in ("studies", "labels"):
        if not isinstance(data.get(name), list) or not data[name]:
            issue("missing_records", name)
    if diagnostics:
        raise InputValidationError(diagnostics)

    def identity(row, field, collection, number):
        value = row.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None
        ):
            issue("invalid_identity", collection, number, field)
            return None
        return value

    label_by_key = {}
    for number, label in enumerate(data["labels"], 1):
        if not isinstance(label, dict):
            issue("invalid_record", "labels", number)
            continue
        if set(label) - {"source_id", "patient_id", "label_record_id", "labels"}:
            issue("unknown_fields", "labels", number)
        source, patient, label_id = (
            identity(label, f, "labels", number)
            for f in ("source_id", "patient_id", "label_record_id")
        )
        values = label.get("labels")
        mapped = {}
        if not isinstance(values, dict):
            issue("missing_plane", "labels", number, "labels")
            values = {}
        if set(values) - {"sagittal", "frontal"}:
            issue("unknown_fields", "labels", number, "labels")
        for plane, prefix, mapper in (
            ("sagittal", "sag", map_sagittal),
            ("frontal", "fr", map_frontal),
        ):
            if plane == "frontal" and plane not in values and sagittal_only:
                continue
            sides = values.get(plane)
            if not isinstance(sides, dict):
                issue("missing_plane", "labels", number, plane)
                continue
            if set(sides) - {"left", "right"}:
                issue("unknown_fields", "labels", number, plane)
            for side in ("left", "right"):
                try:
                    mapped[f"{prefix}_{side}"] = mapper(sides.get(side))
                except ValueError:
                    issue("invalid_label_code", "labels", number, f"{plane}.{side}")
        if source is not None and label_id is not None:
            key = (source, label_id)
            if key in label_by_key:
                issue("duplicate_label_record", "labels", number, "label_record_id")
            else:
                label_by_key[key] = (patient, mapped)

    records = []
    seen_studies = set()
    seen_paths = set()

    def checked_path(value, number, field, directory=False):
        if not isinstance(value, str) or not value:
            issue("invalid_path", "studies", number, field)
            return None
        try:
            relative = Path(value)
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):
                issue("path_escape", "studies", number, field)
                return None
            if not (path.is_dir() if directory else path.is_file()):
                issue(
                    "missing_directory" if directory else "missing_file", "studies", number, field
                )
                return None
        except (OSError, RuntimeError, ValueError):
            issue("invalid_path", "studies", number, field)
            return None
        if path in seen_paths:
            issue("duplicate_input_path", "studies", number, field)
        seen_paths.add(path)
        return str(path)

    for number, study in enumerate(data["studies"], 1):
        if not isinstance(study, dict):
            issue("invalid_record", "studies", number)
            continue
        if set(study) - {
            "source_id",
            "study_id",
            "patient_id",
            "label_record_id",
            "label_applicability",
            "series_path",
            "crops",
        }:
            issue("unknown_fields", "studies", number)
        source, study_id, patient, label_id = (
            identity(study, f, "studies", number)
            for f in ("source_id", "study_id", "patient_id", "label_record_id")
        )
        key = (source, study_id)
        if key in seen_studies:
            issue("duplicate_study", "studies", number, "study_id")
        seen_studies.add(key)
        if study.get("label_applicability") != "confirmed":
            issue("label_applicability_unconfirmed", "studies", number, "label_applicability")
        matched = label_by_key.get((source, label_id))
        mapped = {}
        if matched is None:
            issue("label_not_found", "studies", number, "label_record_id")
        else:
            if patient != matched[0]:
                issue("label_patient_mismatch", "studies", number, "patient_id")
            mapped = matched[1]
        record = {
            "source_id": source,
            "study_id": study_id,
            "patient_id": patient,
            "label_record_id": label_id,
            **mapped,
        }
        if source is not None and patient is not None:
            record["patient_key"] = patient_group_key(record)
        if "series_path" in study:
            record["dicom_dir"] = checked_path(
                study["series_path"], number, "series_path", directory=True
            )
        elif not require_crops:
            issue("missing_series", "studies", number, "series_path")
        if "crops" in study or require_crops:
            paths = study.get("crops")
            if not isinstance(paths, dict):
                issue("missing_crop", "studies", number, "crops")
            else:
                if set(paths) - {"left", "right"}:
                    issue("unknown_fields", "studies", number, "crops")
                record["crop_paths"] = {}
                for side in ("left", "right"):
                    if side not in paths:
                        issue("missing_crop", "studies", number, f"crops.{side}")
                    else:
                        record["crop_paths"][side] = checked_path(
                            paths[side], number, f"crops.{side}"
                        )
        records.append(record)
    if diagnostics:
        raise InputValidationError(diagnostics)
    return sorted(records, key=lambda record: (record["source_id"], record["study_id"]))


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def build_index(
    manifest_path: str = "data/dataset_cbct_public/manifest_private.json",
    labels_path: str = "data/tmj_position_labels.json",
    dataset_root: str = "data/dataset_cbct_public",
    cache_path: Optional[str] = None,
) -> List[Dict]:
    """
    Legacy name join; strict research entrypoints use build_canonical_index().

    Args:
        manifest_path: Path to manifest_private.json.
        labels_path:   Path to tmj_position_labels.json.
        dataset_root:  Root directory of the CBCT dataset (study_* folders live here).
        cache_path:    If given, save the resulting index as JSON for debugging.

    Returns:
        List of dicts with keys:
            study_id     – e.g. "study_0001"
            dicom_dir    – absolute path to the folder with .dcm files
            patient_name – raw patient name from manifest
            sag_right    – 0/1/2
            sag_left     – 0/1/2
            fr_right     – 0/1/2
            fr_left      – 0/1/2
    """
    manifest_path = Path(manifest_path)
    labels_path = Path(labels_path)
    dataset_root = Path(dataset_root)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    # Build name → labels dict
    label_by_name: Dict[str, Dict] = {}
    for patient in labels_data["patients"]:
        name = patient["name_raw"].strip()
        label_by_name[name] = patient["labels"]

    records: List[Dict] = []
    skipped = 0

    for study in manifest["studies"]:
        patient_name = study["patient_name"].strip()

        if patient_name not in label_by_name:
            logger.debug("Study without matching labels — skipped")
            skipped += 1
            continue

        lbl = label_by_name[patient_name]
        dicom_dir = dataset_root / study["study_id"]

        records.append(
            {
                "study_id": study["study_id"],
                "dicom_dir": str(dicom_dir),
                "patient_name": patient_name,
                "sag_right": map_sagittal(lbl["sagittal"]["right"]),
                "sag_left": map_sagittal(lbl["sagittal"]["left"]),
                "fr_right": map_frontal(lbl["frontal"]["right"]),
                "fr_left": map_frontal(lbl["frontal"]["left"]),
            }
        )

    logger.info(
        "build_index: %d records matched, %d studies skipped (no label match)",
        len(records),
        skipped,
    )

    if cache_path is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Saved private legacy index cache")

    return records


# ---------------------------------------------------------------------------
# Train / val split by patient
# ---------------------------------------------------------------------------


def split_by_patient(
    records: List[Dict],
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by source-qualified supplied patient identity (patient_name for legacy).

    All records with one patient group remain together; identity evidence is supplied.

    Args:
        records:     Output of build_index().
        split_ratio: Fraction of patients assigned to train.
        seed:        Random seed for reproducibility.

    Returns:
        (train_records, val_records)
    """
    patients = sorted(set(patient_group_key(r) for r in records))
    n = len(patients)
    if n == 0:
        return [], []
    if n == 1:
        logger.warning("split_by_patient: only one patient — all records go to train, val is empty")
        return records, []

    rng = random.Random(seed)
    rng.shuffle(patients)

    # At least 1 patient in train and 1 in val when n >= 2 (avoids empty val → crash in training loop)
    split_idx = int(n * split_ratio)
    split_idx = min(max(1, split_idx), n - 1)
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])

    train_records = [r for r in records if patient_group_key(r) in train_patients]
    val_records = [r for r in records if patient_group_key(r) in val_patients]

    logger.info(
        "split_by_patient: train=%d records (%d patients) / val=%d records (%d patients)",
        len(train_records),
        len(train_patients),
        len(val_records),
        len(val_patients),
    )
    return train_records, val_records


# ---------------------------------------------------------------------------
# Binarize labels
# ---------------------------------------------------------------------------


def binarize_labels(records: List[Dict], crop_dir: str) -> List[Dict]:
    """
    Explode validated class records into side-specific binary records.

    Each input record produces 2 output records (left + right).
    Class 0 → 0, classes 1/2 → 1. Absent frontal remains absent; invalid classes raise.

    Args:
        records:  Output of build_index().
        crop_dir: Root directory containing detector-generated NIfTI crops,
                  structured as {crop_dir}/{study_id}/{study_id}_{side}.nii.gz

    Returns:
        List of dicts with keys:
            study_id    – e.g. "study_0001"
            patient_key/source_id/patient_id (or legacy patient_name)
            side        – "left" | "right"
            sag         – 0 (central) or 1 (non-central)
            fr          – optional 0/1, only when the corresponding class is present
            crop_path   – absolute path to the NIfTI crop file
    """
    crop_dir = Path(crop_dir).resolve()
    binary_records: List[Dict] = []

    for rec in records:
        study_id = rec["study_id"]
        identity = {
            key: rec[key]
            for key in ("patient_name", "source_id", "patient_id", "patient_key", "label_record_id")
            if key in rec
        }
        for side in ("left", "right"):
            row = {"study_id": study_id, **identity, "side": side}
            for plane in ("sag", "fr"):
                key = f"{plane}_{side}"
                if plane == "fr" and key not in rec:
                    continue
                value = rec.get(key)
                if type(value) is not int or value not in (0, 1, 2):
                    raise ValueError("Invalid class for binary conversion")
                row[plane] = int(value != 0)
            if "crop_paths" in rec:
                row["crop_path"] = rec["crop_paths"][side]
            else:
                row["crop_path"] = str(crop_dir / study_id / f"{study_id}_{side}.nii.gz")
            binary_records.append(row)

    logger.info(
        "binarize_labels: %d records → %d binary side-records", len(records), len(binary_records)
    )
    return binary_records


# ---------------------------------------------------------------------------
# Stratified Group K-Fold (patient-level groups, sagittal binary strat label)
# ---------------------------------------------------------------------------


def patient_sagittal_strat_labels(binary_records: List[Dict]) -> Dict[str, int]:
    """
    Per-patient stratification label for sagittal binary (0=central, 1=non-central).

    Policy: ``max`` over all side-records for that patient group — if any side is
    non-central (1), the patient is treated as positive for stratification.
    This keeps asymmetric (0/1) patients in the positive stratum.
    """
    strat: Dict[str, int] = {}
    for rec in binary_records:
        p = patient_group_key(rec)
        s = int(rec["sag"])
        strat[p] = max(strat.get(p, 0), s)
    return strat


def iter_stratified_group_kfold_indices(
    binary_records: List[Dict],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield ``(train_idx, val_idx)`` index arrays into ``binary_records``.

    - **Groups:** source-qualified patient IDs; legacy records use patient_name.
    - **Stratification:** per-patient sagittal binary label from
      :func:`patient_sagittal_strat_labels`.

    Requires ``scikit-learn`` (``StratifiedGroupKFold``).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n = len(binary_records)
    if n == 0:
        return

    indices = np.arange(n, dtype=np.int64)
    groups = np.array([patient_group_key(rec) for rec in binary_records], dtype=object)
    strat_map = patient_sagittal_strat_labels(binary_records)
    y = np.array([strat_map[patient_group_key(rec)] for rec in binary_records], dtype=np.int64)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    for train_idx, val_idx in sgkf.split(indices, y, groups):
        yield train_idx, val_idx
