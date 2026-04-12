#!/usr/bin/env python3
"""
TMJ Position Label Table

Reads manifest_private.json (study_id, patient_name) and
tmj_position_labels.json (name_raw, sagittal/frontal labels),
joins them by exact match of patient_name == name_raw,
and returns a list of records ready for training.

Label mapping:
    sagittal codes 1-3  →  class index 0-2  (code - 1)
    frontal  codes 4-6  →  class index 0-2  (code - 4)

Split is performed strictly by patient_name to avoid data leakage
(multiple studies of the same patient always land in the same split).
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code mapping helpers
# ---------------------------------------------------------------------------

def map_sagittal(code: int) -> int:
    """Sagittal code 1-3 → class index 0-2."""
    if code not in (1, 2, 3):
        raise ValueError(f"Invalid sagittal code: {code!r} (expected 1-3)")
    return code - 1


def map_frontal(code: int) -> int:
    """Frontal code 4-6 → class index 0-2."""
    if code not in (4, 5, 6):
        raise ValueError(f"Invalid frontal code: {code!r} (expected 4-6)")
    return code - 4


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
    Build training index by joining manifest and labels on patient_name == name_raw.

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
            logger.debug(
                "No labels for patient %r (study %s) — skipped",
                patient_name, study["study_id"],
            )
            skipped += 1
            continue

        lbl = label_by_name[patient_name]
        dicom_dir = dataset_root / study["study_id"]

        records.append({
            "study_id": study["study_id"],
            "dicom_dir": str(dicom_dir),
            "patient_name": patient_name,
            "sag_right": map_sagittal(lbl["sagittal"]["right"]),
            "sag_left":  map_sagittal(lbl["sagittal"]["left"]),
            "fr_right":  map_frontal(lbl["frontal"]["right"]),
            "fr_left":   map_frontal(lbl["frontal"]["left"]),
        })

    logger.info(
        "build_index: %d records matched, %d studies skipped (no label match)",
        len(records), skipped,
    )

    if cache_path is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Saved index cache → %s", cache_path)

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
    Split records into train/val strictly by patient_name.

    All studies of the same patient go to the same split, preventing leakage.

    Args:
        records:     Output of build_index().
        split_ratio: Fraction of patients assigned to train.
        seed:        Random seed for reproducibility.

    Returns:
        (train_records, val_records)
    """
    patients = sorted(set(r["patient_name"] for r in records))
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

    train_records = [r for r in records if r["patient_name"] in train_patients]
    val_records = [r for r in records if r["patient_name"] in val_patients]

    logger.info(
        "split_by_patient: train=%d records (%d patients) / val=%d records (%d patients)",
        len(train_records), len(train_patients),
        len(val_records), len(val_patients),
    )
    return train_records, val_records


# ---------------------------------------------------------------------------
# Binarize labels
# ---------------------------------------------------------------------------

def binarize_labels(records: List[Dict], crop_dir: str) -> List[Dict]:
    """
    Explode 4-head multi-class records into side-specific binary records.

    Each input record produces 2 output records (left + right).
    Labels are binarized: class 0 (central) → 0, classes 1 or 2 → 1 (non-central).

    Args:
        records:  Output of build_index().
        crop_dir: Root directory containing detector-generated NIfTI crops,
                  structured as {crop_dir}/{study_id}/{study_id}_{side}.nii.gz

    Returns:
        List of dicts with keys:
            study_id    – e.g. "study_0001"
            patient_name
            side        – "left" | "right"
            sag         – 0 (central) or 1 (non-central)
            fr          – 0 (central) or 1 (non-central)
            crop_path   – absolute path to the NIfTI crop file
    """
    crop_dir = Path(crop_dir).resolve()
    binary_records: List[Dict] = []

    for rec in records:
        study_id = rec["study_id"]
        patient_name = rec["patient_name"]

        for side in ("left", "right"):
            sag_val = rec[f"sag_{side}"]
            fr_val = rec[f"fr_{side}"]

            binary_records.append({
                "study_id": study_id,
                "patient_name": patient_name,
                "side": side,
                "sag": 0 if sag_val == 0 else 1,
                "fr": 0 if fr_val == 0 else 1,
                "crop_path": str(crop_dir / study_id / f"{study_id}_{side}.nii.gz"),
            })

    logger.info("binarize_labels: %d records → %d binary side-records", len(records), len(binary_records))
    return binary_records


# ---------------------------------------------------------------------------
# Stratified Group K-Fold (patient-level groups, sagittal binary strat label)
# ---------------------------------------------------------------------------


def patient_sagittal_strat_labels(binary_records: List[Dict]) -> Dict[str, int]:
    """
    Per-patient stratification label for sagittal binary (0=central, 1=non-central).

    Policy: ``max`` over all side-records for that patient — if any side is
    non-central (1), the patient is treated as positive for stratification.
    This keeps asymmetric (0/1) patients in the positive stratum.
    """
    strat: Dict[str, int] = {}
    for rec in binary_records:
        p = rec["patient_name"]
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

    - **Groups:** ``patient_name`` (no leakage across folds).
    - **Stratification:** per-patient sagittal binary label from
      :func:`patient_sagittal_strat_labels`.

    Requires ``scikit-learn`` (``StratifiedGroupKFold``).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n = len(binary_records)
    if n == 0:
        return

    indices = np.arange(n, dtype=np.int64)
    groups = np.array([rec["patient_name"] for rec in binary_records], dtype=object)
    strat_map = patient_sagittal_strat_labels(binary_records)
    y = np.array([strat_map[rec["patient_name"]] for rec in binary_records], dtype=np.int64)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    for train_idx, val_idx in sgkf.split(indices, y, groups):
        yield train_idx, val_idx
