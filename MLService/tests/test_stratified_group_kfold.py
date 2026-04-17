"""StratifiedGroupKFold splits for binary sagittal records."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.tmj_position_label_table import (
    iter_stratified_group_kfold_indices,
    patient_sagittal_strat_labels,
)


def _synthetic_binary_records():
    """20 patients: 10 all-central, 10 all non-central; 2 sides each."""
    recs = []
    for pi in range(20):
        sag = 0 if pi < 10 else 1
        for side in ("left", "right"):
            recs.append(
                {
                    "patient_name": f"Patient_{pi:02d}",
                    "study_id": f"study_{pi:02d}",
                    "side": side,
                    "sag": sag,
                    "fr": 0,
                    "crop_path": f"/tmp/crop_{pi}_{side}.nii.gz",
                }
            )
    return recs


def test_patient_strat_max_takes_asymmetric():
    recs = [
        {
            "patient_name": "A",
            "study_id": "s1",
            "side": "left",
            "sag": 0,
            "fr": 0,
            "crop_path": "/x",
        },
        {
            "patient_name": "A",
            "study_id": "s1",
            "side": "right",
            "sag": 1,
            "fr": 0,
            "crop_path": "/y",
        },
    ]
    strat = patient_sagittal_strat_labels(recs)
    assert strat["A"] == 1


def test_kfold_no_patient_leakage():
    recs = _synthetic_binary_records()
    folds = list(
        iter_stratified_group_kfold_indices(recs, n_splits=5, shuffle=True, random_state=0)
    )
    assert len(folds) == 5
    for tr_idx, va_idx in folds:
        tr_pat = {recs[i]["patient_name"] for i in tr_idx}
        va_pat = {recs[i]["patient_name"] for i in va_idx}
        assert tr_pat.isdisjoint(va_pat)
        assert len(tr_pat | va_pat) == 20


def test_kfold_covers_all_indices():
    recs = _synthetic_binary_records()
    for tr_idx, va_idx in iter_stratified_group_kfold_indices(recs, n_splits=5, random_state=1):
        covered = set(tr_idx.tolist()) | set(va_idx.tolist())
        assert covered == set(range(len(recs)))
