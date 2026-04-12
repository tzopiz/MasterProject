"""
Paths and environment detection for **Yandex DataSphere** vs local/Colab.

Typical DataSphere layout (after ``google_colab/init_datasphere_dataset.ipynb``
or manual wget from GitHub Releases):

- ``/home/jupyter/datasets/tmj/manifest_private.json``
- ``/home/jupyter/datasets/tmj/tmj_position_labels.json``
- ``/home/jupyter/datasets/tmj/detector_crops_v2/study_XXXX/*.nii.gz``

Writable artifacts: ``/home/jupyter/filestore/experiments/`` (dataset mount is read-only).

Repo code often lives under ``/home/jupyter/project/.../MLService`` after ``git clone``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


JUPYTER_HOME = Path("/home/jupyter")
DATASPHERE_DATASET_TMJ = JUPYTER_HOME / "datasets" / "tmj"
FILESTORE_EXPERIMENTS = JUPYTER_HOME / "filestore" / "experiments"


def is_datasphere() -> bool:
    return JUPYTER_HOME.exists()


def default_tmj_dataset_dir() -> Path:
    """Root folder with manifest, labels, and detector crop tree."""
    env = os.environ.get("TMJ_DATASET_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if is_datasphere():
        return DATASPHERE_DATASET_TMJ
    return Path("data")


def resolve_detector_crop_dir(dataset_dir: Optional[Path] = None) -> Path:
    """
    Prefer ``detector_crops_v2`` (DataSphere / GitHub release), else ``detector_crops``.
    """
    base = dataset_dir or default_tmj_dataset_dir()
    v2 = base / "detector_crops_v2"
    v1 = base / "detector_crops"
    if v2.is_dir():
        return v2
    if v1.is_dir():
        return v1
    return v2


def infer_mlservice_root(start: Optional[Path] = None) -> Path:
    """
    Best-effort MLService root: walk ``start`` parents, then common DataSphere paths.

    Override: set env ``ML_SERVICE_ROOT`` to the ``MLService`` directory if detection fails.
    """
    env_root = os.environ.get("ML_SERVICE_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "training" / "sagittal_binary_cv.py").is_file():
            return p
    if start is not None:
        p = start.resolve()
        for _ in range(8):
            if (p / "training" / "sagittal_binary_cv.py").is_file():
                return p
            if p.parent == p:
                break
            p = p.parent
    cwd = Path.cwd().resolve()
    candidates = (
        cwd,
        cwd / "MLService",
        cwd.parent,
        cwd.parent / "MLService",
        Path("/home/jupyter/project/MasterProject/MLService"),
        Path("/home/jupyter/project/MasterProject") / "MLService",
    )
    for p in candidates:
        if (p / "training" / "sagittal_binary_cv.py").is_file():
            return p
    return cwd


def sagittal_binary_cv_path_kwargs(
    dataset_dir: Optional[Path] = None,
    crop_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Keyword paths for :class:`training.sagittal_binary_cv.SagittalBinaryCVConfig`.

    ``dataset_root`` is set to the **crop root** so ``build_index``'s synthetic
    ``dicom_dir`` paths point at existing ``study_*`` folders (NIfTI-only OK).
    """
    d = Path(dataset_dir or default_tmj_dataset_dir()).resolve()
    crops = Path(crop_dir or resolve_detector_crop_dir(d)).resolve()
    return {
        "crop_dir": str(crops),
        "manifest_path": str(d / "manifest_private.json"),
        "labels_path": str(d / "tmj_position_labels.json"),
        "dataset_root": str(crops),
    }


def default_cv_output_json(mlservice_root: Path, filename: str = "sagittal_cv_last.json") -> Path:
    """Writable JSON path: filestore on DataSphere, else ``MLService/experiments``."""
    if is_datasphere():
        FILESTORE_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        return FILESTORE_EXPERIMENTS / filename
    out_dir = mlservice_root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename
