"""
Paths and environment detection for **Yandex DataSphere** vs local/Colab.

Typical DataSphere layout (after ``google_colab/init_datasphere_dataset.ipynb``
or manual wget from GitHub Releases):

- ``/home/jupyter/datasets/tmj/tmj_position_labels.json``
- Манифест: ``manifest_private.json`` **или** ``manifest.json`` в каталоге датасета;
  если в датасете только кропы — часто лежит в ``/home/jupyter/filestore/`` (как после wget в ``train_binary_position_classifier``).

Переопределение путей: ``TMJ_DATASET_DIR``, ``TMJ_CROP_DIR`` (прямо на каталог с ``study_*``),
``TMJ_MANIFEST_PATH``, ``TMJ_LABELS_PATH``.

Кропы: сначала под датасетом ``.../datasets/tmj/detector_crops_v2``, иначе (как в
``train_binary_position_classifier`` без смонтированного датасета) —
``/home/jupyter/filestore/detector_crops_v2``.

Writable artifacts: ``/home/jupyter/filestore/experiments/`` (dataset mount is read-only).

Repo code often lives under ``/home/jupyter/project/.../MLService`` after ``git clone``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

JUPYTER_HOME = Path("/home/jupyter")
DATASPHERE_DATASET_TMJ = JUPYTER_HOME / "datasets" / "tmj"
FILESTORE = JUPYTER_HOME / "filestore"
FILESTORE_EXPERIMENTS = FILESTORE / "experiments"


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
    Resolve directory that contains ``study_XXXX/`` subfolders with NIfTI crops.

    Order:

    1. ``TMJ_CROP_DIR`` — если задан и существует (указывает на ``detector_crops_v2`` или аналог).
    2. ``{dataset_dir}/detector_crops_v2`` then ``detector_crops`` — если ``dataset_dir`` существует.
    3. На DataSphere: ``/home/jupyter/filestore/detector_crops_v2`` (типичный layout binary-ноутбука,
       когда датасет ``datasets/tmj`` не подключён, а кропы распакованы в filestore).

    If nothing exists, returns the first candidate path (for downstream error messages).
    """
    env = os.environ.get("TMJ_CROP_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    base = Path(dataset_dir) if dataset_dir is not None else default_tmj_dataset_dir()
    base = base.resolve()
    candidates: list[Path] = []
    if base.is_dir():
        candidates.extend([base / "detector_crops_v2", base / "detector_crops"])
    if is_datasphere():
        candidates.extend([FILESTORE / "detector_crops_v2", FILESTORE / "detector_crops"])
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    if is_datasphere():
        return (FILESTORE / "detector_crops_v2").resolve()
    return (base / "detector_crops_v2").resolve()


def _manifest_search_paths(dataset_dir: Path) -> list[Path]:
    paths = [
        dataset_dir / "manifest_private.json",
        dataset_dir / "manifest.json",
        dataset_dir / "dataset_public" / "manifest_private.json",
        dataset_dir / "dataset_cbct_public" / "manifest_private.json",
    ]
    if is_datasphere():
        paths.extend(
            [
                FILESTORE / "manifest_private.json",
                FILESTORE / "manifest.json",
            ]
        )
    return paths


def _labels_search_paths(dataset_dir: Path) -> list[Path]:
    paths = [
        dataset_dir / "tmj_position_labels.json",
        dataset_dir / "dataset_public" / "tmj_position_labels.json",
        dataset_dir / "dataset_cbct_public" / "tmj_position_labels.json",
    ]
    if is_datasphere():
        paths.append(FILESTORE / "tmj_position_labels.json")
    return paths


def resolve_manifest_path(dataset_dir: Path) -> Path:
    """
    Resolve manifest JSON (``studies`` list) for ``build_index``.

    Order: ``TMJ_MANIFEST_PATH`` env, then dataset dir (private / public name),
    then on DataSphere ``/home/jupyter/filestore`` (same filenames).
    """
    env = os.environ.get("TMJ_MANIFEST_PATH", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    for p in _manifest_search_paths(dataset_dir):
        if p.is_file():
            return p
    tried = [str(p) for p in _manifest_search_paths(dataset_dir)]
    raise FileNotFoundError(
        "Manifest not found. Tried:\n  "
        + "\n  ".join(tried)
        + "\nSet TMJ_MANIFEST_PATH, copy manifest into the dataset dir, or run "
        "google_colab/init_datasphere_dataset.ipynb (wget manifest_private.json)."
    )


def resolve_labels_path(dataset_dir: Path) -> Path:
    """Resolve ``tmj_position_labels.json`` (dataset dir, then filestore on DataSphere)."""
    env = os.environ.get("TMJ_LABELS_PATH", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    for p in _labels_search_paths(dataset_dir):
        if p.is_file():
            return p
    tried = [str(p) for p in _labels_search_paths(dataset_dir)]
    raise FileNotFoundError(
        "tmj_position_labels.json not found. Tried:\n  "
        + "\n  ".join(tried)
        + "\nSet TMJ_LABELS_PATH or add labels to the dataset / filestore directory."
    )


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
        "manifest_path": str(resolve_manifest_path(d)),
        "labels_path": str(resolve_labels_path(d)),
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
