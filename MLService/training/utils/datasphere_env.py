"""Strict local/DataSphere legacy discovery; canonical config paths stay explicit.

An explicit environment value or argument must exist and is never replaced.
Unset legacy paths may discover exactly one candidate, otherwise fail safely.
No directory/probe writes happen during path discovery.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

JUPYTER_HOME = Path("/home/jupyter")
DATASPHERE_DATASET_TMJ = JUPYTER_HOME / "datasets" / "tmj"
FILESTORE = JUPYTER_HOME / "filestore"
FILESTORE_EXPERIMENTS = FILESTORE / "experiments"


class PathResolutionError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def is_datasphere() -> bool:
    return JUPYTER_HOME.exists()


def _checked_path(value, *, directory, code):
    try:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise PathResolutionError(code)
        path = Path(value).expanduser().resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise PathResolutionError(code)
        return path
    except PathResolutionError:
        raise
    except (OSError, ValueError, RuntimeError, TypeError):
        raise PathResolutionError(code) from None


def _environment_path(name, *, directory):
    if name not in os.environ:
        return None
    return _checked_path(os.environ[name], directory=directory, code="invalid_explicit_path")


def _unique_existing(candidates, *, directory):
    try:
        matches = {
            path.resolve()
            for path in candidates
            if (path.is_dir() if directory else path.is_file())
        }
    except (OSError, ValueError, RuntimeError, TypeError):
        raise PathResolutionError("unreadable_discovery_path") from None
    if not matches:
        raise PathResolutionError("missing_discovery_path")
    if len(matches) != 1:
        raise PathResolutionError("ambiguous_discovery_path")
    return matches.pop()


def default_tmj_dataset_dir() -> Path:
    explicit = _environment_path("TMJ_DATASET_DIR", directory=True)
    if explicit is not None:
        return explicit
    candidates = [Path("data")]
    if is_datasphere():
        candidates = [DATASPHERE_DATASET_TMJ]
        if any(
            (FILESTORE / name).exists()
            for name in (
                "detector_crops_v2",
                "detector_crops",
                "manifest_private.json",
                "manifest.json",
                "tmj_position_labels.json",
            )
        ):
            candidates.append(FILESTORE)
    return _unique_existing(candidates, directory=True)


def _dataset_base(dataset_dir):
    return (
        default_tmj_dataset_dir()
        if dataset_dir is None
        else _checked_path(dataset_dir, directory=True, code="invalid_explicit_path")
    )


def resolve_detector_crop_dir(dataset_dir: Optional[Path] = None) -> Path:
    base = _dataset_base(dataset_dir) if dataset_dir is not None else None
    explicit = _environment_path("TMJ_CROP_DIR", directory=True)
    if explicit is not None:
        return explicit
    if base is None:
        base = default_tmj_dataset_dir()
    return _unique_existing([base / "detector_crops_v2", base / "detector_crops"], directory=True)


def _manifest_search_paths(dataset_dir: Path) -> list[Path]:
    return [
        dataset_dir / "manifest_private.json",
        dataset_dir / "manifest.json",
        dataset_dir / "dataset_public" / "manifest_private.json",
        dataset_dir / "dataset_cbct_public" / "manifest_private.json",
    ]


def _labels_search_paths(dataset_dir: Path) -> list[Path]:
    return [
        dataset_dir / "tmj_position_labels.json",
        dataset_dir / "dataset_public" / "tmj_position_labels.json",
        dataset_dir / "dataset_cbct_public" / "tmj_position_labels.json",
    ]


def resolve_manifest_path(dataset_dir: Optional[Path] = None) -> Path:
    base = _dataset_base(dataset_dir) if dataset_dir is not None else None
    explicit = _environment_path("TMJ_MANIFEST_PATH", directory=False)
    if explicit is not None:
        return explicit
    return _unique_existing(
        _manifest_search_paths(base or default_tmj_dataset_dir()), directory=False
    )


def resolve_labels_path(dataset_dir: Optional[Path] = None) -> Path:
    base = _dataset_base(dataset_dir) if dataset_dir is not None else None
    explicit = _environment_path("TMJ_LABELS_PATH", directory=False)
    if explicit is not None:
        return explicit
    return _unique_existing(
        _labels_search_paths(base or default_tmj_dataset_dir()), directory=False
    )


def _is_mlservice_root(path):
    return (path / "training" / "sagittal_binary_cv.py").is_file()


def infer_mlservice_root(start: Optional[Path] = None) -> Path:
    explicit = _environment_path("ML_SERVICE_ROOT", directory=True)
    if explicit is not None:
        if not _is_mlservice_root(explicit):
            raise PathResolutionError("invalid_mlservice_root")
        return explicit
    context = _checked_path(start or Path.cwd(), directory=True, code="invalid_discovery_context")
    candidates = []
    for parent in (context, *list(context.parents)[:8]):
        candidates.extend([parent, parent / "MLService"])
    matches = [path for path in candidates if _is_mlservice_root(path)]
    if not matches:
        matches = [
            path
            for path in (
                Path("/home/jupyter/project/MasterProject/MLService"),
                Path("/content/MasterProject/MLService"),
            )
            if _is_mlservice_root(path)
        ]
    return _unique_existing(matches, directory=True)


def validate_explicit_environment() -> None:
    for name, directory in (
        ("TMJ_DATASET_DIR", True),
        ("TMJ_CROP_DIR", True),
        ("TMJ_MANIFEST_PATH", False),
        ("TMJ_LABELS_PATH", False),
        ("ML_SERVICE_ROOT", True),
    ):
        path = _environment_path(name, directory=directory)
        if name == "ML_SERVICE_ROOT" and path is not None and not _is_mlservice_root(path):
            raise PathResolutionError("invalid_mlservice_root")


def sagittal_binary_cv_path_kwargs(
    dataset_dir: Optional[Path] = None, crop_dir: Optional[Path] = None
) -> Dict[str, str]:
    """Compatibility paths for explicitly opted-in legacy name-based CV only."""
    dataset = _dataset_base(dataset_dir)
    crops = (
        resolve_detector_crop_dir(dataset)
        if crop_dir is None
        else _checked_path(crop_dir, directory=True, code="invalid_explicit_path")
    )
    return {
        "crop_dir": str(crops),
        "manifest_path": str(resolve_manifest_path(dataset)),
        "labels_path": str(resolve_labels_path(dataset)),
        "dataset_root": str(crops),
    }


def default_cv_output_json(mlservice_root: Path, filename: str = "sagittal_cv_last.json") -> Path:
    """Legacy protected-directory alias; path only, no mkdir during discovery."""
    root = _checked_path(mlservice_root, directory=True, code="invalid_mlservice_root")
    if not isinstance(filename, str) or not filename.strip() or Path(filename).name != filename:
        raise PathResolutionError("invalid_output_name")
    return (FILESTORE_EXPERIMENTS if is_datasphere() else root / "experiments") / filename
