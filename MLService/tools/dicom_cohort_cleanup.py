"""
Очистка папок CBCT/DICOM-когорты от мусора перед сборкой датасета.

Удаляем типичные вьюеры, логи, превью, инсталляторы и служебные каталоги архивов,
оставляя .dcm и прочие медицинские файлы (в т.ч. без расширения .dcm, если не попали под правила).

Логика согласована с tmj_classification_tool/services/file_cleaner.py, но расширена.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

try:
    import pydicom
except ImportError:
    pydicom = None

# Точные имена (нижний регистр)
JUNK_FILENAMES_EXACT: set[str] = {
    "runthis.exe",
    "vision.exe",
    "visionrus.dll",
    "autorun.inf",
    "license.txt",
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    "icon\r",
}

# Расширения — удалять, кроме особых случаев (.bin может быть DICOM)
JUNK_EXTENSIONS: set[str] = {
    ".exe",
    ".dll",
    ".inf",
    ".db",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".txt",
    ".pdf",
    ".log",
    ".htm",
    ".html",
    ".url",
    ".lnk",
    ".bat",
    ".cmd",
    ".msi",
    ".msm",
    ".chm",
    ".tmp",
    ".temp",
    ".bak",
    ".ini",
    ".cfg",
    ".cab",
}

# Если имя файла содержит подстроку — мусор (только basename)
JUNK_NAME_SUBSTRINGS: tuple[str, ...] = (
    "readme",
    "license",
    "licence",
    "changelog",
    "install",
    "uninstall",
    "setup",
    "vision",
    "viewer",
    "dcmview",
    "dicomview",
    "weasis",
    "radiant",
    "autorun",
    "release notes",
    "release_notes",
    "версия",
    "инструкция",
)


def _basename_lower(path: Path) -> str:
    return path.name.lower()


def is_dataset_junk_file(filepath: Path) -> bool:
    """True, если файл можно удалить при подготовке датасета."""
    name = _basename_lower(filepath)
    if name in JUNK_FILENAMES_EXACT:
        return True
    for part in JUNK_NAME_SUBSTRINGS:
        if part in name:
            return True
    suf = filepath.suffix.lower()
    if filepath.name.count(".") > 1:
        # .dcm.bak и т.п.
        full = name
        for ext in sorted(JUNK_EXTENSIONS, key=len, reverse=True):
            if full.endswith(ext):
                suf = ext
                break
    if suf not in JUNK_EXTENSIONS:
        return False
    if suf == ".bin" and pydicom is not None:
        try:
            pydicom.dcmread(str(filepath), stop_before_pixels=True)
            return False
        except Exception:
            return True
    if suf == ".bin":
        return False
    return True


def remove_service_directories(root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Удаляет каталоги __MACOSX (артефакт zip на macOS)."""
    stats = {"removed_dirs": [], "errors": 0}
    root = root.resolve()
    if not root.is_dir():
        return stats
    to_remove: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == "__MACOSX" or d.startswith("__MACOSX"):
                to_remove.append(Path(dirpath) / d)
    for p in sorted(to_remove, key=lambda x: len(x.parts), reverse=True):
        try:
            if not dry_run:
                shutil.rmtree(p, ignore_errors=True)
            stats["removed_dirs"].append(str(p))
        except OSError:
            stats["errors"] += 1
    return stats


def remove_empty_directories(root: Path, dry_run: bool = False) -> int:
    """Снизу вверх удаляет пустые каталоги (кроме самого root)."""
    removed = 0
    root = root.resolve()
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p.resolve() == root:
            continue
        try:
            if p.is_dir() and not any(p.iterdir()):
                if not dry_run:
                    p.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def clean_patient_folder(
    root_dir: Path,
    *,
    dry_run: bool = False,
    remove_macosx: bool = True,
    prune_empty_dirs: bool = True,
) -> dict[str, Any]:
    """
    Удаляет мусорные файлы под root_dir, опционально __MACOSX и пустые папки.
    """
    stats: dict[str, Any] = {
        "root": str(root_dir.resolve()),
        "files_scanned": 0,
        "files_removed": 0,
        "removed_paths": [],
        "errors": 0,
        "empty_dirs_removed": 0,
        "service_dirs": [],
    }
    if not root_dir.is_dir():
        return stats

    for dirpath, _dirs, files in os.walk(root_dir):
        for filename in files:
            fp = Path(dirpath) / filename
            stats["files_scanned"] += 1
            try:
                if is_dataset_junk_file(fp):
                    stats["removed_paths"].append(str(fp.resolve()))
                    if not dry_run:
                        fp.unlink(missing_ok=True)
                    stats["files_removed"] += 1
            except OSError:
                stats["errors"] += 1

    if remove_macosx:
        sd = remove_service_directories(root_dir, dry_run=dry_run)
        stats["service_dirs"] = sd["removed_dirs"]
        stats["errors"] += sd["errors"]

    if prune_empty_dirs:
        stats["empty_dirs_removed"] = remove_empty_directories(root_dir, dry_run=dry_run)

    return stats


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Распаковка с проверкой zip-slip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (dest_dir / info.filename).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"Небезопасный путь в архиве {zip_path.name!r}: {info.filename!r}")
        zf.extractall(dest_dir)
