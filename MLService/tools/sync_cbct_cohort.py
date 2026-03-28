#!/usr/bin/env python3
"""
Единая точка входа для CBCT-когорты (метки в JSON, zip с Яндекс.Диска): распаковка → датасет.

По умолчанию (без --download):
  1) Распаковать **новые** .zip из zips-dir → extracted (build_cbct_zip_dataset)
  2) Собрать плоский датасет study_0001… + manifest (organize_dataset), с **анонимизацией DICOM**

С флагом --download перед этим вызывается download_yandex_cbct_cohort (новые пациенты в JSON).

Интерпретатор: если есть `MLService/venv/`, подставляется его `python` (нужен pydicom для --anonymize).

Примеры:
  cd MLService
  python3 tools/sync_cbct_cohort.py
  python3 tools/sync_cbct_cohort.py --download
  python3 tools/sync_cbct_cohort.py --force-extract --dataset-out data/dataset_cbct_public
  python3 tools/sync_cbct_cohort.py --no-anonymize   # копия DICOM без снятия PHI (только локально)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> int:
    print("→", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def _python_executable(ml_root: Path) -> str:
    """Prefer project venv so pydicom etc. are available."""
    for rel in ("venv/bin/python3", "venv/bin/python"):
        p = ml_root / rel
        if p.is_file():
            return str(p)
    return sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--download",
        action="store_true",
        help="Сначала скачать zip с Яндекс.Диска по tmj_position_labels.json",
    )
    ap.add_argument(
        "--strict-download-match",
        action="store_true",
        help="Без --download-below-threshold у загрузчика",
    )
    ap.add_argument("--labels", type=Path, default=Path("data/tmj_position_labels.json"))
    ap.add_argument("--public-url", default="https://disk.360.yandex.ru/d/iZwyDWpG2A9Fiw")
    ap.add_argument("--zips-dir", type=Path, default=Path("data/cbct_public_zips"))
    ap.add_argument("--extract-dir", type=Path, default=Path("data/cbct_public_extracted"))
    ap.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("data/dataset_cbct_public"),
        help="Выход organize_dataset (study_0001 + manifest.json)",
    )
    ap.add_argument(
        "--force-extract",
        action="store_true",
        help="Передать в build_cbct_zip_dataset (всё перераспаковать)",
    )
    ap.add_argument(
        "--organize-clean-input",
        action="store_true",
        help="Передать organize_dataset --clean (ещё раз чистить дерево extracted)",
    )
    ap.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Не анонимизировать DICOM (по умолчанию organize_dataset --anonymize)",
    )
    args = ap.parse_args()

    ml_root = Path(__file__).resolve().parent.parent
    os.chdir(ml_root)
    py = _python_executable(ml_root)
    tools = ml_root / "tools"

    if args.download:
        cmd = [
            py,
            str(tools / "download_yandex_cbct_cohort.py"),
            "--labels",
            str(args.labels),
            "--output-dir",
            str(args.zips_dir),
            "--public-url",
            args.public_url,
        ]
        if not args.strict_download_match:
            cmd.append("--download-below-threshold")
        rc = _run(cmd, ml_root)
        if rc != 0:
            print(f"Загрузка завершилась с кодом {rc}, продолжаем распаковку…", flush=True)

    build_cmd = [
        py,
        "-u",
        str(tools / "build_cbct_zip_dataset.py"),
        "--zips-dir",
        str(args.zips_dir),
        "--output-dir",
        str(args.extract_dir),
        "--report",
        str(args.extract_dir / "dataset_build_report.json"),
    ]
    if args.force_extract:
        build_cmd.append("--force-extract")
    rc = _run(build_cmd, ml_root)
    if rc != 0:
        return rc

    org_cmd = [
        py,
        str(tools / "organize_dataset.py"),
        "--input",
        str(args.extract_dir),
        "--output",
        str(args.dataset_out),
    ]
    if not args.no_anonymize:
        org_cmd.append("--anonymize")
    if args.organize_clean_input:
        org_cmd.append("--clean")
    rc = _run(org_cmd, ml_root)
    if rc != 0:
        return rc

    print("\nГотово. Датасет:", (ml_root / args.dataset_out).resolve(), flush=True)
    print("Manifest:", (ml_root / args.dataset_out / "manifest.json").resolve(), flush=True)
    if not args.no_anonymize:
        print("PHI linkage (не публиковать):", (ml_root / args.dataset_out / "manifest_private.json").resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
