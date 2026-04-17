#!/usr/bin/env python3
"""
Сборка датасета из ZIP-исследований: распаковка + очистка мусора.

Зачем: для обучения нужны в основном DICOM-срезы; вьюеры (.exe), логи, PDF/HTML
инструкции, превью-картинки и служебные папки (__MACOSX) только занимают место и
путают сканирование каталогов.

Не удаляет:
  - .dcm и файлы без «мусорного» расширения;
  - .bin, если pydicom успешно читает их как DICOM.

Примеры:
  cd MLService
  python3 tools/build_cbct_zip_dataset.py --dry-run
  python3 tools/build_cbct_zip_dataset.py --force-extract
  python3 tools/build_cbct_zip_dataset.py --zips-dir data/my_zips --output-dir data/my_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from dicom_cohort_cleanup import clean_patient_folder, safe_extract_zip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--zips-dir",
        type=Path,
        default=Path("data/cbct_public_zips"),
        help="Каталог с .zip по пациентам",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cbct_public_extracted"),
        help="Куда распаковывать (подпапка на архив: <stem>/...)",
    )
    ap.add_argument(
        "--force-extract",
        action="store_true",
        help="Удалить существующую папку пациента и распаковать заново",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Не удалять файлы и не писать при распаковке — только план",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON-отчёт (по умолчанию: <output-dir>/dataset_build_report.json)",
    )
    ap.add_argument(
        "--no-prune-empty-dirs",
        action="store_true",
        help="Не удалять пустые каталоги после чистки",
    )
    args = ap.parse_args()

    ml_root = Path(__file__).resolve().parent.parent
    os.chdir(ml_root)

    zips_dir = args.zips_dir if args.zips_dir.is_absolute() else ml_root / args.zips_dir
    out_root = args.output_dir if args.output_dir.is_absolute() else ml_root / args.output_dir

    zips = sorted(zips_dir.glob("*.zip"))
    if not zips:
        print(f"Нет .zip в {zips_dir}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "zips_dir": str(zips_dir),
        "output_dir": str(out_root),
        "dry_run": args.dry_run,
        "patients": [],
    }

    for zp in zips:
        stem = zp.stem
        dest = out_root / stem
        entry: dict[str, Any] = {"zip": zp.name, "folder": str(dest)}

        try:
            if dest.exists() and any(dest.iterdir()):
                if args.force_extract:
                    if not args.dry_run:
                        shutil.rmtree(dest)
                else:
                    entry["extract"] = "skipped_exists"
                    print(f"SKIP extract (exists): {stem}", flush=True)
                    clean_stats = clean_patient_folder(
                        dest,
                        dry_run=args.dry_run,
                        prune_empty_dirs=not args.no_prune_empty_dirs,
                    )
                    entry["cleanup"] = clean_stats
                    report["patients"].append(entry)
                    continue

            if args.dry_run:
                entry["extract"] = "would_extract"
                print(f"PLAN extract: {stem}", flush=True)
            else:
                safe_extract_zip(zp, dest)
                entry["extract"] = "ok"
                print(f"OK extract: {stem}", flush=True)
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            entry["extract"] = "failed"
            entry["error"] = str(e)
            print(f"FAIL extract {stem}: {e}", file=sys.stderr, flush=True)

        if dest.is_dir():
            clean_stats = clean_patient_folder(
                dest,
                dry_run=args.dry_run,
                prune_empty_dirs=not args.no_prune_empty_dirs,
            )
            entry["cleanup"] = clean_stats
            cr = clean_stats.get("files_removed", 0)
            print(f"  cleanup: removed {cr} junk files", flush=True)

        report["patients"].append(entry)

    report["summary"] = {
        "zip_count": len(zips),
        "extract_ok": sum(1 for p in report["patients"] if p.get("extract") == "ok"),
        "files_removed": sum(
            p.get("cleanup", {}).get("files_removed", 0) for p in report["patients"]
        ),
        "empty_dirs_removed": sum(
            p.get("cleanup", {}).get("empty_dirs_removed", 0) for p in report["patients"]
        ),
    }

    rep_path = args.report or (out_root / "dataset_build_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run or rep_path:
        rep_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"\nОтчёт: {rep_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
