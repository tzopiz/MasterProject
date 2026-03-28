#!/usr/bin/env python3
"""
Полный цикл для CBCT-когорты (публичная папка Яндекс.Диска):
  1) скачать ZIP с Яндекс.Диска (все пациенты из labels JSON);
  2) распаковать каждый архив в отдельную папку;
  3) удалить мусорные файлы (как в tmj_classification_tool file_cleaner).

По умолчанию включает --download-below-threshold у загрузчика, чтобы скачать и спорные
совпадения (см. предупреждение в download_yandex_cbct_cohort.py).

Пример:
  cd MLService
  python3 tools/prepare_cbct_cohort.py

Только распаковка и очистка уже скачанных zip:
  python3 tools/prepare_cbct_cohort.py --no-download
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from dicom_cohort_cleanup import clean_patient_folder, safe_extract_zip


def run_download(
    tools_dir: Path,
    labels: Path,
    zips_dir: Path,
    public_url: str,
    download_below_threshold: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(tools_dir / "download_yandex_cbct_cohort.py"),
        "--labels",
        str(labels),
        "--output-dir",
        str(zips_dir),
        "--public-url",
        public_url,
    ]
    if download_below_threshold:
        cmd.append("--download-below-threshold")
    if overwrite:
        cmd.append("--overwrite")
    if dry_run:
        cmd.append("--dry-run")
    print("Запуск:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, default=Path("data/tmj_position_labels.json"))
    ap.add_argument("--public-url", default="https://disk.360.yandex.ru/d/iZwyDWpG2A9Fiw")
    ap.add_argument("--zips-dir", type=Path, default=Path("data/cbct_public_zips"))
    ap.add_argument("--extract-dir", type=Path, default=Path("data/cbct_public_extracted"))
    ap.add_argument("--no-download", action="store_true", help="Пропустить фазу скачивания")
    ap.add_argument(
        "--strict-download-match",
        action="store_true",
        help="Не передавать --download-below-threshold (пропуск низкой уверенности)",
    )
    ap.add_argument("--overwrite-download", action="store_true", help="Передать --overwrite загрузчику")
    ap.add_argument("--dry-run-download", action="store_true", help="Загрузчик только dry-run")
    ap.add_argument(
        "--force-extract",
        action="store_true",
        help="Удалить папку назначения и распаковать заново, если она уже есть",
    )
    ap.add_argument("--dry-run-clean", action="store_true", help="Только показать, какие файлы удалили бы")
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON-отчёт (по умолчанию: <extract-dir>/prepare_report.json)",
    )
    args = ap.parse_args()

    ml_root = Path(__file__).resolve().parent.parent
    os.chdir(ml_root)
    tools_dir = Path(__file__).resolve().parent

    labels = args.labels if args.labels.is_absolute() else ml_root / args.labels
    zips_dir = args.zips_dir if args.zips_dir.is_absolute() else ml_root / args.zips_dir
    extract_dir = args.extract_dir if args.extract_dir.is_absolute() else ml_root / args.extract_dir

    if not labels.is_file():
        print(f"Нет файла меток: {labels}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "labels": str(labels),
        "zips_dir": str(zips_dir),
        "extract_dir": str(extract_dir),
        "phases": [],
    }

    if not args.no_download:
        rc = run_download(
            tools_dir,
            labels,
            zips_dir,
            args.public_url,
            download_below_threshold=not args.strict_download_match,
            overwrite=args.overwrite_download,
            dry_run=args.dry_run_download,
        )
        report["phases"].append({"name": "download", "exit_code": rc})
        if rc != 0:
            print(
                f"Загрузка завершилась с кодом {rc} (если есть частично скачанные .zip, ниже — распаковка).",
                file=sys.stderr,
                flush=True,
            )

    zips = sorted(zips_dir.glob("*.zip"))
    if not zips:
        print(f"В {zips_dir} нет .zip файлов.", file=sys.stderr)
        out_path = args.report or (extract_dir / "prepare_report.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1

    extract_report: list[dict[str, Any]] = []
    for zp in zips:
        stem = zp.stem
        out_sub = extract_dir / stem
        entry: dict[str, Any] = {"zip": str(zp.name), "extract_to": str(out_sub)}
        try:
            if out_sub.exists() and any(out_sub.iterdir()):
                if args.force_extract:
                    shutil.rmtree(out_sub)
                else:
                    entry["extract_status"] = "skipped_exists"
                    extract_report.append(entry)
                    print(f"EXTRACT SKIP (exists) {stem}", flush=True)
                    clean_stats = clean_patient_folder(out_sub, dry_run=args.dry_run_clean)
                    entry["clean"] = clean_stats
                    continue
            safe_extract_zip(zp, out_sub)
            entry["extract_status"] = "ok"
            print(f"EXTRACT OK {stem}", flush=True)
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            entry["extract_status"] = "failed"
            entry["error"] = str(e)
            print(f"EXTRACT FAIL {stem}: {e}", file=sys.stderr, flush=True)
        clean_stats = (
            clean_patient_folder(out_sub, dry_run=args.dry_run_clean) if out_sub.is_dir() else {}
        )
        entry["clean"] = clean_stats
        extract_report.append(entry)

    report["phases"].append({"name": "extract_and_clean", "items": extract_report})
    report["summary"] = {
        "zip_count": len(zips),
        "extracted_ok": sum(1 for x in extract_report if x.get("extract_status") == "ok"),
        "files_removed": sum(x.get("clean", {}).get("files_removed", 0) for x in extract_report),
    }

    out_path = args.report or (extract_dir / "prepare_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {out_path}", flush=True)
    print(
        f"Итого: zip={len(zips)}, распаковано (в этом прогоне ok)={report['summary']['extracted_ok']}, "
        f"удалено мусорных файлов={report['summary']['files_removed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
