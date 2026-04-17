#!/usr/bin/env python3
"""
Скачивание ZIP-исследований с публичной папки Яндекс.Диска по списку пациентов из JSON
(например data/tmj_position_labels.json).

Сопоставление имён: token-sort ratio (устойчиво к порядку «Фамилия Имя Отчество» и мелким опечаткам).
Порог по умолчанию 0.82 отсекает явные несовпадения (например, Артём vs Марк у одной фамилии).
Флаг --download-below-threshold всё равно качает лучший матч (в манифесте будет downloaded_low_confidence).

Пример:
  cd MLService
  python3 tools/download_yandex_cbct_cohort.py \\
    --labels data/tmj_position_labels.json \\
    --public-url 'https://disk.360.yandex.ru/d/iZwyDWpG2A9Fiw' \\
    --output-dir data/cbct_public_zips \\
    --dry-run

Без --dry-run — реальная загрузка (архивы большие, десятки ГБ суммарно).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("ё", "е").split())


def token_sort_ratio(a: str, b: str) -> float:
    ta, tb = sorted(_norm(a).split()), sorted(_norm(b).split())
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def list_public_zips(public_url: str, limit: int = 1000) -> list[dict[str, Any]]:
    api = "https://cloud-api.yandex.net/v1/disk/public/resources?" + urllib.parse.urlencode(
        {"public_key": public_url, "limit": limit}
    )
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    items = data.get("_embedded", {}).get("items", [])
    return [
        i for i in items if i.get("type") == "file" and i.get("name", "").lower().endswith(".zip")
    ]


def get_download_href(public_url: str, remote_path: str) -> str:
    api = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
        + urllib.parse.urlencode({"public_key": public_url, "path": remote_path})
    )
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)
    return body["href"]


def download_href_to_file(href: str, dest: Path, chunk: int = 256 * 1024) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(href, method="GET")
    total = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        with open(dest, "wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                total += len(block)
    return total


def best_zip_match(name_raw: str, zips: list[dict[str, Any]]) -> tuple[str | None, float]:
    best_name, best_score = None, 0.0
    for item in zips:
        zname = item["name"]
        stem = zname[:-4] if zname.lower().endswith(".zip") else zname
        sc = token_sort_ratio(name_raw, stem)
        if sc > best_score:
            best_score, best_name = sc, zname
    return best_name, best_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--labels", type=Path, required=True, help="JSON от parse_tmj_position_labels_docx.py"
    )
    ap.add_argument(
        "--public-url",
        default="https://disk.360.yandex.ru/d/iZwyDWpG2A9Fiw",
        help="Публичная ссылка на папку Яндекс.Диска",
    )
    ap.add_argument("--output-dir", type=Path, required=True, help="Локальная папка для .zip")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.82,
        help="Мин. token_sort_ratio для «уверенной» загрузки",
    )
    ap.add_argument(
        "--download-below-threshold",
        action="store_true",
        help="Скачивать лучший матч даже при score < min-score (осторожно: возможен неверный пациент)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Только план и manifest, без скачивания")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Перекачать, даже если файл уже есть (по умолчанию пропуск существующих)",
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Куда записать отчёт (по умолчанию: <output-dir>/download_manifest.json)",
    )
    args = ap.parse_args()

    if not args.labels.is_file():
        print(f"Нет файла: {args.labels}", file=sys.stderr)
        return 1

    labels_data = json.loads(args.labels.read_text(encoding="utf-8"))
    patients = labels_data.get("patients", [])
    if not patients:
        print("В JSON нет patients[]", file=sys.stderr)
        return 1

    print("Запрос списка файлов с Яндекс.Диска…", flush=True)
    try:
        zips = list_public_zips(args.public_url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as e:
        print(f"Ошибка API Диска: {e}", file=sys.stderr)
        return 1

    manifest: dict[str, Any] = {
        "public_url": args.public_url,
        "labels_file": str(args.labels.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "min_score": args.min_score,
        "download_below_threshold": args.download_below_threshold,
        "dry_run": args.dry_run,
        "disk_zip_count": len(zips),
        "patients": [],
        "skipped": [],
        "downloaded": [],
    }

    for p in patients:
        num = p.get("patient_number")
        name_raw = p.get("name_raw", "")
        match_name, score = best_zip_match(name_raw, zips)
        entry: dict[str, Any] = {
            "patient_number": num,
            "name_raw": name_raw,
            "matched_zip": match_name,
            "score": round(score, 4),
        }
        if match_name:
            entry["remote_size"] = next(
                (i.get("size") for i in zips if i["name"] == match_name), None
            )

        low_conf = match_name is not None and score < args.min_score
        if match_name is None:
            entry["status"] = "skipped_no_match"
            manifest["skipped"].append(entry)
            manifest["patients"].append(entry)
            print(f"SKIP no zip  #{num} {name_raw!r}", flush=True)
            continue
        if low_conf and not args.download_below_threshold:
            entry["status"] = "skipped_low_confidence"
            manifest["skipped"].append(entry)
            manifest["patients"].append(entry)
            print(f"SKIP {score:.3f}  #{num} {name_raw!r} -> {match_name!r}", flush=True)
            continue
        if low_conf:
            entry["parse_warning"] = "score_below_min_threshold"
            print(
                f"LOW {score:.3f}  #{num} {name_raw!r} -> {match_name!r} (download-below-threshold)",
                flush=True,
            )

        dest = args.output_dir / match_name
        entry["local_path"] = str(dest.resolve())
        if dest.exists() and not args.overwrite:
            entry["status"] = "skipped_exists"
            if low_conf:
                entry["parse_warning"] = "score_below_min_threshold"
            manifest["patients"].append(entry)
            print(f"EXISTS       #{num} {match_name}", flush=True)
            continue

        if args.dry_run:
            entry["status"] = "would_download_low_confidence" if low_conf else "would_download"
            manifest["patients"].append(entry)
            print(f"PLAN {score:.3f}  #{num} -> {match_name}", flush=True)
            continue

        remote_path = "/" + match_name
        try:
            href = get_download_href(args.public_url, remote_path)
            t0 = time.perf_counter()
            nbytes = download_href_to_file(href, dest)
            sec = time.perf_counter() - t0
            entry["status"] = "downloaded_low_confidence" if low_conf else "downloaded"
            entry["bytes"] = nbytes
            entry["seconds"] = round(sec, 2)
            manifest["downloaded"].append(entry)
            manifest["patients"].append(entry)
            mb_s = (nbytes / (1024 * 1024)) / sec if sec > 0 else 0
            print(
                f"OK {score:.3f}  #{num} {match_name}  {nbytes / (1024**2):.1f} MB  {sec:.0f}s  ({mb_s:.1f} MB/s)",
                flush=True,
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            TimeoutError,
            KeyError,
        ) as e:
            entry["status"] = "download_failed"
            entry["error"] = str(e)
            manifest["patients"].append(entry)
            print(f"FAIL #{num} {match_name}: {e}", file=sys.stderr, flush=True)

    mpath = args.manifest_out or (args.output_dir / "download_manifest.json")
    if not args.dry_run or mpath:
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nManifest: {mpath}", flush=True)

    failed = sum(1 for x in manifest["patients"] if x.get("status") == "download_failed")
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
