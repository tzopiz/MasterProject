#!/usr/bin/env python3
"""
Парсинг DOCX с клиническими случаями ВНЧС (формат «Пациент N. …» + 4 строки классов).

Ожидаемая структура документа (типичный клинический отчёт по случаям ВНЧС):
- В начале — блок с кодами 1–6 (положение головки), текст может идти подряд в одном абзаце.
- Далее блоки «Пациент <номер>. <ФИО>/ <пол>/ <дата рождения>/ <контекст КТ>»
- Четыре строки (по одной на абзац или в одном абзаце через перенос):
  Правый/Левый ВНЧС, сагиттальная/фронтальная плоскость: <цифра>

Выход: JSON с однозначной схемой (см. build_output).

Пример:
  python3 parse_tmj_position_labels_docx.py \\
    --input ~/Downloads/файл.docx \\
    --output labels.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ")
    return s.strip()


def _paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for t in p.iter(_w("t")):
        if t.text:
            parts.append(t.text)
        if t.tail:
            parts.append(t.tail)
    return "".join(parts)


def _iter_paragraphs_in_order(body: ET.Element) -> list[str]:
    """Обход w:body в порядке документа: таблицы по ячейкам, затем абзацы."""

    out: list[str] = []

    def walk(el: ET.Element) -> None:
        tag = el.tag
        if tag == _w("p"):
            txt = _paragraph_text(el)
            if txt.strip():
                out.append(txt)
        elif tag == _w("tbl"):
            for tr in el.findall(_w("tr")):
                for tc in tr.findall(_w("tc")):
                    for child in tc:
                        walk(child)
        else:
            for child in el:
                walk(child)

    for child in body:
        walk(child)
    return out


def _load_paragraphs_from_docx(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        data = zf.read("word/document.xml")
    root = ET.fromstring(data)
    body = root.find(_w("body"))
    if body is None:
        raise ValueError("В document.xml не найден w:body")
    return _iter_paragraphs_in_order(body)


def _flatten_to_lines(paragraphs: list[str]) -> list[str]:
    lines: list[str] = []
    for p in paragraphs:
        for raw in p.splitlines():
            line = _normalize_line(raw)
            if line:
                lines.append(line)
    return lines


_PATIENT_HEADER = re.compile(
    r"^Пациент\s+(?P<num>\d+)\.\s*(?P<rest>.+)$",
    re.IGNORECASE,
)

_LABEL_LINE = re.compile(
    r"^(?P<side>Правый|Левый)\s+ВНЧС,\s*(?P<plane>сагиттальная|фронтальная)\s+плоскость:\s*(?P<code>\d+)\s*$",
    re.IGNORECASE,
)

_EXPECTED_ORDER = (
    ("Правый", "сагиттальная"),
    ("Левый", "сагиттальная"),
    ("Правый", "фронтальная"),
    ("Левый", "фронтальная"),
)


def _parse_header_rest(rest: str) -> dict[str, Any]:
    """Делит хвост строки пациента по '/' на части (ФИО, пол, дата рождения, визит)."""
    parts = [_normalize_line(x) for x in rest.split("/")]
    parts = [x for x in parts if x]
    name_raw = parts[0] if parts else rest
    sex = None
    birth_date_raw = None
    visit_raw = None
    if len(parts) >= 2:
        s = parts[1].lower().strip()
        if s in ("ж", "м", "м.", "ж."):
            sex = s.rstrip(".")
    if len(parts) >= 3:
        birth_date_raw = parts[2]
    if len(parts) >= 4:
        visit_raw = "/".join(parts[3:])
    return {
        "name_raw": name_raw,
        "sex": sex,
        "birth_date_raw": birth_date_raw,
        "visit_raw": visit_raw,
        "header_parts": parts,
    }


def _extract_class_legend(intro_text: str) -> list[dict[str, Any]]:
    """
    Выделяет пары (code, text) для кодов 1–6 из вступительного текста до первого «Пациент».
    """
    intro_text = _normalize_line(intro_text.replace("\n", " "))
    matches = list(re.finditer(r"(?<![0-9])([1-6])\.\s*", intro_text))
    if not matches:
        return []
    items: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        code = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(intro_text)
        text = intro_text[start:end].strip().strip(",").strip()
        items.append({"code": code, "description": text})
    return items


def _split_intro_and_body_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    first_patient_idx = None
    for i, line in enumerate(lines):
        if _PATIENT_HEADER.match(line):
            first_patient_idx = i
            break
    if first_patient_idx is None:
        return lines, []
    return lines[:first_patient_idx], lines[first_patient_idx:]


def _apply_clinical_name_aliases(record: dict[str, Any]) -> None:
    """
    Известные расхождения DOCX ↔ имена архивов на Яндекс.Диске.
    Якунин: в документе иногда «Артём/Артем», на Диске — «Марк» (подтверждено как опечатка).
    """
    name = record.get("name_raw", "")
    wrong = ("Якунин Артем Викторович", "Якунин Артём Викторович")
    if name not in wrong:
        return
    record["name_raw"] = "Якунин Марк Викторович"
    parts = record.get("header_parts") or []
    if parts:
        parts[0] = record["name_raw"]
    hl = record.get("header_line", "")
    for w in wrong:
        if w in hl:
            record["header_line"] = hl.replace(w, record["name_raw"], 1)
            break
    msg = "В DOCX указано «Артём/Артем»; заменено на «Марк» для совпадения с архивом на Диске (опечатка в источнике)."
    notes = record.setdefault("parse_notes", [])
    if msg not in notes:
        notes.append(msg)


def parse_docx_lines(lines: list[str]) -> dict[str, Any]:
    intro_lines, body_lines = _split_intro_and_body_lines(lines)
    intro_joined = " ".join(intro_lines)
    class_legend = _extract_class_legend(intro_joined)

    patients: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    i = 0
    while i < len(body_lines):
        m = _PATIENT_HEADER.match(body_lines[i])
        if not m:
            errors.append(
                {
                    "type": "unexpected_line_before_patient",
                    "line_index": i,
                    "line": body_lines[i],
                }
            )
            i += 1
            continue
        patient_number = int(m.group("num"))
        rest = m.group("rest")
        header_info = _parse_header_rest(rest)
        record: dict[str, Any] = {
            "patient_number": patient_number,
            "header_line": body_lines[i],
            **header_info,
            "labels": {
                "sagittal": {"right": None, "left": None},
                "frontal": {"right": None, "left": None},
            },
            "label_lines_raw": [],
            "parse_notes": [],
        }
        i += 1
        labels_parsed: list[tuple[str, str, int]] = []
        for expected_side, expected_plane in _EXPECTED_ORDER:
            if i >= len(body_lines):
                errors.append(
                    {
                        "type": "missing_label_lines",
                        "patient_number": patient_number,
                        "expected": f"{expected_side} {expected_plane}",
                    }
                )
                break
            raw = body_lines[i]
            record["label_lines_raw"].append(raw)
            lm = _LABEL_LINE.match(raw)
            if not lm:
                errors.append(
                    {
                        "type": "label_line_unmatched",
                        "patient_number": patient_number,
                        "line_index": i,
                        "line": raw,
                    }
                )
                record["parse_notes"].append(f"Строка класса не распознана: {raw!r}")
                i += 1
                break
            side = lm.group("side").capitalize()
            plane_ru = lm.group("plane").lower()
            if plane_ru == "сагиттальная":
                plane = "sagittal"
            else:
                plane = "frontal"
            lr = "right" if side == "Правый" else "left"
            code = int(lm.group("code"))
            if side != expected_side or lm.group("plane").lower() != expected_plane.lower():
                record["parse_notes"].append(
                    f"Ожидалось «{expected_side}, {expected_plane}», получено «{side}, {lm.group('plane')}»"
                )
            labels_parsed.append((plane, lr, code))
            record["labels"][plane][lr] = code
            i += 1
        else:
            # все 4 строки обработаны циклом for-else
            for plane, lr, code in labels_parsed:
                if plane == "sagittal" and code not in range(1, 7):
                    record["parse_notes"].append(
                        f"Нетипичный код сагиттали {code} (ожидают обычно 1–6)"
                    )
                if plane == "frontal" and code not in (4, 5, 6):
                    record["parse_notes"].append(
                        f"Нетипичный код фронтали {code} (по легенде обычно 4–6)"
                    )
        _apply_clinical_name_aliases(record)
        patients.append(record)

    return {
        "class_legend": class_legend,
        "legend_intro_raw": intro_joined,
        "patients": patients,
        "errors": errors,
    }


def build_output(docx_path: Path, parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_file": str(docx_path.resolve()),
        "class_legend": parsed["class_legend"],
        "legend_intro_raw": parsed["legend_intro_raw"],
        "patients": parsed["patients"],
        "errors": parsed["errors"],
        "stats": {
            "patient_count": len(parsed["patients"]),
            "error_count": len(parsed["errors"]),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", "-i", type=Path, required=True, help="Путь к .docx")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Куда записать JSON (по умолчанию: stdout)",
    )
    ap.add_argument("--pretty", action="store_true", help="Форматировать JSON с отступами")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Код выхода 2, если есть ошибки парсинга (errors не пустой)",
    )
    args = ap.parse_args()
    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 1
    if args.input.suffix.lower() != ".docx":
        print("Ожидается расширение .docx", file=sys.stderr)
        return 1

    try:
        paragraphs = _load_paragraphs_from_docx(args.input)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, ValueError) as e:
        print(f"Не удалось прочитать DOCX: {e}", file=sys.stderr)
        return 1

    lines = _flatten_to_lines(paragraphs)
    parsed = parse_docx_lines(lines)
    out = build_output(args.input, parsed)

    text = json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None)
    text += "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.strict and out["errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
