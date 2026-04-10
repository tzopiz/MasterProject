#!/usr/bin/env python3
"""
Anonymize patient data in manifest_private.json and tmj_position_labels.json.

Replaces all PII (names, birth dates, visit dates, raw headers) with
deterministic anonymous IDs. The join key (patient_name == name_raw) is
preserved using the same anon ID in both files.

Usage:
    ./venv/bin/python tools/anonymize_labels.py \
        --manifest data/dataset_cbct_public/manifest_private.json \
        --labels   data/tmj_position_labels.json \
        --out-dir  data/anonymized
"""

import argparse
import hashlib
import json
from pathlib import Path


def anon_id(name: str, prefix: str = "patient") -> str:
    """Deterministic anonymous ID from name — same name always maps to same ID."""
    h = hashlib.sha256(name.strip().encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"


def anonymize_manifest(manifest: dict, name_to_anon: dict) -> dict:
    """Keep same field names (patient_name) so build_index() works unchanged."""
    clean_studies = []
    for study in manifest["studies"]:
        name = study["patient_name"].strip()
        anon = name_to_anon.get(name, anon_id(name))
        clean_studies.append({
            "study_id":     study["study_id"],
            "patient_name": anon,   # same key, anonymized value
            "num_files":    study.get("num_files", 0),
        })
    return {
        "note": "Anonymized — patient_name replaced with anon ID",
        "studies": clean_studies,
    }


def anonymize_labels(labels: dict, name_to_anon: dict) -> dict:
    """Keep same field names (name_raw) so build_index() works unchanged."""
    clean_patients = []
    for p in labels["patients"]:
        name = p["name_raw"].strip()
        anon = name_to_anon.get(name, anon_id(name))
        clean_patients.append({
            "patient_number": p["patient_number"],
            "name_raw":       anon,   # same key, anonymized value
            "labels":         p["labels"],
        })
    return {
        "schema_version": labels.get("schema_version", "1.0"),
        "note": "Anonymized — name_raw replaced with anon ID",
        "class_legend": labels.get("class_legend", {}),
        "patients": clean_patients,
        "stats": labels.get("stats", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/dataset_cbct_public/manifest_private.json")
    parser.add_argument("--labels",   default="data/tmj_position_labels.json")
    parser.add_argument("--out-dir",  default="data/anonymized")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    labels_path   = Path(args.labels)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    # Build unified name → anon_id map from labels (patient_number as index)
    name_to_anon: dict = {}
    for p in labels["patients"]:
        name = p["name_raw"].strip()
        name_to_anon[name] = f"patient_{p['patient_number']:04d}"

    # Anonymize
    clean_manifest = anonymize_manifest(manifest, name_to_anon)
    clean_labels   = anonymize_labels(labels, name_to_anon)

    out_manifest = out_dir / "manifest.json"
    out_labels   = out_dir / "tmj_position_labels.json"

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=2, ensure_ascii=False)
    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(clean_labels, f, indent=2, ensure_ascii=False)

    print(f"manifest → {out_manifest}  ({len(clean_manifest['studies'])} studies)")
    print(f"labels   → {out_labels}  ({len(clean_labels['patients'])} patients)")
    print("\nПример записи manifest:")
    print(json.dumps(clean_manifest["studies"][0], ensure_ascii=False, indent=2))
    print("\nПример записи labels:")
    p = clean_labels["patients"][0]
    print(json.dumps({k: v for k, v in p.items() if k != "labels"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
