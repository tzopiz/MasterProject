"""Explicit review is required before failed patients leave the training cohort."""

import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from tools import curate_tmj_od3d as curation


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "data").mkdir()
    names = [f"private-patient-{i}" for i in range(20)]
    metadata = root / "metadata.private.csv"
    fields = [
        "anonymous_id",
        "patient_sex",
        "age_years",
        "age_group",
        "selected_slice_min",
        "selected_slice_max",
        "selected_slice_count",
        "label_L",
        "label_R",
    ]
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow(dict(zip(fields, [name, "M", 30, "30-39", 1, 2, 2, "0", "0"])))
    monkeypatch.setattr(curation, "EXPECTED_PATIENT_COUNT", 20)
    monkeypatch.setattr(curation, "METADATA_SHA256", digest(metadata))
    records, receipts = [], []
    for i, name in enumerate(names):
        patient = hashlib.sha256((curation.SOURCE_DOI + "\0" + name).encode()).hexdigest()
        failure = i == 0
        receipts.append(
            dict(
                patient_id=patient,
                tar_start=i * 512,
                tar_end=curation.ARCHIVE_BYTES if i == 19 else (i + 1) * 512,
                status="failed" if failure else "prepared",
                code="annotation_side_conflict" if failure else "ok",
                side_count=0 if failure else 2,
            )
        )
        if failure:
            continue
        for side in ("L", "R"):
            crop = root / "data" / f"{patient}-{side}.npz"
            np.savez_compressed(crop, images=np.full((2, 1, 8, 8), i / 20, dtype=np.float16))
            records.append(
                dict(
                    patient_id=patient,
                    side=side,
                    codes=[0],
                    binary_target=0,
                    crop_path=str(crop.relative_to(root)),
                    crop_sha256=digest(crop),
                    source_sha256="a" * 64,
                    annotation_sha256="b" * 64,
                    instance_numbers=[1, 2],
                    roi_source="author_annotation",
                    assessment_input="oracle_roi",
                )
            )
    state = dict(
        schema_version=1,
        task=curation.TASK,
        source_doi=curation.SOURCE_DOI,
        codebook_commit=curation.CODEBOOK_COMMIT,
        metadata_sha256=digest(metadata),
        preprocessing=dict(slice_count=2, image_size=8, intensity="per-slice-p1-p99"),
        records=records,
        processed_patients=names,
        exclusions={},
        failures={"annotation_side_conflict": 1},
        complete=True,
        next_offset=curation.ARCHIVE_BYTES,
        patient_receipts=receipts,
    )
    state_path = root / "preparation.private.json"
    index_path = root / "index.private.json"
    write(state_path, state)

    def save_index():
        value = {
            k: v
            for k, v in state.items()
            if k not in ("processed_patients", "next_offset", "patient_receipts")
        }
        value["patient_count"] = len(state["processed_patients"])
        write(index_path, value)

    save_index()
    policy = dict(
        schema_version=1,
        task=curation.TASK,
        source_preparation_sha256=digest(state_path),
        reviews=[
            dict(
                patient_id=receipts[0]["patient_id"],
                code="annotation_side_conflict",
                disposition="exclude",
                reason="Reviewed source filenames conflict with author side labels.",
            )
        ],
    )
    policy_path = tmp_path / "policy.private.json"
    write(policy_path, policy)
    return root, state, policy, policy_path, save_index


def test_success_preserves_failure_provenance_copies_only_crops_and_protects_identity(
    tmp_path, monkeypatch, capsys
):
    root, state, policy, policy_path, _ = cohort(tmp_path, monkeypatch)
    before = {
        name: digest(root / name)
        for name in ("preparation.private.json", "index.private.json", "metadata.private.csv")
    }
    output = tmp_path / "curated"
    report = curation.curate(root, policy_path, output)
    index = json.loads((output / "index.private.json").read_text())
    assert index["complete"] is True and index["failures"] == {}
    assert index["source_failures"] == {"annotation_side_conflict": 1}
    assert index["patient_count"] == 19
    assert len(index["records"]) == 38
    assert index["curation"]["source_preparation_sha256"] == before["preparation.private.json"]
    assert index["curation"]["source_index_sha256"] == before["index.private.json"]
    assert index["curation"]["policy_sha256"] == digest(policy_path)
    assert report["excluded_patients"] == 1 and report["accepted_sides"] == 38
    assert "private-patient-" not in json.dumps(index)
    assert "private-patient-" not in json.dumps(report)
    assert "processed_patients" not in index and "patient_receipts" not in index
    assert set(p.name for p in output.iterdir()) == {
        "data",
        "index.private.json",
        "curation-report.json",
    }
    assert len(list((output / "data").iterdir())) == 38
    assert (output.stat().st_mode & 0o777) == 0o700
    assert (output / "index.private.json").stat().st_mode & 0o777 == 0o600
    assert before == {name: digest(root / name) for name in before}
    assert (
        curation.main(
            [
                "--source-root",
                str(root),
                "--policy",
                str(policy_path),
                "--output-root",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["accepted_patients"] == 19


@pytest.mark.parametrize(
    "mutation",
    [
        "incomplete",
        "offset",
        "coverage",
        "receipt_count",
        "receipt_side_count",
        "receipt_hash",
        "index",
        "metadata",
        "lock",
        "crop",
        "partial_failed_record",
    ],
)
def test_source_integrity_failures_are_rejected_without_output(tmp_path, monkeypatch, mutation):
    root, state, policy, policy_path, save_index = cohort(tmp_path, monkeypatch)
    if mutation == "incomplete":
        state["complete"] = False
    elif mutation == "offset":
        state["next_offset"] -= 1
    elif mutation == "coverage":
        state["processed_patients"][-1] = "unknown-private-patient"
    elif mutation == "receipt_count":
        state["patient_receipts"].pop()
    elif mutation == "receipt_side_count":
        state["patient_receipts"][1]["side_count"] = 1
    elif mutation == "receipt_hash":
        state["patient_receipts"][1]["patient_id"] = "f" * 64
    elif mutation == "index":
        pass
    elif mutation == "metadata":
        (root / "metadata.private.csv").write_text("altered")
    elif mutation == "lock":
        (root / "preparation.lock").write_text("{}")
    elif mutation == "crop":
        (root / state["records"][0]["crop_path"]).write_bytes(b"changed")
    else:
        state["records"][0]["patient_id"] = state["patient_receipts"][0]["patient_id"]
    write(root / "preparation.private.json", state)
    policy["source_preparation_sha256"] = digest(root / "preparation.private.json")
    write(policy_path, policy)
    if mutation != "index":
        save_index()
    else:
        index = json.loads((root / "index.private.json").read_text())
        index["patient_count"] = 19
        write(root / "index.private.json", index)
    output = tmp_path / "curated"
    with pytest.raises(curation.CurationError) as caught:
        curation.curate(root, policy_path, output)
    assert not output.exists()
    assert str(tmp_path) not in str(caught.value)
    assert "private-patient-" not in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unreviewed",
        "unknown_code",
        "wrong_hash",
        "wrong_state",
        "include",
        "blank_reason",
        "duplicate",
    ],
)
def test_policy_requires_exact_reviewed_exclusions(tmp_path, monkeypatch, mutation):
    root, state, policy, policy_path, save_index = cohort(tmp_path, monkeypatch)
    if mutation == "missing":
        policy_path.unlink()
    elif mutation == "unreviewed":
        policy["reviews"] = []
    elif mutation == "unknown_code":
        state["patient_receipts"][0]["code"] = "unknown_annotation_failure"
        state["failures"] = {"unknown_annotation_failure": 1}
        write(root / "preparation.private.json", state)
        save_index()
        policy["source_preparation_sha256"] = digest(root / "preparation.private.json")
        policy["reviews"][0]["code"] = "unknown_annotation_failure"
    elif mutation == "wrong_hash":
        policy["reviews"][0]["patient_id"] = "f" * 64
    elif mutation == "wrong_state":
        policy["source_preparation_sha256"] = "f" * 64
    elif mutation == "include":
        policy["reviews"][0]["disposition"] = "include"
    elif mutation == "blank_reason":
        policy["reviews"][0]["reason"] = " "
    else:
        policy["reviews"].append(copy.deepcopy(policy["reviews"][0]))
    if mutation != "missing":
        write(policy_path, policy)
    with pytest.raises(curation.CurationError):
        curation.curate(root, policy_path, tmp_path / "curated")
    assert not (tmp_path / "curated").exists()


def test_overlap_and_existing_destination_are_never_modified(tmp_path, monkeypatch):
    root, _, _, policy_path, _ = cohort(tmp_path, monkeypatch)
    marker = tmp_path / "existing"
    marker.mkdir()
    (marker / "keep").write_text("retained")
    for output in (root / "curated", tmp_path, marker):
        with pytest.raises(curation.CurationError):
            curation.curate(root, policy_path, output)
    assert (marker / "keep").read_text() == "retained"


def test_size_limit_is_checked_before_copy(tmp_path, monkeypatch):
    root, _, _, policy_path, _ = cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(curation, "MAX_PREPARED_BYTES", 1)
    with pytest.raises(curation.CurationError):
        curation.curate(root, policy_path, tmp_path / "curated")
    assert not (tmp_path / "curated").exists()


def test_zero_crop_prepared_patients_do_not_inflate_retained_patient_count(tmp_path, monkeypatch):
    root, state, policy, policy_path, save_index = cohort(tmp_path, monkeypatch)
    empty = state["patient_receipts"][1]["patient_id"]
    state["records"] = [r for r in state["records"] if r["patient_id"] != empty]
    state["patient_receipts"][1]["side_count"] = 0
    state["exclusions"] = {"missing_label": 2}
    write(root / "preparation.private.json", state)
    save_index()
    policy["source_preparation_sha256"] = digest(root / "preparation.private.json")
    write(policy_path, policy)
    output = tmp_path / "curated"
    report = curation.curate(root, policy_path, output)
    index = json.loads((output / "index.private.json").read_text())
    assert index["patient_count"] == 18
    assert index["accepted_source_patient_count"] == 19
    assert index["exclusions"] == {"missing_label": 2}
    assert report["accepted_patients"] == 18 and report["accepted_source_patients"] == 19


def test_copy_race_cannot_publish_changed_source(tmp_path, monkeypatch):
    root, state, _, policy_path, _ = cohort(tmp_path, monkeypatch)
    original = curation.shutil.copyfile

    def mutate_source(src, dest):
        result = original(src, dest)
        with (root / "preparation.private.json").open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(curation.shutil, "copyfile", mutate_source)
    output = tmp_path / "curated"
    with pytest.raises(curation.CurationError):
        curation.curate(root, policy_path, output)
    assert not (output / "index.private.json").exists()


def test_late_mutation_of_already_copied_crop_is_rejected(tmp_path, monkeypatch):
    root, state, _, policy_path, _ = cohort(tmp_path, monkeypatch)
    original = curation.shutil.copyfile
    first = root / state["records"][0]["crop_path"]
    calls = []

    def change_earlier_crop(src, dest):
        result = original(src, dest)
        calls.append(src)
        if len(calls) == 2:
            first.write_bytes(b"modified after its own copy verification")
        return result

    monkeypatch.setattr(curation.shutil, "copyfile", change_earlier_crop)
    output = tmp_path / "curated"
    with pytest.raises(curation.CurationError):
        curation.curate(root, policy_path, output)
    assert not (output / "index.private.json").exists()


def archive_prefix(tmp_path, monkeypatch):
    root, state, policy, policy_path, save_index = cohort(tmp_path, monkeypatch)
    state["processed_patients"] = state["processed_patients"][:6]
    state["patient_receipts"] = state["patient_receipts"][:6]
    patients = {row["patient_id"] for row in state["patient_receipts"]}
    state["records"] = [row for row in state["records"] if row["patient_id"] in patients]
    state.update(complete=False, next_offset=state["patient_receipts"][-1]["tar_end"])
    write(root / "preparation.private.json", state)
    save_index()
    policy["source_preparation_sha256"] = digest(root / "preparation.private.json")
    policy["cohort"] = dict(
        kind="archive-prefix", source_patient_count=6, source_end_offset=state["next_offset"]
    )
    write(policy_path, policy)
    return root, state, policy, policy_path, save_index


def source_snapshot(root):
    return {str(path.relative_to(root)): digest(path) for path in root.rglob("*") if path.is_file()}


def test_explicit_archive_prefix_is_curated_without_completing_or_mutating_source(
    tmp_path, monkeypatch
):
    root, state, policy, policy_path, _ = archive_prefix(tmp_path, monkeypatch)
    before = source_snapshot(root)
    report = curation.curate(root, policy_path, tmp_path / "curated")
    index = json.loads((tmp_path / "curated" / "index.private.json").read_text())
    assert index["complete"] is True
    assert index["cohort"] == dict(
        kind="archive-prefix",
        source_complete=False,
        release_patient_count=20,
        source_patient_count=6,
        source_end_offset=state["next_offset"],
    )
    assert report["cohort"] == index["cohort"]
    assert index["patient_count"] == 5 and len(index["records"]) == 10
    assert report["source_patients"] == 6 and report["accepted_sides"] == 10
    assert index["source_failures"] == {"annotation_side_conflict": 1}
    assert index["curation"]["source_preparation_sha256"] == policy["source_preparation_sha256"]
    assert "private-patient-" not in json.dumps(index) + json.dumps(report)
    assert "processed_patients" not in index and "patient_receipts" not in index
    assert before == source_snapshot(root)
    assert json.loads((root / "preparation.private.json").read_text())["complete"] is False


def test_archive_prefix_still_requires_explicit_opt_in(tmp_path, monkeypatch):
    root, _, policy, policy_path, _ = archive_prefix(tmp_path, monkeypatch)
    del policy["cohort"]
    write(policy_path, policy)
    before = source_snapshot(root)
    with pytest.raises(curation.CurationError, match="^source_incomplete$"):
        curation.curate(root, policy_path, tmp_path / "curated")
    assert not (tmp_path / "curated").exists() and before == source_snapshot(root)


@pytest.mark.parametrize(
    "selection",
    [
        None,
        [],
        False,
        {},
        {"kind": "unknown"},
        {"kind": "archive-prefix", "source_patient_count": 6},
        {"kind": "archive-prefix", "source_patient_count": True, "source_end_offset": 3072},
        {"kind": "archive-prefix", "source_patient_count": 6.0, "source_end_offset": 3072},
        {"kind": "archive-prefix", "source_patient_count": 0, "source_end_offset": 3072},
        {"kind": "archive-prefix", "source_patient_count": 20, "source_end_offset": 3072},
        {"kind": "archive-prefix", "source_patient_count": 6, "source_end_offset": True},
        {"kind": "archive-prefix", "source_patient_count": 6, "source_end_offset": "3072"},
        {"kind": "archive-prefix", "source_patient_count": 6, "source_end_offset": 0},
        {
            "kind": "archive-prefix",
            "source_patient_count": 6,
            "source_end_offset": curation.ARCHIVE_BYTES,
        },
        {
            "kind": "archive-prefix",
            "source_patient_count": 6,
            "source_end_offset": 3072,
            "extra": 1,
        },
    ],
)
def test_archive_prefix_rejects_malformed_policy(tmp_path, monkeypatch, selection):
    root, _, policy, policy_path, _ = archive_prefix(tmp_path, monkeypatch)
    policy["cohort"] = selection
    write(policy_path, policy)
    with pytest.raises(curation.CurationError, match="^invalid_cohort_policy$"):
        curation.curate(root, policy_path, tmp_path / "curated")
    assert not (tmp_path / "curated").exists()


@pytest.mark.parametrize(
    "mutation, code",
    [
        ("count", "source_coverage_mismatch"),
        ("offset", "source_incomplete"),
        ("complete", "source_incomplete"),
        ("complete_type", "source_incomplete"),
        ("offset_type", "source_incomplete"),
        ("hash", "policy_source_mismatch"),
        ("unknown_patient", "source_coverage_mismatch"),
        ("duplicate_patient", "source_coverage_mismatch"),
        ("metadata_count", "source_coverage_mismatch"),
        ("receipt_count", "receipt_coverage_mismatch"),
        ("receipt_gap", "invalid_receipt"),
        ("receipt_end", "source_failure_mismatch"),
        ("unreviewed", "policy_receipt_mismatch"),
        ("lock", "source_locked"),
        ("crop", "crop_changed"),
    ],
)
def test_archive_prefix_retains_integrity_review_and_privacy_gates(
    tmp_path, monkeypatch, mutation, code
):
    root, state, policy, policy_path, save_index = archive_prefix(tmp_path, monkeypatch)
    if mutation == "count":
        policy["cohort"]["source_patient_count"] += 1
    elif mutation == "offset":
        policy["cohort"]["source_end_offset"] += 512
    elif mutation == "complete":
        state["complete"] = True
    elif mutation == "complete_type":
        state["complete"] = 0
    elif mutation == "offset_type":
        state["next_offset"] = float(state["next_offset"])
    elif mutation == "unknown_patient":
        state["processed_patients"][-1] = "unknown-private-patient"
    elif mutation == "duplicate_patient":
        state["processed_patients"][-1] = state["processed_patients"][0]
    elif mutation == "metadata_count":
        metadata = root / "metadata.private.csv"
        metadata.write_text("\n".join(metadata.read_text().splitlines()[:-1]) + "\n")
        monkeypatch.setattr(curation, "METADATA_SHA256", digest(metadata))
        state["metadata_sha256"] = digest(metadata)
    elif mutation == "receipt_count":
        state["patient_receipts"].pop()
    elif mutation == "receipt_gap":
        state["patient_receipts"][2]["tar_start"] += 1
    elif mutation == "receipt_end":
        state["patient_receipts"][-1]["tar_end"] += 512
    elif mutation == "unreviewed":
        policy["reviews"] = []
    elif mutation == "lock":
        (root / "preparation.lock").write_text("{}")
    elif mutation == "crop":
        (root / state["records"][0]["crop_path"]).write_bytes(b"changed")
    write(root / "preparation.private.json", state)
    save_index()
    policy["source_preparation_sha256"] = (
        "f" * 64 if mutation == "hash" else digest(root / "preparation.private.json")
    )
    write(policy_path, policy)
    before = source_snapshot(root)
    with pytest.raises(curation.CurationError, match=f"^{code}$") as caught:
        curation.curate(root, policy_path, tmp_path / "curated")
    assert not (tmp_path / "curated").exists() and before == source_snapshot(root)
    assert "private-patient-" not in str(caught.value) and str(tmp_path) not in str(caught.value)
