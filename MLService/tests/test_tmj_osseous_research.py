"""Patient split and source integrity before osseous model training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training import tmj_osseous_research as research


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(tmp_path, count=40):
    records = []
    data = tmp_path / "data"
    data.mkdir()
    for patient in range(count):
        for side in ("L", "R"):
            target = patient % 2
            path = data / f"{patient}-{side}.npz"
            rng = np.random.default_rng(patient * 2 + (side == "R"))
            images = (rng.random((2, 1, 8, 8)) * 0.1 + target * 0.8).astype(np.float16)
            np.savez_compressed(path, images=images)
            records.append(
                dict(
                    patient_id="sciencedb:" + digest(str(patient)),
                    side=side,
                    codes=[1 + patient % 6] if target else [0],
                    binary_target=target,
                    crop_path=str(path.relative_to(tmp_path)),
                    crop_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sha256=digest(f"source-{patient}"),
                    annotation_sha256=digest(f"annotation-{patient}"),
                    instance_numbers=[1, 9],
                    roi_source="author_annotation",
                    assessment_input="oracle_roi",
                )
            )
    index = dict(
        schema_version=1,
        task="tmj-osseous-author-roi-v1",
        source_doi="10.57760/sciencedb.37727",
        codebook_commit="a4bce89c89b7175e330193f04a282887b5e3f8b0",
        preprocessing=dict(slice_count=2, image_size=8, intensity="per-slice-p1-p99"),
        records=records,
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    config = dict(
        index_path=str(index_path),
        output_dir=str(tmp_path / "run"),
        epochs=2,
        patience=2,
        batch_size=4,
        max_runtime_seconds=30,
        bootstrap_draws=200,
        device="cpu",
    )
    return index, config


def test_preflight_keeps_bilateral_patients_together_and_detects_duplicate_sources(tmp_path):
    index, config = fixture(tmp_path)
    report = research.preflight(config)
    assert report["partitions"]["train"]["patients"] == 28
    assert report["partitions"]["validation"]["patients"] == 6
    assert report["partitions"]["test"]["patients"] == 6
    split = research.build_split(index["records"])
    assert not set(split["train"]) & set(split["test"])
    assert set().union(*map(set, split.values())) == {r["patient_id"] for r in index["records"]}
    patients = {p: part for part, ids in split.items() for p in ids}
    pair = next(
        (a, b)
        for a in index["records"]
        for b in index["records"]
        if patients[a["patient_id"]] != patients[b["patient_id"]]
    )
    pair[1]["source_sha256"] = pair[0]["source_sha256"]
    Path(config["index_path"]).write_text(json.dumps(index))
    with pytest.raises(research.ResearchError, match="cross_partition_duplicate"):
        research.preflight(config)


@pytest.mark.parametrize(
    "mutation", ["checksum", "shape", "nan", "range", "dtype", "codes", "task", "instances"]
)
def test_preflight_rejects_mutated_inputs_without_private_details(tmp_path, mutation):
    index, config = fixture(tmp_path)
    record = index["records"][0]
    crop = tmp_path / record["crop_path"]
    if mutation == "checksum":
        crop.write_bytes(b"private bad data")
    elif mutation in ("shape", "nan", "range", "dtype"):
        images = np.ones((2, 1, 8, 8), dtype=np.float16)
        if mutation == "shape":
            images = images[:, 0]
        if mutation == "nan":
            images[0, 0, 0, 0] = np.nan
        if mutation == "range":
            images[0, 0, 0, 0] = 2
        if mutation == "dtype":
            images = images.astype(np.float32)
        np.savez_compressed(crop, images=images)
        record["crop_sha256"] = hashlib.sha256(crop.read_bytes()).hexdigest()
    elif mutation == "codes":
        record["codes"] = [0, 1]
    elif mutation == "task":
        index["task"] = "position"
    else:
        record["instance_numbers"] = [1]
    Path(config["index_path"]).write_text(json.dumps(index))
    with pytest.raises(research.ResearchError) as caught:
        research.preflight(config)
    assert str(tmp_path) not in str(caught.value)
    assert record["patient_id"] not in str(caught.value)


def test_config_paths_are_relative_and_unbounded_runs_are_rejected(tmp_path):
    _, config = fixture(tmp_path)
    config.update(index_path="index.json", output_dir="run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = research.load_config(path)
    assert loaded["index_path"] == str(tmp_path / "index.json")
    config["epochs"] = 41
    with pytest.raises(research.ResearchError):
        research.preflight(config)


def test_development_patients_never_enter_test_even_on_split_replay(tmp_path):
    index, config = fixture(tmp_path)
    initial = research.build_split(index["records"])
    inspected = initial["test"][:2]
    split = research.build_split(index["records"], development_only_patients=inspected)
    assert not set(inspected) & set(split["test"])
    assert len(split["test"]) == 6
    config["development_only_patients"] = inspected
    research.preflight(config)
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            dict(
                task=research.TASK,
                seed=42,
                membership=initial,
                split_digest=hashlib.sha256(
                    json.dumps(initial, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )
        )
    )
    config["split_path"] = str(path)
    with pytest.raises(research.ResearchError, match="development_patient_in_test"):
        research.preflight(config)


@pytest.mark.parametrize("kind", ["array", "source_pixel"])
def test_duplicate_pixels_cannot_hide_behind_archive_or_dicom_headers(tmp_path, kind):
    index, config = fixture(tmp_path)
    first, other = index["records"][0], index["records"][2]
    if kind == "source_pixel":
        first["source_pixel_sha256"] = other["source_pixel_sha256"] = digest("same pixels")
    else:
        original = np.load(tmp_path / first["crop_path"])["images"]
        destination = tmp_path / other["crop_path"]
        np.savez(destination, images=original)  # distinct container, identical data
        other["crop_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    Path(config["index_path"]).write_text(json.dumps(index))
    with pytest.raises(research.ResearchError, match="duplicate_pixel_patient_identity"):
        research.preflight(config)


def test_multilabel_selection_requires_train_and_validation_supported_label(tmp_path):
    index, config = fixture(tmp_path)
    split = research.build_split(index["records"])
    for record in index["records"]:
        if record["binary_target"]:
            record["codes"] = [2] if record["patient_id"] in split["validation"] else [1]
    Path(config["index_path"]).write_text(json.dumps(index))
    config["mode"] = "multilabel"
    with pytest.raises(research.ResearchError, match="no_validation_supported_labels"):
        research.preflight(config)
    assert not Path(config["output_dir"]).exists()


def test_bare_source_qualified_hash_patient_ids_are_accepted(tmp_path):
    index, config = fixture(tmp_path)
    for record in index["records"]:
        record["patient_id"] = record["patient_id"].split(":")[-1]
    Path(config["index_path"]).write_text(json.dumps(index))
    assert research.preflight(config)["status"] == "ready"


def test_repeated_sparse_instance_padding_is_valid(tmp_path):
    index, config = fixture(tmp_path)
    for record in index["records"]:
        record["instance_numbers"] = [9, 9]
    Path(config["index_path"]).write_text(json.dumps(index))
    assert research.preflight(config)["status"] == "ready"
