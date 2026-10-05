"""Behavioral checks for the author-ROI osseous research baseline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("mode, outputs", [("binary", 1), ("multilabel", 6)])
def test_cpu_training_checkpoint_reload_and_private_artifacts(tmp_path, mode, outputs):
    _, config = fixture(tmp_path)
    config["mode"] = mode
    report = research.train(config)
    assert report["status"] == "complete"
    assert report["selected_epoch"] in (1, 2)
    assert report["test"]["model"]["auroc"] is not None
    assert report["test"]["constant_train_prevalence"]["auroc"] == 0.5
    assert report["assessment_input"] == "oracle_roi"
    run = Path(config["output_dir"])
    predictions = json.loads((run / "predictions.private.json").read_text())
    model, metadata = research.load_checkpoint(run / "checkpoint.private.pt")
    crop = research.load_crop(
        tmp_path / "data" / "0-L.npz",
        hashlib.sha256((tmp_path / "data" / "0-L.npz").read_bytes()).hexdigest(),
        (2, 1, 8, 8),
    )
    with torch.no_grad():
        result = torch.sigmoid(model(torch.from_numpy(crop.astype(np.float32))[None])).numpy()
    assert result.shape == (1, outputs)
    assert metadata["mode"] == mode
    assert metadata["bindings"]["input_digest"] == report["bindings"]["input_digest"]
    assert report["checkpoint_reload_verified"] is True
    assert len(predictions["records"]) == 80
    public = json.dumps(report)
    assert "sciencedb:" not in public and str(tmp_path) not in public
    assert report["test"]["model"]["bootstrap"]["requested_draws"] == 200


def test_test_label_perturbation_cannot_change_selection(tmp_path):
    index, config = fixture(tmp_path)
    first = research.train(config)
    split_path = Path(config["output_dir"]) / "split.private.json"
    split = json.loads(split_path.read_text())["membership"]
    for record in index["records"]:
        if record["patient_id"] in split["test"]:
            record["binary_target"] = 1 - record["binary_target"]
            record["codes"] = [1] if record["binary_target"] else [0]
    Path(config["index_path"]).write_text(json.dumps(index))
    config.update(output_dir=str(tmp_path / "changed"), split_path=str(split_path))
    second = research.train(config)
    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["thresholds"] == second["thresholds"]
    assert first["selection_history"] == second["selection_history"]


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


def test_model_mean_pooling_is_invariant_to_slice_order():
    torch.manual_seed(7)
    model = research.SliceBagClassifier(6).eval()
    images = torch.rand(2, 3, 1, 8, 8)
    with torch.no_grad():
        assert torch.allclose(model(images), model(images[:, [2, 0, 1]]), atol=1e-7)


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
    with pytest.raises(research.ResearchError, match="no_validation_supported_labels"):
        research.train(config)


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


def test_cli_preflight_is_read_only_and_failure_output_is_private(tmp_path, capsys):
    from tools.run_osseous_research import main

    index, config = fixture(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert main(["--config", str(path), "--preflight"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert not Path(config["output_dir"]).exists()
    index["records"][0]["codes"] = [7]
    Path(config["index_path"]).write_text(json.dumps(index))
    assert main(["--config", str(path), "--preflight"]) == 2
    error = capsys.readouterr().err
    assert json.loads(error)["status"] == "rejected"
    assert str(tmp_path) not in error
    assert index["records"][0]["patient_id"] not in error


def test_train_filesystem_failures_do_not_expose_private_paths(tmp_path):
    _, config = fixture(tmp_path)
    destination = tmp_path / "private-patient-file"
    destination.write_text("reserved")
    config["output_dir"] = str(destination)
    with pytest.raises(research.ResearchError) as caught:
        research.train(config)
    assert str(tmp_path) not in str(caught.value)


def test_split_is_persisted_before_first_optimizer_step_interruption(tmp_path, monkeypatch):
    _, config = fixture(tmp_path)
    expected_digest = research.preflight(config)["bindings"]["split_digest"]

    def interrupt_first_step(self, *args, **kwargs):
        raise RuntimeError("intentional optimizer interruption")

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt_first_step)
    with pytest.raises(research.ResearchError):
        research.train(config)
    output = Path(config["output_dir"])
    assert (output / "split.private.json").is_file()
    split = json.loads((output / "split.private.json").read_text())
    actual_digest = hashlib.sha256(
        json.dumps(split["membership"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert split["split_digest"] == expected_digest == actual_digest
    assert (output / "split.private.json").stat().st_mode & 0o777 == 0o600
    assert not (output / "completion.json").exists()
    assert not (output / "checkpoint.private.pt").exists()


def test_multilabel_calibration_support_excludes_unvalidated_test_labels(tmp_path):
    index, config = fixture(tmp_path)
    split = research.build_split(index["records"])
    ownership = {patient: part for part, patients in split.items() for patient in patients}
    for record in index["records"]:
        if record["binary_target"]:
            record["codes"] = {"train": [1, 2], "validation": [1, 3], "test": [1, 2, 3]}[
                ownership[record["patient_id"]]
            ]
    Path(config["index_path"]).write_text(json.dumps(index))
    config.update(mode="multilabel", bootstrap_draws=0)
    report = research.train(config)
    assert report["train_supported_label_mask"] == [True, True, False, False, False, False]
    assert report["validation_supported_label_mask"] == [True, False, True, False, False, False]
    assert report["evaluation_supported_label_mask"] == [True, False, False, False, False, False]
    assert report["threshold_sources"][:3] == [
        "validation_youden_j",
        "fallback_0.5_no_validation_support",
        "fallback_0.5_no_train_support",
    ]
    assert report["thresholds"][1:3] == [0.5, 0.5]
    assert report["label_limitations"][1] == ["no_validation_class_support"]
    assert report["label_limitations"][2] == ["no_train_class_support"]
    for name in ("model", "constant_train_prevalence"):
        metric = report["test"][name]
        assert metric["labels"][0]["auroc"] is not None
        assert metric["auroc"] == metric["labels"][0]["auroc"]
        assert metric["auprc"] == metric["labels"][0]["auprc"]
        for label in (1, 2):
            assert metric["labels"][label]["auroc"] is None
            assert metric["labels"][label]["auprc"] is None
            assert metric["labels"][label]["supported_for_evaluation"] is False
            assert sum(metric["labels"][label]["confusion"].values()) == 12
            assert metric["labels"][label]["limitations"] == report["label_limitations"][label]
    _, metadata = research.load_checkpoint(Path(config["output_dir"]) / "checkpoint.private.pt")
    for field in (
        "train_supported_label_mask",
        "validation_supported_label_mask",
        "evaluation_supported_label_mask",
        "threshold_sources",
        "label_limitations",
    ):
        assert metadata[field] == report[field]
