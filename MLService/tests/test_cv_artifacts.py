"""Actual tiny CPU training/save/reload; no real medical data or quality claim."""

import json
from pathlib import Path

import pytest
import torch

from training import sagittal_binary_cv as cv

from .test_sagittal_binary_cv import _canonical_synthetic_config


@pytest.fixture(autouse=True)
def artifact_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _run(tmp_path):
    cfg = _canonical_synthetic_config(tmp_path)
    cfg.output_dir = str(tmp_path / "runs")
    cfg.run_id = "SYNTHETIC_PRIVATE_RUN"
    result = cv.run_sagittal_binary_cv(cfg)
    return cfg, result, Path(cfg.output_dir) / cfg.run_id


def test_real_cpu_save_reload_and_recompute(tmp_path):
    from models.tmj_binary_position_classifier import load_binary_position_checkpoint
    from training.datasets.tmj_position_dataset import TMJBinaryPositionDataset

    cfg, report, directory = _run(tmp_path)
    analyzed = cv.analyze_sagittal_cv_result(report, show_plots=False, save_curves=False)
    assert analyzed["summary"] == report["summary"]
    assert (
        analyzed["fold_summaries"][0]["train_majority_baseline"]
        == report["folds"][0]["train_majority_baseline"]
    )
    public = (directory / "report.json").read_text()
    assert "SYNTHETIC_PRIVATE_RUN" not in public
    assert str(tmp_path) not in public
    assert "synthetic_" not in public
    assert '"source_id"' not in public
    assert '"patient_id"' not in public
    assert '"study_id"' not in public
    assert '"group_key"' not in public
    assert '"sample_key"' not in public
    assert '"crop_sha256"' not in public
    assert "manifest_path" not in report["config"]
    assert report["status"] == "complete"
    assert report["provenance"]["detector_training_identity"] == "unknown"
    replay = json.loads((directory / "replay.private.json").read_text())
    assert replay["config"]["input_path"] == cfg.input_path
    lookup = {row["sample_key"]: row for row in replay["records"]}
    assert report["provenance"]["source_rechecked_studies"] == 0
    assert report["provenance"]["coordinate_spaces"] == {"voxel": 8}
    seen, recomputed_folds = [], []
    for fold in report["folds"]:
        artifacts = fold["artifacts"]
        model, metadata = load_binary_position_checkpoint(
            directory / artifacts["checkpoint"]["file"],
            expected_sha256=artifacts["checkpoint"]["sha256"],
        )
        serialized_metadata = json.dumps(metadata)
        assert str(tmp_path) not in serialized_metadata
        assert "synthetic" not in serialized_metadata
        assert "source_id" not in serialized_metadata
        assert metadata["trained_tasks"] == ["sagittal"]
        assert metadata["decision_rule"] == ">="
        assert metadata["model_kwargs"]["features"] == [2]
        rows = [
            json.loads(line)
            for line in (directory / artifacts["predictions"]["file"]).read_text().splitlines()
        ]
        recomputed_folds.append(cv.recompute_fold_metrics(rows))
        assert cv.recompute_fold_metrics(rows) == {
            key: value for key, value in fold.items() if key in cv.recompute_fold_metrics(rows)
        }
        train_groups = {r["group_key"] for r in rows if r["role"] == "training"}
        val_groups = {r["group_key"] for r in rows if r["role"] == "validation"}
        assert train_groups.isdisjoint(val_groups)
        for row in rows:
            record = lookup[row["sample_key"]]
            tensor, label = TMJBinaryPositionDataset([record], sagittal_only=True)[0]
            with torch.no_grad():
                logit = model(tensor.unsqueeze(0))[0].item()
                probability = torch.sigmoid(torch.tensor(logit)).item()
            assert label.item() == row["label"]
            assert logit == pytest.approx(row["logit"], abs=1e-6)
            assert probability == pytest.approx(row["probability"], abs=1e-6)
            assert row["decision"] == int(row["probability"] >= metadata["threshold"])
            if row["role"] == "validation":
                seen.append(row["sample_key"])
        assert len(set(seen)) == len(seen)
    assert len(seen) == 16
    assert cv._build_cv_report_dict(cfg, recomputed_folds)["summary"] == report["summary"]
    from models.tmj_binary_position_classifier import TMJBinaryPositionClassifier
    from training.utils.seed import set_seed

    set_seed(cfg.seed)
    initial = TMJBinaryPositionClassifier(
        features=list(cfg.features), fc_hidden=cfg.fc_hidden, dropout=cfg.dropout
    )
    saved = torch.load(
        directory / report["folds"][0]["artifacts"]["checkpoint"]["file"], weights_only=True
    )
    assert not torch.equal(
        initial.backbone[0][0].weight, saved["model_state_dict"]["backbone.0.0.weight"]
    )
    assert (directory / "replay.private.json").stat().st_mode & 0o077 == 0
    sentinel = (directory / "report.json").read_bytes()
    with pytest.raises(ValueError, match="run_directory_exists"):
        cv.run_sagittal_binary_cv(cfg)
    assert (directory / "report.json").read_bytes() == sentinel


def test_failed_second_fold_preserves_evidence(tmp_path, monkeypatch):
    original = cv._train_one_fold

    def train(*args, **kwargs):
        if kwargs["fold_idx"] == 1:
            raise RuntimeError("PRIVATE_PATIENT_NAME /PRIVATE/DATA")
        return original(*args, **kwargs)

    monkeypatch.setattr(cv, "_train_one_fold", train)
    cfg = _canonical_synthetic_config(tmp_path)
    cfg.output_dir, cfg.run_id = str(tmp_path / "runs"), "failure"
    with pytest.raises(RuntimeError, match="cv_run_failed") as caught:
        cv.run_sagittal_binary_cv(cfg)
    assert "PRIVATE" not in str(caught.value)
    directory = Path(cfg.output_dir) / cfg.run_id
    public = (directory / "report.json").read_text()
    assert "PRIVATE" not in public
    report = json.loads(public)
    assert report["status"] == "failed"
    assert report["completed_folds"] == 1
    assert (directory / report["folds"][0]["artifacts"]["checkpoint"]["file"]).is_file()
    assert len((directory / "epochs.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "wrong"),
        ("model_kwargs", {"features": []}),
        ("threshold", float("nan")),
        ("threshold", -0.1),
        ("threshold", 1.1),
        ("model_kwargs", {"in_channels": 1, "features": [2], "fc_hidden": 5, "dropout": 0}),
        ("decision_rule", ">"),
        ("trained_tasks", ["frontal"]),
        ("preprocessing", {}),
        ("model_state_dict", {"bad": torch.tensor([1.0])}),
        ("model_state_dict", {"bad": torch.tensor([float("nan")])}),
    ],
)
def test_checkpoint_contract_refuses_wrong_metadata(tmp_path, field, value):
    from models.tmj_binary_position_classifier import load_binary_position_checkpoint

    _, report, directory = _run(tmp_path)
    path = directory / report["folds"][0]["artifacts"]["checkpoint"]["file"]
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    wrong = tmp_path / "PRIVATE_TOKEN.pth"
    torch.save(payload, wrong)
    with pytest.raises(ValueError, match="invalid_binary_checkpoint") as caught:
        load_binary_position_checkpoint(wrong)
    assert "PRIVATE_TOKEN" not in str(caught.value)


def test_checkpoint_refuses_corruption_and_checksum(tmp_path):
    from models.tmj_binary_position_classifier import load_binary_position_checkpoint

    path = tmp_path / "PRIVATE_TOKEN.pth"
    path.write_bytes(b"corrupt")
    for digest in (None, "0" * 64):
        with pytest.raises(ValueError, match="invalid_binary_checkpoint") as caught:
            load_binary_position_checkpoint(path, expected_sha256=digest)
        assert "PRIVATE_TOKEN" not in str(caught.value)


def test_missing_roi_passport_blocks_before_run_directory(tmp_path):
    cfg = _canonical_synthetic_config(tmp_path)
    from training.roi_provenance import ROIValidationError

    next(Path(cfg.dataset_root).rglob("*.passport.json")).unlink()
    with pytest.raises(ROIValidationError):
        cv.run_sagittal_binary_cv(cfg)
    assert not Path(cfg.output_dir).exists()


def test_legacy_output_json_alias_is_protected(tmp_path):
    cfg = _canonical_synthetic_config(tmp_path)
    cfg.output_json = str(tmp_path / "PRIVATE_OUTPUT_NAME.json")
    report = cv.run_sagittal_binary_cv(cfg)
    directory = tmp_path / "PRIVATE_OUTPUT_NAME"
    assert not Path(cfg.output_json).exists()
    assert (directory / "report.json").is_file()
    assert "PRIVATE_OUTPUT_NAME" not in json.dumps(report)
    with pytest.raises(ValueError, match="run_directory_exists"):
        cv.run_sagittal_binary_cv(cfg)
