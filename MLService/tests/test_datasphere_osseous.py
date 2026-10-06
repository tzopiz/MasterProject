import json
import subprocess
from pathlib import Path

import pytest

from tools import datasphere_osseous as cloud
from tools.datasphere_research import CloudLaunchError, _check_bundle, confirm_bundle
from training.tmj_osseous_research import train

from .test_tmj_osseous_research import fixture


def config_fixture(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    index, config = fixture(data)
    index.update(complete=True, failures={})
    Path(config["index_path"]).write_text(json.dumps(index))
    path = tmp_path / "config.private.json"
    path.write_text(json.dumps(config))
    return path, index, config


def test_local_staging_keeps_frozen_patient_members_and_osseous_mapping(tmp_path, monkeypatch):
    path, index, config = config_fixture(tmp_path)

    def forbidden(*a, **k):
        raise AssertionError("prepare must not call provider")

    monkeypatch.setattr(subprocess, "run", forbidden)
    bundle = tmp_path / "bundle"
    plan = cloud.prepare_bundle(path, bundle, project_id="synthetic-project")
    staged = json.loads((bundle / "payload/index.private.json").read_text())
    assert staged == index
    split = json.loads((bundle / "payload/split.private.json").read_text())
    assert split["split_digest"] == plan["split_digest"] == plan["bindings"]["split_digest"]
    config = json.loads((bundle / "payload/research.private.json").read_text())
    assert config["device"] == "cuda" and config["split_path"] == "split.private.json"
    staged_summary = cloud.preflight(cloud.load_config(bundle / "payload/research.private.json"))
    assert plan["bindings"] == staged_summary["bindings"]
    assert plan["training_window_compute_estimate"] == 168.48 * 30 / 3600
    assert _check_bundle(bundle, plan["plan_sha256"])["task"] == "tmj-osseous-author-roi-v1"
    job = json.loads((bundle / "job.yaml").read_text())
    assert job["env"]["docker"] == "system-python-3-10"
    assert plan["docker_image"] == "system-python-3-10"
    assert "python" not in job["env"]
    assert plan["python"] == "3.10"
    assert "tools/osseous_cloud_bootstrap.sh" in plan["manifest"]
    assert job["cmd"].startswith("bash payload/tools/osseous_cloud_bootstrap.sh ")
    assert "torch==2.6.0+cu118" in (bundle / "payload/requirements.txt").read_text()


def test_incomplete_dataset_and_changed_payload_refused_before_any_submission(tmp_path):
    path, index, config = config_fixture(tmp_path)
    index["complete"] = False
    Path(config["index_path"]).write_text(json.dumps(index))
    with pytest.raises(CloudLaunchError, match="dataset_preparation_incomplete"):
        cloud.prepare_bundle(path, tmp_path / "bundle", project_id="synthetic")
    index["complete"] = True
    Path(config["index_path"]).write_text(json.dumps(index))
    bundle = tmp_path / "bundle"
    plan = cloud.prepare_bundle(path, bundle, project_id="synthetic")
    crop = bundle / "payload" / index["records"][0]["crop_path"]
    crop.chmod(0o600)
    crop.write_bytes(b"changed")
    with pytest.raises(CloudLaunchError, match="bundle_changed"):
        confirm_bundle(
            bundle, plan["plan_sha256"], transport=lambda *a, **k: pytest.fail("must not submit")
        )
    assert not (bundle / "submission.private.json").exists()


def test_completion_requires_actual_artifacts_and_bound_hashes(tmp_path):
    _, _, config = config_fixture(tmp_path)
    train(config)
    root = Path(config["output_dir"])
    assert cloud.artifacts_complete(root)
    (root / "predictions.private.json").write_text("{}")
    assert not cloud.artifacts_complete(root)


def test_worker_timeout_kills_own_group_and_reports_failure(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.mkdir()
    config = payload / "config.json"
    config.write_text("{}")

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("private-command", timeout)
            return -9

    monkeypatch.setattr(cloud, "preflight", lambda config: {"bindings": {}})
    monkeypatch.setattr(cloud, "load_config", lambda path: {})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    killed = []
    monkeypatch.setattr(cloud.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    report = cloud.worker(config, 1)
    assert report["training_status"] == "timeout" and killed == [(12345, cloud.signal.SIGKILL)]
    assert (
        json.loads((tmp_path / "results/training-status.json").read_text())["status"] == "timeout"
    )


def test_completion_rejects_wrong_semantics_and_unexpected_plan_bindings(tmp_path):
    _, _, config = config_fixture(tmp_path)
    report = train(config)
    root = Path(config["output_dir"])
    expected = dict(report["bindings"])
    assert cloud.artifacts_complete(root, expected)
    wrong = dict(expected, input_digest="a" * 64)
    assert not cloud.artifacts_complete(root, wrong)
    for name in ("report.json", "completion.json"):
        document = json.loads((root / name).read_text())
        document["bindings"]["codebook_commit"] = "wrong-codebook"
        (root / name).write_text(json.dumps(document))
    completion = json.loads((root / "completion.json").read_text())
    completion["artifact_digests"]["report.json"] = cloud._hash_file(root / "report.json")
    (root / "completion.json").write_text(json.dumps(completion))
    assert not cloud.artifacts_complete(root)


def test_full_cohort_index_above_one_mib_stages_without_cloud_io(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    data.mkdir()
    index, config = fixture(data, count=1043)
    index.update(complete=True, failures={})
    index_path = Path(config["index_path"])
    index_path.write_text(json.dumps(index))
    assert index_path.stat().st_size > 1024**2 and len(index["records"]) == 2086
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("offline staging contacted provider")

    monkeypatch.setattr(subprocess, "run", forbidden)
    plan = cloud.prepare_bundle(config_path, tmp_path / "bundle", project_id="synthetic")
    staged = cloud._read_index(tmp_path / "bundle/payload/index.private.json")
    assert staged == index and plan["partitions"]["train"]["patients"] > 700
    with pytest.raises(CloudLaunchError, match="invalid_private_file"):
        cloud._read(index_path)  # Lifecycle reader keeps its narrower bound.


@pytest.mark.parametrize(
    "raw",
    [
        b'{"records":[],"records":[]}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e999}',
        b'{"value":"\xff"}',
        b"{",
    ],
)
def test_index_reader_refuses_ambiguous_nonfinite_or_non_utf8_json(tmp_path, raw):
    path = tmp_path / "private-index.json"
    path.write_bytes(raw)
    with pytest.raises(CloudLaunchError, match="^invalid_private_index$") as error:
        cloud._read_index(path)
    assert error.value.__cause__ is None


def test_index_reader_has_a_separate_enforced_byte_limit(tmp_path, monkeypatch):
    path = tmp_path / "private-index.json"
    path.write_bytes(b'{"records":[]}' + b" " * 32)
    monkeypatch.setattr(cloud, "MAX_INDEX_BYTES", 32)
    with pytest.raises(CloudLaunchError, match="^invalid_private_index$"):
        cloud._read_index(path)


@pytest.mark.parametrize("action", ["status", "cancel", "results", "reconcile"])
def test_lifecycle_cli_dispatches_authorized_command_without_submission(
    tmp_path, monkeypatch, capsys, action
):
    import sys

    calls = []
    bundle = str(tmp_path / "bundle")
    destination = str(tmp_path / "download")

    def unexpected(*args, **kwargs):
        raise AssertionError("lifecycle command must not submit/train")

    monkeypatch.setattr(cloud, "confirm_bundle", unexpected)
    monkeypatch.setattr(cloud, "worker", unexpected)

    def status(bundle_dir, *, cli_path):
        calls.append((bundle_dir, cli_path))
        return {"provider_status": "EXECUTING"}

    def cancel(bundle_dir, *, cli_path):
        calls.append((bundle_dir, cli_path))
        return {"status": "cancel_requested"}

    def results(bundle_dir, destination, cli_path):
        calls.append((bundle_dir, destination, cli_path))
        return {"results_ready": False, "training_status": "failed"}

    def reconcile(bundle_dir, job_id, operation_id, *, acknowledge_match, cli_path):
        calls.append((bundle_dir, job_id, operation_id, acknowledge_match, cli_path))
        return {"status": "reconciled"}

    monkeypatch.setattr(cloud, "job_status", status)
    monkeypatch.setattr(cloud, "cancel_job", cancel)
    monkeypatch.setattr(cloud, "results", results)
    monkeypatch.setattr(cloud, "reconcile_submission", reconcile, raising=False)
    argv = ["datasphere_osseous", action, "--bundle", bundle, "--cli-path", "synthetic-cli"]
    if action == "results":
        argv += ["--destination", destination]
    if action == "reconcile":
        argv += [
            "--job-id",
            "synthetic-job",
            "--operation-id",
            "synthetic-operation",
            "--acknowledge-match",
        ]
    monkeypatch.setattr(sys, "argv", argv)
    assert cloud.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert bundle not in json.dumps(output)
    expected = {
        "status": (bundle, "synthetic-cli"),
        "cancel": (bundle, "synthetic-cli"),
        "results": (bundle, destination, "synthetic-cli"),
        "reconcile": (bundle, "synthetic-job", "synthetic-operation", True, "synthetic-cli"),
    }
    assert calls == [expected[action]]


def test_reconcile_cli_requires_explicit_match_acknowledgment(tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "datasphere_osseous",
            "reconcile",
            "--bundle",
            str(tmp_path),
            "--job-id",
            "synthetic-job",
            "--operation-id",
            "synthetic-operation",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cloud.main()
    assert error.value.code == 2
    assert "acknowledge" in capsys.readouterr().err


def test_prepare_rejects_oversized_index_before_unbounded_consumer_read(tmp_path, monkeypatch):
    config_path, _, _ = config_fixture(tmp_path)
    monkeypatch.setattr(cloud, "MAX_INDEX_BYTES", 32)

    def forbidden(config):
        raise AssertionError("consumer read preceded byte bound")

    monkeypatch.setattr(cloud, "preflight", forbidden)
    with pytest.raises(CloudLaunchError, match="^invalid_private_index$"):
        cloud.prepare_bundle(config_path, tmp_path / "bundle", project_id="synthetic")


def test_cloud_requirements_are_accepted_by_datasphere_requirement_parser():
    # DataSphere's parser accepts these pip flags but rejects comment lines.
    from packaging.requirements import Requirement

    path = Path(cloud.__file__).with_name("osseous_cloud_requirements.txt")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    for line in lines:
        if line.startswith("--extra-index-url "):
            continue
        requirement = Requirement(line)
        assert not requirement.marker and not requirement.url
