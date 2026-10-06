"""Local synthetic bundles and official-CLI transport doubles; no cloud writes."""

import json
import random
import subprocess
from pathlib import Path

import pytest

from .test_sagittal_binary_cv import _canonical_synthetic_config


def launcher():
    from tools import datasphere_research

    return datasphere_research


def prepare(tmp_path):
    cfg = _canonical_synthetic_config(tmp_path)
    raw = {
        "schema_version": 1,
        "mode": "binary",
        "input_path": cfg.input_path,
        "dataset_root": cfg.dataset_root,
        "output_dir": str(tmp_path / "runs"),
        "run_id": "SyntheticSensitiveRun",
        "n_splits": 2,
        "epochs": 1,
        "batch_size": 2,
        "features": [2],
        "fc_hidden": 4,
        "device": "cpu",
    }
    config = tmp_path / "SyntheticSensitiveConfig.private.json"
    config.write_text(json.dumps(raw))
    module = launcher()
    bundle = tmp_path / "bundle"
    plan = module.prepare_cloud_bundle(
        config,
        bundle,
        project_id="synthetic-project",
        resource="gt4.1",
        max_runtime_seconds=120,
        hourly_price="10.5",
        currency="RUB",
        price_as_of="2026-10-03T10:00:00+03:00",
    )
    return module, bundle, plan


def test_prepare_uploads_only_validated_rekeyed_payload(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    payload = bundle / "payload"
    uploaded = [p for p in payload.rglob("*") if p.is_file()]
    assert plan["study_count"] == 8 and plan["patient_count"] == 8
    assert plan["training_runtime_seconds"] == 120
    assert plan["money_cap_guaranteed"] is False
    assert plan["upload_bytes"] == sum(p.stat().st_size for p in uploaded)
    for path in uploaded:
        assert "synthetic" not in path.name.lower()
        if path.suffix in (".json", ".yaml"):
            text = path.read_text()
            assert str(tmp_path) not in text
            assert "SyntheticSensitive" not in text
            assert "synthetic-project" not in text
            assert 'patient_id": "p0' not in text
            assert "series_path" not in text
    from tools.run_research import load_research_config, preflight_research

    assert preflight_research(load_research_config(payload / "research.private.json"))["ready"]
    index = json.loads((payload / "index.private.json").read_text())
    assert len({(row["source_id"], row["patient_id"]) for row in index["studies"]}) == 8
    assert json.loads((payload / "research.private.json").read_text())["run_id"] == "research"
    assert not any(p.name in {"AGENTS.md", "tasks.json", ".git"} for p in uploaded)
    with pytest.raises(module.CloudLaunchError, match="bundle_exists"):
        module.prepare_cloud_bundle(
            payload / "research.private.json",
            bundle,
            project_id="x",
            max_runtime_seconds=120,
            hourly_price="1",
            currency="RUB",
            price_as_of="2026-10-03T10:00:00Z",
        )


class Transport:
    def __init__(self, failure=False):
        self.calls = []
        self.failure = failure

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.failure:
            raise subprocess.TimeoutExpired(argv, 1, output="SyntheticSensitiveCloudLog")
        if "execute" in argv:
            (Path(kwargs["cwd"]) / argv[argv.index("-o") + 1]).write_text(
                json.dumps({"job_id": "job-1", "operation_id": "op-1"})
            )
        elif "get" in argv:
            (Path(kwargs["cwd"]) / argv[argv.index("-o") + 1]).write_text(
                json.dumps({"id": "job-1", "status": "SUCCESS", "name": "SyntheticSensitiveName"})
            )
        elif "download-files" in argv:
            dest = Path(argv[argv.index("--output-dir") + 1]) / "results"
            dest.mkdir()
            (dest / "training-status.json").write_text(
                json.dumps({"schema_version": 1, "status": "timeout", "exit_code": -9})
            )
        return subprocess.CompletedProcess(
            argv, 0, "SyntheticSensitiveCloudLog", "SyntheticSensitiveError"
        )


def test_confirm_async_receipt_guards_duplicate_and_paths_are_not_uploaded(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    transport = Transport()
    assert (
        module.confirm_bundle(
            bundle, plan["plan_sha256"], cli_path="datasphere", transport=transport
        )["status"]
        == "submitted"
    )
    argv, kwargs = transport.calls[0]
    assert argv[:4] == ["datasphere", "project", "job", "execute"]
    assert "--async" in argv and "-o" in argv
    assert kwargs["cwd"] == str(bundle)
    assert kwargs["capture_output"] is True
    assert "SyntheticSensitive" not in json.dumps(module.job_status(bundle, transport=transport))
    with pytest.raises(module.CloudLaunchError, match="submission_already_attempted"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert len([a for a, k in transport.calls if "execute" in a]) == 1


@pytest.mark.parametrize("mutation", ["crop", "plan", "extra", "symlink"])
def test_confirm_refuses_changed_bundle_before_transport(tmp_path, mutation):
    module, bundle, plan = prepare(tmp_path)
    if mutation == "crop":
        crop = next((bundle / "payload/data").rglob("*.nii.gz"))
        crop.chmod(0o600)
        crop.write_bytes(b"changed")
    elif mutation == "plan":
        p = bundle / "plan.private.json"
        raw = json.loads(p.read_text())
        raw["resource"] = "other"
        p.write_text(json.dumps(raw))
    elif mutation == "extra":
        (bundle / "payload").chmod(0o700)
        (bundle / "payload/private-account.json").write_text("secret")
    else:
        (bundle / "payload").chmod(0o700)
        (bundle / "payload/link").symlink_to(tmp_path)
    transport = Transport()
    with pytest.raises(module.CloudLaunchError, match="bundle_changed"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert not transport.calls


def test_ambiguous_submission_retains_lock_never_retries(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    transport = Transport(failure=True)
    with pytest.raises(module.CloudLaunchError, match="submission_ambiguous") as error:
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert "SyntheticSensitive" not in str(error.value)
    assert json.loads((bundle / "submission.private.json").read_text())["status"] == "ambiguous"
    with pytest.raises(module.CloudLaunchError, match="submission_already_attempted"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert len(transport.calls) == 1


def test_results_preserves_training_timeout_and_cancel_uses_receipt(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    transport = Transport()
    module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    destination = tmp_path / "download.private"
    result = module.download_results(bundle, destination, transport=transport)
    assert result["training_status"] == "timeout" and result["training_exit_code"] == -9
    assert module.cancel_job(bundle, transport=transport)["status"] == "cancel_requested"
    assert transport.calls[-1][0][-3:] == ["cancel", "--id", "job-1"]
    with pytest.raises(module.CloudLaunchError, match="destination_exists"):
        module.download_results(bundle, destination, transport=transport)


def test_drive_transport_refuses_collision_and_hides_url_errors(tmp_path):
    module = launcher()
    dest = tmp_path / "source.private"
    url = tmp_path / "link.private.txt"
    url.write_text("https://drive.google.com/drive/folders/SyntheticSensitiveToken")
    transport = Transport(failure=True)
    with pytest.raises(module.CloudLaunchError, match="source_download_failed"):
        module.download_source(url, dest, transport=transport)
    assert transport.calls[0][0][:3] == ["gdown", "--folder", "--output"]
    with pytest.raises(module.CloudLaunchError, match="destination_exists"):
        module.download_source(url, dest, transport=transport)


def test_reconcile_requires_explicit_match_and_never_launches(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    with pytest.raises(module.CloudLaunchError, match="submission_ambiguous"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=Transport(failure=True))
    transport = Transport()
    with pytest.raises(module.CloudLaunchError, match="reconciliation_acknowledgment_required"):
        module.reconcile_submission(bundle, "job-1", "op-1", transport=transport)
    assert not transport.calls
    result = module.reconcile_submission(
        bundle, "job-1", "op-1", acknowledge_match=True, transport=transport
    )
    assert result["status"] == "reconciled"
    assert len(transport.calls) == 1 and "get" in transport.calls[0][0]
    assert module.job_status(bundle, transport=transport)["provider_status"] == "SUCCESS"
    with pytest.raises(module.CloudLaunchError, match="submission_already_attempted"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)


def test_worker_actual_staged_cpu_training_and_private_results(tmp_path, monkeypatch):
    module, bundle, plan = prepare(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    command = [
        __import__("sys").executable,
        str(bundle / "payload/tools/datasphere_research.py"),
        "worker",
        "--config",
        str(bundle / "payload/research.private.json"),
        "--max-runtime-seconds",
        "60",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0
    assert json.loads(result.stdout)["training_status"] == "complete"
    outputs = bundle / "results"
    state = json.loads((outputs / "training-status.json").read_text())
    assert state == {"schema_version": 1, "status": "complete", "exit_code": 0}
    report = json.loads((outputs / "runs/research/report.json").read_text())
    assert report["status"] == "complete" and len(report["folds"]) == 2
    assert all(
        (outputs / "runs/research" / fold["artifacts"]["checkpoint"]["file"]).is_file()
        for fold in report["folds"]
    )
    assert "SyntheticSensitive" not in (outputs / "training.private.log").read_text()
    assert str(tmp_path) not in (outputs / "runs/research/report.json").read_text()
    assert module.downloaded_artifacts_complete(outputs) is True
    checkpoint = outputs / "runs/research" / report["folds"][0]["artifacts"]["checkpoint"]["file"]
    checkpoint.unlink()
    assert module.downloaded_artifacts_complete(outputs) is False


@pytest.mark.parametrize(
    "runner,status",
    [
        ("raise RuntimeError('SyntheticSensitive')", "failed"),
        ("import time; time.sleep(30)", "timeout"),
    ],
)
def test_worker_failure_timeout_returns_artifacts_without_success(tmp_path, runner, status):
    module = launcher()
    # A real subprocess exercises timeout/process cleanup; no cloud or classifier mock.
    root = tmp_path / "payload"
    (root / "tools").mkdir(parents=True)
    (root / "tools/run_research.py").write_text(runner)
    shutil = __import__("shutil")
    shutil.copyfile(Path(module.__file__), root / "tools/datasphere_research.py")
    config = root / "research.private.json"
    config.write_text("{}")
    result = subprocess.run(
        [
            __import__("sys").executable,
            str(root / "tools/datasphere_research.py"),
            "worker",
            "--config",
            str(config),
            "--max-runtime-seconds",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["training_status"] == status
    assert "SyntheticSensitive" not in result.stdout + result.stderr
    assert json.loads((tmp_path / "results/training-status.json").read_text())["exit_code"] != 0


def test_invalid_cli_is_private(tmp_path):
    module = launcher()
    result = subprocess.run(
        [__import__("sys").executable, module.__file__, "confirm", "--private=SyntheticSensitive"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["code"] == "invalid_arguments"
    assert "SyntheticSensitive" not in result.stdout + result.stderr


@pytest.mark.parametrize("alias_seed", [7, 23, 29])
def test_shared_patient_group_is_preserved_in_private_rekeyed_index(
    tmp_path, monkeypatch, alias_seed
):
    cfg = _canonical_synthetic_config(tmp_path)
    index = json.loads(Path(cfg.input_path).read_text())
    index["studies"][4]["patient_id"] = index["studies"][0]["patient_id"]
    index["labels"][4]["patient_id"] = index["labels"][0]["patient_id"]
    # Grouping is the contract under test; each group supports both classes.
    for row in index["labels"]:
        row["labels"]["sagittal"] = {"left": 1, "right": 2}
    Path(cfg.input_path).write_text(json.dumps(index))
    config = tmp_path / "research.private.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "binary",
                "input_path": cfg.input_path,
                "dataset_root": cfg.dataset_root,
                "output_dir": str(tmp_path / "outputs"),
                "n_splits": 2,
                "features": list(cfg.features),
                "fc_hidden": cfg.fc_hidden,
            }
        )
    )
    module = launcher()
    monkeypatch.setattr(module.os, "urandom", random.Random(alias_seed).randbytes)
    plan = module.prepare_cloud_bundle(
        config,
        tmp_path / "bundle",
        project_id="project",
        max_runtime_seconds=10,
        hourly_price="1",
        currency="RUB",
        price_as_of="2026-10-03T00:00:00Z",
    )
    staged = json.loads((tmp_path / "bundle/payload/index.private.json").read_text())
    assert plan["patient_count"] == 7
    from collections import Counter

    assert sorted(
        Counter((r["source_id"], r["patient_id"]) for r in staged["studies"]).values()
    ) == [1, 1, 1, 1, 1, 1, 2]


@pytest.mark.parametrize("case", ["nonzero", "missing_receipt", "malformed_receipt"])
def test_submission_failure_is_ambiguous_and_output_is_never_echoed(tmp_path, case):
    module, bundle, plan = prepare(tmp_path)
    calls = []

    def transport(argv, **kwargs):
        calls.append(argv)
        if case == "malformed_receipt":
            (bundle / "execution.private.json").write_text('{"job_id":"SyntheticSensitive/path"}')
        return subprocess.CompletedProcess(
            argv, 7 if case == "nonzero" else 0, "SyntheticSensitive", "SyntheticSensitive"
        )

    with pytest.raises(module.CloudLaunchError, match="submission_ambiguous") as error:
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert "SyntheticSensitive" not in str(error.value)
    with pytest.raises(module.CloudLaunchError, match="submission_already_attempted"):
        module.confirm_bundle(bundle, plan["plan_sha256"], transport=transport)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"status": "SyntheticSensitive", "exit_code": 0},
        {"status": "complete", "exit_code": True},
    ],
)
def test_malformed_training_status_has_fixed_safe_error(tmp_path, payload):
    module, bundle, plan = prepare(tmp_path)
    module.confirm_bundle(bundle, plan["plan_sha256"], transport=Transport())

    def transport(argv, **kwargs):
        dest = Path(argv[argv.index("--output-dir") + 1]) / "results"
        dest.mkdir()
        (dest / "training-status.json").write_text(json.dumps(payload))
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(module.CloudLaunchError, match="invalid_training_status"):
        module.download_results(bundle, tmp_path / "download.private", transport=transport)


def test_no_outputs_is_not_complete_training(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    module.confirm_bundle(bundle, plan["plan_sha256"], transport=Transport())

    def transport(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    assert (
        module.download_results(bundle, tmp_path / "empty.private", transport=transport)[
            "training_status"
        ]
        == "no_result"
    )


def test_drive_file_link_is_refused_before_transport(tmp_path):
    module = launcher()
    link = tmp_path / "link.private.txt"
    link.write_text("https://drive.google.com/file/d/SyntheticSensitive/view")
    transport = Transport()
    with pytest.raises(module.CloudLaunchError, match="source_download_failed"):
        module.download_source(link, tmp_path / "download.private", transport=transport)
    assert not transport.calls


def test_prepare_cli_summary_is_reviewable_without_private_account(tmp_path):
    module, bundle, plan = prepare(tmp_path)
    result = subprocess.run(
        [
            __import__("sys").executable,
            module.__file__,
            "prepare",
            "--config",
            str(tmp_path / "SyntheticSensitiveConfig.private.json"),
            "--bundle",
            str(tmp_path / "cli-bundle"),
            "--project-id",
            "SyntheticSensitiveProject",
            "--max-runtime-seconds",
            "120",
            "--hourly-price",
            "10.5",
            "--currency",
            "RUB",
            "--price-as-of",
            "2026-10-03T10:00:00+03:00",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["resource"] == "gt4.1" and summary["training_runtime_seconds"] == 120
    assert summary["currency"] == "RUB" and summary["hourly_price"] == 10.5
    assert summary["training_window_compute_estimate"] == pytest.approx(0.35)
    assert "logs_results_storage" in summary["excluded_costs"]
    assert "SyntheticSensitive" not in result.stdout + result.stderr
    assert "project_id" not in summary and "manifest" not in summary


def test_named_profile_is_private_and_used_in_every_official_command(tmp_path):
    module, original, first_plan = prepare(tmp_path)
    bundle = tmp_path / "profile-bundle"
    plan = module.prepare_cloud_bundle(
        tmp_path / "SyntheticSensitiveConfig.private.json",
        bundle,
        project_id="project",
        max_runtime_seconds=120,
        hourly_price="10.5",
        currency="RUB",
        price_as_of="2026-10-03T10:00:00+03:00",
        profile="SyntheticSensitiveProfile",
    )
    assert plan["profile"] == "SyntheticSensitiveProfile"
    assert "SyntheticSensitiveProfile" not in (bundle / "job.yaml").read_text()
    assert not any(
        "SyntheticSensitiveProfile" in p.read_text() for p in (bundle / "payload").rglob("*.json")
    )
    transport = Transport()
    module.confirm_bundle(bundle, plan["plan_sha256"], cli_path="cli", transport=transport)
    module.job_status(bundle, cli_path="cli", transport=transport)
    module.download_results(
        bundle, tmp_path / "profile-results", cli_path="cli", transport=transport
    )
    module.cancel_job(bundle, cli_path="cli", transport=transport)
    assert all(
        argv[:3] == ["cli", "--profile", "SyntheticSensitiveProfile"]
        for argv, kwargs in transport.calls
    )


def test_cloud_auto_device_requires_cuda_instead_of_silent_cpu_fallback(tmp_path):
    module, _, _ = prepare(tmp_path)
    config = tmp_path / "SyntheticSensitiveConfig.private.json"
    raw = json.loads(config.read_text())
    raw["device"] = None
    config.write_text(json.dumps(raw))
    bundle = tmp_path / "auto-device-bundle"
    module.prepare_cloud_bundle(
        config,
        bundle,
        project_id="synthetic-project",
        max_runtime_seconds=120,
        hourly_price="10.5",
        currency="RUB",
        price_as_of="2026-10-03T10:00:00+03:00",
    )
    remote = json.loads((bundle / "payload/research.private.json").read_text())
    assert remote["device"] == "cuda"


def test_cloud_refuses_mac_only_device_before_staging(tmp_path):
    module, _, _ = prepare(tmp_path)
    config = tmp_path / "SyntheticSensitiveConfig.private.json"
    raw = json.loads(config.read_text())
    raw["device"] = "mps"
    config.write_text(json.dumps(raw))
    bundle = tmp_path / "mps-bundle"
    with pytest.raises(module.CloudLaunchError, match="invalid_prepare_inputs"):
        module.prepare_cloud_bundle(
            config,
            bundle,
            project_id="synthetic-project",
            max_runtime_seconds=120,
            hourly_price="10.5",
            currency="RUB",
            price_as_of="2026-10-03T10:00:00+03:00",
        )
    assert not bundle.exists()
