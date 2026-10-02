#!/usr/bin/env python3
"""Private, reviewed DataSphere Jobs bundles; official CLI transport only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_FILES = (
    "tools/run_research.py",
    "tools/datasphere_research.py",
    "training/sagittal_binary_cv.py",
    "training/tmj_position_label_table.py",
    "training/roi_provenance.py",
    "training/datasets/tmj_position_dataset.py",
    "training/losses/focal_loss.py",
    "training/utils/binary_metrics.py",
    "training/utils/datasphere_env.py",
    "training/utils/seed.py",
    "training/utils/volume_aug_3d.py",
    "models/__init__.py",
    "models/blocks.py",
    "models/tmj_binary_position_classifier.py",
)
MAX_UPLOAD_BYTES = 5 * 1024**3  # Official per-input limit; payload is one directory input.


class CloudLaunchError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        with Path(path).open() as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError
        return json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, ValueError, TypeError, RecursionError):
        raise CloudLaunchError("invalid_private_file") from None


def _write(path, value, *, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    with os.fdopen(os.open(path, flags, 0o600), "w") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(payload):
    result = {}
    if payload.is_symlink() or not payload.is_dir():
        raise CloudLaunchError("bundle_changed")
    for path in sorted(payload.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise CloudLaunchError("bundle_changed")
        if path.is_file():
            result[str(path.relative_to(payload))] = {
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
    return result


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise CloudLaunchError("invalid_launch_options")
    return value


def prepare_cloud_bundle(
    config_path,
    bundle_dir,
    *,
    project_id,
    max_runtime_seconds,
    hourly_price,
    currency,
    price_as_of,
    resource="gt4.1",
    profile=None,
):
    """No external IO. Plan and reverse mapping are local-only private files."""
    from tools.run_research import CV_CONFIG_FIELDS, load_research_config, preflight_research
    from training.roi_provenance import passport_path, study_key
    from training.tmj_position_label_table import build_canonical_index

    bundle = Path(bundle_dir).expanduser().resolve()
    if bundle.exists():
        raise CloudLaunchError("bundle_exists")
    try:
        _identifier(project_id)
        if profile is not None and (
            not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", profile)
        ):
            raise ValueError
        if not isinstance(resource, str) or not re.fullmatch(
            r"[a-z][a-z0-9]*\.[1-9][0-9]*", resource
        ):
            raise ValueError
        if type(max_runtime_seconds) is not int or not 1 <= max_runtime_seconds <= 7 * 24 * 3600:
            raise ValueError
        if isinstance(hourly_price, bool):
            raise ValueError
        price = float(hourly_price)
        if not math.isfinite(price) or price <= 0 or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError
        if datetime.fromisoformat(price_as_of.replace("Z", "+00:00")).utcoffset() is None:
            raise ValueError
        config = load_research_config(config_path)
        if config.cv.device and config.cv.device.split(":", 1)[0] == "mps":
            raise ValueError
        summary = preflight_research(config)
        records = build_canonical_index(
            config.cv.input_path, config.cv.dataset_root, sagittal_only=True
        )
        raw = _read(config_path)
        dataset = Path(config.cv.dataset_root)
        if (
            bundle.is_relative_to(dataset)
            or dataset.is_relative_to(bundle)
            or Path(config.cv.input_path).is_relative_to(bundle)
        ):
            raise ValueError
    except Exception:
        raise CloudLaunchError("invalid_prepare_inputs") from None
    bundle.mkdir(mode=0o700, parents=True)
    payload = bundle / "payload"
    payload.mkdir(mode=0o700)
    try:
        # Random plan-local tokens preserve source-qualified patient and label relationships.
        mappings = {name: {} for name in ("source", "patient", "study", "label")}

        def token(kind, original):
            key = _json(original)
            return mappings[kind].setdefault(key, kind[0] + os.urandom(16).hex())

        studies, labels = [], {}
        for record in records:
            source = token("source", record["source_id"])
            patient = token("patient", [record["source_id"], record["patient_id"]])
            study = token("study", [record["source_id"], record["study_id"]])
            label = token("label", [record["source_id"], record["label_record_id"]])
            shared = {"source_id": source, "patient_id": patient, "label_record_id": label}
            row = {**shared, "study_id": study, "label_applicability": "confirmed", "crops": {}}
            for side in ("left", "right"):
                relative = f"{study}/{side}.nii.gz"
                target = payload / "data" / relative
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(record["crop_paths"][side], target)
                passport = _read(passport_path(record["crop_paths"][side]))
                passport["study_key"] = study_key(row)
                common = {
                    key: passport[key]
                    for key in ("study_key", "source", "detectors", "preprocessing")
                }
                passport["input_sha256"] = _digest(common)
                _write(passport_path(target), passport)
                row["crops"][side] = relative
            planes = {"sagittal": {side: record[f"sag_{side}"] + 1 for side in ("left", "right")}}
            if "fr_left" in record:
                planes["frontal"] = {side: record[f"fr_{side}"] + 4 for side in ("left", "right")}
            labels[(source, label)] = {**shared, "labels": planes}
            studies.append(row)
        _write(
            payload / "index.private.json",
            {"schema_version": 1, "studies": studies, "labels": list(labels.values())},
        )
        remote = {key: raw[key] for key in CV_CONFIG_FIELDS if key in raw}
        remote.update(
            schema_version=1,
            mode=config.mode,
            input_path="index.private.json",
            dataset_root="data",
            output_dir="../results/runs",
            run_id="research",
        )
        if remote.get("device") is None:
            remote["device"] = "cuda"
        remote.update(tqdm_disable=True, log_each_epoch=False)
        _write(payload / "research.private.json", remote)
        root = Path(__file__).resolve().parents[1]
        for relative in SOURCE_FILES:
            target = payload / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
        shutil.copyfile(
            root / "tools/research_cloud_requirements.txt", payload / "requirements.txt"
        )
        # JSON is a strict YAML subset accepted by the official parser.
        job = {
            "name": "tmj-research",
            "cmd": f"python3 payload/tools/datasphere_research.py worker --config payload/research.private.json --max-runtime-seconds {max_runtime_seconds}",
            "inputs": ["payload"],
            "outputs": ["results"],
            "cloud-instance-types": [resource],
            "env": {
                "python": {
                    "type": "manual",
                    "version": "3.12",
                    "requirements-file": "payload/requirements.txt",
                    "local-paths": [],
                }
            },
        }
        _write(bundle / "job.yaml", job)
        # Staged passports/index must pass the real consumer too, after rekey/copy.
        preflight_research(load_research_config(payload / "research.private.json"))
        manifest = _manifest(payload)
        size = sum(row["bytes"] for row in manifest.values())
        if size > MAX_UPLOAD_BYTES:
            raise CloudLaunchError("upload_size_limit")
        plan = {
            "schema_version": 1,
            "project_id": project_id,
            "profile": profile,
            "resource": resource,
            "python": "3.12",
            "study_count": summary["study_count"],
            "sample_count": summary["sample_count"],
            "patient_count": summary["patient_count"],
            "upload_bytes": size,
            "training_runtime_seconds": max_runtime_seconds,
            "hourly_price": price,
            "currency": currency,
            "price_as_of": price_as_of,
            "price_version": "operator-reviewed-quote",
            "training_window_compute_estimate": price * max_runtime_seconds / 3600,
            "money_cap_guaranteed": False,
            "excluded_costs": [
                "setup_compute",
                "input_cache_storage",
                "logs_results_storage",
                "egress",
            ],
            "provider_default_retention_days": 14,
            "pixels_remain_private": True,
            "detector_training_identity": "unknown",
            "source_rechecked_study_count": summary["source_rechecked_study_count"],
            "manifest": manifest,
            "job_sha256": _hash_file(bundle / "job.yaml"),
        }
        plan["plan_sha256"] = _digest(plan)
        _write(bundle / "plan.private.json", plan)
        _write(bundle / "mapping.private.json", mappings)
        for path in payload.rglob("*"):
            path.chmod(0o500 if path.is_dir() else 0o400)
        payload.chmod(0o500)
        return plan
    except CloudLaunchError:
        raise
    except Exception:
        raise CloudLaunchError("bundle_prepare_failed") from None


def _check_bundle(bundle, expected_digest):
    try:
        plan = _read(bundle / "plan.private.json")
        digest = plan.pop("plan_sha256")
        if (
            digest != expected_digest
            or _digest(plan) != digest
            or plan["manifest"] != _manifest(bundle / "payload")
            or plan["job_sha256"] != _hash_file(bundle / "job.yaml")
        ):
            raise ValueError
        plan["plan_sha256"] = digest
        return plan
    except Exception:
        raise CloudLaunchError("bundle_changed") from None


def _invoke(argv, *, cwd, transport=subprocess.run):
    # CLI stdout/stderr can contain account names, private paths and URLs. Never echo.
    try:
        result = transport(argv, cwd=str(cwd), capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise ValueError
    except Exception:
        raise CloudLaunchError("cloud_command_failed") from None


def _cli_prefix(cli_path, bundle):
    plan = _read(bundle / "plan.private.json")
    profile = plan.get("profile")
    if profile is None:
        return [str(cli_path)]
    if not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", profile):
        raise CloudLaunchError("invalid_launch_options")
    return [str(cli_path), "--profile", profile]


def confirm_bundle(bundle_dir, expected_digest, *, cli_path="datasphere", transport=subprocess.run):
    bundle = Path(bundle_dir).resolve()
    if (bundle / "submission.private.json").exists():
        raise CloudLaunchError("submission_already_attempted")
    plan = _check_bundle(bundle, expected_digest)
    receipt = {"schema_version": 1, "plan_sha256": expected_digest, "status": "submission_started"}
    try:
        _write(bundle / "submission.private.json", receipt, exclusive=True)
    except FileExistsError:
        raise CloudLaunchError("submission_already_attempted") from None
    try:
        _invoke(
            _cli_prefix(cli_path, bundle)
            + [
                "project",
                "job",
                "execute",
                "-p",
                plan["project_id"],
                "-c",
                "job.yaml",
                "--async",
                "-o",
                "execution.private.json",
            ],
            cwd=bundle,
            transport=transport,
        )
        raw = _read(bundle / "execution.private.json")
        receipt.update(
            status="submitted",
            job_id=_identifier(raw["job_id"]),
            operation_id=_identifier(raw["operation_id"]),
        )
        _write(bundle / "submission.private.json", receipt)
        return {"schema_version": 1, "status": "submitted", "plan_sha256": expected_digest}
    except (Exception, KeyboardInterrupt):
        receipt["status"] = "ambiguous"
        _write(bundle / "submission.private.json", receipt)
        raise CloudLaunchError("submission_ambiguous") from None


def reconcile_submission(
    bundle_dir,
    job_id,
    operation_id,
    *,
    acknowledge_match=False,
    cli_path="datasphere",
    transport=subprocess.run,
):
    """Operator attests the job matches this plan after inspecting official job list.

    This reads the existing job; it never resubmits or clears the submission lock.
    """
    if acknowledge_match is not True:
        raise CloudLaunchError("reconciliation_acknowledgment_required")
    bundle = Path(bundle_dir).resolve()
    receipt = _read(bundle / "submission.private.json")
    if receipt.get("status") not in ("submission_started", "ambiguous"):
        raise CloudLaunchError("submission_not_ambiguous")
    _check_bundle(bundle, receipt.get("plan_sha256"))
    job_id, operation_id = _identifier(job_id), _identifier(operation_id)
    _invoke(
        _cli_prefix(cli_path, bundle)
        + [
            "project",
            "job",
            "get",
            "--id",
            job_id,
            "--format",
            "json",
            "-o",
            "reconciliation.private.json",
        ],
        cwd=bundle,
        transport=transport,
    )
    raw = _read(bundle / "reconciliation.private.json")
    if not isinstance(raw, dict) or raw.get("id") != job_id:
        raise CloudLaunchError("invalid_provider_response")
    receipt.update(status="reconciled", job_id=job_id, operation_id=operation_id)
    _write(bundle / "submission.private.json", receipt)
    return {"schema_version": 1, "status": "reconciled"}


def _job(bundle):
    receipt = _read(bundle / "submission.private.json")
    if receipt.get("status") not in ("submitted", "reconciled"):
        raise CloudLaunchError("submission_requires_reconciliation")
    return _identifier(receipt.get("job_id"))


def job_status(bundle_dir, *, cli_path="datasphere", transport=subprocess.run):
    bundle = Path(bundle_dir).resolve()
    job = _job(bundle)
    _invoke(
        _cli_prefix(cli_path, bundle)
        + [
            "project",
            "job",
            "get",
            "--id",
            job,
            "--format",
            "json",
            "-o",
            "provider-status.private.json",
        ],
        cwd=bundle,
        transport=transport,
    )
    raw = _read(bundle / "provider-status.private.json")
    if not isinstance(raw, dict) or raw.get("id") != job:
        raise CloudLaunchError("invalid_provider_response")
    allowed = {
        "JOB_STATUS_UNSPECIFIED",
        "CREATING",
        "PREPARING",
        "EXECUTING",
        "UPLOADING_OUTPUT",
        "ERROR",
        "CANCELLED",
        "CANCELLING",
        "SUCCESS",
    }
    state = raw.get("status")
    return {
        "schema_version": 1,
        "provider_status": state if isinstance(state, str) and state in allowed else "unknown",
        "training_status": "not_downloaded",
    }


def cancel_job(bundle_dir, *, cli_path="datasphere", transport=subprocess.run):
    bundle = Path(bundle_dir).resolve()
    _invoke(
        _cli_prefix(cli_path, bundle) + ["project", "job", "cancel", "--id", _job(bundle)],
        cwd=bundle,
        transport=transport,
    )
    return {"schema_version": 1, "status": "cancel_requested"}


def downloaded_artifacts_complete(results_dir):
    """Check the full declared fold artifacts without exposing individual rows."""
    directory = Path(results_dir) / "runs/research"
    try:
        report = _read(directory / "report.json")
        count = report["n_splits"]
        folds = report["folds"]
        if (
            report["status"] != "complete"
            or type(count) is not int
            or count < 2
            or report["completed_folds"] != count
            or len(folds) != count
            or sorted(fold["fold"] for fold in folds) != list(range(count))
        ):
            return False
        refs = [report["artifacts"]["replay"]]
        for fold in folds:
            refs.extend(fold["artifacts"][key] for key in ("checkpoint", "predictions"))
        for ref in refs:
            name = ref["file"]
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or name in (".", "..")
            ):
                return False
            path = directory / name
            if (
                path.is_symlink()
                or not path.is_file()
                or path.resolve().parent != directory.resolve()
                or _hash_file(path) != ref["sha256"]
            ):
                return False
        return True
    except (CloudLaunchError, OSError, ValueError, TypeError, KeyError):
        return False


def download_results(bundle_dir, destination, *, cli_path="datasphere", transport=subprocess.run):
    bundle = Path(bundle_dir).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise CloudLaunchError("destination_exists")
    if destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
        raise CloudLaunchError("destination_overlap")
    job = _job(bundle)
    destination.mkdir(mode=0o700, parents=True)
    _invoke(
        _cli_prefix(cli_path, bundle)
        + [
            "project",
            "job",
            "download-files",
            "--id",
            job,
            "--output-dir",
            str(destination),
            "--with-logs",
        ],
        cwd=bundle,
        transport=transport,
    )
    path = destination / "results/training-status.json"
    if not path.is_file():
        return {"schema_version": 1, "training_status": "no_result", "training_exit_code": None}
    raw = _read(path)
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
        or raw.get("status") not in ("complete", "failed", "timeout")
        or type(raw.get("exit_code")) is not int
        or (raw["status"] == "complete" and raw["exit_code"] != 0)
    ):
        raise CloudLaunchError("invalid_training_status")
    complete = raw["status"] == "complete" and downloaded_artifacts_complete(
        destination / "results"
    )
    return {
        "schema_version": 1,
        "training_status": raw["status"],
        "training_exit_code": raw["exit_code"],
        "results_ready": complete,
        "artifacts_status": "complete" if complete else "partial",
    }


def download_source(url_file, destination, *, gdown_path="gdown", transport=subprocess.run):
    destination = Path(destination).resolve()
    if destination.exists():
        raise CloudLaunchError("destination_exists")
    try:
        url = Path(url_file).read_text().strip()
        if not re.fullmatch(
            r"https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+(?:\?[^\s]*)?", url
        ):
            raise ValueError
        destination.mkdir(mode=0o700, parents=True)
        _invoke(
            [str(gdown_path), "--folder", "--output", str(destination), url],
            cwd=destination.parent,
            transport=transport,
        )
        return {"schema_version": 1, "status": "source_downloaded", "training_ready": False}
    except Exception:
        raise CloudLaunchError("source_download_failed") from None


def worker(config_path, max_runtime_seconds):
    """Provider success only means wrapper returned; consult training-status.json."""
    if type(max_runtime_seconds) is not int or not 1 <= max_runtime_seconds <= 7 * 24 * 3600:
        raise CloudLaunchError("invalid_launch_options")
    config_path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[1]
    results = config_path.parent.parent / "results"
    results.mkdir(mode=0o700, exist_ok=True)
    status, exit_code = "failed", 2
    with (results / "training.private.log").open("x") as log:
        try:
            process = subprocess.Popen(
                [sys.executable, str(root / "tools/run_research.py"), "--config", str(config_path)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=max_runtime_seconds)
                status = "complete" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait()
                status = "timeout"
        except (OSError, ValueError):
            pass
    _write(
        results / "training-status.json",
        {"schema_version": 1, "status": status, "exit_code": exit_code},
    )
    return {"schema_version": 1, "training_status": status, "training_exit_code": exit_code}


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        print(_json({"schema_version": 1, "code": "invalid_arguments"}))
        self.exit(2)


def main():
    parser = _SafeParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_SafeParser)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--project-id", required=True)
    prepare.add_argument(
        "--profile", help="Private official CLI profile; omitted uses native default"
    )
    prepare.add_argument("--resource", default="gt4.1")
    prepare.add_argument("--max-runtime-seconds", required=True, type=int)
    prepare.add_argument("--hourly-price", required=True)
    prepare.add_argument("--currency", required=True)
    prepare.add_argument("--price-as-of", required=True)
    for command in ("confirm", "status", "results", "cancel", "reconcile"):
        sub = commands.add_parser(command)
        sub.add_argument("--bundle", required=True)
        sub.add_argument("--cli", default="datasphere")
        if command == "confirm":
            sub.add_argument("--plan-sha256", required=True)
        if command == "reconcile":
            sub.add_argument("--job-id", required=True)
            sub.add_argument("--operation-id", required=True)
            sub.add_argument("--acknowledge-match", action="store_true")
        if command == "results":
            sub.add_argument("--destination", required=True)
    download = commands.add_parser("download-source")
    download.add_argument("--url-file", required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--gdown", default="gdown")
    runner = commands.add_parser("worker")
    runner.add_argument("--config", required=True)
    runner.add_argument("--max-runtime-seconds", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_cloud_bundle(
                args.config,
                args.bundle,
                project_id=args.project_id,
                profile=args.profile,
                resource=args.resource,
                max_runtime_seconds=args.max_runtime_seconds,
                hourly_price=args.hourly_price,
                currency=args.currency,
                price_as_of=args.price_as_of,
            )
            result = {
                key: result[key]
                for key in (
                    "schema_version",
                    "plan_sha256",
                    "study_count",
                    "patient_count",
                    "upload_bytes",
                    "money_cap_guaranteed",
                    "resource",
                    "python",
                    "training_runtime_seconds",
                    "hourly_price",
                    "currency",
                    "price_as_of",
                    "training_window_compute_estimate",
                    "excluded_costs",
                    "provider_default_retention_days",
                    "pixels_remain_private",
                )
            }
        elif args.command == "confirm":
            result = confirm_bundle(args.bundle, args.plan_sha256, cli_path=args.cli)
        elif args.command == "reconcile":
            result = reconcile_submission(
                args.bundle,
                args.job_id,
                args.operation_id,
                acknowledge_match=args.acknowledge_match,
                cli_path=args.cli,
            )
        elif args.command == "status":
            result = job_status(args.bundle, cli_path=args.cli)
        elif args.command == "cancel":
            result = cancel_job(args.bundle, cli_path=args.cli)
        elif args.command == "results":
            result = download_results(args.bundle, args.destination, cli_path=args.cli)
        elif args.command == "download-source":
            result = download_source(args.url_file, args.destination, gdown_path=args.gdown)
        else:
            result = worker(args.config, args.max_runtime_seconds)
        print(_json(result))
        return 0
    except CloudLaunchError as error:
        print(_json({"schema_version": 1, "code": error.code}))
        return 2
    except Exception:
        print(_json({"schema_version": 1, "code": "launcher_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
