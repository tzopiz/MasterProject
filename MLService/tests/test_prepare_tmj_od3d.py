import io
import json
import re
import tarfile

import pytest

from tools import prepare_tmj_od3d as tool
from training.tmj_od3d_images import PreparationError, digest_file

from .test_tmj_od3d_images import source


def archive_fixture(tmp_path, monkeypatch):
    roots = []
    for i in range(2):
        base = tmp_path / f"patient{i}"
        base.mkdir()
        folder, _ = source(base)
        renamed = folder.with_name(f"synthetic-{i}")
        folder.rename(renamed)
        roots.append(renamed)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for root in roots:
            archive.add(root, arcname="tmj/" + root.name)
    raw = stream.getvalue()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "anonymous_id,patient_sex,age_years,age_group,selected_slice_min,selected_slice_max,selected_slice_count,label_L,label_R\n"
        + "".join(f"{p.name},unknown,20,adult,10,30,3,0,1|5\n" for p in roots)
    )
    monkeypatch.setattr(tool, "METADATA_SHA256", digest_file(metadata))
    monkeypatch.setattr(tool, "ARCHIVE_BYTES", len(raw))
    calls = []

    class Response(io.BytesIO):
        status = 206

    def urlopen(request, timeout):
        lo, hi = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        calls.append((lo, hi))
        result = Response(raw[lo : hi + 1])
        result.headers = {
            "Content-Range": f"bytes {lo}-{hi}/{len(raw)}",
            "Content-Length": str(hi - lo + 1),
        }
        return result

    monkeypatch.setattr(tool.urllib.request, "urlopen", urlopen)
    return metadata, calls


def test_resume_preserves_first_patient_hash_and_reaches_real_eof(tmp_path, monkeypatch):
    metadata, calls = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    first = tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    assert not first["complete"] and len(first["records"]) == 2 and first["next_offset"] > 0
    before = [r["crop_sha256"] for r in first["records"]]
    second = tool.prepare_archive(out, metadata, max_patients=2, slice_count=4, image_size=16)
    assert second["complete"] and len(second["records"]) == 4 and second["failures"] == {}
    assert [r["crop_sha256"] for r in second["records"][:2]] == before
    assert calls[0][0] == 0 and calls[1][0] == first["next_offset"]
    assert not (out / "raw-cache").exists() and not (out / "preparation.lock").exists()
    assert json.loads((out / "index.private.json").read_text())["complete"]


def test_resume_refuses_changed_prepared_pixels(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    first = tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    (out / first["records"][0]["crop_path"]).write_bytes(b"changed")
    with pytest.raises(PreparationError, match="prepared_artifact_changed"):
        tool.prepare_archive(out, metadata, slice_count=4, image_size=16)


def test_lock_and_metadata_mismatch_refuse_without_network(tmp_path, monkeypatch):
    metadata, calls = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "preparation.lock").write_text("")
    with pytest.raises(PreparationError, match="preparation_already_locked"):
        tool.prepare_archive(out, metadata)
    assert calls == []
    (out / "preparation.lock").unlink()
    metadata.write_text("corrupted")
    with pytest.raises(PreparationError, match="source_metadata_changed"):
        tool.prepare_archive(out, metadata)
    assert calls == []


def range_source(raw, damaged=None, transient=False):
    import threading

    guard = threading.Lock()
    calls = []
    closed = []
    attempts = {}

    class Response(io.BytesIO):
        status = 206

        def close(self):
            closed.append(True)
            super().close()

    def opener(request, timeout):
        lo, hi = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        with guard:
            calls.append((lo, hi))
            attempts[lo] = attempts.get(lo, 0) + 1
        if transient and attempts[lo] == 1:
            raise TimeoutError("private transport diagnostic")
        response = Response(raw[lo : hi + 1] if damaged != "truncated" else raw[lo:hi])
        response.headers = {
            "Content-Range": f"bytes {lo}-{hi}/{len(raw)}",
            "Content-Length": str(hi - lo + 1),
        }
        if damaged == "range":
            response.headers["Content-Range"] = "private invalid range"
        if damaged == "length":
            response.headers["Content-Length"] = str(hi - lo)
        return response

    return opener, calls, closed


def test_prefetch_reader_delivers_ordered_bytes_and_honors_requested_budget():
    raw = bytes(range(100))
    opener, calls, closed = range_source(raw)
    assert hasattr(tool, "RangeReader"), "bounded range reader is missing"
    with tool.RangeReader(
        "https://example.test/data", 5, 28, len(raw), chunk_size=8, opener=opener
    ) as reader:
        parts = []
        while block := reader.read(3):
            parts.append(block)
    assert b"".join(parts) == raw[5:29]
    assert sorted(calls) == [(5, 12), (13, 20), (21, 28)]
    assert len(closed) == 3


@pytest.mark.parametrize(
    "damaged,code",
    [
        ("range", "source_range_not_confirmed"),
        ("length", "source_range_not_confirmed"),
        ("truncated", "truncated_range_response"),
    ],
)
def test_prefetch_rejects_invalid_ranges_or_truncated_body_with_safe_errors(damaged, code):
    opener, calls, closed = range_source(b"abcdefgh", damaged)
    with pytest.raises(PreparationError, match=f"^{code}$"):
        with tool.RangeReader(
            "https://example.test/private-url", 0, 7, 8, chunk_size=8, opener=opener
        ) as reader:
            reader.read(8)
    assert calls == [(0, 7)] * 3 and len(closed) == 3


def test_prefetch_retry_and_context_exit_release_resources():
    opener, calls, closed = range_source(bytes(range(48)), transient=True)
    with tool.RangeReader(
        "https://example.test/data", 0, 47, 48, chunk_size=8, opener=opener
    ) as reader:
        assert reader.read(8) == bytes(range(8))
    assert len(calls) <= 10  # One consumed buffer plus four queued chunks, two attempts each.
    assert len(closed) == sum(
        count > 1 for count in __import__("collections").Counter(lo for lo, _ in calls).values()
    )


@pytest.mark.parametrize(
    "name,kind",
    [
        ("tmj/../escape.dcm", "file"),
        ("tmj/patient/escape.dcm", "symlink"),
        ("tmj/patient/escape.dcm", "hardlink"),
        ("wrong/patient/escape.dcm", "file"),
        ("tmj/patient/subdir", "dir"),
        ("tmj/patient/escape.dcm", "fifo"),
    ],
)
def test_unsafe_archive_layout_and_member_types_rejected(tmp_path, monkeypatch, name, kind):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.size = 0
        member.type = {
            "file": tarfile.REGTYPE,
            "dir": tarfile.DIRTYPE,
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "fifo": tarfile.FIFOTYPE,
        }[kind]
        member.linkname = "private-link-target"
        archive.addfile(member)
    raw = stream.getvalue()
    monkeypatch.setattr(tool, "ARCHIVE_BYTES", len(raw))
    opener, _, _ = range_source(raw)
    monkeypatch.setattr(tool.urllib.request, "urlopen", opener)
    out = tmp_path / "out"
    with pytest.raises(PreparationError, match="unsafe_archive_member|unsupported_archive_layout"):
        tool.prepare_archive(out, metadata)
    assert not (out / "preparation.lock").exists() and not (out / "raw-cache").exists()


def test_duplicate_directory_members_are_rejected(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for _ in range(2):
            info = tarfile.TarInfo("tmj/synthetic-0")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
    raw = stream.getvalue()
    monkeypatch.setattr(tool, "ARCHIVE_BYTES", len(raw))
    opener, _, _ = range_source(raw)
    monkeypatch.setattr(tool.urllib.request, "urlopen", opener)
    with pytest.raises(PreparationError, match="duplicate_archive_file"):
        tool.prepare_archive(tmp_path / "out", metadata)


def test_prefetch_close_at_patient_limit_never_fetches_past_budget(tmp_path, monkeypatch):
    metadata, calls = archive_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, "RANGE_CHUNK_BYTES", 1024)
    out = tmp_path / "out"
    state = tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    assert len(state["processed_patients"]) == 1
    assert all(hi < tool.ARCHIVE_BYTES for _, hi in calls)
    assert not (out / "raw-cache").exists() and not (out / "preparation.lock").exists()


def test_download_budget_exhaustion_keeps_last_committed_offset(tmp_path, monkeypatch):
    metadata, calls = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(PreparationError, match="download_budget_exhausted|unexpected end of data"):
        tool.prepare_archive(out, metadata, max_download_bytes=1024, slice_count=4, image_size=16)
    assert all(hi < 1024 for _, hi in calls)
    assert not (out / "preparation.lock").exists() and not (out / "raw-cache").exists()


def test_explicit_stale_lock_reconciliation_refuses_live_and_legacy_locks(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    lock = out / "preparation.lock"
    lock.write_text(json.dumps({"pid": __import__("os").getpid()}))
    with pytest.raises(PreparationError, match="preparation_already_locked"):
        tool.reconcile_stale_lock(out)
    assert lock.exists()
    lock.write_text("")
    with pytest.raises(PreparationError, match="unverifiable_preparation_lock"):
        tool.reconcile_stale_lock(out)
    lock.write_text(json.dumps({"pid": 43210}))

    def absent(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(tool.os, "kill", absent)
    tool.reconcile_stale_lock(out)
    assert not lock.exists()


def test_out_of_order_workers_preserve_tar_byte_order_and_read_memory_bound():
    import threading

    raw = bytes(range(64))
    release = threading.Event()
    calls = []
    opener, _, closed = range_source(raw)

    def delayed(request, timeout):
        lo = int(request.headers["Range"].split("=")[1].split("-")[0])
        calls.append(lo)
        if lo == 0:
            assert release.wait(2)
        if lo == 8:
            release.set()
        return opener(request, timeout)

    with tool.RangeReader(
        "https://example.test/data", 0, 63, 64, chunk_size=8, opener=delayed
    ) as reader:
        first = reader.read(10**9)
        assert len(first) <= 8
        parts = [first]
        while block := reader.read(7):
            assert len(reader.pending) <= 4
            parts.append(block)
    assert b"".join(parts) == raw and len(closed) == 8


def test_hard_crash_orphan_is_flagged_and_explicitly_quarantined(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    first = tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    orphan = out / "data" / ("a" * 64 + "-L.npz")
    orphan.write_bytes(b"partial private pixels")
    with pytest.raises(PreparationError, match="prepared_orphan_artifact"):
        tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    lock = out / "preparation.lock"
    lock.write_text(json.dumps({"pid": 43210}))

    def absent(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(tool.os, "kill", absent)
    tool.reconcile_stale_lock(out)
    assert not orphan.exists() and not lock.exists()
    quarantined = list(out.glob("orphan-quarantine-*/*.npz"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"partial private pixels"
    second = tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    assert second["complete"] and second["records"][:2] == first["records"]


def test_finished_patient_storage_cap_rolls_back_new_uncommitted_outputs(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(tool, "MAX_PREPARED_BYTES", 1)
    with pytest.raises(PreparationError, match="prepared_storage_limit"):
        tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    assert not list((out / "data").glob("*.npz"))
    assert not (out / "preparation.private.json").exists()
    assert not (out / "preparation.lock").exists()


def test_private_patient_receipts_bind_offsets_and_failures_without_index_rows(
    tmp_path, monkeypatch
):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    first = tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    receipt = first["patient_receipts"][0]
    assert (
        receipt["tar_end"] == first["next_offset"]
        and 0 <= receipt["tar_start"] < receipt["tar_end"]
    )
    assert (
        receipt["status"] == "prepared" and receipt["code"] == "ok" and receipt["side_count"] == 2
    )
    assert re.fullmatch("[0-9a-f]{64}", receipt["patient_id"])

    def fail(*args):
        raise PreparationError("csv_annotation_label_mismatch")

    monkeypatch.setattr(tool, "prepare_patient", fail)
    second = tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    last = second["patient_receipts"][1]
    assert last["tar_start"] == receipt["tar_end"] and last["tar_end"] == tool.ARCHIVE_BYTES
    assert (
        last["status"] == "failed"
        and last["code"] == "csv_annotation_label_mismatch"
        and last["side_count"] == 0
    )
    assert "patient_receipts" not in json.loads((out / "index.private.json").read_text())
    assert (
        json.loads((out / "preparation.private.json").read_text())["patient_receipts"]
        == second["patient_receipts"]
    )


def test_exhausted_transport_retries_never_expose_exception_text():
    calls = []

    def failed(request, timeout):
        calls.append(request)
        raise OSError("private signed-url and local path")

    with pytest.raises(PreparationError, match="^source_range_failed$") as error:
        with tool.RangeReader("https://example.test/data", 0, 7, 8, opener=failed) as reader:
            reader.read(8)
    assert len(calls) == 3 and error.value.__cause__ is None


def test_explicit_orphan_recovery_without_lock_acquires_exclusive_owner(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    orphan = out / "data" / ("b" * 64 + "-R.npz")
    orphan.write_bytes(b"uncommitted")
    tool.reconcile_stale_lock(out)
    assert not orphan.exists() and not (out / "preparation.lock").exists()
    assert len(list(out.glob("orphan-quarantine-*/*.npz"))) == 1


def test_explicit_recovery_rejects_unknown_files_or_invalid_state(tmp_path, monkeypatch):
    metadata, _ = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    tool.prepare_archive(out, metadata, max_patients=1, slice_count=4, image_size=16)
    unknown = out / "data" / "not-owned.bin"
    unknown.write_bytes(b"preserve")
    with pytest.raises(PreparationError, match="unsafe_prepared_orphan"):
        tool.reconcile_stale_lock(out)
    assert unknown.read_bytes() == b"preserve"
    unknown.unlink()
    (out / "preparation.private.json").write_text('{"records":[]}')
    with pytest.raises(PreparationError, match="invalid_preparation_state"):
        tool.reconcile_stale_lock(out)


@pytest.mark.parametrize(
    "retry_after, expected", [(None, [30, 60]), ("7", [7, 7]), ("999", [60, 60]), ("bad", [30, 60])]
)
def test_rate_limit_has_bounded_interruptible_backoff(monkeypatch, retry_after, expected):
    import urllib.error

    calls = []
    delays = []

    def opener(request, timeout):
        calls.append(True)
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        raise urllib.error.HTTPError(
            "https://example.test/private", 429, "private reason", headers, None
        )

    monkeypatch.setattr(
        tool.RangeReader,
        "_wait_retry",
        lambda self, delay: delays.append(delay) or False,
        raising=False,
    )
    with pytest.raises(PreparationError, match="^source_rate_limited$"):
        with tool.RangeReader("https://example.test/private", 0, 7, 8, opener=opener) as reader:
            reader.read(8)
    assert len(calls) == 3 and delays == expected


def test_stop_during_rate_backoff_does_not_retry(monkeypatch):
    import urllib.error

    calls = []

    def opener(request, timeout):
        calls.append(True)
        raise urllib.error.HTTPError("https://example.test/private", 429, "private", {}, None)

    monkeypatch.setattr(tool.RangeReader, "_wait_retry", lambda self, delay: True, raising=False)
    with pytest.raises(PreparationError, match="^source_rate_limited$"):
        with tool.RangeReader("https://example.test/private", 0, 7, 8, opener=opener) as reader:
            reader.read(8)
    assert len(calls) == 1


@pytest.mark.parametrize("boundary", ["last_patient_state", "complete_index"])
def test_resume_finalizes_eof_checkpoint_and_repairs_index_without_network(
    tmp_path, monkeypatch, boundary
):
    metadata, calls = archive_fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    write = tool._write

    def interrupted(path, value):
        if (
            boundary == "complete_index"
            and path.name == "index.private.json"
            and value.get("complete") is True
        ):
            raise OSError("synthetic crash before derived index")
        write(path, value)
        if (
            boundary == "last_patient_state"
            and path.name == "preparation.private.json"
            and value.get("next_offset") == tool.ARCHIVE_BYTES
            and value.get("complete") is False
        ):
            raise OSError("synthetic crash after EOF patient checkpoint")

    monkeypatch.setattr(tool, "_write", interrupted)
    with pytest.raises(OSError, match="synthetic crash"):
        tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    state = json.loads((out / "preparation.private.json").read_text())
    assert state["next_offset"] == tool.ARCHIVE_BYTES and len(state["records"]) == 4
    before = [r["crop_sha256"] for r in state["records"]]
    monkeypatch.setattr(tool, "_write", write)
    monkeypatch.setattr(
        tool.urllib.request, "urlopen", lambda *a, **k: pytest.fail("EOF resume must not download")
    )
    resumed = tool.prepare_archive(out, metadata, slice_count=4, image_size=16)
    index = json.loads((out / "index.private.json").read_text())
    assert resumed["complete"] is True and index["complete"] is True
    assert [r["crop_sha256"] for r in index["records"]] == before
    assert index["patient_count"] == 2
