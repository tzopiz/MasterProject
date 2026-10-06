"""Optional spatial capacity: synthetic bags, replay and legacy provenance."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import datasphere_osseous as cloud
from training import tmj_osseous_research as research

from .test_tmj_osseous_research import fixture


@pytest.mark.parametrize("mode,outputs", [("binary", 1), ("multilabel", 6)])
def test_spatial_development_replays_without_test_crops(tmp_path, mode, outputs):
    index, config = fixture(tmp_path)
    split = research.build_split(index["records"])
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            dict(
                task=research.TASK, seed=42, membership=split, split_digest=research._digest(split)
            )
        )
    )
    config.update(
        architecture="spatial_head", mode=mode, split_path=str(frozen), epochs=1, bootstrap_draws=0
    )
    for record in index["records"]:
        if record["patient_id"] in split["test"]:
            (tmp_path / record["crop_path"]).unlink()
    preview = research.preflight(config, development=True)
    assert preview["bindings"]["architecture"] == "spatial_head"
    report = research.train(config, development=True)
    assert "test" not in report and report["architecture"] == "spatial_head"
    run = Path(config["output_dir"])
    model, metadata = research.load_checkpoint(run / "checkpoint.private.pt")
    assert metadata["architecture"] == metadata["config"]["architecture"] == "spatial_head"
    predictions = json.loads((run / "predictions.private.json").read_text())["records"]
    record = next(
        r
        for r in index["records"]
        if r["patient_id"] == predictions[0]["patient_id"] and r["side"] == predictions[0]["side"]
    )
    with np.load(tmp_path / record["crop_path"]) as crop:
        images = torch.from_numpy(crop["images"].astype(np.float32))[None]
    with torch.no_grad():
        result = torch.sigmoid(model(images)).numpy()[0]
    assert result.shape == (outputs,)
    np.testing.assert_allclose(result, predictions[0]["probability"], atol=1e-7)
    assert {r["partition"] for r in predictions} == {"train", "validation"}
    assert cloud.artifacts_complete(run, preview["bindings"])
    legacy_binding = dict(preview["bindings"])
    legacy_binding.pop("architecture")
    assert not cloud.artifacts_complete(run, legacy_binding)
    for name in ("global_mean", "SyntheticPrivateUnknown"):
        report["architecture"] = name
        (run / "report.json").write_text(json.dumps(report))
        completion = json.loads((run / "completion.json").read_text())
        completion["artifact_digests"]["report.json"] = cloud._hash_file(run / "report.json")
        (run / "completion.json").write_text(json.dumps(completion))
        assert not cloud.artifacts_complete(run)


@pytest.mark.parametrize("architecture", ["global_mean", "spatial_head"])
def test_bag_slice_permutation_preserves_logits(architecture):
    torch.manual_seed(7)
    model = research.build_model(6, architecture).eval()
    images = torch.rand(2, 3, 1, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(model(images), model(images[:, [2, 0, 1]]), atol=1e-7, rtol=0)


def test_legacy_checkpoint_and_default_model_keep_exact_logits_and_binding(tmp_path):
    _, config = fixture(tmp_path)
    implicit = research.preflight(config)["bindings"]
    explicit = research.preflight(dict(config, architecture="global_mean"))["bindings"]
    assert implicit == explicit and "architecture" not in implicit
    torch.manual_seed(9)
    original = research.SliceBagClassifier(1).eval()
    torch.manual_seed(9)
    default = research.build_model(1).eval()
    images = torch.rand(2, 3, 1, 8, 8)
    path = tmp_path / "legacy.pt"
    torch.save(
        dict(
            state_dict=original.state_dict(),
            metadata=dict(mode="binary", config={}, bindings=implicit),
        ),
        path,
    )
    reloaded, _ = research.load_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(original(images), default(images))
        assert torch.equal(original(images), reloaded(images))


@pytest.mark.parametrize("name", ["SyntheticPrivateUnknown", None, [], 3])
def test_unknown_architecture_fails_before_input_io(tmp_path, name):
    config = dict(
        index_path=str(tmp_path / "absent.json"),
        output_dir=str(tmp_path / "run"),
        architecture=name,
    )
    with pytest.raises(research.ResearchError, match="^unsupported_architecture$"):
        research.preflight(config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize(
    "metadata_name,config_name,binding_name,code",
    [
        ("SyntheticPrivateUnknown", "global_mean", None, "unsupported_architecture"),
        ("global_mean", "SyntheticPrivateUnknown", None, "unsupported_architecture"),
        ("global_mean", "spatial_head", None, "checkpoint_architecture_mismatch"),
        ("spatial_head", "spatial_head", None, "checkpoint_architecture_mismatch"),
        ("global_mean", "global_mean", "spatial_head", "checkpoint_architecture_mismatch"),
        ("global_mean", "global_mean", "SyntheticPrivateUnknown", "unsupported_architecture"),
    ],
)
def test_checkpoint_rejects_unknown_or_conflicting_architecture(
    tmp_path, metadata_name, config_name, binding_name, code
):
    binding = dict(task=research.TASK, codebook_commit=research.CODEBOOK)
    if binding_name is not None:
        binding["architecture"] = binding_name
    path = tmp_path / "bad.pt"
    torch.save(
        dict(
            state_dict=research.SliceBagClassifier().state_dict(),
            metadata=dict(
                mode="binary",
                architecture=metadata_name,
                config=dict(architecture=config_name),
                bindings=binding,
            ),
        ),
        path,
    )
    with pytest.raises(research.ResearchError, match="^" + code + "$"):
        research.load_checkpoint(path)
