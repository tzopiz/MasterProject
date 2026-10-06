"""Bounded synthetic checks: no cohort, pretrained weights or accelerator needed."""

import contextlib
import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch import nn


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(14), nn.Conv2d(3, 256, 1))
        self.bn1, self.relu, self.maxpool = nn.BatchNorm2d(256), nn.ReLU(), nn.Identity()
        self.layer1, self.layer2, self.layer3 = nn.Identity(), nn.Identity(), nn.Identity()
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 1, groups=256), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.avgpool, self.fc = nn.AdaptiveAvgPool2d(1), nn.Identity()


@pytest.fixture
def lib():
    assert importlib.util.find_spec("training.tmj_osseous_layer4") is not None
    return importlib.import_module("training.tmj_osseous_layer4")


@pytest.fixture
def model(lib):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return lib.BagLayer4Model(TinyResNet(), device="cpu")


def cached():
    x = torch.ones(4, 16, 256, 14, 14)
    x[2:] = 3
    return x


def test_layer4_library_exists():
    assert importlib.util.find_spec("training.tmj_osseous_layer4") is not None


def test_transform_matches_whole_roi_interpolation_and_rgb_normalization(lib):
    crop = np.zeros((16, 1, 96, 96), np.float16)
    crop[:, :, :, 48:] = 1
    result = lib.normalize_crop(crop)
    assert result.shape == (16, 3, 224, 224) and result.dtype == torch.float32
    edge = result[0, 0, 111, 111:113] * 0.229 + 0.485
    torch.testing.assert_close(edge, torch.tensor([2 / 7, 5 / 7]), atol=5e-6, rtol=0)
    torch.testing.assert_close(
        result[0, :, 0, 0], torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    )
    torch.testing.assert_close(lib.normalize_crop(crop.astype(np.float32)), result)


def test_prefix_cache_matches_direct_predictions_is_detached_and_order_invariant(lib, model):
    crops = np.random.default_rng(2).random((2, 16, 1, 96, 96), dtype=np.float32)
    with torch.no_grad():
        model.head.weight.fill_(0.001)
    calls = []
    cache = lib.cache_prefix(model, crops, deadline=lambda: calls.append(1))
    assert cache.shape == (2, 16, 256, 14, 14)
    assert cache.dtype == torch.float32 and cache.device.type == "cpu" and not cache.requires_grad
    with torch.no_grad():
        direct = []
        for crop in crops:
            prefix = model.prefix(lib.normalize_crop(crop))
            pooled = model.layer4(prefix).mean((2, 3)).mean(0)
            direct.append(model.head(pooled).sigmoid().reshape(()))
    torch.testing.assert_close(lib.predict(model, cache), torch.stack(direct), atol=2e-6, rtol=0)
    torch.testing.assert_close(lib.predict(model, cache.flip(1)), lib.predict(model, cache))
    assert len(calls) == 2 and all(p.grad is None for p in model.prefix.parameters())


def test_fit_warmup_freezes_bn_prefix_then_only_updates_layer4_conv_and_head(lib, model):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    snapshots, calls = [], []

    def observe(fitted, epoch, row):
        snapshots.append({name: value.clone() for name, value in fitted.state_dict().items()})
        assert epoch == len(snapshots) and row["epoch"] == epoch and row["steps"] == 1
        assert all(not bn.training for bn in fitted.modules() if isinstance(bn, nn.BatchNorm2d))

    history = lib.fit_epochs(
        model,
        cached(),
        np.array([0, 0, 1, 1]),
        3,
        observer=observe,
        deadline=lambda: calls.append(1),
    )
    assert len(history) == 3 and len(calls) == 3
    assert all(np.isfinite(row["loss"]) and row["loss"] > 0 for row in history)
    assert not torch.equal(snapshots[0]["head.weight"], before["head.weight"])
    for name, value in before.items():
        if name.startswith("head."):
            continue
        if name in ("layer4.0.weight", "layer4.0.bias"):
            assert torch.equal(snapshots[0][name], value) and torch.equal(snapshots[1][name], value)
        else:
            assert torch.equal(model.state_dict()[name], value), name
    assert not torch.equal(snapshots[2]["layer4.0.weight"], before["layer4.0.weight"])
    model.train()
    assert not model.prefix.training
    assert all(not bn.training for bn in model.modules() if isinstance(bn, nn.BatchNorm2d))
    assert all(
        not p.requires_grad
        for bn in model.modules()
        if isinstance(bn, nn.BatchNorm2d)
        for p in bn.parameters()
    )
    replay = lib.BagLayer4Model(TinyResNet())
    replay.load_state_dict(model.state_dict())
    torch.testing.assert_close(lib.predict(replay, cached()), lib.predict(model, cached()))


def test_zero_head_weighted_bce_and_fit_seed_are_reproducible(lib, model):
    torch.testing.assert_close(lib.predict(model, cached()), torch.full((4,), 0.5))
    replay = lib.BagLayer4Model(TinyResNet())
    replay.load_state_dict(model.state_dict())
    y = torch.tensor([0, 0, 0, 1])
    history = lib.fit_epochs(model, cached(), y, 1)
    assert history[0]["loss"] == pytest.approx(1.5 * np.log(2), rel=1e-6)
    torch.manual_seed(999)
    lib.fit_epochs(replay, cached(), y, 1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(replay.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "low", "high"])
def test_invalid_crop_is_rejected(lib, mutation):
    x = np.zeros((16, 1, 96, 96), np.float32)
    if mutation == "shape":
        x = x[:15]
    elif mutation == "dtype":
        x = x.astype(np.float64)
    else:
        x.flat[0] = {"nan": np.nan, "low": -0.1, "high": 1.1}[mutation]
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_crop$"):
        lib.normalize_crop(x)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "empty", "grad"])
def test_invalid_cache_is_rejected_before_prediction(lib, model, mutation):
    x = cached()
    if mutation == "shape":
        x = x[:, :15]
    elif mutation == "dtype":
        x = x.half()
    elif mutation == "nan":
        x[0, 0, 0, 0].fill_(float("nan"))
    elif mutation == "empty":
        x = x[:0]
    else:
        x.requires_grad_()
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_cache$"):
        lib.predict(model, x)


@pytest.mark.parametrize(
    "targets",
    [[0, 0, 0, 0], [1, 1, 1, 1], [0, 2, 0, 1], [0, np.nan, 0, 1], [[0], [0], [1], [1]], [0, 1]],
)
def test_fit_rejects_invalid_or_single_class_targets(lib, model, targets):
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_targets$"):
        lib.fit_epochs(model, cached(), np.asarray(targets), 1)


@pytest.mark.parametrize("epochs", [0, 17, True, 1.0])
def test_fit_rejects_invalid_epochs(lib, model, epochs):
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_epochs$"):
        lib.fit_epochs(model, cached(), torch.tensor([0, 0, 1, 1]), epochs)


def test_bf16_cpu_and_unknown_precision_fail_before_updates(lib, model):
    for precision, code in [
        ("bf16", "layer4_bf16_unavailable"),
        ("fp16", "invalid_layer4_precision"),
    ]:
        with pytest.raises(lib.ResearchError, match=f"^{code}$"):
            lib.fit_epochs(model, cached(), torch.tensor([0, 0, 1, 1]), 1, precision=precision)
    torch.testing.assert_close(model.head.weight, torch.zeros_like(model.head.weight))
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_device$"):
        lib.BagLayer4Model(TinyResNet(), device="mps")


def test_offline_loader_uses_existing_pinned_weight_validation(lib, monkeypatch, tmp_path):
    with pytest.raises(lib.ResearchError, match="^feature_weights_checksum_mismatch$"):
        path = tmp_path / "bad.pt"
        path.write_bytes(b"bad")
        lib.load_layer4_model(path)
    monkeypatch.setattr(lib, "load_backbone", lambda path: TinyResNet())
    model = lib.load_layer4_model("caller-local-path")
    torch.testing.assert_close(lib.predict(model, cached()), torch.full((4,), 0.5))


def test_nonfinite_model_outputs_and_gradients_fail_safely(lib, model):
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_logits$"):
        lib.predict(model, cached())
    with torch.no_grad():
        model.head.weight.zero_()
    handle = model.head.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_gradients$"):
        lib.fit_epochs(model, cached(), torch.tensor([0, 0, 1, 1]), 1)
    handle.remove()


def test_direct_forward_converts_invalid_backbone_output_to_safe_error(lib, model):
    model.layer4 = nn.Identity()
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_logits$"):
        model(cached())
    model.layer4 = nn.Flatten(start_dim=1)
    with pytest.raises(lib.ResearchError, match="^invalid_layer4_logits$"):
        model(cached())
    model.layer4 = nn.Sequential(nn.Conv2d(1, 512, 1))
    with pytest.raises(lib.ResearchError, match="^layer4_forward_failed$"):
        model(cached())


def test_fit_multi_batch_seed_deadline_and_observer_are_enforced(lib, model):
    x = torch.cat((cached(), cached() * 2))
    y = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1])
    replay = lib.BagLayer4Model(TinyResNet())
    replay.load_state_dict(model.state_dict())
    history = lib.fit_epochs(model, x, y, 3)
    torch.manual_seed(123)
    lib.fit_epochs(replay, x, y, 3)
    assert [row["steps"] for row in history] == [2, 2, 2]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(replay.state_dict()[name], value, rtol=0, atol=0)
    observed = []

    def expired():
        raise lib.ResearchError("caller_deadline")

    with pytest.raises(lib.ResearchError, match="^caller_deadline$"):
        lib.fit_epochs(model, x, y, 1, deadline=expired, observer=lambda *args: observed.append(1))
    assert not observed


def test_cuda_bf16_gate_requires_native_capability_before_fit(lib, model, monkeypatch):
    monkeypatch.setattr(lib, "_model_device", lambda fitted: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    calls = []

    def supported(*, including_emulation):
        calls.append(including_emulation)
        return False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    with pytest.raises(lib.ResearchError, match="^layer4_bf16_unavailable$"):
        lib.fit_epochs(model, cached(), torch.tensor([0, 0, 1, 1]), 1, precision="bf16")
    assert calls == [False]
    torch.testing.assert_close(model.head.weight, torch.zeros_like(model.head.weight))
