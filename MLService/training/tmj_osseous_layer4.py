"""Offline whole-ROI layer4 adaptation; callers own splits and model selection."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.tmj_osseous_features import load_backbone
from training.tmj_osseous_research import ResearchError


def _fail(code):
    raise ResearchError(code) from None


def _device(value):
    try:
        device = torch.device(value)
        if (
            device.type not in ("cpu", "cuda")
            or (device.type == "cpu" and device.index is not None)
            or (
                device.type == "cuda"
                and (
                    not torch.cuda.is_available()
                    or (device.index is not None and device.index >= torch.cuda.device_count())
                )
            )
        ):
            _fail("invalid_layer4_device")
        return device
    except ResearchError:
        raise
    except Exception:
        _fail("invalid_layer4_device")


def _model_device(model):
    if not isinstance(model, BagLayer4Model):
        _fail("invalid_layer4_model")
    parameters = list(model.parameters())
    device = _device(parameters[0].device)
    if any(p.device != device or p.dtype != torch.float32 for p in parameters):
        _fail("invalid_layer4_model")
    return device


def _cache(value):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or value.shape[1:] != (16, 256, 14, 14)
        or not len(value)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.device.type not in ("cpu", "cuda")
        or not torch.isfinite(value).all()
    ):
        _fail("invalid_layer4_cache")
    return value


def normalize_crop(crop):
    """Apply the existing whole-ROI RGB/ImageNet recipe to one normalized bag."""
    if (
        not isinstance(crop, np.ndarray)
        or crop.shape != (16, 1, 96, 96)
        or crop.dtype not in (np.dtype("float16"), np.dtype("float32"))
        or not np.isfinite(crop).all()
        or crop.min() < 0
        or crop.max() > 1
    ):
        _fail("invalid_layer4_crop")
    images = torch.from_numpy(crop.astype(np.float32, copy=True))
    images = F.interpolate(
        images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    ).repeat(1, 3, 1, 1)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


class BagLayer4Model(nn.Module):
    """Frozen prefix/BN, trainable layer4 convolutions, zero binary bag head."""

    def __init__(self, backbone, *, device="cpu"):
        super().__init__()
        device = _device(device)
        try:
            self.prefix = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            )
            self.layer4 = backbone.layer4
            self.head = nn.Linear(512, 1)
            self.requires_grad_(False)
            for module in self.layer4.modules():
                if isinstance(module, nn.Conv2d):
                    module.requires_grad_(True)
            self.head.requires_grad_(True)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.to(device=device, dtype=torch.float32)
            self.eval()
        except Exception:
            _fail("invalid_layer4_backbone")

    def train(self, mode=True):
        super().train(mode)
        self.prefix.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(False).eval()
        return self

    def forward(self, cached):
        cached = _cache(cached)
        if cached.device != _model_device(self):
            _fail("invalid_layer4_device")
        try:
            encoded = self.layer4(cached.reshape(-1, 256, 14, 14))
            if (
                not isinstance(encoded, torch.Tensor)
                or encoded.ndim != 4
                or encoded.shape[:2] != (len(cached) * 16, 512)
            ):
                _fail("invalid_layer4_logits")
            pooled = encoded.float().mean((2, 3)).reshape(len(cached), 16, 512).mean(1)
            logits = self.head(pooled).float().reshape(-1)
            if logits.shape != (len(cached),) or not torch.isfinite(logits).all():
                _fail("invalid_layer4_logits")
            return logits
        except ResearchError:
            raise
        except Exception:
            _fail("layer4_forward_failed")


def load_layer4_model(local_weights, *, device="cpu"):
    """Use the existing digest-verified offline loader; never fetch weights."""
    _device(device)
    return BagLayer4Model(load_backbone(local_weights), device=device)


def cache_prefix(model, crops, *, deadline=None):
    """Return detached CPU FP32 [N,16,256,14,14], independent of labels/statistics."""
    device = _model_device(model)
    if (
        not isinstance(crops, np.ndarray)
        or crops.ndim != 5
        or crops.shape[1:] != (16, 1, 96, 96)
        or not len(crops)
    ):
        _fail("invalid_layer4_crop")
    try:
        model.eval()
        result = []
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for crop in crops:
                if deadline is not None:
                    deadline()
                result.append(model.prefix(normalize_crop(crop).to(device)).float().cpu())
        return _cache(torch.stack(result))
    except ResearchError:
        raise
    except Exception:
        _fail("layer4_cache_failed")


def predict(model, cached, *, deadline=None):
    """Return CPU FP32 positive probabilities with FP32 inference in batches of four."""
    device, cached = _model_device(model), _cache(cached)
    try:
        model.eval()
        result = []
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            for batch in cached.split(4):
                if deadline is not None:
                    deadline()
                result.append(model(batch.to(device)).sigmoid().cpu())
        probabilities = torch.cat(result)
        if (
            not torch.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            _fail("invalid_layer4_probabilities")
        return probabilities
    except ResearchError:
        raise
    except Exception:
        _fail("layer4_prediction_failed")


def fit_epochs(
    model, cached_fit, targets, epochs, *, precision="fp32", observer=None, deadline=None
):
    """Fit supplied rows only; observer(model, 1-based epoch, row) runs after each epoch."""
    device, cached_fit = _model_device(model), _cache(cached_fit)
    if (
        isinstance(epochs, (bool, np.bool_))
        or not isinstance(epochs, (int, np.integer))
        or not 1 <= epochs <= 16
    ):
        _fail("invalid_layer4_epochs")
    if not isinstance(precision, str) or precision not in ("fp32", "bf16"):
        _fail("invalid_layer4_precision")
    if precision == "bf16":
        if device.type != "cuda":
            _fail("layer4_bf16_unavailable")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                _fail("layer4_bf16_unavailable")
    try:
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets.copy())
        if (
            not isinstance(targets, torch.Tensor)
            or targets.shape != (len(cached_fit),)
            or targets.is_complex()
            or not torch.isfinite(targets).all()
            or not ((targets == 0) | (targets == 1)).all()
            or not (targets == 0).any()
            or not (targets == 1).any()
        ):
            _fail("invalid_layer4_targets")
        targets = targets.detach().to(device=device, dtype=torch.float32)
    except ResearchError:
        raise
    except Exception:
        _fail("invalid_layer4_targets")
    try:
        torch.manual_seed(42)
        conv = [
            p
            for module in model.layer4.modules()
            if isinstance(module, nn.Conv2d)
            for p in module.parameters(recurse=False)
        ]
        optimizer = torch.optim.AdamW(
            [{"params": model.head.parameters(), "lr": 1e-3}, {"params": conv, "lr": 1e-5}],
            weight_decay=1e-4,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=(targets == 0).sum() / (targets == 1).sum())
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            for parameter in conv:
                parameter.requires_grad_(epoch > 2)
            loss_sum, steps = 0.0, 0
            for indices in torch.randperm(len(cached_fit), device="cpu").split(4):
                if deadline is not None:
                    deadline()
                batch = cached_fit[indices.to(cached_fit.device)].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
                ):
                    logits = model(batch)
                loss = criterion(logits.float(), targets[indices.to(device)])
                if not torch.isfinite(loss):
                    _fail("invalid_layer4_loss")
                loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    _fail("invalid_layer4_gradients")
                if not torch.isfinite(nn.utils.clip_grad_norm_(parameters, 1.0)):
                    _fail("invalid_layer4_gradients")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    _fail("invalid_layer4_parameters")
                loss_sum += float(loss.detach()) * len(indices)
                steps += 1
            row = {"epoch": epoch, "loss": loss_sum / len(cached_fit), "steps": steps}
            history.append(row)
            if observer is not None:
                observer(model, epoch, row)
        return history
    except ResearchError:
        raise
    except Exception:
        _fail("layer4_fit_failed")
