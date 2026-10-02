#!/usr/bin/env python3
"""Per-side ROI classifier sharing the original binary backbone.

Binary default: sagittal/frontal single logits (the research run trains sagittal
only). Multiclass: three sagittal logits central/anterior/posterior, no frontal
head/output. Both consume one normalized NCDHW detector ROI, no resampling.
Research checkpoints declare trained tasks and exact inference decision rule.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.blocks import _conv_block, validate_research_architecture


class TMJBinaryPositionClassifier(nn.Module):
    """
    ROI TMJ position classifier, per side; binary default or sagittal3-class.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale CBCT crop).
    features : list of int
        Output channels for each encoder block.
    fc_hidden : int
        Width of the hidden layer in each head.
    dropout : float
        Dropout probability before the final linear layer.
    num_classes : int
        2 keeps legacybinary logits/heads;3 creates only sagittal3-class logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Optional[List[int]] = None,
        fc_hidden: int = 256,
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes not in (2, 3):
            raise ValueError("invalid_position_model_classes")
        self.num_classes = num_classes

        if features is None:
            features = [16, 32, 64, 128]

        validate_research_architecture(features, fc_hidden=fc_hidden, in_channels=in_channels)

        blocks: List[nn.Module] = []
        prev = in_channels
        for out_ch in features:
            blocks.append(_conv_block(prev, out_ch))
            prev = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        feat_dim = features[-1]

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feat_dim, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_hidden, 1 if num_classes == 2 else 3),
            )

        self.head_sag = _head()
        if num_classes == 2:
            self.head_fr = _head()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 1, D, H, W) — normalised 128³ crop for one condyle side.

        Returns:
            Binary: (sag_logit,fr_logit), each(B,1). Multiclass: ((B,3),None).
        """
        feat = self.backbone(x)  # (B, C, d, h, w)
        feat = self.global_pool(feat)  # (B, C, 1, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, C)
        return self.head_sag(feat), self.head_fr(feat) if self.num_classes == 2 else None


BINARY_CHECKPOINT_FAMILY = "tmj_binary_position_classifier"
BINARY_PREPROCESSING = {
    "version": "tmj-binary-nifti-percentile-v1",
    "input": "one-side-nifti-array",
    "layout": "NCDHW",
    "channels": 1,
    "clip_percentiles": [2, 98],
    "scale": "per-volume-0-1;constant-volume-zero",
    "resampling": "none",
}
BINARY_CLASS_SEMANTICS = {"0": "central", "1": "non-central"}
MULTICLASS_CHECKPOINT_FAMILY = "tmj_roi_sagittal_multiclass_classifier"
MULTICLASS_CLASS_SEMANTICS = {"0": "central", "1": "anterior", "2": "posterior"}
MULTICLASS_DECISION_RULE = "argmax;ties-lowest-class-index"


def load_position_checkpoint(path, device="cpu", expected_sha256=None, expected_mode=None):
    """Load only the versioned sagittal research contract, using weights_only.

    Metadata is deliberately free of patient identifiers and private paths.
    Binary frontal head is structural/untrained; multiclass has no frontal head.
    Normalized metadata always includes mode and number of declared classes.
    The optional digest binds this local checkpoint to its run report.
    """
    import hashlib
    import math
    from pathlib import Path

    try:
        data = Path(path).read_bytes()
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError
        import io

        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        required = {
            "schema_version",
            "family",
            "model_kwargs",
            "trained_tasks",
            "preprocessing",
            "class_semantics",
            "threshold",
            "decision_rule",
            "model_state_dict",
            "fold",
            "best_epoch",
        }
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("family") == MULTICLASS_CHECKPOINT_FAMILY:
            mode = "multiclass"
            required |= {"mode", "num_classes"}
            if (
                payload.get("mode") != mode
                or type(payload.get("num_classes")) is not int
                or payload["num_classes"] != 3
            ):
                raise ValueError
        elif payload.get("family") == BINARY_CHECKPOINT_FAMILY:
            mode = "binary"
        else:
            raise ValueError
        if expected_mode is not None and expected_mode != mode:
            raise ValueError
        if set(payload) != required:
            raise ValueError
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError
        if payload["trained_tasks"] != ["sagittal"] or payload["decision_rule"] != (
            ">=" if mode == "binary" else MULTICLASS_DECISION_RULE
        ):
            raise ValueError
        if payload["preprocessing"] != BINARY_PREPROCESSING:
            raise ValueError
        if payload["class_semantics"] != (
            BINARY_CLASS_SEMANTICS if mode == "binary" else MULTICLASS_CLASS_SEMANTICS
        ):
            raise ValueError
        for field, minimum in (("fold", 0), ("best_epoch", 1)):
            if type(payload[field]) is not int or payload[field] < minimum:
                raise ValueError
        threshold = payload["threshold"]
        if mode == "multiclass" and threshold is not None:
            raise ValueError
        if mode == "binary" and (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError
        kwargs = payload["model_kwargs"]
        expected_kwargs = {"in_channels", "features", "fc_hidden", "dropout"}
        if mode == "multiclass":
            expected_kwargs.add("num_classes")
        if not isinstance(kwargs, dict) or set(kwargs) != expected_kwargs:
            raise ValueError
        if mode == "multiclass" and (
            type(kwargs["num_classes"]) is not int or kwargs["num_classes"] != 3
        ):
            raise ValueError
        if type(kwargs["in_channels"]) is not int or kwargs["in_channels"] != 1:
            raise ValueError
        if not isinstance(kwargs["features"], list) or not kwargs["features"]:
            raise ValueError
        if any(type(v) is not int or v < 1 for v in kwargs["features"]):
            raise ValueError
        if type(kwargs["fc_hidden"]) is not int or kwargs["fc_hidden"] < 1:
            raise ValueError
        if type(kwargs["dropout"]) not in (int, float) or not 0 <= kwargs["dropout"] < 1:
            raise ValueError
        validate_research_architecture(
            kwargs["features"], fc_hidden=kwargs["fc_hidden"], in_channels=kwargs["in_channels"]
        )
        state = payload["model_state_dict"]
        if not isinstance(state, dict) or not state:
            raise ValueError
        if any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()
        ):
            raise ValueError
        model = TMJBinaryPositionClassifier(**kwargs)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
        metadata.update(mode=mode, num_classes=2 if mode == "binary" else 3)
        return model, metadata
    except Exception:
        raise ValueError("invalid_position_checkpoint") from None


def load_binary_position_checkpoint(path, device="cpu", expected_sha256=None):
    """Backward-compatible binary-only loader; refuses multiclass checkpoints."""
    try:
        return load_position_checkpoint(
            path, device=device, expected_sha256=expected_sha256, expected_mode="binary"
        )
    except ValueError:
        raise ValueError("invalid_binary_checkpoint") from None
