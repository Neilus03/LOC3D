"""Strict official Sonata Small initialization with the matched native readout.

The downstream representative voxels, RGBN6, coordinates, exact pooling maps,
and task input projections are shared with Concerto. Sonata's own PTv3 source
and released weights supply the encoder; no pretrained task decoder is loaded.
"""
from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import torch
from torch import nn

from delimit3d.models.concerto_wrapper import (
    ConcertoBackbone, ConcertoMaskAdapters, agile_adapter, component_counts,
    configure_direct_decoder_inputs, file_sha256, gradient_report, tensor_hash,
)
from delimit3d.training.sonata_contract import (
    NATIVE_STAGE_RECIPE, SONATA_CHECKPOINT_SHA256, SONATA_ENCODER_PARAMETERS,
    SONATA_SOURCE_COMMIT, sonata_feature_channels, validate_checkpoint_config,
)

# These projections are deliberately identical to the Concerto comparison.
SonataMaskAdapters = ConcertoMaskAdapters


class SonataBackbone(ConcertoBackbone):
    """Reuse the voxel/hierarchy bridge, initializing only from Sonata Small."""

    def __init__(self, *, source_root, checkpoint, frozen,
                 activation_checkpointing=False, grid_size=.05,
                 feature_recipe=NATIVE_STAGE_RECIPE):
        nn.Module.__init__(self)
        self.feature_recipe = feature_recipe
        self.feature_channels = sonata_feature_channels(feature_recipe)
        if float(grid_size) != .05:
            raise ValueError("Sonata's matched downstream voxel size must be 5 cm")
        source = Path(source_root).resolve(strict=True)
        sys.path.insert(0, str(source))
        module = importlib.import_module("sonata.model")
        if Path(module.__file__).resolve() != source / "sonata/model.py":
            raise RuntimeError("Imported Sonata model from a different source snapshot")
        observed = file_sha256(checkpoint)
        if observed != SONATA_CHECKPOINT_SHA256:
            raise ValueError(f"Sonata Small checkpoint SHA256 mismatch: {observed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(payload["config"])
        validate_checkpoint_config(config)
        self.encoder = module.PointTransformerV3(**config)
        expected, incoming = self.encoder.state_dict(), payload["state_dict"]
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        incompatible = {
            name: {"expected": list(expected[name].shape), "supplied": list(value.shape)}
            for name, value in incoming.items()
            if name in expected and expected[name].shape != value.shape
        }
        self.load_report = {
            "model": "sonata_small", "file_sha256": observed,
            "source_commit": SONATA_SOURCE_COMMIT, "tensor_count": len(incoming),
            "missing_keys": missing, "unexpected_keys": unexpected,
            "incompatible_tensors": incompatible, "config": config, "strict": True,
        }
        if missing or unexpected or incompatible:
            raise RuntimeError(f"Pretrained Sonata encoder mismatch: {self.load_report}")
        self.encoder.load_state_dict(incoming, strict=True)
        if any(not torch.equal(expected[name], value) for name, value in incoming.items()):
            raise RuntimeError("Loaded Sonata tensors differ from the supplied checkpoint")
        count = sum(parameter.numel() for parameter in self.encoder.parameters())
        if count != SONATA_ENCODER_PARAMETERS:
            raise ValueError(f"Unexpected Sonata Small parameter count: {count}")
        self.in_channels = 6
        self.grid_size = float(grid_size)
        self.voxel_reduce = "representative"
        self.representative_sampling = "first"
        self.activation_checkpointing = bool(activation_checkpointing)
        self.frozen = bool(frozen)
        self.set_frozen(frozen)
        self.load_report.update(
            parameters=count, state_sha256=tensor_hash(self),
            normalization_modules={
                "batchnorm": sum(isinstance(m, nn.modules.batchnorm._BatchNorm)
                                 for m in self.modules()),
                "layernorm": sum(isinstance(m, nn.LayerNorm) for m in self.modules()),
            },
        )

    def forward(self, *args, **kwargs):
        pyramid = super().forward(*args, **kwargs)
        return replace(pyramid, memory_levels=tuple(
            replace(level, name=level.name.replace("concerto_", "sonata_"))
            for level in pyramid.memory_levels
        ))
