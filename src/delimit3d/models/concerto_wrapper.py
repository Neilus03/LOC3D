"""Pinned Concerto encoder with the existing downstream voxel/scene contract.

No Concerto default data transforms are applied: the downstream 5 cm grid,
representative points, RGB, normals, and metric coordinates are retained.
Concerto's required nine-channel stem receives [XYZ, RGB, normal].
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from delimit3d.models.litept_wrapper import (
    LitePTBackbone, LitePTFeaturePyramid, LitePTPyramidLevel,
)
from delimit3d.training.concerto_contract import (
    ALL_FIVE_STAGE_RECIPE, NATIVE_STAGE_RECIPE, THREE_STAGE_RECIPE,
    DIRECT_INPUT_PROJECTION, concerto_feature_channels,
)

CHECKPOINT_SHA256 = "251441d92ac7d9cff9e0c3ae22bdf2dc416d1c9dbb7a75f1922c08bfd1a18868"
CONCERTO_COMMIT = "5b41117160cdd92c3538be1911277b17e7ef5871"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def component_counts(model):
    return {
        name: {
            "total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
        }
        for name in ("backbone", "adapters", "decoder")
        for module in (getattr(model, name),)
    }


def aggregate_encoder_features(features, parent_maps, *, recipe=THREE_STAGE_RECIPE):
    """Concatenate ancestor features at each stage using exact pooling maps.

    Inputs and outputs are ordered enc0..enc4. parent_maps[i] maps tokens at
    enc{i} to enc{i+1}; no scene or spatial pooling is introduced here.
    """
    concerto_feature_channels(recipe)
    if len(features) != 5 or len(parent_maps) != 4:
        raise ValueError("Concerto aggregation requires five stages and four parent maps")
    if recipe == NATIVE_STAGE_RECIPE:
        # Attention receives the untouched native stages. Only the finest
        # mask/click descriptor gathers all ancestors; this adds no weights.
        finest = features[-1]
        for index in range(3, -1, -1):
            finest = torch.cat((features[index], finest[parent_maps[index]]), dim=1)
        return (finest, *features[1:])
    fused = list(features)
    for index in range(3, -1, -1):
        ancestor = fused[index + 1][parent_maps[index]]
        if recipe == ALL_FIVE_STAGE_RECIPE or index >= 2:
            fused[index] = torch.cat((features[index], ancestor), dim=1)
        else:
            fused[index] = ancestor
    return tuple(fused)


class ConcertoBackbone(nn.Module):
    """Pure pretrained encoder; adapters deliberately live outside this module."""

    # Reuse precisely the existing downstream quantization and scene boundaries.
    _normalize_inputs = LitePTBackbone._normalize_inputs
    _normalize_offsets = LitePTBackbone._normalize_offsets
    _voxelize_single = LitePTBackbone._voxelize_single
    _voxelize_batched = LitePTBackbone._voxelize_batched
    _prepare_prequantized = LitePTBackbone._prepare_prequantized

    def __init__(self, *, source_root, checkpoint, frozen, activation_checkpointing=False,
                 grid_size=.05, feature_recipe=THREE_STAGE_RECIPE):
        super().__init__()
        self.feature_recipe = feature_recipe
        self.feature_channels = concerto_feature_channels(feature_recipe)
        if float(grid_size) not in (.02, .05):
            raise ValueError("Concerto downstream voxel size must be 2 cm or 5 cm")
        sys.path.insert(0, str(Path(source_root).resolve()))
        from concerto.model import PointTransformerV3

        observed = file_sha256(checkpoint)
        if observed != CHECKPOINT_SHA256:
            raise ValueError(f"Concerto checkpoint SHA256 mismatch: {observed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(payload["config"])
        if config["in_channels"] != 9 or not config["enc_mode"] or not config["traceable"]:
            raise ValueError("Unexpected pretrained Concerto interface")
        self.encoder = PointTransformerV3(**config)
        expected, incoming = self.encoder.state_dict(), payload["state_dict"]
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        incompatible = {
            name: {"expected": list(expected[name].shape), "supplied": list(value.shape)}
            for name, value in incoming.items()
            if name in expected and expected[name].shape != value.shape
        }
        self.load_report = {
            "file_sha256": observed, "source_commit": CONCERTO_COMMIT,
            "tensor_count": len(incoming), "missing_keys": missing,
            "unexpected_keys": unexpected, "incompatible_tensors": incompatible,
            "config": config, "strict": True,
        }
        if missing or unexpected or incompatible:
            raise RuntimeError(f"Pretrained encoder mismatch: {self.load_report}")
        self.encoder.load_state_dict(incoming, strict=True)
        if any(not torch.equal(expected[name], value) for name, value in incoming.items()):
            raise RuntimeError("Loaded encoder tensor values differ from supplied weights")
        self.in_channels = 6
        self.grid_size = float(grid_size)
        self.voxel_reduce = "representative"
        self.representative_sampling = "first"
        self.activation_checkpointing = bool(activation_checkpointing)
        self.frozen = bool(frozen)
        self.set_frozen(frozen)
        self.load_report["state_sha256"] = tensor_hash(self)
        self.load_report["normalization_modules"] = {
            "batchnorm": sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.modules()),
            "layernorm": sum(isinstance(m, nn.LayerNorm) for m in self.modules()),
        }

    def set_frozen(self, frozen):
        self.frozen = bool(frozen)
        for parameter in self.parameters():
            parameter.requires_grad_(not self.frozen)
        for module in self.encoder.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = not self.frozen
        self.train(self.training)

    def train(self, mode=True):
        super().train(False if self.frozen else mode)
        return self

    def _encode(self, coord, grid, feat, offsets):
        point = self.encoder({"coord": coord, "grid_coord": grid,
                              "feat": torch.cat((coord, feat), dim=1), "offset": offsets})
        descending = [point]
        while "pooling_parent" in point:
            point = point["pooling_parent"]
            descending.append(point)
        levels = list(reversed(descending))
        if len(levels) != 5:
            raise RuntimeError(f"Expected five Concerto levels, got {len(levels)}")
        parent_maps = tuple(p.pooling_inverse.long() for p in levels[1:])
        features = aggregate_encoder_features(
            [p.feat for p in levels], parent_maps, recipe=self.feature_recipe)
        if tuple(feature.shape[1] for feature in features) != self.feature_channels:
            raise RuntimeError("Concerto checkpoint feature widths differ from the selected recipe")
        return (features, tuple(p.grid_coord for p in levels),
                tuple(p.coord for p in levels), tuple(p.offset for p in levels),
                parent_maps)

    def forward(self, coord, feat, *, point_offsets=None, grid_coord=None):
        coord, feat = self._normalize_inputs(coord, feat)
        offsets = self._normalize_offsets(point_offsets, num_points=len(coord), device=coord.device)
        if grid_coord is None:
            data, inverse, _, _, representatives = self._voxelize_batched(coord, feat, offsets)
        else:
            data, inverse, _, _, representatives = self._prepare_prequantized(coord, feat, grid_coord, offsets)
        args = (data["coord"], data["grid_coord"], data["feat"], data["offset"])
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            if self.activation_checkpointing and self.training and not self.frozen:
                from torch.utils.checkpoint import checkpoint
                result = checkpoint(self._encode, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                result = self._encode(*args)
        features, grids, xyz, level_offsets, maps = result
        if not torch.equal(grids[0], data["grid_coord"]) or not torch.equal(level_offsets[0], data["offset"]):
            raise RuntimeError("Concerto changed the downstream finest token order")
        memory = tuple(LitePTPyramidLevel(
            name=f"concerto_enc{i}", stride=2**i, channels=features[i].shape[1],
            features=features[i], grid=grids[i], xyz=xyz[i], offsets=level_offsets[i],
        ) for i in (4, 3, 2, 1))
        return LitePTFeaturePyramid(
            finest_features=features[0], finest_grid=grids[0], finest_xyz=xyz[0],
            finest_offsets=level_offsets[0], memory_levels=memory,
            point_xyz=coord, point_offsets=offsets, original_to_finest=inverse,
            representative_indices=representatives, parent_maps=maps,
        )


class ConcertoMaskAdapters(nn.Module):
    """Preserve the existing decoder interface at every spatial level."""

    def __init__(self, seed=42, *, feature_recipe=THREE_STAGE_RECIPE):
        super().__init__()
        channels = concerto_feature_channels(feature_recipe)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.finest = nn.Linear(channels[0], 72)
            self.memory = nn.ModuleList(nn.Linear(src, dst) for src, dst in
                                        zip(channels[:0:-1], (504, 252, 144, 72)))

    def forward(self, pyramid):
        return replace(pyramid,
            finest_features=self.finest(pyramid.finest_features),
            memory_levels=tuple(replace(level, features=adapter(level.features), channels=adapter.out_features)
                                for level, adapter in zip(pyramid.memory_levels, self.memory, strict=True)))


def agile_adapter(seed=42, *, feature_recipe=THREE_STAGE_RECIPE):
    channels = concerto_feature_channels(feature_recipe)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Linear(channels[0], 72)


def configure_direct_decoder_inputs(decoder, *, task, feature_recipe, seed=42):
    """Resize only task input projections after strict loading of the old head.

    The attention, mask and objectness tensors remain byte-identical to the
    existing template. New projections initialize on CPU with an isolated RNG,
    then move to the decoder's device, also supporting GPU engine initialization.
    """
    if feature_recipe not in (ALL_FIVE_STAGE_RECIPE, NATIVE_STAGE_RECIPE):
        raise ValueError("Direct projections require an explicit all-five feature recipe")
    if task not in ("agile3d", "mask3d") or decoder.hidden_dim != 128:
        raise ValueError("Expected AGILE3D or Mask3D with hidden width 128")
    channels = concerto_feature_channels(feature_recipe)
    prefixes = (("input_adapter.",) if task == "agile3d" else
                ("mask_features_head.", "memory_projections."))
    shared = {name: value.detach().clone() for name, value in decoder.state_dict().items()
              if not name.startswith(prefixes)}
    reference = next(decoder.parameters())

    def linear(width):
        return nn.Linear(width, decoder.hidden_dim).to(
            device=reference.device, dtype=reference.dtype).train(decoder.training)

    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed would also reset CUDA generators, which this CPU
        # fork deliberately does not save. All new tensors initialize on CPU.
        torch.default_generator.manual_seed(seed)
        if task == "agile3d":
            decoder.input_adapter = linear(channels[0])
            decoder.feature_dim = channels[0]
            projections = [decoder.input_adapter]
        else:
            decoder.mask_features_head = linear(channels[0])
            decoder.memory_channels = tuple(reversed(channels[1:]))
            decoder.memory_projections = nn.ModuleList(linear(c) for c in decoder.memory_channels)
            projections = [decoder.mask_features_head, *decoder.memory_projections]
    current = decoder.state_dict()
    if any(not torch.equal(value, current[name]) for name, value in shared.items()):
        raise RuntimeError("Direct input initialization changed a shared task-decoder tensor")
    return {
        "input_projection": DIRECT_INPUT_PROJECTION, "seed": seed,
        "feature_recipe": feature_recipe, "finest_input": channels[0],
        "memory_input": list(reversed(channels[1:])) if task == "mask3d" else [],
        "hidden_width": decoder.hidden_dim,
        "projection_parameters": sum(p.numel() for m in projections for p in m.parameters()),
        "projection_state_sha256": tensor_hash(nn.ModuleList(projections)),
        "reinitialized_keys": sorted(name for name in current if name.startswith(prefixes)),
        "shared_tensors_preserved": len(shared),
    }


def gradient_report(model):
    report = {}
    for component in ("backbone", "adapters", "decoder"):
        module = getattr(model, component)
        nonzero, absent, squared_norm = [], [], 0.0
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                absent.append(name)
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Nonfinite gradient in {component}.{name}")
            norm = float(grad.float().norm().item())
            if norm > 0:
                nonzero.append(name)
            squared_norm += norm**2
        report[component] = {"nonzero_tensors": len(nonzero), "norm": squared_norm**0.5,
                             "absent": absent}
    return report
