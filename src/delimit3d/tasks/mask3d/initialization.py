"""Strict, provenance-friendly initialization for LitePT -> Mask3D arms."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from delimit3d.tasks.mask3d.model import LitePTMask3DModel


_PRETRAINED_BACKBONE_SOURCES = ("structured3d", "scannet_sam2")
_BACKBONE_SOURCES = ("scratch", *_PRETRAINED_BACKBONE_SOURCES)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in stable key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _candidate_backbone_states(
    source: Mapping[str, torch.Tensor],
) -> list[OrderedDict[str, torch.Tensor]]:
    normalized = OrderedDict(
        (str(key).removeprefix("module."), value) for key, value in source.items()
    )
    candidates: list[OrderedDict[str, torch.Tensor]] = [normalized]
    for prefix in ("backbone.", "student.backbone."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            candidates.append(
                OrderedDict(
                    (key.removeprefix(prefix), value)
                    for key, value in normalized.items()
                )
            )
    candidates.extend(
        OrderedDict(
            (
                key if key.startswith("model.") else f"model.{key}",
                value,
            )
            for key, value in candidate.items()
        )
        for candidate in list(candidates)
    )
    return candidates


def _extract_backbone_state(checkpoint: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Backbone checkpoint must contain a mapping")
    if "backbone_state_dict" in checkpoint:
        state = checkpoint["backbone_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint['backbone_state_dict'] must be a mapping")
        return state, "backbone_state_dict"

    if "model_state_dict" in checkpoint:
        full_state = checkpoint["model_state_dict"]
        if not isinstance(full_state, Mapping):
            raise TypeError("checkpoint['model_state_dict'] must be a mapping")
        stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in full_state.items():
            normalized = str(key).removeprefix("module.")
            if normalized in stripped:
                raise RuntimeError(f"Duplicate normalized checkpoint key: {normalized}")
            if not torch.is_tensor(value):
                raise TypeError(f"Non-tensor checkpoint entry: {key}")
            stripped[normalized] = value
        backbone = OrderedDict(
            (key.removeprefix("backbone."), value)
            for key, value in stripped.items()
            if key.startswith("backbone.")
        )
        ignored = sorted(key for key in stripped if not key.startswith("backbone."))
        # Native contrastive scratch/pretraining checkpoints also contain this
        # learned loss scalar. It is not part of the encoder or task decoder.
        invalid_ignored = [
            key for key in ignored
            if not (key.startswith("projector.") or key == "criterion.log_temperature")
        ]
        if invalid_ignored:
            raise RuntimeError(
                "Dedicated backbone initialization refuses non-backbone state other "
                "than the contrastive projector and loss temperature: "
                f"{invalid_ignored[:20]}"
            )
        temperature = stripped.get("criterion.log_temperature")
        if temperature is not None and (
            temperature.numel() != 1 or not torch.isfinite(temperature).all()
        ):
            raise RuntimeError("Pretraining loss temperature must be one finite scalar")
        if not backbone:
            raise RuntimeError("model_state_dict contains no backbone.* tensors")
        return backbone, "model_state_dict.backbone"

    # Native LitePT training checkpoints store the complete LitePT tensor
    # mapping under ``model`` alongside optimizer/scheduler state.  Exact key
    # and tensor-shape matching below ensures no non-LitePT tensor can load.
    if "model" in checkpoint:
        state = checkpoint["model"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint['model'] must be a mapping")
        non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
        if non_tensors:
            raise TypeError(
                "checkpoint['model'] contains non-tensor entries: "
                f"{non_tensors[:20]}"
            )
        return state, "model"

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint, "plain_state_dict"
    raise KeyError(
        "Checkpoint must contain backbone_state_dict, model_state_dict, model, "
        "or be a plain tensor state dict"
    )


def load_litept_backbone_checkpoint(
    model: LitePTMask3DModel,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Strictly initialize only LitePT; no decoder tensor is accepted."""
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    source, source_field = _extract_backbone_state(_torch_load(path))
    expected = model.backbone.state_dict()
    matching: OrderedDict[str, torch.Tensor] | None = None
    for candidate in _candidate_backbone_states(source):
        if set(candidate) == set(expected):
            matching = candidate
            break
    if matching is None:
        observed_sets = [set(candidate) for candidate in _candidate_backbone_states(source)]
        best = min(observed_sets, key=lambda keys: len(keys ^ set(expected)))
        raise RuntimeError(
            "Pretrained LitePT backbone key mismatch: "
            f"missing={sorted(set(expected) - best)[:20]}, "
            f"unexpected={sorted(best - set(expected))[:20]}"
        )

    shape_mismatches = {
        key: (tuple(matching[key].shape), tuple(expected[key].shape))
        for key in expected
        if matching[key].shape != expected[key].shape
    }
    dtype_mismatches = {
        key: (str(matching[key].dtype), str(expected[key].dtype))
        for key in expected
        if matching[key].dtype != expected[key].dtype
    }
    if shape_mismatches or dtype_mismatches:
        raise RuntimeError(
            "Pretrained LitePT backbone tensor mismatch: "
            f"shapes={shape_mismatches}, dtypes={dtype_mismatches}"
        )
    load_result = model.backbone.load_state_dict(matching, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict backbone load unexpectedly failed: {load_result}")
    loaded = model.backbone.state_dict()
    if tensor_state_sha256(loaded) != tensor_state_sha256(matching):
        raise RuntimeError("Loaded LitePT backbone bytes differ from checkpoint bytes")
    return {
        "checkpoint": str(path),
        "source_field": source_field,
        "tensor_count": len(matching),
        "state_sha256": tensor_state_sha256(loaded),
        "strict": True,
        "decoder_tensors_loaded": 0,
    }


def save_decoder_template(
    decoder: nn.Module,
    output_path: str | Path,
    *,
    downstream_seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save every fresh decoder tensor as the paired-arm template."""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    state = OrderedDict(
        (key, value.detach().cpu().clone())
        for key, value in decoder.state_dict().items()
    )
    payload = {
        "schema": "litept_mask3d_decoder_template/v1",
        "downstream_seed": int(downstream_seed),
        "decoder_state_dict": state,
        "decoder_state_sha256": tensor_state_sha256(state),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output)
    return {
        "path": str(output.resolve()),
        "downstream_seed": int(downstream_seed),
        "tensor_count": len(state),
        "decoder_state_sha256": payload["decoder_state_sha256"],
    }


def load_decoder_template(
    model: LitePTMask3DModel,
    template_path: str | Path,
) -> dict[str, Any]:
    """Strictly load a paired fresh decoder and verify its stored hash."""
    path = Path(template_path).expanduser().resolve(strict=True)
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        "litept_mask3d_decoder_template/v1"
    ):
        raise RuntimeError(f"Not a LitePT-Mask3D decoder template: {path}")
    state = payload.get("decoder_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("decoder_state_dict is missing from decoder template")
    expected_hash = str(payload.get("decoder_state_sha256", ""))
    observed_hash = tensor_state_sha256(state)
    if observed_hash != expected_hash:
        raise RuntimeError("Decoder template tensor hash does not match its manifest")
    result = model.decoder.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict decoder template load failed: {result}")
    loaded_hash = model.decoder_state_hash()
    if loaded_hash != expected_hash:
        raise RuntimeError("Loaded decoder differs byte-for-byte from the template")
    return {
        "path": str(path),
        "downstream_seed": int(payload["downstream_seed"]),
        "tensor_count": len(state),
        "decoder_state_sha256": loaded_hash,
        "strict": True,
    }


def initialize_litept_mask3d_arm(
    model: LitePTMask3DModel,
    *,
    decoder_template: str | Path,
    backbone_source: str,
    backbone_checkpoint: str | Path | None = None,
    freeze_backbone: bool = False,
) -> dict[str, Any]:
    """Initialize one matched arm without ever accepting official Mask3D weights."""
    source = str(backbone_source)
    if source not in _BACKBONE_SOURCES:
        raise ValueError(
            f"backbone_source must be one of {_BACKBONE_SOURCES}, got {source!r}"
        )
    if source == "scratch" and backbone_checkpoint is not None:
        raise ValueError("scratch backbone arm must not provide a checkpoint")
    if source in _PRETRAINED_BACKBONE_SOURCES and backbone_checkpoint is None:
        raise ValueError(f"{source} backbone arm requires a checkpoint")

    decoder_report = load_decoder_template(model, decoder_template)
    backbone_report = None
    if source in _PRETRAINED_BACKBONE_SOURCES:
        backbone_report = load_litept_backbone_checkpoint(
            model, backbone_checkpoint  # type: ignore[arg-type]
        )
    model.set_backbone_frozen(freeze_backbone)
    return {
        "backbone_source": source,
        "backbone_frozen": bool(freeze_backbone),
        "backbone_state_sha256": model.backbone_state_hash(),
        "decoder_state_sha256": model.decoder_state_hash(),
        "decoder_template": decoder_report,
        "backbone_checkpoint": backbone_report,
        "official_mask3d_weights_loaded": False,
    }


__all__ = [
    "initialize_litept_mask3d_arm",
    "load_decoder_template",
    "load_litept_backbone_checkpoint",
    "save_decoder_template",
    "tensor_state_sha256",
]
