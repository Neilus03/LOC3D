"""Pretraining records extracted from the executed trainer."""

from __future__ import annotations


import hashlib

import json


import os

import random


from pathlib import Path

from tempfile import NamedTemporaryFile

from typing import Any, Mapping, Sequence

import numpy as np

import torch


from delimit3d.training.adaptation import LitePTContrastiveModel


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        temp_path.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _content_addressed_report(value: Any) -> Any:
    """Drop extraction-location fields while preserving identity hashes."""

    if isinstance(value, dict):
        return {
            key: _content_addressed_report(item)
            for key, item in value.items()
            if key != "path"
            and key != "pack_dir"
            and not key.endswith("_path")
            and not key.endswith("_root")
        }
    if isinstance(value, list):
        return [_content_addressed_report(item) for item in value]
    return value


def _semantic_config_sha256(resolved: dict[str, Any]) -> str:
    """Hash every semantic field while excluding attempt-local lineage."""

    payload = _content_addressed_report(resolved)
    payload.pop("semantic_config_sha256", None)
    payload.pop("resume_parent", None)
    payload.pop("resume_into_fresh_attempt", None)
    payload.pop("lineage_parent_checkpoints", None)
    payload.pop("pause_after_update", None)
    payload.pop("execution_stop_update", None)
    payload.pop("executed_scene_visits_per_rank", None)
    execution = payload.get("execution_provenance")
    if isinstance(execution, dict):
        payload["execution_provenance"] = {
            "schema_version": execution.get("schema_version"),
            "trust_root": execution.get("trust_root"),
            "passed": execution.get("passed"),
        }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_jsonl_noclobber(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        os.link(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def _json_copy(value: Any) -> Any:
    """Copy JSON telemetry without retaining mutable controller references."""

    return json.loads(json.dumps(value))


def _tensor_mapping_sha256(
    state: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """Content hash tensor values without depending on torch.save serialization."""

    selected = tuple(sorted(state if keys is None else keys))
    digest = hashlib.sha256()
    for key in selected:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batchnorm_state_report(model: LitePTContrastiveModel) -> dict[str, Any]:
    state = model.state_dict()
    keys = tuple(
        sorted(
            key
            for key in state
            if key.endswith(".running_mean")
            or key.endswith(".running_var")
            or key.endswith(".num_batches_tracked")
        )
    )
    return {
        "schema_version": "litept_batchnorm_state/v1",
        "key_count": len(keys),
        "keys": list(keys),
        "sha256": _tensor_mapping_sha256(state, keys=keys),
    }


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture all mutable per-rank generators used by the runner."""

    return {
        "schema_version": "litept_pretraining_rng_state/v2",
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if state.get("schema_version") != "litept_pretraining_rng_state/v2":
        raise ValueError("Resume RNG-state schema drift")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=device)
