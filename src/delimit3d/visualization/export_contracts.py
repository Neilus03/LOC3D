"""CPU-only, append-only saved-evaluation interfaces for future evaluator runs.

A bundle is ``record.json`` plus ``arrays.npz``. The writer receives already
computed outputs; it never runs a model or changes evaluator postprocessing.
Native token arrays and inverse maps survive alongside original point geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .contracts import Scene

EPISODE_SCHEMA = "delimit3d.agile3d_episode.v1"
PROPOSAL_SCHEMA = "delimit3d.mask3d_proposals.v1"


def geometry_hash(points: np.ndarray) -> str:
    array = np.ascontiguousarray(points)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def pairing_status(model: Mapping[str, Any], expected: Mapping[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Report incomplete records; reject contradictions instead of guessing pairing."""
    missing = []
    for role in ("encoder", "decoder"):
        identity = model.get(role, {})
        if not isinstance(identity, Mapping):
            raise ValueError(f"model.{role} must be an identity record")
        for key in ("checkpoint", "sha256"):
            if not identity.get(key):
                missing.append(f"model.{role}.{key}")
    trained_with = model.get("decoder_trained_with_encoder_sha256")
    if not trained_with:
        missing.append("model.decoder_trained_with_encoder_sha256")
    encoder_hash = model.get("encoder", {}).get("sha256")
    if trained_with and encoder_hash and trained_with != encoder_hash:
        raise ValueError("Decoder/encoder pairing mismatch in recorded checkpoint identities")
    preprocessing = model.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or not preprocessing:
        missing.append("model.preprocessing")
    if not model.get("preprocessing_sha256"):
        missing.append("model.preprocessing_sha256")
    elif preprocessing:
        digest = hashlib.sha256(json.dumps(json_value(preprocessing), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if model["preprocessing_sha256"] != digest:
            raise ValueError("Recorded preprocessing_sha256 differs from preprocessing record")
    def compare(actual: Mapping[str, Any], wanted: Mapping[str, Any], prefix: str) -> None:
        for key, value in wanted.items():
            if isinstance(value, Mapping):
                if not isinstance(actual.get(key), Mapping):
                    raise ValueError(f"Expected pairing field absent: {prefix}{key}")
                compare(actual[key], value, f"{prefix}{key}.")
            elif actual.get(key) != value:
                raise ValueError(f"Expected pairing mismatch: {prefix}{key}")
    if expected:
        compare(model, expected, "model.")
    return {"status": "incomplete" if missing else "record_complete", "missing": missing,
            "verification": "record consistency only; checkpoint bytes have not been checked by this validator"}


def validate_geometry(record: Mapping[str, Any], arrays: Mapping[str, np.ndarray], scene: Scene) -> None:
    if str(record.get("scene_id")) != scene.scene_id:
        raise ValueError("Export scene_id differs from scene")
    expected = geometry_hash(scene.points)
    if record.get("source_geometry_sha256") != expected:
        raise ValueError("Export ordered geometry hash differs from scene")
    if "points" in arrays:
        points = np.asarray(arrays["points"])
        if points.shape != scene.points.shape or not np.array_equal(points, scene.points):
            raise ValueError("Export points do not preserve original point correspondence")


def integer_array(value: Any, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != len(shape) or any(v is not None and array.shape[i] != v
                                       for i, v in enumerate(shape))
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    return array


def validate_inverse(arrays: Mapping[str, np.ndarray], n: int, native_count: int) -> np.ndarray:
    if "inverse_map" not in arrays:
        raise ValueError("Token/segment outputs require exact inverse_map")
    inverse = integer_array(arrays["inverse_map"], "inverse_map", (n,))
    if np.any(inverse < 0) or np.any(inverse >= native_count):
        raise ValueError("inverse_map contains out-of-range native point IDs")
    if "representative_indices" in arrays:
        reps = integer_array(arrays["representative_indices"], "representative_indices", (native_count,))
        if np.any(reps < 0) or np.any(reps >= n) or not np.array_equal(inverse[reps], np.arange(native_count)):
            raise ValueError("representative_indices do not invert inverse_map")
    return inverse


def read_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path).resolve(strict=True)
    record_path = source / "record.json" if source.is_dir() else source
    record = json.loads(record_path.read_text())
    array_file = record.get("arrays_file", "arrays.npz")
    if Path(array_file).name != array_file:
        raise ValueError("arrays_file must name an NPZ beside record.json")
    array_path = record_path.parent / array_file
    if record.get("arrays_sha256") and file_hash(array_path) != record["arrays_sha256"]:
        raise ValueError("Export arrays SHA256 mismatch")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    record["source_record"] = str(record_path)
    record["source_record_sha256"] = file_hash(record_path)
    return record, arrays


def _save_bundle(path: str | Path, scene: Scene, record: Mapping[str, Any],
                 arrays: Mapping[str, Any], schema: str) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite saved evaluation: {destination}")
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload.setdefault("points", np.asarray(scene.points))
    if any(value.dtype.hasobject for value in payload.values()):
        raise ValueError("Object arrays are forbidden in saved evaluation exports")
    manifest = deepcopy(json_value(record))
    if "scene_id" in manifest and str(manifest["scene_id"]) != scene.scene_id:
        raise ValueError("Supplied record scene_id differs from scene")
    if "source_geometry_sha256" in manifest and manifest["source_geometry_sha256"] != geometry_hash(scene.points):
        raise ValueError("Supplied record geometry hash differs from scene")
    manifest.update(schema=schema, scene_id=scene.scene_id,
                    source_geometry_sha256=geometry_hash(scene.points), arrays_file="arrays.npz")
    if schema == EPISODE_SCHEMA:
        from .episodes import episode_from_arrays
        episode_from_arrays(manifest, payload, scene)
    else:
        from .proposals import proposals_from_arrays
        proposals_from_arrays(manifest, payload, scene)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        np.savez_compressed(temporary / "arrays.npz", **payload)
        manifest["arrays_sha256"] = file_hash(temporary / "arrays.npz")
        (temporary / "record.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # mkdir provides an exclusive claim, including against concurrent writers.
        destination.mkdir()
        for filename in ("arrays.npz", "record.json"):
            (temporary / filename).rename(destination / filename)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination / "record.json"


def save_episode_export(path: str | Path, scene: Scene, *, record: Mapping[str, Any],
                        arrays: Mapping[str, Any]) -> Path:
    """Save exact evaluated frames/clicks. Omit predictions only for explicit trace-only exports."""
    return _save_bundle(path, scene, record, arrays, EPISODE_SCHEMA)


def save_proposal_export(path: str | Path, scene: Scene, *, record: Mapping[str, Any],
                         arrays: Mapping[str, Any]) -> Path:
    """Save native masks, every query/score, and evaluated settings before display resolution."""
    return _save_bundle(path, scene, record, arrays, PROPOSAL_SCHEMA)
