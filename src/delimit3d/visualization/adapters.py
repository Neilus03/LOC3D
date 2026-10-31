"""Dataset and model adapters for full-resolution, order-checked visualization.

All configured paths are relative to the manifest. NPZ and NumPy paths need
only NumPy; PyTorch, LitePT and PLY dependencies are imported on demand. Native
AGILE3D support refers to this repository's compatible LitePT click decoder.
Official AGILE3D and Mask3D environments can expose a ``module:function``
callable accepting ``scene=Scene, config=dict`` and returning an Observation or
its field dictionary, with arrays in the original scene point order.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Observation, Scene


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Expand environment/home variables, then resolve against the manifest."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve(strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_sha256(points: np.ndarray) -> str:
    """Hash ordered geometry in canonical little-endian float32 representation."""
    return hashlib.sha256(np.asarray(points, dtype="<f4", order="C").tobytes()).hexdigest()


def _scaled_geometry(value: Any, scale: float) -> np.ndarray:
    """Keep stored floating precision so exact cache witnesses remain valid."""
    points = _array(value)
    if (not np.issubdtype(points.dtype, np.number) or
            np.issubdtype(points.dtype, np.complexfloating)):
        raise ValueError("Geometry must contain real numeric coordinates")
    return points.astype(np.result_type(points.dtype, np.float32), copy=False) * scale


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _field(payload: Mapping[str, Any], selector: Any) -> Any:
    if selector is None:
        return None
    if isinstance(selector, (list, tuple)):
        return np.column_stack([_array(payload[str(key)]) for key in selector])
    if isinstance(selector, Mapping):
        value = _array(payload[str(selector["key"])])
        return value[:, selector["columns"]] if "columns" in selector else value
    # Dotted keys support nested exported feature dictionaries.
    if str(selector) in payload:
        return payload[str(selector)]
    value: Any = payload
    for part in str(selector).split("."):
        value = value[part]
    return value


def _optional(payload: Mapping[str, Any], fields: Mapping[str, Any], name: str) -> Any:
    selector = fields.get(name, name)
    if name not in fields and name not in payload:
        return None
    return _field(payload, selector)


def _torch_load(path: Path) -> Mapping[str, Any]:
    # PyTorch archives execute pickle code: these paths must be trusted local
    # experiment artifacts. NumPy archives never enable allow_pickle.
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older pinned experiment runtimes predate weights_only.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a mapping in the trusted torch archive")
    return value


def _read_payload(path: Path, kind: str | None = None) -> tuple[dict[str, Any], list[Path]]:
    kind = kind or ("directory" if path.is_dir() else path.suffix.lstrip(".").lower())
    if kind == "npz":
        with np.load(path, allow_pickle=False) as value:
            return {key: value[key] for key in value.files}, [path]
    if kind == "npy":
        return {"data": np.load(path, allow_pickle=False)}, [path]
    if kind in {"directory", "training_pack"}:
        files = sorted(path.glob("*.npy"))
        if not files:
            raise ValueError(f"{path}: directory has no .npy arrays")
        return {file.stem: np.load(file, allow_pickle=False) for file in files}, files
    if kind == "ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("PLY scenes require the optional plyfile package") from exc
        vertex = PlyData.read(str(path))["vertex"].data
        return {name: np.asarray(vertex[name]) for name in vertex.dtype.names}, [path]
    if kind in {"pt", "pth", "torch"}:
        return dict(_torch_load(path)), [path]
    raise ValueError(f"Unsupported array format {kind!r}: {path}")


def _colors(value: Any, declared_range: str | None) -> np.ndarray:
    color = _array(value)
    if declared_range is None:
        if color.dtype == np.uint8:
            declared_range = "0_255"
        elif np.issubdtype(color.dtype, np.floating) and np.all((color >= 0) & (color <= 1)):
            declared_range = "0_1"
        else:
            raise ValueError("Declare color_range as 0_1, 0_255 or -1_1 for noncanonical RGB")
    limits = {"0_1": (0.0, 1.0), "0_255": (0.0, 255.0), "-1_1": (-1.0, 1.0)}
    if declared_range not in limits:
        raise ValueError(f"Unsupported color_range {declared_range!r}")
    low, high = limits[declared_range]
    color = color.astype(np.float32)
    if not np.isfinite(color).all() or np.any(color < low) or np.any(color > high):
        raise ValueError(f"RGB values lie outside declared color_range {declared_range}")
    return (color - low) / (high - low)


def _label_ids(value: Any, name: str) -> np.ndarray:
    array = _array(value)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name} must contain finite integer label IDs")
        if np.any(array < np.iinfo(np.int64).min) or np.any(array >= 2**63):
            raise ValueError(f"{name} label IDs do not fit int64")
    return array.astype(np.int64, copy=False)


def load_scene(spec: dict[str, Any], base_dir: Path) -> Scene:
    """Load one dataset using explicit fields, without axis or unit guessing.

    ``fields`` maps canonical names (points, colors, normals, labels,
    semantic_labels) to archive keys, PLY property-name lists, or
    ``{key: data, columns: [0,1,2]}`` selectors for packed NPY files.
    Directory defaults are canonical names, except ``points`` falls back to
    the repository's unambiguous ``coords.npy`` training-pack convention.
    """
    path = resolve_path(spec["path"], base_dir)
    kind = spec.get("kind", spec.get("format"))
    payload, files = _read_payload(path, kind)
    fields = dict(spec.get("fields", {}))
    if "points" not in fields and "points" not in payload and "coords" in payload:
        fields["points"] = "coords"
    scale = float(spec.get("coordinate_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    points = _scaled_geometry(_field(payload, fields.get("points", "points")), scale)
    optional = {name: _optional(payload, fields, name) for name in (
        "colors", "normals", "labels", "semantic_labels",
    )}
    color = optional.pop("colors")
    values = {name: None if value is None else _array(value) for name, value in optional.items()}
    for name in ("labels", "semantic_labels"):
        if values[name] is not None:
            values[name] = _label_ids(values[name], name)
    metadata = dict(spec.get("metadata", {}))
    metadata.update({
        "source_path": str(path), "dataset": spec.get("dataset", "custom"),
        "source_files": {str(file): sha256_file(file) for file in files},
        "geometry_sha256": geometry_sha256(points), "coordinate_scale": scale,
        "fields": fields, "color_range": spec.get("color_range"),
    })
    parent_value = _optional(payload, fields, "part_parent_labels")
    if parent_value is not None:
        parents = _label_ids(parent_value, "part_parent_labels")
        if parents.shape != (len(points),):
            raise ValueError("part_parent_labels must align with original scene points")
        metadata["part_parent_labels"] = parents
    for key in ("background_ids", "ignore_label_ids"):
        if key in spec:
            metadata[key] = list(spec[key])
    scene = Scene(
        scene_id=str(spec.get("scene_id", path.stem)), points=points,
        colors=None if color is None else _colors(color, spec.get("color_range")),
        metadata=metadata, **values,
    )
    scene.validate()
    return scene


def _ordered_geometry(
    payload: Mapping[str, Any], spec: Mapping[str, Any], scene: Scene, base_dir: Path,
) -> str:
    """Require a complete ordered-geometry witness before accepting a cache."""
    fields = spec.get("fields", {})
    cached_scene = _scalar(payload.get("scene_id", payload.get("scene")))
    if cached_scene is not None and str(cached_scene) != scene.scene_id:
        raise ValueError(f"Cache scene {cached_scene!r} differs from {scene.scene_id!r}")
    raw = _optional(payload, fields, "points")
    # Historical caches use points as a scalar count, and store shifted_points.
    if raw is not None and _array(raw).ndim == 0:
        raw = None
    shifted = False
    if raw is None and "shifted_points" in payload and "shift" in payload:
        raw = _array(payload["shifted_points"])
        shift = _array(payload["shift"])
        if shift.shape != (3,) or not np.isfinite(shift).all():
            raise ValueError("Cache shift must be a finite length-three vector")
        expected = np.asarray(scene.points, dtype=np.float32) - shift.astype(np.float32)
        shifted = True
    else:
        expected = scene.points
    if raw is None and "geometry_path" in spec:
        witness, _ = _read_payload(resolve_path(spec["geometry_path"], base_dir))
        raw = _field(witness, spec.get("geometry_field", "points"))
    if raw is not None:
        raw = _array(raw)
        tolerance = float(spec.get("geometry_atol", 0.0))
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("geometry_atol must be finite and nonnegative")
        if raw.shape != expected.shape or not np.allclose(raw, expected, rtol=0, atol=tolerance):
            raise ValueError("Cache geometry/order differs from the full-resolution scene")
        return "shifted_points" if shifted else "ordered_points"
    expected_hash = _scalar(payload.get("source_geometry_sha256"))
    if expected_hash is not None and str(expected_hash) == geometry_sha256(scene.points):
        return "source_geometry_sha256"
    # Existing caches can carry a source-file hash instead of all raw vertices.
    source_hash = _scalar(payload.get("source_data_sha256"))
    hashes = scene.metadata.get("source_files", {}).values()
    if source_hash is not None and str(source_hash) in hashes:
        return "source_data_sha256"
    raise ValueError(
        "Cache requires matching ordered points, shifted_points+shift, "
        "source_geometry_sha256, matching source_data_sha256, or geometry_path; "
        "equal point counts and representative vertices do not establish point order"
    )


def _integer_indices(value: Any, name: str, count: int, upper: int) -> np.ndarray:
    result = _array(value)
    if result.shape != (count,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array of shape [{count}]")
    result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} contains indices outside [0, {upper})")
    return result


def _cache_data(spec: dict[str, Any], scene: Scene, base_dir: Path) -> dict[str, Any]:
    path = resolve_path(spec["path"], base_dir)
    payload, _ = _read_payload(path, spec.get("format"))
    alignment = _ordered_geometry(payload, spec, scene, base_dir)
    fields = spec.get("fields", {})
    selectors = spec.get("features")
    stage_spec = spec.get("feature_stages")
    if stage_spec is not None and selectors is not None:
        raise ValueError("Choose features or feature_stages, not both")
    if selectors is None:
        value = payload.get("features")
        selectors = ({str(key): f"features.{key}" for key in value}
                     if isinstance(value, Mapping) else ({"dec0": "features"} if value is not None else {}))
    if not isinstance(selectors, Mapping):
        raise ValueError("features must map level names to cache array keys")
    features = {str(name): _array(_field(payload, key)) for name, key in selectors.items()}
    if stage_spec is not None:
        from .stages import split_feature_stages
        if not isinstance(stage_spec, Mapping):
            raise ValueError("feature_stages must be a mapping")
        features = split_feature_stages(
            _array(_field(payload, stage_spec.get("key", "features"))),
            stage_spec.get("widths"),
            normalization=stage_spec.get("normalization", "as_saved"))
    inverse_value = _optional(payload, fields, "inverse_map")
    token_xyz = _optional(payload, fields, "scene_xyz")
    inverse = None
    token_count = None
    if inverse_value is not None:
        # Infer token count from an actual token array, never inverse.max()+1.
        if token_xyz is not None:
            token_count = len(_array(token_xyz))
        elif features:
            token_count = len(next(iter(features.values())))
        elif "token_count" in spec:
            token_count = int(spec["token_count"])
        else:
            raise ValueError("A token cache needs scene_xyz, features, or an explicit token_count")
        inverse = _integer_indices(inverse_value, "inverse_map", len(scene.points), token_count)
    reps_value = _optional(payload, fields, "representative_indices")
    if reps_value is not None:
        if inverse is None:
            raise ValueError("representative_indices requires inverse_map")
        reps = _integer_indices(reps_value, "representative_indices", token_count, len(scene.points))
        if len(np.unique(reps)) != len(reps) or not np.array_equal(inverse[reps], np.arange(token_count)):
            raise ValueError("representative_indices does not invert inverse_map in token order")
    aligned_features = {}
    for level, value in features.items():
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"Feature level {level!r} must be a finite matrix")
        expected = token_count if inverse is not None else len(scene.points)
        if len(value) != expected:
            raise ValueError(f"Feature level {level!r} has {len(value)} rows, expected {expected}")
        aligned_features[level] = value[inverse] if inverse is not None else value
    aligned: dict[str, Any] = {}
    for name in ("prediction", "confidence", "steps"):
        value = _optional(payload, fields, name)
        if value is None:
            aligned[name] = None
            continue
        value = _array(value)
        point_axis = 1 if name == "steps" else 0
        required_ndim = 2 if name == "steps" else 1
        expected = token_count if inverse is not None else len(scene.points)
        if value.ndim != required_ndim or value.shape[point_axis] != expected:
            raise ValueError(f"Cache {name} does not align with {'tokens' if inverse is not None else 'points'}")
        aligned[name] = np.take(value, inverse, axis=point_axis) if inverse is not None else value
    checkpoint_hash = _scalar(payload.get("checkpoint_sha256"))
    expected_hash = spec.get("expected_checkpoint_sha256")
    if expected_hash is not None and str(checkpoint_hash) != str(expected_hash):
        raise ValueError("Cache encoder checkpoint hash mismatch")
    metadata = {
        "adapter": "cache", "cache_path": str(path), "cache_sha256": sha256_file(path),
        "alignment_check": alignment, "source_geometry_sha256": geometry_sha256(scene.points),
        "checkpoint_sha256": checkpoint_hash, "token_count": token_count,
        "token_ids": None if inverse is None else {level: inverse for level in features},
        "confidence_semantics": spec.get("confidence_semantics", spec.get("metadata", {}).get("confidence_semantics", "exported model score")),
        "confidence_calibrated": bool(spec.get("confidence_calibrated", spec.get("metadata", {}).get("confidence_calibrated", False))),
    }
    for key in ("wrapper_settings", "preprocessing", "repo_commit", "schema", "arm",
                "encoder_tensor_sha256", "encoder_checkpoint_sha256", "epoch", "checkpoint_epoch", "update", "feature_coordinate_system"):
        if key in payload:
            metadata[key] = _scalar(payload[key])
    if stage_spec is not None:
        metadata["feature_stages"] = {
            "key": stage_spec.get("key", "features"), "widths": dict(stage_spec["widths"]),
            "normalization": stage_spec.get("normalization", "as_saved"),
            "source_normalization": stage_spec.get("source_normalization", "unspecified; inspect export provenance"),
            "pca_domain": "original point rows after inverse-map expansion, with the configured fit sample",
        }
    if "experiment_record_json" in payload:
        import json
        source_record = json.loads(str(_scalar(payload["experiment_record_json"])))
        if not isinstance(source_record, dict):
            raise ValueError("experiment_record_json must decode to an object")
        metadata["experiment_record"] = source_record
        for key in ("preprocessing", "epoch", "feature_coordinate_system"):
            if key not in metadata and key in source_record:
                metadata[key] = source_record[key]
    metadata["artifact_identity"] = {"scene_id": scene.scene_id, **{
        key: metadata[key] for key in ("checkpoint_sha256", "encoder_checkpoint_sha256", "epoch", "preprocessing")
        if metadata.get(key) is not None}}
    return dict(features=aligned_features, raw_features=features, inverse=inverse,
                token_xyz=None if token_xyz is None else _array(token_xyz),
                metadata=metadata, **aligned)


def _load_cache(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    values = _cache_data(spec, scene, base_dir)
    return Observation(
        name=str(spec.get("name", "cached")), features=values["features"],
        prediction=values["prediction"], confidence=values["confidence"], steps=values["steps"],
        clicks=list(spec.get("clicks", [])), metadata=values["metadata"],
    )


def _checkpoint_state(payload: Mapping[str, Any], state_key: str | None = None) -> Mapping[str, Any]:
    if state_key is not None:
        state = _field(payload, state_key)
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint state_key {state_key!r} is not a mapping")
        return state
    # V2 checkpoints may duplicate backbone weights; model_state_dict contains
    # authoritative recalibrated BatchNorm buffers.
    for key in ("model_state_dict", "backbone_state_dict", "encoder_state_dict", "state_dict", "model"):
        if isinstance(payload.get(key), Mapping):
            return payload[key]
    return payload


def _strict_backbone_state(source: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Map public, wrapper, and whole-model checkpoints without key collisions."""
    result = {}
    for source_key, value in source.items():
        if not hasattr(value, "shape"):
            continue
        key = str(source_key).removeprefix("module.")
        prefixes = ("model.backbone.model.", "model.backbone.", "backbone.model.",
                    "backbone.", "encoder.model.", "encoder.")
        explicitly_backbone = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = "model." + key[len(prefix):]
                explicitly_backbone = True
                break
        if key not in expected and "model." + key in expected:
            key = "model." + key
        if key not in expected:
            if explicitly_backbone or key.startswith("model."):
                raise ValueError(f"Unexpected backbone tensor {source_key!r}")
            continue  # Whole-model heads are outside the encoder contract.
        if key in result:
            raise ValueError(f"Duplicate checkpoint tensors map to {key!r}")
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"Checkpoint shape mismatch for {key!r}")
        result[key] = value
    missing = set(expected) - set(result)
    if missing:
        raise ValueError(f"Checkpoint misses {len(missing)} backbone tensors: {sorted(missing)[:8]}")
    return result


def _load_litept(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    import torch

    from delimit3d.models.litept_wrapper import LitePTBackbone

    checkpoint = resolve_path(spec["checkpoint"], base_dir)
    checkpoint_hash = sha256_file(checkpoint)
    if spec.get("expected_checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("LitePT checkpoint hash mismatch")
    if "grid_size" not in spec or "rgb_mean" not in spec or "rgb_std" not in spec:
        raise ValueError("LitePT requires explicit grid_size, rgb_mean and rgb_std matching training")
    mode = spec.get("input_features", "rgbn6")
    if mode not in {"rgb3", "rgbn6"}:
        raise ValueError("input_features must be rgb3 or rgbn6")
    if scene.colors is None or (mode == "rgbn6" and scene.normals is None):
        raise ValueError(f"LitePT {mode} requires RGB" + (" and normals" if mode == "rgbn6" else ""))
    mean = np.asarray(spec["rgb_mean"], dtype=np.float32)
    std = np.asarray(spec["rgb_std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("rgb_mean/rgb_std must be finite length-three arrays with positive std")
    grid_size = float(spec["grid_size"])
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("grid_size must be finite and positive")
    features = (scene.colors - mean) / std
    if mode == "rgbn6":
        normals = np.asarray(scene.normals, dtype=np.float32)
        if spec.get("normalize_normals", True):
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        features = np.concatenate([features, normals], axis=1)
    points = np.asarray(scene.points, dtype=np.float32).copy()
    shift_mode = spec.get("center_shift", "none")
    shift = np.zeros(3, dtype=np.float32)
    if shift_mode == "center_xy_ground_z":
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        shift = np.array([(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]], dtype=np.float32)
        points -= shift
    elif shift_mode != "none":
        raise ValueError("center_shift must be none or center_xy_ground_z")
    levels = list(spec.get("levels", ["dec0"]))
    if not levels or not set(levels) <= {"dec0", "dec1", "dec2", "dec3", "enc4"}:
        raise ValueError("levels must select dec0, dec1, dec2, dec3, enc4")
    capture = any(level != "dec0" for level in levels)
    device = torch.device(spec.get("device", "cuda"))
    torch.manual_seed(int(spec.get("seed", 0)))
    model = LitePTBackbone(
        litept_root=str(resolve_path(spec["litept_root"], base_dir)),
        in_channels=3 if mode == "rgb3" else 6, grid_size=grid_size,
        litept_variant=spec.get("litept_variant", "litept_s_star"),
        litept_kwargs=spec.get("litept_kwargs"), voxel_reduce=spec.get("voxel_reduce", "representative"),
        representative_sampling="first", cache_training_voxelization=False,
        capture_hierarchy=capture,
    )
    payload = _torch_load(checkpoint)
    state = _checkpoint_state(payload, spec.get("state_key"))
    model.load_state_dict(_strict_backbone_state(state, model.state_dict()), strict=True)
    model.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        output = model(torch.as_tensor(points, device=device),
                       torch.as_tensor(features, dtype=torch.float32, device=device))
    result = {"dec0": _array(output.point_feat)}
    token_ids = {"dec0": _array(output.inverse_map)}
    if capture:
        level_map = dict(zip(output.hierarchy_stage_names, output.hierarchy_tokens))
        point_to_level = output.inverse_map
        for level, parent_map in zip(("dec1", "dec2", "dec3", "enc4"), output.hierarchy_parent_maps):
            point_to_level = parent_map[point_to_level]
            if level in levels:
                result[level] = _array(level_map[level][point_to_level])
                token_ids[level] = _array(point_to_level)
    return Observation(name=str(spec.get("name", "LitePT")),
                       features={level: result[level] for level in levels}, metadata={
        "adapter": "litept", "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "source_geometry_sha256": geometry_sha256(scene.points),
        "preprocessing": {"input_features": mode, "grid_size": grid_size,
            "rgb_mean": mean.tolist(), "rgb_std": std.tolist(), "center_shift": shift_mode,
            "shift": shift.tolist(), "normalize_normals": spec.get("normalize_normals", True),
            "voxel_reduce": spec.get("voxel_reduce", "representative"),
            "representative_sampling": "first", "seed": int(spec.get("seed", 0))},
        "litept_variant": spec.get("litept_variant", "litept_s_star"),
        "levels": levels, "inference_points": len(points),
        "token_ids": {level: token_ids[level] for level in levels},
    })


def _load_agile3d(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    import torch

    from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder, state_sha256

    cache = _cache_data(spec["cache"], scene, base_dir)
    level = str(spec.get("level", "dec0"))
    if cache["inverse"] is None or cache["token_xyz"] is None:
        raise ValueError("AGILE3D requires cached token features, scene_xyz and inverse_map")
    token_features = cache["raw_features"][level]
    token_xyz = cache["token_xyz"]
    if token_xyz.shape != (len(token_features), 3) or not np.isfinite(token_xyz).all():
        raise ValueError("AGILE3D scene_xyz must be finite and aligned with tokens")
    checkpoint = resolve_path(spec["checkpoint"], base_dir)
    checkpoint_hash = sha256_file(checkpoint)
    if spec.get("expected_checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("AGILE3D decoder checkpoint hash mismatch")
    payload = _torch_load(checkpoint)
    if "arm" in spec and payload.get("arm") != spec["arm"]:
        raise ValueError("AGILE3D decoder arm differs from the configured arm")
    pairing_checks = []
    for key in ("encoder_tensor_sha256", "encoder_checkpoint_sha256", "epoch", "checkpoint_epoch", "update", "feature_coordinate_system"):
        encoder_hash = cache["metadata"].get(key)
        if key == "encoder_checkpoint_sha256" and encoder_hash is None:
            encoder_hash = cache["metadata"].get("checkpoint_sha256")
        if payload.get(key) is not None and encoder_hash is not None:
            if str(payload[key]) != str(encoder_hash):
                raise ValueError(f"AGILE3D cache/decoder encoder mismatch: {key}")
            pairing_checks.append(key)
    if cache["metadata"].get("arm") is not None and payload.get("arm") is not None:
        if cache["metadata"]["arm"] != payload["arm"]:
            raise ValueError("AGILE3D cache/decoder arm mismatch")
    kwargs = dict(payload.get("decoder_kwargs", spec.get("decoder_kwargs", {})))
    if "decoder_kwargs" in spec and kwargs != spec["decoder_kwargs"]:
        raise ValueError("AGILE3D decoder_kwargs differ from checkpoint")
    device = torch.device(spec.get("device", "cpu"))
    decoder = Agile3DClickDecoder(**kwargs)
    state = _checkpoint_state(payload, spec.get("state_key", "decoder_state_dict"))
    decoder.load_state_dict(state, strict=True)
    if payload.get("decoder_tensor_sha256") is not None:
        if state_sha256(decoder.state_dict()) != payload["decoder_tensor_sha256"]:
            raise ValueError("AGILE3D decoder tensor hash mismatch")
    decoder.to(device).eval().requires_grad_(False)
    events = [dict(click) for click in spec.get("clicks", [])]
    if not events:
        raise ValueError("AGILE3D requires explicit clicks [{point_index: ..., object_id: 1}, ...]")
    features = torch.as_tensor(token_features, dtype=torch.float32, device=device)
    xyz = torch.as_tensor(token_xyz, dtype=torch.float32, device=device)
    clicks: dict[int, list[int]] = {0: []}
    times: dict[int, list[int]] = {0: []}
    steps = []
    inverse = cache["inverse"]
    with torch.inference_mode():
        for time, event in enumerate(events):
            point = event.get("point_index")
            object_id = event.get("object_id")
            if not isinstance(point, int) or not 0 <= point < len(scene.points):
                raise ValueError("Each click point_index must be an in-range original scene index")
            if not isinstance(object_id, int) or object_id < 0:
                raise ValueError("Click object_id must be nonnegative; 0 is background")
            token = int(inverse[point])
            # A later click on a token replaces an earlier conflicting label.
            for label in clicks:
                pairs = [(index, stamp) for index, stamp in zip(clicks[label], times[label]) if index != token]
                clicks[label] = [pair[0] for pair in pairs]
                times[label] = [pair[1] for pair in pairs]
            clicks.setdefault(object_id, []).append(token)
            times.setdefault(object_id, []).append(time)
            active = sorted(label for label, ids in clicks.items() if label > 0 and ids)
            if not active:
                raise ValueError("The first AGILE3D click must select a foreground object")
            label_map = [0] + active
            dense_clicks = {new: clicks[old] for new, old in enumerate(label_map)}
            dense_times = {new: times[old] for new, old in enumerate(label_map)}
            logits = decoder(features, xyz, clicks=dense_clicks, click_times=dense_times)["pred_masks"]
            probabilities = logits.softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            predicted = np.asarray(label_map)[_array(prediction)]
            if spec.get("enforce_clicks", True):
                for label, ids in clicks.items():
                    predicted[ids] = label
            steps.append(predicted[inverse])
    metadata = dict(cache["metadata"])
    metadata.update({"adapter": "agile3d_compatible", "decoder_checkpoint": str(checkpoint),
                     "decoder_checkpoint_sha256": checkpoint_hash,
                     "decoder_kwargs": kwargs, "enforce_clicks": spec.get("enforce_clicks", True),
                     "encoder_pairing_checks": pairing_checks,
                     "prediction_background_ids": [0],
                     "confidence_semantics": "maximum softmax score before click enforcement",
                     "confidence_calibrated": False})
    return Observation(name=str(spec.get("name", "AGILE3D-compatible")),
                       features=cache["features"], prediction=steps[-1],
                       confidence=_array(confidence)[inverse], steps=np.stack(steps),
                       clicks=events, metadata=metadata)


def observation_from_mask3d_output(
    output: Mapping[str, Any], scene: Scene, *, name: str = "Mask3D",
    inverse_map: Any = None, features: Mapping[str, Any] | None = None,
    mask_key: str = "pred_masks", class_key: str = "pred_logits",
    mask_layout: str = "NQ", mask_threshold: float = 0.5,
    class_threshold: float = 0.0, no_object_index: int = -1,
) -> Observation:
    """Convert real Mask3D query logits to point-aligned visualization arrays.

    Official outputs use ``pred_masks`` [N,Q] (or one-item list) and
    ``pred_logits`` [Q,C+1] (or [1,Q,C+1]). CHORUS outputs use ``mask_logits``
    [Q,N], ``object_logits`` [Q,C+1]: set those keys and ``mask_layout='QN'``.
    An inverse map expands voxel/segment predictions to original points.

    This is an explicit visualization policy, not benchmark postprocessing:
    drop queries whose top class is no-object; keep points above the mask
    threshold; resolve overlap by sigmoid(mask)*softmax(class). Instance IDs
    retain original query index + 1; 0 means unassigned. Scores are uncalibrated.
    Feature arrays, if supplied, must already use original point order.
    """
    scene.validate()
    masks = output[mask_key]
    if isinstance(masks, (list, tuple)):
        if len(masks) != 1:
            raise ValueError("Mask3D bridge accepts exactly one scene at a time")
        masks = masks[0]
    masks = _array(masks)
    classes = _array(output[class_key])
    if classes.ndim == 3 and classes.shape[0] == 1:
        classes = classes[0]
    if masks.ndim != 2 or classes.ndim != 2 or classes.shape[1] < 2:
        raise ValueError("Mask3D expects 2D mask logits and [Q,C+1] class logits")
    if mask_layout == "QN":
        masks = masks.T
    elif mask_layout != "NQ":
        raise ValueError("mask_layout must be explicitly NQ or QN")
    if masks.shape[1] != len(classes) or not len(classes):
        raise ValueError("Mask3D mask and class query counts differ or are empty")
    if not np.isfinite(masks).all() or not np.isfinite(classes).all():
        raise ValueError("Mask3D logits must be finite")
    if not 0 <= mask_threshold <= 1 or not 0 <= class_threshold <= 1:
        raise ValueError("Mask3D mask/class thresholds must lie in [0,1]")
    no_object = int(no_object_index)
    if no_object < 0:
        no_object += classes.shape[1]
    if not 0 <= no_object < classes.shape[1]:
        raise ValueError("no_object_index is outside the class logits")
    probabilities = np.exp(classes.astype(np.float64) - classes.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    class_ids = probabilities.argmax(axis=1)
    class_scores = probabilities[np.arange(len(classes)), class_ids]
    accepted = (class_ids != no_object) & (class_scores >= class_threshold)
    # Chunk point rows to bound memory for full-resolution scenes.
    prediction = np.zeros(len(masks), dtype=np.int64)
    confidence = np.zeros(len(masks), dtype=np.float32)
    for start in range(0, len(masks), 65536):
        stop = min(start + 65536, len(masks))
        chunk = masks[start:stop].astype(np.float32, copy=False)
        probability = 1.0 / (1.0 + np.exp(-np.clip(chunk, -80, 80)))
        scores = probability * class_scores[None, :]
        scores[(probability < mask_threshold) | ~accepted[None, :]] = -1
        winners = scores.argmax(axis=1)
        best = scores[np.arange(stop - start), winners]
        selected = best >= 0
        prediction[start:stop] = np.where(selected, winners + 1, 0)
        confidence[start:stop] = np.where(selected, best, 0)
    if inverse_map is not None:
        inverse = _integer_indices(inverse_map, "inverse_map", len(scene.points), len(masks))
        prediction, confidence = prediction[inverse], confidence[inverse]
    elif len(masks) != len(scene.points):
        raise ValueError("Mask3D mask rows differ from scene points; supply the exact inverse_map")
    observation = Observation(
        name=name, features={key: _array(value) for key, value in (features or {}).items()},
        prediction=prediction, confidence=confidence, metadata={
            "adapter": "mask3d_output", "prediction_kind": "instance",
            "prediction_background_ids": [0],
            "confidence_semantics": "sigmoid(mask logit) times selected class softmax score",
            "confidence_calibrated": False,
            "mask3d_postprocessing": {"mask_threshold": mask_threshold,
                "class_threshold": class_threshold, "no_object_index": no_object,
                "overlap_policy": "maximum mask probability times class probability",
                "background_instance_id": 0, "benchmark_postprocessing": False},
            "queries": [{"instance_id": query + 1, "semantic_index": int(class_ids[query]),
                         "class_score": float(class_scores[query]), "accepted": bool(accepted[query])}
                        for query in range(len(classes))],
        },
    )
    return observation.validate(scene)


def _mask3d_options(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {key: spec[key] for key in (
        "mask_key", "class_key", "mask_layout", "mask_threshold", "class_threshold",
        "no_object_index",
    ) if key in spec}


def _load_mask3d(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    """Read exported official/CHORUS query outputs with geometry verification."""
    path = resolve_path(spec["path"], base_dir)
    payload, _ = _read_payload(path, spec.get("format"))
    alignment = _ordered_geometry(payload, spec, scene, base_dir)
    fields = spec.get("fields", {})
    inverse = _optional(payload, fields, "inverse_map")
    checkpoint_hash = _scalar(payload.get("checkpoint_sha256"))
    if spec.get("expected_checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("Mask3D output checkpoint hash mismatch")
    observation = observation_from_mask3d_output(
        payload, scene, name=str(spec.get("name", "Mask3D")), inverse_map=inverse,
        **_mask3d_options(spec),
    )
    observation.metadata.update({"cache_path": str(path), "cache_sha256": sha256_file(path),
                                 "alignment_check": alignment,
                                 "checkpoint_sha256": checkpoint_hash,
                                 "source_geometry_sha256": geometry_sha256(scene.points)})
    return observation


def _load_callable(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    reference = str(spec["callable"])
    module_name, separator, name = reference.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("callable must be 'package.module:function'")
    callback = getattr(importlib.import_module(module_name), name)
    config = dict(spec)
    config["base_dir"] = str(base_dir.resolve())
    for key in ("checkpoint", "config_path", "model_root"):
        if key in config:
            config[key] = str(resolve_path(config[key], base_dir))
    checkpoint_hash = None
    if "checkpoint" in config:
        checkpoint_hash = sha256_file(Path(config["checkpoint"]))
        if spec.get("expected_checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
            raise ValueError("Callable checkpoint hash mismatch")
    output = callback(scene=scene, config=config)
    if spec.get("output_format") == "mask3d":
        output = observation_from_mask3d_output(
            output, scene, name=str(spec.get("name", "Mask3D")),
            inverse_map=output.get("inverse_map"), **_mask3d_options(spec),
        )
    if isinstance(output, Mapping):
        output = Observation(**{"name": spec.get("name", name), **dict(output)})
    if not isinstance(output, Observation):
        raise TypeError("Model callable must return Observation or an Observation field dictionary")
    output.metadata.update({"adapter": "callable", "callable": reference,
                            "source_geometry_sha256": geometry_sha256(scene.points)})
    if "checkpoint" in config:
        output.metadata["checkpoint"] = config["checkpoint"]
        output.metadata["checkpoint_sha256"] = checkpoint_hash
    return output


def load_observation(spec: dict[str, Any], scene: Scene, base_dir: Path) -> Observation:
    """Run/load a model on the full scene; rendering subsamples only afterwards."""
    scene.validate()
    kind = spec.get("kind", "cache")
    adapters = {"cache": _load_cache, "npz": _load_cache, "litept": _load_litept,
                "agile3d": _load_agile3d, "mask3d": _load_mask3d, "callable": _load_callable}
    if kind not in adapters:
        raise ValueError(f"Unsupported model kind {kind!r}; use a callable for external Mask3D/AGILE3D")
    observation = adapters[kind](spec, scene, base_dir)
    # Descriptive fields can be added by a manifest, but cannot replace the
    # adapter's measured checkpoint hashes, array geometry or preprocessing.
    observation.metadata = {**dict(spec.get("metadata", {})), **observation.metadata}
    if "prediction_kind" in spec:
        observation.metadata["prediction_kind"] = str(spec["prediction_kind"])
    observation.validate(len(scene.points))
    return observation
