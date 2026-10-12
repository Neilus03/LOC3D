"""Loading and validating frozen point-feature caches.

The AGILE3D data loader owns the official 5 cm quantization.  A frozen
external encoder (for example LitePT) may use a different internal voxel
size, so its exporter stores one feature row per AGILE3D voxel together with
the integer voxel coordinate.  This module reorders the cache to the exact
order produced by the official loader and refuses ambiguous caches.
"""

from pathlib import Path

import numpy as np


def _scene_dirs(feature_root, scene_name):
    root = Path(feature_root)
    return (
        root / scene_name,
        root / "train" / scene_name,
        root / "val" / scene_name,
        root / "test" / scene_name,
    )


def _find_scene_file(feature_root, scene_name, filename):
    for scene_dir in _scene_dirs(feature_root, scene_name):
        path = scene_dir / filename
        if path.is_file():
            return path
    return None


def load_scene_features(
    feature_root,
    scene_name,
    expected_quantized_coords,
    expected_dim=None,
):
    """Load one feature cache and align it to AGILE3D's voxel order.

    Args:
        feature_root: Root containing ``<scene>/features.npy`` and
            ``<scene>/quantized_coords.npy`` (split subdirectories are also
            accepted).
        scene_name: Official scene identifier, e.g. ``scene0000_00``.
        expected_quantized_coords: The integer coordinates returned by
            ``ME.utils.sparse_quantize`` for the current official PLY.
        expected_dim: Optional feature width check.

    Returns:
        A contiguous float32 array in the exact order of
        ``expected_quantized_coords``.
    """

    if not feature_root:
        raise ValueError("feature_root is required for external features")

    feature_path = _find_scene_file(feature_root, scene_name, "features.npy")
    key_path = _find_scene_file(feature_root, scene_name, "quantized_coords.npy")
    if feature_path is None or key_path is None:
        missing = []
        if feature_path is None:
            missing.append("features.npy")
        if key_path is None:
            missing.append("quantized_coords.npy")
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} for external feature cache "
            f"scene {scene_name} under {feature_root}"
        )

    features = np.asarray(np.load(feature_path, mmap_mode="r"))
    expected_coords = np.asarray(expected_quantized_coords)
    saved_coords = np.asarray(np.load(key_path, mmap_mode="r"))

    if features.ndim != 2:
        raise ValueError(
            f"External features for {scene_name} must be 2-D, got "
            f"{features.shape} from {feature_path}"
        )
    if expected_coords.ndim != 2 or expected_coords.shape[1] != 3:
        raise ValueError(
            f"Expected quantized coordinates for {scene_name} must have shape "
            f"(N, 3), got {expected_coords.shape}"
        )
    if saved_coords.ndim != 2 or saved_coords.shape[1] != 3:
        raise ValueError(
            f"Cached quantized coordinates for {scene_name} must have shape "
            f"(N, 3), got {saved_coords.shape} from {key_path}"
        )
    if features.shape[0] != saved_coords.shape[0]:
        raise ValueError(
            f"Feature/key row mismatch for {scene_name}: features have "
            f"{features.shape[0]} rows but keys have {saved_coords.shape[0]}"
        )
    if features.shape[0] != expected_coords.shape[0]:
        raise ValueError(
            f"External feature point-count mismatch for {scene_name}: cache has "
            f"{features.shape[0]} rows but AGILE3D has {expected_coords.shape[0]}"
        )
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"External feature width mismatch for {scene_name}: cache has "
            f"{features.shape[1]}, expected {expected_dim}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"External features for {scene_name} contain non-finite values")

    expected_coords = expected_coords.astype(np.int64, copy=False)
    saved_coords = saved_coords.astype(np.int64, copy=False)
    saved_keys = [tuple(row) for row in saved_coords.tolist()]
    if len(set(saved_keys)) != len(saved_keys):
        raise ValueError(f"Cached quantized coordinates for {scene_name} contain duplicates")

    key_to_row = {key: row for row, key in enumerate(saved_keys)}
    try:
        reorder = np.asarray(
            [key_to_row[tuple(row)] for row in expected_coords.tolist()],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"External feature cache voxel keys do not match AGILE3D for {scene_name}"
        ) from exc

    return np.ascontiguousarray(features[reorder], dtype=np.float32)
