"""Materialized ScanNet GT labels for the LitePT-Mask3D training path.

The cache stores exactly the arrays produced by ``load_scannet_gt_instances``
after the existing student convention converts ignored instance id 0 to -1.
It removes repeated PLY/aggregation/segment JSON parsing from every training
sample without changing the target construction or augmentation pipeline.
"""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from delimit3d.tasks.mask3d.data.training_pack import load_training_pack_scene_multi


SCANNET_GT_CACHE_VERSION = 1


def cache_file(root: str | Path, scene_id: str) -> Path:
    """Return the deterministic cache file for one ScanNet scene."""

    return Path(root) / f"{scene_id}.npz"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_cache_manifest(root: str | Path) -> dict[str, Any]:
    """Load and validate the cache manifest."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ScanNet GT cache manifest is missing: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid ScanNet GT cache manifest: {manifest_path}")
    if int(payload.get("cache_version", -1)) != SCANNET_GT_CACHE_VERSION:
        raise RuntimeError(
            f"Unsupported ScanNet GT cache version in {manifest_path}: "
            f"{payload.get('cache_version')!r}"
        )
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict):
        raise RuntimeError(f"ScanNet GT cache manifest has no scenes map: {manifest_path}")
    return payload


def validate_cache_manifest(
    root: str | Path,
    scene_ids: Iterable[str],
    *,
    eval_benchmark: str,
) -> dict[str, Any]:
    """Validate that all requested scenes have materialized GT entries."""

    root = Path(root)
    manifest = load_cache_manifest(root)
    manifest_benchmark = str(manifest.get("eval_benchmark", ""))
    if manifest_benchmark != str(eval_benchmark):
        raise RuntimeError(
            "ScanNet GT cache benchmark mismatch: "
            f"cache={manifest_benchmark!r} requested={eval_benchmark!r}"
        )
    scenes = manifest["scenes"]
    for scene_id in scene_ids:
        scene_id = str(scene_id)
        entry = scenes.get(scene_id)
        path = cache_file(root, scene_id)
        if not isinstance(entry, dict) or not path.is_file():
            raise FileNotFoundError(
                f"ScanNet GT cache entry is missing for {scene_id}: {path}"
            )
        if str(entry.get("cache_file", "")) != path.name:
            raise RuntimeError(
                f"ScanNet GT cache manifest path mismatch for {scene_id}: {entry}"
            )
        if entry.get("sha256"):
            from delimit3d.runtime.records import file_sha256
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"ScanNet target cache changed: {scene_id}")
    return manifest


def load_cached_scannet_gt(
    root: str | Path,
    scene_id: str,
    *,
    expected_num_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Load one cached GT record and validate its shape contract."""

    path = cache_file(root, scene_id)
    try:
        with np.load(path, allow_pickle=False) as payload:
            instance_ids = np.asarray(payload["instance_ids"], dtype=np.int64).copy()
            segment_ids = np.asarray(payload["segment_ids"], dtype=np.int64).copy()
            class_pairs = np.asarray(
                payload["instance_class_pairs"], dtype=np.int64
            ).copy()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ScanNet GT cache entry is missing: {path}") from exc

    expected_shape = (int(expected_num_points),)
    if instance_ids.shape != expected_shape or segment_ids.shape != expected_shape:
        raise RuntimeError(
            f"Cached ScanNet GT shape mismatch for {scene_id}: "
            f"instances={instance_ids.shape} segments={segment_ids.shape} "
            f"expected={expected_shape}"
        )
    if class_pairs.ndim != 2 or class_pairs.shape[1] != 2:
        raise RuntimeError(
            f"Cached ScanNet GT class map has invalid shape for {scene_id}: "
            f"{class_pairs.shape}"
        )
    class_map = {
        int(instance_id): int(class_id)
        for instance_id, class_id in class_pairs.tolist()
    }
    return instance_ids, segment_ids, class_map


def materialize_scannet_gt_cache_scene(
    scene_dir: str | Path,
    output_root: str | Path,
    *,
    eval_benchmark: str,
    granularity: tuple[str, ...] = ("g05",),
) -> dict[str, Any]:
    """Validate and materialize one scene's ScanNet GT arrays."""

    from delimit3d.source.datasets.scannet.gt import load_scannet_gt_instances

    scene = load_training_pack_scene_multi(Path(scene_dir), granularity)
    gt = load_scannet_gt_instances(
        scene.scene_dir,
        scene.scene_id,
        eval_benchmark=eval_benchmark,
    )
    if gt.segment_ids is None or gt.vertex_xyz is None:
        raise RuntimeError(f"ScanNet GT is incomplete for {scene.scene_id}")

    segment_ids = np.asarray(gt.segment_ids, dtype=np.int64)
    vertex_xyz = np.asarray(gt.vertex_xyz, dtype=np.float32)
    if segment_ids.shape != (scene.num_points,):
        raise RuntimeError(
            f"{scene.scene_id}: segment ids {segment_ids.shape} do not align "
            f"with training-pack points {(scene.num_points,)}"
        )
    if vertex_xyz.shape != scene.points.shape or not np.allclose(
        vertex_xyz, np.asarray(scene.points, dtype=np.float32), rtol=0.0, atol=1e-5
    ):
        max_error = (
            float(np.max(np.abs(vertex_xyz - scene.points)))
            if vertex_xyz.shape == scene.points.shape and vertex_xyz.size
            else float("inf")
        )
        raise RuntimeError(
            f"{scene.scene_id}: training-pack point order is not aligned with "
            f"ScanNet mesh (max_abs_xyz_error={max_error})"
        )

    instance_ids = np.asarray(gt.instance_ids, dtype=np.int64)
    if instance_ids.shape != (scene.num_points,):
        raise RuntimeError(
            f"{scene.scene_id}: instance ids {instance_ids.shape} do not align "
            f"with training-pack points {(scene.num_points,)}"
        )
    # Match MultiSceneDataset's existing convention exactly.
    instance_ids = np.where(instance_ids == 0, -1, instance_ids).astype(
        np.int64, copy=False
    )
    class_pairs = np.asarray(
        sorted((int(instance_id), int(class_id)) for instance_id, class_id in gt.instance_class_ids.items()),
        dtype=np.int64,
    )
    if class_pairs.size == 0:
        class_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        class_pairs = class_pairs.reshape(-1, 2)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = cache_file(output_root, scene.scene_id)
    _atomic_write_npz(
        output_path,
        instance_ids=instance_ids,
        segment_ids=segment_ids,
        instance_class_pairs=class_pairs,
    )
    cached_instance_ids, cached_segment_ids, cached_class_map = load_cached_scannet_gt(
        output_root,
        scene.scene_id,
        expected_num_points=scene.num_points,
    )
    if (
        not np.array_equal(cached_instance_ids, instance_ids)
        or not np.array_equal(cached_segment_ids, segment_ids)
        or cached_class_map != {
            int(instance_id): int(class_id)
            for instance_id, class_id in class_pairs.tolist()
        }
    ):
        raise RuntimeError(f"Serialized ScanNet GT cache round-trip failed for {scene.scene_id}")
    return {
        "scene_id": scene.scene_id,
        "cache_file": output_path.name,
        "num_points": int(scene.num_points),
        "num_instances": int(np.unique(instance_ids[instance_ids > 0]).size),
    }


def materialize_scannet_gt_cache(
    scene_dirs: Iterable[str | Path],
    output_root: str | Path,
    *,
    eval_benchmark: str,
    workers: int = 1,
) -> dict[str, Any]:
    """Materialize a complete cache and write its manifest atomically."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dirs = [Path(scene_dir) for scene_dir in scene_dirs]
    entries: dict[str, dict[str, Any]] = {}
    jobs = [
        (scene_dir, output_root, str(eval_benchmark), index)
        for index, scene_dir in enumerate(scene_dirs, start=1)
    ]
    worker_count = max(int(workers), 1)
    if worker_count == 1:
        results = (_materialize_or_reuse_cache_scene(job) for job in jobs)
        for entry in results:
            _record_cache_result(entries, entry)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_materialize_or_reuse_cache_scene, job): job[-1]
                for job in jobs
            }
            for future in as_completed(futures):
                try:
                    entry = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                _record_cache_result(entries, entry)

    manifest = {
        "cache_version": SCANNET_GT_CACHE_VERSION,
        "eval_benchmark": str(eval_benchmark),
        "scene_count": len(entries),
        "scenes": entries,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _materialize_or_reuse_cache_scene(
    job: tuple[Path, Path, str, int],
) -> dict[str, Any]:
    """Worker-safe resumable cache operation."""

    scene_dir, output_root, eval_benchmark, index = job
    scene = load_training_pack_scene_multi(scene_dir, ("g05",))
    existing_path = cache_file(output_root, scene.scene_id)
    if existing_path.is_file():
        try:
            cached_instance_ids, _, _ = load_cached_scannet_gt(
                output_root,
                scene.scene_id,
                expected_num_points=scene.num_points,
            )
        except (OSError, KeyError, RuntimeError, ValueError):
            pass
        else:
            return {
                "scene_id": scene.scene_id,
                "cache_file": existing_path.name,
                "num_points": int(scene.num_points),
                "num_instances": int(
                    np.unique(cached_instance_ids[cached_instance_ids > 0]).size
                ),
                "_reused": True,
                "_index": index,
            }

    entry = materialize_scannet_gt_cache_scene(
        scene_dir,
        output_root,
        eval_benchmark=eval_benchmark,
    )
    entry["_reused"] = False
    entry["_index"] = index
    return entry


def _record_cache_result(
    entries: dict[str, dict[str, Any]],
    entry: dict[str, Any],
) -> None:
    scene_id = str(entry["scene_id"])
    reused = bool(entry.pop("_reused", False))
    index = int(entry.pop("_index", 0))
    entries[scene_id] = entry
    action = "reused" if reused else "cached"
    detail = (
        f"({entry['num_points']} points)"
        if reused
        else f"({entry['num_points']} points, {entry['num_instances']} instances)"
    )
    print(f"[{index}] {action} {scene_id} {detail}", flush=True)
