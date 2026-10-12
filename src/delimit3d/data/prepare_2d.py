"""Geometry and 2D-only sources. Historical GT-assisted packs keep their schema."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from delimit3d.cli.prepare_structured3d_raw import Structured3DReader, Structured3DArchiveAdapter
from delimit3d.cli.prepare_structured3d_sources import (
    Structured3DRawProposalProvider, _select_all_complete_valid_pose_frames,
)
from delimit3d.cli.prepare_sources import _array_record, _file_record, _project_proposals, _write_raw_cache, _global_to_local, _local_frame
from delimit3d.data.region_sampling import sphere_crop_indices
from delimit3d.runtime.records import write_json
from delimit3d.training.contracts import TWO_D_CELLS, EXPERIMENT_ID, sha256_file
from .source_2d import SCHEMA, audit_2d_source


def geometry_reference(scene_root, reader):
    scene_root = Path(scene_root)
    scene = scene_root.name
    manifest_path = scene_root / "source_manifest.json"
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text())
        if value.get("schema_version") != "delimit3d_structured3d_geometry/v1":
            raise ValueError("Existing directory is not a geometry-only reference")
        for record in value["artifacts"].values():
            if sha256_file(scene_root / record["path"]) != record["sha256"]:
                raise ValueError("Geometry reference changed")
        return value
    points = np.load(scene_root / "points.npy", allow_pickle=False)
    np.save(scene_root / "crop_indices.npy", np.arange(len(points), dtype=np.int64))
    adapter = Structured3DArchiveAdapter(scene_id=scene, reader=reader)
    frames, invalid, train, heldout = _select_all_complete_valid_pose_frames(adapter)
    value = {
        "schema_version": "delimit3d_structured3d_geometry/v1", "scene_id": scene,
        "dataset": "structured3d", "crop": {"mode": "full_scene", "num_points": len(points)},
        "frames": {"physical": [frame.frame_id for frame in frames], "train": train, "heldout": heldout,
                   "invalid_pose_candidates": invalid, "selection_mode": "all_complete_valid_pose"},
        "artifacts": {key: _array_record(scene_root / (key + ".npy"), relative_to=scene_root)
                      for key in ("points", "colors", "normals", "crop_indices")},
        "labels": None, "geometry_policy": "RGBD_2cm_first_representative_v1",
        "normal_policy": "open3d_hybrid_radius_0.1_max_nn_30_orient_toward_origin_v1",
    }
    write_json(manifest_path, value)
    return value


def counts_2d(indices, projected, num_points):
    lookup = _global_to_local(indices, num_points)
    counts = {split: {cell: 0 for cell in TWO_D_CELLS} for split in ("train", "heldout")}
    for key, frames in projected.items():
        for frame in frames:
            _, members, _ = _local_frame(frame, lookup)
            counts[frame.split]["2d/" + key] += len(members)
    return counts


def build_2d_scene(*, scene_id, geometry_root, proposal_root, output_root, reader, crop_mode, seed=42):
    target = Path(output_root) / scene_id
    if (target / "source_manifest.json").is_file():
        return audit_2d_source(target / "source_manifest.json")
    if target.exists():
        raise FileExistsError(f"Preserve incomplete source and use a fresh directory: {target}")
    root = Path(geometry_root) / scene_id
    reference = geometry_reference(root, reader)
    points, colors, normals = (np.load(root / (key + ".npy"), allow_pickle=False) for key in ("points", "colors", "normals"))
    adapter = Structured3DArchiveAdapter(scene_id=scene_id, reader=reader)
    by_id = {frame.frame_id: frame for frame in adapter.list_frames()}
    frames = [by_id[value] for value in reference["frames"]["physical"]]
    provider = Structured3DRawProposalProvider(Path(proposal_root))
    provider.validate_scene(scene_id=scene_id, frame_ids=[frame.frame_id for frame in frames])
    projected = _project_proposals(adapter=adapter, points=points, selected_frames=frames,
                                   train_frame_ids=reference["frames"]["train"], provider=provider)
    if crop_mode == "full_scene":
        indices = np.arange(len(points), dtype=np.int64)
        counts = counts_2d(indices, projected, len(points))
    elif crop_mode == "sphere_50k":
        choices = []
        for candidate_seed in range(seed, seed + 8):
            candidate = sphere_crop_indices(points, point_max=50000, rng=np.random.default_rng(candidate_seed))
            candidate_counts = counts_2d(candidate, projected, len(points))
            if all(count > 0 for split in candidate_counts.values() for count in split.values()):
                choices.append((sum(candidate_counts["train"].values()), candidate, candidate_counts))
        if not choices:
            raise ValueError(f"{scene_id}: no 2D-only crop has every granularity in both frame splits")
        _, indices, counts = max(choices, key=lambda item: item[0])
    else:
        raise ValueError("crop_mode must be full_scene or sphere_50k")
    if not all(count > 0 for split in counts.values() for count in split.values()):
        raise ValueError(f"{scene_id}: incomplete 2D proposal coverage")
    target.mkdir(parents=True, exist_ok=False)
    for key, array in (("points", points[indices]), ("colors", colors[indices]), ("normals", normals[indices]), ("crop_indices", indices)):
        np.save(target / (key + ".npy"), array)
    cache_counts = _write_raw_cache(path=target / "raw_projection_cache.npz", crop_indices=indices,
                                    projected=projected, num_points=len(points))
    manifest = {
        "schema_version": SCHEMA, "scene_id": scene_id, "dataset": "structured3d", "complete": True,
        "experiment_id": EXPERIMENT_ID, "source_cells": list(TWO_D_CELLS), "labels": None,
        "crop": {"mode": crop_mode, "num_points": len(indices), "seed": seed,
                 "selection_policy": "all_points" if crop_mode == "full_scene" else "2d_only_proposal_coverage_v1"},
        "frames": reference["frames"],
        "counts": {"trainable_proposals": counts["train"], "heldout_trainable_proposals": counts["heldout"], "cache": cache_counts},
        "artifacts": {key: _array_record(target / (key + ".npy"), relative_to=target)
                      for key in ("points", "colors", "normals", "crop_indices")},
        "teacher": provider.provenance(),
        "reference_geometry_sha256": sha256_file(root / "source_manifest.json"),
        "historical_crop_selection_reproduced": False,
    }
    manifest["artifacts"]["raw_projection_cache"] = _file_record(target / "raw_projection_cache.npz", relative_to=target)
    write_json(target / "source_manifest.json", manifest)
    return audit_2d_source(target / "source_manifest.json")
