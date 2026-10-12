"""Versioned 2D-only source packs, with no GT/hierarchy arrays."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from delimit3d.training.contracts import GRANULARITY_KEYS, RAW_CACHE_SCHEMA, TWO_D_CELLS, SceneSourceAudit, sha256_file
from delimit3d.data.single_scene_dataset import build_input_features
from delimit3d.data.point_augmentations import canonicalize_structured3d_points_normals_to_scannet

SCHEMA = "delimit3d_structured3d_2d_source/v1"


def artifact_path(manifest_path, record):
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("2D source artifacts must use portable relative paths")
    return (manifest_path.parent / relative).resolve(strict=True)


def audit_2d_source(manifest_path, *, expected_scene_id=None, verify_hashes=True):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != SCHEMA or manifest.get("source_cells") != list(TWO_D_CELLS)
            or manifest.get("labels") is not None or not manifest.get("complete")):
        raise ValueError("Invalid 2D-only source pack")
    scene_id = manifest["scene_id"]
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError("2D-only source identity differs")
    for key in ("points", "colors", "normals", "crop_indices", "raw_projection_cache"):
        record = manifest["artifacts"][key]
        path = artifact_path(manifest_path, record)
        if path.stat().st_size != record["size_bytes"] or (verify_hashes and sha256_file(path) != record["sha256"]):
            raise ValueError(f"2D source artifact changed: {key}")
    frames = manifest["frames"]
    if not frames["train"] or not frames["heldout"] or set(frames["train"]) & set(frames["heldout"]):
        raise ValueError("2D source frame split is missing or overlaps")
    return SceneSourceAudit(scene_id=scene_id, manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path), crop_mode=manifest["crop"]["mode"],
        crop_points=manifest["crop"]["num_points"], train_frames=tuple(frames["train"]),
        heldout_frames=tuple(frames["heldout"]), trainable_proposals=manifest["counts"]["trainable_proposals"],
        heldout_trainable_proposals=manifest["counts"]["heldout_trainable_proposals"])


def load_2d_source(manifest_path, *, input_features="rgbn6", coordinate_normalization_policy="native",
                   verify_hashes=True, color_mean=None, color_std=None):
    from delimit3d.pretraining.sources import RawFrame, SceneSource
    audit = audit_2d_source(manifest_path, verify_hashes=verify_hashes)
    manifest = json.loads(Path(manifest_path).read_text())
    artifacts = manifest["artifacts"]
    _resolve_artifact = artifact_path
    points = np.asarray(np.load(artifact_path(manifest_path, artifacts["points"]), allow_pickle=False), dtype=np.float32)
    colors = np.asarray(np.load(artifact_path(manifest_path, artifacts["colors"]), allow_pickle=False))
    normals = np.asarray(np.load(artifact_path(manifest_path, artifacts["normals"]), allow_pickle=False), dtype=np.float32)
    if coordinate_normalization_policy == "structured3d_local_y_up_to_scannet_z_up_v1":
        points, normals = canonicalize_structured3d_points_normals_to_scannet(points, normals)
    elif coordinate_normalization_policy != "native":
        raise ValueError("Unknown 2D source coordinate policy")
    input_channels = {"rgb3": 3, "rgbn6": 6}[input_features]
    n = len(points)
    if points.shape != (n,3) or colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError("2D geometry arrays are not aligned")
    if not all(np.isfinite(a).all() for a in (points, colors, normals)):
        raise ValueError("2D geometry contains nonfinite values")
    features = build_input_features(points, colors, use_colors=True, use_normals=input_features=="rgbn6",
        normals=normals, color_mean=color_mean, color_std=color_std).astype(np.float32)
    labels_by_key = {}
    cache_path = _resolve_artifact(manifest_path, artifacts["raw_projection_cache"])
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache["schema_version"].item()) != RAW_CACHE_SCHEMA:
            raise ValueError(f"{cache_path}: raw cache schema drift")
        crop_path = _resolve_artifact(manifest_path, artifacts["crop_indices"])
        if (
            verify_hashes
            and str(cache["crop_indices_sha256"].item()) != sha256_file(crop_path)
        ):
            raise ValueError(f"{cache_path}: crop hash drift")
        physical_ids = [str(value) for value in cache["physical_frame_ids"].tolist()]
        granularities = [str(value) for value in cache["granularity_keys"].tolist()]
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        frame_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
        visible_indices = np.asarray(cache["eligible_point_indices"], dtype=np.int64)
        proposal_frame_offsets = np.asarray(
            cache["proposal_frame_offsets"], dtype=np.int64
        )
        proposal_point_offsets = np.asarray(
            cache["proposal_point_offsets"], dtype=np.int64
        )
        proposal_point_indices = np.asarray(
            cache["proposal_point_indices"], dtype=np.int64
        )
    num_frames = len(physical_ids)
    if (num_frames == 0 or not np.isin(split_codes, [0, 1]).all()
            or not set(granularities).issubset(GRANULARITY_KEYS)
            or any(a.size == 0 or int(a[0]) != 0 or (np.diff(a) < 0).any()
                   for a in (frame_offsets, proposal_frame_offsets, proposal_point_offsets))):
        raise ValueError(f"{cache_path}: invalid frame/cache ordering")
    if (
        len(granularities) != num_frames
        or split_codes.shape != (num_frames,)
        or frame_offsets.shape != (num_frames + 1,)
        or proposal_frame_offsets.shape != (num_frames + 1,)
        or int(frame_offsets[-1]) != int(visible_indices.size)
        or int(proposal_frame_offsets[-1]) + 1 != int(proposal_point_offsets.size)
        or int(proposal_point_offsets[-1]) != int(proposal_point_indices.size)
    ):
        raise ValueError(f"{cache_path}: compact-cache offset drift")

    raw_frames: list[RawFrame] = []
    for frame_index in range(num_frames):
        visible_start = int(frame_offsets[frame_index])
        visible_end = int(frame_offsets[frame_index + 1])
        proposal_start = int(proposal_frame_offsets[frame_index])
        proposal_end = int(proposal_frame_offsets[frame_index + 1])
        point_start = int(proposal_point_offsets[proposal_start])
        point_end = int(proposal_point_offsets[proposal_end])
        local_offsets = (
            proposal_point_offsets[proposal_start : proposal_end + 1] - point_start
        )
        visible = visible_indices[visible_start:visible_end].copy()
        members = proposal_point_indices[point_start:point_end].copy()
        if visible.size and (int(visible.min()) < 0 or int(visible.max()) >= n):
            raise ValueError(f"{cache_path}: visible index outside crop")
        if members.size and (int(members.min()) < 0 or int(members.max()) >= n):
            raise ValueError(f"{cache_path}: proposal index outside crop")
        raw_frames.append(
            RawFrame(
                granularity=granularities[frame_index],
                physical_frame_id=physical_ids[frame_index],
                split="train" if int(split_codes[frame_index]) == 0 else "heldout",
                visible_indices=visible,
                proposal_offsets=local_offsets.copy(),
                proposal_point_indices=members,
            )
        )
    expected_physical = len(audit.train_frames) + len(audit.heldout_frames)
    for key in GRANULARITY_KEYS:
        for split, expected in (("train", audit.train_frames), ("heldout", audit.heldout_frames)):
            actual = [frame.physical_frame_id for frame in raw_frames if frame.granularity == key and frame.split == split]
            if len(set(actual)) != len(actual) or set(actual) != set(expected):
                raise ValueError(f"{cache_path}: physical frame identity drift for {key}/{split}")
        if len([frame for frame in raw_frames if frame.granularity == key]) != expected_physical:
            raise ValueError(
                f"{cache_path}: expected {expected_physical} virtual frames for {key}"
            )
        if len([frame for frame in raw_frames if frame.granularity == key and frame.split == "train"]) != len(audit.train_frames):
            raise ValueError(f"{cache_path}: train-frame cardinality drift for {key}")
        if len([frame for frame in raw_frames if frame.granularity == key and frame.split == "heldout"]) != len(audit.heldout_frames):
            raise ValueError(f"{cache_path}: heldout-frame cardinality drift for {key}")
    return SceneSource(
        scene_id=audit.scene_id,
        manifest_path=manifest_path,
        manifest_sha256=audit.manifest_sha256,
        points=points,
        features=features,
        labels_by_key=labels_by_key,
        raw_frames=tuple(raw_frames),
        colors=colors,
        normals=normals,
        crop_mode=audit.crop_mode,
        coordinate_normalization_policy=coordinate_normalization_policy,
    )
