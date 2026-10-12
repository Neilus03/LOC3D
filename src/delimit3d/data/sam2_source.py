"""Versioned SAM2 full-scene sources; never impersonate an UnSAM granularity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

SCENE_SCHEMA = "delimit3d_structured3d_sam2_scene_source/v2"
CACHE_SCHEMA = "delimit3d_structured3d_sam2_projection_cache/v1"
SOURCE_CELL = "2d/sam2"
# The measured 526,742-point B* FP32 backward used 14.4 GB on a 24 GB 4090.
# Keep a conservative raw-point ceiling, including worst-case voxel occupancy.
MAX_VISIT_POINTS = 500_000
FRAGMENT_POLICY = "geometry_median_partition_epoch_rotation_500k/v1"


def artifact_path(manifest_path: Path, record: dict) -> Path:
    path = Path(record["path"])
    return (path if path.is_absolute() else manifest_path.parent / path).resolve(strict=True)


def audit_sam2_source(manifest_path: Path, *, expected_scene_id=None, verify_hashes=True):
    from delimit3d.training.contracts import SceneSourceAudit, sha256_file

    path = manifest_path.resolve(strict=True)
    manifest = json.loads(path.read_text())
    scene_id = str(manifest["scene_id"])
    if (
        manifest.get("schema_version") != SCENE_SCHEMA
        or manifest.get("dataset") != "structured3d"
        or manifest.get("complete") is not True
        or manifest.get("source_cells") != [SOURCE_CELL]
        or manifest.get("teacher_source_key") != SOURCE_CELL
        or manifest.get("single_granularity") is not True
        or manifest.get("crop", {}).get("mode") != "full_scene"
        or not manifest.get("checks")
        or not all(value is True for value in manifest["checks"].values())
    ):
        raise ValueError(f"{path}: SAM2 full-scene contract drift")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(f"{path}: scene identity drift")
    semantics = manifest.get("semantics", {})
    for key in ("frame_local", "overlap_preserved", "unseen_points_are_unknown",
                "visible_but_unmasked_points_are_unknown", "zero_proposal_frames_contribute_no_supervision"):
        if semantics.get(key) is not True:
            raise ValueError(f"{path}: unsafe SAM2 relation semantics: {key}")
    point_count = int(manifest["crop"]["num_points"])
    if point_count <= 1:
        raise ValueError(f"{path}: invalid scene size")
    for key in ("points", "colors", "normals", "crop_indices", "sam2_projection_cache"):
        record = manifest["artifacts"][key]
        artifact = artifact_path(path, record)
        if artifact.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"{artifact}: size drift")
        if verify_hashes and sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"{artifact}: hash drift")
    physical = tuple(manifest["frames"]["physical"])
    train = tuple(manifest["frames"]["train"])
    heldout = tuple(manifest["frames"]["heldout"])
    if len(physical) != len(set(physical)) or set(train) & set(heldout) or set(train) | set(heldout) != set(physical):
        raise ValueError(f"{path}: physical-frame split drift")
    return SceneSourceAudit(
        scene_id=scene_id, manifest_path=path, manifest_sha256=sha256_file(path),
        crop_mode="full_scene", crop_points=point_count,
        train_frames=train, heldout_frames=heldout,
        trainable_proposals=manifest["counts"]["trainable_proposals"],
        heldout_trainable_proposals=manifest["counts"]["heldout_trainable_proposals"],
    )


def load_sam2_source(manifest_path: Path, *, input_features, coordinate_normalization_policy,
                     verify_hashes, color_mean, color_std):
    from delimit3d.training.adaptation import (
        RawFrame, SceneSource, STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
        build_input_features, canonicalize_structured3d_points_normals_to_scannet,
    )

    if input_features != "rgbn6" or coordinate_normalization_policy != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY:
        raise ValueError("SAM2 pretraining requires canonical Structured3D RGBN6")
    audit = audit_sam2_source(manifest_path, verify_hashes=verify_hashes)
    path = audit.manifest_path
    manifest = json.loads(path.read_text())
    artifacts = manifest["artifacts"]
    arrays = {key: np.load(artifact_path(path, artifacts[key]), mmap_mode="r", allow_pickle=False)
              for key in ("points", "colors", "normals")}
    n = audit.crop_points
    if any(value.shape != (n, 3) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{path}: invalid RGBN6 geometry")
    points, normals = canonicalize_structured3d_points_normals_to_scannet(arrays["points"], arrays["normals"])
    colors = np.asarray(arrays["colors"], dtype=np.float32)
    features = build_input_features(points, colors, use_colors=True, use_normals=True,
                                    normals=normals, color_mean=color_mean, color_std=color_std).astype(np.float32, copy=False)
    cache_path = artifact_path(path, artifacts["sam2_projection_cache"])
    with np.load(cache_path, allow_pickle=False) as cache:
        if (str(cache["schema_version"].item()) != CACHE_SCHEMA
                or str(cache["scene_id"].item()) != audit.scene_id
                or str(cache["teacher_source_key"].item()) != SOURCE_CELL
                or str(cache["crop_indices_sha256"].item()) != artifacts["crop_indices"]["sha256"]):
            raise ValueError(f"{cache_path}: cache identity drift")
        ids = [str(value) for value in cache["physical_frame_ids"]]
        splits = np.asarray(cache["split_codes"], dtype=np.int8)
        # Crucial: the known union, NOT all visible camera points. Unmasked
        # points may not become contrastive negatives.
        known_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
        known = np.asarray(cache["eligible_point_indices"], dtype=np.int64)
        frame_offsets = np.asarray(cache["proposal_frame_offsets"], dtype=np.int64)
        offsets = np.asarray(cache["proposal_point_offsets"], dtype=np.int64)
        members = np.asarray(cache["proposal_point_indices"], dtype=np.int64)
    if ids != manifest["frames"]["physical"] or splits.shape != (len(ids),) or not np.isin(splits, (0, 1)).all():
        raise ValueError(f"{cache_path}: frame/split identity drift")
    for value, count, terminal in ((known_offsets, len(ids), len(known)),
                                   (frame_offsets, len(ids), len(offsets) - 1),
                                   (offsets, len(offsets) - 1, len(members))):
        if value.shape != (count + 1,) or value[0] != 0 or value[-1] != terminal or (np.diff(value) < 0).any():
            raise ValueError(f"{cache_path}: corrupt CSR offsets")
    for value in (known, members):
        if value.size and (value.min() < 0 or value.max() >= n):
            raise ValueError(f"{cache_path}: point index outside scene")
    frames = []
    for index, frame_id in enumerate(ids):
        split = "train" if splits[index] == 0 else "heldout"
        if frame_id not in manifest["frames"][split]:
            raise ValueError(f"{cache_path}: frame split drift")
        begin, end = frame_offsets[index:index + 2]
        start, stop = offsets[begin], offsets[end]
        frames.append(RawFrame(
            granularity="sam2", physical_frame_id=frame_id, split=split,
            visible_indices=known[known_offsets[index]:known_offsets[index + 1]],
            proposal_offsets=offsets[begin:end + 1] - start,
            # Keep every overlap relation, not only selected/trainable masks.
            proposal_point_indices=members[start:stop],
        ))
    return SceneSource(scene_id=audit.scene_id, manifest_path=path, manifest_sha256=audit.manifest_sha256,
                       points=points, features=features, labels_by_key={}, raw_frames=tuple(frames),
                       colors=colors, normals=normals, crop_mode="full_scene",
                       coordinate_normalization_policy=coordinate_normalization_policy,
                       source_variant_id=FRAGMENT_POLICY)


def geometry_fragments(points: np.ndarray, maximum_points: int = MAX_VISIT_POINTS) -> tuple[np.ndarray, ...]:
    """Disjoint, exhaustive spatial partitions selected independently of masks."""
    if maximum_points < 2:
        raise ValueError("fragment ceiling must be at least two points")
    leaves = []

    def partition(indices):
        if len(indices) <= maximum_points:
            leaves.append(np.sort(indices))
            return
        geometry = points[indices]
        axis = int(np.argmax(np.ptp(geometry, axis=0)))
        order = np.argsort(geometry[:, axis], kind="stable")
        ordered = indices[order]
        middle = len(ordered) // 2
        partition(ordered[:middle])
        partition(ordered[middle:])

    partition(np.arange(len(points), dtype=np.int64))
    return tuple(leaves)


def sam2_scene_for_visit(scene, *, epoch: int, global_seed: int):
    """One geometry fragment per logical visit; all points rotate into training."""
    from delimit3d.training.adaptation import RawFrame
    from delimit3d.training.contracts import stable_seed

    if scene.source_variant_id != FRAGMENT_POLICY or len(scene.points) <= MAX_VISIT_POINTS or scene.crop_id is not None:
        return scene
    fragments = geometry_fragments(scene.points, MAX_VISIT_POINTS)
    position = (stable_seed(global_seed, scene.scene_id, FRAGMENT_POLICY) + int(epoch)) % len(fragments)
    indices = fragments[position]
    inverse = np.full(len(scene.points), -1, dtype=np.int64)
    inverse[indices] = np.arange(len(indices))
    frames = []
    for frame in scene.raw_frames:
        known = inverse[frame.visible_indices]
        known = known[known >= 0]
        mapped = inverse[frame.proposal_point_indices]
        keep = mapped >= 0
        # Preserve proposal identities (including empty intersections) so a
        # fragment cannot manufacture or erase a shared-membership relation.
        prefix = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))
        offsets = prefix[frame.proposal_offsets]
        frames.append(RawFrame(frame.granularity, frame.physical_frame_id, frame.split,
                               known, offsets, mapped[keep]))
    digest = hashlib.sha256(indices.tobytes()).hexdigest()[:16]
    return replace(scene, points=scene.points[indices], features=scene.features[indices],
                   colors=scene.colors[indices], normals=scene.normals[indices], raw_frames=tuple(frames),
                   crop_id=f"sam2_fragment_{position:02d}_of_{len(fragments):02d}_{digest}")
