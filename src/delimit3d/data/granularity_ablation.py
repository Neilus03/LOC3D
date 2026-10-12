"""Read-only 2D-mask inputs and necessary geometric checks for the ablation.

No label_*.npy arrays are read. These checks are optimistic: they do not replace
co-membership-safe contrastive sampling or the actual two-fold hierarchy loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from delimit3d.data.point_augmentations import (
    augment_points_mask3d_scannet_rgbn6,
    canonicalize_structured3d_points_normals_to_scannet,
)
from delimit3d.training.contracts import RAW_CACHE_SCHEMA, stable_seed
from delimit3d.training.granularity_ablation import GRANULARITIES, file_record


@dataclass(frozen=True)
class MaskCatalog:
    granularity: str
    frame_id: str
    visible: np.ndarray
    offsets: np.ndarray
    members: np.ndarray

    def proposals(self):
        for index in range(len(self.offsets) - 1):
            yield index, np.unique(self.members[self.offsets[index]:self.offsets[index + 1]])


@dataclass(frozen=True)
class AblationScene:
    scene_id: str
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    catalogs: tuple[MaskCatalog, ...]
    artifacts: tuple[dict[str, Any], ...]


def load_scene(source_root: Path, scene_id: str) -> AblationScene:
    """Load only RGBN6 geometry and the train split of the raw 2D mask cache."""
    path = source_root / scene_id / "source_manifest.json"
    manifest = json.loads(path.read_text())
    # The original compact-source v1 schema predates crop.mode. Its omitted
    # value means sphere_50k, as in training.contracts.audit_scene_manifest.
    if manifest["crop"].get("mode", "sphere_50k") != "sphere_50k":
        raise ValueError(f"Unsupported ablation crop in {path}")
    records = [file_record(path)]

    def artifact(name):
        declared = manifest["artifacts"][name]
        source = (path.parent / declared["path"]).resolve(strict=True)
        actual = file_record(source)
        if actual["sha256"] != declared["sha256"]:
            raise ValueError(f"Source hash drift: {source}")
        records.append(actual)
        return source

    points = np.asarray(np.load(artifact("points"), allow_pickle=False), dtype=np.float32)
    colors = np.asarray(np.load(artifact("colors"), allow_pickle=False), dtype=np.float32)
    normals = np.asarray(np.load(artifact("normals"), allow_pickle=False), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError("RGBN6 source array shape mismatch")
    if not all(np.isfinite(array).all() for array in (points, colors, normals)):
        raise ValueError("Non-finite RGBN6 source")
    points, normals = canonicalize_structured3d_points_normals_to_scannet(points, normals)
    catalogs = []
    with np.load(artifact("raw_projection_cache"), allow_pickle=False) as cache:
        if str(cache["schema_version"].item()) != RAW_CACHE_SCHEMA:
            raise ValueError("Unknown raw projection schema")
        granularities = cache["granularity_keys"]
        frame_ids = cache["physical_frame_ids"]
        splits = cache["split_codes"]
        frame_offsets = cache["frame_offsets"]
        visible = cache["eligible_point_indices"]
        proposal_frames = cache["proposal_frame_offsets"]
        offsets = cache["proposal_point_offsets"]
        members = cache["proposal_point_indices"]
        n = len(granularities)
        if (len(frame_ids) != n or splits.shape != (n,)
                or frame_offsets.shape != (n + 1,) or proposal_frames.shape != (n + 1,)
                or frame_offsets[0] != 0 or frame_offsets[-1] != len(visible)
                or proposal_frames[0] != 0 or proposal_frames[-1] + 1 != len(offsets)
                or offsets[0] != 0 or offsets[-1] != len(members)
                or not np.isin(splits, [0, 1]).all()
                or any(np.any(np.diff(array) < 0) for array in (frame_offsets, proposal_frames, offsets))):
            raise ValueError("Projection-cache CSR/split mismatch")
        identities = set()
        for i in range(n):
            if str(granularities[i]) not in GRANULARITIES:
                raise ValueError("Unknown granularity")
            if int(splits[i]) != 0:
                continue
            identity = (str(granularities[i]), str(frame_ids[i]))
            if identity in identities:
                raise ValueError("Duplicate physical-frame/granularity catalog")
            identities.add(identity)
            start, end = int(proposal_frames[i]), int(proposal_frames[i + 1])
            base, stop = int(offsets[start]), int(offsets[end])
            catalog = MaskCatalog(
                *identity, np.unique(visible[frame_offsets[i]:frame_offsets[i + 1]]),
                np.asarray(offsets[start:end + 1] - base, dtype=np.int64),
                np.asarray(members[base:stop], dtype=np.int64),
            )
            if any(array.size and (array.min() < 0 or array.max() >= len(points))
                   for array in (catalog.visible, catalog.members)):
                raise ValueError("Mask indices outside geometry")
            if not np.isin(catalog.members, catalog.visible).all():
                raise ValueError("Proposal extends outside its declared visibility")
            catalogs.append(catalog)
    return AblationScene(scene_id, points, colors, normals, tuple(catalogs), tuple(records))


def augmented_scene(scene: AblationScene, *, epoch: int = 0, seed: int = 42):
    """Exactly the existing RGBN6 augmentation seed/transform, without label reads."""
    augmentation_seed = stable_seed(seed, scene.scene_id, epoch, "input-augmentation")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed % (2**32))
        values = augment_points_mask3d_scannet_rgbn6(
            scene.points, scene.colors, normals=scene.normals, use_colors=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return tuple(np.ascontiguousarray(array, dtype=np.float32) for array in values)


def point_to_finest(points: np.ndarray, *, grid_size: float = 0.05) -> np.ndarray:
    # Torch's float32 division matches the actual backbone/planner. NumPy only
    # groups the resulting integer grid; its label ordering is immaterial here.
    import torch

    grid = torch.floor(torch.from_numpy(points).float() / float(grid_size)).to(torch.int64).numpy()
    grid -= grid.min(axis=0)
    return np.unique(grid, axis=0, return_inverse=True)[1].astype(np.int64)


def fully_known_tokens(visible: np.ndarray, relation_members: np.ndarray,
                       inverse: np.ndarray, all_counts: np.ndarray) -> np.ndarray:
    """Proposal-invariant frame support, cached once per catalog and context."""
    known = np.intersect1d(visible, relation_members, assume_unique=False)
    known_counts = np.bincount(inverse[known], minlength=len(all_counts))
    return (known_counts == all_counts) & (all_counts > 0)


def proposal_upper_bound(members: np.ndarray, visible: np.ndarray,
                         relation_members: np.ndarray, inverse: np.ndarray,
                         all_counts: np.ndarray, *,
                         fully_known: np.ndarray | None = None) -> dict[str, Any]:
    """Necessary pure-query and known-negative conditions, no semantic labels."""
    members = np.unique(members)
    if fully_known is None:
        fully_known = fully_known_tokens(visible, relation_members, inverse, all_counts)
    member_counts = np.bincount(inverse[members], minlength=len(all_counts))
    pure_positive = fully_known & (member_counts == all_counts) & (all_counts > 0)
    pure_negative = fully_known & (member_counts == 0) & (all_counts > 0)
    positives, negatives = int(pure_positive.sum()), int(pure_negative.sum())
    return {"pure_query_tokens": positives, "known_negative_tokens": negatives,
            "possibly_eligible": positives >= 2 and negatives >= 1,
            "raw_members": len(members)}


def geometric_capacity(scene: AblationScene, *, quota: int, epoch: int = 0,
                       seed: int = 42, grid_size: float = 0.05) -> dict[str, Any]:
    points, _, _ = augmented_scene(scene, epoch=epoch, seed=seed)
    inverse = point_to_finest(points, grid_size=grid_size)
    all_counts = np.bincount(inverse)
    same_frame: dict[str, list[np.ndarray]] = {}
    for catalog in scene.catalogs:
        same_frame.setdefault(catalog.frame_id, []).append(catalog.members)
    multigranular_relations = {
        frame: np.unique(np.concatenate(arrays)) for frame, arrays in same_frame.items()
    }
    counts = {g: {"raw": 0, "single": 0, "multi": 0, "common": 0}
              for g in GRANULARITIES}
    records = []
    for catalog in scene.catalogs:
        single_relations = np.unique(catalog.members)
        multi_relations = multigranular_relations[catalog.frame_id]
        single_known = fully_known_tokens(catalog.visible, single_relations, inverse, all_counts)
        multi_known = fully_known_tokens(catalog.visible, multi_relations, inverse, all_counts)
        for index, members in catalog.proposals():
            if len(members) < 2:
                continue
            single = proposal_upper_bound(members, catalog.visible, single_relations,
                                          inverse, all_counts, fully_known=single_known)
            multi = proposal_upper_bound(members, catalog.visible,
                                         multi_relations, inverse, all_counts, fully_known=multi_known)
            common = single["possibly_eligible"] and multi["possibly_eligible"]
            cell = counts[catalog.granularity]
            cell["raw"] += 1
            cell["single"] += int(single["possibly_eligible"])
            cell["multi"] += int(multi["possibly_eligible"])
            cell["common"] += int(common)
            if not common:
                records.append({"granularity": catalog.granularity, "frame": catalog.frame_id,
                                "proposal_index": index, "single": single, "multi": multi})
    return {"scene_id": scene.scene_id, "augmentation_epoch": epoch,
            "counts": counts, "admission_upper_bound": all(row["common"] >= quota for row in counts.values()),
            "rejected_proposals": records, "artifacts": list(scene.artifacts)}
