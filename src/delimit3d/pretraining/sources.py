"""Scene sources and deterministic geometry inputs."""

from __future__ import annotations

import json


import random

from dataclasses import dataclass

from pathlib import Path


from typing import Any, Mapping, Sequence

import numpy as np


from delimit3d.training.contracts import GRANULARITY_KEYS, RAW_CACHE_SCHEMA, TWO_D_CELLS, audit_scene_source_manifest, sha256_file, stable_seed


from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    augment_points_mask3d_scannet_rgbn6,
    canonicalize_structured3d_points_normals_to_scannet,
)

from delimit3d.data.single_scene_dataset import build_input_features


from .settings import (
    CLEAN_CROP_PACK_SCHEMA,
    CLEAN_PROJECTION_CACHE_SCHEMA,
    CLEAN_TRAINING_SOURCE_SCHEMA,
    TRAIN_AUGMENTATION_CHOICES,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
    TRAIN_AUGMENTATION_NONE,
    augmentation_contract_for_profile,
    input_channels_for,
)


@dataclass(frozen=True)
class RawFrame:
    granularity: str
    physical_frame_id: str
    split: str
    visible_indices: np.ndarray
    proposal_offsets: np.ndarray
    proposal_point_indices: np.ndarray


@dataclass(frozen=True)
class SceneSource:
    scene_id: str
    manifest_path: Path
    manifest_sha256: str
    points: np.ndarray
    features: np.ndarray
    labels_by_key: dict[str, np.ndarray]
    raw_frames: tuple[RawFrame, ...]
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    crop_mode: str = "sphere_50k"
    coordinate_normalization_policy: str = "native"
    crop_id: str | None = None
    source_variant_id: str | None = None

    def frames(self, granularity: str, split: str) -> tuple[RawFrame, ...]:
        return tuple(
            frame
            for frame in self.raw_frames
            if (
                frame.granularity == granularity
                and frame.split == split
                and frame.proposal_offsets.size > 1
            )
        )


@dataclass(frozen=True)
class MultiCropSceneSource:
    """One logical scene whose immutable clean crops rotate across epochs."""

    scene_id: str
    manifest_path: Path
    manifest_sha256: str
    crops: tuple[SceneSource, ...]
    crop_mode: str = "sphere_50k"
    coordinate_normalization_policy: str = (
        STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    )
    crop_rotation_policy: str = "deterministic_epoch_rotation_v1"

    def __post_init__(self) -> None:
        if not self.crops:
            raise ValueError("a multi-crop source requires at least one crop")
        if any(crop.scene_id != self.scene_id for crop in self.crops):
            raise ValueError("all clean crops must retain the logical scene ID")
        if any(crop.crop_mode != self.crop_mode for crop in self.crops):
            raise ValueError("clean crop modes differ within one logical scene")
        if any(
            crop.coordinate_normalization_policy
            != self.coordinate_normalization_policy
            for crop in self.crops
        ):
            raise ValueError("clean crop coordinate policies differ")


def scene_source_for_visit(
    scene: SceneSource | MultiCropSceneSource,
    *,
    epoch: int,
    global_seed: int,
) -> SceneSource:
    """Resolve one clean crop without changing logical scene optimization mass."""

    if isinstance(scene, SceneSource):
        if scene.source_variant_id == "geometry_median_partition_epoch_rotation_500k/v1":
            from delimit3d.data.sam2_source import sam2_scene_for_visit
            return sam2_scene_for_visit(scene, epoch=epoch, global_seed=global_seed)
        return scene
    if scene.crop_rotation_policy != "deterministic_epoch_rotation_v1":
        raise ValueError(f"unsupported crop rotation {scene.crop_rotation_policy!r}")
    offset = stable_seed(
        int(global_seed), scene.scene_id, "clean-pack-crop-rotation"
    ) % len(scene.crops)
    return scene.crops[(offset + int(epoch)) % len(scene.crops)]


@dataclass(frozen=True)
class SceneVisitInput:
    points: np.ndarray
    features: np.ndarray
    augmentation_seed: int | None


def build_scene_visit_input(
    *,
    scene: SceneSource,
    epoch: int,
    global_seed: int,
    input_features: str,
    augmentation_profile: str,
    color_mean: Sequence[float] | None = None,
    color_std: Sequence[float] | None = None,
) -> SceneVisitInput:
    """Build one point/feature view without mutating the cached scene arrays.

    The augmentation seed intentionally excludes DDP rank, scene-order position,
    selected frame, proposal, and granularity.  A scene therefore receives one
    reproducible transform per epoch/visit, shared by all proposal draws, while
    single- and multi-granularity arms can consume the same augmented views.
    """

    profile = str(augmentation_profile)
    if profile not in TRAIN_AUGMENTATION_CHOICES:
        augmentation_contract_for_profile(profile)
    if profile == TRAIN_AUGMENTATION_NONE:
        if (color_mean is None) != (color_std is None):
            raise ValueError("color_mean and color_std must be supplied together")
        if color_mean is not None:
            features = build_input_features(
                scene.points,
                scene.colors,
                use_colors=True,
                use_normals=input_features == "rgbn6",
                normals=scene.normals,
                color_mean=color_mean,
                color_std=color_std,
            ).astype(np.float32, copy=False)
        else:
            features = scene.features
        return SceneVisitInput(
            points=scene.points,
            features=features,
            augmentation_seed=None,
        )
    required_crop_modes = {
        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: "full_scene",
        TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K: "sphere_50k",
    }
    if profile not in required_crop_modes:
        raise AssertionError(f"Unhandled train augmentation profile {profile!r}")
    if input_features != "rgbn6":
        raise ValueError(f"{profile} requires input_features='rgbn6'")
    required_crop_mode = required_crop_modes[profile]
    if scene.crop_mode != required_crop_mode:
        raise ValueError(
            f"{scene.scene_id}: {profile} requires "
            f"crop.mode={required_crop_mode!r}, "
            f"got {scene.crop_mode!r}"
        )
    if scene.colors is None or scene.normals is None:
        raise ValueError(f"{scene.scene_id}: {profile} requires RGB and normals")

    augmentation_seed = stable_seed(
        int(global_seed),
        scene.scene_id,
        int(epoch),
        "input-augmentation",
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed % (2**32))
        aug_points, aug_colors, aug_normals = (
            augment_points_mask3d_scannet_rgbn6(
                scene.points,
                scene.colors,
                use_colors=True,
                normals=scene.normals,
            )
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    if aug_colors is None:
        raise RuntimeError(f"{scene.scene_id}: RGB augmentation returned no colors")
    expected_shape = scene.points.shape
    if aug_points.shape != expected_shape or aug_colors.shape != expected_shape:
        raise RuntimeError(
            f"{scene.scene_id}: augmentation changed point/color cardinality"
        )
    if aug_normals.shape != expected_shape:
        raise RuntimeError(f"{scene.scene_id}: augmentation changed normal cardinality")
    if not (
        np.isfinite(aug_points).all()
        and np.isfinite(aug_colors).all()
        and np.isfinite(aug_normals).all()
    ):
        raise RuntimeError(f"{scene.scene_id}: augmentation produced non-finite values")
    features = build_input_features(
        aug_points,
        aug_colors,
        use_colors=True,
        use_normals=True,
        normals=aug_normals,
        color_mean=color_mean,
        color_std=color_std,
    ).astype(np.float32, copy=False)
    return SceneVisitInput(
        points=np.ascontiguousarray(aug_points, dtype=np.float32),
        features=np.ascontiguousarray(features, dtype=np.float32),
        augmentation_seed=int(augmentation_seed),
    )


def _resolve_artifact(manifest_path: Path, record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(strict=True)


def _load_clean_multicrop_scene_source(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any],
    input_features: str,
    coordinate_normalization_policy: str,
    verify_hashes: bool,
    color_mean: Sequence[float] | None,
    color_std: Sequence[float] | None,
) -> MultiCropSceneSource:
    """Load a hierarchy-free clean-pack adapter manifest fail-closed."""

    if input_features != "rgbn6":
        raise ValueError("clean-pack V2 sources require RGBN6")
    if (
        coordinate_normalization_policy
        != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    ):
        raise ValueError(
            "clean-pack V2 sources require the canonical Structured3D-to-ScanNet "
            "coordinate policy"
        )
    required_checks = {
        "rgbn6_all_crops",
        "three_2d_cells_all_crops",
        "known_support_preserved",
        "no_hierarchy_or_gt",
        "single_variant_only",
        "artifact_hashes_recorded",
    }
    checks = manifest.get("checks")
    source_kind = str(manifest.get("source_kind", ""))
    if (
        manifest.get("dataset") != "structured3d"
        or source_kind not in {"current_pack", "clean_pack"}
        or manifest.get("source_cells") != list(TWO_D_CELLS)
        or manifest.get("crop_mode") != "sphere_50k"
        or manifest.get("crop_rotation_policy")
        != "deterministic_epoch_rotation_v1"
        or manifest.get("coordinate_normalization_policy")
        != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
        or not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or not all(bool(checks[key]) for key in required_checks)
    ):
        raise ValueError(f"{manifest_path}: clean training-source contract drift")
    scene_id = str(manifest.get("scene_id", ""))
    if not scene_id.startswith("scene_"):
        raise ValueError(f"{manifest_path}: invalid Structured3D scene ID")
    crop_records = manifest.get("crops")
    if not isinstance(crop_records, list) or not 1 <= len(crop_records) <= 4:
        raise ValueError(f"{manifest_path}: expected one to four clean crops")

    def artifact(record: Mapping[str, Any], key: str) -> Path:
        value = record.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"{manifest_path}: crop artifact {key!r} is missing")
        path = Path(str(value.get("path", "")))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{manifest_path}: non-portable crop artifact {key!r}")
        resolved = (manifest_path.parent / path).resolve(strict=True)
        if int(value.get("size_bytes", -1)) != resolved.stat().st_size:
            raise ValueError(f"{manifest_path}: crop artifact size drift for {key}")
        if verify_hashes and str(value.get("sha256", "")) != sha256_file(resolved):
            raise ValueError(f"{manifest_path}: crop artifact hash drift for {key}")
        return resolved

    crops: list[SceneSource] = []
    seen_crop_ids: set[str] = set()
    variant_id = str(manifest.get("variant", {}).get("variant_id", ""))
    if not variant_id:
        raise ValueError(f"{manifest_path}: clean variant ID is missing")
    for crop_record in crop_records:
        if not isinstance(crop_record, Mapping):
            raise ValueError(f"{manifest_path}: invalid clean crop record")
        crop_id = str(crop_record.get("crop_id", ""))
        if not crop_id or crop_id in seen_crop_ids:
            raise ValueError(f"{manifest_path}: duplicate/empty clean crop ID")
        seen_crop_ids.add(crop_id)
        crop_manifest_path = artifact(crop_record, "crop_manifest")
        crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
        if (
            crop_manifest.get("schema_version") != CLEAN_CROP_PACK_SCHEMA
            or crop_manifest.get("scene_id") != scene_id
            or crop_manifest.get("crop_id") != crop_id
            or crop_manifest.get("variant") != manifest.get("variant")
            or crop_manifest.get("labels") is not None
            or crop_manifest.get("features", {}).get("channels") != "rgbn6"
        ):
            raise ValueError(f"{crop_manifest_path}: clean crop manifest drift")
        points_path = artifact(crop_record, "points")
        colors_path = artifact(crop_record, "colors")
        normals_path = artifact(crop_record, "normals")
        cache_path = artifact(crop_record, "projection_cache")
        points = np.asarray(np.load(points_path, allow_pickle=False), dtype=np.float32)
        colors = np.asarray(np.load(colors_path, allow_pickle=False), dtype=np.float32)
        normals = np.asarray(np.load(normals_path, allow_pickle=False), dtype=np.float32)
        n = int(points.shape[0])
        if (
            n <= 1
            or n > 50_000
            or points.shape != (n, 3)
            or colors.shape != (n, 3)
            or normals.shape != (n, 3)
        ):
            raise ValueError(f"{crop_manifest_path}: RGBN6 crop arrays are invalid")
        points, normals = canonicalize_structured3d_points_normals_to_scannet(
            points, normals
        )
        features = build_input_features(
            points,
            colors,
            use_colors=True,
            use_normals=True,
            normals=normals,
            color_mean=color_mean,
            color_std=color_std,
        ).astype(np.float32, copy=False)
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_schema = str(cache["schema_version"].item())
            declared_cache_schema = str(
                crop_record.get("projection_cache_schema", "")
            )
            if (
                cache_schema != declared_cache_schema
                or cache_schema
                not in {CLEAN_PROJECTION_CACHE_SCHEMA, RAW_CACHE_SCHEMA}
                or (
                    source_kind == "clean_pack"
                    and cache_schema != CLEAN_PROJECTION_CACHE_SCHEMA
                )
                or (
                    source_kind == "current_pack"
                    and cache_schema != RAW_CACHE_SCHEMA
                )
            ):
                raise ValueError(f"{cache_path}: projection-cache schema drift")
            if (
                cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA
                and str(cache["variant_id"].item()) != variant_id
            ):
                raise ValueError(f"{cache_path}: clean projection variant drift")
            physical_ids = [
                str(value) for value in cache["physical_frame_ids"].tolist()
            ]
            granularities = [
                str(value) for value in cache["granularity_keys"].tolist()
            ]
            split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
            if cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA:
                visible_offsets = np.asarray(
                    cache["visible_frame_offsets"], dtype=np.int64
                )
                visible_indices = np.asarray(
                    cache["visible_point_indices"], dtype=np.int64
                )
                known_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
                known_indices = np.asarray(
                    cache["eligible_point_indices"], dtype=np.int64
                )
            else:
                visible_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
                visible_indices = np.asarray(
                    cache["eligible_point_indices"], dtype=np.int64
                )
                known_offsets = None
                known_indices = None
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
        if (
            len(granularities) != num_frames
            or split_codes.shape != (num_frames,)
            or visible_offsets.shape != (num_frames + 1,)
            or proposal_frame_offsets.shape != (num_frames + 1,)
            or int(visible_offsets[-1]) != int(visible_indices.size)
            or int(proposal_frame_offsets[-1]) + 1
            != int(proposal_point_offsets.size)
            or int(proposal_point_offsets[-1])
            != int(proposal_point_indices.size)
        ):
            raise ValueError(f"{cache_path}: clean cache offsets drift")
        if cache_schema == CLEAN_PROJECTION_CACHE_SCHEMA and (
            known_offsets is None
            or known_indices is None
            or known_offsets.shape != (num_frames + 1,)
            or int(known_offsets[-1]) != int(known_indices.size)
        ):
            raise ValueError(f"{cache_path}: clean known-support offsets drift")
        raw_frames: list[RawFrame] = []
        trainable_cells: set[str] = set()
        for frame_index in range(num_frames):
            cell = f"2d/{granularities[frame_index]}"
            if cell not in TWO_D_CELLS:
                raise ValueError(f"{cache_path}: unexpected clean cell {cell!r}")
            visible = visible_indices[
                int(visible_offsets[frame_index]) : int(visible_offsets[frame_index + 1])
            ].copy()
            proposal_start = int(proposal_frame_offsets[frame_index])
            proposal_end = int(proposal_frame_offsets[frame_index + 1])
            member_start = int(proposal_point_offsets[proposal_start])
            member_end = int(proposal_point_offsets[proposal_end])
            members = proposal_point_indices[member_start:member_end].copy()
            offsets = (
                proposal_point_offsets[proposal_start : proposal_end + 1]
                - member_start
            ).copy()
            if known_offsets is not None and known_indices is not None:
                known = known_indices[
                    int(known_offsets[frame_index]) : int(
                        known_offsets[frame_index + 1]
                    )
                ]
            else:
                known = np.unique(members).astype(np.int64, copy=False)
            for values, label in ((visible, "visible"), (known, "known"), (members, "proposal")):
                if values.size and (int(values.min()) < 0 or int(values.max()) >= n):
                    raise ValueError(f"{cache_path}: {label} index outside crop")
            if np.setdiff1d(
                np.unique(known), np.unique(visible), assume_unique=True
            ).size:
                raise ValueError(f"{cache_path}: known support is not visible")
            if np.setdiff1d(
                np.unique(members), np.unique(known), assume_unique=True
            ).size:
                raise ValueError(f"{cache_path}: proposal lies outside known support")
            split = "train" if int(split_codes[frame_index]) == 0 else "heldout"
            if split == "train" and any(
                int(offsets[index + 1] - offsets[index]) >= 2
                and int(offsets[index + 1] - offsets[index]) < np.unique(visible).size
                for index in range(max(int(offsets.size) - 1, 0))
            ):
                trainable_cells.add(cell)
            raw_frames.append(
                RawFrame(
                    granularity=granularities[frame_index],
                    physical_frame_id=physical_ids[frame_index],
                    split=split,
                    visible_indices=np.unique(visible).astype(np.int64, copy=False),
                    proposal_offsets=offsets,
                    proposal_point_indices=members,
                )
            )
        if trainable_cells != set(TWO_D_CELLS):
            raise ValueError(
                f"{cache_path}: clean crop lacks trainable three-cell support: "
                f"{sorted(trainable_cells)}"
            )
        crops.append(
            SceneSource(
                scene_id=scene_id,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                points=points,
                features=features,
                labels_by_key={},
                raw_frames=tuple(raw_frames),
                colors=colors,
                normals=normals,
                crop_mode="sphere_50k",
                coordinate_normalization_policy=coordinate_normalization_policy,
                crop_id=crop_id,
                source_variant_id=variant_id,
            )
        )
    return MultiCropSceneSource(
        scene_id=scene_id,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        crops=tuple(crops),
        coordinate_normalization_policy=coordinate_normalization_policy,
    )


def load_scene_source(
    manifest_path: Path,
    *,
    input_features: str = "rgbn6",
    coordinate_normalization_policy: str = "native",
    verify_hashes: bool = True,
    color_mean: Sequence[float] | None = None,
    color_std: Sequence[float] | None = None,
) -> SceneSource | MultiCropSceneSource:
    input_features = str(input_features)
    coordinate_normalization_policy = str(coordinate_normalization_policy)
    if coordinate_normalization_policy not in {
        "native",
        STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    }:
        raise ValueError(
            "Unknown coordinate-normalization policy "
            f"{coordinate_normalization_policy!r}"
        )
    input_channels = input_channels_for(input_features)
    resolved_manifest = manifest_path.resolve(strict=True)
    initial_manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if initial_manifest.get("schema_version") == "delimit3d_structured3d_2d_source/v1":
        from delimit3d.data.source_2d import load_2d_source
        return load_2d_source(resolved_manifest, input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy, verify_hashes=verify_hashes,
            color_mean=color_mean, color_std=color_std)
    if initial_manifest.get("schema_version") == "delimit3d_structured3d_sam2_scene_source/v2":
        from delimit3d.data.sam2_source import load_sam2_source
        return load_sam2_source(
            resolved_manifest, input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            verify_hashes=verify_hashes, color_mean=color_mean, color_std=color_std,
        )
    if initial_manifest.get("schema_version") == CLEAN_TRAINING_SOURCE_SCHEMA:
        return _load_clean_multicrop_scene_source(
            resolved_manifest,
            manifest=initial_manifest,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            verify_hashes=verify_hashes,
            color_mean=color_mean,
            color_std=color_std,
        )
    audit = audit_scene_source_manifest(
        manifest_path,
        verify_hashes=verify_hashes,
    )
    manifest_path = audit.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    points = np.asarray(
        np.load(_resolve_artifact(manifest_path, artifacts["points"]), allow_pickle=False),
        dtype=np.float32,
    )
    colors = np.asarray(
        np.load(_resolve_artifact(manifest_path, artifacts["colors"]), allow_pickle=False),
        dtype=np.float32,
    )
    normals = None
    if input_features == "rgbn6":
        normals = np.asarray(
            np.load(
                _resolve_artifact(manifest_path, artifacts["normals"]),
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
    if (
        coordinate_normalization_policy
        == STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
    ):
        if input_features != "rgbn6" or normals is None:
            raise ValueError(
                f"{coordinate_normalization_policy} requires RGBN6 points and normals"
            )
        points, normals = canonicalize_structured3d_points_normals_to_scannet(
            points,
            normals,
        )
    features = build_input_features(
        points,
        colors,
        use_colors=True,
        use_normals=input_features == "rgbn6",
        normals=normals,
        color_mean=color_mean,
        color_std=color_std,
    ).astype(np.float32, copy=False)
    labels_by_key = {
        key: np.asarray(
            np.load(
                _resolve_artifact(manifest_path, artifacts[f"labels_{key}"]),
                allow_pickle=False,
            ),
            dtype=np.int64,
        )
        for key in GRANULARITY_KEYS
    }
    n = int(points.shape[0])
    if (
        points.shape != (n, 3)
        or colors.shape != (n, 3)
        or features.shape != (n, input_channels)
        or (normals is not None and normals.shape != (n, 3))
    ):
        raise ValueError(f"{manifest_path}: point/input shape drift")
    if any(labels.shape != (n,) for labels in labels_by_key.values()):
        raise ValueError(f"{manifest_path}: label shape drift")

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
