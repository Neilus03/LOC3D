"""Multi-scene dataset for epoch-based training.

Wraps multiple CHORUS training-pack scenes into a single
:class:`torch.utils.data.Dataset` where ``__len__`` is the
number of scenes and ``__getitem__(i)`` yields the same dict
structure as ``MultiGranSceneDataset[0]``.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from delimit3d.tasks.mask3d.data.region_sampling import sphere_crop_indices
from delimit3d.tasks.mask3d.data.training_pack import (
    MultiGranTrainingPackScene,
    _load_scene_meta,
    _resolve_pack_dir,
    load_training_pack_scene_multi,
)
from delimit3d.tasks.mask3d.data.point_augmentations import (
    augment_points_litept_scannet,
    augment_points_mask3d_scannet,
    augment_points_mask3d_scannet_rgbn6,
)
from delimit3d.tasks.mask3d.data.scannet_gt_cache import (
    load_cached_scannet_gt,
    validate_cache_manifest,
)
from delimit3d.tasks.mask3d.data.single_scene_dataset import build_input_features

from .aligned_augmentation import augment_points_mask3d_scannet_rgbn6_aligned

log = logging.getLogger(__name__)

_LABEL_SOURCE_PACK = "pack"
_LABEL_SOURCE_SCANNET_GT = "scannet_gt"


def _normals_array_for_scene(
    scene: MultiGranTrainingPackScene,
    *,
    use_normals: bool,
    warned_ids: set[str],
    require_normals: bool = False,
) -> np.ndarray | None:
    """Return per-point normals for augmentation/features, or None if disabled."""
    if not use_normals:
        return None
    if scene.normals is not None:
        return np.asarray(scene.normals, dtype=np.float32)
    if require_normals:
        raise RuntimeError(
            f"[{scene.scene_id}] mask3d_scannet_rgbn6 requires normals.npy"
        )
    if scene.scene_id not in warned_ids:
        warned_ids.add(scene.scene_id)
        log.warning(
            "[%s] use_normals=True but normals.npy missing; using zeros",
            scene.scene_id,
        )
    return np.zeros((scene.num_points, 3), dtype=np.float32)


def build_scene_list(
    scene_list_file: Path,
    scans_root: Path,
) -> list[Path]:
    """Read a text file of scene IDs and resolve to full paths.

    Parameters
    ----------
    scene_list_file:
        Text file with one scene ID per line.  Blank lines and lines
        starting with ``#`` are skipped.
    scans_root:
        Root directory containing scene subdirectories.

    Returns
    -------
    List of resolved, validated scene directory paths.

    Raises
    ------
    FileNotFoundError
        If the list file, a scene directory, or its training pack is missing.
    ValueError
        If the file contains no scenes.
    """
    scene_list_file = Path(scene_list_file)
    scans_root = Path(scans_root)

    if not scene_list_file.is_file():
        raise FileNotFoundError(f"Scene list file not found: {scene_list_file}")

    scene_dirs: list[Path] = []
    for line in scene_list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scene_dir = scans_root / line
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        _resolve_pack_dir(scene_dir)
        scene_dirs.append(scene_dir)

    if not scene_dirs:
        raise ValueError(f"No scenes found in {scene_list_file}")

    log.info("Loaded %d scene paths from %s", len(scene_dirs), scene_list_file)
    return scene_dirs


def build_scene_list_from_root(scans_root: Path) -> list[Path]:
    """Return all scene directories under *scans_root* that contain a training pack."""
    scans_root = Path(scans_root)
    if not scans_root.is_dir():
        raise FileNotFoundError(f"Scans root not found: {scans_root}")

    scene_dirs: list[Path] = []
    for scene_dir in sorted(path for path in scans_root.iterdir() if path.is_dir()):
        try:
            _resolve_pack_dir(scene_dir)
        except FileNotFoundError:
            continue
        scene_dirs.append(scene_dir)

    if not scene_dirs:
        raise ValueError(f"No training-pack scenes found under {scans_root}")

    log.info("Loaded %d scene paths from %s", len(scene_dirs), scans_root)
    return scene_dirs


class MultiSceneDataset(Dataset):
    """Dataset that iterates over multiple scenes.

    Each item is a complete sample dict with the same keys and types
    as ``MultiGranSceneDataset[0]``.

    Parameters
    ----------
    scene_dirs:
        Resolved scene directory paths.
    granularities:
        Dot-free granularity keys, e.g. ``("g02", "g05", "g08")``.
    use_colors:
        Use RGB from ``colors.npy`` as input features.
    append_xyz:
        Append XYZ coordinates to the feature vector.
    use_normals:
        If True, concatenate mesh normals (from ``normals.npy`` or zeros if missing).
    preload:
        Load all scenes into memory at init time.  When
        ``scannet_gt_cache_root`` is also set, each GT npz is loaded once
        here so ``__getitem__`` does not re-read pack or GT files.
    preload_workers:
        Thread count for the preload pass.  I/O-bound; keep modest on Lustre.
    max_points:
        If set, caps the number of points per ``__getitem__`` (same indices for
        points, features, labels, masks).  Ignored when *subsampling_mode* is
        ``"none"`` and neither *max_points* nor *sphere_point_max* applies.
    subsampling_mode:
        ``"sphere_crop"`` (default): Point-style sphere around a random point,
        keep nearest *cap* points.  ``"randperm"``: legacy global random subset.
        ``"none"``: never subsample (full scene each step; needs enough GPU RAM).
    sphere_point_max:
        Optional override for the sphere crop size; defaults to *max_points*
        when unset.
    train_augmentations:
        If True, apply the selected ScanNet training transform on training
        samples only. Ignores precomputed feature cache so augmentations vary
        each epoch.
    augmentation_profile:
        ``"litept_scannet"`` preserves the historical LitePT transform;
        ``"mask3d_scannet"`` matches the released RGB-only Mask3D transform;
        ``"mask3d_scannet_rgbn6"`` is its normal-aware RGBN6 variant with
        elastic distortion disabled.
    color_mean / color_std:
        Optional three-channel RGB normalization applied after augmentation.
    scannet_gt_supervise_all_points:
        When ``label_source="scannet_gt"``, keep every point in the mask-loss
        supervision domain.  ScanNet GT points with id 0 remain ``-1`` labels,
        so they are not target instances, but they become negatives for every
        foreground instance mask.  Pack/pseudo-label pretraining is unaffected.
    scannet_gt_cache_root:
        Optional materialized ScanNet GT cache.  The cache must have been
        created by ``build_litept_mask3d_scannet_gt_cache.py`` and is checked
        against the requested benchmark and scene set at construction time.
    """

    def __init__(
        self,
        scene_dirs: list[Path],
        granularities: tuple[str, ...],
        *,
        use_colors: bool = True,
        append_xyz: bool = False,
        use_normals: bool = False,
        preload: bool = True,
        max_points: int | None = None,
        subsampling_mode: str = "sphere_crop",
        sphere_point_max: int | None = None,
        train_augmentations: bool = False,
        augmentation_profile: str = "litept_scannet",
        color_mean: Sequence[float] | None = None,
        color_std: Sequence[float] | None = None,
        label_source: str = _LABEL_SOURCE_PACK,
        scannet_eval_benchmark: str = "all",
        scannet_gt_supervise_all_points: bool = False,
        scannet_gt_cache_root: str | Path | None = None,
        preload_workers: int = 1,
        scene_sources: Sequence[str] | None = None,
        default_source_name: str = "default",
    ) -> None:
        super().__init__()
        self._scene_dirs = list(scene_dirs)
        if scene_sources is None:
            self._scene_source_names = [str(default_source_name)] * len(self._scene_dirs)
        else:
            self._scene_source_names = [str(source) for source in scene_sources]
            if len(self._scene_source_names) != len(self._scene_dirs):
                raise ValueError(
                    "scene_sources length must match scene_dirs length: "
                    f"{len(self._scene_source_names)} != {len(self._scene_dirs)}"
                )
        source_to_idx: dict[str, int] = {}
        source_names: list[str] = []
        source_indices: list[int] = []
        for source_name in self._scene_source_names:
            if source_name not in source_to_idx:
                source_to_idx[source_name] = len(source_to_idx)
                source_names.append(source_name)
            source_indices.append(source_to_idx[source_name])
        self._source_names = source_names
        self._scene_source_indices = source_indices
        self._granularities = granularities
        self._use_colors = use_colors
        self._append_xyz = append_xyz
        self._use_normals = bool(use_normals)
        self._missing_normals_warned: set[str] = set()
        self._max_points = max_points
        self._subsampling_mode = subsampling_mode
        self._sphere_point_max = sphere_point_max
        self._train_augmentations = bool(train_augmentations)
        self._augmentation_profile = str(augmentation_profile).strip().lower()
        augmentation_profiles = {
            "litept_scannet",
            "mask3d_scannet",
            "mask3d_scannet_rgbn6",
        }
        if self._augmentation_profile not in augmentation_profiles:
            raise ValueError(
                "augmentation_profile must be one of "
                f"{sorted(augmentation_profiles)}, got {augmentation_profile!r}"
            )
        if (
            self._augmentation_profile == "mask3d_scannet_rgbn6"
            and not self._use_normals
        ):
            raise ValueError("mask3d_scannet_rgbn6 requires use_normals=True")
        if (color_mean is None) != (color_std is None):
            raise ValueError("color_mean and color_std must be supplied together")
        self._color_mean = None if color_mean is None else tuple(float(x) for x in color_mean)
        self._color_std = None if color_std is None else tuple(float(x) for x in color_std)
        if self._color_mean is not None and (
            len(self._color_mean) != 3
            or len(self._color_std or ()) != 3
            or any(value <= 0 for value in (self._color_std or ()))
        ):
            raise ValueError("color_mean and color_std must each contain three valid values")
        self._label_source = str(label_source)
        self._scannet_eval_benchmark = str(scannet_eval_benchmark)
        self._scannet_gt_supervise_all_points = bool(scannet_gt_supervise_all_points)
        self._scannet_gt_cache_root = (
            Path(scannet_gt_cache_root).expanduser().resolve()
            if scannet_gt_cache_root is not None
            else None
        )
        self._scenes: list[MultiGranTrainingPackScene] = []
        self._gt_cache: list[
            tuple[np.ndarray, np.ndarray, dict[int, int]] | None
        ] = []
        self._scene_point_counts: list[int] = []
        self._features_cache: list[np.ndarray] | None = [] if not train_augmentations else None
        self._preload_workers = max(int(preload_workers), 1)

        if self._label_source not in (_LABEL_SOURCE_PACK, _LABEL_SOURCE_SCANNET_GT):
            raise ValueError(
                f"label_source must be one of "
                f"{(_LABEL_SOURCE_PACK, _LABEL_SOURCE_SCANNET_GT)}, got {self._label_source!r}"
            )
        if self._scannet_gt_cache_root is not None:
            if self._label_source != _LABEL_SOURCE_SCANNET_GT:
                raise ValueError(
                    "scannet_gt_cache_root requires label_source='scannet_gt'"
                )
            validate_cache_manifest(
                self._scannet_gt_cache_root,
                (scene_dir.name for scene_dir in self._scene_dirs),
                eval_benchmark=self._scannet_eval_benchmark,
            )

        if self._subsampling_mode not in ("sphere_crop", "randperm", "none"):
            raise ValueError(
                f"subsampling_mode must be 'sphere_crop', 'randperm', or 'none', "
                f"got {self._subsampling_mode!r}",
            )
        if self._subsampling_mode == "none" and (
            self._max_points is not None or self._sphere_point_max is not None
        ):
            log.warning(
                "subsampling_mode=none: ignoring max_points / sphere_point_max (full scenes)",
            )

        if (
            self._train_augmentations
            and self._augmentation_profile != "mask3d_scannet_rgbn6"
        ):
            try:
                import scipy.ndimage  # noqa: F401
            except ImportError:
                log.warning(
                    "train_augmentations=True but scipy is not installed; "
                    "elastic distortion will be skipped (install scipy for full LitePT-style aug).",
                )
            else:
                log.info(
                    "train_augmentations=True (profile=%s; elastic distortion enabled)",
                    self._augmentation_profile,
                )
        elif self._train_augmentations:
            log.info(
                "train_augmentations=True (profile=mask3d_scannet_rgbn6; "
                "normal-aware geometry, elastic distortion disabled)",
            )

        self._output_cap: int | None
        if self._sphere_point_max is not None:
            self._output_cap = int(self._sphere_point_max)
        elif self._max_points is not None:
            self._output_cap = int(self._max_points)
        else:
            self._output_cap = None

        if preload:
            self._preload_scenes()
        else:
            for sd in self._scene_dirs:
                meta = _load_scene_meta(_resolve_pack_dir(sd))
                effective_points = int(meta["num_points"])
                if self._output_cap is not None:
                    effective_points = min(effective_points, self._output_cap)
                self._scene_point_counts.append(effective_points)

    def _preload_one_scene(
        self, scene_dir: Path
    ) -> tuple[
        MultiGranTrainingPackScene,
        np.ndarray | None,
        tuple[np.ndarray, np.ndarray, dict[int, int]] | None,
        int,
    ]:
        scene = load_training_pack_scene_multi(scene_dir, self._granularities)
        feats: np.ndarray | None = None
        if self._features_cache is not None:
            n_arr = _normals_array_for_scene(
                scene,
                use_normals=self._use_normals,
                warned_ids=self._missing_normals_warned,
                require_normals=(
                    self._augmentation_profile == "mask3d_scannet_rgbn6"
                ),
            )
            feats = build_input_features(
                scene.points,
                scene.colors,
                use_colors=self._use_colors,
                append_xyz=self._append_xyz,
                use_normals=self._use_normals,
                normals=n_arr if self._use_normals else None,
                color_mean=self._color_mean,
                color_std=self._color_std,
            )
        gt_record: tuple[np.ndarray, np.ndarray, dict[int, int]] | None = None
        if (
            self._label_source == _LABEL_SOURCE_SCANNET_GT
            and self._scannet_gt_cache_root is not None
        ):
            gt_record = load_cached_scannet_gt(
                self._scannet_gt_cache_root,
                scene.scene_id,
                expected_num_points=scene.num_points,
            )
        effective_points = scene.num_points
        if self._output_cap is not None:
            effective_points = min(effective_points, self._output_cap)
        return scene, feats, gt_record, int(effective_points)

    def _preload_scenes(self) -> None:
        n_scenes = len(self._scene_dirs)
        workers = min(self._preload_workers, n_scenes) if n_scenes else 1
        log.info(
            "Preloading %d scenes (workers=%d, gt_cache=%s, augmentations=%s)",
            n_scenes,
            workers,
            self._scannet_gt_cache_root is not None
            and self._label_source == _LABEL_SOURCE_SCANNET_GT,
            self._train_augmentations,
        )
        if n_scenes == 0:
            return
        if workers <= 1:
            loaded = [self._preload_one_scene(path) for path in self._scene_dirs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                loaded = list(pool.map(self._preload_one_scene, self._scene_dirs))
        gt_cached = 0
        for scene, feats, gt_record, effective_points in loaded:
            self._scenes.append(scene)
            self._gt_cache.append(gt_record)
            self._scene_point_counts.append(effective_points)
            if self._features_cache is not None:
                if feats is None:
                    raise RuntimeError(
                        f"[{scene.scene_id}] preload feature cache is missing"
                    )
                self._features_cache.append(feats)
            if gt_record is not None:
                gt_cached += 1
        log.info(
            "Preloaded %d scenes (%d with ScanNet GT arrays, features_cached=%s)",
            n_scenes,
            gt_cached,
            self._features_cache is not None,
        )

    def __len__(self) -> int:
        return len(self._scene_dirs)

    def _should_subsample(self) -> bool:
        if self._subsampling_mode == "none":
            return False
        return self._output_cap is not None

    def _subsample_indices(self, n: int, points_np: np.ndarray) -> torch.Tensor:
        cap = self._output_cap
        assert cap is not None
        if n <= cap:
            return torch.arange(n, dtype=torch.long)

        if self._subsampling_mode == "randperm":
            return torch.randperm(n)[:cap]

        if self._subsampling_mode == "sphere_crop":
            assert self._output_cap is not None
            rng = np.random.default_rng()
            idx_np = sphere_crop_indices(
                np.asarray(points_np, dtype=np.float32),
                rng=rng,
                point_max=int(self._output_cap),
            )
            return torch.from_numpy(idx_np).long()

        raise RuntimeError(f"Unhandled subsampling_mode={self._subsampling_mode!r}")

    def _compose_item(self, idx: int, *, subsample: bool) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self._scenes:
            scene = self._scenes[idx]
            base_points = scene.points
            base_colors = scene.colors
        else:
            scene = load_training_pack_scene_multi(
                self._scene_dirs[idx], self._granularities,
            )
            base_points = scene.points
            base_colors = scene.colors

        n_arr = _normals_array_for_scene(
            scene,
            use_normals=self._use_normals,
            warned_ids=self._missing_normals_warned,
            require_normals=(
                self._augmentation_profile == "mask3d_scannet_rgbn6"
            ),
        )

        if self._train_augmentations:
            augmentations = {
                "litept_scannet": augment_points_litept_scannet,
                "mask3d_scannet": (augment_points_mask3d_scannet_rgbn6_aligned if self._use_normals else augment_points_mask3d_scannet),
                "mask3d_scannet_rgbn6": augment_points_mask3d_scannet_rgbn6,
            }
            augmentation = augmentations[self._augmentation_profile]
            aug_pts, aug_cols, aug_nrm = augmentation(
                base_points,
                base_colors,
                use_colors=self._use_colors,
                normals=n_arr,
            )
            features = build_input_features(
                aug_pts,
                aug_cols,
                use_colors=self._use_colors,
                append_xyz=self._append_xyz,
                use_normals=self._use_normals,
                normals=aug_nrm if self._use_normals else None,
                color_mean=self._color_mean,
                color_std=self._color_std,
            )
            points_np = aug_pts
        elif self._scenes and self._features_cache is not None:
            features = self._features_cache[idx]
            points_np = base_points
        else:
            features = build_input_features(
                base_points,
                base_colors,
                use_colors=self._use_colors,
                append_xyz=self._append_xyz,
                use_normals=self._use_normals,
                normals=n_arr if self._use_normals else None,
                color_mean=self._color_mean,
                color_std=self._color_std,
            )
            points_np = base_points

        labels_by_gran = {
            g: torch.from_numpy(scene.labels_by_granularity[g]).long()
            for g in self._granularities
        }
        instance_classes_by_gran: dict[str, dict[int, int] | None] = {
            g: None for g in self._granularities
        }

        points_t = torch.from_numpy(np.asarray(points_np)).float()
        features_t = torch.from_numpy(features).float()
        valid_np = scene.valid_points
        sup_np = scene.supervision_mask
        scan_segments_np: np.ndarray | None = None
        if self._label_source == _LABEL_SOURCE_SCANNET_GT:
            cached_gt = None
            if self._scenes and idx < len(self._gt_cache):
                cached_gt = self._gt_cache[idx]
            if cached_gt is not None:
                cached_labels, cached_segments, cached_class_map = cached_gt
                gt_labels = np.array(cached_labels, dtype=np.int64, copy=True)
                scan_segments_np = np.array(cached_segments, dtype=np.int64, copy=True)
                gt_class_map = dict(cached_class_map)
            elif self._scannet_gt_cache_root is not None:
                gt_labels, scan_segments_np, gt_class_map = load_cached_scannet_gt(
                    self._scannet_gt_cache_root,
                    scene.scene_id,
                    expected_num_points=scene.num_points,
                )
            else:
                from delimit3d.source.datasets.scannet.gt import load_scannet_gt_instances

                gt_instances = load_scannet_gt_instances(
                    scene.scene_dir,
                    scene.scene_id,
                    eval_benchmark=self._scannet_eval_benchmark,
                )
                if gt_instances.segment_ids is None:
                    raise RuntimeError(
                        f"[{scene.scene_id}] ScanNet GT loader did not return segment ids"
                    )
                scan_segments_np = np.asarray(gt_instances.segment_ids, dtype=np.int64)
                if scan_segments_np.shape != (scene.num_points,):
                    raise RuntimeError(
                        f"[{scene.scene_id}] segment ids {scan_segments_np.shape} do not "
                        f"align with training-pack vertices ({scene.num_points},)"
                    )
                if gt_instances.vertex_xyz is None:
                    raise RuntimeError(
                        f"[{scene.scene_id}] ScanNet GT loader did not return mesh XYZ"
                    )
                mesh_xyz = np.asarray(gt_instances.vertex_xyz, dtype=np.float32)
                source_xyz = np.asarray(base_points, dtype=np.float32)
                if mesh_xyz.shape != source_xyz.shape or not np.allclose(
                    mesh_xyz, source_xyz, rtol=0.0, atol=1e-5
                ):
                    max_error = (
                        float(np.max(np.abs(mesh_xyz - source_xyz)))
                        if mesh_xyz.shape == source_xyz.shape and mesh_xyz.size
                        else float("inf")
                    )
                    raise RuntimeError(
                        f"[{scene.scene_id}] training-pack point order is not aligned "
                        f"with the ScanNet mesh (max_abs_xyz_error={max_error})"
                    )
                gt_labels = np.asarray(gt_instances.instance_ids, dtype=np.int64)
                gt_class_map = dict(gt_instances.instance_class_ids)
            # Convert ScanNet GT convention (0=ignore/background) to student convention (-1=ignore).
            gt_labels = np.where(gt_labels == 0, -1, gt_labels)

            labels_by_gran = {
                g: torch.from_numpy(gt_labels).long()
                for g in self._granularities
            }
            instance_classes_by_gran = {
                g: dict(gt_class_map)
                for g in self._granularities
            }
            if self._scannet_gt_supervise_all_points:
                sup_np = np.ones_like(gt_labels, dtype=bool)
            else:
                sup_np = gt_labels != -1
            valid_np = sup_np.copy()

        valid_t = torch.from_numpy(valid_np).bool()
        seen_t = torch.from_numpy(scene.seen_points).bool()
        sup_t = torch.from_numpy(sup_np).bool()

        n = points_t.shape[0]
        original_num_points = int(n)
        vertex_indices: torch.Tensor | None = None
        if subsample and self._should_subsample():
            idx_t = self._subsample_indices(n, np.asarray(points_np, dtype=np.float32))
            is_identity = (
                int(idx_t.numel()) == n
                and torch.equal(idx_t.cpu(), torch.arange(n, dtype=torch.long))
            )
            vertex_indices = None if is_identity else idx_t.clone()
            points_t = points_t[idx_t]
            features_t = features_t[idx_t]
            valid_t = valid_t[idx_t]
            seen_t = seen_t[idx_t]
            sup_t = sup_t[idx_t]
            labels_by_gran = {g: labels_by_gran[g][idx_t] for g in self._granularities}
            if scan_segments_np is not None:
                scan_segments_np = scan_segments_np[idx_t.cpu().numpy()]

        out: dict[str, Any] = {
            "scene_id": scene.scene_id,
            "scene_dir": str(scene.scene_dir),
            "source_name": self._scene_source_names[idx],
            "source_index": self._scene_source_indices[idx],
            "points": points_t,
            "features": features_t,
            "labels_by_granularity": labels_by_gran,
            "instance_classes_by_granularity": instance_classes_by_gran,
            "valid_points": valid_t,
            "seen_points": seen_t,
            "supervision_mask": sup_t,
            "scene_meta": scene.scene_meta,
            "granularities": self._granularities,
            "original_num_points": original_num_points,
            "eval_input_points": int(points_t.shape[0]),
            "full_scene": vertex_indices is None and int(points_t.shape[0]) == original_num_points,
            "subsampling_mode": self._subsampling_mode,
            "scan_segments": (
                torch.from_numpy(scan_segments_np).long()
                if scan_segments_np is not None
                else None
            ),
        }
        if vertex_indices is not None:
            out["vertex_indices"] = vertex_indices
        return out

    def __getitem__(self, idx: int | tuple[int, bool]) -> dict[str, Any]:
        is_padding = False
        if isinstance(idx, tuple):
            idx, is_padding = idx
        item = self._compose_item(int(idx), subsample=True)
        if is_padding:
            item["is_padding"] = True
        return item

    def get_full_item(self, idx: int) -> dict[str, Any]:
        """Same as ``__getitem__`` but without point subsampling (full scene).

        For optional fragment-based evaluation that needs the full vertex list.
        """
        return self._compose_item(idx, subsample=False)

    @property
    def scene_ids(self) -> list[str]:
        """List of all scene IDs in dataset order."""
        if self._scenes:
            return [s.scene_id for s in self._scenes]
        return [d.name for d in self._scene_dirs]

    @property
    def scene_source_names(self) -> list[str]:
        """Source name for each scene in dataset order."""
        return list(self._scene_source_names)

    @property
    def scene_source_indices(self) -> list[int]:
        """Integer source id for each scene in dataset order."""
        return list(self._scene_source_indices)

    @property
    def source_names(self) -> list[str]:
        """Unique source names in stable first-seen order."""
        return list(self._source_names)

    @property
    def scene_point_counts(self) -> list[int]:
        """Effective point counts after optional max-point clipping."""
        return list(self._scene_point_counts)


# ── smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Smoke test for MultiSceneDataset",
    )
    parser.add_argument("--scene-list", type=str, required=True)
    parser.add_argument("--scans-root", type=str, required=True)
    args = parser.parse_args()

    dirs = build_scene_list(Path(args.scene_list), Path(args.scans_root))
    ds = MultiSceneDataset(dirs, granularities=("g02", "g05", "g08"))

    print(f"\nlen(dataset): {len(ds)}")
    for i in range(len(ds)):
        s = ds[i]
        print(
            f"  [{i:2d}] {s['scene_id']:16s}  "
            f"points={tuple(s['points'].shape)}  "
            f"features={tuple(s['features'].shape)}  "
            f"grans={list(s['labels_by_granularity'].keys())}  "
            f"supervised={s['supervision_mask'].sum().item()}"
        )
