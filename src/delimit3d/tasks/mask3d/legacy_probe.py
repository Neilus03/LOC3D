"""Deterministic matched ScanNet-train development probes for V2 transfer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from delimit3d.models.litept_v2_transfer import (
    sha256_file,
    tensor_state_sha256,
    validate_backbone_export,
)
from delimit3d.tasks.mask3d.model import (
    MASK3D_DEC0_POOLED_MEMORY_CHANNELS,
    MASK3D_DECODER_ONLY_MEMORY_CHANNELS,
    MASK3D_MEMORY_CHANNELS,
)


SUBSET_SCHEMA = "litept_mask3d_scannet_train_dev_subset/v1"
PROBE_BUNDLE_SCHEMA = "litept_mask3d_v2_dev_probe_bundle/v1"
PROBE_ARM_SCHEMA = "litept_mask3d_v2_dev_probe_arm/v1"
PROBE_PAIR_SCHEMA = "litept_mask3d_v2_dev_probe_pair/v1"
PROBE_RESULT_SCHEMA = "litept_mask3d_v2_dev_probe_result/v1"
PROBE_AGGREGATE_SCHEMA = "litept_mask3d_v2_dev_probe_aggregate/v1"
DECODER_TEMPLATE_SCHEMA = "litept_mask3d_decoder_template/v1"
SCANNET_GT_CACHE_VERSION = 1
SCANNET20_CLASS_SLOT_COUNT = 20
TRAIN_SCENE_COUNT = 50
DEVELOPMENT_SCENE_COUNT = 10
PROBE_UPDATES = 1_000
PROBE_TRAIN_CAPACITY_PROFILE = "sphere_50k_train_full_scene_eval_v1"
PROBE_TRAIN_MAX_POINTS = 50_000
AP_METRICS = ("AP", "AP50", "AP25")
QUERY_METRICS = (
    "valid_proposals_per_scene",
    "foreground_query_fraction",
    "no_object_query_fraction",
)
STRATIFICATION_FIELDS = (
    "num_points",
    "num_foreground_instances",
    "foreground_fraction",
    "num_foreground_classes",
)
FEATURE_PYRAMID_MEMORY_CHANNELS = {
    "mask3d_fpn": MASK3D_MEMORY_CHANNELS,
    "mask3d_dec0_pooled_fpn": MASK3D_DEC0_POOLED_MEMORY_CHANNELS,
    "mask3d_decoder_only_fpn": MASK3D_DECODER_ONLY_MEMORY_CHANNELS,
}


def _probe_train_capacity_contract() -> dict[str, Any]:
    """The explicit 24 GiB training envelope for both matched arms.

    ``run_litept_mask3d.py`` applies these subsampling values only to training
    samples; the fixed ten-scene development evaluation remains full-scene.
    Keeping the distinction in the manifest avoids silently turning a capacity
    workaround into a different evaluation protocol.
    """

    return {
        "profile": PROBE_TRAIN_CAPACITY_PROFILE,
        "subsampling_mode": "sphere_crop",
        "max_points": PROBE_TRAIN_MAX_POINTS,
        "sphere_point_max": PROBE_TRAIN_MAX_POINTS,
        "evaluation_point_policy": "full_scene_train_cap_ignored",
    }


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256("/".join(str(value) for value in parts).encode()).digest()[:8],
        "little",
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expand_config_environment(value: Any) -> Any:
    """Resolve the same environment syntax as config_utils.load_config."""

    if isinstance(value, Mapping):
        return {key: _expand_config_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_environment(item) for item in value)
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        if ":-" in expression:
            variable, default = expression.split(":-", 1)
            observed = os.environ.get(variable, "")
            return observed if observed else default
        return os.environ.get(expression, match.group(0))

    expanded = os.path.expanduser(os.path.expandvars(pattern.sub(replace, value)))
    if re.search(r"\$(?:\{|[A-Za-z_])", expanded):
        raise ValueError(f"unresolved environment expression in base config: {expanded!r}")
    return expanded


def read_scene_ids(path: Path) -> list[str]:
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"scene split must be non-empty and unique: {path}")
    return values


def ordered_scene_sha256(scene_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in scene_ids).encode("utf-8")
    ).hexdigest()


def collect_scannet_train_statistics(
    *,
    official_train_split: Path,
    scans_root: Path,
    gt_cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Collect selection-only statistics from available official-train scenes."""

    train_path = Path(official_train_split).resolve(strict=True)
    scan_root = Path(scans_root).resolve(strict=True)
    cache_root = Path(gt_cache_root).resolve(strict=True)
    scene_ids = read_scene_ids(train_path)
    cache_manifest_path = cache_root / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (
        cache_manifest.get("cache_version") != SCANNET_GT_CACHE_VERSION
        or cache_manifest.get("eval_benchmark") != "scannet20"
        or not isinstance(cache_manifest.get("scenes"), Mapping)
    ):
        raise ValueError(f"invalid ScanNet20 GT cache manifest: {cache_manifest_path}")
    cache_rows = cache_manifest["scenes"]
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for scene_id in scene_ids:
        scene_dir = scan_root / scene_id
        points_candidates = (
            scene_dir / "training_pack" / "points.npy",
            scene_dir / "points.npy",
        )
        points_path = next((path for path in points_candidates if path.is_file()), None)
        cache_row = cache_rows.get(scene_id)
        cache_path = cache_root / f"{scene_id}.npz"
        if not scene_dir.is_dir() or points_path is None:
            excluded.append({"scene_id": scene_id, "reason": "training_pack_unavailable"})
            continue
        if not isinstance(cache_row, Mapping) or not cache_path.is_file():
            excluded.append({"scene_id": scene_id, "reason": "gt_cache_unavailable"})
            continue
        points = np.load(points_path, mmap_mode="r", allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"{points_path}: expected non-empty [N,3]")
        with np.load(cache_path, allow_pickle=False) as cache:
            instance_ids = np.asarray(cache["instance_ids"], dtype=np.int64)
            class_pairs = np.asarray(cache["instance_class_pairs"], dtype=np.int64)
        if instance_ids.shape != (points.shape[0],):
            raise ValueError(f"{scene_id}: GT cache and training points do not align")
        if class_pairs.ndim != 2 or class_pairs.shape[1] != 2:
            raise ValueError(f"{scene_id}: malformed instance_class_pairs")
        # The shared ScanNet20 cache stores 20 benchmark-contiguous slots:
        # wall/floor at 0/1 and the 18 Mask3D instance classes at 2..19.  This
        # is the same conversion contract used by run_litept_mask3d.py.
        if class_pairs.size and (
            int(class_pairs[:, 1].min()) < 0
            or int(class_pairs[:, 1].max()) >= SCANNET20_CLASS_SLOT_COUNT
        ):
            raise ValueError(f"{scene_id}: ScanNet20 cache class ID outside [0,19]")
        foreground_pairs = class_pairs[
            (class_pairs[:, 1] >= 2) & (class_pairs[:, 1] < 20)
        ]
        foreground_instances = np.unique(foreground_pairs[:, 0])
        foreground_mask = np.isin(instance_ids, foreground_instances)
        if foreground_instances.size == 0 or not bool(foreground_mask.any()):
            excluded.append(
                {
                    "scene_id": scene_id,
                    "reason": "no_mask3d_foreground_for_per_scene_ap",
                }
            )
            continue
        records.append(
            {
                "scene_id": scene_id,
                "num_points": int(points.shape[0]),
                "num_foreground_instances": int(foreground_instances.size),
                "foreground_fraction": float(np.mean(foreground_mask)),
                "num_foreground_classes": int(
                    np.unique(foreground_pairs[:, 1]).size
                ),
                "points_path": str(points_path.resolve(strict=True)),
                "gt_cache_path": str(cache_path.resolve(strict=True)),
            }
        )
    provenance = {
        "official_train_split": str(train_path),
        "official_train_split_sha256": sha256_file(train_path),
        "official_train_order_sha256": ordered_scene_sha256(scene_ids),
        "official_train_scene_count": len(scene_ids),
        "scans_root": str(scan_root),
        "gt_cache_root": str(cache_root),
        "gt_cache_manifest": str(cache_manifest_path.resolve(strict=True)),
        "gt_cache_manifest_sha256": sha256_file(cache_manifest_path),
        "selection_source_split": "ScanNet official train only",
    }
    return records, excluded, provenance


def _rank_bins(
    records: Sequence[Mapping[str, Any]],
    field: str,
    bins: int,
) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda row: (float(row[field]), str(row["scene_id"])),
    )
    return {
        str(row["scene_id"]): min(bins - 1, index * bins // len(ordered))
        for index, row in enumerate(ordered)
    }


def _proportional_quotas(
    populations: Mapping[tuple[int, ...], int],
    count: int,
    *,
    seed: int,
    label: str,
) -> dict[tuple[int, ...], int]:
    total = int(sum(populations.values()))
    if not 0 <= int(count) <= total:
        raise ValueError(f"invalid proportional allocation {count}/{total}")
    quotas = {
        key: min(size, int(count) * size // total)
        for key, size in populations.items()
    }
    remaining = int(count) - sum(quotas.values())
    while remaining:
        candidates = [key for key, size in populations.items() if quotas[key] < size]
        chosen = max(
            candidates,
            key=lambda key: (
                int(count) * populations[key] / total - quotas[key],
                -_stable_seed(seed, label, *key),
            ),
        )
        quotas[chosen] += 1
        remaining -= 1
    return quotas


def stratified_train_development_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    official_train_scene_ids: Sequence[str],
    seed: int = 42,
    bins_per_feature: int = 4,
    train_count: int = TRAIN_SCENE_COUNT,
    development_count: int = DEVELOPMENT_SCENE_COUNT,
) -> dict[str, Any]:
    """Choose disjoint 50/10 roles from official ScanNet train only."""

    if (
        int(train_count) != TRAIN_SCENE_COUNT
        or int(development_count) != DEVELOPMENT_SCENE_COUNT
    ):
        raise ValueError(
            "the development-probe panel is fixed at exactly "
            f"{TRAIN_SCENE_COUNT} train and {DEVELOPMENT_SCENE_COUNT} development scenes"
        )
    rows = [dict(row) for row in records]
    total_count = int(train_count) + int(development_count)
    if len(rows) < total_count:
        raise ValueError(f"need at least {total_count} available official-train scenes")
    official = list(official_train_scene_ids)
    official_set = set(official)
    if len(official) != len(official_set):
        raise ValueError("official train scene list contains duplicates")
    row_ids = [str(row.get("scene_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("scene statistics contain duplicate IDs")
    outside = sorted(set(row_ids) - official_set)
    if outside:
        raise ValueError(f"selection candidates outside official ScanNet train: {outside[:20]}")
    for row in rows:
        for field in STRATIFICATION_FIELDS:
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"{row.get('scene_id')}: invalid {field}")
    bins = int(bins_per_feature)
    if bins < 2:
        raise ValueError("bins_per_feature must be at least two")
    assignments = {
        field: _rank_bins(rows, field, bins) for field in STRATIFICATION_FIELDS
    }
    strata: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        stratum = tuple(assignments[field][scene_id] for field in STRATIFICATION_FIELDS)
        row["stratum"] = list(stratum)
        strata.setdefault(stratum, []).append(row)
    selection_quotas = _proportional_quotas(
        {key: len(value) for key, value in strata.items()},
        total_count,
        seed=int(seed),
        label="panel",
    )
    selected_by_stratum: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for stratum, values in strata.items():
        ordered = sorted(
            values,
            key=lambda row: (
                _stable_seed(seed, "select", row["scene_id"]),
                str(row["scene_id"]),
            ),
        )
        selected_by_stratum[stratum] = ordered[: selection_quotas[stratum]]
    development_quotas = _proportional_quotas(
        {key: len(value) for key, value in selected_by_stratum.items()},
        int(development_count),
        seed=int(seed),
        label="development",
    )
    train_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for stratum in sorted(selected_by_stratum):
        selected = sorted(
            selected_by_stratum[stratum],
            key=lambda row: (
                _stable_seed(seed, "role", row["scene_id"]),
                str(row["scene_id"]),
            ),
        )
        dev_quota = development_quotas[stratum]
        dev = selected[:dev_quota]
        train = selected[dev_quota:]
        development_rows.extend(dev)
        train_rows.extend(train)
        stratum_rows.append(
            {
                "stratum": list(stratum),
                "candidate_count": len(strata[stratum]),
                "selected_count": len(selected),
                "train_count": len(train),
                "development_count": len(dev),
            }
        )
    train_rows.sort(
        key=lambda row: (
            _stable_seed(seed, "train-order", row["scene_id"]),
            str(row["scene_id"]),
        )
    )
    development_rows.sort(
        key=lambda row: (
            _stable_seed(seed, "development-order", row["scene_id"]),
            str(row["scene_id"]),
        )
    )
    train_ids = [str(row["scene_id"]) for row in train_rows]
    development_ids = [str(row["scene_id"]) for row in development_rows]
    if (
        len(train_ids) != int(train_count)
        or len(development_ids) != int(development_count)
        or set(train_ids) & set(development_ids)
        or not set(train_ids + development_ids).issubset(official_set)
    ):
        raise RuntimeError("stratified ScanNet train/development allocation drift")
    payload = {
        "schema_version": SUBSET_SCHEMA,
        "dataset": "scannetv2",
        "source_split": "official_train",
        "seed": int(seed),
        "stratification": {
            "method": "joint_rank_quantile_proportional_allocation",
            "fields": list(STRATIFICATION_FIELDS),
            "bins_per_feature": bins,
            "available_candidate_count": len(rows),
            "strata": stratum_rows,
        },
        "train": {
            "count": len(train_ids),
            "scene_ids": train_ids,
            "ordered_sha256": ordered_scene_sha256(train_ids),
        },
        "development": {
            "count": len(development_ids),
            "scene_ids": development_ids,
            "ordered_sha256": ordered_scene_sha256(development_ids),
            "claim_boundary": "development_only_not_final_scannet_validation",
        },
        "records": {
            "train": train_rows,
            "development": development_rows,
        },
        "checks": {
            "fixed_50_train": len(train_ids) == TRAIN_SCENE_COUNT,
            "fixed_10_development": len(development_ids) == DEVELOPMENT_SCENE_COUNT,
            "roles_disjoint": not bool(set(train_ids) & set(development_ids)),
            "all_scenes_from_official_train": set(train_ids + development_ids).issubset(
                official_set
            ),
        },
    }
    payload["selection_sha256"] = _canonical_hash(payload)
    return payload


def _write_scene_split(path: Path, *, label: str, scene_ids: Sequence[str]) -> None:
    text = f"# {label}\n" + "".join(f"{scene_id}\n" for scene_id in scene_ids)
    Path(path).write_text(text, encoding="utf-8")


def materialize_subset(
    output_dir: Path,
    *,
    subset: Mapping[str, Any],
    provenance: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, Any]],
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if subset.get("schema_version") != SUBSET_SCHEMA:
        raise ValueError("invalid subset payload")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.") as name:
        work = Path(name) / "subset"
        work.mkdir()
        train_path = work / "train_50.txt"
        development_path = work / "development_10.txt"
        _write_scene_split(
            train_path,
            label="Fixed 50-scene ScanNet-train development-probe training split",
            scene_ids=subset["train"]["scene_ids"],
        )
        _write_scene_split(
            development_path,
            label="Fixed 10-scene ScanNet-train development-only split",
            scene_ids=subset["development"]["scene_ids"],
        )
        manifest = copy.deepcopy(dict(subset))
        manifest["provenance"] = dict(provenance)
        manifest["excluded_official_train_scenes"] = [dict(row) for row in exclusions]
        manifest["artifacts"] = {
            "train_split": {
                "path": "train_50.txt",
                "sha256": sha256_file(train_path),
                "ordered_sha256": ordered_scene_sha256(
                    subset["train"]["scene_ids"]
                ),
            },
            "development_split": {
                "path": "development_10.txt",
                "sha256": sha256_file(development_path),
                "ordered_sha256": ordered_scene_sha256(
                    subset["development"]["scene_ids"]
                ),
            },
        }
        manifest["checks"]["split_file_order_matches_manifest"] = True
        manifest_path = work / "subset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        work.rename(destination)
    return destination / "subset_manifest.json"


def validate_subset_manifest(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    checks = payload.get("checks")
    if (
        payload.get("schema_version") != SUBSET_SCHEMA
        or payload.get("source_split") != "official_train"
        or not isinstance(checks, Mapping)
        or not all(bool(value) for value in checks.values())
        or payload.get("train", {}).get("count") != TRAIN_SCENE_COUNT
        or payload.get("development", {}).get("count") != DEVELOPMENT_SCENE_COUNT
    ):
        raise ValueError(f"invalid fixed ScanNet train/development subset: {source}")
    train_ids = payload.get("train", {}).get("scene_ids", [])
    development_ids = payload.get("development", {}).get("scene_ids", [])
    if (
        not isinstance(train_ids, list)
        or not isinstance(development_ids, list)
        or len(train_ids) != TRAIN_SCENE_COUNT
        or len(development_ids) != DEVELOPMENT_SCENE_COUNT
        or len(set(train_ids)) != len(train_ids)
        or len(set(development_ids)) != len(development_ids)
        or bool(set(train_ids) & set(development_ids))
    ):
        raise ValueError(f"subset role membership drift: {source}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"subset lacks official-train provenance: {source}")
    official_path = Path(str(provenance.get("official_train_split", ""))).resolve(
        strict=True
    )
    official_ids = read_scene_ids(official_path)
    if (
        sha256_file(official_path) != provenance.get("official_train_split_sha256")
        or ordered_scene_sha256(official_ids)
        != provenance.get("official_train_order_sha256")
        or not set(train_ids + development_ids).issubset(set(official_ids))
    ):
        raise ValueError(f"subset is not provably contained in official train: {source}")
    for role, artifact_key in (
        ("train", "train_split"),
        ("development", "development_split"),
    ):
        record = payload["artifacts"][artifact_key]
        split_path = (source.parent / str(record["path"])).resolve(strict=True)
        if (
            sha256_file(split_path) != record["sha256"]
            or read_scene_ids(split_path) != payload[role]["scene_ids"]
            or ordered_scene_sha256(payload[role]["scene_ids"])
            != record["ordered_sha256"]
        ):
            raise ValueError(f"{role} split artifact drift: {split_path}")
    payload["manifest_path"] = str(source)
    payload["manifest_sha256"] = sha256_file(source)
    return payload


def validate_decoder_template(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != DECODER_TEMPLATE_SCHEMA:
        raise ValueError(f"not a fresh Mask3D decoder template: {source}")
    state = payload.get("decoder_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("decoder template lacks state")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("decoder template state contains a non-tensor")
    observed = tensor_state_sha256(state)
    if observed != payload.get("decoder_state_sha256"):
        raise ValueError("decoder template tensor hash drift")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("decoder template metadata must be a mapping")
    if any(
        metadata.get(key) is True
        for key in (
            "official_mask3d_weights_used",
            "official_mask3d_weights_loaded",
        )
    ):
        raise ValueError("decoder template reports official Mask3D weights")
    mode = metadata.get("feature_pyramid_mode")
    memory_channels = metadata.get("memory_channels")
    if mode is not None and str(mode) not in FEATURE_PYRAMID_MEMORY_CHANNELS:
        raise ValueError(f"decoder template has unsupported FPN mode: {mode!r}")
    if memory_channels is not None:
        if not isinstance(memory_channels, Sequence) or isinstance(
            memory_channels, (str, bytes)
        ):
            raise TypeError("decoder template memory_channels must be a sequence")
        # Keep the public record JSON-stable. Tuple/list equivalence is only
        # normalized at comparisons with the runtime channel contract.
        memory_channels = [int(value) for value in memory_channels]
    return {
        "path": str(source),
        "file_sha256": sha256_file(source),
        "tensor_state_sha256": observed,
        "tensor_count": len(state),
        "downstream_seed": int(payload["downstream_seed"]),
        "schema": DECODER_TEMPLATE_SCHEMA,
        "fresh_mask3d_decoder": True,
        "feature_pyramid_mode": None if mode is None else str(mode),
        "memory_channels": memory_channels,
    }


def _validate_probe_base_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    initialization = config.get("initialization", {})
    train = config.get("train", {})
    scheduler = config.get("scheduler", {})
    evaluation = config.get("eval", {})
    loss = config.get("loss", {})
    requirements = {
        "data.label_source": (data.get("label_source"), "scannet_gt"),
        "data.use_colors": (data.get("use_colors"), True),
        "data.use_normals": (data.get("use_normals"), True),
        "data.append_xyz_to_features": (
            data.get("append_xyz_to_features"),
            False,
        ),
        "data.augmentation_profile": (
            data.get("augmentation_profile"),
            "mask3d_scannet_rgbn6",
        ),
        "data.augmentations": (data.get("augmentations"), True),
        "data.subsampling_mode": (data.get("subsampling_mode"), "sphere_crop"),
        "data.max_points": (data.get("max_points"), PROBE_TRAIN_MAX_POINTS),
        "data.sphere_point_max": (
            data.get("sphere_point_max"),
            PROBE_TRAIN_MAX_POINTS,
        ),
        "model.architecture": (
            model.get("architecture"),
            "litept_mask3d_class_aware_scannet20",
        ),
        "model.input_features": (model.get("input_features"), "rgbn6"),
        "model.input_channels": (model.get("input_channels"), 6),
        "model.litept_variant": (model.get("litept_variant"), "litept_s_star"),
        "model.voxel_reduce": (model.get("voxel_reduce"), "representative"),
        "model.representative_sampling": (
            model.get("representative_sampling"),
            "first",
        ),
        "model.num_classes": (model.get("num_classes"), 19),
        "model.train_queries": (model.get("train_queries"), 100),
        "model.eval_queries": (model.get("eval_queries"), 150),
        "initialization.freeze_backbone": (
            initialization.get("freeze_backbone"),
            False,
        ),
        "initialization.official_mask3d_checkpoint": (
            initialization.get("official_mask3d_checkpoint"),
            None,
        ),
        "train.drop_last": (train.get("drop_last"), False),
        "train.backbone_freeze_epochs": (
            train.get("backbone_freeze_epochs", 0),
            0,
        ),
        "train.ddp_find_unused_parameters": (
            train.get("ddp_find_unused_parameters"),
            True,
        ),
        "train.ddp_padding_policy": (
            train.get("ddp_padding_policy"),
            "zero_and_rescale",
        ),
        "train.batch_scenes_per_step": (train.get("batch_scenes_per_step"), 1),
        "train.gradient_accumulation_steps": (
            train.get("gradient_accumulation_steps"),
            1,
        ),
        "loss.include_auxiliary": (loss.get("include_auxiliary"), True),
        "scheduler.name": (str(scheduler.get("name", "")).lower(), "onecyclelr"),
        "scheduler.interval": (scheduler.get("interval"), "optimizer_step"),
        "eval.scannet_benchmark": (
            evaluation.get("scannet_benchmark"),
            "scannet20",
        ),
        "eval.primary": (evaluation.get("primary"), "raw_queries"),
        "eval.report_dbscan": (evaluation.get("report_dbscan"), False),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in requirements.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(f"base config violates matched RGBN6 probe contract: {mismatches}")
    feature_pyramid_mode = str(model.get("feature_pyramid_mode", "")).lower()
    if feature_pyramid_mode not in FEATURE_PYRAMID_MEMORY_CHANNELS:
        raise ValueError(
            "probe requires a supported explicit model.feature_pyramid_mode; got "
            f"{feature_pyramid_mode!r}"
        )
    if evaluation.get("backbone_batchnorm_policy") not in {"eval", "train_no_update"}:
        raise ValueError("probe requires one explicit supported evaluation BN policy")
    if int(train.get("batch_scenes_per_step", 0)) <= 0:
        raise ValueError("train.batch_scenes_per_step must be positive")
    if int(train.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive")
    for key in ("backbone_lr", "decoder_lr"):
        if not math.isfinite(float(train.get(key, 0.0))) or float(train[key]) <= 0.0:
            raise ValueError(f"train.{key} must be finite and positive")
    if (
        not math.isfinite(float(train.get("weight_decay", -1.0)))
        or float(train["weight_decay"]) < 0.0
    ):
        raise ValueError("train.weight_decay must be finite and non-negative")


def _optimizer_steps_per_epoch(
    *,
    scenes: int,
    world_size: int,
    batch_scenes_per_step: int,
    gradient_accumulation_steps: int,
    drop_last: bool,
) -> int:
    if scenes % world_size:
        local_samples = math.ceil(scenes / world_size)
    else:
        local_samples = scenes // world_size
    local_batches = (
        local_samples // batch_scenes_per_step
        if drop_last
        else math.ceil(local_samples / batch_scenes_per_step)
    )
    if gradient_accumulation_steps > 1 and (
        world_size > 1
        and (
            batch_scenes_per_step != 1
            or local_batches != gradient_accumulation_steps
        )
    ):
        raise ValueError(
            "distributed accumulation must match run_litept_mask3d.py: "
            "batch_scenes_per_step=1 and one complete local loader per update"
        )
    if local_batches <= 0 or local_batches % gradient_accumulation_steps:
        raise ValueError(
            "local loader batches must be positive and divisible by accumulation"
        )
    return local_batches // gradient_accumulation_steps


def _common_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.get("experiment", {}).pop("name", None)
    value.get("experiment", {}).pop("output_root", None)
    init = value.get("initialization", {})
    init.pop("backbone_source", None)
    init.pop("backbone_checkpoint", None)
    return value


def _safe_arm_id(value: str) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", normalized):
        raise ValueError(f"invalid probe arm ID {value!r}")
    return normalized


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(value), sort_keys=False),
        encoding="utf-8",
    )


def materialize_probe_bundle(
    output_dir: Path,
    *,
    base_config_path: Path,
    base_config_provenance_path: Path | None = None,
    subset_manifest_path: Path,
    decoder_template_path: Path,
    candidates: Mapping[str, Path],
    runs_root: Path,
    seed: int,
    world_size: int,
    total_updates: int,
    evaluation_interval_updates: int,
    targets: Mapping[str, float],
    noninferiority_margins: Mapping[str, float],
) -> Path:
    """Write one shared scratch arm plus matched candidate pair manifests."""

    if int(total_updates) != PROBE_UPDATES:
        raise ValueError(f"development probe must run exactly {PROBE_UPDATES} updates")
    if int(world_size) != 2:
        raise ValueError("the matched development probe requires exactly two GPU ranks")
    if set(targets) != set(AP_METRICS) or set(noninferiority_margins) != set(AP_METRICS):
        raise ValueError(f"targets and margins must contain exactly {AP_METRICS}")
    if any(not 0.0 <= float(value) <= 1.0 for value in targets.values()):
        raise ValueError("AP targets must lie in [0,1]")
    if any(float(value) < 0.0 for value in noninferiority_margins.values()):
        raise ValueError("non-inferiority margins must be non-negative")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    base_source = Path(base_config_path).expanduser().resolve(strict=True)
    base_provenance_source = (
        base_source
        if base_config_provenance_path is None
        else Path(base_config_provenance_path).expanduser().resolve(strict=True)
    )
    if sha256_file(base_provenance_source) != sha256_file(base_source):
        raise ValueError(
            "base_config_provenance_path does not contain the exact staged base config"
        )
    base_raw = yaml.safe_load(base_source.read_text(encoding="utf-8"))
    if not isinstance(base_raw, Mapping):
        raise ValueError(f"base config is not a mapping: {base_source}")
    base = _expand_config_environment(base_raw)
    _validate_probe_base_config(base)
    subset = validate_subset_manifest(subset_manifest_path)
    template = validate_decoder_template(decoder_template_path)
    feature_pyramid_mode = str(base["model"]["feature_pyramid_mode"]).lower()
    expected_memory_channels = FEATURE_PYRAMID_MEMORY_CHANNELS[feature_pyramid_mode]
    if template["feature_pyramid_mode"] != feature_pyramid_mode:
        raise ValueError(
            "fresh decoder template feature_pyramid_mode does not match probe "
            f"config: {template['feature_pyramid_mode']!r} != {feature_pyramid_mode!r}"
        )
    if tuple(template["memory_channels"] or ()) != expected_memory_channels:
        raise ValueError(
            "fresh decoder template memory channels do not match probe FPN "
            f"contract: {template['memory_channels']} != {expected_memory_channels}"
        )
    if template["downstream_seed"] != int(seed):
        raise ValueError("decoder template seed differs from the paired downstream seed")
    candidate_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for raw_id, raw_path in sorted(candidates.items()):
        arm_id = _safe_arm_id(raw_id)
        if arm_id == "shared_scratch" or arm_id in candidate_records:
            raise ValueError(f"duplicate/reserved candidate arm ID {arm_id!r}")
        candidate_records[arm_id] = validate_backbone_export(raw_path)
    if not candidate_records:
        raise ValueError("at least one V2 candidate export is required")

    train = base["train"]
    if (
        int(train.get("gradient_accumulation_steps", 1)) > 1
        and train.get("normalization_target_instances") is None
    ):
        raise ValueError(
            "gradient accumulation requires normalization_target_instances "
            "under the existing Mask3D runner contract"
        )
    steps_per_epoch = _optimizer_steps_per_epoch(
        scenes=TRAIN_SCENE_COUNT,
        world_size=int(world_size),
        batch_scenes_per_step=int(train["batch_scenes_per_step"]),
        gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 1)),
        drop_last=bool(train.get("drop_last", False)),
    )
    if int(total_updates) % steps_per_epoch:
        raise ValueError("1,000 updates are not epoch-aligned for this loader contract")
    if (
        int(evaluation_interval_updates) <= 0
        or int(total_updates) % int(evaluation_interval_updates)
        or int(evaluation_interval_updates) % steps_per_epoch
    ):
        raise ValueError("evaluation interval must divide the run and align to epochs")
    max_epochs = int(total_updates) // steps_per_epoch
    eval_every_epochs = int(evaluation_interval_updates) // steps_per_epoch
    expected_updates = list(
        range(0, int(total_updates) + 1, int(evaluation_interval_updates))
    )
    if expected_updates[-1] != int(total_updates):
        raise RuntimeError("evaluation schedule does not include the final update")

    train_split = (
        Path(subset["manifest_path"]).parent
        / subset["artifacts"]["train_split"]["path"]
    ).resolve(strict=True)
    development_split = (
        Path(subset["manifest_path"]).parent
        / subset["artifacts"]["development_split"]["path"]
    ).resolve(strict=True)
    common = copy.deepcopy(dict(base))
    common["experiment"]["seed"] = int(seed)
    common["data"]["train_split"] = str(train_split)
    common["data"]["val_split"] = str(development_split)
    common["initialization"]["decoder_template"] = template["path"]
    common["initialization"]["freeze_backbone"] = False
    common["initialization"]["official_mask3d_checkpoint"] = None
    common["train"]["max_epochs"] = max_epochs
    common["train"]["eval_every_epochs"] = eval_every_epochs
    common["train"]["save_every_epochs"] = eval_every_epochs
    common["train"]["report_loss_components"] = True
    common["eval"]["report_train_metrics"] = False

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}.") as name:
        work = Path(name) / "probe"
        # A bundle must remain valid after a node-local materialization job
        # exits.  Keep the exact base YAML inside the immutable bundle instead
        # of recording a transient staging path as the verification target.
        base_copy_path = work / "base_config.yaml"
        base_copy_path.parent.mkdir(parents=True, exist_ok=True)
        base_copy_path.write_bytes(base_source.read_bytes())
        configs_dir = work / "configs"
        arms_dir = work / "arms"
        pairs_dir = work / "pairs"
        for directory in (configs_dir, arms_dir, pairs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        arm_inputs: OrderedDict[str, Path | None] = OrderedDict(
            [("shared_scratch", None)]
        )
        arm_inputs.update(
            (arm_id, Path(record["path"]))
            for arm_id, record in candidate_records.items()
        )
        arm_records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        common_hash: str | None = None
        for arm_id, checkpoint in arm_inputs.items():
            config = copy.deepcopy(common)
            config["experiment"]["name"] = arm_id
            config["experiment"]["output_root"] = str(Path(runs_root).resolve())
            if checkpoint is None:
                config["initialization"]["backbone_source"] = "scratch"
                config["initialization"]["backbone_checkpoint"] = None
                backbone = None
            else:
                config["initialization"]["backbone_source"] = "structured3d"
                config["initialization"]["backbone_checkpoint"] = str(
                    checkpoint.resolve(strict=True)
                )
                backbone = candidate_records[arm_id]
            observed_common_hash = _canonical_hash(_common_config(config))
            if common_hash is None:
                common_hash = observed_common_hash
            elif observed_common_hash != common_hash:
                raise RuntimeError("paired configs differ outside backbone initialization")
            config_path = configs_dir / f"{arm_id}.yaml"
            _write_yaml(config_path, config)
            config_hash = sha256_file(config_path)
            arm_manifest = {
                "schema_version": PROBE_ARM_SCHEMA,
                "arm_id": arm_id,
                "role": "shared_scratch" if checkpoint is None else "v2_candidate",
                "config": {
                    "path": f"configs/{config_path.name}",
                    "sha256": config_hash,
                },
                "common_config_sha256": common_hash,
                "backbone_export": backbone,
                "decoder_template": template,
                "subset_manifest": {
                    "path": subset["manifest_path"],
                    "sha256": subset["manifest_sha256"],
                },
                "execution_contract": {
                    "world_size": int(world_size),
                    "global_optimizer_updates": int(total_updates),
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "max_epochs": max_epochs,
                    "evaluation_interval_updates": int(evaluation_interval_updates),
                    "expected_evaluation_updates": expected_updates,
                    "zero_update_evaluation": "required_before_first_optimizer_step",
                },
                "matched_contract": {
                    "rgbn6": True,
                    "seed": int(seed),
                    "world_size": int(world_size),
                    "batchnorm_policy": common["eval"][
                        "backbone_batchnorm_policy"
                    ],
                    "ordered_train_split_sha256": subset["train"][
                        "ordered_sha256"
                    ],
                    "ordered_development_split_sha256": subset["development"][
                        "ordered_sha256"
                    ],
                    "augmentation_profile": common["data"][
                        "augmentation_profile"
                    ],
                    "training_capacity": _probe_train_capacity_contract(),
                    "feature_pyramid_mode": feature_pyramid_mode,
                    "decoder_memory_channels": list(expected_memory_channels),
                    "train_sampler_order": (
                        "ExactDistributedSceneSampler(seed+epoch); identical manifest, "
                        "scene order, seed, and world size"
                    ),
                    "optimizer_and_scheduler_in_common_config_hash": True,
                    "decoder_tensor_state_sha256": template[
                        "tensor_state_sha256"
                    ],
                    "decoder_byte_identical_across_arms": True,
                },
                "result_contract": PROBE_RESULT_SCHEMA,
            }
            arm_manifest_path = arms_dir / f"{arm_id}.json"
            arm_manifest_path.write_text(
                json.dumps(arm_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            arm_records[arm_id] = {
                **arm_manifest,
                "manifest": f"arms/{arm_manifest_path.name}",
                "manifest_sha256": sha256_file(arm_manifest_path),
            }

        pair_records: list[dict[str, Any]] = []
        scratch = arm_records["shared_scratch"]
        for arm_id in candidate_records:
            candidate = arm_records[arm_id]
            if (
                candidate["common_config_sha256"] != scratch["common_config_sha256"]
                or candidate["decoder_template"]["file_sha256"]
                != scratch["decoder_template"]["file_sha256"]
                or candidate["decoder_template"]["tensor_state_sha256"]
                != scratch["decoder_template"]["tensor_state_sha256"]
            ):
                raise RuntimeError("candidate/scratch pairing contract drift")
            pair_id = f"{arm_id}_vs_shared_scratch"
            pair = {
                "schema_version": PROBE_PAIR_SCHEMA,
                "pair_id": pair_id,
                "candidate_arm_id": arm_id,
                "scratch_arm_id": "shared_scratch",
                "candidate_arm_manifest_sha256": candidate["manifest_sha256"],
                "scratch_arm_manifest_sha256": scratch["manifest_sha256"],
                "common_config_sha256": common_hash,
                "decoder_file_sha256": template["file_sha256"],
                "decoder_tensor_state_sha256": template["tensor_state_sha256"],
                "targets": {key: float(targets[key]) for key in AP_METRICS},
                "noninferiority_margins": {
                    key: float(noninferiority_margins[key]) for key in AP_METRICS
                },
                "matched": True,
            }
            pair_path = pairs_dir / f"{pair_id}.json"
            pair_path.write_text(
                json.dumps(pair, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            pair_records.append(
                {
                    **pair,
                    "manifest": f"pairs/{pair_path.name}",
                    "manifest_sha256": sha256_file(pair_path),
                }
            )
        bundle = {
            "schema_version": PROBE_BUNDLE_SCHEMA,
            "base_config": {
                "path": base_copy_path.name,
                "sha256": sha256_file(base_copy_path),
                "source_path": str(base_provenance_source),
                "source_sha256": sha256_file(base_provenance_source),
                "resolved_sha256": _canonical_hash(base),
            },
            "subset_manifest": {
                "path": subset["manifest_path"],
                "sha256": subset["manifest_sha256"],
            },
            "shared_scratch_arm_id": "shared_scratch",
            "candidate_arm_ids": list(candidate_records),
            "arms": arm_records,
            "pairs": pair_records,
            "common_config_sha256": common_hash,
            "execution_contract": scratch["execution_contract"],
            "metrics_contract": {
                "schema_version": PROBE_RESULT_SCHEMA,
                "strict_initialization_audit_required": True,
                "aggregate_metrics": list(AP_METRICS),
                "query_diagnostics": [
                    "valid_proposals",
                    "foreground_queries",
                    "no_object_queries",
                    "total_queries",
                ],
                "queries_per_scene": int(common["model"]["eval_queries"]),
                "per_scene_required": True,
                "early_auc_horizon_updates": int(total_updates),
            },
            "checks": {
                "one_shared_scratch": True,
                "fixed_50_train_10_development": True,
                "all_arms_common_config_hash_identical": True,
                "all_arms_decoder_file_hash_identical": True,
                "all_arms_decoder_tensor_hash_identical": True,
                "exact_1000_updates": int(total_updates) == PROBE_UPDATES,
                "zero_update_evaluation_required": True,
                "official_scannet_validation_unused": True,
            },
            "runner_integration": {
                "command": "torchrun --nproc_per_node=<world_size> student/scripts/run_litept_mask3d.py --config <arm-config>",
                "required_additions": [
                    "enforce manifest world_size and exact optimizer update count",
                    "evaluate initialized model at update zero",
                    "emit valid/foreground/no-object query counts per development scene",
                    "write one result JSON conforming to metrics_contract",
                ],
            },
        }
        bundle_path = work / "probe_bundle_manifest.json"
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        work.rename(destination)
    return destination / "probe_bundle_manifest.json"


def validate_probe_bundle(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    bundle = json.loads(source.read_text(encoding="utf-8"))
    checks = bundle.get("checks")
    if (
        bundle.get("schema_version") != PROBE_BUNDLE_SCHEMA
        or bundle.get("shared_scratch_arm_id") != "shared_scratch"
        or not isinstance(checks, Mapping)
        or not all(bool(value) for value in checks.values())
        or bundle.get("execution_contract", {}).get("global_optimizer_updates")
        != PROBE_UPDATES
        or bundle.get("execution_contract", {}).get("world_size") != 2
    ):
        raise ValueError(f"invalid matched development-probe bundle: {source}")
    base_reference = Path(bundle["base_config"]["path"])
    base_path = (
        base_reference
        if base_reference.is_absolute()
        else source.parent / base_reference
    ).resolve(strict=True)
    subset = validate_subset_manifest(Path(bundle["subset_manifest"]["path"]))
    if (
        sha256_file(base_path) != bundle["base_config"]["sha256"]
        or subset["manifest_sha256"] != bundle["subset_manifest"]["sha256"]
    ):
        raise ValueError("probe base config or subset provenance drift")
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    _validate_probe_base_config(base_config)
    arm_ids = ["shared_scratch", *bundle.get("candidate_arm_ids", [])]
    if set(bundle.get("arms", {})) != set(arm_ids):
        raise ValueError("probe bundle arm set drift")
    for arm_id in arm_ids:
        arm = bundle["arms"][arm_id]
        config_path = (source.parent / arm["config"]["path"]).resolve(strict=True)
        manifest_path = (source.parent / arm["manifest"]).resolve(strict=True)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        disk_arm = json.loads(manifest_path.read_text(encoding="utf-8"))
        embedded_arm = {
            key: value
            for key, value in arm.items()
            if key not in {"manifest", "manifest_sha256"}
        }
        if (
            sha256_file(config_path) != arm["config"]["sha256"]
            or sha256_file(manifest_path) != arm["manifest_sha256"]
            or arm["common_config_sha256"] != bundle["common_config_sha256"]
            or disk_arm != embedded_arm
            or _canonical_hash(_common_config(config))
            != bundle["common_config_sha256"]
        ):
            raise ValueError(f"probe arm artifact drift: {arm_id}")
        _validate_probe_base_config(config)
        template = validate_decoder_template(Path(arm["decoder_template"]["path"]))
        if template != arm["decoder_template"]:
            raise ValueError(f"probe decoder template drift: {arm_id}")
        feature_pyramid_mode = str(config["model"].get("feature_pyramid_mode", "")).lower()
        expected_memory_channels = FEATURE_PYRAMID_MEMORY_CHANNELS.get(
            feature_pyramid_mode
        )
        if (
            expected_memory_channels is None
            or template["feature_pyramid_mode"] != feature_pyramid_mode
            or tuple(template["memory_channels"] or ()) != expected_memory_channels
            or arm["matched_contract"].get("feature_pyramid_mode")
            != feature_pyramid_mode
            or arm["matched_contract"].get("decoder_memory_channels")
            != list(expected_memory_channels)
            or arm["matched_contract"].get("training_capacity")
            != _probe_train_capacity_contract()
        ):
            raise ValueError(f"probe FPN/decoder contract drift: {arm_id}")
        backbone = arm.get("backbone_export")
        if arm_id == "shared_scratch":
            if backbone is not None:
                raise ValueError("shared scratch arm unexpectedly has a backbone export")
        elif validate_backbone_export(Path(backbone["path"])) != backbone:
            raise ValueError(f"probe candidate backbone drift: {arm_id}")
    for pair in bundle.get("pairs", []):
        pair_path = (source.parent / pair["manifest"]).resolve(strict=True)
        disk_pair = json.loads(pair_path.read_text(encoding="utf-8"))
        embedded_pair = {
            key: value
            for key, value in pair.items()
            if key not in {"manifest", "manifest_sha256"}
        }
        if (
            pair.get("schema_version") != PROBE_PAIR_SCHEMA
            or sha256_file(pair_path) != pair.get("manifest_sha256")
            or disk_pair != embedded_pair
            or pair.get("scratch_arm_id") != "shared_scratch"
            or pair.get("candidate_arm_id") not in bundle["candidate_arm_ids"]
            or pair.get("matched") is not True
            or pair.get("candidate_arm_manifest_sha256")
            != bundle["arms"][pair.get("candidate_arm_id")]["manifest_sha256"]
            or pair.get("scratch_arm_manifest_sha256")
            != bundle["arms"]["shared_scratch"]["manifest_sha256"]
            or pair.get("common_config_sha256")
            != bundle["common_config_sha256"]
            or pair.get("decoder_file_sha256")
            != bundle["arms"]["shared_scratch"]["decoder_template"][
                "file_sha256"
            ]
            or pair.get("decoder_tensor_state_sha256")
            != bundle["arms"]["shared_scratch"]["decoder_template"][
                "tensor_state_sha256"
            ]
        ):
            raise ValueError(f"probe pair artifact drift: {pair.get('pair_id')}")
        if set(pair.get("targets", {})) != set(AP_METRICS) or set(
            pair.get("noninferiority_margins", {})
        ) != set(AP_METRICS):
            raise ValueError(f"probe pair metric contract drift: {pair['pair_id']}")
        if any(
            not math.isfinite(float(pair["targets"][key]))
            or not 0.0 <= float(pair["targets"][key]) <= 1.0
            or not math.isfinite(float(pair["noninferiority_margins"][key]))
            or float(pair["noninferiority_margins"][key]) < 0.0
            for key in AP_METRICS
        ):
            raise ValueError(f"probe pair target/margin invalid: {pair['pair_id']}")
    if (
        len(bundle.get("pairs", [])) != len(bundle["candidate_arm_ids"])
        or {pair["candidate_arm_id"] for pair in bundle["pairs"]}
        != set(bundle["candidate_arm_ids"])
        or any(
            pair["targets"] != bundle["pairs"][0]["targets"]
            for pair in bundle["pairs"][1:]
        )
    ):
        raise ValueError("probe pair set does not cover each candidate exactly once")
    bundle["manifest_path"] = str(source)
    bundle["manifest_sha256"] = sha256_file(source)
    return bundle


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be finite in [0,1]")
    return parsed


def _validate_query_diagnostics(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} query_diagnostics missing")
    keys = (
        "valid_proposals",
        "foreground_queries",
        "no_object_queries",
        "total_queries",
    )
    if any(
        isinstance(value.get(key), bool) or not isinstance(value.get(key), int)
        for key in keys
    ):
        raise ValueError(f"{label} query diagnostics must be JSON integers")
    output = {key: int(value[key]) for key in keys}
    if (
        min(output.values()) < 0
        or output["total_queries"] <= 0
        or output["valid_proposals"] > output["total_queries"]
        or output["foreground_queries"] + output["no_object_queries"]
        != output["total_queries"]
    ):
        raise ValueError(f"{label} query counts are inconsistent: {output}")
    return output


def validate_probe_result(
    result_path: Path,
    *,
    arm: Mapping[str, Any],
    bundle: Mapping[str, Any],
    development_scene_ids: Sequence[str],
) -> dict[str, Any]:
    source = Path(result_path).expanduser().resolve(strict=True)
    result = json.loads(source.read_text(encoding="utf-8"))
    initialization = result.get("initialization")
    expected_source = "scratch" if arm["role"] == "shared_scratch" else "structured3d"
    expected_backbone_hash = (
        None
        if arm["role"] == "shared_scratch"
        else arm["backbone_export"]["tensor_state_sha256"]
    )
    valid_scratch_hash = (
        isinstance(initialization, Mapping)
        and isinstance(initialization.get("backbone_state_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", initialization["backbone_state_sha256"]))
    )
    initialization_ok = (
        isinstance(initialization, Mapping)
        and initialization.get("backbone_source") == expected_source
        and initialization.get("decoder_state_sha256")
        == arm["decoder_template"]["tensor_state_sha256"]
        and initialization.get("official_mask3d_weights_loaded") is False
        and (
            valid_scratch_hash
            if expected_backbone_hash is None
            else initialization.get("backbone_state_sha256") == expected_backbone_hash
        )
    )
    if (
        result.get("schema_version") != PROBE_RESULT_SCHEMA
        or result.get("arm_id") != arm["arm_id"]
        or result.get("arm_manifest_sha256") != arm["manifest_sha256"]
        or result.get("common_config_sha256") != bundle["common_config_sha256"]
        or result.get("decoder_tensor_state_sha256")
        != arm["decoder_template"]["tensor_state_sha256"]
        or result.get("global_optimizer_updates") != PROBE_UPDATES
        or result.get("world_size") != bundle["execution_contract"]["world_size"]
        or result.get("arm_config_sha256") != arm["config"]["sha256"]
        or result.get("decoder_file_sha256")
        != arm["decoder_template"]["file_sha256"]
        or result.get("ordered_train_split_sha256")
        != arm["matched_contract"]["ordered_train_split_sha256"]
        or result.get("ordered_development_split_sha256")
        != arm["matched_contract"]["ordered_development_split_sha256"]
        or result.get("backbone_batchnorm_policy")
        != arm["matched_contract"]["batchnorm_policy"]
        or result.get("augmentation_profile")
        != arm["matched_contract"]["augmentation_profile"]
        or not initialization_ok
    ):
        raise ValueError(f"probe result identity/contract drift: {source}")
    evaluations = result.get("evaluations")
    expected_updates = bundle["execution_contract"]["expected_evaluation_updates"]
    if (
        not isinstance(evaluations, list)
        or any(
            not isinstance(row, Mapping)
            or isinstance(row.get("update"), bool)
            or not isinstance(row.get("update"), int)
            for row in evaluations
        )
        or [row["update"] for row in evaluations] != expected_updates
    ):
        raise ValueError(f"probe result evaluation schedule drift: {source}")
    expected_scenes = list(development_scene_ids)
    normalized = []
    for evaluation in evaluations:
        update = int(evaluation["update"])
        metrics = {
            key: _finite_probability(
                evaluation.get("metrics", {}).get(key),
                f"update {update} {key}",
            )
            for key in AP_METRICS
        }
        query = _validate_query_diagnostics(
            evaluation.get("query_diagnostics"),
            f"update {update}",
        )
        per_scene = evaluation.get("per_scene")
        if not isinstance(per_scene, Mapping) or set(per_scene) != set(expected_scenes):
            raise ValueError(f"update {update}: development per-scene set drift")
        normalized_scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        aggregate_query = {
            "valid_proposals": 0,
            "foreground_queries": 0,
            "no_object_queries": 0,
            "total_queries": 0,
        }
        for scene_id in expected_scenes:
            scene = per_scene[scene_id]
            scene_metrics = {
                key: _finite_probability(
                    scene.get("metrics", {}).get(key),
                    f"update {update} {scene_id} {key}",
                )
                for key in AP_METRICS
            }
            scene_query = _validate_query_diagnostics(
                scene.get("query_diagnostics"),
                f"update {update} {scene_id}",
            )
            if scene_query["total_queries"] != int(
                bundle["metrics_contract"]["queries_per_scene"]
            ):
                raise ValueError(
                    f"update {update} {scene_id}: total query count drift"
                )
            for key in aggregate_query:
                aggregate_query[key] += scene_query[key]
            normalized_scenes[scene_id] = {
                "metrics": scene_metrics,
                "query_diagnostics": scene_query,
            }
        if query != aggregate_query:
            raise ValueError(
                f"update {update}: aggregate query counts do not equal per-scene sums"
            )
        normalized.append(
            {
                "update": update,
                "metrics": metrics,
                "query_diagnostics": query,
                "per_scene": normalized_scenes,
            }
        )
    return {
        **result,
        "evaluations": normalized,
        "result_path": str(source),
        "result_sha256": sha256_file(source),
    }


def _normalized_auc(updates: Sequence[int], values: Sequence[float]) -> float:
    if len(updates) != len(values) or len(values) < 2:
        raise ValueError("AUC requires aligned trajectories with at least two points")
    span = float(updates[-1] - updates[0])
    if span <= 0:
        raise ValueError("AUC update span must be positive")
    area = sum(
        (float(right_x) - float(left_x))
        * (float(left_y) + float(right_y))
        * 0.5
        for left_x, right_x, left_y, right_y in zip(
            updates[:-1],
            updates[1:],
            values[:-1],
            values[1:],
            strict=True,
        )
    )
    return float(area / span)


def _query_metric(query: Mapping[str, int], key: str, scene_count: int) -> float:
    if key == "valid_proposals_per_scene":
        return float(query["valid_proposals"] / scene_count)
    if key == "foreground_query_fraction":
        return float(query["foreground_queries"] / query["total_queries"])
    if key == "no_object_query_fraction":
        return float(query["no_object_queries"] / query["total_queries"])
    raise KeyError(key)


def _arm_summary(
    result: Mapping[str, Any],
    *,
    scene_count: int,
    targets: Mapping[str, float],
) -> dict[str, Any]:
    evaluations = result["evaluations"]
    updates = [int(row["update"]) for row in evaluations]
    trajectories: dict[str, list[float]] = {
        key: [float(row["metrics"][key]) for row in evaluations]
        for key in AP_METRICS
    }
    trajectories.update(
        {
            key: [
                _query_metric(row["query_diagnostics"], key, scene_count)
                for row in evaluations
            ]
            for key in QUERY_METRICS
        }
    )
    steps_to_target = {
        key: next(
            (
                update
                for update, value in zip(updates, trajectories[key], strict=True)
                if value >= float(targets[key])
            ),
            None,
        )
        for key in AP_METRICS
    }
    return {
        "arm_id": result["arm_id"],
        "result_path": result["result_path"],
        "result_sha256": result["result_sha256"],
        "updates": updates,
        "trajectories": trajectories,
        "final": {key: values[-1] for key, values in trajectories.items()},
        "early_auc_0_1000": {
            key: _normalized_auc(updates, values)
            for key, values in trajectories.items()
        },
        "steps_to_target": steps_to_target,
    }


def _paired_delta_summary(
    values: Sequence[float],
    *,
    margin: float,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("paired deltas must be a non-empty finite vector")
    rng = np.random.default_rng(int(seed))
    sampled = array[
        rng.integers(0, array.size, size=(int(bootstrap_samples), array.size))
    ].mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975]).tolist()
    return {
        "scene_count": int(array.size),
        "mean_delta": float(array.mean()),
        "median_delta": float(np.median(array)),
        "minimum_delta": float(array.min()),
        "maximum_delta": float(array.max()),
        "candidate_win_fraction": float(np.mean(array > 0.0)),
        "tie_fraction": float(np.mean(array == 0.0)),
        "paired_bootstrap_mean_ci95": [float(lower), float(upper)],
        "noninferiority_margin": float(margin),
        "noninferior_by_paired_ci": bool(float(lower) >= -float(margin)),
    }


def aggregate_probe_results(
    *,
    bundle_manifest_path: Path,
    result_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Aggregate paired candidate-versus-shared-scratch development results."""

    bundle = validate_probe_bundle(bundle_manifest_path)
    subset = validate_subset_manifest(Path(bundle["subset_manifest"]["path"]))
    dev_ids = subset["development"]["scene_ids"]
    expected_arms = {"shared_scratch", *bundle["candidate_arm_ids"]}
    if set(result_paths) != expected_arms:
        raise ValueError(
            f"result arm set must be exact: missing={sorted(expected_arms - set(result_paths))} "
            f"extra={sorted(set(result_paths) - expected_arms)}"
        )
    results = {
        arm_id: validate_probe_result(
            path,
            arm=bundle["arms"][arm_id],
            bundle=bundle,
            development_scene_ids=dev_ids,
        )
        for arm_id, path in result_paths.items()
    }
    targets = bundle["pairs"][0]["targets"]
    summaries = {
        arm_id: _arm_summary(
            result,
            scene_count=len(dev_ids),
            targets=targets,
        )
        for arm_id, result in results.items()
    }
    scratch_result = results["shared_scratch"]
    scratch_summary = summaries["shared_scratch"]
    pair_summaries = []
    for pair in bundle["pairs"]:
        candidate_id = pair["candidate_arm_id"]
        candidate_result = results[candidate_id]
        candidate_summary = summaries[candidate_id]
        final_candidate = candidate_result["evaluations"][-1]
        final_scratch = scratch_result["evaluations"][-1]
        paired_scenes: dict[str, Any] = {}
        noninferiority: dict[str, Any] = {}
        for metric in AP_METRICS:
            deltas = [
                float(final_candidate["per_scene"][scene_id]["metrics"][metric])
                - float(final_scratch["per_scene"][scene_id]["metrics"][metric])
                for scene_id in dev_ids
            ]
            margin = float(pair["noninferiority_margins"][metric])
            paired = _paired_delta_summary(
                deltas,
                margin=margin,
                seed=_stable_seed(pair["pair_id"], metric, "paired-bootstrap"),
            )
            aggregate_delta = (
                candidate_summary["final"][metric]
                - scratch_summary["final"][metric]
            )
            auc_delta = (
                candidate_summary["early_auc_0_1000"][metric]
                - scratch_summary["early_auc_0_1000"][metric]
            )
            paired["per_scene_deltas"] = dict(zip(dev_ids, deltas, strict=True))
            paired_scenes[metric] = paired
            noninferiority[metric] = {
                "margin": margin,
                "final_aggregate_delta": float(aggregate_delta),
                "early_auc_delta": float(auc_delta),
                "aggregate_final_noninferior": bool(aggregate_delta >= -margin),
                "paired_ci_noninferior": paired["noninferior_by_paired_ci"],
                "passed": bool(
                    aggregate_delta >= -margin
                    and paired["noninferior_by_paired_ci"]
                ),
            }
        pair_summaries.append(
            {
                "pair_id": pair["pair_id"],
                "candidate_arm_id": candidate_id,
                "scratch_arm_id": "shared_scratch",
                "final_deltas": {
                    key: candidate_summary["final"][key]
                    - scratch_summary["final"][key]
                    for key in (*AP_METRICS, *QUERY_METRICS)
                },
                "early_auc_deltas": {
                    key: candidate_summary["early_auc_0_1000"][key]
                    - scratch_summary["early_auc_0_1000"][key]
                    for key in (*AP_METRICS, *QUERY_METRICS)
                },
                "steps_to_target": {
                    key: {
                        "target": float(pair["targets"][key]),
                        "candidate": candidate_summary["steps_to_target"][key],
                        "scratch": scratch_summary["steps_to_target"][key],
                    }
                    for key in AP_METRICS
                },
                "paired_per_scene": paired_scenes,
                "noninferiority": noninferiority,
                "passed_all_ap_metrics": all(
                    row["passed"] for row in noninferiority.values()
                ),
            }
        )
    return {
        "schema_version": PROBE_AGGREGATE_SCHEMA,
        "bundle_manifest": bundle["manifest_path"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "development_subset": {
            "scene_count": len(dev_ids),
            "scene_ids": dev_ids,
            "ordered_sha256": subset["development"]["ordered_sha256"],
            "source_split": "official_train",
            "claim_boundary": "development_noninferiority_not_final_test_evidence",
        },
        "shared_scratch_arm_id": "shared_scratch",
        "arms": summaries,
        "pairs": pair_summaries,
        "all_candidates_passed": all(
            row["passed_all_ap_metrics"] for row in pair_summaries
        ),
    }


def write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "AP_METRICS",
    "DEVELOPMENT_SCENE_COUNT",
    "PROBE_AGGREGATE_SCHEMA",
    "PROBE_ARM_SCHEMA",
    "PROBE_BUNDLE_SCHEMA",
    "PROBE_PAIR_SCHEMA",
    "PROBE_RESULT_SCHEMA",
    "PROBE_UPDATES",
    "QUERY_METRICS",
    "STRATIFICATION_FIELDS",
    "SUBSET_SCHEMA",
    "TRAIN_SCENE_COUNT",
    "aggregate_probe_results",
    "collect_scannet_train_statistics",
    "materialize_probe_bundle",
    "materialize_subset",
    "ordered_scene_sha256",
    "read_scene_ids",
    "stratified_train_development_subset",
    "validate_decoder_template",
    "validate_probe_bundle",
    "validate_probe_result",
    "validate_subset_manifest",
    "write_json_no_clobber",
]
