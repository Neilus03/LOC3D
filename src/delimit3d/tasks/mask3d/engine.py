#!/usr/bin/env python3
"""Dedicated ScanNet20-things training path for faithful LitePT -> Mask3D."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


from delimit3d.tasks.mask3d.config import load_config, parse_granularities, set_seed
from delimit3d.tasks.mask3d.legacy_probe import (
    PROBE_RESULT_SCHEMA,
    validate_probe_bundle,
    validate_probe_result,
    validate_subset_manifest,
)
from delimit3d.tasks.mask3d.data.batched_scene_batch import collate_multi_scene_samples
from delimit3d.tasks.mask3d.data.distributed_scene_sampler import ExactDistributedSceneSampler
from delimit3d.tasks.mask3d.data.multi_scene_dataset import MultiSceneDataset, build_scene_list
from delimit3d.tasks.mask3d.data.segment_targets import build_segment_instance_targets
from delimit3d.tasks.mask3d.scheduler import (
    build_onecycle_scheduler,
    build_train_acceptance_report,
)
from delimit3d.tasks.mask3d.losses.mask3d_segment_loss import Mask3DSegmentSetCriterion
from delimit3d.tasks.mask3d.metrics.litept_mask3d_ca_ap import (
    dbscan_split_query_proposals,
    dbscan_split_query_proposals_with_classes,
    raw_query_proposals,
    raw_query_proposals_with_classes,
)
from delimit3d.metrics.official_instance_ap import (
    SCANNET_IGNORE_MODE,
    SCANNET_MIN_REGION_SIZE,
    build_instance_ap_records,
    evaluate_official_instance_ap,
    merge_ap_record_sets,
)
from delimit3d.tasks.mask3d.model import LitePTMask3DModel
from delimit3d.tasks.mask3d.initialization import initialize_litept_mask3d_arm
from delimit3d.models.litept_v2_transfer import sha256_file
from delimit3d.runtime.records import config_sha256
from delimit3d.evaluation.backbone_batchnorm import current_batch_without_updates
from contextlib import nullcontext


log = logging.getLogger("run_litept_mask3d")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_NCCL_TIMEOUT = timedelta(minutes=30)
_EVAL_BARRIER_TIMEOUT = timedelta(hours=24)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--eval-queries",
        type=int,
        choices=(100, 150),
        help="Diagnostic override for the number of non-parametric eval queries.",
    )
    parser.add_argument(
        "--skip-dbscan",
        action="store_true",
        help="Skip secondary DBSCAN metrics during repeated validation.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Strict full-model checkpoint for --eval-only.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model and optimizer state from a training checkpoint.",
    )
    parser.add_argument(
        "--v2-probe-bundle",
        type=Path,
        help=(
            "Validated matched V2 development-probe bundle. Requires "
            "--v2-probe-arm-manifest and --v2-probe-result."
        ),
    )
    parser.add_argument(
        "--v2-probe-arm-manifest",
        type=Path,
        help="One arm manifest inside --v2-probe-bundle.",
    )
    parser.add_argument(
        "--v2-probe-result",
        type=Path,
        help="No-clobber V2 development-probe result JSON written by rank zero.",
    )
    parser.add_argument(
        "--v2-runtime-litept-root",
        type=Path,
        help=(
            "Optional node-local LitePT root for a V2 development probe. Must be "
            "paired with the two V2 data-root overrides."
        ),
    )
    parser.add_argument(
        "--v2-runtime-scans-root",
        type=Path,
        help=(
            "Optional node-local ScanNet scans root for a V2 development probe. "
            "Must be paired with the other V2 runtime overrides."
        ),
    )
    parser.add_argument(
        "--v2-runtime-scannet-gt-cache-root",
        type=Path,
        help=(
            "Optional node-local ScanNet GT-cache root for a V2 development probe. "
            "Must be paired with the other V2 runtime overrides."
        ),
    )
    parser.add_argument(
        "--v2-runtime-decoder-template",
        type=Path,
        help=(
            "Optional byte-verified node-local copy of the frozen V2 decoder "
            "template."
        ),
    )
    parser.add_argument(
        "--v2-runtime-backbone-checkpoint",
        type=Path,
        help=(
            "Optional byte-verified node-local copy of a V2 candidate backbone "
            "export; forbidden for the shared-scratch arm."
        ),
    )
    return parser.parse_args(argv)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _split_scene_ids(path: str | Path) -> set[str]:
    """Read scene IDs without requiring the scan root to be mounted yet."""
    split_path = _path(path).resolve(strict=True)
    return {
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _validate_config(config):
    """Validate the real task interface for every supported backbone."""
    from delimit3d.models.backbones import backbone_name, ptv3_recipe
    name = backbone_name(config)
    data, model = config["data"], config["model"]
    if data.get("label_source") != "scannet_gt":
        raise ValueError("Mask3D requires ordered ScanNet GT instance targets")
    if config.get("eval", {}).get("scannet_benchmark") != "scannet20":
        raise ValueError("Mask3D uses the ScanNet20 instance universe")
    if data.get("append_xyz_to_features", False):
        raise ValueError("XYZ is a separate input; RGBN6 contains six feature channels")
    features = model.get("input_features", "rgbn6")
    channels = {"rgb3": 3, "rgbn6": 6}.get(features)
    if channels is None or int(model.get("input_channels", channels)) != channels:
        raise ValueError("Mask3D input feature width does not match its channels")
    if not data.get("use_colors") or bool(data.get("use_normals")) != (features == "rgbn6"):
        raise ValueError("Data features differ from the model's RGB/normal contract")
    if name != "litept_s_star" and features != "rgbn6":
        raise ValueError("PTv3 task bridges require unchanged RGBN6 plus separate XYZ")
    if name == "sonata" and float(model.get("voxel_size", 0)) != 0.05:
        raise ValueError("The Sonata native reference uses 5 cm voxels")
    if float(model.get("voxel_size", 0)) <= 0:
        raise ValueError("voxel_size must be positive")
    if model.get("voxel_reduce", "representative") != "representative":
        raise ValueError("The task bridge requires representative point voxelization")
    if model.get("feature_pyramid_mode", "mask3d_fpn") not in {
        "mask3d_fpn", "mask3d_dec0_pooled_fpn", "mask3d_decoder_only_fpn"
    }:
        raise ValueError("Unknown Mask3D memory readout")
    class_aware = model.get("architecture") == "litept_mask3d_class_aware_scannet20"
    if int(model.get("num_classes", 2)) != (19 if class_aware else 2):
        raise ValueError("Mask3D class width must include its no-object class")
    if int(model.get("train_queries", 100)) != 100:
        raise ValueError("The Mask3D reference uses 100 training queries")
    if int(model.get("eval_queries", 100)) not in (100, 150):
        raise ValueError("Use 100 reference queries or the explicit 150-query diagnostic")
    init = config["initialization"]
    if not init.get("decoder_template") or init.get("official_mask3d_checkpoint"):
        raise ValueError("A fresh shared decoder template is required")
    train = config["train"]
    for key in ("max_epochs", "batch_scenes_per_step", "gradient_accumulation_steps"):
        if int(train.get(key, 0)) <= 0:
            raise ValueError(f"train.{key} must be positive")
    if config.get("eval", {}).get("backbone_batchnorm_policy", "eval") not in {"eval", "train_no_update"}:
        raise ValueError("Select an explicit BatchNorm inference policy")


def _distributed_runtime(requested_device="cuda") -> tuple[
    bool, int, int, int, torch.device, dist.ProcessGroup | None
]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    requested = torch.device(requested_device)
    if requested.type != "cuda":
        raise ValueError("Native Mask3D backbones require a CUDA device")
    if requested.index is not None:
        if distributed and requested.index != local_rank:
            raise ValueError("In DDP select GPUs with CUDA_VISIBLE_DEVICES; local ranks choose devices")
        local_rank = requested.index
    if not torch.cuda.is_available():
        raise RuntimeError("LitePT-Mask3D training requires a CUDA worker")
    torch.cuda.set_device(local_rank)
    eval_group: dist.ProcessGroup | None = None
    if distributed:
        dist.init_process_group(
            "nccl",
            init_method="env://",
            timeout=_NCCL_TIMEOUT,
        )
        # Rank-0-only ScanNet validation can run for hours.  NCCL's default
        # 10-minute watchdog kills the idle rank waiting on dist.barrier().
        eval_group = dist.new_group(
            backend="gloo",
            timeout=_EVAL_BARRIER_TIMEOUT,
        )
    return (
        distributed,
        rank,
        local_rank,
        world_size,
        torch.device("cuda", local_rank),
        eval_group,
    )


def _wait_for_rank0_eval(
    distributed: bool, eval_group: dist.ProcessGroup | None
) -> None:
    if distributed:
        dist.barrier(group=eval_group)


def _shutdown_distributed(
    distributed: bool, eval_group: dist.ProcessGroup | None
) -> None:
    if not distributed:
        return
    if eval_group is not None:
        dist.destroy_process_group(eval_group)
    dist.destroy_process_group()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _load_v2_probe_contract(
    args: argparse.Namespace,
    *,
    config_path: Path,
    world_size: int,
) -> dict[str, Any] | None:
    """Validate the immutable V2 probe artifacts before allocating compute."""
    values = (
        args.v2_probe_bundle,
        args.v2_probe_arm_manifest,
        args.v2_probe_result,
    )
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "--v2-probe-bundle, --v2-probe-arm-manifest, and "
            "--v2-probe-result must be supplied together"
        )
    bundle = validate_probe_bundle(args.v2_probe_bundle)
    if int(bundle["execution_contract"]["world_size"]) != int(world_size):
        raise ValueError(
            "V2 probe world-size drift: "
            f"runtime={world_size} bundle={bundle['execution_contract']['world_size']}"
        )
    arm_path = Path(args.v2_probe_arm_manifest).expanduser().resolve(strict=True)
    arm_sha256 = sha256_file(arm_path)
    arm_payload = json.loads(arm_path.read_text(encoding="utf-8"))
    bundle_root = Path(bundle["manifest_path"]).parent
    matching = [
        record
        for record in bundle["arms"].values()
        if (bundle_root / record["manifest"]).resolve() == arm_path
        and record["manifest_sha256"] == arm_sha256
        and {
            key: value
            for key, value in record.items()
            if key not in {"manifest", "manifest_sha256"}
        }
        == arm_payload
    ]
    if len(matching) != 1:
        raise ValueError("V2 probe arm manifest is not uniquely bound to the bundle")
    arm = matching[0]
    source_config = Path(config_path).expanduser().resolve(strict=True)
    if sha256_file(source_config) != arm["config"]["sha256"]:
        raise ValueError("V2 probe config hash differs from its frozen arm manifest")
    result_path = Path(args.v2_probe_result).expanduser().resolve()
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite V2 probe result: {result_path}")
    subset = validate_subset_manifest(Path(bundle["subset_manifest"]["path"]))
    return {
        "bundle": bundle,
        "arm": arm,
        "subset": subset,
        "result_path": result_path,
    }


def _apply_v2_runtime_overrides(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    v2_probe: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Use a byte-identical local copy of V2 inputs without mutating its contract.

    The frozen arm YAML remains the object whose hash is bound to the bundle.
    These optional paths are physical-location overrides only: each selected
    ScanNet scene and its GT cache row must be present before training starts.
    """

    data_location_values = {
        "litept_root": args.v2_runtime_litept_root,
        "scans_root": args.v2_runtime_scans_root,
        "scannet_gt_cache_root": args.v2_runtime_scannet_gt_cache_root,
    }
    initialization_location_values = {
        "decoder_template": getattr(args, "v2_runtime_decoder_template", None),
        "backbone_checkpoint": getattr(
            args, "v2_runtime_backbone_checkpoint", None
        ),
    }
    if not any(
        value is not None
        for value in (*data_location_values.values(), *initialization_location_values.values())
    ):
        return None
    if v2_probe is None:
        raise ValueError("V2 runtime path overrides require a V2 probe bundle")
    missing_data = [
        key for key, value in data_location_values.items() if value is None
    ]
    if missing_data and len(missing_data) != len(data_location_values):
        raise ValueError(
            "V2 runtime LitePT/data path overrides must be supplied together; "
            f"missing {missing_data}"
        )
    resolved: dict[str, str] = {}
    if not missing_data:
        data_locations = {
            key: str(Path(value).expanduser().resolve(strict=True))
            for key, value in data_location_values.items()
        }
        litept_root = Path(data_locations["litept_root"])
        scans_root = Path(data_locations["scans_root"])
        gt_cache_root = Path(data_locations["scannet_gt_cache_root"])
        if not (litept_root / "litept").is_dir():
            raise RuntimeError("V2 runtime LitePT root lacks its litept package")
        if not (gt_cache_root / "manifest.json").is_file():
            raise RuntimeError("V2 runtime ScanNet GT cache lacks manifest.json")
        expected_scene_ids = tuple(
            v2_probe["subset"]["train"]["scene_ids"]
        ) + tuple(v2_probe["subset"]["development"]["scene_ids"])
        missing_scenes = [
            scene_id
            for scene_id in expected_scene_ids
            if not (scans_root / scene_id / "training_pack").is_dir()
        ]
        missing_cache = [
            scene_id
            for scene_id in expected_scene_ids
            if not (gt_cache_root / f"{scene_id}.npz").is_file()
        ]
        if missing_scenes or missing_cache:
            raise RuntimeError(
                "V2 local-data override is incomplete: "
                f"missing_scenes={missing_scenes[:8]} "
                f"missing_gt_cache={missing_cache[:8]}"
            )
        config["model"] = dict(config["model"])
        config["data"] = dict(config["data"])
        config["model"]["litept_root"] = data_locations["litept_root"]
        config["data"]["scans_root"] = data_locations["scans_root"]
        config["data"]["scannet_gt_cache_root"] = data_locations[
            "scannet_gt_cache_root"
        ]
        resolved.update(data_locations)
    config["initialization"] = dict(config["initialization"])
    decoder_template = initialization_location_values["decoder_template"]
    if decoder_template is not None:
        local_decoder = Path(decoder_template).expanduser().resolve(strict=True)
        if sha256_file(local_decoder) != v2_probe["arm"]["decoder_template"]["file_sha256"]:
            raise RuntimeError("V2 local decoder-template file hash drift")
        config["initialization"]["decoder_template"] = str(local_decoder)
        resolved["decoder_template"] = str(local_decoder)
    backbone_checkpoint = initialization_location_values["backbone_checkpoint"]
    if backbone_checkpoint is not None:
        expected_export = v2_probe["arm"]["backbone_export"]
        if not isinstance(expected_export, Mapping):
            raise ValueError(
                "V2 shared-scratch arm cannot receive a runtime backbone checkpoint"
            )
        local_backbone = Path(backbone_checkpoint).expanduser().resolve(strict=True)
        if sha256_file(local_backbone) != expected_export["file_sha256"]:
            raise RuntimeError("V2 local backbone-export file hash drift")
        config["initialization"]["backbone_checkpoint"] = str(local_backbone)
        resolved["backbone_checkpoint"] = str(local_backbone)
    return resolved


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create an immutable result file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prediction_query_diagnostics(
    prediction: Mapping[str, torch.Tensor],
    query_indices: Any,
) -> dict[str, int]:
    """Count every evaluation query and the raw-query proposals it yielded."""
    object_logits = prediction["object_logits"]
    if object_logits.ndim != 2 or object_logits.shape[1] < 2:
        raise ValueError("Mask3D prediction has invalid object logits")
    predicted_class = object_logits.argmax(dim=1)
    total_queries = int(predicted_class.numel())
    no_object_class = int(object_logits.shape[1] - 1)
    no_object_queries = int((predicted_class == no_object_class).sum().item())
    foreground_queries = total_queries - no_object_queries
    valid_proposals = (
        int(query_indices.numel())
        if torch.is_tensor(query_indices)
        else int(np.asarray(query_indices).size)
    )
    if valid_proposals > total_queries:
        raise RuntimeError("raw-query proposal count exceeds available Mask3D queries")
    return {
        "valid_proposals": valid_proposals,
        "foreground_queries": foreground_queries,
        "no_object_queries": no_object_queries,
        "total_queries": total_queries,
    }


def _v2_probe_evaluation_record(
    metrics: Mapping[str, Any],
    *,
    update: int,
    expected_scene_ids: Sequence[str],
) -> dict[str, Any]:
    raw = metrics.get("raw_queries")
    if not isinstance(raw, Mapping):
        raise RuntimeError("V2 probe evaluation is missing raw-query metrics")
    per_scene = raw.get("per_scene")
    if not isinstance(per_scene, Mapping) or set(per_scene) != set(expected_scene_ids):
        raise RuntimeError("V2 probe evaluation scene set drift")
    aggregate_query = raw.get("query_diagnostics")
    if not isinstance(aggregate_query, Mapping):
        raise RuntimeError("V2 probe evaluation lacks aggregate query diagnostics")
    return {
        "update": int(update),
        "metrics": {key: float(raw[key]) for key in ("AP", "AP50", "AP25")},
        "query_diagnostics": {
            key: int(aggregate_query[key])
            for key in (
                "valid_proposals",
                "foreground_queries",
                "no_object_queries",
                "total_queries",
            )
        },
        "per_scene": {
            scene_id: {
                "metrics": {
                    key: float(per_scene[scene_id][key])
                    for key in ("AP", "AP50", "AP25")
                },
                "query_diagnostics": {
                    key: int(per_scene[scene_id]["query_diagnostics"][key])
                    for key in (
                        "valid_proposals",
                        "foreground_queries",
                        "no_object_queries",
                        "total_queries",
                    )
                },
            }
            for scene_id in expected_scene_ids
        },
    }


def _compact_eval_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    compact: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            compact[key] = {
                name: item for name, item in value.items() if name != "per_scene"
            }
        else:
            compact[key] = value
    return compact


def _metric_summary(metrics: dict[str, Any], class_aware: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("raw_queries", "dbscan_secondary"):
        container = metrics.get(key)
        if not isinstance(container, dict):
            continue
        summary[key] = {name: value for name, value in container.items() if name != "per_scene"}
    metric_key = "AP50" if class_aware else "CA-AP50"
    min_metric_key = "min_scene_AP50" if class_aware else "min_scene_CA-AP50"
    raw = metrics.get("raw_queries", {})
    if isinstance(raw, dict):
        summary["headline"] = {
            metric_key: raw.get(metric_key),
            min_metric_key: raw.get(min_metric_key),
        }
    return summary


def _configure_eval_backbone_batchnorm(
    model: LitePTMask3DModel,
    policy: str,
) -> Callable[[], None]:
    """Apply an explicit evaluation BN policy and return a restoration hook.

    ``train_no_update`` keeps the backbone root in eval mode, so voxel
    representative selection and stochastic layers remain deterministic, while
    BatchNorm layers use current-batch statistics like training.  Momentum is
    forced to zero and all BN buffers are restored after the forward pass, so
    this policy cannot alter checkpoint state.
    """
    normalized = str(policy).strip().lower()
    if normalized == "eval":
        return lambda: None
    if normalized != "train_no_update":
        raise ValueError(
            "backbone batchnorm policy must be 'eval' or 'train_no_update', "
            f"got {policy!r}"
        )

    saved: list[
        tuple[
            torch.nn.Module,
            bool,
            float | None,
            dict[str, torch.Tensor],
        ]
    ] = []
    for module in model.backbone.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        buffers = {
            name: buffer.detach().clone()
            for name, buffer in module.named_buffers(recurse=False)
            if buffer is not None
        }
        saved.append((module, bool(module.training), module.momentum, buffers))
        module.train()
        module.momentum = 0.0

    def restore() -> None:
        with torch.no_grad():
            for module, was_training, momentum, buffers in saved:
                for name, buffer in buffers.items():
                    current = getattr(module, name, None)
                    if current is not None:
                        current.copy_(buffer)
                module.training = was_training
                module.momentum = momentum

    return restore


def _make_dataset(
    config: dict[str, Any],
    *,
    train: bool,
    split_key: str | None = None,
) -> MultiSceneDataset:
    data = config["data"]
    selected_split_key = split_key or ("train_split" if train else "val_split")
    scene_dirs = build_scene_list(
        _path(data[selected_split_key]), Path(data["scans_root"])
    )
    return MultiSceneDataset(
        scene_dirs,
        parse_granularities(data),
        use_colors=bool(data.get("use_colors", True)),
        use_normals=bool(data.get("use_normals", False)),
        append_xyz=bool(data.get("append_xyz_to_features", False)),
        preload=bool(data.get("preload", False)),
        preload_workers=int(data.get("preload_workers", 1)),
        max_points=(data.get("max_points") if train else None),
        subsampling_mode=(data.get("subsampling_mode", "sphere_crop") if train else "none"),
        sphere_point_max=(data.get("sphere_point_max") if train else None),
        train_augmentations=bool(data.get("augmentations", False)) if train else False,
        augmentation_profile=str(
            data.get("augmentation_profile", "litept_scannet")
        ),
        color_mean=data.get("color_mean"),
        color_std=data.get("color_std"),
        label_source="scannet_gt",
        scannet_eval_benchmark="scannet20",
        scannet_gt_supervise_all_points=True,
        scannet_gt_cache_root=data.get("scannet_gt_cache_root"),
    )


def _count_dataset_target_instances(
    dataset: MultiSceneDataset,
    granularity: str,
    *,
    class_aware: bool = False,
) -> list[int]:
    """Count valid GT instance IDs once for static logical-batch validation."""
    counts: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = torch.as_tensor(sample["labels_by_granularity"][granularity])
        positive_ids = {
            int(instance_id)
            for instance_id in torch.unique(labels[labels > 0]).tolist()
        }
        if class_aware:
            instance_class_map = sample["instance_classes_by_granularity"][granularity]
            if instance_class_map is None:
                raise ValueError(
                    "Class-aware ScanNet20 training requires per-instance class labels"
                )
            positive_ids = {
                instance_id
                for instance_id in positive_ids
                if 2 <= int(instance_class_map.get(instance_id, -1)) < 20
            }
        counts.append(len(positive_ids))
    return counts


def _canonical_scannet20_class_map(
    instance_class_map: dict[int, int] | None,
) -> dict[int, int] | None:
    """Map the repository's 20-slot benchmark ids to Mask3D's 18 classes.

    The shared ScanNet GT loader retains the standard 20-slot semantic order,
    where wall and floor occupy slots 0 and 1. They are intentionally ignored
    for ScanNet instance AP, leaving 18 foreground classes. Mask3D's classifier
    expects those foreground classes to be contiguous at ids 0..17.
    """
    if instance_class_map is None:
        return None
    foreground_benchmark_ids = set(range(2, 20))
    canonical = {}
    for instance_id, benchmark_id in instance_class_map.items():
        benchmark_id = int(benchmark_id)
        if benchmark_id in (0, 1):
            continue
        if benchmark_id not in foreground_benchmark_ids:
            raise ValueError(
                "ScanNet20 class-aware targets contain a non-foreground class "
                f"slot {benchmark_id} for instance {instance_id}"
            )
        canonical[int(instance_id)] = benchmark_id - 2
    return canonical


def _make_targets(
    batch: Any,
    granularity: str,
    device: torch.device,
    *,
    class_aware: bool,
) -> list[Any]:
    targets = []
    for scene_index in range(batch.num_scenes):
        segments = batch.scan_segments[scene_index]
        if segments is None:
            raise RuntimeError("ScanNet segment ids are missing from the batch")
        class_map = _canonical_scannet20_class_map(
            batch.instance_classes_by_granularity[granularity][scene_index]
        )
        if class_aware and class_map is None:
            raise RuntimeError(
                "Class-aware ScanNet20 training requires per-instance class labels"
            )
        if not class_aware:
            class_map = None
        instance_labels = batch.labels_by_granularity[granularity][scene_index]
        if class_aware:
            # ScanNet's benchmark slots 0/1 (wall/floor) are valid ignored
            # regions, not Mask3D foreground classes. Remove their instance
            # ids from the label tensor before constructing segment targets so
            # they remain explicit negatives and are not looked up in the
            # contiguous 18-class map.
            instance_labels = instance_labels.clone()
            kept_instance_ids = set(class_map or {})
            for instance_id in torch.unique(instance_labels[instance_labels > 0]).tolist():
                if int(instance_id) not in kept_instance_ids:
                    instance_labels[instance_labels == int(instance_id)] = -1
        targets.append(
            build_segment_instance_targets(
                instance_labels,
                segments,
                min_instance_points=1,
                instance_class_map=class_map,
            ).to(device)
        )
    return targets


def _concat_segments(batch: Any, device: torch.device) -> torch.Tensor:
    if any(segment is None for segment in batch.scan_segments):
        raise RuntimeError("Every scene must provide ScanNet segment ids")
    return torch.cat(
        [segment for segment in batch.scan_segments if segment is not None], dim=0
    ).to(device=device, non_blocking=True)


def _ca_named(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "CA-AP": metrics["AP"],
        "CA-AP50": metrics["AP50"],
        "CA-AP25": metrics["AP25"],
        "class_agnostic": True,
        "canonical_scannet_leaderboard_metric": False,
        **metrics,
    }


def _class_aware_named(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **metrics,
        "mAP": metrics["AP"],
        "mAP50": metrics["AP50"],
        "mAP25": metrics["AP25"],
        "min_scene_AP50": float("nan"),
        "protocol": "mask3d_class_aware_scannet20",
        "class_agnostic": False,
        "num_foreground_classes": 18,
        "canonical_scannet_leaderboard_metric": True,
    }


@torch.no_grad()
def _evaluate_impl(
    model: LitePTMask3DModel,
    loader: DataLoader,
    *,
    granularity: str,
    class_aware: bool,
    device: torch.device,
    report_dbscan: bool,
    dbscan_eps: float,
    dbscan_min_samples: int,
    topk_per_image: int | None = 100,
    backbone_batchnorm_policy: str = "eval",
    collect_query_diagnostics: bool = False,
) -> dict[str, Any]:
    model.eval()
    raw_records: list[dict[str, Any]] = []
    dbscan_records: list[dict[str, Any]] = []
    raw_records_by_scene: dict[str, dict[str, Any]] = {}
    dbscan_records_by_scene: dict[str, dict[str, Any]] = {}
    raw_query_diagnostics_by_scene: dict[str, dict[str, int]] = {}
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == total_batches:
            log.info("evaluation batch %d/%d", batch_index, total_batches)
        points = batch.points.to(device=device, non_blocking=True)
        predictions = model(
            points,
            batch.features.to(device=device, non_blocking=True),
            _concat_segments(batch, device),
            scene_offsets=batch.point_offsets.to(device),
        )
        point_parts = batch.split_tensor(batch.points)
        for scene_index, prediction in enumerate(predictions):
            gt_labels = batch.labels_by_granularity[granularity][scene_index]
            gt_ids = torch.where(gt_labels > 0, gt_labels, torch.zeros_like(gt_labels))
            scene_id = batch.scene_ids[scene_index]
            gt_class_map = _canonical_scannet20_class_map(
                batch.instance_classes_by_granularity[granularity][scene_index]
            )
            if class_aware:
                if gt_class_map is None:
                    raise RuntimeError(
                        "Class-aware ScanNet20 evaluation requires per-instance class labels"
                    )
                proposals, scores, pred_class_ids, query_indices = (
                    raw_query_proposals_with_classes(
                        prediction, topk=topk_per_image
                    )
                )
                raw_scene_records = build_instance_ap_records(
                    scene_id=scene_id,
                    gt_ids=gt_ids.numpy(),
                    proposals=proposals,
                    scores=scores,
                    query_indices=query_indices,
                    class_agnostic=False,
                    gt_instance_class_ids=gt_class_map,
                    pred_class_ids=pred_class_ids,
                    min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                    min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                )
            else:
                proposals, scores, query_indices = raw_query_proposals(
                    prediction, topk=topk_per_image
                )
                raw_scene_records = build_instance_ap_records(
                    scene_id=scene_id,
                    gt_ids=gt_ids.numpy(),
                    proposals=proposals,
                    scores=scores,
                    query_indices=query_indices,
                    class_agnostic=True,
                    min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                    min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                )
            raw_records.append(raw_scene_records)
            raw_records_by_scene[scene_id] = raw_scene_records
            if collect_query_diagnostics:
                raw_query_diagnostics_by_scene[scene_id] = (
                    _prediction_query_diagnostics(prediction, query_indices)
                )
            if report_dbscan:
                if class_aware:
                    (
                        split_masks,
                        split_scores,
                        split_class_ids,
                        split_queries,
                    ) = dbscan_split_query_proposals_with_classes(
                        prediction,
                        point_parts[scene_index],
                        eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        topk=topk_per_image,
                    )
                    dbscan_scene_records = build_instance_ap_records(
                        scene_id=scene_id,
                        gt_ids=gt_ids.numpy(),
                        proposals=split_masks,
                        scores=split_scores,
                        query_indices=split_queries,
                        class_agnostic=False,
                        gt_instance_class_ids=gt_class_map,
                        pred_class_ids=split_class_ids,
                        min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                        min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                    )
                else:
                    split_masks, split_scores, split_queries = dbscan_split_query_proposals(
                        prediction,
                        point_parts[scene_index],
                        eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        topk=topk_per_image,
                    )
                    dbscan_scene_records = build_instance_ap_records(
                        scene_id=scene_id,
                        gt_ids=gt_ids.numpy(),
                        proposals=split_masks,
                        scores=split_scores,
                        query_indices=split_queries,
                        class_agnostic=True,
                        min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                        min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                    )
                dbscan_records.append(dbscan_scene_records)
                dbscan_records_by_scene[scene_id] = dbscan_scene_records

    raw = evaluate_official_instance_ap(
        merge_ap_record_sets(raw_records),
        threshold_preset="scannet_official",
        ignore_mode=SCANNET_IGNORE_MODE,
    )
    raw_named = _class_aware_named(raw) if class_aware else _ca_named(raw)
    raw_named["per_scene"] = {
        scene_id: (_class_aware_named if class_aware else _ca_named)(
            evaluate_official_instance_ap(
                records,
                threshold_preset="scannet_official",
                ignore_mode=SCANNET_IGNORE_MODE,
            )
        )
        for scene_id, records in raw_records_by_scene.items()
    }
    if collect_query_diagnostics:
        if set(raw_query_diagnostics_by_scene) != set(raw_named["per_scene"]):
            raise RuntimeError("raw-query diagnostic scene set drift")
        raw_named["query_diagnostics"] = {
            key: sum(values[key] for values in raw_query_diagnostics_by_scene.values())
            for key in (
                "valid_proposals",
                "foreground_queries",
                "no_object_queries",
                "total_queries",
            )
        }
        for scene_id, diagnostics in raw_query_diagnostics_by_scene.items():
            raw_named["per_scene"][scene_id]["query_diagnostics"] = diagnostics
    if class_aware:
        raw_named["min_scene_AP50"] = min(
            scene_metrics["AP50"]
            for scene_metrics in raw_named["per_scene"].values()
        )
    else:
        raw_named["min_scene_CA-AP50"] = min(
            scene_metrics["CA-AP50"]
            for scene_metrics in raw_named["per_scene"].values()
        )
    result: dict[str, Any] = {
        "primary": "raw_queries",
        "raw_queries": raw_named,
        "backbone_batchnorm_policy": str(backbone_batchnorm_policy).lower(),
    }
    if report_dbscan:
        clustered = evaluate_official_instance_ap(
            merge_ap_record_sets(dbscan_records),
            threshold_preset="scannet_official",
            ignore_mode=SCANNET_IGNORE_MODE,
        )
        dbscan_named = {
            **(_class_aware_named(clustered) if class_aware else _ca_named(clustered)),
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
        }
        dbscan_named["per_scene"] = {
            scene_id: (_class_aware_named if class_aware else _ca_named)(
                evaluate_official_instance_ap(
                    records,
                    threshold_preset="scannet_official",
                    ignore_mode=SCANNET_IGNORE_MODE,
                )
            )
            for scene_id, records in dbscan_records_by_scene.items()
        }
        if class_aware:
            dbscan_named["min_scene_AP50"] = min(
                scene_metrics["AP50"]
                for scene_metrics in dbscan_named["per_scene"].values()
            )
        else:
            dbscan_named["min_scene_CA-AP50"] = min(
                scene_metrics["CA-AP50"]
                for scene_metrics in dbscan_named["per_scene"].values()
            )
        result["dbscan_secondary"] = dbscan_named
    return result


@torch.no_grad()
def _evaluate(model, loader, **kwargs):
    model.eval()
    policy = kwargs.get("backbone_batchnorm_policy", "eval")
    if policy not in {"eval", "train_no_update"}:
        raise ValueError("Unknown backbone BatchNorm evaluation policy")
    context = current_batch_without_updates(model.backbone) if policy == "train_no_update" else nullcontext()
    with context:
        return _evaluate_impl(model, loader, **kwargs)


def main(argv=None, *, config=None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config) if config is None else config
    _validate_config(config)
    from delimit3d.tasks.data_identity import mask3d_data_identity
    from delimit3d.tasks.contracts import model_contract
    config["_data_identity"] = mask3d_data_identity(config)
    if args.v2_probe_bundle is not None and (
        args.max_epochs is not None
        or args.eval_only
        or args.checkpoint is not None
        or args.resume is not None
    ):
        raise ValueError(
            "A V2 development probe must use its frozen config and start from "
            "the immutable fresh initialization; --max-epochs, --eval-only, "
            "--checkpoint, and --resume are forbidden"
        )
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    config["_config_sha256"] = config_sha256(config)
    distributed, rank, local_rank, world_size, device, eval_group = (
        _distributed_runtime(args.device)
    )
    v2_probe = _load_v2_probe_contract(
        args,
        config_path=args.config,
        world_size=world_size,
    )
    v2_runtime_overrides = _apply_v2_runtime_overrides(
        args,
        config=config,
        v2_probe=v2_probe,
    )
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s",
    )

    seed = int(config["experiment"].get("seed", 42))
    # Model/template initialization must be identical before DDP broadcasts.
    set_seed(seed)
    model_config = config["model"]
    class_aware = str(model_config["architecture"]) == (
        "litept_mask3d_class_aware_scannet20"
    )
    input_features = str(model_config.get("input_features", "rgbn6")).lower()
    default_input_channels = {"rgb3": 3, "rgbn6": 6}[input_features]
    from .factory import build_model
    model, init_report = build_model(config, device)
    initialization = config["initialization"]
    template_seed = int(init_report["decoder_template"]["downstream_seed"])
    if template_seed != seed:
        raise RuntimeError(
            f"Decoder template seed {template_seed} != experiment seed {seed}"
        )
    if v2_probe is not None:
        probe_arm = v2_probe["arm"]
        expected_source = (
            "scratch" if probe_arm["role"] == "shared_scratch" else "structured3d"
        )
        expected_backbone_hash = (
            None
            if expected_source == "scratch"
            else probe_arm["backbone_export"]["tensor_state_sha256"]
        )
        expected_mode = probe_arm["matched_contract"]["feature_pyramid_mode"]
        if (
            init_report["backbone_source"] != expected_source
            or init_report["decoder_state_sha256"]
            != probe_arm["decoder_template"]["tensor_state_sha256"]
            or (
                expected_backbone_hash is not None
                and init_report["backbone_state_sha256"] != expected_backbone_hash
            )
            or str(model_config.get("feature_pyramid_mode", "")).lower()
            != expected_mode
            or list(model.decoder.memory_channels)
            != probe_arm["matched_contract"]["decoder_memory_channels"]
        ):
            raise RuntimeError("V2 probe initialization/FPN contract drift")
    if args.checkpoint is not None and not args.eval_only:
        raise ValueError("--checkpoint is currently supported only with --eval-only")
    if args.eval_only and args.checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint")
    if args.resume is not None and (args.eval_only or args.checkpoint is not None):
        raise ValueError("--resume cannot be combined with --eval-only or --checkpoint")
    baseline_config_hash = config["_config_sha256"] if config["train"].get("save_latest_each_epoch", False) else None
    resume_checkpoint = None
    loaded_checkpoint_epoch: int | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume.expanduser().resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("data_identity") != config["_data_identity"]:
            raise ValueError("Mask3D resume data changed; convert a historical checkpoint explicitly")
        if baseline_config_hash and resume_checkpoint.get("baseline_config_sha256") != baseline_config_hash:
            raise ValueError("Baseline resume configuration changed")
        for key in ("backbone_source", "backbone_checkpoint", "decoder_template"):
            if resume_checkpoint.get("initialization", {}).get(key) != init_report.get(key):
                raise ValueError(f"Mask3D resume initialization changed: {key}")
        result = model.load_state_dict(
            resume_checkpoint["model_state_dict"], strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict resume model load failed: {result}")
        loaded_checkpoint_epoch = int(resume_checkpoint["epoch"])
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Strict full-model checkpoint load failed: {result}")
        loaded_checkpoint_epoch = int(checkpoint["epoch"])
    if args.eval_queries is not None:
        model.decoder.eval_queries = int(args.eval_queries)
        log.info("using %d evaluation queries", model.decoder.eval_queries)
    train_config = config["train"]
    ddp_find_unused_parameters = bool(
        train_config.get("ddp_find_unused_parameters", False)
    )
    backbone_freeze_epochs = int(train_config.get("backbone_freeze_epochs", 0))
    if backbone_freeze_epochs < 0:
        raise ValueError(
            "train.backbone_freeze_epochs must be non-negative, got "
            f"{backbone_freeze_epochs}"
        )
    if backbone_freeze_epochs > 0 and bool(initialization.get("freeze_backbone", False)):
        raise ValueError(
            "Use initialization.freeze_backbone=false with the staged "
            "train.backbone_freeze_epochs schedule"
        )
    frozen_hash_before: str | None = None
    backbone_hash_at_unfreeze: str | None = None
    wrapped: torch.nn.Module = model
    if distributed:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            # Mask3D's criterion intentionally lives outside ``forward`` and
            # particular decoder branches can have no matched target on one
            # rank.  The probe contract opts into unused-parameter discovery
            # so both ranks can complete the same reduction safely.
            find_unused_parameters=ddp_find_unused_parameters,
        )
    # DDP is wrapped while the scheduled backbone is trainable so it installs
    # gradient hooks before the first freeze window.  The model then enforces
    # eval mode and requires_grad=False until the scheduled unfreeze epoch.
    if backbone_freeze_epochs > 0:
        model.set_backbone_frozen(True)
    frozen_hash_before = model.backbone_state_hash() if model.backbone_frozen else None
    init_report["backbone_frozen_at_start"] = bool(model.backbone_frozen)
    init_report["backbone_freeze_epochs"] = backbone_freeze_epochs

    data = config["data"]
    eval_config = config.get("eval", {})
    eval_batchnorm_policy = str(
        eval_config.get("backbone_batchnorm_policy", "eval")
    ).lower()
    ddp_padding_policy = str(
        train_config.get("ddp_padding_policy", "legacy")
    ).lower()
    include_auxiliary = bool(
        config.get("loss", {}).get("include_auxiliary", True)
    )
    granularity = parse_granularities(data)[0]
    train_dataset = _make_dataset(config, train=True)
    if v2_probe is not None:
        expected_train = int(v2_probe["subset"]["train"]["count"])
        expected_development = int(v2_probe["subset"]["development"]["count"])
        observed_development = len(_split_scene_ids(data["val_split"]))
        if (
            len(train_dataset) != expected_train
            or observed_development != expected_development
        ):
            raise RuntimeError(
                "V2 probe split cardinality drift: "
                f"train={len(train_dataset)}/{expected_train} "
                f"development={observed_development}/{expected_development}"
            )
    val_dataset = _make_dataset(config, train=False) if rank == 0 else None
    acceptance_config = eval_config.get("acceptance", {})
    report_train_metrics = bool(
        eval_config.get(
            "report_train_metrics",
            acceptance_config.get(
                "require_high_train_AP50",
                acceptance_config.get("require_high_train_CA_AP50", False),
            ),
        )
    )
    train_eval_split_key = str(
        eval_config.get("train_eval_split") or data.get("train_eval_split") or ""
    ).strip()
    if train_eval_split_key:
        config["data"]["train_eval_split"] = train_eval_split_key
    train_eval_dataset = (
        _make_dataset(
            config,
            train=False,
            split_key="train_eval_split" if train_eval_split_key else "train_split",
        )
        if rank == 0 and report_train_metrics
        else None
    )
    train_sampler = (
        ExactDistributedSceneSampler(
            len(train_dataset),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=bool(train_config.get("drop_last", False)),
        )
        if distributed
        else None
    )
    loader_workers = int(train_config.get("num_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_scenes_per_step", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=loader_workers,
        collate_fn=collate_multi_scene_samples,
        pin_memory=True,
        drop_last=bool(train_config.get("drop_last", False)),
        persistent_workers=loader_workers > 0 and bool(train_config.get("persistent_workers", True)),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=loader_workers,
            collate_fn=collate_multi_scene_samples,
            pin_memory=True,
            persistent_workers=loader_workers > 0 and bool(train_config.get("persistent_workers", True)),
        )
        if val_dataset is not None
        else None
    )
    train_eval_loader = (
        DataLoader(
            train_eval_dataset,
            batch_size=int(eval_config.get("train_eval_batch_scenes", 1)),
            shuffle=False,
            num_workers=loader_workers,
            collate_fn=collate_multi_scene_samples,
            pin_memory=True,
            persistent_workers=loader_workers > 0 and bool(train_config.get("persistent_workers", True)),
        )
        if train_eval_dataset is not None
        else None
    )

    if model_config.get("mask_only", False):
        from delimit3d.tasks.mask3d.losses.mask3d_mask_only_loss import Mask3DMaskOnlyCriterion
        criterion_type = Mask3DMaskOnlyCriterion
        if class_aware or eval_config.get("report_dbscan", False):
            raise ValueError("Mask-only baseline requires class-agnostic raw-query evaluation")
    else:
        criterion_type = Mask3DSegmentSetCriterion
    criterion = criterion_type(
        num_classes=int(model.decoder.num_classes),
        include_auxiliary=include_auxiliary,
    ).to(device)
    parameter_groups = model.parameter_groups(
        backbone_lr=float(train_config.get("backbone_lr", 1e-4)),
        decoder_lr=float(train_config.get("decoder_lr", 1e-4)),
        include_frozen_backbone=backbone_freeze_epochs > 0,
    )
    intended_max_lrs = [float(group["lr"]) for group in parameter_groups]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(train_config.get("weight_decay", 1e-4))
    )
    scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None
    start_epoch = 1
    first_train_set_loss: float | None = None
    if resume_checkpoint is not None:
        # Load optimizer/scheduler *after* OneCycleLR is constructed. Loading
        # AdamW first makes group["lr"] the live scheduled value, and
        # OneCycleLR(max_lr=group["lr"]) rebuilds a shrunken cycle.
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        first_train_set_loss = resume_checkpoint.get("first_train_set_loss")
        init_report["resume"] = {
            "checkpoint": str(args.resume),
            "completed_epoch": int(resume_checkpoint["epoch"]),
            "backbone_state_sha256": resume_checkpoint.get(
                "backbone_state_sha256"
            ),
            "decoder_state_sha256": resume_checkpoint.get(
                "decoder_state_sha256"
            ),
        }
        log.info(
            "resuming checkpoint=%s at epoch=%d",
            args.resume,
            start_epoch,
        )
    output_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "initialization_report.json").write_text(
            json.dumps(init_report, indent=2), encoding="utf-8"
        )

    if args.eval_only:
        if rank == 0 and val_loader is not None:
            metrics = _evaluate(
                model,
                val_loader,
                granularity=granularity,
                class_aware=class_aware,
                device=device,
                report_dbscan=(
                    bool(eval_config.get("report_dbscan", True))
                    and not args.skip_dbscan
                ),
                dbscan_eps=float(eval_config.get("dbscan_eps", 0.95)),
                dbscan_min_samples=int(eval_config.get("dbscan_min_samples", 1)),
                topk_per_image=(
                    None
                    if int(eval_config.get("topk_per_image", 100)) < 0
                    else int(eval_config.get("topk_per_image", 100))
                ),
                backbone_batchnorm_policy=str(
                    eval_config.get("backbone_batchnorm_policy", "eval")
                ).lower(),
            )
            eval_epoch = int(loaded_checkpoint_epoch or 0)
            report = {
                "epoch": eval_epoch,
                "checkpoint": str(args.checkpoint),
                "validation": metrics,
            }
            report_path = output_dir / f"eval_epoch_{eval_epoch:04d}.json"
            report_path.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            summary = _metric_summary(metrics, class_aware)
            headline = summary.get("headline", {})
            metric_key = "AP50" if class_aware else "CA-AP50"
            min_metric_key = "min_scene_AP50" if class_aware else "min_scene_CA-AP50"
            log.info(
                "wrote %s epoch=%d val raw %s=%s min-scene %s=%s",
                report_path,
                eval_epoch,
                metric_key,
                headline.get(metric_key),
                min_metric_key,
                headline.get(min_metric_key),
            )
            print(
                json.dumps(
                    {"epoch": eval_epoch, "path": str(report_path), **summary},
                    indent=2,
                )
            )
            _append_jsonl(
                output_dir / "metrics.jsonl",
                {
                    "epoch": eval_epoch,
                    "source": "eval_only",
                    "validation": {
                        key: summary[key]
                        for key in ("raw_queries", "dbscan_secondary")
                        if key in summary
                    },
                },
            )
        _wait_for_rank0_eval(distributed, eval_group)
        _shutdown_distributed(distributed, eval_group)
        return

    max_epochs = int(train_config.get("max_epochs", 100))
    eval_every = int(train_config.get("eval_every_epochs", 10))
    save_every = int(train_config.get("save_every_epochs", 10))
    report_loss_components = bool(
        train_config.get("report_loss_components", False)
    )
    if backbone_freeze_epochs > max_epochs:
        raise ValueError(
            "train.backbone_freeze_epochs cannot exceed train.max_epochs: "
            f"freeze={backbone_freeze_epochs} max_epochs={max_epochs}"
        )
    batch_scenes_per_step = int(train_config.get("batch_scenes_per_step", 1))
    gradient_accumulation_steps = max(
        1, int(train_config.get("gradient_accumulation_steps", 1))
    )
    if batch_scenes_per_step <= 0:
        raise ValueError(
            "train.batch_scenes_per_step must be positive, "
            f"got {batch_scenes_per_step}"
        )
    if ddp_padding_policy == "zero_and_rescale" and distributed:
        if not isinstance(train_sampler, ExactDistributedSceneSampler):
            raise ValueError(
                "train.ddp_padding_policy=zero_and_rescale requires the exact "
                "distributed scene sampler"
            )
        if batch_scenes_per_step != 1:
            raise ValueError(
                "train.ddp_padding_policy=zero_and_rescale currently requires "
                "batch_scenes_per_step=1"
            )
    if max_epochs <= 0 or eval_every <= 0 or save_every <= 0:
        raise ValueError(
            "train.max_epochs, eval_every_epochs, and save_every_epochs "
            "must be positive"
        )
    normalization_target_instances = train_config.get(
        "normalization_target_instances"
    )
    if gradient_accumulation_steps > 1 and normalization_target_instances is None:
        raise ValueError(
            "Mask3D-equivalent gradient accumulation requires a logical-batch "
            "normalization_target_instances value. Use a real joint batch or "
            "configure the total target-instance count explicitly."
        )
    if distributed and gradient_accumulation_steps > 1:
        if not isinstance(train_sampler, ExactDistributedSceneSampler):
            raise ValueError(
                "Distributed logical accumulation requires the exact sampler "
                "that marks padded scenes."
            )
        if batch_scenes_per_step != 1:
            raise ValueError(
                "Distributed logical accumulation currently requires "
                "batch_scenes_per_step=1 so padded scenes can be zero-weighted."
            )
        if len(train_loader) != gradient_accumulation_steps:
            raise ValueError(
                "Distributed accumulation must cover the complete local loader: "
                f"loader_batches={len(train_loader)} != "
                f"accumulation={gradient_accumulation_steps}"
            )
    if gradient_accumulation_steps > 1:
        if len(train_loader) % gradient_accumulation_steps != 0:
            raise ValueError(
                "The logical accumulation group must contain complete loader "
                "batches; set drop_last or choose a compatible accumulation "
                f"length (loader_batches={len(train_loader)}, "
                f"accumulation={gradient_accumulation_steps})"
            )
        if bool(train_config.get("require_full_logical_batch", False)):
            expected_scenes = (
                len(train_dataset)
                if distributed
                else batch_scenes_per_step * gradient_accumulation_steps
            )
            if expected_scenes != len(train_dataset):
                raise ValueError(
                    "require_full_logical_batch expects one accumulation group "
                    "to cover the entire training split: "
                    f"{expected_scenes} != {len(train_dataset)} scenes"
                )
            if distributed:
                observed_tensor = torch.zeros(1, device=device, dtype=torch.float32)
                if rank == 0:
                    observed_tensor[0] = sum(
                        _count_dataset_target_instances(
                            train_dataset, granularity, class_aware=class_aware
                        )
                    )
                dist.broadcast(observed_tensor, src=0)
                observed_target_instances = int(observed_tensor.item())
            else:
                observed_target_instances = sum(
                    _count_dataset_target_instances(
                        train_dataset, granularity, class_aware=class_aware
                    )
                )
            configured_target_instances = float(normalization_target_instances)
            if configured_target_instances != float(observed_target_instances):
                raise ValueError(
                    "normalization_target_instances does not match the training "
                    "split: "
                    f"configured={configured_target_instances}, "
                    f"observed={observed_target_instances}"
                )
            log.info(
                "validated one logical accumulation group: scenes=%d "
                "target_instances=%d local_batches=%d padding=%d world_size=%d",
                expected_scenes,
                observed_target_instances,
                len(train_loader),
                (
                    train_sampler.padding_count
                    if isinstance(train_sampler, ExactDistributedSceneSampler)
                    else 0
                ),
                world_size,
            )

    scheduler_config = dict(config.get("scheduler", {}))
    if scheduler_config.get("max_lr") is None:
        scheduler_config["max_lr"] = intended_max_lrs
    scheduler, scheduler_steps_per_epoch = build_onecycle_scheduler(
        optimizer,
        max_epochs=max_epochs,
        num_batches=len(train_loader),
        gradient_accumulation_steps=gradient_accumulation_steps,
        config=scheduler_config,
    )
    log.info(
        "using OneCycleLR: optimizer_steps_per_epoch=%d total_optimizer_steps=%d",
        scheduler_steps_per_epoch,
        scheduler_steps_per_epoch * max_epochs,
    )
    init_report["training_protocol"] = {
        "scheduler": "OneCycleLR",
        "scheduler_interval": "optimizer_step",
        "max_epochs": max_epochs,
        "optimizer_steps_per_epoch": scheduler_steps_per_epoch,
        "scheduler_total_steps": scheduler_steps_per_epoch * max_epochs,
        "batch_scenes_per_step": batch_scenes_per_step,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "backbone_lr": float(train_config.get("backbone_lr", 1e-4)),
        "decoder_lr": float(train_config.get("decoder_lr", 1e-4)),
        "backbone_freeze_epochs": backbone_freeze_epochs,
        "backbone_unfreeze_epoch": (
            backbone_freeze_epochs + 1 if backbone_freeze_epochs < max_epochs else None
        ),
        "optimizer_group_names": [
            str(group.get("name", f"group_{index}"))
            for index, group in enumerate(parameter_groups)
        ],
        "include_auxiliary_losses": include_auxiliary,
        "evaluation_backbone_batchnorm_policy": eval_batchnorm_policy,
        "train_queries": int(model.decoder.train_queries),
        "eval_queries": int(model.decoder.eval_queries),
        "class_agnostic": not class_aware,
        "classification_head": not bool(model_config.get("mask_only", False)),
        "classification_loss": not bool(model_config.get("mask_only", False)),
        "semantic_loss": False if model_config.get("mask_only", False) else class_aware,
        "num_classes_with_no_object": int(model.decoder.num_classes),
        "num_foreground_classes": int(model.decoder.num_classes - 1),
        "evaluation_metric": "AP/AP50/AP25" if class_aware else "CA-AP/CA-AP50/CA-AP25",
        "train_metrics_reported": bool(train_eval_loader is not None),
        "train_eval_scenes": (
            int(len(train_eval_dataset)) if train_eval_dataset is not None else 0
        ),
        "train_eval_split": train_eval_split_key or None,
        "validation_metrics_reported": bool(val_loader is not None),
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "global_train_scenes": int(len(train_dataset)),
        "local_train_batches": int(len(train_loader)),
        "padded_train_samples": int(
            train_sampler.padding_count
            if isinstance(train_sampler, ExactDistributedSceneSampler)
            else 0
        ),
        "ddp_logical_loss_scale": int(world_size) if distributed else 1,
        "ddp_padding_policy": ddp_padding_policy,
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
    }
    if rank == 0:
        (output_dir / "initialization_report.json").write_text(
            json.dumps(init_report, indent=2), encoding="utf-8"
        )
    if resume_checkpoint is not None:
        scheduler_state = resume_checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise RuntimeError(
                "Cannot resume this run safely: checkpoint has no "
                "scheduler_state_dict. Start a fresh run with the OneCycleLR "
                "setup instead of mixing it with a constant-LR checkpoint."
            )
        checkpoint_total_steps = int(scheduler_state.get("total_steps", -1))
        expected_total_steps = scheduler_steps_per_epoch * max_epochs
        if checkpoint_total_steps != expected_total_steps:
            raise RuntimeError(
                "Cannot resume with a different OneCycleLR horizon: "
                f"checkpoint_total_steps={checkpoint_total_steps}, "
                f"current_total_steps={expected_total_steps}. "
                "Use the same max_epochs and logical batch layout, or start "
                "a fresh run."
            )
        expected_frontier = int(resume_checkpoint["epoch"]) * scheduler_steps_per_epoch
        if int(scheduler_state.get("last_epoch", -1)) != expected_frontier:
            raise ValueError("Mask3D resume scheduler differs from the completed epoch")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(scheduler_state)
        log.info(
            "restored OneCycleLR last_epoch=%d lr=%s group_max_lr=%s",
            scheduler.last_epoch,
            [f"{lr:.6g}" for lr in scheduler.get_last_lr()],
            [f"{float(group['max_lr']):.6g}" for group in optimizer.param_groups],
        )
    if backbone_freeze_epochs > 0 and start_epoch > backbone_freeze_epochs:
        if model.backbone_frozen:
            backbone_hash_at_unfreeze = model.backbone_state_hash()
            if (
                frozen_hash_before is not None
                and backbone_hash_at_unfreeze != frozen_hash_before
            ):
                raise RuntimeError(
                    "Scheduled-frozen LitePT changed before the resumed unfreeze"
                )
            model.set_backbone_frozen(False)
            log.info(
                "resumed after staged backbone freeze: unfreezing LitePT at epoch=%d",
                start_epoch,
            )
    v2_probe_evaluations: list[dict[str, Any]] = []
    v2_expected_scene_ids: tuple[str, ...] = ()
    v2_expected_updates: list[int] = []
    v2_expected_total_updates: int | None = None
    observed_optimizer_updates = 0
    if v2_probe is not None:
        v2_expected_scene_ids = tuple(
            v2_probe["subset"]["development"]["scene_ids"]
        )
        v2_expected_updates = list(
            v2_probe["bundle"]["execution_contract"]["expected_evaluation_updates"]
        )
        v2_expected_total_updates = int(
            v2_probe["bundle"]["execution_contract"]["global_optimizer_updates"]
        )
        observed_schedule = [
            0,
            *[
                epoch * scheduler_steps_per_epoch
                for epoch in range(eval_every, max_epochs + 1, eval_every)
            ],
        ]
        if (
            scheduler_steps_per_epoch * max_epochs != v2_expected_total_updates
            or observed_schedule != v2_expected_updates
            or (rank == 0 and val_loader is None)
        ):
            raise RuntimeError(
                "V2 probe update/evaluation schedule drift: "
                f"total={scheduler_steps_per_epoch * max_epochs}/"
                f"{v2_expected_total_updates} evaluations={observed_schedule}/"
                f"{v2_expected_updates}"
            )
        init_report["v2_development_probe"] = {
            "bundle_manifest": v2_probe["bundle"]["manifest_path"],
            "bundle_manifest_sha256": v2_probe["bundle"]["manifest_sha256"],
            "arm_id": v2_probe["arm"]["arm_id"],
            "arm_manifest_sha256": v2_probe["arm"]["manifest_sha256"],
            "result_path": str(v2_probe["result_path"]),
            "expected_updates": v2_expected_updates,
            "runtime_location_overrides": v2_runtime_overrides,
        }
        if rank == 0:
            (output_dir / "initialization_report.json").write_text(
                json.dumps(init_report, indent=2), encoding="utf-8"
            )
            zero_metrics = _evaluate(
                model,
                val_loader,
                granularity=granularity,
                class_aware=class_aware,
                device=device,
                report_dbscan=False,
                dbscan_eps=float(eval_config.get("dbscan_eps", 0.95)),
                dbscan_min_samples=int(eval_config.get("dbscan_min_samples", 1)),
                topk_per_image=(
                    None
                    if int(eval_config.get("topk_per_image", 100)) < 0
                    else int(eval_config.get("topk_per_image", 100))
                ),
                backbone_batchnorm_policy=eval_batchnorm_policy,
                collect_query_diagnostics=True,
            )
            v2_probe_evaluations.append(
                _v2_probe_evaluation_record(
                    zero_metrics,
                    update=0,
                    expected_scene_ids=v2_expected_scene_ids,
                )
            )
        _wait_for_rank0_eval(distributed, eval_group)
    best_acceptance_score = float("-inf")
    if resume_checkpoint is not None and not resume_checkpoint.get("validation_completed", True):
        _wait_for_rank0_eval(distributed, eval_group)
        if rank == 0:
            resumed_metrics = {"epoch": int(resume_checkpoint["epoch"]), "resumed_validation": True}
            for key, loader in (("train", train_eval_loader), ("validation", val_loader)):
                if loader is not None:
                    resumed_metrics[key] = _evaluate(model, loader, granularity=granularity,
                        class_aware=class_aware, device=device,
                        report_dbscan=bool(eval_config.get("report_dbscan", True)) and not args.skip_dbscan,
                        dbscan_eps=float(eval_config.get("dbscan_eps", .95)),
                        dbscan_min_samples=int(eval_config.get("dbscan_min_samples", 1)),
                        topk_per_image=None if int(eval_config.get("topk_per_image", 100)) < 0 else int(eval_config.get("topk_per_image", 100)),
                        backbone_batchnorm_policy=eval_batchnorm_policy)
            _append_jsonl(output_dir / "metrics.jsonl", resumed_metrics)
            from delimit3d.runtime.records import save_torch
            save_torch(output_dir / "checkpoints/latest.pt", {**resume_checkpoint, "validation_completed": True})
        _wait_for_rank0_eval(distributed, eval_group)
    if start_epoch > max_epochs + 1:
        raise ValueError(
            f"Resume starts at epoch {start_epoch}, beyond max_epochs={max_epochs}"
        )
    if start_epoch == max_epochs + 1 and rank == 0:
        (output_dir / "status.json").write_text(json.dumps({"epoch": max_epochs, "complete": True}, indent=2))
    run_start_time = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        if train_config.get("deterministic_epoch_seed", False):
            if train_config.get("persistent_workers", True):
                raise ValueError("Epoch seeding requires persistent_workers=false")
            set_seed(seed + epoch)
        epoch_start_time = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if backbone_freeze_epochs > 0 and epoch == backbone_freeze_epochs + 1:
            backbone_hash_at_unfreeze = model.backbone_state_hash()
            if (
                frozen_hash_before is not None
                and backbone_hash_at_unfreeze != frozen_hash_before
            ):
                raise RuntimeError(
                    "Scheduled-frozen LitePT changed before the unfreeze epoch"
                )
            model.set_backbone_frozen(False)
            log.info(
                "staged backbone schedule: unfreezing LitePT at epoch=%d",
                epoch,
            )
        wrapped.train()
        epoch_loss = 0.0
        epoch_steps = 0
        loss_component_names = (
            "total",
            "final_only",
            "auxiliary_total",
            "final_class",
            "final_mask_bce",
            "final_mask_dice",
        )
        loss_component_sums = torch.zeros(
            len(loss_component_names) + 1,
            device=device,
            dtype=torch.float64,
        )
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        accumulated_logical_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            accumulation_start = ((batch_index - 1) // gradient_accumulation_steps) * (
                gradient_accumulation_steps
            )
            accumulation_size = min(
                gradient_accumulation_steps,
                total_batches - accumulation_start,
            )
            points = batch.points.to(device=device, non_blocking=True)
            features = batch.features.to(device=device, non_blocking=True)
            segments = _concat_segments(batch, device)
            targets = _make_targets(
                batch,
                granularity,
                device,
                class_aware=class_aware,
            )
            predictions = wrapped(
                points,
                features,
                segments,
                scene_offsets=batch.point_offsets.to(device),
            )
            losses = criterion(
                predictions,
                targets,
                normalization_denominator=(
                    normalization_target_instances
                    if gradient_accumulation_steps > 1
                    else None
                ),
            )
            if report_loss_components and not any(batch.padding):
                loss_component_sums[:-1] += torch.tensor(
                    [
                        float(losses["loss_total"].detach().item()),
                        float(
                            (
                                losses["loss_total"].detach()
                                - losses["loss_aux"]
                            ).item()
                        ),
                        float(losses["loss_aux"].item()),
                        float(losses["loss_class"].item()),
                        float(losses["loss_mask_bce"].item()),
                        float(losses["loss_mask_dice"].item()),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                loss_component_sums[-1] += 1.0
            if gradient_accumulation_steps > 1:
                # The memory-safe capacity gate uses one scene per microbatch
                # and a validated full logical batch. Reproduce Mask3D's
                # concatenated-batch denominators while freeing each graph
                # immediately after its backward pass.
                batch_size = batch_scenes_per_step
                if distributed:
                    # DDP ranks jointly cover the global logical batch. The
                    # exact sampler may add marked padding positions only to
                    # keep all ranks synchronized; those positions do not
                    # increase the logical query denominator.
                    logical_scene_count = len(train_dataset)
                else:
                    rank_dataset_size = (
                        len(train_sampler)
                        if train_sampler is not None
                        else len(train_dataset)
                    )
                    group_scene_start = accumulation_start * batch_size
                    group_scene_end = min(
                        group_scene_start + accumulation_size * batch_size,
                        rank_dataset_size,
                    )
                    logical_scene_count = max(group_scene_end - group_scene_start, 1)
                total_query_slots = logical_scene_count * int(
                    model.decoder.train_queries
                )
                total_target_instances = float(normalization_target_instances)
                if total_target_instances > total_query_slots:
                    raise ValueError(
                        "logical-batch target count exceeds available query slots: "
                        f"targets={total_target_instances} slots={total_query_slots}"
                    )
                total_class_denominator = total_target_instances + criterion.no_object_weight * (
                    total_query_slots - total_target_instances
                )
                backward_loss = criterion.microbatch_loss(
                    losses,
                    total_target_instances=total_target_instances,
                    total_class_denominator=total_class_denominator,
                )
                if distributed:
                    # DDP averages gradients across ranks. Scale real local
                    # contributions by world size to recover the global sum;
                    # marked padding scenes still execute the collective but
                    # contribute exactly zero.
                    if any(batch.padding):
                        backward_loss = backward_loss * 0.0
                    else:
                        backward_loss = backward_loss * float(world_size)
            else:
                backward_loss = losses["loss_total"]
                if (
                    distributed
                    and ddp_padding_policy == "zero_and_rescale"
                    and isinstance(train_sampler, ExactDistributedSceneSampler)
                    and train_sampler.padding_count > 0
                ):
                    # With a non-divisible scene count, ExactDistributedSceneSampler
                    # marks synthetic positions. DDP averages gradients across
                    # ranks, so retain only real scenes and scale them back to the
                    # real global sum. This is the one-scene/four-GPU diagnostic
                    # correction; divisible layouts retain legacy scaling.
                    if any(batch.padding):
                        backward_loss = backward_loss * 0.0
                    else:
                        backward_loss = backward_loss * float(world_size)
            backward_loss.backward()
            accumulated_logical_loss += float(backward_loss.detach().item())
            accumulation_end = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == total_batches
            )
            if accumulation_end:
                grad_clip_norm = train_config.get("grad_clip_norm", 10.0)
                if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for group in parameter_groups
                            for parameter in group["params"]
                        ],
                        float(grad_clip_norm),
                    )
                optimizer.step()
                scheduler.step()
                observed_optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)
                epoch_loss += accumulated_logical_loss
                epoch_steps += 1
                accumulated_logical_loss = 0.0

        reduced = torch.tensor(
            [epoch_loss, epoch_steps], device=device, dtype=torch.float64
        )
        if distributed:
            dist.all_reduce(reduced)
            if report_loss_components:
                dist.all_reduce(loss_component_sums)
        mean_loss = float(reduced[0].item() / max(reduced[1].item(), 1.0))
        epoch_seconds = time.perf_counter() - epoch_start_time
        if first_train_set_loss is None:
            first_train_set_loss = mean_loss
        if model.backbone_frozen:
            frozen_hash_after = model.backbone_state_hash()
            if frozen_hash_after != frozen_hash_before:
                raise RuntimeError("Frozen LitePT parameters or buffers changed during training")
        if rank == 0:
            log.info(
                "epoch=%d set_loss=%.6f lr=%s",
                epoch,
                mean_loss,
                [f"{lr:.6g}" for lr in scheduler.get_last_lr()],
            )
            epoch_record: dict[str, Any] | None = {
                "epoch": epoch,
                "train_set_loss": mean_loss,
                "learning_rates": [float(lr) for lr in scheduler.get_last_lr()],
                "optimizer_group_names": [
                    str(group.get("name", f"group_{index}"))
                    for index, group in enumerate(optimizer.param_groups)
                ],
                "backbone_frozen": bool(model.backbone_frozen),
                "optimizer_steps_per_epoch": scheduler_steps_per_epoch,
                "epoch_seconds": epoch_seconds,
                "optimizer_steps_per_second": scheduler_steps_per_epoch / max(epoch_seconds, 1e-9),
                "scenes_per_second": len(train_dataset) / max(epoch_seconds, 1e-9),
            }
            if report_loss_components:
                forward_batches = int(loss_component_sums[-1].item())
                if forward_batches <= 0:
                    raise RuntimeError(
                        "Component-loss reporting observed no real training batches"
                    )
                epoch_record["train_forward_loss_components"] = {
                    "averaging": "unweighted_mean_over_real_forward_batches",
                    "forward_batches": forward_batches,
                    **{
                        name: float(loss_component_sums[index].item())
                        / float(forward_batches)
                        for index, name in enumerate(loss_component_names)
                    },
                }
        else:
            epoch_record = None
        if rank == 0:
            if epoch_record is None:
                raise RuntimeError("rank-0 epoch record was not initialized")
            _append_jsonl(output_dir / "metrics.jsonl", epoch_record)

        if epoch % save_every == 0 and rank == 0:
            checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_contract": model_contract(config),
                    "data_identity": config["_data_identity"],
                    "baseline_config_sha256": baseline_config_hash,
                    "deterministic_epoch_seed": seed + epoch + 1,
                    "validation_completed": epoch % eval_every != 0,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "first_train_set_loss": first_train_set_loss,
                    "initialization": init_report,
                    "backbone_state_sha256": model.backbone_state_hash(),
                    "decoder_state_sha256": model.decoder_state_hash(),
                },
                checkpoint_path,
            )
            log.info("saved %s", checkpoint_path)

        if epoch % eval_every == 0:
            _wait_for_rank0_eval(distributed, eval_group)
            if rank == 0:
                report_dbscan = bool(eval_config.get("report_dbscan", True)) and not args.skip_dbscan
                log.info(
                    "epoch=%d starting rank-0 evaluation (train_eval_scenes=%s val=%s dbscan=%s)",
                    epoch,
                    0 if train_eval_loader is None else len(train_eval_loader.dataset),
                    val_loader is not None,
                    report_dbscan,
                )
                train_metrics = (
                    _evaluate(
                        model,
                        train_eval_loader,
                        granularity=granularity,
                        class_aware=class_aware,
                        device=device,
                        report_dbscan=report_dbscan,
                        dbscan_eps=float(eval_config.get("dbscan_eps", 0.95)),
                        dbscan_min_samples=int(eval_config.get("dbscan_min_samples", 1)),
                        topk_per_image=(
                            None
                            if int(eval_config.get("topk_per_image", 100)) < 0
                            else int(eval_config.get("topk_per_image", 100))
                        ),
                        backbone_batchnorm_policy=eval_batchnorm_policy,
                    )
                    if train_eval_loader is not None
                    else None
                )
                validation_metrics = (
                    _evaluate(
                        model,
                        val_loader,
                        granularity=granularity,
                        class_aware=class_aware,
                        device=device,
                        report_dbscan=report_dbscan,
                        dbscan_eps=float(eval_config.get("dbscan_eps", 0.95)),
                        dbscan_min_samples=int(eval_config.get("dbscan_min_samples", 1)),
                        topk_per_image=(
                            None
                            if int(eval_config.get("topk_per_image", 100)) < 0
                            else int(eval_config.get("topk_per_image", 100))
                        ),
                        backbone_batchnorm_policy=eval_batchnorm_policy,
                        collect_query_diagnostics=v2_probe is not None,
                    )
                    if val_loader is not None
                    else None
                )
                if epoch_record is None:
                    raise RuntimeError("rank-0 epoch record was not initialized")
                if train_metrics is not None:
                    epoch_record["train"] = train_metrics
                    train_raw = train_metrics["raw_queries"]
                    metric_key = "AP50" if class_aware else "CA-AP50"
                    min_metric_key = "min_scene_AP50" if class_aware else "min_scene_CA-AP50"
                    log.info(
                        "epoch=%d train raw %s=%.4f min-scene %s=%.4f",
                        epoch,
                        metric_key,
                        train_raw[metric_key],
                        min_metric_key,
                        train_raw[min_metric_key],
                    )
                if validation_metrics is not None:
                    epoch_record["validation"] = validation_metrics
                    if v2_probe is not None:
                        v2_probe_evaluations.append(
                            _v2_probe_evaluation_record(
                                validation_metrics,
                                update=epoch * scheduler_steps_per_epoch,
                                expected_scene_ids=v2_expected_scene_ids,
                            )
                        )
                    val_raw = validation_metrics["raw_queries"]
                    metric_key = "AP50" if class_aware else "CA-AP50"
                    min_metric_key = "min_scene_AP50" if class_aware else "min_scene_CA-AP50"
                    log.info(
                        "epoch=%d val raw %s=%.4f min-scene %s=%.4f",
                        epoch,
                        metric_key,
                        val_raw[metric_key],
                        min_metric_key,
                        val_raw[min_metric_key],
                    )

                acceptance = eval_config.get("acceptance", {})
                if train_metrics is not None and acceptance:
                    report = build_train_acceptance_report(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        train_set_loss=mean_loss,
                        first_train_set_loss=first_train_set_loss,
                        acceptance=acceptance,
                    )
                    if class_aware:
                        scene_threshold = acceptance.get(
                            "min_train_scene_AP50",
                            acceptance.get("min_per_scene_AP50"),
                        )
                        selection_key = "min_scene_AP50"
                        aggregate_key = "AP50"
                        validation_prefix = "AP50"
                    else:
                        scene_threshold = acceptance.get(
                            "min_train_scene_CA_AP50",
                            acceptance.get("min_per_scene_CA_AP50"),
                        )
                        selection_key = "min_scene_CA-AP50"
                        aggregate_key = "CA-AP50"
                        validation_prefix = "CA-AP50"
                    selection_score = (
                        report[selection_key]
                        if scene_threshold is not None
                        else report[aggregate_key]
                    )
                    is_new_best = float(selection_score) > best_acceptance_score
                    if is_new_best:
                        best_acceptance_score = float(selection_score)
                    if validation_metrics is not None:
                        val_raw = validation_metrics["raw_queries"]
                        report["validation_summary"] = {
                            "aggregate_" + validation_prefix: val_raw[aggregate_key],
                            "min_scene_" + validation_prefix: val_raw[selection_key],
                            "per_scene_" + validation_prefix: {
                                scene_id: scene_metrics[aggregate_key]
                                for scene_id, scene_metrics in val_raw[
                                    "per_scene"
                                ].items()
                            },
                        }
                    accepted_path = output_dir / "checkpoints" / (
                        "accepted_raw_ap50.pt"
                        if class_aware
                        else "accepted_raw_ca_ap50.pt"
                    )
                    report["best_selection_score"] = best_acceptance_score
                    report["checkpoint"] = (
                        str(accepted_path)
                        if report["passed"] and is_new_best
                        else None
                    )
                    (output_dir / "overfit_acceptance.json").write_text(
                        json.dumps(report, indent=2), encoding="utf-8"
                    )
                    if report["passed"] and is_new_best:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "model_contract": model_contract(config),
                                "data_identity": config["_data_identity"],
                                "baseline_config_sha256": baseline_config_hash,
                                "deterministic_epoch_seed": seed + epoch + 1,
                                "validation_completed": True,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "first_train_set_loss": first_train_set_loss,
                                "initialization": init_report,
                                "backbone_state_sha256": model.backbone_state_hash(),
                                "decoder_state_sha256": model.decoder_state_hash(),
                                "acceptance": report,
                            },
                            accepted_path,
                        )
                if epoch_record is None:
                    raise RuntimeError("rank-0 epoch record was not initialized")
                if "train" in epoch_record or "validation" in epoch_record:
                    eval_report = {
                        "epoch": epoch,
                        "train": epoch_record.get("train"),
                        "validation": epoch_record.get("validation"),
                    }
                    (
                        output_dir / f"eval_epoch_{epoch:04d}.json"
                    ).write_text(
                        json.dumps(eval_report, indent=2), encoding="utf-8"
                    )
                    compact_record = {
                        key: value
                        for key, value in epoch_record.items()
                        if key not in {"train", "validation"}
                    }
                    if "train" in epoch_record:
                        compact_record["train"] = _compact_eval_metrics(
                            epoch_record["train"]
                        )
                    if "validation" in epoch_record:
                        compact_record["validation"] = _compact_eval_metrics(
                            epoch_record["validation"]
                        )
                    _append_jsonl(output_dir / "metrics.jsonl", compact_record)
            _wait_for_rank0_eval(distributed, eval_group)

        if train_config.get("save_latest_each_epoch", False) and rank == 0:
            # Save after validation; each next epoch reseeds workers and shuffling.
            latest = output_dir / "checkpoints" / "latest.pt"
            temporary = latest.with_suffix(".tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_contract": model_contract(config),
                "data_identity": config["_data_identity"],
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "first_train_set_loss": first_train_set_loss,
                "initialization": init_report,
                "backbone_state_sha256": model.backbone_state_hash(),
                "decoder_state_sha256": model.decoder_state_hash(),
                "deterministic_epoch_seed": seed + epoch + 1,
                "baseline_config_sha256": baseline_config_hash,
                "validation_completed": True,
            }, temporary)
            temporary.replace(latest)
            (output_dir / "status.json").write_text(json.dumps({
                "epoch": epoch, "complete": epoch == max_epochs,
                "epoch_seconds": epoch_seconds,
            }, indent=2))
        if train_config.get("stop_after_epochs") and epoch >= int(train_config["stop_after_epochs"]):
            break
        if time.monotonic() - run_start_time > float(train_config.get("max_run_hours", 1e9))*3600:
            break

    if v2_probe is not None:
        if v2_expected_total_updates is None:
            raise RuntimeError("V2 probe expected update count was not initialized")
        local_steps = torch.tensor(
            int(observed_optimizer_updates), device=device, dtype=torch.int64
        )
        min_steps = local_steps.clone()
        max_steps = local_steps.clone()
        if distributed:
            dist.all_reduce(min_steps, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_steps, op=dist.ReduceOp.MAX)
        observed_min = int(min_steps.item())
        observed_max = int(max_steps.item())
        if (
            observed_min != observed_max
            or observed_min != v2_expected_total_updates
        ):
            raise RuntimeError(
                "V2 probe optimizer-update count drift across ranks: "
                f"min={observed_min} max={observed_max} "
                f"expected={v2_expected_total_updates}"
            )
        if rank == 0:
            observed_updates = [row["update"] for row in v2_probe_evaluations]
            if observed_updates != v2_expected_updates:
                raise RuntimeError(
                    "V2 probe evaluation result schedule drift: "
                    f"observed={observed_updates} expected={v2_expected_updates}"
                )
            probe_arm = v2_probe["arm"]
            result_payload = {
                "schema_version": PROBE_RESULT_SCHEMA,
                "arm_id": probe_arm["arm_id"],
                "arm_manifest_sha256": probe_arm["manifest_sha256"],
                "arm_config_sha256": probe_arm["config"]["sha256"],
                "common_config_sha256": v2_probe["bundle"]["common_config_sha256"],
                "decoder_tensor_state_sha256": probe_arm["decoder_template"][
                    "tensor_state_sha256"
                ],
                "decoder_file_sha256": probe_arm["decoder_template"][
                    "file_sha256"
                ],
                "global_optimizer_updates": int(observed_optimizer_updates),
                "world_size": int(world_size),
                "ordered_train_split_sha256": probe_arm["matched_contract"][
                    "ordered_train_split_sha256"
                ],
                "ordered_development_split_sha256": probe_arm["matched_contract"][
                    "ordered_development_split_sha256"
                ],
                "backbone_batchnorm_policy": probe_arm["matched_contract"][
                    "batchnorm_policy"
                ],
                "augmentation_profile": probe_arm["matched_contract"][
                    "augmentation_profile"
                ],
                "initialization": {
                    "backbone_source": init_report["backbone_source"],
                    "backbone_state_sha256": init_report["backbone_state_sha256"],
                    "decoder_state_sha256": init_report["decoder_state_sha256"],
                    "official_mask3d_weights_loaded": False,
                },
                "evaluations": v2_probe_evaluations,
            }
            _write_json_no_clobber(v2_probe["result_path"], result_payload)
            validate_probe_result(
                v2_probe["result_path"],
                arm=probe_arm,
                bundle=v2_probe["bundle"],
                development_scene_ids=v2_expected_scene_ids,
            )
        _wait_for_rank0_eval(distributed, eval_group)

    if rank == 0 and backbone_freeze_epochs > 0:
        final_backbone_hash = model.backbone_state_hash()
        (output_dir / "backbone_freeze_schedule_audit.json").write_text(
            json.dumps(
                {
                    "freeze_epochs": backbone_freeze_epochs,
                    "before": frozen_hash_before,
                    "at_unfreeze": backbone_hash_at_unfreeze,
                    "after": final_backbone_hash,
                    "unchanged_during_freeze": (
                        frozen_hash_before == backbone_hash_at_unfreeze
                    ),
                    "changed_after_unfreeze": (
                        backbone_hash_at_unfreeze is not None
                        and final_backbone_hash != backbone_hash_at_unfreeze
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif rank == 0 and model.backbone_frozen:
        (output_dir / "frozen_backbone_hash_audit.json").write_text(
            json.dumps(
                {
                    "before": frozen_hash_before,
                    "after": model.backbone_state_hash(),
                    "identical": frozen_hash_before == model.backbone_state_hash(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    _shutdown_distributed(distributed, eval_group)


if __name__ == "__main__":
    main()
