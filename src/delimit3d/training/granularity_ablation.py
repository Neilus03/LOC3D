"""Immutable design and fail-closed readiness checks for the six-arm ablation.

This module deliberately has no Torch import and does not modify any existing
trainer. Dataset feasibility must be established before a CUDA run is prepared.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEGACY_SCHEMA = "delimit3d_granularity_alignment_20ep/v1"
SCHEMA = "delimit3d_granularity_alignment_20ep/v2_fixed9"
PROPOSALS_PER_SCENE = 9
ARMS = ("A02", "A05", "A08", "B", "C", "D")
GRANULARITIES = ("g02", "g05", "g08")
STAGES = ("dec1", "dec2", "dec3", "enc4")
ROUTES = {
    "C": {"g02": ["dec2", "dec3"], "g05": ["dec3", "enc4"],
          "g08": ["dec1", "dec2"]},
    "D": {"g02": ["dec1", "dec2"], "g05": ["dec2", "dec3"],
          "g08": ["dec3", "enc4"]},
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"path": str(path), "sha256": checksum.hexdigest(),
            "size_bytes": path.stat().st_size}


def verify_file(record: dict[str, Any]) -> None:
    if file_record(Path(record["path"])) != record:
        raise ValueError(f"Pinned artifact changed: {record['path']}")


def arm_spec(name: str, *, proposals_per_scene: int = PROPOSALS_PER_SCENE) -> dict[str, Any]:
    if name not in ARMS:
        raise ValueError(f"Unknown arm: {name}")
    if proposals_per_scene not in (9, 15):
        raise ValueError("Only the approved fixed-nine and archived fixed-fifteen designs exist")
    granularities = ["g" + name[1:]] if name.startswith("A") else list(GRANULARITIES)
    return {
        "arm": name, "granularities": granularities,
        "proposal_quota": {g: proposals_per_scene // len(granularities) for g in granularities},
        "contrastive_level": "dec0", "contrastive_weight": 1.0,
        "mask_routes": copy.deepcopy(ROUTES.get(name, {})),
        "mask_weight": 1.0 if name in ROUTES else 0.0,
        "vicreg_weight": 2.0 if name in ROUTES else 0.0,
        "stage_weights": dict.fromkeys(STAGES, 1.0) if name in ROUTES else {},
        "query_projection_policy": "per_stage_dec0" if name in ROUTES else None,
        "routing_fallback": False, "adaptive_loss_controller": False,
        "pcgrad": False, "loss_reduction": "proposal_then_granularity_mean",
    }


def assignment_counts(spec: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for granularity, stages in spec["mask_routes"].items():
        for stage in stages:
            counts[stage] += spec["proposal_quota"][granularity]
    return {stage: counts[stage] for stage in STAGES}


def expected_common(*, proposals_per_scene: int = PROPOSALS_PER_SCENE) -> dict[str, Any]:
    if proposals_per_scene not in (9, 15):
        raise ValueError("Unapproved proposal quota")
    return {
        "seed": 42, "model": "LitePT-S*", "input_features": "rgbn6",
        "initialization": "scratch", "grid_size_m": 0.05, "crop": "sphere_50k",
        "dataset": "Structured3D", "train_scene_count": 2930,
        "ground_truth_segmentation_training_supervision": False,
        "nominal_epochs": 20, "updates": 4883, "warmup_updates": 244,
        "world_size": 2, "scenes_per_rank_per_update": 6,
        "proposals_per_scene": proposals_per_scene, "distinct_proposals_per_scene": True,
        "precision": "fp32", "normalization": "sync_batchnorm",
        "optimizer": "adamw", "backbone_lr": 1e-4, "head_lr": 3e-3,
        "weight_decay": 0.05, "clip_norm": 5.0,
        "scheduler": "linear_warmup_cosine", "final_lr_factor": 0.01,
        "feature_hard_start_update": 245, "positive_pairs_per_proposal": 64,
        "uniform_negatives": 256, "spatial_negatives": 128,
        "feature_negatives": 128, "spatial_candidate_pool": 1024,
        "feature_candidate_pool": 1024,
        "vicreg_variance_target": 1.0, "vicreg_covariance_weight": 0.04,
        "vicreg_max_tokens": 4096,
        "snapshot_updates": [0, 1220, 2441, 4883],
        "encoder_selection": "final_update_4883_raw_bn",
        "common_eligibility_min_retention_per_granularity": 0.90,
        "common_visit_min_admission": 0.95,
    }


def expected_downstream() -> dict[str, Any]:
    return {
        "frozen": {
            "arms": list(ARMS), "primary_level": "dec0",
            "levels": ["dec0", *STAGES], "scannet_scenes": 312,
            "articulate3d_scenes": 42, "part_min_tokens": 10,
            "part_sensitivity_tokens": [5, 20], "bn": "raw_checkpoint",
            "common_query_and_candidate_ids": True,
        },
        "mask3d": {
            "arms": list(ARMS), "mask_only": False, "input_features": "rgbn6",
            "encoder_lr": 1e-5, "decoder_lr": 1e-4, "epochs": 600,
            "screen_epoch": 150, "eval_every_epochs": 50,
            "schedule_horizon_at_initialization": 600,
            "primary": "raw_query_class_agnostic_ap_epoch_600",
            "continue_all_healthy_arms": True,
            "shared_fresh_decoder": True,
        },
        "agile3d": {
            "arms": ["C", "D"], "epochs": 1100, "encoder_lr_multiplier": 0.1,
            "decoder_lr": 1e-4, "modes": ["single_object", "multiobject"],
            "metrics": ["iou@1", "iou@5", "iou@15", "noc@80"],
            "shared_fresh_decoder": True,
        },
        "seed_replicates": 1, "bootstrap": "paired_base_scene_groups",
        "bootstrap_does_not_measure_training_seed_uncertainty": True,
    }


def make_contract(*, train_split: Path, source_root: Path,
                  reference_recipe: Path) -> dict[str, Any]:
    scenes = train_split.read_text().splitlines()
    if len(scenes) != 2930 or len(set(scenes)) != 2930 or any(
        not scene.startswith("scene_") or "/" in scene or ".." in scene for scene in scenes
    ):
        raise ValueError("Expected the unchanged 2,930 unique Structured3D scene IDs")
    return {
        "schema": SCHEMA, "common": expected_common(),
        "arms": {arm: arm_spec(arm) for arm in ARMS},
        "downstream": expected_downstream(),
        "data": {"train_split": file_record(train_split),
                 "source_root": str(source_root.resolve(strict=True)),
                 "reference_recipe": file_record(reference_recipe)},
        "pretraining_order": ["C", "D", "B", "A05", "A02", "A08"],
        "mask3d_pairs": [["C", "D"], ["B", "A05"], ["A02", "A08"]],
        "historical_checkpoint_reuse": False,
    }


def validate_contract(contract: dict[str, Any], *, verify_artifacts: bool = False) -> None:
    schema = contract.get("schema")
    if schema not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Unknown ablation schema")
    quota = 9 if schema == SCHEMA else 15
    for key, expected in (("common", expected_common(proposals_per_scene=quota)),
                          ("arms", {arm: arm_spec(arm, proposals_per_scene=quota) for arm in ARMS}),
                          ("downstream", expected_downstream())):
        if contract.get(key) != expected:
            raise ValueError(f"Unapproved ablation contract difference: {key}")
    if contract.get("historical_checkpoint_reuse") is not False:
        raise ValueError("Historical D is not the balanced 20-epoch D")
    if assignment_counts(contract["arms"]["C"]) != assignment_counts(contract["arms"]["D"]):
        raise ValueError("C/D supervision assignment counts differ")
    if verify_artifacts:
        verify_file(contract["data"]["train_split"])
        verify_file(contract["data"]["reference_recipe"])


def lr_factor(update: int) -> float:
    """One-based optimizer-update LR factor, with the exact final floor."""
    if not 1 <= update <= 4883:
        raise ValueError("Update must be in [1, 4883]")
    if update <= 244:
        return update / 244.0
    phase = (update - 244) / (4883 - 244)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * phase))


def require_training_readiness(contract: dict[str, Any], *, inventory: dict[str, Any],
                               geometry: dict[str, Any], smoke: dict[str, Any]) -> None:
    """No catalog-only report, partial audit or CPU test can authorize CUDA training."""
    validate_contract(contract, verify_artifacts=True)
    if contract["schema"] != SCHEMA:
        raise ValueError("The archived fixed-fifteen design is not approved for training")
    contract_hash = digest(contract)
    for label, report in (("catalog", inventory), ("geometry", geometry), ("GPU smoke", smoke)):
        if report.get("contract_sha256") != contract_hash or report.get("passed") is not True:
            raise ValueError(f"Missing or failed {label} gate for this contract")
    if geometry.get("validation_level") != "full_training_split_geometry_and_two_fold_support":
        raise ValueError("The complete geometry audit is required")
    if smoke.get("validation_level") != "two_gpu_eight_update_all_six_arms":
        raise ValueError("CPU tests are not the six-arm CUDA smoke")
