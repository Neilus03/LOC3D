"""Approved fixed-nine, common-subset, one-GPU-per-arm revision.

The archived two-GPU contracts remain unchanged. Accumulation preserves twelve
scenes per optimizer update; local BatchNorm is explicitly a hardware revision,
not claimed equivalent to the earlier two-rank SyncBatchNorm design.
"""
from __future__ import annotations

import copy
from pathlib import Path

from delimit3d.training.granularity_ablation import (
    ARMS, arm_spec, digest, expected_common, expected_downstream, file_record,
    validate_contract as validate_parent,
)

SCHEMA = "delimit3d_granularity_alignment_20ep/v3_pf69_single_gpu"
PAIRS = [["C", "D"], ["B", "A05"], ["A02", "A08"]]


def common_settings():
    return {**expected_common(), "world_size": 1, "scenes_per_rank_per_update": 12,
            "normalization": "batchnorm", "normalization_microbatch_scenes": 1,
            "global_scenes_per_update": 12, "common_visit_min_admission": None,
            "admission_policy": "shared_eligible_subset_and_common_visit_filter",
            "initial_eligibility_epoch": 0, "sampler_version": "fixed9_pf69/v1"}


def make_contract(parent_path: Path):
    parent = __import__("json").loads(parent_path.read_text())
    validate_parent(parent, verify_artifacts=True)
    return {"schema": SCHEMA, "parent": file_record(parent_path),
            "common": common_settings(), "arms": {arm: arm_spec(arm) for arm in ARMS},
            "data": copy.deepcopy(parent["data"]), "downstream": expected_downstream(),
            "execution": {"host": "pf-pc69", "gpu_model": "NVIDIA GeForce RTX 4090",
                          "gpus_per_run": 1, "parallel_runs": 2, "pairs": PAIRS,
                          "pair_barrier": "both_arms_complete_before_next_pair",
                          "failure_policy": "stop_queue_without_cancelling_peer",
                          "wait_for_idle_gpus": True},
            "authorization": "User approved fixed nine on a common eligible subset and three sequential pairs on pf-pc69, one arm per 4090.",
            "historical_checkpoint_reuse": False}


def validate_contract(contract):
    for key, value in (("schema", SCHEMA), ("common", common_settings()),
                       ("arms", {arm: arm_spec(arm) for arm in ARMS}),
                       ("downstream", expected_downstream())):
        if contract.get(key) != value:
            raise ValueError(f"Unapproved pf69 contract difference: {key}")
    execution = contract["execution"]
    if (execution["host"] != "pf-pc69" or execution["pairs"] != PAIRS
            or execution["gpus_per_run"] != 1 or execution["parallel_runs"] != 2
            or contract.get("historical_checkpoint_reuse") is not False):
        raise ValueError("One-GPU paired execution contract drift")


def verify_runtime_split(contract, path: Path):
    record = file_record(path)
    if record["sha256"] != contract["data"]["train_split"]["sha256"]:
        raise ValueError("Runtime training split differs from the pinned original")
    scenes = path.read_text().splitlines()
    if len(scenes) != 2930 or len(set(scenes)) != 2930:
        raise ValueError("Expected the original 2930 unique scenes")
    return scenes


def require_geometry(contract, report):
    validate_contract(contract)
    if (report.get("contract_sha256") != digest(contract) or report.get("passed") is not True
            or report.get("complete") is not True
            or report.get("validation_level") != "full_training_split_geometry_and_two_fold_support"
            or not report.get("admitted_scene_ids")):
        raise ValueError("Complete common-subset geometry gate is required")
    if any(value < .90 for value in report["retention_per_granularity"].values()):
        raise ValueError("The unchanged 90% proposal-retention gate failed")
