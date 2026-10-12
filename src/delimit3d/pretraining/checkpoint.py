"""Pretraining checkpoint extracted from the executed trainer."""

from __future__ import annotations

import argparse


from pathlib import Path


from typing import Any


import torch


from delimit3d.identity import project_name, project_slug

from delimit3d.training.contracts import EXPERIMENT_ID, sha256_file
from delimit3d.data.partfield_contrastive import negative_policy_name

from delimit3d.training.adaptation import LitePTContrastiveModel, MULTISCALE_LEVEL_LOSS_WEIGHTS, MULTISCALE_LEVEL_NAMES


from .engine_settings import (
    V2_SAMPLING_MODE,
)
from .health import (
    _merge_count_dicts,
)
from .records import (
    _batchnorm_state_report,
    _tensor_mapping_sha256,
)


def validate_negative_policy_resume(previous, current, checkpoint=None):
    """Reject a policy change before loading model or optimizer state."""
    def read(config):
        flag = config.get("allow_shared_proposal_negatives", False)
        policy = negative_policy_name(flag)
        if config.get("negative_policy", policy) != policy:
            raise ValueError("SAM2 negative policy stamp disagrees with its boolean")
        return flag

    before, after = read(previous), read(current)
    if checkpoint is not None and (
        "allow_shared_proposal_negatives" in checkpoint or "negative_policy" in checkpoint
    ) and read(checkpoint) != before:
        raise ValueError("Checkpoint negative policy differs from its resolved configuration")
    if before != after:
        raise ValueError("SAM2 negative policy differs on resume; start a separate run")


def _checkpoint_payload(
    *,
    epoch: int,
    update: int,
    model: LitePTContrastiveModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    resolved: dict[str, Any],
    evaluation: dict[str, Any] | None,
    evaluation_history: dict[int, dict[str, Any]],
    rank_exposure_counts: list[dict[str, int]],
    rank_coverage_states: list[dict[str, Any]] | None = None,
    rank_sampling_totals: list[dict[str, Any]] | None = None,
    rank_rng_states: list[dict[str, Any]] | None = None,
    feature_hard_state: dict[str, Any] | None = None,
    gradient_conflict_state: dict[str, Any] | None = None,
    cumulative_health: dict[str, Any] | None = None,
    checkpoint_kind: str = "resumable_epoch_boundary",
    nominal_epoch: int | None = None,
) -> dict[str, Any]:
    backbone_state = model.backbone.state_dict()
    pointgroup_state = {
        f"backbone.{key.removeprefix('model.')}": value
        for key, value in backbone_state.items()
        if key.startswith("model.")
    }
    if len(pointgroup_state) != len(backbone_state):
        invalid = sorted(
            key for key in backbone_state if not key.startswith("model.")
        )
        raise RuntimeError(
            f"Cannot map wrapper backbone into PointGroup state: {invalid}"
        )
    is_v2 = str(getattr(args, "sampling_mode", "")) == V2_SAMPLING_MODE
    flag = getattr(args, "allow_shared_proposal_negatives", False)
    validate_negative_policy_resume(resolved, {"allow_shared_proposal_negatives": flag})
    auxiliary_prefixes = (
        "multiscale_projectors.",
        "multiscale_criteria.",
        "hierarchy_mask_supervisor.",
    )
    auxiliary_keys = sorted(
        key
        for key in model.state_dict()
        if key.startswith(auxiliary_prefixes)
    )
    if (is_v2 or bool(getattr(args, "multiscale_supervision", False))) and (
        cumulative_health is None
    ):
        raise ValueError(
            "V2/legacy-multiscale checkpoint requires cumulative health telemetry"
        )
    return {
        "schema_version": (
            "litept_rgbn6_multigranular_pretrain_checkpoint/v2"
            if is_v2
            else "litept_gt3_unsam3_pretrain_checkpoint/v1"
        ),
        "project_name": project_name(),
        "project_slug": project_slug(),
        "experiment_id": EXPERIMENT_ID,
        "epoch": int(epoch),
        "nominal_epoch": int(epoch if nominal_epoch is None else nominal_epoch),
        "update": int(update),
        "checkpoint_kind": str(checkpoint_kind),
        "resumable": checkpoint_kind == "resumable_epoch_boundary",
        **({"allow_shared_proposal_negatives": flag, "negative_policy": negative_policy_name(flag)}
           if getattr(args, "source_granularity", None) == "sam2" else {}),
        "seed": int(args.seed),
        "model_state_dict": model.state_dict(),
        "backbone_state_dict": backbone_state,
        "pointgroup_model_state_dict": pointgroup_state,
        "projection_head_state_dict": model.projector.state_dict(),
        "multiscale_projection_head_state_dict": (
            model.multiscale_projectors.state_dict()
        ),
        "multiscale_criterion_state_dict": (
            model.multiscale_criteria.state_dict()
        ),
        "criterion_state_dict": model.criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rank_exposure_counts": rank_exposure_counts,
        "global_exposure_counts": _merge_count_dicts(rank_exposure_counts),
        "rank_coverage_states": rank_coverage_states,
        "rank_sampling_totals": rank_sampling_totals,
        "rank_rng_states": rank_rng_states,
        "feature_hard_state": feature_hard_state,
        "gradient_conflict_state": gradient_conflict_state,
        "cumulative_health": cumulative_health,
        "batchnorm_raw": _batchnorm_state_report(model),
        "batchnorm_recalibrated": None,
        "auxiliary_state_keys": auxiliary_keys,
        "transfer_export_contract": {
            "allowed_prefixes": ["backbone."],
            "rejected_prefixes": [
                "projector.",
                "criterion.",
                *auxiliary_prefixes,
            ],
            "chosen_batchnorm_buffers_only": True,
        },
        "evaluation": evaluation,
        "evaluation_history": {
            str(key): value
            for key, value in sorted(evaluation_history.items())
        },
        "resolved_config": resolved,
        "public_checkpoint_used": bool(
            resolved["initialization"]["public_checkpoint_used"]
        ),
    }


def _strict_initial_load(
    *,
    model: LitePTContrastiveModel,
    path: Path,
    input_features: str,
    multiscale_supervision: bool = False,
    v2_auxiliary_initialization: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "litept_gt3_unsam3_initial/v1":
        raise ValueError(f"Unexpected initialization schema: {path}")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Initialization experiment drift: {path}")
    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get(
        "voxel_reduce"
    ) != "representative":
        raise ValueError(
            "Initialization does not declare representative-point voxelization; "
            "regenerate it rather than silently mixing preprocessing contracts"
        )
    if float(architecture.get("grid_size", 0.02)) != float(model.backbone.grid_size):
        raise ValueError("Initialization grid size differs from this run")
    if architecture.get("litept_variant", "litept_s_star") != model.backbone.litept_variant:
        raise ValueError("Initialization LitePT variant differs from this run")
    declared_input_features = architecture.get("input_features")
    if declared_input_features is None:
        # The original v1 RGBN6 initialization predates the explicit feature
        # field. Preserve its compatibility while rejecting it for RGB3.
        declared_input_features = (
            "rgbn6" if architecture.get("in_channels") == 6 else None
        )
    if declared_input_features != str(input_features):
        raise ValueError(
            "Initialization input-feature contract differs from this run: "
            f"checkpoint={declared_input_features!r} run={input_features!r}"
        )
    declared_representative_sampling = architecture.get(
        "representative_sampling", "first"
    )
    if declared_representative_sampling != model.backbone.representative_sampling:
        raise ValueError(
            "Initialization representative-point sampling differs from this run: "
            f"checkpoint={declared_representative_sampling!r} "
            f"run={model.backbone.representative_sampling!r}"
        )
    declared_multiscale = bool(
        architecture.get("multiscale_supervision", False)
    )
    if v2_auxiliary_initialization and declared_multiscale:
        raise ValueError(
            "V2 arms require the common dec0-only initialization; disposable "
            "auxiliary heads are initialized deterministically by the V2 model"
        )
    if not v2_auxiliary_initialization and declared_multiscale != bool(
        multiscale_supervision
    ):
        raise ValueError(
            "Initialization multi-scale recipe differs from this run: "
            f"checkpoint={declared_multiscale!r} run={bool(multiscale_supervision)!r}"
        )
    if declared_multiscale:
        expected_multiscale = {
            "pretraining_recipe": model.recipe_name,
            "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
            "multiscale_level_loss_weights": dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            ),
            "multiscale_loss_weight": float(model.multiscale_loss_weight),
            "multiscale_warmup_epochs": int(model.multiscale_warmup_epochs),
            "multiscale_token_policy": "strict_pure_token_resampling_v1",
            "multiscale_temperature_policy": (
                "separate_learnable_temperature_per_auxiliary_level"
            ),
            "multiscale_feature_hard_policy": (
                "separate_candidate_selection_per_positive_endpoint"
            ),
        }
        multiscale_mismatches = {
            key: {
                "checkpoint": architecture.get(key),
                "run": expected,
            }
            for key, expected in expected_multiscale.items()
            if architecture.get(key) != expected
        }
        if multiscale_mismatches:
            raise ValueError(
                "Initialization multi-scale contract drift: "
                f"{multiscale_mismatches}"
            )
    expected_channels = 3 if input_features == "rgb3" else 6
    if int(architecture.get("in_channels", -1)) != expected_channels:
        raise ValueError(
            "Initialization input-channel count differs from this run: "
            f"checkpoint={architecture.get('in_channels')!r} "
            f"run={expected_channels}"
        )
    public_checkpoint_used = payload.get("public_checkpoint_used")
    if not isinstance(public_checkpoint_used, bool):
        raise ValueError(f"Initialization lacks an explicit public-checkpoint flag: {path}")
    initialization_kind = payload.get(
        "initialization_kind",
        "scratch" if not public_checkpoint_used else None,
    )
    if initialization_kind not in ("scratch", "public_litept_backbone"):
        raise ValueError(
            f"Unsupported initialization kind {initialization_kind!r}: {path}"
        )
    if public_checkpoint_used != (initialization_kind == "public_litept_backbone"):
        raise ValueError(f"Initialization kind/public-checkpoint flag drift: {path}")
    public_checkpoint = payload.get("public_checkpoint")
    if public_checkpoint_used:
        if not isinstance(public_checkpoint, dict):
            raise ValueError(f"Public initialization lacks checkpoint provenance: {path}")
        public_path = Path(public_checkpoint["path"]).resolve(strict=True)
        if sha256_file(public_path) != public_checkpoint["sha256"]:
            raise ValueError(f"Public initialization checkpoint hash drift: {public_path}")
        if int(public_checkpoint.get("loaded_backbone_keys", -1)) != 302:
            raise ValueError("Public initialization did not load exactly 302 backbone keys")
    checkpoint_state = payload.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Initialization lacks model_state_dict: {path}")
    auxiliary_prefixes = (
        "multiscale_projectors.",
        "multiscale_criteria.",
        "hierarchy_mask_supervisor.",
    )
    if v2_auxiliary_initialization:
        current_state = model.state_dict()
        expected_base_keys = {
            key
            for key in current_state
            if key.startswith(("backbone.", "projector.", "criterion."))
        }
        checkpoint_keys = set(checkpoint_state)
        if checkpoint_keys != expected_base_keys:
            raise ValueError(
                "V2 common initialization is not an exact backbone/primary-head "
                "state: "
                f"missing={sorted(expected_base_keys - checkpoint_keys)} "
                f"extra={sorted(checkpoint_keys - expected_base_keys)}"
            )
        for key in expected_base_keys:
            if (
                checkpoint_state[key].shape != current_state[key].shape
                or checkpoint_state[key].dtype != current_state[key].dtype
            ):
                raise ValueError(f"V2 initialization tensor contract drift: {key}")
        auxiliary_keys = sorted(
            key for key in current_state if key.startswith(auxiliary_prefixes)
        )
        merged_state = dict(current_state)
        merged_state.update(checkpoint_state)
        model.load_state_dict(merged_state, strict=True)
        base_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=sorted(expected_base_keys)
        )
        auxiliary_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=auxiliary_keys
        )
    else:
        model.load_state_dict(checkpoint_state, strict=True)
        expected_base_keys = set(checkpoint_state)
        auxiliary_keys = sorted(
            key for key in model.state_dict() if key.startswith(auxiliary_prefixes)
        )
        base_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=sorted(expected_base_keys)
        )
        auxiliary_state_sha256 = _tensor_mapping_sha256(
            model.state_dict(), keys=auxiliary_keys
        )
    report = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "seed": int(payload["seed"]),
        "strict": True,
        "strict_scope": (
            "backbone_primary_exact_auxiliary_deterministic"
            if v2_auxiliary_initialization
            else "complete_model_exact"
        ),
        "loaded_base_key_count": len(expected_base_keys),
        "loaded_base_state_sha256": base_state_sha256,
        "auxiliary_key_count": len(auxiliary_keys),
        "auxiliary_keys": auxiliary_keys,
        "auxiliary_initial_state_sha256": auxiliary_state_sha256,
        "input_features": str(input_features),
        "in_channels": expected_channels,
        "representative_sampling": model.backbone.representative_sampling,
        "initialization_kind": initialization_kind,
        "public_checkpoint_used": public_checkpoint_used,
    }
    if public_checkpoint_used:
        report["public_checkpoint"] = public_checkpoint
        report["source_random_initialization"] = payload[
            "source_random_initialization"
        ]
    return report
