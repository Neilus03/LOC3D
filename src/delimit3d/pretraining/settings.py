"""Explicit supervision, sampling and augmentation settings."""

from __future__ import annotations

import json

import math


from dataclasses import dataclass


from typing import Any, Mapping


import torch


from delimit3d.losses.litept_hierarchy_supervision import HIERARCHY_COARSE_TO_FINE


@dataclass(frozen=True)
class SamplingConfig:
    proposals_per_forward: int = 4
    positive_pairs_per_proposal: int = 64
    uniform_negatives: int = 256
    spatial_hard_negatives: int = 128
    spatial_candidate_pool: int = 1024
    feature_hard_negatives: int = 128
    feature_candidate_pool: int = 1024


SAMPLING_PROFILES: dict[str, SamplingConfig] = {
    "pf_hnm_512_v1": SamplingConfig(),
    "pf_hnm_512_p16_v1": SamplingConfig(proposals_per_forward=16),
    "pf_hnm_768_v1": SamplingConfig(
        uniform_negatives=256,
        spatial_hard_negatives=256,
        spatial_candidate_pool=1024,
        feature_hard_negatives=256,
        feature_candidate_pool=1024,
    ),
}


FIXED_EVAL_PACK_SCHEMA = "litept_gt3_unsam3_fixed_eval_pack/v1"


FIXED_EVAL_SCENE_SCHEMA = "litept_gt3_unsam3_fixed_eval_scene/v1"


TOKEN_PURE_FIXED_EVAL_SCHEMA = "litept_token_pure_dec0_fixed_pack/v5"


TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY = (
    "rebuilt_after_voxelization_from_saved_deferred_widths"
)


TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION = (
    "runtime_rebuild_from_selected_proposal_csr_after_voxelization"
)


CLEAN_TRAINING_SOURCE_SCHEMA = "structured3d_clean_2d_training_scene_source/v1"


CLEAN_CROP_PACK_SCHEMA = "structured3d_clean_2d_crop_pack/v1"


CLEAN_PROJECTION_CACHE_SCHEMA = "structured3d_clean_2d_projection_cache/v1"


TRAINING_SAMPLING_MODES = (
    "random_with_replacement",
    "coverage_without_replacement",
    "hybrid_2d_random_3d_coverage",
)


INPUT_FEATURE_CHANNELS: dict[str, int] = {
    "rgb3": 3,
    "rgbn6": 6,
}


INPUT_FEATURE_CHOICES = tuple(INPUT_FEATURE_CHANNELS)


COMMON_RGB_COLOR_MEAN = (0.0, 0.0, 0.0)


COMMON_RGB_COLOR_STD = (1.0, 1.0, 1.0)


RGBN6_COLOR_NORMALIZATION = "litept_rgb_0_1_no_mean_std"


MULTISCALE_RECIPE_NAME = "final_dec0_plus_decoder_stage_deep_supervision_v1"


MULTISCALE_LEVEL_NAMES = ("dec3", "dec2", "dec1")


MULTISCALE_LEVEL_INDICES = (0, 1, 2)


MULTISCALE_LEVEL_LOSS_WEIGHTS: dict[str, float] = {
    "dec3": 0.20,
    "dec2": 0.30,
    "dec1": 0.50,
}


MULTISCALE_AUXILIARY_WEIGHT = 0.4


MULTISCALE_AUXILIARY_WARMUP_EPOCHS = 20


HIERARCHY_RECIPE_NAME = "final_dec0_plus_proposal_query_hierarchy_v1"


HIERARCHY_MASK_LOSS_WEIGHT = 0.25


HIERARCHY_VICREG_LOSS_WEIGHT = 0.0


HIERARCHY_WARMUP_FRACTION = 0.05


HIERARCHY_VARIANCE_TARGET = 1.0


HIERARCHY_COVARIANCE_WEIGHT = 0.04


HIERARCHY_VICREG_MAX_TOKENS = 4096


HIERARCHY_MASK_STAGE_NAMES = tuple(
    stage for stage in HIERARCHY_COARSE_TO_FINE if stage != "dec0"
)


TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL = 64


TRAIN_AUGMENTATION_NONE = "none"


TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE = (
    "mask3d_scannet_rgbn6_fullscene_v1"
)


TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K = (
    "mask3d_scannet_rgbn6_sphere50k_v1"
)


TRAIN_AUGMENTATION_CHOICES = (
    TRAIN_AUGMENTATION_NONE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
)


def normalize_hierarchy_mask_stage_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return a complete, validated relative query-mask stage weighting.

    This is intentionally relative to ``hierarchy_mask_loss_weight``.  It
    lets a calibration increase only a demonstrably under-served stage while
    preserving the established global objective, rather than changing every
    routed edge together.  Omitting the option is exactly equivalent to the
    pre-existing all-one weighting.
    """

    supplied = {} if weights is None else dict(weights)
    unknown = sorted(set(supplied) - set(HIERARCHY_MASK_STAGE_NAMES))
    if unknown:
        raise ValueError(
            "Unknown hierarchy-mask stage weights: "
            f"{unknown!r}; expected {list(HIERARCHY_MASK_STAGE_NAMES)!r}"
        )
    normalized: dict[str, float] = {}
    for stage in HIERARCHY_MASK_STAGE_NAMES:
        value = float(supplied.get(stage, 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"hierarchy-mask stage weight for {stage} must be finite and non-negative"
            )
        normalized[stage] = value
    return normalized


def normalize_hierarchy_mask_stage_weight_schedule(
    schedule: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate an optional deterministic early/late stage-weight schedule.

    The schedule is deliberately a single hard transition rather than a new
    learnable controller: it preserves exact reproducibility across resume and
    makes the causal question explicit.  It is useful when a coarse stage
    needs stronger early shaping while decoder-stage auxiliaries benefit from
    a lower late coefficient.
    """

    if schedule is None:
        return None
    supplied = dict(schedule)
    expected = {"transition_fraction", "early", "late"}
    unknown = sorted(set(supplied) - expected)
    missing = sorted(expected - set(supplied))
    if unknown or missing:
        raise ValueError(
            "hierarchy-mask stage-weight schedule must contain exactly "
            f"{sorted(expected)!r}; missing={missing!r}, unknown={unknown!r}"
        )
    fraction = float(supplied["transition_fraction"])
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            "hierarchy-mask stage-weight schedule transition_fraction must lie in (0, 1)"
        )
    early = normalize_hierarchy_mask_stage_weights(supplied["early"])
    late = normalize_hierarchy_mask_stage_weights(supplied["late"])
    return {
        "transition_fraction": fraction,
        "early": early,
        "late": late,
    }


def hierarchy_mask_stage_weights_for_step(
    *,
    static_weights: Mapping[str, float],
    schedule: Mapping[str, Any] | None,
    supervision_step: int | None,
    supervision_total_steps: int | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select static or scheduled stage weights and emit replayable telemetry."""

    normalized_static = normalize_hierarchy_mask_stage_weights(static_weights)
    normalized_schedule = normalize_hierarchy_mask_stage_weight_schedule(schedule)
    if normalized_schedule is None:
        return normalized_static, {
            "enabled": False,
            "phase": "static",
            "progress_fraction": None,
            "transition_fraction": None,
        }
    if supervision_step is None or supervision_total_steps is None:
        return dict(normalized_schedule["early"]), {
            "enabled": True,
            "phase": "early_missing_step_context",
            "progress_fraction": None,
            "transition_fraction": normalized_schedule["transition_fraction"],
        }
    step = int(supervision_step)
    total = int(supervision_total_steps)
    if step < 0 or total <= 0 or step > total:
        raise ValueError(
            "hierarchy-mask stage-weight schedule requires 0 <= supervision_step <= total"
        )
    # The runner numbers optimizer updates from one.  Select the phase from
    # progress *before* the current update so a 0.5 transition in a 256-step
    # calibration gives exactly 128 early and 128 late objective evaluations,
    # rather than silently giving the early phase only 127 updates.
    progress = float(max(step - 1, 0)) / float(total)
    phase = (
        "early"
        if progress < float(normalized_schedule["transition_fraction"])
        else "late"
    )
    return dict(normalized_schedule[phase]), {
        "enabled": True,
        "phase": phase,
        "progress_fraction": progress,
        "transition_fraction": normalized_schedule["transition_fraction"],
    }


def weighted_hierarchy_mask_loss(
    *,
    loss_total: torch.Tensor,
    stage_loss_tensors: Mapping[str, torch.Tensor],
    stage_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply relative stage weights to the already cell-balanced mask loss.

    ``ProposalConditionedHierarchyLoss`` has already performed the important
    within-cell and cross-granularity normalization before exposing its stage
    contributions.  We only rescale those final contributions; we do not
    reweight masks, frames, or granularities here.  The all-one branch returns
    the original scalar verbatim to preserve legacy arithmetic.
    """

    normalized = normalize_hierarchy_mask_stage_weights(stage_weights)
    weighted = {
        stage: float(normalized[stage])
        * stage_loss_tensors.get(stage, loss_total * 0.0)
        for stage in HIERARCHY_MASK_STAGE_NAMES
    }
    if all(weight == 1.0 for weight in normalized.values()):
        return loss_total, weighted
    total = loss_total * 0.0
    for stage in HIERARCHY_MASK_STAGE_NAMES:
        total = total + weighted[stage]
    return total, weighted


TRAIN_AUGMENTATION_CONTRACTS: dict[str, dict[str, Any]] = {
    TRAIN_AUGMENTATION_NONE: {
        "profile": TRAIN_AUGMENTATION_NONE,
        "stochastic": False,
        "point_removal": False,
    },
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: {
        "profile": TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
        "stochastic": True,
        "training_only": True,
        "point_removal": False,
        "point_order_preserved": True,
        "rgb_normalization": RGBN6_COLOR_NORMALIZATION,
        "rgb_color_mean": list(COMMON_RGB_COLOR_MEAN),
        "rgb_color_std": list(COMMON_RGB_COLOR_STD),
        "geometry": {
            "center_then_random_translate": True,
            "reflect_x_probability": 0.5,
            "reflect_y_probability": 0.5,
            "anisotropic_scale": {
                "range_per_axis": [0.9, 1.1],
                "probability": 0.5,
                "normal_transform": "inverse_transpose_then_unit_normalize",
            },
            "rotate_z": {
                "range_degrees": [-180.0, 180.0],
                "probability": 0.5,
            },
            "rotate_y": {
                "range_degrees": [-7.5, 7.5],
                "probability": 0.5,
            },
            "rotate_x": {
                "range_degrees": [-7.5, 7.5],
                "probability": 0.5,
            },
            "rigid_normal_transform": True,
            "elastic_distortion": False,
            "coordinate_jitter": False,
        },
        "color": {
            "brightness_contrast": {
                "brightness_range": [-0.2, 0.2],
                "contrast_range": [-0.2, 0.2],
                "probability": 0.5,
            },
            "rgb_shift": {
                "integer_255_range_per_channel": [-20, 20],
                "probability": 0.5,
            },
            "clip_range": [0.0, 1.0],
        },
        "excluded": [
            "point_dropout",
            "sphere_crop",
            "coordinate_jitter",
            "elastic_distortion",
            "normal_jitter",
            "color_dropout",
        ],
        "seed_key": ["global_seed", "scene_id", "epoch", "input-augmentation"],
        "seed_depends_on_rank": False,
        "seed_depends_on_granularity": False,
    },
}


_sphere50k_augmentation_contract = json.loads(
    json.dumps(
        TRAIN_AUGMENTATION_CONTRACTS[
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE
        ]
    )
)


_sphere50k_augmentation_contract.update(
    {
        "profile": TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
        "required_crop_mode": "sphere_50k",
        "stored_point_contract": (
            "use every stored point from the frozen source crop; at most 50000"
        ),
    }
)


_sphere50k_augmentation_contract["excluded"] = [
    "point_dropout",
    "additional_sphere_crop",
    "point_subsampling",
    "coordinate_jitter",
    "elastic_distortion",
    "normal_jitter",
    "color_dropout",
]


TRAIN_AUGMENTATION_CONTRACTS[
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K
] = _sphere50k_augmentation_contract


def input_channels_for(input_features: str) -> int:
    try:
        return int(INPUT_FEATURE_CHANNELS[str(input_features)])
    except KeyError as exc:
        raise ValueError(
            f"Unknown input feature contract {input_features!r}; "
            f"expected one of {INPUT_FEATURE_CHOICES}"
        ) from exc


def sampling_config_for_profile(profile: str) -> SamplingConfig:
    try:
        return SAMPLING_PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sampling profile {profile!r}; "
            f"expected one of {sorted(SAMPLING_PROFILES)}"
        ) from exc


def augmentation_contract_for_profile(profile: str) -> dict[str, Any]:
    """Return a JSON-safe copy of the frozen train-augmentation contract."""

    try:
        contract = TRAIN_AUGMENTATION_CONTRACTS[str(profile)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown train augmentation profile {profile!r}; "
            f"expected one of {TRAIN_AUGMENTATION_CHOICES}"
        ) from exc
    return json.loads(json.dumps(contract))
