"""Pretraining engine settings extracted from the executed trainer."""

from __future__ import annotations


import logging


from delimit3d.identity import project_slug


from delimit3d.training.adaptation import MULTISCALE_LEVEL_NAMES


from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
)


log = logging.getLogger(f"{project_slug()}.adaptation")


DEFAULT_EVAL_EPOCHS = (
    0,
    17,
    34,
    85,
    169,
    252,
    336,
    504,
    672,
    756,
    840,
    924,
    1008,
)


V2_SAMPLING_MODE = "coverage_multigranular_v2"


TRAINING_SAMPLING_MODES = (
    "random_with_replacement",
    "coverage_without_replacement",
    V2_SAMPLING_MODE,
)


HIERARCHY_SUPERVISION_MODES = (
    "off",
    "legacy",
    "query_mask",
    "query_mask_vicreg",
)


NORMALIZATION_POLICIES = ("batchnorm", "sync_batchnorm")


BACKBONE_LR_CHOICES = (1e-4, 3e-4, 1e-3)


COORDINATE_NORMALIZATION_POLICIES = (
    "native",
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
)


CUMULATIVE_HEALTH_SCHEMA = "litept_pretraining_cumulative_health/v1"


RANK_CUMULATIVE_HEALTH_SCHEMA = "litept_pretraining_rank_cumulative_health/v1"


TOKEN_CUMULATIVE_HEALTH_SCHEMA = "litept_token_pure_delivery_cumulative/v1"


HIERARCHY_CUMULATIVE_HEALTH_SCHEMA = "litept_hierarchy_delivery_cumulative/v1"


LEGACY_MULTISCALE_CUMULATIVE_SCHEMA = (
    "litept_legacy_multiscale_delivery_cumulative/v1"
)


GRADIENT_PANEL_CUMULATIVE_SCHEMA = "litept_gradient_panel_cumulative/v1"


HIERARCHY_HEALTH_STAGES = ("enc4", "dec3", "dec2", "dec1", "dec0")


HIERARCHY_ROUTE_STAGES = {
    "g02": ("dec1", "dec2"),
    "g05": ("dec2", "dec3"),
    "g08": ("dec3", "enc4"),
}


LEGACY_MULTISCALE_CUMULATIVE_STAGES = tuple(MULTISCALE_LEVEL_NAMES)


LEGACY_MULTISCALE_INTEGER_KEYS = (
    "input_pairs",
    "valid_pairs",
    "kept_pairs",
    "input_proposals",
    "kept_proposals",
    "collapsed_positive_pairs",
    "dropped_proposals_insufficient_pure_positive",
    "dropped_proposals_no_pure_negative",
    "pure_positive_tokens",
    "pure_negative_tokens",
    "mixed_or_unknown_tokens_ignored",
    "accepted_mixed_tokens",
    "zero_negative_pairs",
    "base_negative_collisions",
    "base_negative_duplicates",
    "feature_candidate_pool",
    "feature_hard_negatives",
    "feature_candidate_collisions",
    "feature_candidate_duplicates",
)


LEGACY_MULTISCALE_FLOAT_KEYS = (
    "loss",
    "hardest_negative_ranking_accuracy",
    "cosine_gap",
    "temperature",
    "base_negatives_per_pair",
    "uniform_negatives_per_pair",
    "spatial_hard_negatives_per_pair",
    "stage_weight",
    "effective_loss_weight",
)


LEGACY_MULTISCALE_CUMULATIVE_KEYS = (
    *LEGACY_MULTISCALE_INTEGER_KEYS,
    *LEGACY_MULTISCALE_FLOAT_KEYS,
)


RANK_CUMULATIVE_HEALTH_KEYS = frozenset(
    {
        "schema_version",
        "rank",
        "through_update",
        "optimizer_update_count",
        "source_cells",
        "granularities",
        "expected_microbatches_per_update",
        "clipping",
        "token_pure_dec0",
        "hierarchy_supervision",
        "legacy_multiscale_delivery",
        "gradient_panels",
    }
)


TOKEN_CUMULATIVE_COUNT_KEYS = (
    "input_groups",
    "retained_groups",
    "input_proposals",
    "retained_proposals",
    "input_point_pairs",
    "requested_token_pairs",
    "realized_positive_pairs",
    "zero_denominator_groups",
)


HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS = (
    "mask_valid_fold_denominator",
    "mask_proposals_attempted",
    "mask_proposals_valid",
    "mask_folds_attempted",
    "mask_folds_valid",
)
