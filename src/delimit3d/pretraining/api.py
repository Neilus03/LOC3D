"""Compatibility exports for existing pretraining callers."""

from __future__ import annotations

import json

import math

import random

from dataclasses import dataclass, replace

from pathlib import Path

from tempfile import NamedTemporaryFile

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import torch

import torch.nn as nn

from delimit3d.training.contracts import (
    GRANULARITY_KEYS,
    RAW_CACHE_SCHEMA,
    SOURCE_CELLS,
    SOURCE_CELL_SCHEDULE_FIXED,
    TWO_D_CELLS,
    normalize_source_cells,
    audit_scene_source_manifest,
    lr_factor,
    sha256_file,
    source_cell_for_scene_epoch,
    stable_seed,
)

from delimit3d.data.partfield_contrastive import (
    SAMPLER_ALGORITHM_VERSION,
    SAMPLER_PRESELECTION_POLICY,
    TOKEN_POSITIVE_PAIR_PAYLOAD_SCHEMA,
    ContrastiveTripletBatch,
    DeferredTokenLossPlan,
    concatenate_triplet_batches,
    sample_overlapping_proposal_triplets,
    sample_partition_triplets,
    validate_token_positive_pair_payload,
)

from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    source_cell_quotas,
    sample_multigranular_frame_group_plan,
)

from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    augment_points_mask3d_scannet_rgbn6,
    canonicalize_structured3d_points_normals_to_scannet,
)

from delimit3d.data.single_scene_dataset import build_input_features

from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion

from delimit3d.losses.litept_hierarchy_supervision import (
    HIERARCHY_COARSE_TO_FINE,
    HIERARCHY_QUERY_PROJECTION_POLICIES,
    MaskSupervisionGroup,
    ProposalConditionedHierarchyLoss,
    build_token_pure_dec0_contrastive_batch,
    hierarchy_gradient_telemetry,
    hierarchy_point_maps,
    native_hierarchy_vicreg_loss,
)

from delimit3d.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput


from .settings import (
    CLEAN_CROP_PACK_SCHEMA,
    CLEAN_PROJECTION_CACHE_SCHEMA,
    CLEAN_TRAINING_SOURCE_SCHEMA,
    COMMON_RGB_COLOR_MEAN,
    COMMON_RGB_COLOR_STD,
    FIXED_EVAL_PACK_SCHEMA,
    FIXED_EVAL_SCENE_SCHEMA,
    HIERARCHY_COVARIANCE_WEIGHT,
    HIERARCHY_MASK_LOSS_WEIGHT,
    HIERARCHY_MASK_STAGE_NAMES,
    HIERARCHY_RECIPE_NAME,
    HIERARCHY_VARIANCE_TARGET,
    HIERARCHY_VICREG_LOSS_WEIGHT,
    HIERARCHY_VICREG_MAX_TOKENS,
    HIERARCHY_WARMUP_FRACTION,
    INPUT_FEATURE_CHANNELS,
    INPUT_FEATURE_CHOICES,
    MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    MULTISCALE_AUXILIARY_WEIGHT,
    MULTISCALE_LEVEL_INDICES,
    MULTISCALE_LEVEL_LOSS_WEIGHTS,
    MULTISCALE_LEVEL_NAMES,
    MULTISCALE_RECIPE_NAME,
    RGBN6_COLOR_NORMALIZATION,
    SAMPLING_PROFILES,
    SamplingConfig,
    TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL,
    TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY,
    TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION,
    TOKEN_PURE_FIXED_EVAL_SCHEMA,
    TRAINING_SAMPLING_MODES,
    TRAIN_AUGMENTATION_CHOICES,
    TRAIN_AUGMENTATION_CONTRACTS,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
    TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
    TRAIN_AUGMENTATION_NONE,
    _sphere50k_augmentation_contract,
    augmentation_contract_for_profile,
    hierarchy_mask_stage_weights_for_step,
    input_channels_for,
    normalize_hierarchy_mask_stage_weight_schedule,
    normalize_hierarchy_mask_stage_weights,
    sampling_config_for_profile,
    weighted_hierarchy_mask_loss,
)
from .sources import (
    MultiCropSceneSource,
    RawFrame,
    SceneSource,
    SceneVisitInput,
    _load_clean_multicrop_scene_source,
    _resolve_artifact,
    build_scene_visit_input,
    load_scene_source,
    scene_source_for_visit,
)
from .contrastive import (
    ContrastiveProjectionHead,
    _cell_balanced_contrastive_results,
    _token_pure_dec0_optimization_metrics,
    _zero_contrastive_result,
    build_pure_token_contrastive_batch,
    normalize_hierarchy_frame_groups,
    remap_contrastive_batch_to_level,
    zero_loss_touching_module_parameters,
)
from .model import (
    LitePTContrastiveModel,
)
from .sampling import (
    _coverage_raw_training_batch,
    _generator,
    _holdout_batches,
    _valid_partition_labels,
    _valid_raw_proposal_indices,
    coverage_queue_indices,
    sample_3d_batch,
    sample_raw_batch,
    training_batch,
)
from .fixed_eval import (
    _atomic_json_write,
    _atomic_torch_save,
    _expected_holdout_record_count,
    _fixed_scene_batch_views,
    _token_pure_fixed_holdout_plan,
    _triplet_batch_from_record,
    _triplet_batch_record,
    _validate_token_pure_fixed_record,
    prepare_fixed_holdout_eval_pack,
    validate_fixed_holdout_eval_pack,
)
from .evaluation import (
    evaluate_holdout,
    feature_health,
)
from .optim import (
    optimizer_parameter_groups,
    set_optimizer_lr,
)
