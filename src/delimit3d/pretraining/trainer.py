"""Pretraining trainer extracted from the executed trainer."""

from __future__ import annotations

from delimit3d.data.sam2_source import fragment_policy as sam2_fragment_policy


import datetime


import json

import logging

import math

import os

import random

import shutil

import time

from concurrent.futures import ThreadPoolExecutor

from contextlib import nullcontext

from pathlib import Path


from typing import Any, Mapping, Sequence

import numpy as np

import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel

from delimit3d.identity import project_name, project_slug

from delimit3d.training.contracts import EXPERIMENT_ID, LOGICAL_BATCH_EVAL_EPOCHS, SCENES_PER_RANK, SOURCE_CELLS, SOURCE_CELL_SCHEDULE_FIXED, SOURCE_FAMILY_2D, WORLD_SIZE, extension_lr_factor, exposure_counts, lr_factor, load_scene_ids, normalize_initialization_report, sha256_file, source_cells_for_family, split_digest, stable_seed

from delimit3d.training.adaptation import LitePTContrastiveModel, MULTISCALE_AUXILIARY_WEIGHT, MULTISCALE_AUXILIARY_WARMUP_EPOCHS, MULTISCALE_LEVEL_LOSS_WEIGHTS, MULTISCALE_LEVEL_NAMES, RGBN6_COLOR_NORMALIZATION, COMMON_RGB_COLOR_MEAN, COMMON_RGB_COLOR_STD, TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE, TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K, augmentation_contract_for_profile, build_scene_visit_input, evaluate_holdout, hierarchy_gradient_telemetry, load_scene_source, normalize_hierarchy_mask_stage_weight_schedule, normalize_hierarchy_mask_stage_weights, optimizer_parameter_groups, prepare_fixed_holdout_eval_pack, sampling_config_for_profile, scene_source_for_visit, set_optimizer_lr, training_batch

from delimit3d.data.contrastive_sampler_v2 import DEFAULT_2D_CELLS as V2_DEFAULT_2D_CELLS, NoRetainedFrameGroupsError, sample_multigranular_frame_group_plan
from delimit3d.data.partfield_contrastive import negative_policy_name

from delimit3d.data.point_augmentations import (
    STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
)

from delimit3d.training.sam2_sampling import (
    COVERAGE_POLICY as SAM2_COVERAGE_POLICY,
    sampling_visit_for_scene,
    validate_saved_coverage_states,
)

from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule

from delimit3d.training.sam2_rank_balance import (
    STRICT as STRICT_RANK_BALANCE, ROTATING as ROTATING_RANK_BALANCE,
    partition_scene_ids, epoch_scene_order, balance_contract, padding_ranks,
)


from .arguments import (
    _comma_separated_nonnegative_ints,
    _resolve_seed_contract,
    parse_args,
)
from .checkpoint import (
    _checkpoint_payload,
    _strict_initial_load,
    validate_negative_policy_resume,
)
from .dataset import (
    _audit_lazy_sources,
    _load_dataset_source_audit,
    _load_logical_scene_group,
    _load_smoke_manifest_paths,
    _preload_sources,
    _v2_frame_catalogs,
)
from .engine_settings import (
    BACKBONE_LR_CHOICES,
    COORDINATE_NORMALIZATION_POLICIES,
    CUMULATIVE_HEALTH_SCHEMA,
    DEFAULT_EVAL_EPOCHS,
    GRADIENT_PANEL_CUMULATIVE_SCHEMA,
    HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
    HIERARCHY_HEALTH_STAGES,
    HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
    HIERARCHY_ROUTE_STAGES,
    HIERARCHY_SUPERVISION_MODES,
    LEGACY_MULTISCALE_CUMULATIVE_KEYS,
    LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
    LEGACY_MULTISCALE_CUMULATIVE_STAGES,
    LEGACY_MULTISCALE_FLOAT_KEYS,
    LEGACY_MULTISCALE_INTEGER_KEYS,
    NORMALIZATION_POLICIES,
    RANK_CUMULATIVE_HEALTH_KEYS,
    RANK_CUMULATIVE_HEALTH_SCHEMA,
    TOKEN_CUMULATIVE_COUNT_KEYS,
    TOKEN_CUMULATIVE_HEALTH_SCHEMA,
    TRAINING_SAMPLING_MODES,
    V2_SAMPLING_MODE,
    log,
)
from .gradients import (
    _add_flat_objective_gradients,
    _add_manual_gradients,
    _clip_gradient_diagnostics,
    _flat_objective_gradients,
    _global_auxiliary_budget_passed,
    _global_shared_gradient_report,
    _gradient_norm,
    _install_global_flat_total_gradient,
    _install_global_pcgrad,
    _manual_objective_gradients,
    _synchronize_accumulated_gradients,
    _telemetry_auxiliary_budget,
    _telemetry_objective_cosine,
)
from .health import (
    _accumulate_rank_cumulative_health_update,
    _append_rank_gradient_panel,
    _canonical_panel_schedule,
    _canonical_source_metadata,
    _canonical_string_sequence,
    _gradient_objective_stage_counts,
    _legacy_multiscale_stage_metrics,
    _merge_count_dicts,
    _merge_rank_cumulative_health,
    _new_rank_cumulative_health,
    _restore_rank_cumulative_health,
    _strict_finite_real,
    _strict_nonnegative_counter,
    _strict_nonnegative_real,
)
from .records import (
    _append_jsonl,
    _atomic_json,
    _atomic_jsonl_noclobber,
    _atomic_torch_save,
    _batchnorm_state_report,
    _capture_rng_state,
    _content_addressed_report,
    _json_copy,
    _restore_rng_state,
    _semantic_config_sha256,
    _tensor_mapping_sha256,
)
from .schedule import (
    _final_evaluation_endpoint,
    _scheduled_gradient_panel_updates,
    _update_for_completed_epoch,
    _updates_for_scene_visit_duration,
)
from .validation import (
    _mean_numeric_tree,
    _native_fixed_pack_cosine_gap,
    _native_token_pure_fixed_pack_cosine_gap,
    _summarize_hierarchy_records,
    _summarize_token_pure_records,
    _summarize_v2_sampling,
)


def _run(args=None) -> None:
    if args is None:
        args = parse_args()
    from delimit3d.training.sampling_ablation import NegativeBudget, negative_budget, ablation_contract, validate_ablation_args
    validate_ablation_args(args)
    sampling_ablation = ablation_contract(args)
    train_allow_shared = bool(args.allow_shared_proposal_negatives) and args.ablation_variant != "overlap_veto"
    # Keep the training/sampling seed independent from the immutable artifacts
    # when a controlled seed repeat is requested.  Legacy invocations retain
    # the previous behavior because both defaults fall back to --seed.
    (
        args.initialization_seed,
        args.fixed_eval_seed,
    ) = _resolve_seed_contract(args)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = int(
        args.expected_world_size
        if args.expected_world_size is not None
        else WORLD_SIZE
    )
    if expected_world_size <= 0:
        raise ValueError("--expected-world-size must be positive")
    if world_size != expected_world_size:
        raise RuntimeError(
            f"This experiment requires exactly {expected_world_size} ranks, "
            f"got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        # SAM2 prepares/evaluates 326 full-scene validation visits on rank zero;
        # rank one must not time out while that bounded work is progressing.
        timeout=datetime.timedelta(minutes=120 if args.source_granularity == "sam2" else 30),
    )
    local_device_report = {
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
    }
    device_reports: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(device_reports, local_device_report)
    source_loader_pool: ThreadPoolExecutor | None = None
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        if (args.resume is None) != (args.resume_sha256 is None):
            raise ValueError("--resume and --resume-sha256 are required together")
        if args.resume_into_fresh_attempt and args.resume is None:
            raise ValueError("--resume-into-fresh-attempt requires --resume")
        if (args.execution_provenance is None) != (
            args.execution_provenance_sha256 is None
        ):
            raise ValueError(
                "--execution-provenance and its SHA-256 are required together"
            )
        resume_sha256: str | None = None
        if args.resume is not None:
            resume_path = args.resume.resolve(strict=True)
            resume_sha256 = sha256_file(resume_path)
            if resume_sha256 != args.resume_sha256:
                raise ValueError(
                    f"Resume checkpoint hash drift: {resume_sha256} != "
                    f"{args.resume_sha256}"
                )
            gathered_resume_hashes: list[str | None] = [None] * world_size
            dist.all_gather_object(gathered_resume_hashes, resume_sha256)
            if set(gathered_resume_hashes) != {resume_sha256}:
                raise RuntimeError(
                    f"Ranks disagree on resume checkpoint identity: "
                    f"{gathered_resume_hashes}"
                )
        execution_provenance_report: dict[str, Any] | None = None
        if args.execution_provenance is not None:
            provenance_path = args.execution_provenance.resolve(strict=True)
            provenance_sha256 = sha256_file(provenance_path)
            if provenance_sha256 != args.execution_provenance_sha256:
                raise ValueError(
                    "Execution-provenance hash differs from the Slurm-spooled "
                    "trust root"
                )
            gathered_provenance_hashes: list[str | None] = [None] * world_size
            dist.all_gather_object(
                gathered_provenance_hashes, provenance_sha256
            )
            if set(gathered_provenance_hashes) != {provenance_sha256}:
                raise RuntimeError(
                    "Ranks disagree on execution-provenance identity"
                )
            provenance_payload = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            if (
                provenance_payload.get("schema_version")
                not in {
                    "structured3d_rgbn6_fullscene_execution_provenance/v1",
                    "structured3d_rgbn6_sphere50k_execution_provenance/v1",
                    "structured3d_rgbn6_pretraining_v2_execution_provenance/v1",
                    "structured3d_sam2_bstar_execution_provenance/v1",
                    "structured3d_sam2_sstar_euler_execution_provenance/v1",
                    "delimit3d_pretraining_execution/v1",
                }
                or provenance_payload.get("passed") is not True
            ):
                raise ValueError("Execution-provenance manifest is not passed")
            if provenance_payload["schema_version"] == "delimit3d_pretraining_execution/v1":
                from .experiment_record import validate_execution_record
                validate_execution_record(provenance_payload, args)
            execution_provenance_report = {
                "path": str(provenance_path),
                "sha256": provenance_sha256,
                "schema_version": provenance_payload["schema_version"],
                "trust_root": provenance_payload.get("trust_root"),
                "passed": True,
            }
        source_family = str(args.source_family)
        source_cells = source_cells_for_family(
            source_family, granularity=args.source_granularity
        )
        single_granularity = args.source_granularity is not None
        if args.source_granularity == "sam2" and args.source_audit is not None:
            sam2_audit = json.loads(args.source_audit.resolve(strict=True).read_text())
            if not (sam2_audit.get("schema_version") == "structured3d_sam2_fullscene_source_audit/v1"
                    or (sam2_audit.get("schema_version") == "delimit3d_2d_source_audit/v1"
                        and sam2_audit.get("source_cells") == ["2d/sam2"])):
                raise ValueError("SAM2 production must use the sealed SAM2 source audit, never an UnSAM audit")
        sampling_mode = str(args.sampling_mode)
        hierarchy_mode = str(args.hierarchy_supervision)
        hierarchy_enabled = hierarchy_mode in {"query_mask", "query_mask_vicreg"}
        if hierarchy_mode == "legacy" and not args.multiscale_supervision:
            raise ValueError(
                "--hierarchy-supervision legacy requires --multiscale-supervision"
            )
        if hierarchy_enabled and args.multiscale_supervision:
            raise ValueError(
                "Proposal-query hierarchy and legacy multiscale supervision are "
                "mutually exclusive"
            )
        if hierarchy_mode == "query_mask" and float(
            args.hierarchy_vicreg_loss_weight
        ) != 0.0:
            raise ValueError("query_mask requires zero VICReg loss weight")
        if hierarchy_mode == "query_mask_vicreg" and float(
            args.hierarchy_vicreg_loss_weight
        ) <= 0.0:
            raise ValueError("query_mask_vicreg requires a positive VICReg loss weight")
        try:
            raw_hierarchy_mask_stage_weights = json.loads(
                str(args.hierarchy_mask_stage_weights)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--hierarchy-mask-stage-weights must be a JSON object"
            ) from exc
        if not isinstance(raw_hierarchy_mask_stage_weights, dict):
            raise ValueError(
                "--hierarchy-mask-stage-weights must decode to an object"
            )
        hierarchy_mask_stage_weights = normalize_hierarchy_mask_stage_weights(
            raw_hierarchy_mask_stage_weights
        )
        try:
            raw_hierarchy_mask_stage_weight_schedule = json.loads(
                str(args.hierarchy_mask_stage_weight_schedule)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--hierarchy-mask-stage-weight-schedule must be JSON"
            ) from exc
        if raw_hierarchy_mask_stage_weight_schedule is not None and not isinstance(
            raw_hierarchy_mask_stage_weight_schedule,
            dict,
        ):
            raise ValueError(
                "--hierarchy-mask-stage-weight-schedule must decode to null or an object"
            )
        hierarchy_mask_stage_weight_schedule = (
            normalize_hierarchy_mask_stage_weight_schedule(
                raw_hierarchy_mask_stage_weight_schedule
            )
        )
        if any(
            value < 0.0
            for value in (
                float(args.hierarchy_mask_loss_weight),
                float(args.hierarchy_vicreg_loss_weight),
                float(args.hierarchy_covariance_weight),
            )
        ):
            raise ValueError("Hierarchy loss weights must be non-negative")
        if not 0.0 <= float(args.hierarchy_warmup_fraction) <= 1.0:
            raise ValueError("--hierarchy-warmup-fraction must lie in [0,1]")
        if not 0.0 <= float(args.feature_hard_warmup_fraction) <= 1.0:
            raise ValueError("--feature-hard-warmup-fraction must lie in [0,1]")
        if int(args.feature_hard_required_panels) <= 0:
            raise ValueError("--feature-hard-required-panels must be positive")
        if int(args.bn_recalibration_scenes) < 0:
            raise ValueError("--bn-recalibration-scenes must be non-negative")
        if sampling_mode == V2_SAMPLING_MODE:
            v2_errors: list[str] = []
            if args.dataset != "structured3d":
                v2_errors.append("dataset must be structured3d")
            if args.input_features != "rgbn6":
                v2_errors.append("input features must be rgbn6")
            if (
                args.coordinate_normalization_policy
                != STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
            ):
                v2_errors.append(
                    "Structured3D stored Y-up points/normals must be canonicalized "
                    "to the ScanNet Z-up model frame"
                )
            if source_family != SOURCE_FAMILY_2D:
                v2_errors.append("source family must be 2d")
            if not single_granularity and tuple(source_cells) != tuple(V2_DEFAULT_2D_CELLS):
                v2_errors.append("source cells must be g02/g05/g08")
            if args.sampling_profile != "pf_hnm_512_p16_v1":
                v2_errors.append("training sampling profile must be p16")
            euler_sstar_sam2 = (
                args.source_granularity == "sam2" and args.litept_variant == "litept_s_star"
                and expected_world_size in (3, 4) and args.rank_balance_policy == ROTATING_RANK_BALANCE
            )
            required_scenes_per_rank = 12 // expected_world_size if euler_sstar_sam2 else 6
            required_world_size = expected_world_size if euler_sstar_sam2 else 2
            if int(args.scenes_per_rank_per_update) != required_scenes_per_rank:
                v2_errors.append(f"{required_scenes_per_rank} scenes per rank/update are required")
            if expected_world_size != required_world_size:
                v2_errors.append(f"exactly {required_world_size} homogeneous GPU ranks are required")
            if args.representative_sampling != "first":
                v2_errors.append("representative sampling must be first")
            expected_v2_crop = "full_scene" if args.source_granularity == "sam2" else "sphere_50k"
            if args.required_crop_mode != expected_v2_crop:
                v2_errors.append(f"crop mode must be {expected_v2_crop}")
            if hierarchy_mode == "off" and not single_granularity:
                v2_errors.append("an explicit legacy/query hierarchy arm is required")
            if int(args.bn_recalibration_scenes) <= 0:
                v2_errors.append("a fixed BN recalibration panel is required")
            if args.fixed_eval_pack is None:
                v2_errors.append("an immutable fixed evaluation pack is required")
            typed_device_reports = [
                report for report in device_reports if report is not None
            ]
            if len(typed_device_reports) != world_size or len(
                {report["name"] for report in typed_device_reports}
            ) != 1:
                v2_errors.append(
                    f"GPU ranks must be homogeneous, observed={typed_device_reports}"
                )
            if v2_errors:
                raise ValueError("Invalid V2 RGBN6 contract: " + "; ".join(v2_errors))
        train_augmentation_profile = str(args.train_augmentation_profile)
        color_mean = (
            tuple(float(value) for value in args.color_mean)
            if args.color_mean is not None
            else None
        )
        color_std = (
            tuple(float(value) for value in args.color_std)
            if args.color_std is not None
            else None
        )
        if (color_mean is None) != (color_std is None):
            raise ValueError("--color-mean and --color-std must be supplied together")
        if train_augmentation_profile in {
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
        }:
            if color_mean is None:
                color_mean = tuple(COMMON_RGB_COLOR_MEAN)
                color_std = tuple(COMMON_RGB_COLOR_STD)
            if color_mean != tuple(COMMON_RGB_COLOR_MEAN) or color_std != tuple(
                COMMON_RGB_COLOR_STD
            ):
                raise ValueError(
                    "Normal-aware RGBN6 augmentation requires the fixed LitePT "
                    "RGB /255 transfer contract"
                )
        required_crop_mode = str(args.required_crop_mode)
        source_loading_mode = str(args.source_loading_mode)
        source_loader_workers = int(args.source_loader_workers)
        if source_loader_workers <= 0:
            raise ValueError("--source-loader-workers must be positive")
        augmentation_crop_modes = {
            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE: "full_scene",
            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K: "sphere_50k",
        }
        if train_augmentation_profile in augmentation_crop_modes:
            if args.dataset != "structured3d":
                raise ValueError(
                    f"{train_augmentation_profile} is Structured3D-only"
                )
            if args.input_features != "rgbn6":
                raise ValueError(
                    f"{train_augmentation_profile} requires --input-features rgbn6"
                )
            augmentation_crop_mode = augmentation_crop_modes[
                train_augmentation_profile
            ]
            if required_crop_mode != augmentation_crop_mode:
                raise ValueError(
                    f"{train_augmentation_profile} requires "
                    f"--required-crop-mode {augmentation_crop_mode}"
                )
        scenes_per_rank_per_update = int(args.scenes_per_rank_per_update)
        if scenes_per_rank_per_update <= 0:
            raise ValueError("--scenes-per-rank-per-update must be positive")
        if source_family == SOURCE_FAMILY_2D and args.three_d_mode != "gt3":
            raise ValueError(
                "--three-d-mode is not used with --source-family 2d; omit it "
                "or leave the default"
            )
        if (
            args.smoke_manifest is not None
            and args.smoke_manifest_list is not None
        ):
            raise ValueError(
                "--smoke-manifest and --smoke-manifest-list are mutually exclusive"
            )
        smoke_manifest_paths = (
            _load_smoke_manifest_paths(args.smoke_manifest_list)
            if args.smoke_manifest_list is not None
            else [args.smoke_manifest.resolve(strict=True)]
            if args.smoke_manifest is not None
            else []
        )
        smoke = bool(smoke_manifest_paths)
        if (
            (args.multiscale_supervision or hierarchy_enabled or sampling_mode == V2_SAMPLING_MODE)
            and not smoke
            and args.execution_provenance is None
        ):
            raise ValueError(
                "Deep-supervision/V2 production requires frozen execution provenance"
            )
        if smoke and source_loading_mode != "preload":
            raise ValueError("Smoke runs require --source-loading-mode preload")
        if smoke:
            epochs = int(args.smoke_epochs)
            scene_visits_per_epoch_per_rank = len(smoke_manifest_paths)
            scenes_per_rank = scene_visits_per_epoch_per_rank
            local_scene_ids = [
                f"smoke-rank-{rank}-scene-{index}"
                for index in range(scene_visits_per_epoch_per_rank)
            ]
            holdout_scene_ids = [
                f"smoke-holdout-{index}"
                for index in range(scene_visits_per_epoch_per_rank)
            ]
            eval_epochs = {0, epochs}
            if epochs == 6:
                eval_epochs.add(3)
            source_audit_report = None
            expected_manifest_hashes = None
        else:
            train_scene_ids = load_scene_ids(args.train_split)
            holdout_scene_ids = load_scene_ids(args.holdout_split)
            if not train_scene_ids:
                raise ValueError("Training split is empty")
            if not holdout_scene_ids:
                raise ValueError("Validation/holdout split is empty")
            if len(train_scene_ids) % world_size != 0 and args.rank_balance_policy == STRICT_RANK_BALANCE:
                raise ValueError(
                    "Training split must divide evenly across ranks: "
                    f"{len(train_scene_ids)} scenes / {world_size} ranks"
                )
            if args.dataset == "scannet":
                if len(train_scene_ids) != WORLD_SIZE * SCENES_PER_RANK:
                    raise ValueError(
                        f"Expected {WORLD_SIZE * SCENES_PER_RANK} ScanNet train "
                        f"scenes, got {len(train_scene_ids)}"
                    )
                if len(holdout_scene_ids) != 21:
                    raise ValueError(
                        "Expected 21 ScanNet upstream holdout scenes, "
                        f"got {len(holdout_scene_ids)}"
                    )
            if set(train_scene_ids) & set(holdout_scene_ids):
                raise ValueError("Training and validation/holdout splits overlap")
            local_scene_ids, scenes_per_rank = partition_scene_ids(
                train_scene_ids, rank=rank, world_size=world_size, policy=args.rank_balance_policy,
            )
            epochs = int(args.epochs)
            scene_visits_per_epoch_per_rank = scenes_per_rank
            eval_epochs = {
                epoch for epoch in DEFAULT_EVAL_EPOCHS if epoch <= epochs
            } | {epochs}
            if args.source_audit is not None or args.dataset == "scannet":
                source_audit_path = (
                    args.source_audit.resolve(strict=True)
                    if args.source_audit is not None
                    else args.source_root.parent
                    / "audits"
                    / "source_dataset_audit.json"
                )
                source_audit_report, expected_manifest_hashes = (
                    _load_dataset_source_audit(
                        path=source_audit_path,
                        dataset=args.dataset,
                        source_root=args.source_root,
                        train_split=args.train_split,
                        holdout_split=args.holdout_split,
                    )
                )
            else:
                # Structured3D has a different scene/source-manifest contract;
                # each manifest and every referenced artifact is still audited
                # by load_scene_source below, but there is not yet a ScanNet-
                # shaped dataset-level audit file to reuse here.
                source_audit_report = None
                expected_manifest_hashes = None
            if (
                source_loading_mode == "logical_batch"
                and expected_manifest_hashes is None
            ):
                raise ValueError(
                    "logical_batch source loading requires a frozen dataset "
                    "source audit"
                )

        logical_batch_schedule = None
        if args.flush_logical_batch_at_eval:
            if args.evaluation_epochs is None or 0 not in args.evaluation_epochs or epochs not in args.evaluation_epochs:
                raise ValueError("Logical-batch flushing requires explicit initial and final evaluation epochs")
            logical_batch_schedule = LogicalBatchSchedule(
                scenes_per_epoch=int(scene_visits_per_epoch_per_rank),
                nominal_batch_size=scenes_per_rank_per_update,
                epochs=epochs,
                flush_epochs=tuple(e for e in args.evaluation_epochs if e > 0),
            )

        def visits_through_update(value):
            return (logical_batch_schedule.consumed_visits(int(value))
                    if logical_batch_schedule is not None
                    else int(value) * scenes_per_rank_per_update)

        nominal_scene_visits_per_rank = (
            int(epochs) * int(scene_visits_per_epoch_per_rank)
        )
        if (
            args.maximum_updates is None
            and logical_batch_schedule is None
            and nominal_scene_visits_per_rank % scenes_per_rank_per_update != 0
        ):
            raise ValueError(
                "The complete run must contain full per-rank logical batches: "
                f"scene_visits={nominal_scene_visits_per_rank} "
                f"scenes_per_rank_per_update={scenes_per_rank_per_update}"
            )
        if args.maximum_updates is None:
            total_updates = (
                logical_batch_schedule.total_updates if logical_batch_schedule is not None
                else nominal_scene_visits_per_rank // scenes_per_rank_per_update
            )
        else:
            total_updates = int(args.maximum_updates)
            if total_updates <= 0:
                raise ValueError("--maximum-updates must be positive")
            consumed = total_updates * scenes_per_rank_per_update
            if consumed > nominal_scene_visits_per_rank:
                raise ValueError(
                    "--maximum-updates exceeds the nominal scene horizon: "
                    f"consumed={consumed} available={nominal_scene_visits_per_rank}"
                )
            if sampling_mode != V2_SAMPLING_MODE:
                raise ValueError("--maximum-updates is reserved for V2 funnel panels")
        total_scene_visits_per_rank = visits_through_update(total_updates)
        if args.pause_after_update is not None:
            if args.maximum_updates is not None:
                raise ValueError(
                    "--pause-after-update and --maximum-updates are mutually exclusive"
                )
            if sampling_mode != V2_SAMPLING_MODE:
                raise ValueError("--pause-after-update is reserved for V2 production")
            execution_stop_update = int(args.pause_after_update)
            if not 0 < execution_stop_update <= total_updates:
                raise ValueError("--pause-after-update lies outside the run horizon")
            if (
                visits_through_update(execution_stop_update)
            ) % scene_visits_per_epoch_per_rank != 0:
                raise ValueError(
                    "--pause-after-update must land on an exact epoch boundary"
                )
        else:
            execution_stop_update = total_updates

        if args.evaluation_epochs is not None:
            eval_epochs = set(args.evaluation_epochs)
            if 0 not in eval_epochs:
                raise ValueError("--evaluation-epochs must include epoch 0")
            if any(value > epochs for value in eval_epochs):
                raise ValueError("An evaluation epoch exceeds --epochs")
        snapshot_updates = tuple(args.audit_snapshot_updates or ())
        snapshot_labels = tuple(args.audit_snapshot_labels or ())
        if bool(snapshot_updates) != bool(snapshot_labels) or len(
            snapshot_updates
        ) != len(snapshot_labels):
            raise ValueError(
                "Audit snapshot updates and labels must be supplied one-to-one"
            )
        if any(value <= 0 or value > total_updates for value in snapshot_updates):
            raise ValueError("Audit snapshot updates must lie inside the run horizon")
        if len(set(snapshot_labels)) != len(snapshot_labels):
            raise ValueError("Audit snapshot labels must be unique")
        snapshot_label_by_update = dict(zip(snapshot_updates, snapshot_labels))
        checkpoint_updates = set(args.checkpoint_updates or ())
        if not checkpoint_updates.issubset(set(snapshot_updates)):
            raise ValueError("Checkpoint updates must be audit snapshot updates")
        if scenes_per_rank_per_update > 1 and args.evaluation_epochs is None:
            if smoke:
                eval_epochs = {
                    epoch
                    for epoch in eval_epochs
                    if (
                        int(epoch) * int(scene_visits_per_epoch_per_rank)
                    )
                    % scenes_per_rank_per_update
                    == 0
                }
                eval_epochs.add(0)
                eval_epochs.add(epochs)
            else:
                eval_epochs = {
                    epoch
                    for epoch in LOGICAL_BATCH_EVAL_EPOCHS
                    if epoch <= epochs
                } | {epochs}
        checkpoint_epochs = (
            set(args.checkpoint_epochs)
            if args.checkpoint_epochs is not None
            else set(eval_epochs)
        )
        if not checkpoint_epochs.issubset(eval_epochs):
            raise ValueError("Checkpoint epochs must also be evaluation epochs")
        # Validate the final retention contract before model initialization or
        # GPU work. Maximum-update screens must end on an explicitly retained
        # audit snapshot; complete runs must retain their true final epoch.
        _final_evaluation_endpoint(
            nominal_epochs=epochs,
            total_updates=total_updates,
            maximum_updates=args.maximum_updates,
            snapshot_label_by_update=snapshot_label_by_update,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_updates=checkpoint_updates,
            checkpoint_dir=args.output_dir / "checkpoints",
        )
        unaligned_eval_epochs = sorted(
            epoch
            for epoch in eval_epochs
            if (
                int(epoch) * int(scene_visits_per_epoch_per_rank)
            )
            % scenes_per_rank_per_update
            != 0
        )
        if unaligned_eval_epochs and logical_batch_schedule is None:
            raise ValueError(
                "Evaluation epochs must end on complete logical batches: "
                f"{unaligned_eval_epochs}"
            )

        extension_enabled = args.extension_start_epoch is not None
        if extension_enabled != (args.extension_parent_checkpoint is not None):
            raise ValueError(
                "--extension-start-epoch and --extension-parent-checkpoint "
                "must be supplied together"
            )
        if extension_enabled and args.resume is None:
            raise ValueError("An extension requires --resume")
        if extension_enabled and not 0 < int(args.extension_start_epoch) < epochs:
            raise ValueError("Extension start epoch must be inside the target schedule")
        extension_start_epoch = (
            int(args.extension_start_epoch) if extension_enabled else 0
        )
        extension_start_update = _update_for_completed_epoch(
            epoch=extension_start_epoch,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
            logical_batch_schedule=logical_batch_schedule,
        )
        if (
            args.sampling_profile == "pf_hnm_768_v1"
            and (args.fixed_eval_pack is None or args.code_manifest is None)
        ):
            raise ValueError(
                "pf_hnm_768_v1 requires --fixed-eval-pack and --code-manifest"
            )
        if (
            (args.multiscale_supervision or hierarchy_enabled or sampling_mode == V2_SAMPLING_MODE)
            and not smoke
            and args.code_manifest is None
        ):
            raise ValueError(
                "Deep-supervision/V2 production requires --code-manifest"
            )
        code_manifest_report: dict[str, Any] | None = None
        if args.code_manifest is not None:
            code_manifest_path = args.code_manifest.resolve(strict=True)
            code_manifest_payload = json.loads(
                code_manifest_path.read_text(encoding="utf-8")
            )
            accepted_code_manifest_schemas = {
                "litept_gt3_unsam3_clean1008_code_snapshot/v1",
                "chorus_rgbn6_sphere50k_pretrain_code/v1",
                "chorus_fullscene_rgbn6_multiscale_pretrain_code/v1",
                "structured3d_rgbn6_pretraining_v2_code_manifest/v1",
                "structured3d_sam2_bstar_code_manifest/v1",
                "structured3d_sam2_sstar_euler_code_manifest/v1",
                "delimit3d_code_manifest/v1",
            }
            if (
                code_manifest_payload.get("schema_version")
                not in accepted_code_manifest_schemas
                or not code_manifest_payload.get("passed")
            ):
                raise ValueError(
                    f"Code-snapshot manifest did not pass: {code_manifest_path}"
                )
            if code_manifest_payload["schema_version"] == "delimit3d_code_manifest/v1":
                from .experiment_record import verify_code_record
                verify_code_record(code_manifest_path)
            code_manifest_report = {
                "path": str(code_manifest_path),
                "sha256": sha256_file(code_manifest_path),
                "schema_version": code_manifest_payload["schema_version"],
                "snapshot_root": code_manifest_payload.get(
                    "snapshot_root", code_manifest_payload.get("archive")
                ),
                "passed": True,
            }

        output_exists = torch.tensor(
            int(args.output_dir.exists()) if rank == 0 else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.broadcast(output_exists, src=0)
        if int(output_exists) == 1 and args.resume is None:
            raise FileExistsError(
                f"Refusing to overwrite run directory: {args.output_dir}"
            )
        if int(output_exists) == 1 and args.resume_into_fresh_attempt:
            raise FileExistsError(
                "Fresh-attempt resume requires a new output directory: "
                f"{args.output_dir}"
            )
        if (
            int(output_exists) == 0
            and args.resume is not None
            and not extension_enabled
            and not args.resume_into_fresh_attempt
        ):
            raise FileNotFoundError(
                f"Resume requested but run directory is missing: {args.output_dir}"
            )
        if int(output_exists) == 0 and extension_enabled:
            if (
                args.resume.resolve(strict=True)
                != args.extension_parent_checkpoint.resolve(strict=True)
            ):
                raise ValueError(
                    "A new extension must resume directly from its pinned parent"
                )
        log.info(
            "rank=%d preparing %d source scenes with mode=%s workers=%d",
            rank,
            len(local_scene_ids),
            source_loading_mode,
            source_loader_workers,
        )
        local_source_audits: list[Any] = []
        smoke_padding_ranks = (
            padding_ranks(len(load_scene_ids(args.train_split)), world_size)
            if smoke and args.rank_balance_policy == ROTATING_RANK_BALANCE else ()
        )
        if smoke:
            smoke_scenes = [
                load_scene_source(
                    path,
                    sam2_maximum_raw_points=args.sam2_maximum_raw_points,
                    input_features=args.input_features,
                    coordinate_normalization_policy=(
                        args.coordinate_normalization_policy
                    ),
                    color_mean=color_mean,
                    color_std=color_std,
                    verify_hashes=True,
                )
                for path in smoke_manifest_paths
            ]
            if len({scene.scene_id for scene in smoke_scenes}) != len(smoke_scenes):
                raise ValueError("Smoke manifest list contains duplicate scene IDs")
            rotation = rank % len(smoke_scenes)
            local_scenes = smoke_scenes[rotation:] + smoke_scenes[:rotation]
            if rank in smoke_padding_ranks:
                # Exercise every production deficit rank and its exact resume
                # on real sources before admitting the production run.
                local_scenes = local_scenes[:-1]
        elif source_loading_mode == "preload":
            local_scenes = _preload_sources(
                scene_ids=local_scene_ids,
                sam2_maximum_raw_points=args.sam2_maximum_raw_points,
                source_root=args.source_root,
                smoke_manifest=None,
                expected_manifest_hashes=expected_manifest_hashes,
                input_features=args.input_features,
                coordinate_normalization_policy=(
                    args.coordinate_normalization_policy
                ),
                color_mean=color_mean,
                color_std=color_std,
            )
        else:
            local_source_audits = _audit_lazy_sources(
                scene_ids=local_scene_ids,
                source_root=args.source_root,
                expected_manifest_hashes=expected_manifest_hashes,
            )
            local_scenes = []
            source_loader_pool = ThreadPoolExecutor(
                max_workers=source_loader_workers,
                thread_name_prefix=f"scene-loader-rank{rank}",
            )
        if len(set(local_scene_ids)) != len(local_scene_ids) and not smoke:
            raise ValueError(f"rank={rank}: duplicate local source scene")
        if rank == 0:
            holdout_scenes = (
                [
                    load_scene_source(
                        path,
                        sam2_maximum_raw_points=500_000,
                        input_features=args.input_features,
                        coordinate_normalization_policy=(
                            args.coordinate_normalization_policy
                        ),
                        color_mean=color_mean,
                        color_std=color_std,
                        verify_hashes=True,
                    )
                    for path in smoke_manifest_paths
                ]
                if smoke
                else _preload_sources(
                    scene_ids=holdout_scene_ids,
                    sam2_maximum_raw_points=500_000,
                    source_root=args.source_root,
                    smoke_manifest=None,
                    expected_manifest_hashes=expected_manifest_hashes,
                    input_features=args.input_features,
                    coordinate_normalization_policy=(
                        args.coordinate_normalization_policy
                    ),
                    color_mean=color_mean,
                    color_std=color_std,
                    evaluation_visit_seed=(args.fixed_eval_seed if args.source_granularity == "sam2" else None),
                )
            )
        else:
            holdout_scenes = []

        loaded_crop_modes = {scene.crop_mode for scene in local_scenes}
        loaded_crop_modes.update(audit.crop_mode for audit in local_source_audits)
        if rank == 0:
            loaded_crop_modes.update(scene.crop_mode for scene in holdout_scenes)
        if required_crop_mode != "any" and loaded_crop_modes != {
            required_crop_mode
        }:
            raise ValueError(
                "Loaded source crop mode differs from the required contract: "
                f"required={required_crop_mode!r} observed={sorted(loaded_crop_modes)}"
            )

        sampling = sampling_config_for_profile(args.sampling_profile)
        evaluation_sampling_profile = str(
            args.evaluation_sampling_profile or args.sampling_profile
        )
        if (
            evaluation_sampling_profile != args.sampling_profile
            and (
                args.sampling_profile,
                evaluation_sampling_profile,
            )
            != ("pf_hnm_512_p16_v1", "pf_hnm_512_v1")
        ):
            raise ValueError(
                "The only supported split train/evaluation sampling contract "
                "is pf_hnm_512_p16_v1 -> pf_hnm_512_v1"
            )
        evaluation_sampling = sampling_config_for_profile(
            evaluation_sampling_profile
        )
        fixed_eval_pack_report: dict[str, Any] | None = None
        if args.fixed_eval_pack is not None:
            report_broadcast: list[dict[str, Any] | None] = [None]
            if rank == 0:
                if args.source_granularity == "sam2":
                    from delimit3d.training.sam2_pretraining import admit_sam2_holdout
                    holdout_scenes = [scene_source_for_visit(scene, epoch=0, global_seed=args.fixed_eval_seed)
                                      for scene in holdout_scenes]
                    holdout_scenes, token_plans = admit_sam2_holdout(
                        holdout_scenes, sampling=evaluation_sampling,
                        seed=args.fixed_eval_seed, grid_size=args.grid_size,
                        report_path=args.fixed_eval_pack.parent / (args.fixed_eval_pack.name + "_admission.json"),
                        allow_shared_proposal_negatives=args.allow_shared_proposal_negatives,
                    )
                else:
                    token_plans = None
                report_broadcast[0] = prepare_fixed_holdout_eval_pack(
                    pack_dir=args.fixed_eval_pack,
                    sam2_maximum_raw_points=500_000,
                    scenes=holdout_scenes,
                    sampling=evaluation_sampling,
                    sampling_profile=evaluation_sampling_profile,
                    seed=args.fixed_eval_seed,
                    cells=source_cells,
                    grid_size=args.grid_size,
                    precomputed_token_plans=token_plans,
                    allow_shared_proposal_negatives=args.allow_shared_proposal_negatives,
                )
            dist.broadcast_object_list(report_broadcast, src=0)
            fixed_eval_pack_report = report_broadcast[0]
            if fixed_eval_pack_report is None:
                raise RuntimeError("Fixed-eval pack report broadcast failed")
            dist.barrier()

        # Every arm loads an explicit, hash-verified initialization artifact.
        torch.manual_seed(args.seed)
        model = LitePTContrastiveModel(
            litept_root=args.litept_root,
            litept_variant=args.litept_variant,
            input_features=args.input_features,
            representative_sampling=args.representative_sampling,
            token_pure_dec0=(sampling_mode == V2_SAMPLING_MODE),
            token_pure_cells=(source_cells if single_granularity else V2_DEFAULT_2D_CELLS),
            grid_size=args.grid_size,
            multiscale_supervision=args.multiscale_supervision,
            multiscale_loss_weight=args.multiscale_loss_weight,
            multiscale_warmup_epochs=args.multiscale_warmup_epochs,
            hierarchy_supervision=hierarchy_enabled,
            hierarchy_mask_loss_weight=args.hierarchy_mask_loss_weight,
            hierarchy_mask_stage_weights=hierarchy_mask_stage_weights,
            hierarchy_mask_stage_weight_schedule=(
                hierarchy_mask_stage_weight_schedule
            ),
            hierarchy_query_projection_policy=(
                args.hierarchy_query_projection_policy
            ),
            hierarchy_vicreg_loss_weight=args.hierarchy_vicreg_loss_weight,
            hierarchy_warmup_fraction=args.hierarchy_warmup_fraction,
            hierarchy_variance_target=args.hierarchy_variance_target,
            hierarchy_covariance_weight=args.hierarchy_covariance_weight,
            hierarchy_vicreg_max_tokens=args.hierarchy_vicreg_max_tokens,
        ).to(device)
        initial_report = _strict_initial_load(
            model=model,
            path=args.initial_checkpoint,
            input_features=args.input_features,
            multiscale_supervision=args.multiscale_supervision,
            v2_auxiliary_initialization=(sampling_mode == V2_SAMPLING_MODE),
        )
        if initial_report["seed"] != int(args.initialization_seed):
            raise ValueError("Initialization seed differs from --initialization-seed")
        if args.normalization_policy == "sync_batchnorm":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(device)
        optimizer_groups, optimizer_group_report = optimizer_parameter_groups(
            model,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(optimizer_groups)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

        if smoke:
            warmup_updates = 1
        else:
            warmup_updates = _updates_for_scene_visit_duration(
                epochs=int(args.warmup_epochs),
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
            )
        scheduled_gradient_panel_updates = _scheduled_gradient_panel_updates(
            total_updates=total_updates,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
            # ``checkpoint_epochs`` is the evaluator's source of truth for
            # checkpoint-native gradient-panel expectations.  It defaults to
            # all evaluation epochs, while an explicit subset must not create
            # undeclared panels in cumulative_health/v1.
            eval_epochs=sorted(checkpoint_epochs),
            snapshot_updates=snapshot_updates,
            auxiliary_enabled=bool(
                hierarchy_enabled or args.multiscale_supervision
            ),
        )
        extension_parent_report = None
        if extension_enabled:
            extension_parent_path = args.extension_parent_checkpoint.resolve(
                strict=True
            )
            parent_digest: list[str | None] = [
                sha256_file(extension_parent_path) if rank == 0 else None
            ]
            dist.broadcast_object_list(parent_digest, src=0)
            if parent_digest[0] is None:
                raise RuntimeError("Extension parent digest broadcast failed")
            extension_parent_report = {
                "path": str(extension_parent_path),
                "sha256": parent_digest[0],
                "epoch": extension_start_epoch,
                "update": extension_start_update,
            }
        resolved = {
            "schema_version": "litept_gt3_unsam3_pretrain_resolved/v1",
            **({"litept_variant": args.litept_variant,
                "resolved_architecture": model.backbone.resolved_architecture}
               if args.litept_variant != "litept_s_star" or args.source_granularity == "sam2" else {}),
            "project_name": project_name(),
            "project_slug": project_slug(),
            "experiment_id": EXPERIMENT_ID,
            "dataset": args.dataset,
            "mode": (
                f"{args.dataset}_smoke" if smoke else f"{args.dataset}_production"
            ),
            "world_size": world_size,
            **({"rank_balance_policy": args.rank_balance_policy, "rank_balance": (
                balance_contract(train_scene_ids, world_size=world_size, epochs=epochs,
                                 policy=args.rank_balance_policy) if not smoke else {
                    "policy": args.rank_balance_policy,
                    "padding_visits_per_epoch": len(smoke_padding_ranks),
                    "padding_visits_total": epochs * len(smoke_padding_ranks),
                    "smoke_only": True,
                }
            )} if args.rank_balance_policy != STRICT_RANK_BALANCE else {}),
            "seed": int(args.seed),
            "initialization_seed": int(args.initialization_seed),
            "fixed_eval_seed": int(args.fixed_eval_seed),
            "epochs": epochs,
            "updates_per_epoch": (
                scene_visits_per_epoch_per_rank
                if scenes_per_rank_per_update == 1
                else None
            ),
            "scene_visits_per_epoch_per_rank": scene_visits_per_epoch_per_rank,
            "nominal_scene_visits_per_rank": nominal_scene_visits_per_rank,
            "executed_scene_visits_per_rank": (
                visits_through_update(execution_stop_update)
            ),
            **({"logical_batch_schedule": logical_batch_schedule.as_dict(),
                "short_update_microbatches_per_rank": {str(u): n for u, n in logical_batch_schedule.short_updates.items()}}
               if logical_batch_schedule is not None else {}),
            "scenes_per_rank_per_update": scenes_per_rank_per_update,
            "scenes_per_global_update": scenes_per_rank_per_update * world_size,
            "total_updates": total_updates,
            "pause_after_update": (
                None
                if args.pause_after_update is None
                else int(args.pause_after_update)
            ),
            "execution_stop_update": execution_stop_update,
            "maximum_updates": (
                None if args.maximum_updates is None else int(args.maximum_updates)
            ),
            "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
            "warmup_updates": warmup_updates,
            "backbone_lr": float(args.backbone_lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
            "min_lr_factor": 0.01,
            "lr_schedule": (
                {
                    "kind": "completed_checkpoint_cosine_restart",
                    "start_epoch": extension_start_epoch,
                    "start_update": extension_start_update,
                    "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
                    "warmup_updates": warmup_updates,
                    "start_factor": 0.01,
                    "peak_factor": 1.0,
                    "final_factor": 0.01,
                }
                if extension_enabled
                else {
                    "kind": "linear_warmup_cosine",
                    "start_epoch": 0,
                    "start_update": 0,
                    "warmup_epochs": 1 if smoke else int(args.warmup_epochs),
                    "warmup_updates": warmup_updates,
                    "start_factor": 0.0,
                    "peak_factor": 1.0,
                    "final_factor": 0.01,
                }
            ),
            "continuation": (
                {
                    "enabled": True,
                    "parent_checkpoint": extension_parent_report,
                    "added_epochs": epochs - extension_start_epoch,
                    "added_updates": total_updates - extension_start_update,
                }
                if extension_enabled
                else {"enabled": False}
            ),
            "precision": "fp32",
            "device_reports": device_reports,
            "input_features": args.input_features,
            "input_channels": int(model.input_channels),
            "coordinate_normalization_policy": str(
                args.coordinate_normalization_policy
            ),
            "model_coordinate_frame": (
                "scannet_z_up_v1"
                if args.coordinate_normalization_policy
                == STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY
                else "source_native"
            ),
            "rgb_color_mean": list(color_mean) if color_mean is not None else None,
            "rgb_color_std": list(color_std) if color_std is not None else None,
            "rgb_color_normalization": (
                RGBN6_COLOR_NORMALIZATION if color_mean is not None else None
            ),
            "pretraining_recipe": model.recipe_name,
            "sampling_mode": sampling_mode,
            "token_pure_dec0": bool(model.token_pure_dec0),
            "hierarchy_supervision": {
                "contract_version": "litept_query_mask_hierarchy/v2",
                "mode": hierarchy_mode,
                "enabled": hierarchy_enabled,
                "mask_loss_weight": float(args.hierarchy_mask_loss_weight),
                "mask_stage_weights": dict(hierarchy_mask_stage_weights),
                "mask_stage_weight_schedule": hierarchy_mask_stage_weight_schedule,
                "query_projection_policy": str(
                    args.hierarchy_query_projection_policy
                ),
                "vicreg_loss_weight": float(args.hierarchy_vicreg_loss_weight),
                "warmup_fraction": float(args.hierarchy_warmup_fraction),
                "variance_target": float(args.hierarchy_variance_target),
                "covariance_weight": float(args.hierarchy_covariance_weight),
                "vicreg_max_tokens": int(args.hierarchy_vicreg_max_tokens),
                "query_source": "detached_native_dec0_two_fold_pure_tokens",
                "query_projection": "layernorm_then_linear_128",
                "target": "positive_known_over_total_known",
                "unknown_and_mixed_policy": "ignored",
                "g02_route": (
                    "dec2_iff_each_shared_fold_has_at_least_two_positive_known_"
                    "tokens_and_nonzero_known_negative_mass_else_dec1_exactly_once"
                ),
                "g05_route": ["dec3", "dec2"],
                "g08_route": ["enc4", "dec3"],
                "vicreg_statistics": (
                    "distributed_scene_balanced_native_fp32_global_min_rank_tokens"
                ),
            },
            "normalization": {
                "policy": str(args.normalization_policy),
                "bn_recalibration_scenes": int(args.bn_recalibration_scenes),
                "retained_checkpoint_recalibration": "required_sidecar",
            },
            "feature_hard_gate": {
                "warmup_fraction": float(args.feature_hard_warmup_fraction),
                "cosine_gap_threshold": float(args.feature_hard_gap_threshold),
                "required_consecutive_panels": int(
                    args.feature_hard_required_panels
                ),
                "required_metric_space": "native_token_pure_dec0_fixed_pack",
                "dense_point_holdout_is_diagnostic_only": True,
            },
            "gradient_conflict_policy": {
                "cosine_threshold": -0.05,
                "required_consecutive_panels": 2,
                "first_action": "halve_objective_coefficient_once",
                "controller_gradient": (
                    "globally_summed_six_microbatch_shared_backbone_gradient"
                ),
                "second_action": "shared_backbone_pcgrad_vs_final_dec0",
                "private_head_policy": "unprojected_total_gradient",
                "third_action": "reject_candidate",
                "total_auxiliary_gradient_budget_vs_primary": 0.5,
            },
            "gradient_panel_schedule": {
                "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
                "updates": scheduled_gradient_panel_updates,
                "sources": [
                    "first_optimizer_update",
                    "evaluation_epoch_boundaries",
                    "audit_snapshot_updates",
                ],
            },
            "multiscale_supervision": bool(args.multiscale_supervision),
            "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
            "multiscale_level_loss_weights": dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            ),
            "multiscale_loss_weight": float(args.multiscale_loss_weight),
            "multiscale_warmup_epochs": int(args.multiscale_warmup_epochs),
            "multiscale_token_policy": (
                "strict_pure_token_resampling_v1"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_temperature_policy": (
                "separate_learnable_temperature_per_auxiliary_level"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_feature_hard_policy": (
                "separate_candidate_selection_per_positive_endpoint"
                if args.multiscale_supervision
                else None
            ),
            "multiscale_transfer_contract": (
                "native_mask3d_fpn_required"
                if args.multiscale_supervision or hierarchy_enabled
                else "final_dec0_feature"
            ),
            "checkpoint_selection_policy": (
                "fixed_dec0_native_selection_macro"
                if single_granularity
                else "external_five_stage_health_then_fixed_dec0_metric"
                if sampling_mode == V2_SAMPLING_MODE
                else "final_dec0_native_selection_macro"
                if args.multiscale_supervision
                else "native_selection_macro"
            ),
            "train_augmentation_profile": train_augmentation_profile,
            "train_augmentation": augmentation_contract_for_profile(
                train_augmentation_profile
            ),
            "evaluation_augmentation_profile": "none",
            "required_crop_mode": required_crop_mode,
            "loaded_crop_modes_rank0": sorted(loaded_crop_modes),
            "point_input_contract": (
                f"all_source_points_preserved_before_{args.grid_size * 100:g}cm_voxelization"
                if required_crop_mode == "full_scene"
                else f"all_stored_source_points_before_{args.grid_size * 100:g}cm_voxelization"
            ),
            "source_loading_mode": source_loading_mode,
            "source_loader_workers": source_loader_workers,
            "source_archive_integrity": (
                "frozen_archive_and_source_audit_verified_before_lazy_loading"
                if source_loading_mode == "logical_batch"
                else "per_scene_artifact_hashes_verified_during_preload"
            ),
            "grad_clip_norm": float(args.grad_clip_norm),
            "cache_training_voxelization": bool(
                model.backbone.cache_training_voxelization
            ),
            "voxel_reduce": model.backbone.voxel_reduce,
            "representative_sampling": model.backbone.representative_sampling,
            "voxelization_mode": "per_forward_single_scene_recompute",
            "source_family": source_family,
            "source_cells": list(source_cells),
            "source_cell_schedule": args.source_cell_schedule,
            "source_cell_schedule_seed": int(args.seed),
            "three_d_mode": None if source_family == SOURCE_FAMILY_2D else args.three_d_mode,
            "three_d_slot_to_label": (
                {}
                if source_family == SOURCE_FAMILY_2D
                else (
                    {
                        "3d/g02": "g02",
                        "3d/g05": "g05",
                        "3d/g08": "g08",
                    }
                    if args.three_d_mode == "gt3"
                    else {
                        "3d/g02": "g08",
                        "3d/g05": "g08",
                        "3d/g08": "g08",
                    }
                )
            ),
            **sampling_ablation,
            "sampling_profile": args.sampling_profile,
            **({"allow_shared_proposal_negatives": args.allow_shared_proposal_negatives,
                "negative_policy": negative_policy_name(args.allow_shared_proposal_negatives)}
               if args.source_granularity == "sam2" else {}),
            "sampling": vars(sampling),
            "sampling_v2_contract": (
                {
                    "schema_version": "multigranular_frame_local_coverage/v2",
                    "quota": ("single_cell_16_without_replacement" if single_granularity
                              else "rotating_6_5_5_without_replacement"),
                    "loss_group": "physical_frame_local",
                    "negative_known_support": (
                        "visible_intersection_same_frame_selected_granularity_relation_union"
                        if single_granularity
                        else "visible_intersection_same_frame_all_granularity_relation_union"
                    ),
                    "post_voxel_policy": (
                        "exact_single_cell_token_pure_or_synchronized_zero_skip"
                        if single_granularity
                        else "exact_three_cell_token_pure_or_synchronized_zero_skip"
                    ),
                    "resume_policy": (
                        "epoch_keyed_deterministic_reconstruction_with_prior_epoch_"
                        "cursor_audit_validation"
                    ),
                }
                if sampling_mode == V2_SAMPLING_MODE
                else None
            ),
            "evaluation_sampling_profile": evaluation_sampling_profile,
            "evaluation_sampling": vars(evaluation_sampling),
            "evaluation_proposals_per_forward": int(
                evaluation_sampling.proposals_per_forward
            ),
            "proposals_per_rank_per_update": (
                scenes_per_rank_per_update * int(sampling.proposals_per_forward)
            ),
            "proposals_per_global_update": (
                world_size
                * scenes_per_rank_per_update
                * int(sampling.proposals_per_forward)
            ),
            "fixed_eval_pack": fixed_eval_pack_report,
            "code_manifest": code_manifest_report,
            "eval_epochs": sorted(eval_epochs),
            "checkpoint_epochs": sorted(checkpoint_epochs),
            "audit_snapshot_updates": [
                {
                    "update": int(snapshot_update),
                    "label": int(snapshot_label_by_update[snapshot_update]),
                    "checkpoint": snapshot_update in checkpoint_updates,
                }
                for snapshot_update in snapshot_updates
            ],
            "initialization": initial_report,
            "execution_provenance": execution_provenance_report,
            "resume_parent": (
                {
                    "sha256": resume_sha256,
                }
                if resume_sha256 is not None
                else None
            ),
            "resume_into_fresh_attempt": bool(
                args.resume_into_fresh_attempt
            ),
            "source_dataset_audit": source_audit_report,
            **({"sam2_scene_visit_contract": {
                "maximum_raw_points": args.sam2_maximum_raw_points,
                "fragment_policy": sam2_fragment_policy(args.sam2_maximum_raw_points),
                **({"evaluation_maximum_raw_points": 500_000,
                    "evaluation_fragment_policy": sam2_fragment_policy(500_000)}
                   if args.sam2_maximum_raw_points != 500_000 else {}),
                "canonical_pack_mode": "full_scene",
                "logical_scene_visits_per_epoch": 1,
                "evaluation_fragment_epoch": 0,
                "evaluation_fragment_seed": args.fixed_eval_seed,
                "fragment_catalog_coverage": SAM2_COVERAGE_POLICY,
            }} if args.source_granularity == "sam2" else {}),
            "optimizer_parameter_groups": optimizer_group_report,
            "train_split": (
                {
                    "path": str(args.train_split.resolve(strict=True)),
                    "sha256": sha256_file(args.train_split),
                    "ordered_digest": split_digest(
                        load_scene_ids(args.train_split)
                    ),
                }
                if not smoke
                else None
            ),
            "holdout_split": (
                {
                    "path": str(args.holdout_split.resolve(strict=True)),
                    "sha256": sha256_file(args.holdout_split),
                    "ordered_digest": split_digest(
                        load_scene_ids(args.holdout_split)
                    ),
                }
                if not smoke
                else None
            ),
            "local_scene_ids": {
                str(candidate_rank): (
                    partition_scene_ids(load_scene_ids(args.train_split),
                                        rank=candidate_rank, world_size=world_size,
                                        policy=args.rank_balance_policy)[0]
                    if not smoke
                    else [
                        smoke_scenes[
                            (candidate_rank + index) % len(smoke_scenes)
                        ].scene_id
                        for index in range(len(smoke_scenes) - int(
                            candidate_rank in smoke_padding_ranks))
                    ]
                )
                for candidate_rank in range(expected_world_size)
            },
            "holdout_scene_ids": (
                load_scene_ids(args.holdout_split)
                if not smoke
                else [scene.scene_id for scene in smoke_scenes]
            ),
            "public_checkpoint_used": bool(
                initial_report["public_checkpoint_used"]
            ),
        }
        if single_granularity:
            resolved["source_granularity"] = args.source_granularity
            resolved["supervised_stages"] = ["dec0"]
        if args.source_granularity == "sam2" or float(args.grid_size) != 0.02:
            resolved["grid_size"] = float(args.grid_size)
        resolved["semantic_config_sha256"] = _semantic_config_sha256(resolved)
        metrics_path = args.output_dir / "metrics.jsonl"

        start_epoch = 0
        inherited_evaluation: dict[str, Any] | None = None
        inherited_evaluations: dict[int, dict[str, Any]] = {}
        lineage_parent_checkpoints: list[dict[str, Any]] = []
        extension_bootstrap = bool(extension_enabled and int(output_exists) == 0)
        rank_exposure_counts: dict[str, int] = {cell: 0 for cell in source_cells}
        rank_coverage_states: dict[str, Any] = {}
        rank_sampling_totals: dict[str, dict[str, int]] = {
            cell: {
                "requested_proposals": 0,
                "realized_unique_proposals": 0,
                "contrastive_retained_proposals": 0,
                "duplicate_proposals": 0,
                "realized_positive_pairs": 0,
                "requested_positive_pairs": 0,
                "active_scene_visits": 0,
                "loss_mass_millionths": 0,
                "optimized_scene_visits": 0,
                "invalid_scene_visits": 0,
                "token_realized_positive_pairs": 0,
                "token_requested_positive_pairs": 0,
                "optimized_loss_mass_millionths": 0,
            }
            for cell in source_cells
        }
        feature_hard_state: dict[str, Any] = {
            "schema_version": "litept_feature_hard_gate/v1",
            "enabled": False,
            "qualifying_panel_streak": 0,
            "panel_count": 0,
            "last_panel_update": None,
            "last_panel_cosine_gap": None,
            "last_dense_point_diagnostic_cosine_gap": None,
            "metric_space": "native_token_pure_dec0_fixed_pack",
            "dense_point_holdout_eligible_for_unlock": False,
            "native_token_metric_eligible_for_unlock": False,
            "activation_update": None,
        }
        conflict_objective_names = (
            ("hierarchy_mask_scaled", "hierarchy_vicreg_scaled")
            if hierarchy_enabled
            else ("legacy_auxiliary_scaled",)
            if args.multiscale_supervision
            else ()
        )
        gradient_conflict_state: dict[str, Any] = {
            "schema_version": "litept_auxiliary_conflict_controller/v2",
            "threshold": -0.05,
            "required_consecutive_panels": 2,
            "rejected": False,
            "rejection_reason": None,
            "auxiliary_budget": {
                "limit": 0.5,
                "last_global_sum_aux_norm_ratio": None,
                "violation_count": 0,
                "passed": None,
            },
            "objectives": {
                objective: {
                    "panel_streak": 0,
                    "coefficient_halved": False,
                    "pcgrad_enabled": False,
                    "post_pcgrad_failure": False,
                    "last_cosine": None,
                    "last_action_update": None,
                }
                for objective in conflict_objective_names
            },
        }
        rank_cumulative_health: dict[str, Any] | None = (
            _new_rank_cumulative_health(
                rank=rank,
                cells=source_cells,
                hierarchy_enabled=hierarchy_enabled,
                expected_microbatches_per_update=scenes_per_rank_per_update,
                logical_batch_schedule=logical_batch_schedule,
                legacy_multiscale_enabled=bool(
                    args.multiscale_supervision and not hierarchy_enabled
                ),
                scheduled_gradient_panel_updates=(
                    scheduled_gradient_panel_updates
                ),
            )
            if sampling_mode == V2_SAMPLING_MODE or args.multiscale_supervision
            else None
        )
        if args.resume is not None:
            checkpoint = torch.load(
                args.resume, map_location="cpu", weights_only=False
            )
            accepted_checkpoint_schemas = {
                "litept_gt3_unsam3_pretrain_checkpoint/v1",
                "litept_rgbn6_multigranular_pretrain_checkpoint/v2",
            }
            if checkpoint.get("schema_version") not in accepted_checkpoint_schemas:
                raise ValueError("Resume checkpoint schema drift")
            if checkpoint.get("experiment_id") != EXPERIMENT_ID:
                raise ValueError("Resume checkpoint experiment drift")
            raw_evaluation_history = checkpoint.get("evaluation_history")
            if not isinstance(raw_evaluation_history, dict):
                if args.resume_into_fresh_attempt:
                    raise ValueError(
                        "Fresh-attempt resume requires checkpoint evaluation history"
                    )
            else:
                inherited_evaluations = {
                    int(epoch): dict(value)
                    for epoch, value in raw_evaluation_history.items()
                }
            resolved["resume_parent"] = {
                "sha256": resume_sha256,
                "schema_version": checkpoint["schema_version"],
                "epoch": int(checkpoint.get("epoch", -1)),
                "update": int(checkpoint.get("update", -1)),
            }
            previous = checkpoint["resolved_config"]
            validate_negative_policy_resume(previous, resolved, checkpoint)
            if (
                args.multiscale_supervision or sampling_mode == V2_SAMPLING_MODE
            ) and not extension_enabled:
                previous_digest = previous.get("semantic_config_sha256")
                if not isinstance(previous_digest, str):
                    raise ValueError(
                        "Multi-scale resume lacks a semantic-config digest"
                    )
                observed_previous_digest = _semantic_config_sha256(previous)
                if previous_digest != observed_previous_digest:
                    raise ValueError(
                        "Resume checkpoint semantic-config digest is corrupt"
                    )
                if previous_digest != resolved["semantic_config_sha256"]:
                    raise ValueError(
                        "Complete semantic configuration differs on resume"
                    )
            base_immutable = (
                "dataset",
                "world_size",
                "seed",
                "initialization_seed",
                "fixed_eval_seed",
                "updates_per_epoch",
                "scene_visits_per_epoch_per_rank",
                "rank_balance_policy",
                "rank_balance",
                "scenes_per_rank_per_update",
                "source_loading_mode",
                "source_loader_workers",
                "input_features",
                "input_channels",
                "coordinate_normalization_policy",
                "model_coordinate_frame",
                "train_augmentation_profile",
                "train_augmentation",
                "evaluation_augmentation_profile",
                "required_crop_mode",
                "pretraining_recipe",
                "sampling_mode",
                "token_pure_dec0",
                "hierarchy_supervision",
                "normalization",
                "feature_hard_gate",
                "gradient_conflict_policy",
                "multiscale_supervision",
                "multiscale_levels",
                "multiscale_level_loss_weights",
                "multiscale_loss_weight",
                "multiscale_warmup_epochs",
                "multiscale_token_policy",
                "multiscale_temperature_policy",
                "multiscale_feature_hard_policy",
                "multiscale_transfer_contract",
                "checkpoint_selection_policy",
                "backbone_lr",
                "head_lr",
                "weight_decay",
                "grad_clip_norm",
                "initialization",
                "source_dataset_audit",
                "three_d_mode",
                "source_family",
                "source_cells",
                "source_granularity",
                "sam2_scene_visit_contract",
                "grid_size",
                "source_cell_schedule",
                "source_cell_schedule_seed",
                "sampling_profile",
                *sampling_ablation,
                "ablation_variant",
                "allow_shared_proposal_negatives",
                "negative_policy",
                "sampling",
                "sampling_v2_contract",
                "evaluation_sampling_profile",
                "evaluation_sampling",
                "cache_training_voxelization",
                "voxel_reduce",
                "voxelization_mode",
                "fixed_eval_pack",
                "code_manifest",
                "train_split",
                "holdout_split",
            )
            base_mismatches: list[str] = []
            for key in base_immutable:
                previous_value = previous.get(key)
                resolved_value = resolved.get(key)
                if key == "ablation_variant":
                    previous_value = previous.get(key, "full")
                    resolved_value = resolved.get(key, "full")
                elif key == "allow_shared_proposal_negatives":
                    previous_value = previous.get(key, False)
                    resolved_value = resolved.get(key, False)
                if key == "negative_policy":
                    previous_value = previous.get(key, negative_policy_name(previous.get("allow_shared_proposal_negatives", False)))
                    resolved_value = resolved.get(key, negative_policy_name(resolved.get("allow_shared_proposal_negatives", False)))
                if key == "initialization":
                    previous_value = _content_addressed_report(
                        normalize_initialization_report(previous_value)
                    )
                    resolved_value = _content_addressed_report(
                        normalize_initialization_report(resolved_value)
                    )
                elif key in {
                    "train_split",
                    "holdout_split",
                    "fixed_eval_pack",
                    "code_manifest",
                    "execution_provenance",
                    "source_dataset_audit",
                }:
                    previous_value = _content_addressed_report(previous_value)
                    resolved_value = _content_addressed_report(resolved_value)
                elif key == "sampling_profile" and previous_value is None:
                    if previous.get("sampling") == vars(
                        sampling_config_for_profile("pf_hnm_512_v1")
                    ):
                        previous_value = "pf_hnm_512_v1"
                elif (
                    key == "evaluation_sampling_profile"
                    and previous_value is None
                ):
                    previous_value = previous.get(
                        "sampling_profile", "pf_hnm_512_v1"
                    )
                elif key == "evaluation_sampling" and previous_value is None:
                    previous_value = previous.get("sampling")
                elif key == "scene_visits_per_epoch_per_rank" and previous_value is None:
                    previous_value = previous.get("updates_per_epoch")
                elif key == "scenes_per_rank_per_update" and previous_value is None:
                    previous_value = 1
                elif key == "source_loading_mode" and previous_value is None:
                    previous_value = "preload"
                elif key == "source_loader_workers" and previous_value is None:
                    previous_value = 1
                elif key == "train_augmentation_profile" and previous_value is None:
                    previous_value = "none"
                elif key == "train_augmentation" and previous_value is None:
                    previous_value = augmentation_contract_for_profile("none")
                elif key == "evaluation_augmentation_profile" and previous_value is None:
                    previous_value = "none"
                elif key == "pretraining_recipe" and previous_value is None:
                    previous_value = "final_dec0_only_v1"
                elif key == "multiscale_supervision" and previous_value is None:
                    previous_value = False
                elif key == "multiscale_levels" and previous_value is None:
                    previous_value = list(MULTISCALE_LEVEL_NAMES)
                elif (
                    key == "multiscale_level_loss_weights"
                    and previous_value is None
                ):
                    previous_value = dict(MULTISCALE_LEVEL_LOSS_WEIGHTS)
                elif key == "multiscale_loss_weight" and previous_value is None:
                    previous_value = MULTISCALE_AUXILIARY_WEIGHT
                elif key == "multiscale_warmup_epochs" and previous_value is None:
                    previous_value = MULTISCALE_AUXILIARY_WARMUP_EPOCHS
                elif key == "multiscale_token_policy" and previous_value is None:
                    previous_value = (
                        "strict_pure_token_resampling_v1"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif (
                    key == "multiscale_temperature_policy"
                    and previous_value is None
                ):
                    previous_value = (
                        "separate_learnable_temperature_per_auxiliary_level"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif (
                    key == "multiscale_feature_hard_policy"
                    and previous_value is None
                ):
                    previous_value = (
                        "separate_candidate_selection_per_positive_endpoint"
                        if bool(previous.get("multiscale_supervision", False))
                        else None
                    )
                elif key == "multiscale_transfer_contract" and previous_value is None:
                    previous_value = "final_dec0_feature"
                elif key == "required_crop_mode" and previous_value is None:
                    previous_value = "any"
                elif key == "source_cell_schedule" and previous_value is None:
                    previous_value = SOURCE_CELL_SCHEDULE_FIXED
                elif key == "source_cell_schedule_seed" and previous_value is None:
                    previous_value = previous.get("seed")
                elif (
                    key == "cache_training_voxelization"
                    and previous_value is None
                    and args.sampling_profile == "pf_hnm_512_v1"
                ):
                    previous_value = False
                elif (
                    key == "voxelization_mode"
                    and previous_value is None
                    and args.sampling_profile == "pf_hnm_512_v1"
                ):
                    previous_value = "per_forward_single_scene_recompute"
                elif key == "source_family" and previous_value is None:
                    previous_cells = previous.get("source_cells")
                    if previous_cells is None or list(previous_cells) == list(
                        SOURCE_CELLS
                    ):
                        previous_value = "2d3d"
                elif (
                    key == "three_d_mode"
                    and previous_value is None
                    and resolved_value is None
                    and source_family == SOURCE_FAMILY_2D
                ):
                    previous_value = None
                if previous_value != resolved_value:
                    base_mismatches.append(key)
            if base_mismatches:
                raise ValueError(
                    "Resume base configuration differs from current run: "
                    + ", ".join(base_mismatches)
                )
            if extension_bootstrap:
                if int(checkpoint.get("epoch", -1)) != extension_start_epoch:
                    raise ValueError(
                        "Extension parent epoch differs from --extension-start-epoch"
                    )
                if int(checkpoint.get("update", -1)) != extension_start_update:
                    raise ValueError("Extension parent update differs from start update")
                if (
                    int(previous.get("epochs", -1)) != extension_start_epoch
                    or int(previous.get("total_updates", -1))
                    != extension_start_update
                ):
                    raise ValueError(
                        "Extension parent is not the completed source schedule"
                    )
                if sampling_mode == V2_SAMPLING_MODE:
                    parent_rank_counts = checkpoint.get("rank_exposure_counts")
                    if (
                        checkpoint.get("resumable") is not True
                        or not isinstance(parent_rank_counts, list)
                        or checkpoint.get("global_exposure_counts")
                        != _merge_count_dicts(parent_rank_counts)
                    ):
                        raise ValueError(
                            "V2 extension parent exposure/replay state is not exact"
                        )
                else:
                    expected_parent_counts = exposure_counts(
                        load_scene_ids(args.train_split),
                        epochs=extension_start_epoch,
                        cells=source_cells,
                        proposals_per_forward=sampling.proposals_per_forward,
                        cell_schedule=args.source_cell_schedule,
                        seed=args.seed,
                    )
                    if checkpoint.get("global_exposure_counts") != expected_parent_counts:
                        raise ValueError("Extension parent exposure counts are not exact")
                inherited_evaluation = checkpoint.get("evaluation")
                if not isinstance(inherited_evaluation, dict):
                    raise ValueError("Extension parent lacks its boundary evaluation")
            else:
                exact_immutable = (
                    "epochs",
                    "total_updates",
                    "warmup_updates",
                    "eval_epochs",
                    "lr_schedule",
                    "continuation",
                )
                if any(
                    previous.get(key) != resolved.get(key)
                    for key in exact_immutable
                ):
                    raise ValueError("Exact resume configuration differs from current run")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"])
            rank_exposure_counts = {
                key: int(value)
                for key, value in checkpoint["rank_exposure_counts"][rank].items()
            }
            if sampling_mode == V2_SAMPLING_MODE:
                if checkpoint.get("resumable") is not True:
                    raise ValueError("V2 resume requires a resumable epoch checkpoint")
                rank_coverage_records = checkpoint.get("rank_coverage_states")
                rank_rng_records = checkpoint.get("rank_rng_states")
                if (
                    not isinstance(rank_coverage_records, list)
                    or len(rank_coverage_records) != world_size
                    or not isinstance(rank_rng_records, list)
                    or len(rank_rng_records) != world_size
                ):
                    raise ValueError("V2 resume lacks complete per-rank state")
                rank_coverage_states = dict(rank_coverage_records[rank])
                validate_saved_coverage_states(
                    rank_coverage_states, start_epoch=start_epoch, seed=args.seed,
                )
                saved_sampling_totals = checkpoint.get("rank_sampling_totals")
                if (
                    not isinstance(saved_sampling_totals, list)
                    or len(saved_sampling_totals) != world_size
                ):
                    raise ValueError("V2 resume lacks cumulative sampling totals")
                rank_sampling_totals = {
                    cell: {
                        key: int(value) for key, value in values.items()
                    }
                    for cell, values in saved_sampling_totals[rank].items()
                }
                saved_feature_hard = checkpoint.get("feature_hard_state")
                if not isinstance(saved_feature_hard, dict):
                    raise ValueError("V2 resume lacks feature-hard gate state")
                feature_hard_state = dict(saved_feature_hard)
                feature_hard_state.setdefault(
                    "last_dense_point_diagnostic_cosine_gap", None
                )
                feature_hard_state.setdefault(
                    "metric_space", "native_token_pure_dec0_fixed_pack"
                )
                feature_hard_state.setdefault(
                    "dense_point_holdout_eligible_for_unlock", False
                )
                feature_hard_state.setdefault(
                    "native_token_metric_eligible_for_unlock", False
                )
                saved_conflict_state = checkpoint.get(
                    "gradient_conflict_state"
                )
                if not isinstance(saved_conflict_state, dict):
                    raise ValueError("V2 resume lacks gradient-conflict state")
                gradient_conflict_state = dict(saved_conflict_state)
                if bool(gradient_conflict_state.get("rejected", False)):
                    raise ValueError(
                        "Refusing to resume a checkpoint whose gradient controller "
                        "has already rejected the candidate"
                    )
                gradient_conflict_state.setdefault("rejection_reason", None)
                gradient_conflict_state.setdefault(
                    "auxiliary_budget",
                    {
                        "limit": 0.5,
                        "last_global_sum_aux_norm_ratio": None,
                        "violation_count": 0,
                        "passed": None,
                    },
                )
                saved_cumulative_health = checkpoint.get("cumulative_health")
                if not isinstance(saved_cumulative_health, Mapping):
                    raise ValueError("V2 resume lacks cumulative health state")
                if saved_cumulative_health.get(
                    "gradient_conflict_controller"
                ) != gradient_conflict_state:
                    raise ValueError(
                        "V2 resume cumulative/controller state is inconsistent"
                    )
                rank_cumulative_health = _restore_rank_cumulative_health(
                    saved_cumulative_health,
                    rank=rank,
                    checkpoint_update=int(checkpoint["update"]),
                    world_size=world_size,
                    scheduled_gradient_panel_updates=(
                        scheduled_gradient_panel_updates
                    ),
                    allow_schedule_extension=extension_enabled,
                )
                _restore_rng_state(rank_rng_records[rank], device)
            elif args.multiscale_supervision:
                if checkpoint.get("resumable") is not True:
                    raise ValueError(
                        "Legacy multiscale resume requires a resumable epoch checkpoint"
                    )
                rank_rng_records = checkpoint.get("rank_rng_states")
                if (
                    not isinstance(rank_rng_records, list)
                    or len(rank_rng_records) != world_size
                ):
                    raise ValueError("Legacy multiscale resume lacks complete RNG state")
                saved_conflict_state = checkpoint.get("gradient_conflict_state")
                if not isinstance(saved_conflict_state, dict):
                    raise ValueError(
                        "Legacy multiscale resume lacks gradient-conflict state"
                    )
                gradient_conflict_state = dict(saved_conflict_state)
                if bool(gradient_conflict_state.get("rejected", False)):
                    raise ValueError(
                        "Refusing to resume a checkpoint whose gradient controller "
                        "has already rejected the candidate"
                    )
                saved_cumulative_health = checkpoint.get("cumulative_health")
                if not isinstance(saved_cumulative_health, Mapping):
                    raise ValueError(
                        "Legacy multiscale resume lacks cumulative health state"
                    )
                if saved_cumulative_health.get(
                    "gradient_conflict_controller"
                ) != gradient_conflict_state:
                    raise ValueError(
                        "Legacy multiscale resume cumulative/controller state is inconsistent"
                    )
                rank_cumulative_health = _restore_rank_cumulative_health(
                    saved_cumulative_health,
                    rank=rank,
                    checkpoint_update=int(checkpoint["update"]),
                    world_size=world_size,
                    scheduled_gradient_panel_updates=(
                        scheduled_gradient_panel_updates
                    ),
                    allow_schedule_extension=extension_enabled,
                )
                _restore_rng_state(rank_rng_records[rank], device)
            expected_resume_update = _update_for_completed_epoch(
                epoch=start_epoch,
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
                logical_batch_schedule=logical_batch_schedule,
            )
            if int(checkpoint["update"]) != expected_resume_update:
                raise ValueError("Resume update is not at an epoch boundary")
            if extension_enabled and start_epoch < extension_start_epoch:
                raise ValueError("Resume checkpoint predates the extension phase")
            if args.resume_into_fresh_attempt:
                expected_history_epochs = {
                    epoch for epoch in eval_epochs if epoch <= start_epoch
                }
                if set(inherited_evaluations) != expected_history_epochs:
                    raise ValueError(
                        "Fresh-attempt checkpoint evaluation history is incomplete"
                    )
                parent_checkpoint_paths = sorted(
                    args.resume.resolve(strict=True).parent.glob("epoch_*.pt")
                )
                observed_parent_epochs = {
                    int(path.stem.removeprefix("epoch_"))
                    for path in parent_checkpoint_paths
                }
                expected_checkpoint_epochs = (
                    expected_history_epochs
                    if sampling_mode == V2_SAMPLING_MODE
                    else expected_history_epochs - {0}
                )
                if observed_parent_epochs != expected_checkpoint_epochs:
                    raise ValueError(
                        "Fresh-attempt parent checkpoint set is incomplete or has later files"
                    )
                if args.resume.resolve(strict=True) != max(
                    parent_checkpoint_paths,
                    key=lambda path: int(path.stem.removeprefix("epoch_")),
                ):
                    raise ValueError(
                        "Fresh-attempt resume must use the latest parent checkpoint"
                    )
                lineage_parent_checkpoints = [
                    {
                        "epoch": int(path.stem.removeprefix("epoch_")),
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for path in parent_checkpoint_paths
                ]
                resolved["lineage_parent_checkpoints"] = (
                    lineage_parent_checkpoints
                )
        if hierarchy_enabled:
            model.hierarchy_mask_loss_weight = float(
                args.hierarchy_mask_loss_weight
            ) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "hierarchy_mask_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
            model.hierarchy_vicreg_loss_weight = float(
                args.hierarchy_vicreg_loss_weight
            ) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "hierarchy_vicreg_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
        elif args.multiscale_supervision:
            model.multiscale_loss_weight = float(args.multiscale_loss_weight) * (
                0.5
                if gradient_conflict_state["objectives"][
                    "legacy_auxiliary_scaled"
                ]["coefficient_halved"]
                else 1.0
            )
        dist.barrier()
        if rank == 0:
            # A failed restart must leave the previous provenance untouched.
            # All checkpoint/config validation and strict state loading above
            # therefore precede this atomic publication.
            if int(output_exists) == 0:
                args.output_dir.mkdir(parents=True, exist_ok=False)
                (args.output_dir / "checkpoints").mkdir()
            if args.resume_into_fresh_attempt:
                for parent in lineage_parent_checkpoints:
                    source_checkpoint = Path(parent["path"])
                    destination_checkpoint = (
                        args.output_dir / "checkpoints" / source_checkpoint.name
                    )
                    with source_checkpoint.open("rb") as source, (
                        destination_checkpoint.open("xb")
                    ) as destination:
                        shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if sha256_file(destination_checkpoint) != parent["sha256"]:
                        raise RuntimeError(
                            "Copied parent checkpoint hash drifted"
                        )
                inherited_rows = [
                    {
                        "kind": "eval",
                        "epoch": epoch,
                        "update": _update_for_completed_epoch(
                            epoch=epoch,
                            scene_visits_per_epoch_per_rank=(
                                scene_visits_per_epoch_per_rank
                            ),
                            scenes_per_rank_per_update=(
                                scenes_per_rank_per_update
                            ),
                            logical_batch_schedule=logical_batch_schedule,
                        ),
                        **inherited_evaluations[epoch],
                    }
                    for epoch in sorted(inherited_evaluations)
                ]
                _atomic_jsonl_noclobber(metrics_path, inherited_rows)
            elif int(output_exists) == 0:
                metrics_path.unlink(missing_ok=True)
            _atomic_json(args.output_dir / "resolved_config.json", resolved)
        dist.barrier()

        evaluations: dict[int, dict[str, Any]] = dict(inherited_evaluations)

        def evaluate_and_record(epoch: int, update: int) -> dict[str, Any]:
            evaluation: dict[str, Any] | None = None
            dist.barrier()
            if rank == 0:
                evaluation = evaluate_holdout(
                    model=model,
                    scenes=holdout_scenes,
                    device=device,
                    sampling=evaluation_sampling,
                    seed=args.fixed_eval_seed,
                    fixed_eval_pack=args.fixed_eval_pack,
                    fixed_eval_pack_report=fixed_eval_pack_report,
                    allow_shared_proposal_negatives=args.allow_shared_proposal_negatives,
                    cells=source_cells,
                )
                if sampling_mode == V2_SAMPLING_MODE:
                    token_cosine_gap = _native_token_pure_fixed_pack_cosine_gap(
                        evaluation, expected_cells=source_cells
                    )
                    qualifies = token_cosine_gap > float(
                        args.feature_hard_gap_threshold
                    )
                    feature_hard_state["qualifying_panel_streak"] = (
                        int(feature_hard_state["qualifying_panel_streak"]) + 1
                        if qualifies
                        else 0
                    )
                    feature_hard_state["panel_count"] = (
                        int(feature_hard_state["panel_count"]) + 1
                    )
                    feature_hard_state["last_panel_update"] = int(update)
                    feature_hard_state["last_panel_cosine_gap"] = token_cosine_gap
                    feature_hard_state[
                        "last_dense_point_diagnostic_cosine_gap"
                    ] = None
                    feature_hard_state[
                        "dense_point_holdout_eligible_for_unlock"
                    ] = False
                    feature_hard_state[
                        "native_token_metric_eligible_for_unlock"
                    ] = True
                    evaluation["feature_hard_gate"] = dict(feature_hard_state)
                evaluations[epoch] = evaluation
                _append_jsonl(
                    metrics_path,
                    {
                        "kind": "eval",
                        "epoch": epoch,
                        "update": update,
                        **evaluation,
                    },
                )
                native_macro = evaluation["native_selection_macro"]
                log.info(
                    "eval epoch=%d update=%d native_macro=%.4f 2d=%s 3d=%s "
                    "effrank=%.2f min_std=%.6f",
                    epoch,
                    update,
                    native_macro["overall"],
                    "n/a"
                    if "2d" not in native_macro
                    else f"{native_macro['2d']:.4f}",
                    "n/a"
                    if "3d" not in native_macro
                    else f"{native_macro['3d']:.4f}",
                    evaluation["native_health"]["effective_rank"],
                    evaluation["native_health"]["feature_std_min"],
                )
            payload: list[dict[str, Any] | None] = [evaluation]
            dist.broadcast_object_list(payload, src=0)
            evaluation = payload[0]
            if evaluation is None:
                raise RuntimeError("Evaluation broadcast failed")
            if sampling_mode == V2_SAMPLING_MODE:
                received_gate = evaluation.get("feature_hard_gate")
                if not isinstance(received_gate, dict):
                    raise RuntimeError("V2 evaluation lacks feature-hard gate state")
                feature_hard_state.clear()
                feature_hard_state.update(received_gate)
            evaluations[epoch] = evaluation
            dist.barrier()
            return evaluation

        def save_checkpoint(
            *,
            epoch_label: int,
            update_value: int,
            evaluation: dict[str, Any] | None,
            resumable: bool,
        ) -> Path:
            """Collect rank-local replay state and atomically retain one raw state."""

            gathered_counts: list[dict[str, int] | None] = [None] * world_size
            gathered_coverage: list[dict[str, Any] | None] = [None] * world_size
            gathered_sampling_totals: list[dict[str, Any] | None] = [
                None
            ] * world_size
            gathered_rng: list[dict[str, Any] | None] = [None] * world_size
            gathered_cumulative_health: list[dict[str, Any] | None] = [
                None
            ] * world_size
            dist.all_gather_object(gathered_counts, rank_exposure_counts)
            dist.all_gather_object(gathered_coverage, rank_coverage_states)
            dist.all_gather_object(
                gathered_sampling_totals, rank_sampling_totals
            )
            dist.all_gather_object(gathered_rng, _capture_rng_state(device))
            dist.all_gather_object(
                gathered_cumulative_health, rank_cumulative_health
            )
            if resumable:
                filename = f"epoch_{int(epoch_label):04d}.pt"
                checkpoint_kind = "resumable_epoch_boundary"
            else:
                filename = (
                    f"audit_e{int(epoch_label):04d}_u{int(update_value):08d}.pt"
                )
                checkpoint_kind = "nonresumable_update_aligned_audit_snapshot"
            path = args.output_dir / "checkpoints" / filename
            if rank == 0:
                typed_counts = [
                    {key: int(value) for key, value in row.items()}
                    for row in gathered_counts
                    if row is not None
                ]
                typed_coverage = [
                    dict(row) for row in gathered_coverage if row is not None
                ]
                typed_rng = [dict(row) for row in gathered_rng if row is not None]
                typed_sampling_totals = [
                    dict(row)
                    for row in gathered_sampling_totals
                    if row is not None
                ]
                typed_cumulative_health = [
                    dict(row)
                    for row in gathered_cumulative_health
                    if row is not None
                ]
                if (
                    len(typed_counts) != world_size
                    or len(typed_coverage) != world_size
                    or len(typed_rng) != world_size
                    or len(typed_sampling_totals) != world_size
                ):
                    raise RuntimeError("Incomplete rank state at checkpoint")
                cumulative_health = None
                if rank_cumulative_health is not None:
                    if len(typed_cumulative_health) != world_size:
                        raise RuntimeError(
                            "Incomplete cumulative health at checkpoint"
                        )
                    cumulative_health = _merge_rank_cumulative_health(
                        typed_cumulative_health,
                        through_update=int(update_value),
                        world_size=world_size,
                        gradient_conflict_controller=(
                            gradient_conflict_state
                        ),
                    )
                payload = _checkpoint_payload(
                    epoch=int(epoch_label),
                    nominal_epoch=int(epoch_label),
                    update=int(update_value),
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    resolved=resolved,
                    evaluation=evaluation,
                    evaluation_history=evaluations,
                    rank_exposure_counts=typed_counts,
                    rank_coverage_states=typed_coverage,
                    rank_sampling_totals=typed_sampling_totals,
                    rank_rng_states=typed_rng,
                    feature_hard_state=dict(feature_hard_state),
                    gradient_conflict_state=json.loads(
                        json.dumps(gradient_conflict_state)
                    ),
                    cumulative_health=cumulative_health,
                    checkpoint_kind=checkpoint_kind,
                )
                _atomic_torch_save(payload, path)
            dist.barrier()
            return path

        if extension_bootstrap and start_epoch in eval_epochs:
            inherited_update = _update_for_completed_epoch(
                epoch=start_epoch,
                scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
                scenes_per_rank_per_update=scenes_per_rank_per_update,
                logical_batch_schedule=logical_batch_schedule,
            )
            if rank == 0:
                evaluations[start_epoch] = inherited_evaluation
                _append_jsonl(
                    metrics_path,
                    {
                        "kind": "eval",
                        "epoch": start_epoch,
                        "update": inherited_update,
                        "inherited_from_parent": True,
                        **inherited_evaluation,
                    },
                )
                log.info(
                    "inherited eval epoch=%d update=%d native_macro=%.4f",
                    start_epoch,
                    inherited_update,
                    inherited_evaluation["native_selection_macro"]["overall"],
                )
            dist.barrier()
        elif start_epoch == 0 and 0 in eval_epochs:
            initial_evaluation = evaluate_and_record(0, 0)
            if 0 in checkpoint_epochs:
                save_checkpoint(
                    epoch_label=0,
                    update_value=0,
                    evaluation=initial_evaluation,
                    resumable=True,
                )

        model.train()
        started = time.perf_counter()
        update = _update_for_completed_epoch(
            epoch=start_epoch,
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
            logical_batch_schedule=logical_batch_schedule,
        )
        pending_visits: list[tuple[int, int]] = []
        stop_training = False
        training_scene_count = (
            len(local_scenes) if smoke else len(local_scene_ids)
        )
        for epoch in range(start_epoch, epochs):
            if stop_training:
                break
            order = epoch_scene_order(
                scene_count=training_scene_count, visits=scene_visits_per_epoch_per_rank,
                rank=rank, epoch=epoch, seed=args.seed, policy=args.rank_balance_policy,
            )
            if len(order) > training_scene_count:
                padded_scene = (local_scenes[order[-1]].scene_id if smoke else local_scene_ids[order[-1]])
                log.info("rank=%d epoch=%d rotating_padding_scene=%s", rank, epoch, padded_scene)
            for scene_position, scene_index in enumerate(order):
                if update >= execution_stop_update:
                    stop_training = True
                    break
                pending_visits.append((epoch, int(scene_index)))
                at_epoch_boundary = scene_position + 1 == len(order)
                flush_boundary = (logical_batch_schedule is not None
                                  and at_epoch_boundary and epoch + 1 in eval_epochs)
                if len(pending_visits) < scenes_per_rank_per_update and not flush_boundary:
                    if at_epoch_boundary and epoch + 1 in eval_epochs:
                        raise RuntimeError(
                            "Evaluation boundary contains an incomplete logical batch"
                        )
                    continue
                group_size = len(pending_visits)
                expected_group_size = (logical_batch_schedule.batch_size(update + 1)
                                       if logical_batch_schedule is not None else scenes_per_rank_per_update)
                if group_size != expected_group_size:
                    raise AssertionError("Logical scene group cardinality drift")

                update += 1
                update_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                factor = (
                    extension_lr_factor(
                        update=update,
                        start_update=extension_start_update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                    )
                    if extension_enabled
                    else lr_factor(
                        update=update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                    )
                )
                lr_values = set_optimizer_lr(optimizer, factor=factor)
                optimizer.zero_grad(set_to_none=True)
                if (
                    sampling_mode == V2_SAMPLING_MODE
                    and args.ablation_variant == "full"
                    and not bool(feature_hard_state["enabled"])
                    and bool(
                        feature_hard_state.get(
                            "native_token_metric_eligible_for_unlock", False
                        )
                    )
                    and int(feature_hard_state["qualifying_panel_streak"])
                    >= int(args.feature_hard_required_panels)
                    and update
                    >= max(
                        1,
                        int(
                            math.ceil(
                                total_updates
                                * float(args.feature_hard_warmup_fraction)
                            )
                        ),
                    )
                ):
                    feature_hard_state["enabled"] = True
                    feature_hard_state["activation_update"] = int(update)
                if (args.ablation_variant != "full" and not feature_hard_state["enabled"]
                        and update >= args.ablation_activation_update):
                    feature_hard_state.update(enabled=True, activation_update=int(update),
                                              activation_policy="fixed_to_reference_actual_activation")
                current_ablation_budget = (
                    NegativeBudget(int(sampling.uniform_negatives), int(sampling.spatial_hard_negatives),
                                   int(sampling.feature_hard_negatives) if feature_hard_state["enabled"] else 0)
                    if args.ablation_variant == "full" else
                    negative_budget(args.ablation_variant, update=update,
                                    activation_update=args.ablation_activation_update)
                )
                scene_metric_sums = [0.0 for _ in range(6)]
                multiscale_metric_sums: dict[str, dict[str, float]] = {
                    name: {
                        "loss": 0.0,
                        "hardest_negative_ranking_accuracy": 0.0,
                        "cosine_gap": 0.0,
                        "temperature": 0.0,
                        "input_pairs": 0.0,
                        "input_proposals": 0.0,
                        "kept_pairs": 0.0,
                        "kept_proposals": 0.0,
                        "collapsed_positive_pairs": 0.0,
                        "dropped_proposals_insufficient_pure_positive": 0.0,
                        "dropped_proposals_no_pure_negative": 0.0,
                        "pure_positive_tokens": 0.0,
                        "pure_negative_tokens": 0.0,
                        "mixed_or_unknown_tokens_ignored": 0.0,
                        "accepted_mixed_tokens": 0.0,
                        "zero_negative_pairs": 0.0,
                        "base_negatives_per_pair": 0.0,
                        "uniform_negatives_per_pair": 0.0,
                        "spatial_hard_negatives_per_pair": 0.0,
                        "feature_candidate_pool": 0.0,
                        "feature_hard_negatives": 0.0,
                        "strict_token_purity": 0.0,
                        "stage_weight": 0.0,
                        "effective_loss_weight": 0.0,
                    }
                    for name in MULTISCALE_LEVEL_NAMES
                }
                final_loss_sum = 0.0
                auxiliary_loss_sum = 0.0
                multiscale_scale_sum = 0.0
                sampling_audit_records: list[dict[str, Any]] = []
                hierarchy_records: list[dict[str, Any]] = []
                token_pure_records: list[dict[str, Any]] = []
                legacy_multiscale_records: list[dict[str, Any]] = []
                gradient_telemetry_records: list[dict[str, Any]] = []
                shared_parameters = [
                    parameter
                    for parameter in model.backbone.parameters()
                    if parameter.requires_grad
                ]
                if not shared_parameters:
                    raise RuntimeError("LitePT shared-backbone parameter set is empty")
                shared_parameter_ids = {id(parameter) for parameter in shared_parameters}
                private_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                    and id(parameter) not in shared_parameter_ids
                ]
                projected_pcgrad_objectives = {
                    objective
                    for objective, state in gradient_conflict_state[
                        "objectives"
                    ].items()
                    if bool(state["pcgrad_enabled"])
                }
                pcgrad_active = bool(
                    (hierarchy_enabled or args.multiscale_supervision)
                    and projected_pcgrad_objectives
                )
                collect_global_gradient_panel = bool(
                    (hierarchy_enabled or args.multiscale_supervision)
                    and update in scheduled_gradient_panel_updates
                )
                manual_gradient_mode = bool(
                    collect_global_gradient_panel or pcgrad_active
                )
                shared_gradient_accumulator: dict[str, torch.Tensor] | None = None
                private_gradient_accumulator: dict[str, torch.Tensor] | None = None
                shared_gradient_report: dict[str, Any] | None = None
                globally_valid_microbatches = 0
                globally_invalid_microbatches = 0
                v2_gradients_pending_sync = False
                group_epochs = [visit_epoch for visit_epoch, _ in pending_visits]
                data_wait_started = time.perf_counter()
                if source_loading_mode == "logical_batch":
                    if source_loader_pool is None:
                        raise RuntimeError("Logical source-loader pool is missing")
                    group_scenes = _load_logical_scene_group(
                        scene_indices=[index for _, index in pending_visits],
                        sam2_maximum_raw_points=args.sam2_maximum_raw_points,
                        scene_ids=local_scene_ids,
                        source_root=args.source_root,
                        expected_manifest_hashes=expected_manifest_hashes,
                        input_features=args.input_features,
                        coordinate_normalization_policy=(
                            args.coordinate_normalization_policy
                        ),
                        color_mean=color_mean,
                        color_std=color_std,
                        executor=source_loader_pool,
                    )
                else:
                    group_scenes = [
                        local_scenes[index] for _, index in pending_visits
                    ]
                data_wait_seconds = time.perf_counter() - data_wait_started
                for microbatch_index, (
                    (visit_epoch, visit_scene_index),
                    scene,
                ) in enumerate(zip(pending_visits, group_scenes, strict=True)):
                    if (
                        source_loading_mode == "logical_batch"
                        and scene.scene_id != local_scene_ids[visit_scene_index]
                    ):
                        raise RuntimeError("Lazy scene-group ordering drift")
                    scene = scene_source_for_visit(
                        scene,
                        epoch=visit_epoch,
                        global_seed=args.seed,
                    )
                    visit_input = build_scene_visit_input(
                        scene=scene,
                        epoch=visit_epoch,
                        global_seed=args.seed,
                        input_features=args.input_features,
                        augmentation_profile=train_augmentation_profile,
                        color_mean=color_mean,
                        color_std=color_std,
                    )
                    points = torch.from_numpy(visit_input.points).to(device)
                    features = torch.from_numpy(visit_input.features).to(device)
                    frame_groups: Sequence[Any] | None = None
                    plan = None
                    if sampling_mode == V2_SAMPLING_MODE:
                        # Each fixed geometry catalog retains its own cursor.
                        # Rotating fragments use catalog-visit epochs, while
                        # geometry augmentation keeps the global visit epoch.
                        sampling_visit = sampling_visit_for_scene(
                            scene, global_epoch=visit_epoch, seed=args.seed,
                            cells=source_cells, saved_states=rank_coverage_states,
                        )
                        try:
                            plan = sample_multigranular_frame_group_plan(
                                scene_id=scene.scene_id,
                                epoch=sampling_visit.epoch,
                                seed=args.seed,
                                points=points,
                                frames_by_cell=_v2_frame_catalogs(
                                    scene=scene,
                                    device=device,
                                    cells=tuple(source_cells),
                                    proposal_area_cutoff=args.proposal_area_cutoff,
                                ),
                                total_proposal_quota=int(
                                    sampling.proposals_per_forward
                                ),
                                cells=tuple(source_cells),
                                positive_pairs_per_proposal=int(
                                    sampling.positive_pairs_per_proposal
                                ),
                                num_uniform_negatives=int(
                                    current_ablation_budget.uniform
                                ),
                                num_spatial_hard_negatives=int(
                                    current_ablation_budget.spatial
                                ),
                                spatial_candidate_pool=int(
                                    sampling.spatial_candidate_pool
                                ),
                                num_feature_hard_negatives=int(current_ablation_budget.feature),
                                feature_candidate_pool=int(
                                    sampling.feature_candidate_pool
                                ),
                                require_negative_proposal_membership=True,
                                allow_shared_proposal_negatives=train_allow_shared,
                                require_hierarchy_routes=not single_granularity,
                                grid_size=args.grid_size,
                                operation=sampling_visit.operation,
                                coverage_states=sampling_visit.states_by_cell,
                            )
                        except NoRetainedFrameGroupsError as error:
                            # Expected data-quality exhaustion is represented by
                            # the complete plan and synchronized before any rank
                            # can enter SyncBN/DDP/VICReg collectives.
                            plan = error.plan
                        frame_groups = plan.groups
                        batch = None
                        sampling_audit_records.append(
                            {
                                **plan.audit_metadata(),
                                "source_crop_id": scene.crop_id,
                                "source_variant_id": scene.source_variant_id,
                                "global_visit_epoch": visit_epoch,
                                "catalog_visit_epoch": sampling_visit.epoch,
                                "coverage_keys": sampling_visit.keys_by_cell,
                                "groups": [
                                    {
                                        "cell": group.cell,
                                        "frame_id": group.frame_id,
                                        "selected_proposal_count": len(
                                            group.selected_proposal_indices
                                        ),
                                        "sampler": dict(
                                            group.coverage_metadata["sampler"]
                                        ),
                                    }
                                    for group in plan.groups
                                ],
                            }
                        )
                        for sampled_cell, coverage in plan.coverage_by_cell.items():
                            rank_exposure_counts[sampled_cell] += int(
                                coverage["realized_count"]
                            )
                            cumulative = rank_sampling_totals[sampled_cell]
                            cumulative["requested_proposals"] += int(
                                coverage["requested_count"]
                            )
                            cumulative["realized_unique_proposals"] += int(
                                coverage["realized_count"]
                            )
                            cumulative[
                                "contrastive_retained_proposals"
                            ] += int(coverage["contrastive_retained_count"])
                            cumulative["duplicate_proposals"] += int(
                                coverage["duplicate_count"]
                            )
                            state_after = coverage.get("state_after")
                            rank_coverage_states[
                                sampling_visit.keys_by_cell[sampled_cell]
                            ] = state_after
                        for group in plan.groups:
                            sampler_report = group.coverage_metadata["sampler"]
                            cumulative = rank_sampling_totals[group.cell]
                            cumulative["realized_positive_pairs"] += int(
                                sampler_report["realized_positive_pairs"]
                            )
                            cumulative["requested_positive_pairs"] += int(
                                sampler_report[
                                    "requested_positive_pairs_per_proposal"
                                ]
                            ) * len(sampler_report["retained_proposal_indices"])
                        local_sampler_valid = (
                            bool(plan.groups)
                            and {group.cell for group in plan.groups}
                            == set(source_cells)
                        )
                        sampler_valid = torch.tensor(
                            int(local_sampler_valid),
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(sampler_valid, op=dist.ReduceOp.MIN)
                        if int(sampler_valid.item()) != 1:
                            globally_invalid_microbatches += 1
                            for sampled_cell in source_cells:
                                rank_sampling_totals[sampled_cell][
                                    "invalid_scene_visits"
                                ] += 1
                            sampling_audit_records[-1][
                                "optimization_valid"
                            ] = False
                            sampling_audit_records[-1][
                                "optimization_drop_reason"
                            ] = "missing_required_frame_local_cell_on_at_least_one_rank"
                            del visit_input, points, features, batch, plan
                            continue
                        sampling_audit_records[-1]["optimization_valid"] = True
                    else:
                        cell, batch = training_batch(
                            scene=scene,
                            epoch=visit_epoch,
                            points=points,
                            device=device,
                            sampling=sampling,
                            sampling_mode=sampling_mode,
                            repeat_g08_for_all_3d_slots=(
                                args.three_d_mode == "g08_repeated"
                            ),
                            cells=source_cells,
                            cell_schedule=args.source_cell_schedule,
                            schedule_seed=args.seed,
                        )
                    forward_seed = stable_seed(
                        args.seed,
                        rank,
                        scene.scene_id,
                        visit_epoch,
                        "train-forward",
                    )
                    random.seed(forward_seed)
                    np.random.seed(forward_seed % (2**32))
                    torch.manual_seed(forward_seed)
                    is_final_microbatch = (
                        microbatch_index + 1 == group_size
                    )
                    # V2 presents six sequential scene forwards as one logical
                    # update.  Keep the first five backward passes local so
                    # the final DDP backward reduces their accumulated buckets
                    # together with the sixth scene.  Manual telemetry/PCGrad
                    # updates intentionally use the raw model and their own
                    # exact global-gradient reductions.
                    sync_context = (
                        nullcontext()
                        if manual_gradient_mode or is_final_microbatch
                        else ddp_model.no_sync()
                    )
                    deferred_v2_sync = bool(
                        sampling_mode == V2_SAMPLING_MODE
                        and not manual_gradient_mode
                        and not is_final_microbatch
                    )
                    with sync_context:
                        if (
                            manual_gradient_mode
                            and args.normalization_policy == "batchnorm"
                        ):
                            for buffer in model.buffers():
                                dist.broadcast(buffer, src=0)
                        forward_model = model if manual_gradient_mode else ddp_model
                        result = forward_model(
                            points,
                            features,
                            batch,
                            multiscale_loss_scale=(
                                model.multiscale_scale_for_epoch(visit_epoch)
                            ),
                            frame_groups=frame_groups,
                            supervision_step=update,
                            supervision_total_steps=total_updates,
                            return_loss_components=(
                                collect_global_gradient_panel or pcgrad_active
                            ),
                            return_multiscale_proposal_diagnostics=bool(
                                args.multiscale_supervision
                            ),
                        )
                        loss = result["loss_total"]
                        finite = torch.tensor(
                            int(bool(torch.isfinite(loss))),
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                        if int(finite.item()) != 1:
                            raise FloatingPointError(
                                "Non-finite loss at "
                                f"epoch={visit_epoch} update={update} "
                                f"microbatch={microbatch_index}"
                            )
                        if "hierarchy_supervision" in result:
                            hierarchy_records.append(result["hierarchy_supervision"])
                        if "token_pure_dec0" in result:
                            token_pure_records.append(result["token_pure_dec0"])
                        if sampling_mode == V2_SAMPLING_MODE:
                            token_health = result.get("token_pure_dec0")
                            local_token_valid = bool(
                                result.get("optimization_valid", False)
                                and
                                isinstance(token_health, Mapping)
                                and token_health.get("optimization_valid", False)
                                and set(token_health.get("retained_cells", ()))
                                == set(source_cells)
                            )
                            hierarchy_health = result.get("hierarchy_supervision")
                            if hierarchy_enabled:
                                local_token_valid = bool(
                                    local_token_valid
                                    and isinstance(hierarchy_health, Mapping)
                                    and hierarchy_health.get(
                                        "optimization_valid", False
                                    )
                                )
                            token_valid = torch.tensor(
                                int(local_token_valid),
                                dtype=torch.int32,
                                device=device,
                            )
                            dist.all_reduce(token_valid, op=dist.ReduceOp.MIN)
                            optimization_valid = int(token_valid.item()) == 1
                        else:
                            optimization_valid = True
                        if not optimization_valid:
                            globally_invalid_microbatches += 1
                            if sampling_audit_records:
                                sampling_audit_records[-1][
                                    "optimization_valid"
                                ] = False
                                sampling_audit_records[-1][
                                    "optimization_drop_reason"
                                ] = (
                                    "post_voxel_token_or_hierarchy_starvation_"
                                    "on_at_least_one_rank"
                                )
                            for sampled_cell in source_cells:
                                rank_sampling_totals[sampled_cell][
                                    "invalid_scene_visits"
                                ] += 1
                            # Every rank drains the identical DDP graph.  The
                            # model's V2 zero-yield contract touches disposable
                            # heads, so find_unused_parameters=False remains safe.
                            if not manual_gradient_mode:
                                (loss * 0.0).backward()
                                if sampling_mode == V2_SAMPLING_MODE:
                                    v2_gradients_pending_sync = deferred_v2_sync
                            del visit_input, points, features, batch, result, loss
                            if plan is not None:
                                del plan
                            continue

                        globally_valid_microbatches += 1
                        if args.multiscale_supervision:
                            multiscale_levels = result.get("multiscale")
                            if not isinstance(multiscale_levels, Mapping) or set(
                                multiscale_levels
                            ) != set(LEGACY_MULTISCALE_CUMULATIVE_STAGES):
                                raise RuntimeError(
                                    "Legacy multiscale result stage set drift"
                                )
                            legacy_levels: dict[str, dict[str, Any]] = {}
                            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
                                stage_metrics = multiscale_levels[stage]
                                if not isinstance(stage_metrics, Mapping) or not set(
                                    LEGACY_MULTISCALE_CUMULATIVE_KEYS
                                ).issubset(stage_metrics):
                                    raise RuntimeError(
                                        "Legacy multiscale result metric set drift"
                                    )
                                # Keep only the canonical cumulative metrics;
                                # proposal-level diagnostics are intentionally
                                # not copied into every update record.
                                legacy_levels[stage] = {
                                    key: stage_metrics[key]
                                    for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
                                }
                            legacy_multiscale_records.append(
                                {
                                    "levels": _json_copy(legacy_levels)
                                }
                            )
                        if collect_global_gradient_panel:
                            loss_components = result.get(
                                "loss_component_tensors"
                            )
                            junctions = result.get("hierarchy_junction_tensors")
                            if not isinstance(loss_components, dict) or not isinstance(
                                junctions, dict
                            ):
                                raise RuntimeError(
                                    "Supervised V2 result lacks gradient telemetry tensors"
                                )
                            if hierarchy_enabled:
                                selected_components = {
                                    key: loss_components[key]
                                    for key in (
                                        "final_dec0",
                                        "hierarchy_mask_scaled",
                                        "hierarchy_vicreg_scaled",
                                    )
                                }
                            else:
                                selected_components = {
                                    key: loss_components[key]
                                    for key in (
                                        "final_dec0",
                                        "legacy_auxiliary_scaled",
                                    )
                                }
                            gradient_telemetry_records.append(
                                hierarchy_gradient_telemetry(
                                    loss_components=selected_components,
                                    junctions=junctions,
                                )
                            )
                        if manual_gradient_mode:
                            manual_components = result.get("loss_component_tensors")
                            if not isinstance(manual_components, dict):
                                raise RuntimeError(
                                    "Shared-gradient mode lacks objective components"
                                )
                            component_names = (
                                (
                                    "final_dec0",
                                    "hierarchy_mask_scaled",
                                    "hierarchy_vicreg_scaled",
                                )
                                if hierarchy_enabled
                                else (
                                    "final_dec0",
                                    "legacy_auxiliary_scaled",
                                )
                            )
                            selected_manual_components = {
                                key: manual_components[key]
                                for key in component_names
                            }
                            shared_gradient_accumulator = (
                                _add_flat_objective_gradients(
                                    shared_gradient_accumulator,
                                    _flat_objective_gradients(
                                    components=selected_manual_components,
                                    parameters=shared_parameters,
                                    scale=float(group_size),
                                ),
                            )
                            )
                            if private_parameters:
                                private_gradient_accumulator = (
                                    _add_flat_objective_gradients(
                                        private_gradient_accumulator,
                                        _flat_objective_gradients(
                                            components={"total": loss},
                                            parameters=private_parameters,
                                            scale=float(
                                                group_size
                                            ),
                                        ),
                                    )
                                )
                        else:
                            (loss / group_size).backward()
                            if sampling_mode == V2_SAMPLING_MODE:
                                v2_gradients_pending_sync = deferred_v2_sync
                    for metric_index, value in enumerate(
                        (
                            float(loss.detach()),
                            float(result["triplet_ranking_accuracy"]),
                            float(result["hardest_negative_ranking_accuracy"]),
                            float(result["positive_cosine_mean"]),
                            float(result["negative_cosine_mean"]),
                            float(result["cosine_gap"]),
                        )
                    ):
                        scene_metric_sums[metric_index] += value
                    final_loss_sum += float(result.get("loss_final_dec0", 0.0))
                    auxiliary_loss_sum += float(result.get("loss_multiscale", 0.0))
                    multiscale_scale_sum += float(
                        result.get("multiscale_loss_scale", 0.0)
                    )
                    if sampling_mode == V2_SAMPLING_MODE:
                        token_health = result["token_pure_dec0"]
                        per_cell_pairs: dict[str, int] = {
                            sampled_cell: 0 for sampled_cell in source_cells
                        }
                        per_cell_requested: dict[str, int] = {
                            sampled_cell: 0 for sampled_cell in source_cells
                        }
                        for group_record in token_health.get("groups", ()):
                            sampled_cell = str(group_record["cell"])
                            per_cell_pairs[sampled_cell] += int(
                                group_record.get("realized_positive_pairs", 0)
                            )
                            proposal_records = group_record.get("proposals", ())
                            requested_token_pairs = sum(
                                int(proposal.get("requested_token_pairs", 0))
                                for proposal in proposal_records
                            )
                            per_cell_requested[sampled_cell] += (
                                requested_token_pairs
                                if proposal_records
                                else int(group_record.get("input_point_pairs", 0))
                            )
                        mass = int(round(1_000_000 / len(source_cells)))
                        for sampled_cell in source_cells:
                            cumulative = rank_sampling_totals[sampled_cell]
                            cumulative["active_scene_visits"] += 1
                            cumulative["optimized_scene_visits"] += 1
                            cumulative["loss_mass_millionths"] += mass
                            cumulative[
                                "optimized_loss_mass_millionths"
                            ] += mass
                            cumulative[
                                "token_realized_positive_pairs"
                            ] += per_cell_pairs[sampled_cell]
                            cumulative[
                                "token_requested_positive_pairs"
                            ] += per_cell_requested[sampled_cell]
                    for name in MULTISCALE_LEVEL_NAMES:
                        level_metrics = result.get("multiscale", {}).get(name, {})
                        for key in multiscale_metric_sums[name]:
                            multiscale_metric_sums[name][key] += float(
                                level_metrics.get(key, 0.0)
                            )
                    if sampling_mode != V2_SAMPLING_MODE:
                        rank_exposure_counts[cell] += sampling.proposals_per_forward
                    del visit_input, points, features, batch, result, loss
                    if plan is not None:
                        del plan

                if source_loading_mode == "logical_batch":
                    del scene, group_scenes

                if (
                    sampling_mode == V2_SAMPLING_MODE
                    and not manual_gradient_mode
                    and (
                        v2_gradients_pending_sync
                        or globally_valid_microbatches <= 0
                    )
                ):
                    _synchronize_accumulated_gradients(
                        parameters=model.parameters(),
                        world_size=world_size,
                    )
                if globally_valid_microbatches <= 0:
                    raise RuntimeError(
                        "V2 logical update retained zero globally valid microbatches"
                    )
                if sampling_mode == V2_SAMPLING_MODE:
                    valid_rescale = (
                        float(group_size)
                        / float(globally_valid_microbatches)
                    )
                    if not manual_gradient_mode:
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(valid_rescale)
                    if shared_gradient_accumulator is not None:
                        for vector in shared_gradient_accumulator.values():
                            vector.mul_(valid_rescale)
                    if private_gradient_accumulator is not None:
                        for vector in private_gradient_accumulator.values():
                            vector.mul_(valid_rescale)
                if manual_gradient_mode:
                    if shared_gradient_accumulator is None:
                        raise RuntimeError(
                            "Shared-gradient logical batch accumulated no gradients"
                        )
                    shared_gradient_report = _global_shared_gradient_report(
                        accumulated=shared_gradient_accumulator,
                        shared_parameters=shared_parameters,
                        projected_objectives=projected_pcgrad_objectives,
                        world_size=world_size,
                        install_projected_gradient=True,
                    )
                    if private_parameters:
                        if (
                            private_gradient_accumulator is None
                            or set(private_gradient_accumulator) != {"total"}
                        ):
                            raise RuntimeError(
                                "Private logical-batch gradient is incomplete"
                            )
                        _install_global_flat_total_gradient(
                            accumulated=private_gradient_accumulator["total"],
                            parameters=private_parameters,
                            world_size=world_size,
                        )

                gradient = _clip_gradient_diagnostics(
                    model.parameters(),
                    max_norm=float(args.grad_clip_norm),
                )
                gradient_status = torch.tensor(
                    [
                        int(
                            all(
                                math.isfinite(float(gradient[key]))
                                for key in (
                                    "pre_clip_norm",
                                    "post_clip_norm",
                                    "clip_return_norm",
                                    "clip_coefficient",
                                )
                            )
                        ),
                        int(bool(gradient["was_clipped"])),
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                # One SUM replaces the former finite-only MIN collective.  It
                # also supplies the exact any-rank/rank-event clipping counts
                # without adding a synchronization to the hot path.
                dist.all_reduce(gradient_status, op=dist.ReduceOp.SUM)
                if int(gradient_status[0].item()) != world_size:
                    raise FloatingPointError(
                        f"Non-finite gradient at epoch={epoch} update={update}"
                    )
                clipped_rank_count = int(gradient_status[1].item())
                optimizer.step()
                if rank_cumulative_health is not None:
                    _accumulate_rank_cumulative_health_update(
                        rank_cumulative_health,
                        update=update,
                        local_was_clipped=bool(gradient["was_clipped"]),
                        clipped_rank_count=clipped_rank_count,
                        world_size=world_size,
                        attempted_microbatch_count=(
                            group_size
                        ),
                        optimization_valid_count=(
                            globally_valid_microbatches
                        ),
                        token_records=token_pure_records,
                        hierarchy_records=hierarchy_records,
                        legacy_multiscale_records=legacy_multiscale_records,
                    )
                update_seconds = time.perf_counter() - update_started

                if gradient_telemetry_records:
                    gathered_gradient_telemetry: list[
                        list[dict[str, Any]] | None
                    ] = [None for _ in range(world_size)]
                    dist.all_gather_object(
                        gathered_gradient_telemetry,
                        gradient_telemetry_records,
                    )
                    if rank == 0:
                        flattened_gradient_telemetry = [
                            record
                            for rank_records in gathered_gradient_telemetry
                            if rank_records is not None
                            for record in rank_records
                        ]
                        conflict_actions: list[dict[str, Any]] = []
                        if shared_gradient_report is None:
                            raise RuntimeError(
                                "Gradient panel lacks global shared-backbone report"
                            )
                        if hierarchy_enabled or args.multiscale_supervision:
                            for objective, state in gradient_conflict_state[
                                "objectives"
                            ].items():
                                objective_report = shared_gradient_report[
                                    "objectives"
                                ].get(objective)
                                if not isinstance(objective_report, Mapping):
                                    raise RuntimeError(
                                        f"Missing global shared gradient for {objective}"
                                    )
                                cosine_key = (
                                    "global_shared_cosine_after"
                                    if bool(state["pcgrad_enabled"])
                                    else "global_shared_cosine_before"
                                )
                                cosine = float(objective_report[cosine_key])
                                state["last_cosine"] = cosine
                                state["last_cosine_kind"] = cosine_key
                                state["panel_streak"] = (
                                    int(state["panel_streak"]) + 1
                                    if cosine
                                    < float(gradient_conflict_state["threshold"])
                                    else 0
                                )
                                if int(state["panel_streak"]) < int(
                                    gradient_conflict_state[
                                        "required_consecutive_panels"
                                    ]
                                ):
                                    continue
                                if not bool(state["coefficient_halved"]):
                                    state["coefficient_halved"] = True
                                    action = "halve_coefficient"
                                elif not bool(state["pcgrad_enabled"]):
                                    state["pcgrad_enabled"] = True
                                    action = "enable_pcgrad"
                                else:
                                    state["post_pcgrad_failure"] = True
                                    gradient_conflict_state["rejected"] = True
                                    gradient_conflict_state[
                                        "rejection_reason"
                                    ] = (
                                        "persistent_global_shared_conflict_"
                                        f"after_pcgrad:{objective}"
                                    )
                                    action = "reject_candidate"
                                state["panel_streak"] = 0
                                state["last_action_update"] = int(update)
                                conflict_actions.append(
                                    {
                                        "objective": objective,
                                        "cosine": cosine,
                                        "action": action,
                                    }
                                )
                            budget_report = shared_gradient_report[
                                "auxiliary_budget"
                            ]
                            budget_ratio = float(
                                budget_report[
                                    "global_sum_aux_norm_ratio_before"
                                ]
                            )
                            budget_state = gradient_conflict_state[
                                "auxiliary_budget"
                            ]
                            budget_state[
                                "last_global_sum_aux_norm_ratio"
                            ] = budget_ratio
                            budget_state["passed"] = bool(
                                budget_report["passed_before"]
                            )
                            if not bool(budget_report["passed_before"]):
                                budget_state["violation_count"] = (
                                    int(budget_state["violation_count"]) + 1
                                )
                                gradient_conflict_state["rejected"] = True
                                gradient_conflict_state[
                                    "rejection_reason"
                                ] = "global_shared_auxiliary_gradient_budget_exceeded"
                                conflict_actions.append(
                                    {
                                        "objective": "combined_auxiliary",
                                        "norm_ratio": budget_ratio,
                                        "limit": float(budget_report["limit"]),
                                        "action": "reject_candidate",
                                    }
                                )
                        gradient_health_row = {
                            "kind": "gradient_health",
                            "epoch": group_epochs[-1],
                            "update": update,
                            "diagnostic_local_junction_telemetry": _mean_numeric_tree(
                                flattened_gradient_telemetry
                            ),
                            "objective_stage_counts": (
                                _gradient_objective_stage_counts(
                                    flattened_gradient_telemetry
                                )
                            ),
                            "diagnostic_conservative_auxiliary_budget": _telemetry_auxiliary_budget(
                                flattened_gradient_telemetry
                            ),
                            "global_shared_gradient": shared_gradient_report,
                            "auxiliary_budget": shared_gradient_report[
                                "auxiliary_budget"
                            ],
                            "auxiliary_budget_limit_vs_primary": 0.5,
                            "conflict_threshold": -0.05,
                            "controller": gradient_conflict_state,
                            "actions": conflict_actions,
                            "pcgrad": (
                                shared_gradient_report
                                if pcgrad_active
                                else None
                            ),
                        }
                        _append_jsonl(metrics_path, gradient_health_row)
                        if rank_cumulative_health is not None:
                            _append_rank_gradient_panel(
                                rank_cumulative_health,
                                gradient_health_row,
                            )
                    conflict_payload: list[dict[str, Any] | None] = [
                        gradient_conflict_state if rank == 0 else None
                    ]
                    dist.broadcast_object_list(conflict_payload, src=0)
                    received_conflict_state = conflict_payload[0]
                    if not isinstance(received_conflict_state, dict):
                        raise RuntimeError("Gradient conflict-state broadcast failed")
                    received_conflict_state = json.loads(
                        json.dumps(received_conflict_state)
                    )
                    gradient_conflict_state.clear()
                    gradient_conflict_state.update(received_conflict_state)
                    if hierarchy_enabled:
                        model.hierarchy_mask_loss_weight = float(
                            args.hierarchy_mask_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "hierarchy_mask_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )
                        model.hierarchy_vicreg_loss_weight = float(
                            args.hierarchy_vicreg_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "hierarchy_vicreg_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )
                    elif args.multiscale_supervision:
                        model.multiscale_loss_weight = float(
                            args.multiscale_loss_weight
                        ) * (
                            0.5
                            if gradient_conflict_state["objectives"][
                                "legacy_auxiliary_scaled"
                            ]["coefficient_halved"]
                            else 1.0
                        )

                if (
                    update == 1
                    or update % int(args.log_every) == 0
                    or update in snapshot_label_by_update
                    or (at_epoch_boundary and epoch + 1 in eval_epochs)
                ):
                    metric_denominator = max(globally_valid_microbatches, 1)
                    scene_metric_means = [
                        value / metric_denominator
                        for value in scene_metric_sums
                    ]
                    values = torch.tensor(
                        [
                            scene_metric_means[0],
                            scene_metric_means[1],
                            scene_metric_means[2],
                            float(gradient["pre_clip_norm"]),
                            float(gradient["post_clip_norm"]),
                            float(gradient["clip_return_norm"]),
                            float(gradient["clip_coefficient"]),
                            float(gradient["was_clipped"]),
                            scene_metric_means[3],
                            scene_metric_means[4],
                            scene_metric_means[5],
                            final_loss_sum / metric_denominator,
                            auxiliary_loss_sum / metric_denominator,
                            multiscale_scale_sum / metric_denominator,
                            *[
                                multiscale_metric_sums[name][key]
                                / metric_denominator
                                for name in MULTISCALE_LEVEL_NAMES
                                for key in multiscale_metric_sums[name]
                            ],
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    values /= world_size
                    gathered_sampling: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    gathered_hierarchy: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    gathered_token_pure: list[list[dict[str, Any]] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(
                        gathered_sampling, sampling_audit_records
                    )
                    dist.all_gather_object(gathered_hierarchy, hierarchy_records)
                    dist.all_gather_object(gathered_token_pure, token_pure_records)
                    local_resource = {
                        "rank": rank,
                        "device": local_device_report,
                        "update_seconds": update_seconds,
                        "data_wait_seconds": data_wait_seconds,
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                    gathered_resources: list[dict[str, Any] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(gathered_resources, local_resource)
                    if rank == 0:
                        row = {
                            "kind": "train",
                            "epoch": group_epochs[-1],
                            "scene_epoch_start": group_epochs[0],
                            "scene_epoch_end": group_epochs[-1],
                            "update": update,
                            "loss_total": float(values[0]),
                            "triplet_ranking_accuracy": float(values[1]),
                            "hardest_negative_ranking_accuracy": float(values[2]),
                            "gradient_norm": float(values[3]),
                            "gradient_norm_pre_clip": float(values[3]),
                            "gradient_norm_post_clip": float(values[4]),
                            "clip_return_norm": float(values[5]),
                            "clip_coefficient": float(values[6]),
                            "gradient_clipped_rank_fraction": float(values[7]),
                            "positive_cosine_mean": float(values[8]),
                            "negative_cosine_mean": float(values[9]),
                            "cosine_gap": float(values[10]),
                            "lr": lr_values,
                            "temperature": float(model.criterion.temperature.detach()),
                            "sampling_profile": args.sampling_profile,
                            "train_augmentation_profile": train_augmentation_profile,
                            "required_crop_mode": required_crop_mode,
                            "scenes_per_rank_per_update": group_size,
                            "nominal_scenes_per_rank_per_update": scenes_per_rank_per_update,
                            "scenes_per_global_update": (
                                group_size * world_size
                            ),
                            "globally_valid_scenes_per_rank": int(
                                globally_valid_microbatches
                            ),
                            "globally_invalid_scenes_per_rank": int(
                                globally_invalid_microbatches
                            ),
                            "proposals_per_scene": int(
                                sampling.proposals_per_forward
                            ),
                            "proposals_per_global_update": (
                                group_size
                                * world_size
                                * int(sampling.proposals_per_forward)
                            ),
                            "negatives_per_pair": int(sampling.uniform_negatives)
                            + int(sampling.spatial_hard_negatives)
                            + int(sampling.feature_hard_negatives),
                            "wall_seconds": time.perf_counter() - started,
                            "normalization_policy": str(
                                args.normalization_policy
                            ),
                            "feature_hard_gate": dict(feature_hard_state),
                            "resource_health": {
                                "ranks": [
                                    resource
                                    for resource in gathered_resources
                                    if resource is not None
                                ],
                                "rank_step_time_imbalance_fraction": (
                                    max(
                                        float(resource["update_seconds"])
                                        for resource in gathered_resources
                                        if resource is not None
                                    )
                                    / max(
                                        min(
                                            float(resource["update_seconds"])
                                            for resource in gathered_resources
                                            if resource is not None
                                        ),
                                        1.0e-12,
                                    )
                                    - 1.0
                                ),
                                "maximum_reserved_memory_fraction": max(
                                    float(resource["peak_reserved_bytes"])
                                    / float(resource["device"]["total_memory_bytes"])
                                    for resource in gathered_resources
                                    if resource is not None
                                ),
                            },
                        }
                        if sampling_mode == V2_SAMPLING_MODE:
                            flattened_sampling = [
                                record
                                for rank_records in gathered_sampling
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["sampling_health"] = _summarize_v2_sampling(
                                flattened_sampling,
                                cells=source_cells,
                                allow_shared_proposal_negatives=train_allow_shared,
                            )
                            row["sampling_health"]["feature_hard_enabled"] = current_ablation_budget.feature > 0
                            if sampling_ablation:
                                row["ablation_negative_budget"] = {
                                    "uniform": current_ablation_budget.uniform,
                                    "spatial": current_ablation_budget.spatial,
                                    "feature": current_ablation_budget.feature,
                                    "requested_total": current_ablation_budget.total,
                                }
                            flattened_token_pure = [
                                record
                                for rank_records in gathered_token_pure
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["token_pure_dec0"] = (
                                _summarize_token_pure_records(
                                    flattened_token_pure,
                                    cells=source_cells,
                                )
                            )
                        if hierarchy_enabled:
                            flattened_hierarchy = [
                                record
                                for rank_records in gathered_hierarchy
                                if rank_records is not None
                                for record in rank_records
                            ]
                            row["hierarchy_supervision"] = (
                                _summarize_hierarchy_records(
                                    flattened_hierarchy,
                                    cells=source_cells,
                                )
                            )
                        row["multiscale"] = {
                            "loss_final_dec0": float(values[11]),
                            "loss_auxiliary": float(values[12]),
                            "loss_scale": float(values[13]),
                            "levels": {},
                        }
                        value_index = 14
                        for name in MULTISCALE_LEVEL_NAMES:
                            row["multiscale"]["levels"][name] = {}
                            for key in multiscale_metric_sums[name]:
                                row["multiscale"]["levels"][name][key] = float(
                                    values[value_index]
                                )
                                value_index += 1
                        _append_jsonl(metrics_path, row)
                        log.info(
                            "epoch=%d update=%d loss=%.4f hard=%.4f lr_bb=%.3e",
                            group_epochs[-1],
                            update,
                            row["loss_total"],
                            row["hardest_negative_ranking_accuracy"],
                            lr_values["backbone/decay"],
                        )
                pending_visits = []

                if bool(gradient_conflict_state["rejected"]):
                    if rank == 0:
                        _atomic_json(
                            args.output_dir / "candidate_rejected.json",
                            {
                                "schema_version": (
                                    "litept_pretraining_candidate_rejection/v2"
                                ),
                                "reason": gradient_conflict_state.get(
                                    "rejection_reason"
                                )
                                or "persistent_post_pcgrad_conflict",
                                "epoch": int(group_epochs[-1]),
                                "update": int(update),
                                "controller": gradient_conflict_state,
                                "checkpoint_published_for_rejected_update": False,
                                "passed": False,
                            },
                        )
                    dist.barrier()
                    raise RuntimeError(
                        "Candidate rejected before evaluation/checkpoint publication: "
                        f"{gradient_conflict_state.get('rejection_reason')}"
                    )

                if update in snapshot_label_by_update:
                    snapshot_label = int(snapshot_label_by_update[update])
                    snapshot_evaluation = evaluate_and_record(
                        snapshot_label, update
                    )
                    if update in checkpoint_updates:
                        save_checkpoint(
                            epoch_label=snapshot_label,
                            update_value=update,
                            evaluation=snapshot_evaluation,
                            resumable=False,
                        )

                if at_epoch_boundary:
                    completed_epoch = epoch + 1
                    completed_update = _update_for_completed_epoch(
                        epoch=completed_epoch,
                        scene_visits_per_epoch_per_rank=(
                            scene_visits_per_epoch_per_rank
                        ),
                        scenes_per_rank_per_update=scenes_per_rank_per_update,
                        logical_batch_schedule=logical_batch_schedule,
                    )
                    if update != completed_update:
                        raise RuntimeError(
                            "Logical update count drift at epoch boundary: "
                            f"observed={update} expected={completed_update}"
                        )
                    evaluation = None
                    if completed_epoch in eval_epochs:
                        if completed_update in snapshot_label_by_update:
                            if int(snapshot_label_by_update[completed_update]) != int(
                                completed_epoch
                            ):
                                raise RuntimeError(
                                    "Snapshot label conflicts with epoch boundary"
                                )
                            evaluation = evaluations[completed_epoch]
                        else:
                            evaluation = evaluate_and_record(
                                completed_epoch, completed_update
                            )
                    if completed_epoch in checkpoint_epochs:
                        save_checkpoint(
                            epoch_label=completed_epoch,
                            update_value=completed_update,
                            evaluation=evaluation,
                            resumable=True,
                        )
                if update >= execution_stop_update:
                    stop_training = True
                    break

        if pending_visits:
            raise RuntimeError(
                "Complete run ended with an incomplete logical scene batch"
            )
        if execution_stop_update < total_updates:
            if update != execution_stop_update:
                raise RuntimeError(
                    "Operational pause update drift: "
                    f"{update} != {execution_stop_update}"
                )
            pause_epoch = (
                visits_through_update(execution_stop_update)
            ) // scene_visits_per_epoch_per_rank
            pause_checkpoint = (
                args.output_dir / "checkpoints" / f"epoch_{pause_epoch:04d}.pt"
            )
            pause_checkpoint_exists = torch.tensor(
                int(pause_checkpoint.is_file()) if rank == 0 else 0,
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(pause_checkpoint_exists, src=0)
            if int(pause_checkpoint_exists.item()) != 1:
                raise RuntimeError(
                    "Operational production pause lacks its resumable epoch "
                    f"checkpoint: {pause_checkpoint}"
                )
            if rank == 0:
                _atomic_json(
                    args.output_dir / "run_paused.json",
                    {
                        "schema_version": "litept_pretraining_operational_pause/v1",
                        "experiment_id": EXPERIMENT_ID,
                        "epoch": int(pause_epoch),
                        "update": int(update),
                        "checkpoint_path": str(pause_checkpoint.resolve()),
                        "checkpoint_sha256": sha256_file(pause_checkpoint),
                        "declared_total_updates": int(total_updates),
                        "lr_schedule_continues_without_restart": True,
                        "candidate_rejected": bool(
                            gradient_conflict_state.get("rejected", False)
                        ),
                        "passed": not bool(
                            gradient_conflict_state.get("rejected", False)
                        ),
                    },
                )
            dist.barrier()
            return
        if update != total_updates:
            raise RuntimeError(
                f"Final logical update count drift: {update} != {total_updates}"
            )

        gathered_counts_final: list[dict[str, int] | None] = [
            None for _ in range(world_size)
        ]
        gathered_sampling_final: list[dict[str, Any] | None] = [
            None for _ in range(world_size)
        ]
        gathered_cumulative_health_final: list[dict[str, Any] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered_counts_final, rank_exposure_counts)
        dist.all_gather_object(gathered_sampling_final, rank_sampling_totals)
        dist.all_gather_object(
            gathered_cumulative_health_final, rank_cumulative_health
        )
        if rank == 0:
            typed_final = [
                {key: int(value) for key, value in row.items()}
                for row in gathered_counts_final
                if row is not None
            ]
            global_counts = _merge_count_dicts(typed_final)
            typed_sampling_final = [
                dict(row) for row in gathered_sampling_final if row is not None
            ]
            typed_cumulative_health_final = [
                dict(row)
                for row in gathered_cumulative_health_final
                if row is not None
            ]
            cumulative_health_final = None
            if rank_cumulative_health is not None:
                if len(typed_cumulative_health_final) != world_size:
                    raise RuntimeError("Incomplete final cumulative health")
                cumulative_health_final = _merge_rank_cumulative_health(
                    typed_cumulative_health_final,
                    through_update=total_updates,
                    world_size=world_size,
                    gradient_conflict_controller=gradient_conflict_state,
                )
            final_eval_label, final_checkpoint_path = _final_evaluation_endpoint(
                nominal_epochs=epochs,
                total_updates=total_updates,
                maximum_updates=args.maximum_updates,
                snapshot_label_by_update=snapshot_label_by_update,
                checkpoint_epochs=checkpoint_epochs,
                checkpoint_updates=checkpoint_updates,
                checkpoint_dir=args.output_dir / "checkpoints",
            )
            final_eval = evaluations.get(final_eval_label)
            if final_eval is None:
                checkpoint = torch.load(
                    final_checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    int(checkpoint.get("epoch", -1)) != final_eval_label
                    or int(checkpoint.get("update", -1)) != total_updates
                ):
                    raise RuntimeError(
                        "Final checkpoint boundary disagrees with the exact run endpoint"
                    )
                final_eval = checkpoint["evaluation"]
            checks = {
                "world_size_exact": world_size == expected_world_size,
                "final_update_exact": (
                    total_scene_visits_per_rank
                    == visits_through_update(total_updates)
                    and (args.maximum_updates is not None
                         or total_scene_visits_per_rank == nominal_scene_visits_per_rank)
                ),
                "logical_batch_exact": (
                    resolved["scenes_per_rank_per_update"]
                    == scenes_per_rank_per_update
                    and resolved["scenes_per_global_update"]
                    == scenes_per_rank_per_update * world_size
                    and resolved["proposals_per_global_update"]
                    == (
                        scenes_per_rank_per_update
                        * world_size
                        * int(sampling.proposals_per_forward)
                    )
                ),
                "input_feature_contract_exact": (
                    resolved["input_features"] == args.input_features
                    and resolved["input_channels"]
                    == (3 if args.input_features == "rgb3" else 6)
                    and initial_report["input_features"] == args.input_features
                ),
                "representative_sampling_contract_exact": (
                    resolved["representative_sampling"]
                    == args.representative_sampling
                    and (
                        train_augmentation_profile
                        not in {
                            TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
                            TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
                        }
                        or args.representative_sampling == "first"
                    )
                ),
                "rgbn6_color_contract_exact": (
                    train_augmentation_profile
                    not in {
                        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
                        TRAIN_AUGMENTATION_MASK3D_RGBN6_SPHERE50K,
                    }
                    or (
                        resolved["rgb_color_normalization"]
                        == RGBN6_COLOR_NORMALIZATION
                        and resolved["rgb_color_mean"]
                        == list(COMMON_RGB_COLOR_MEAN)
                        and resolved["rgb_color_std"]
                        == list(COMMON_RGB_COLOR_STD)
                    )
                ),
                "pretraining_recipe_exact": (
                    resolved["pretraining_recipe"] == model.recipe_name
                    and resolved["multiscale_supervision"]
                    == bool(args.multiscale_supervision)
                    and resolved["multiscale_levels"]
                    == list(MULTISCALE_LEVEL_NAMES)
                    and resolved["multiscale_level_loss_weights"]
                    == dict(MULTISCALE_LEVEL_LOSS_WEIGHTS)
                    and resolved["multiscale_loss_weight"]
                    == float(args.multiscale_loss_weight)
                    and resolved["multiscale_warmup_epochs"]
                    == int(args.multiscale_warmup_epochs)
                    and resolved["multiscale_token_policy"]
                    == (
                        "strict_pure_token_resampling_v1"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_temperature_policy"]
                    == (
                        "separate_learnable_temperature_per_auxiliary_level"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_feature_hard_policy"]
                    == (
                        "separate_candidate_selection_per_positive_endpoint"
                        if args.multiscale_supervision
                        else None
                    )
                    and resolved["multiscale_transfer_contract"]
                    == (
                        "native_mask3d_fpn_required"
                        if args.multiscale_supervision or hierarchy_enabled
                        else "final_dec0_feature"
                    )
                    and resolved["sampling_mode"] == sampling_mode
                    and resolved["token_pure_dec0"]
                    == (sampling_mode == V2_SAMPLING_MODE)
                    and resolved["hierarchy_supervision"]["mode"]
                    == hierarchy_mode
                    and resolved["normalization"]["policy"]
                    == str(args.normalization_policy)
                ),
                "train_augmentation_contract_exact": (
                    resolved["train_augmentation_profile"]
                    == train_augmentation_profile
                    and resolved["train_augmentation"]
                    == augmentation_contract_for_profile(
                        train_augmentation_profile
                    )
                    and resolved["evaluation_augmentation_profile"] == "none"
                ),
                "source_crop_mode_exact": (
                    resolved["required_crop_mode"] == required_crop_mode
                    and (
                        required_crop_mode == "any"
                        or resolved["loaded_crop_modes_rank0"]
                        == [required_crop_mode]
                    )
                ),
                "source_loading_contract_exact": (
                    resolved["source_loading_mode"] == source_loading_mode
                    and resolved["source_loader_workers"]
                    == source_loader_workers
                ),
                "all_metrics_finite": all(
                    math.isfinite(float(value))
                    for value in final_eval["native_selection_macro"].values()
                ),
                "initial_checkpoint_exact": (
                    initial_report["sha256"]
                    == resolved["initialization"]["sha256"]
                ),
                "initialization_seed_exact": (
                    initial_report["seed"] == int(args.initialization_seed)
                    and resolved["initialization_seed"]
                    == int(args.initialization_seed)
                ),
                "fixed_eval_seed_exact": (
                    fixed_eval_pack_report is None
                    or (
                        int(fixed_eval_pack_report.get("seed", -1))
                        == int(args.fixed_eval_seed)
                        and resolved["fixed_eval_seed"]
                        == int(args.fixed_eval_seed)
                    )
                ),
                "source_cells_exact": set(global_counts) == set(source_cells),
                "scene_voxelization_cache_disabled": (
                    resolved["cache_training_voxelization"] is False
                    and resolved["voxelization_mode"]
                    == "per_forward_single_scene_recompute"
                ),
                "sampling_profile_exact": (
                    resolved["sampling_profile"] == args.sampling_profile
                    and resolved["sampling"] == vars(sampling)
                ),
                "negative_policy_exact": (
                    resolved.get("allow_shared_proposal_negatives", False) == args.allow_shared_proposal_negatives
                    and resolved.get("negative_policy", negative_policy_name(False)) == negative_policy_name(args.allow_shared_proposal_negatives)
                    and (fixed_eval_pack_report is None or
                         fixed_eval_pack_report.get("allow_shared_proposal_negatives", False)
                         == args.allow_shared_proposal_negatives)
                ),
                "evaluation_sampling_exact": (
                    resolved["evaluation_sampling_profile"]
                    == evaluation_sampling_profile
                    and resolved["evaluation_sampling"]
                    == vars(evaluation_sampling)
                ),
                "source_cell_schedule_exact": (
                    resolved["source_cell_schedule"]
                    == args.source_cell_schedule
                    and resolved["source_cell_schedule_seed"] == int(args.seed)
                ),
                "code_manifest_pinned": bool(
                    code_manifest_report
                    and sha256_file(Path(code_manifest_report["path"]))
                    == code_manifest_report["sha256"]
                )
                if (
                    args.sampling_profile == "pf_hnm_768_v1"
                    or args.multiscale_supervision
                    or sampling_mode == V2_SAMPLING_MODE
                )
                else True,
            }
            if args.multiscale_supervision:
                legacy_cumulative = (
                    cumulative_health_final.get("legacy_multiscale_delivery")
                    if isinstance(cumulative_health_final, Mapping)
                    else None
                )
                expected_legacy_microbatches = (
                    total_scene_visits_per_rank * world_size
                )
                checks["legacy_multiscale_cumulative_exact"] = bool(
                    isinstance(legacy_cumulative, Mapping)
                    and legacy_cumulative.get("enabled") is True
                    and legacy_cumulative.get("expected_microbatch_count")
                    == expected_legacy_microbatches
                    and legacy_cumulative.get("microbatch_count")
                    == expected_legacy_microbatches
                    and legacy_cumulative.get("recorded_microbatch_count")
                    == legacy_cumulative.get("optimization_valid_count")
                    and legacy_cumulative.get("optimization_valid_count", 0)
                    + legacy_cumulative.get("optimization_invalid_count", 0)
                    == expected_legacy_microbatches
                    and set(legacy_cumulative.get("per_stage", {}))
                    == set(LEGACY_MULTISCALE_CUMULATIVE_STAGES)
                    and isinstance(cumulative_health_final, Mapping)
                    and cumulative_health_final["gradient_panels"][
                        "complete_through_update"
                    ] is True
                    and cumulative_health_final["gradient_panels"][
                        "expected_updates"
                    ]
                    == cumulative_health_final["gradient_panels"][
                        "observed_updates"
                    ]
                )
                diagnostics = final_eval.get("multiscale_diagnostics", {})
                checks["multiscale_holdout_diagnostics_valid"] = (
                    set(diagnostics) == set(MULTISCALE_LEVEL_NAMES)
                    and all(
                        int(diagnostics[name].get("attempted_batches", 0)) > 0
                        and int(diagnostics[name].get("valid_batches", 0)) > 0
                        and float(
                            diagnostics[name].get("valid_batch_fraction", 0.0)
                        )
                        > 0.0
                        and float(
                            diagnostics[name]
                            .get("mapping", {})
                            .get("accepted_mixed_tokens", -1.0)
                        )
                        == 0.0
                        and math.isclose(
                            float(
                                diagnostics[name]
                                .get("mapping", {})
                                .get("strict_token_purity", 0.0)
                            ),
                            1.0,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        and all(
                            math.isfinite(float(value))
                            for section in (
                                "conditional_metrics",
                                "native_health",
                                "projected_health",
                            )
                            for value in diagnostics[name]
                            .get(section, {})
                            .values()
                        )
                        for name in MULTISCALE_LEVEL_NAMES
                    )
                )
            if args.fixed_eval_pack is not None:
                checks["fixed_eval_pack_pinned"] = bool(
                    fixed_eval_pack_report
                    and fixed_eval_pack_report.get("passed")
                    and final_eval.get("selection_view") == "fixed_comparable"
                    and final_eval.get("fixed_eval_pack", {}).get(
                        "manifest_sha256"
                    )
                    == fixed_eval_pack_report["manifest_sha256"]
                )
                if sampling_mode == V2_SAMPLING_MODE:
                    checks["fixed_eval_pack_exact_token_relation_contract"] = bool(
                        isinstance(
                            fixed_eval_pack_report.get(
                                "token_pure_dec0_fixed_pack"
                            ),
                            Mapping,
                        )
                        and final_eval.get(
                            "native_fixed_comparable_dec0_macro_cosine_gap_valid"
                        )
                        is True
                    )
            elif args.sampling_profile == "pf_hnm_768_v1":
                checks["fixed_eval_pack_pinned"] = False
            if extension_enabled:
                checks["continuation_parent_pinned"] = (
                    resolved["continuation"]["enabled"] is True
                    and resolved["continuation"]["parent_checkpoint"]["epoch"]
                    == extension_start_epoch
                    and resolved["continuation"]["parent_checkpoint"]["update"]
                    == extension_start_update
                    and resolved["lr_schedule"]["kind"]
                    == "completed_checkpoint_cosine_restart"
                )
            if initial_report["public_checkpoint_used"]:
                checks["public_checkpoint_used_and_pinned"] = (
                    resolved["public_checkpoint_used"] is True
                    and initial_report["initialization_kind"]
                    == "public_litept_backbone"
                    and int(
                        initial_report["public_checkpoint"][
                            "loaded_backbone_keys"
                        ]
                    )
                    == 302
                )
            else:
                checks["public_checkpoint_unused"] = (
                    resolved["public_checkpoint_used"] is False
                )
            if sampling_mode == V2_SAMPLING_MODE:
                global_sampling_totals = {
                    cell: {
                        key: sum(
                            int(rank_totals[cell][key])
                            for rank_totals in typed_sampling_final
                        )
                        for key in rank_sampling_totals[cell]
                    }
                    for cell in source_cells
                }
                aggregate_sampling = {
                    key: sum(
                        global_sampling_totals[cell][key]
                        for cell in source_cells
                    )
                    for key in rank_sampling_totals[source_cells[0]]
                }
                loss_mass_total = aggregate_sampling[
                    "optimized_loss_mass_millionths"
                ]
                loss_shares = {
                    cell: global_sampling_totals[cell][
                        "optimized_loss_mass_millionths"
                    ]
                    / max(loss_mass_total, 1)
                    for cell in source_cells
                }
                checks["source_exposure_exact"] = (
                    global_counts
                    == {
                        cell: global_sampling_totals[cell][
                            "realized_unique_proposals"
                        ]
                        for cell in source_cells
                    }
                )
                checks["sampling_zero_duplicates"] = (
                    aggregate_sampling["duplicate_proposals"] == 0
                )
                checks["sampling_proposal_survival_at_least_90pct"] = (
                    aggregate_sampling["contrastive_retained_proposals"]
                    / max(aggregate_sampling["realized_unique_proposals"], 1)
                    >= 0.90
                )
                checks["sampling_pair_yield_at_least_85pct"] = (
                    aggregate_sampling["token_realized_positive_pairs"]
                    / max(
                        aggregate_sampling["token_requested_positive_pairs"],
                        1,
                    )
                    >= 0.85
                )
                checks["optimized_visit_fraction_at_least_95pct"] = (
                    aggregate_sampling["optimized_scene_visits"]
                    / max(
                        aggregate_sampling["optimized_scene_visits"]
                        + aggregate_sampling["invalid_scene_visits"],
                        1,
                    )
                    >= 0.95
                )
                checks[
                    "selected_cell_reached_optimized_token_loss" if single_granularity
                    else "all_three_cells_reached_optimized_token_loss"
                ] = all(
                    int(
                        global_sampling_totals[cell][
                            "token_realized_positive_pairs"
                        ]
                    )
                    > 0
                    and int(
                        global_sampling_totals[cell]["optimized_scene_visits"]
                    )
                    > 0
                    for cell in source_cells
                )
                checks["granularity_loss_share_within_5pp"] = all(
                    abs(float(loss_shares[cell]) - 1.0 / len(source_cells)) <= 0.05
                    for cell in source_cells
                )
                checks["global_shared_auxiliary_budget_passed"] = _global_auxiliary_budget_passed(
                    gradient_conflict_state,
                    auxiliary_enabled=bool(hierarchy_enabled or args.multiscale_supervision),
                )
                if cumulative_health_final is None:
                    raise RuntimeError("V2 final cumulative health is missing")
                expected_rank_microbatches = (
                    total_scene_visits_per_rank * world_size
                )
                cumulative_clipping = cumulative_health_final["clipping"]
                cumulative_panels = cumulative_health_final["gradient_panels"]
                cumulative_token = cumulative_health_final["token_delivery"]
                cumulative_hierarchy = cumulative_health_final[
                    "hierarchy_delivery"
                ]
                checks["cumulative_health_exact"] = bool(
                    cumulative_health_final["through_update"] == total_updates
                    and cumulative_health_final["optimizer_update_count"]
                    == total_updates
                    and cumulative_clipping["clipped_update_count"]
                    + cumulative_clipping["unclipped_update_count"]
                    == total_updates
                    and cumulative_clipping["rank_update_event_count"]
                    == total_updates * world_size
                    and cumulative_token["microbatch_count"]
                    == expected_rank_microbatches
                    and cumulative_token["optimization_valid_count"]
                    + cumulative_token["optimization_invalid_count"]
                    == expected_rank_microbatches
                    and (
                        not hierarchy_enabled
                        or (
                            cumulative_hierarchy["microbatch_count"]
                            == expected_rank_microbatches
                            and cumulative_hierarchy[
                                "optimization_valid_count"
                            ]
                            + cumulative_hierarchy[
                                "optimization_invalid_count"
                            ]
                            == expected_rank_microbatches
                            and cumulative_hierarchy["vicreg_valid_count"]
                            + cumulative_hierarchy["vicreg_invalid_count"]
                            == expected_rank_microbatches
                        )
                    )
                    and cumulative_panels["complete_through_update"] is True
                    and cumulative_panels["expected_updates"]
                    == cumulative_panels["observed_updates"]
                    and cumulative_health_final[
                        "gradient_conflict_controller"
                    ]
                    == gradient_conflict_state
                )
                expected_counts = {
                    "sampling_totals": global_sampling_totals,
                    "loss_shares": loss_shares,
                }
            elif not smoke:
                expected_counts = exposure_counts(
                    load_scene_ids(args.train_split),
                    epochs=epochs,
                    cells=source_cells,
                    proposals_per_forward=sampling.proposals_per_forward,
                    cell_schedule=args.source_cell_schedule,
                    seed=args.seed,
                )
                checks["source_exposure_exact"] = global_counts == expected_counts
            else:
                expected_counts = exposure_counts(
                    [
                        scene.scene_id
                        for _ in range(world_size)
                        for scene in smoke_scenes
                    ],
                    epochs=epochs,
                    cells=source_cells,
                    proposals_per_forward=sampling.proposals_per_forward,
                    cell_schedule=args.source_cell_schedule,
                    seed=args.seed,
                )
                checks["source_exposure_exact"] = global_counts == expected_counts
            summary = {
                "schema_version": (
                    "litept_rgbn6_multigranular_pretrain_summary/v2"
                    if sampling_mode == V2_SAMPLING_MODE
                    else "litept_gt3_unsam3_pretrain_summary/v1"
                ),
                "project_name": project_name(),
                "project_slug": project_slug(),
                "experiment_id": EXPERIMENT_ID,
                "dataset": args.dataset,
                "mode": (
                    f"{args.dataset}_smoke"
                    if smoke
                    else f"{args.dataset}_production"
                ),
                "epochs": epochs,
                "updates": total_updates,
                "seed": int(args.seed),
                "initialization_seed": int(args.initialization_seed),
                "fixed_eval_seed": int(args.fixed_eval_seed),
                "scene_visits_per_epoch_per_rank": (
                    scene_visits_per_epoch_per_rank
                ),
                "scenes_per_rank_per_update": scenes_per_rank_per_update,
                "scenes_per_global_update": scenes_per_rank_per_update * world_size,
                "backbone_lr": float(args.backbone_lr),
                "head_lr": float(args.head_lr),
                "pretraining_recipe": model.recipe_name,
                "sampling_mode": sampling_mode,
                "token_pure_dec0": bool(model.token_pure_dec0),
                "hierarchy_supervision": resolved["hierarchy_supervision"],
                "normalization": resolved["normalization"],
                **sampling_ablation,
                "feature_hard_gate": feature_hard_state,
                "cumulative_health": cumulative_health_final,
                "auxiliary_budget_applicable": bool(hierarchy_enabled or args.multiscale_supervision),
                "multiscale_supervision": bool(args.multiscale_supervision),
                "multiscale_levels": list(MULTISCALE_LEVEL_NAMES),
                "multiscale_level_loss_weights": dict(
                    MULTISCALE_LEVEL_LOSS_WEIGHTS
                ),
                "multiscale_loss_weight": float(args.multiscale_loss_weight),
                "multiscale_warmup_epochs": int(args.multiscale_warmup_epochs),
                "multiscale_token_policy": (
                    "strict_pure_token_resampling_v1"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_temperature_policy": (
                    "separate_learnable_temperature_per_auxiliary_level"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_feature_hard_policy": (
                    "separate_candidate_selection_per_positive_endpoint"
                    if args.multiscale_supervision
                    else None
                ),
                "multiscale_transfer_contract": (
                    "native_mask3d_fpn_required"
                    if args.multiscale_supervision or hierarchy_enabled
                    else "final_dec0_feature"
                ),
                "input_features": args.input_features,
                "input_channels": int(model.input_channels),
                "coordinate_normalization_policy": str(
                    args.coordinate_normalization_policy
                ),
                "train_augmentation_profile": train_augmentation_profile,
                "evaluation_augmentation_profile": "none",
                "required_crop_mode": required_crop_mode,
                "source_loading_mode": source_loading_mode,
                "source_loader_workers": source_loader_workers,
                "sampling_profile": args.sampling_profile,
                **({"allow_shared_proposal_negatives": args.allow_shared_proposal_negatives,
                    "negative_policy": negative_policy_name(args.allow_shared_proposal_negatives)}
                   if args.source_granularity == "sam2" else {}),
                "sampling": vars(sampling),
                "evaluation_sampling_profile": evaluation_sampling_profile,
                "evaluation_sampling": vars(evaluation_sampling),
                "evaluation_proposals_per_forward": int(
                    evaluation_sampling.proposals_per_forward
                ),
                "proposals_per_global_update": (
                    scenes_per_rank_per_update
                    * world_size
                    * int(sampling.proposals_per_forward)
                ),
                "source_cell_schedule": args.source_cell_schedule,
                "source_cell_schedule_seed": int(args.seed),
                "fixed_eval_pack": fixed_eval_pack_report,
                "code_manifest": code_manifest_report,
                "cache_training_voxelization": False,
                "voxelization_mode": "per_forward_single_scene_recompute",
                "lr_schedule": resolved["lr_schedule"],
                "continuation": resolved["continuation"],
                "initialization": initial_report,
                "public_checkpoint_used": bool(
                    initial_report["public_checkpoint_used"]
                ),
                "checks": checks,
                "mechanics_passed": all(checks.values()),
                "health_selection_status": (
                    "not_applicable_dec0_only"
                    if single_granularity
                    else "pending_external_five_stage_bn_recalibrated_gate"
                    if sampling_mode == V2_SAMPLING_MODE
                    else "not_applicable"
                ),
                "passed": (
                    None
                    if sampling_mode == V2_SAMPLING_MODE and not single_granularity
                    else all(checks.values())
                ),
                "global_exposure_counts": global_counts,
                "expected_exposure_counts": expected_counts,
                "final_evaluation": final_eval,
                "training_wall_seconds": time.perf_counter() - started,
                "final_checkpoint": str(final_checkpoint_path),
            }
            _atomic_json(args.output_dir / "run_summary.json", summary)
            if not summary["mechanics_passed"]:
                raise RuntimeError(f"Final run checks failed: {checks}")
        dist.barrier()
    finally:
        if source_loader_pool is not None:
            source_loader_pool.shutdown(wait=True, cancel_futures=True)
        dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _run()


if __name__ == "__main__":
    main()
