"""LitePT pretraining model and disposable supervision heads."""

from __future__ import annotations


import math


from pathlib import Path


from typing import Any, Mapping, Sequence


import torch

import torch.nn as nn

from delimit3d.training.contracts import TWO_D_CELLS, normalize_source_cells

from delimit3d.data.partfield_contrastive import ContrastiveTripletBatch


from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion

from delimit3d.losses.litept_hierarchy_supervision import HIERARCHY_COARSE_TO_FINE, HIERARCHY_QUERY_PROJECTION_POLICIES, MaskSupervisionGroup, ProposalConditionedHierarchyLoss, build_token_pure_dec0_contrastive_batch, hierarchy_point_maps, native_hierarchy_vicreg_loss

from delimit3d.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput


from .contrastive import (
    ContrastiveProjectionHead,
    _cell_balanced_contrastive_results,
    _token_pure_dec0_optimization_metrics,
    _zero_contrastive_result,
    build_pure_token_contrastive_batch,
    normalize_hierarchy_frame_groups,
    zero_loss_touching_module_parameters,
)
from .settings import (
    HIERARCHY_COVARIANCE_WEIGHT,
    HIERARCHY_MASK_LOSS_WEIGHT,
    HIERARCHY_RECIPE_NAME,
    HIERARCHY_VARIANCE_TARGET,
    HIERARCHY_VICREG_LOSS_WEIGHT,
    HIERARCHY_VICREG_MAX_TOKENS,
    HIERARCHY_WARMUP_FRACTION,
    MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
    MULTISCALE_AUXILIARY_WEIGHT,
    MULTISCALE_LEVEL_INDICES,
    MULTISCALE_LEVEL_LOSS_WEIGHTS,
    MULTISCALE_LEVEL_NAMES,
    MULTISCALE_RECIPE_NAME,
    TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL,
    hierarchy_mask_stage_weights_for_step,
    input_channels_for,
    normalize_hierarchy_mask_stage_weight_schedule,
    normalize_hierarchy_mask_stage_weights,
    weighted_hierarchy_mask_loss,
)


class LitePTContrastiveModel(nn.Module):
    def __init__(
        self,
        *,
        litept_root: Path,
        input_features: str = "rgbn6",
        litept_variant: str = "litept_s_star",
        representative_sampling: str = "first",
        token_pure_dec0: bool = False,
        token_pure_cells: Sequence[str] = TWO_D_CELLS,
        grid_size: float = 0.02,
        multiscale_supervision: bool = False,
        multiscale_loss_weight: float = MULTISCALE_AUXILIARY_WEIGHT,
        multiscale_warmup_epochs: int = MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
        hierarchy_supervision: bool = False,
        hierarchy_mask_loss_weight: float = HIERARCHY_MASK_LOSS_WEIGHT,
        hierarchy_mask_stage_weights: Mapping[str, float] | None = None,
        hierarchy_mask_stage_weight_schedule: Mapping[str, Any] | None = None,
        hierarchy_query_projection_policy: str = "shared_dec0",
        hierarchy_vicreg_loss_weight: float = HIERARCHY_VICREG_LOSS_WEIGHT,
        hierarchy_warmup_fraction: float = HIERARCHY_WARMUP_FRACTION,
        hierarchy_variance_target: float = HIERARCHY_VARIANCE_TARGET,
        hierarchy_covariance_weight: float = HIERARCHY_COVARIANCE_WEIGHT,
        hierarchy_vicreg_max_tokens: int = HIERARCHY_VICREG_MAX_TOKENS,
    ) -> None:
        super().__init__()
        self.input_features = str(input_features)
        self.input_channels = input_channels_for(self.input_features)
        self.token_pure_dec0 = bool(token_pure_dec0)
        self.token_pure_cells = normalize_source_cells(token_pure_cells)
        if self.token_pure_cells != TWO_D_CELLS and not (
            len(self.token_pure_cells) == 1 and self.token_pure_cells[0] in (*TWO_D_CELLS, "2d/sam2")
        ):
            raise ValueError("Token-pure dec0 expects one 2D cell or the original three")
        if len(self.token_pure_cells) == 1 and (multiscale_supervision or hierarchy_supervision):
            raise ValueError("Single-granularity supervision is dec0-only")
        if not math.isfinite(grid_size) or grid_size <= 0:
            raise ValueError("grid_size must be finite and positive")
        self.multiscale_supervision = bool(multiscale_supervision)
        self.multiscale_loss_weight = float(multiscale_loss_weight)
        self.multiscale_warmup_epochs = int(multiscale_warmup_epochs)
        self.hierarchy_supervision = bool(hierarchy_supervision)
        self.hierarchy_mask_loss_weight = float(hierarchy_mask_loss_weight)
        self.hierarchy_mask_stage_weights = (
            normalize_hierarchy_mask_stage_weights(
                hierarchy_mask_stage_weights
            )
        )
        self.hierarchy_mask_stage_weight_schedule = (
            normalize_hierarchy_mask_stage_weight_schedule(
                hierarchy_mask_stage_weight_schedule
            )
        )
        self.hierarchy_query_projection_policy = str(
            hierarchy_query_projection_policy
        )
        self.hierarchy_vicreg_loss_weight = float(hierarchy_vicreg_loss_weight)
        self.hierarchy_warmup_fraction = float(hierarchy_warmup_fraction)
        self.hierarchy_variance_target = float(hierarchy_variance_target)
        self.hierarchy_covariance_weight = float(hierarchy_covariance_weight)
        self.hierarchy_vicreg_max_tokens = int(hierarchy_vicreg_max_tokens)
        if self.multiscale_loss_weight < 0.0:
            raise ValueError("multiscale_loss_weight must be non-negative")
        if self.multiscale_warmup_epochs < 0:
            raise ValueError("multiscale_warmup_epochs must be non-negative")
        if self.multiscale_supervision and self.hierarchy_supervision:
            raise ValueError(
                "Legacy pure-token multiscale supervision and hierarchy "
                "supervision are mutually exclusive"
            )
        if self.hierarchy_mask_loss_weight < 0.0:
            raise ValueError("hierarchy_mask_loss_weight must be non-negative")
        if (
            self.hierarchy_query_projection_policy
            not in HIERARCHY_QUERY_PROJECTION_POLICIES
        ):
            raise ValueError(
                "Unknown hierarchy_query_projection_policy: "
                f"{self.hierarchy_query_projection_policy!r}"
            )
        if self.hierarchy_vicreg_loss_weight < 0.0:
            raise ValueError("hierarchy_vicreg_loss_weight must be non-negative")
        if not 0.0 <= self.hierarchy_warmup_fraction <= 1.0:
            raise ValueError("hierarchy_warmup_fraction must lie in [0,1]")
        if self.hierarchy_variance_target <= 0.0:
            raise ValueError("hierarchy_variance_target must be positive")
        if self.hierarchy_covariance_weight < 0.0:
            raise ValueError("hierarchy_covariance_weight must be non-negative")
        if self.hierarchy_vicreg_max_tokens < 2:
            raise ValueError("hierarchy_vicreg_max_tokens must be at least two")
        self.backbone = LitePTBackbone(
            litept_root=str(litept_root),
            in_channels=self.input_channels,
            grid_size=grid_size,
            litept_variant=litept_variant,
            multi_scale=self.multiscale_supervision,
            multi_scale_indices=(
                list(MULTISCALE_LEVEL_INDICES)
                if self.multiscale_supervision
                else None
            ),
            # Transfer contract shared with ScanNet LitePT -> Mask3D. This
            # deliberately replaces the wrapper's legacy mean reduction.
            voxel_reduce="representative",
            representative_sampling=representative_sampling,
            # Every V2 arm is judged on the same five native transfer stages.
            # Arm A keeps its legacy decoder-only auxiliary objective, but it
            # must still expose enc4 so final-dec0 and auxiliary gradient flow
            # through the shared backbone can be measured rather than marked
            # permanently unavailable by the evaluator.
            capture_hierarchy=(
                self.hierarchy_supervision or self.multiscale_supervision
            ),
            # Each update can use a different ScanNet scene. The wrapper's
            # legacy cache is only valid for repeated optimization of one scene.
            cache_training_voxelization=False,
        )
        self.projector = ContrastiveProjectionHead(72, 128, 128)
        self.criterion = PartFieldContrastiveCriterion(
            temperature=0.07,
            learnable_temperature=True,
            symmetric_feature_hard_mining=self.token_pure_dec0,
        )
        self.multiscale_projectors = nn.ModuleDict()
        self.multiscale_criteria = nn.ModuleDict()
        self.hierarchy_mask_supervisor: ProposalConditionedHierarchyLoss | None = None
        if self.multiscale_supervision:
            if set(MULTISCALE_LEVEL_LOSS_WEIGHTS) != set(MULTISCALE_LEVEL_NAMES):
                raise RuntimeError("Multi-scale stage-weight names drifted")
            if not math.isclose(
                sum(MULTISCALE_LEVEL_LOSS_WEIGHTS.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("Multi-scale stage weights must sum to one")
            level_channels = self.backbone.multi_scale_channels
            expected_channels = tuple(self.backbone.hierarchy_channels[name] for name in MULTISCALE_LEVEL_NAMES)
            if level_channels != list(expected_channels):
                raise RuntimeError(
                    "Unexpected LitePT-S* multi-scale channels: "
                    f"{level_channels!r} != {list(expected_channels)!r}"
                )
            self.multiscale_projectors.update(
                {
                    name: ContrastiveProjectionHead(channels, 128, 128)
                    for name, channels in zip(
                        MULTISCALE_LEVEL_NAMES,
                        expected_channels,
                        strict=True,
                    )
                }
            )
            # Keep dec0's learnable temperature exactly isolated from the
            # auxiliary distributions.  These level-specific temperatures,
            # like their projection heads, are disposable at transfer time.
            self.multiscale_criteria.update(
                {
                    name: PartFieldContrastiveCriterion(
                        temperature=0.07,
                        learnable_temperature=True,
                        symmetric_feature_hard_mining=True,
                    )
                    for name in MULTISCALE_LEVEL_NAMES
                }
            )
        if self.hierarchy_supervision:
            self.hierarchy_mask_supervisor = ProposalConditionedHierarchyLoss(
                stage_channels=self.backbone.hierarchy_channels,
                embedding_dim=128,
                temperature=0.07,
                query_projection_policy=(
                    self.hierarchy_query_projection_policy
                ),
            )

    @property
    def recipe_name(self) -> str:
        if self.hierarchy_supervision:
            return HIERARCHY_RECIPE_NAME
        if self.multiscale_supervision:
            return MULTISCALE_RECIPE_NAME
        return "final_dec0_only_v1"

    def multiscale_scale_for_epoch(self, epoch: int) -> float:
        """Return the scheduled auxiliary-loss multiplier for a scene visit."""
        if not self.multiscale_supervision:
            return 0.0
        if int(epoch) < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if self.multiscale_warmup_epochs == 0:
            ramp = 1.0
        else:
            ramp = min(1.0, float(epoch) / float(self.multiscale_warmup_epochs))
        return self.multiscale_loss_weight * ramp

    def hierarchy_ramp_for_step(
        self,
        *,
        supervision_step: int | None,
        supervision_total_steps: int | None,
        hierarchy_ramp_fraction: float | None,
    ) -> float:
        """Resolve an explicit ramp or a step-based first-5%-of-run ramp."""

        if not self.hierarchy_supervision:
            return 0.0
        if hierarchy_ramp_fraction is not None:
            value = float(hierarchy_ramp_fraction)
            if not 0.0 <= value <= 1.0:
                raise ValueError("hierarchy_ramp_fraction must lie in [0,1]")
            return value
        if supervision_step is None and supervision_total_steps is None:
            return 1.0
        if supervision_step is None or supervision_total_steps is None:
            raise ValueError(
                "supervision_step and supervision_total_steps must be supplied together"
            )
        step = int(supervision_step)
        total = int(supervision_total_steps)
        if step < 0 or total <= 0 or step > total:
            raise ValueError(
                f"Invalid hierarchy supervision progress step={step}, total={total}"
            )
        if self.hierarchy_warmup_fraction == 0.0:
            return 1.0
        ramp_steps = max(1, int(math.ceil(total * self.hierarchy_warmup_fraction)))
        return min(1.0, float(step) / float(ramp_steps))

    def _hierarchy_stage_features(
        self,
        output: LitePTBackboneOutput,
    ) -> dict[str, torch.Tensor]:
        expected = tuple(HIERARCHY_COARSE_TO_FINE)
        if output.hierarchy_stage_names != expected:
            raise RuntimeError(
                "LitePT hierarchy stage order drift: "
                f"{output.hierarchy_stage_names!r} != {expected!r}"
            )
        if len(output.hierarchy_tokens) != len(expected):
            raise RuntimeError("LitePT hierarchy token count drift")
        if len(output.hierarchy_parent_maps) != len(expected) - 1:
            raise RuntimeError("LitePT hierarchy parent-map count drift")
        stage_features = dict(
            zip(expected, output.hierarchy_tokens, strict=True)
        )
        for stage, expected_channels in self.backbone.hierarchy_channels.items():
            features = stage_features[stage]
            if features.ndim != 2 or int(features.shape[1]) != expected_channels:
                raise RuntimeError(
                    f"LitePT hierarchy {stage} shape drift: {tuple(features.shape)}"
                )
        return stage_features

    def _encode_backbone(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> LitePTBackboneOutput:
        output = self.backbone(points, features)
        if not isinstance(output, LitePTBackboneOutput):
            raise TypeError("Contrastive pretraining requires a dense LitePT output")
        return output

    def encode_with_multiscale(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        *,
        token_space: bool = False,
    ) -> tuple[
        LitePTBackboneOutput,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Encode the final feature and optional disposable level projections."""
        backbone_output = self._encode_backbone(points, features)
        native = backbone_output.point_feat
        if native.shape[0] != points.shape[0]:
            raise RuntimeError(
                "LitePT per-point feature cardinality mismatch: "
                f"input_points={points.shape[0]} output_features={native.shape[0]}"
            )
        projector_input = (
            backbone_output.scene_tokens if token_space else native
        )
        projected = self.projector(projector_input)
        level_embeddings: dict[str, torch.Tensor] = {}
        if self.multiscale_supervision:
            if len(backbone_output.multi_scale_tokens) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError("LitePT multi-scale output count drift")
            level_embeddings = {
                name: self.multiscale_projectors[name](tokens)
                for name, tokens in zip(
                    MULTISCALE_LEVEL_NAMES,
                    backbone_output.multi_scale_tokens,
                    strict=True,
                )
            }
        return backbone_output, native, projected, level_embeddings

    def encode(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, native, projected, _ = self.encode_with_multiscale(points, features)
        return native, projected

    def _legacy_multiscale_for_batch(
        self,
        *,
        backbone_output: LitePTBackboneOutput,
        level_embeddings_by_name: dict[str, torch.Tensor],
        batch: ContrastiveTripletBatch,
        return_proposal_diagnostics: bool,
    ) -> dict[str, Any]:
        """Evaluate the pre-existing pure-token objective for one group."""

        weighted_losses: list[torch.Tensor] = []
        unweighted_losses: dict[str, torch.Tensor] = {}
        stages: dict[str, dict[str, Any]] = {}
        for name, tokens, level_xyz, parent_map in zip(
            MULTISCALE_LEVEL_NAMES,
            backbone_output.multi_scale_tokens,
            backbone_output.multi_scale_xyz,
            backbone_output.finest_to_multi_scale,
            strict=True,
        ):
            level_embeddings = level_embeddings_by_name[name].float()
            point_to_level = parent_map[backbone_output.inverse_map.long()]
            level_batch, mapping_stats = build_pure_token_contrastive_batch(
                batch,
                point_to_level,
                level_xyz,
                level_name=name,
                include_proposal_diagnostics=return_proposal_diagnostics,
            )
            if level_batch is None:
                level_loss = zero_loss_touching_module_parameters(
                    level_embeddings, self.multiscale_criteria[name]
                )
                level_metrics = {
                    "loss": 0.0,
                    "hardest_negative_ranking_accuracy": 0.0,
                    "cosine_gap": 0.0,
                    "temperature": float(
                        self.multiscale_criteria[name].temperature.detach()
                    ),
                }
            else:
                level_result = self.multiscale_criteria[name](
                    level_embeddings, level_batch
                )
                level_loss = level_result["loss_total"]
                level_metrics = {
                    "loss": float(level_loss.detach()),
                    "hardest_negative_ranking_accuracy": float(
                        level_result["hardest_negative_ranking_accuracy"]
                    ),
                    "cosine_gap": float(level_result["cosine_gap"]),
                    "temperature": float(level_result["temperature"]),
                }
            stage_weight = float(MULTISCALE_LEVEL_LOSS_WEIGHTS[name])
            unweighted_losses[name] = level_loss
            weighted_losses.append(stage_weight * level_loss)
            stages[name] = {
                **level_metrics,
                **mapping_stats,
                "token_count": int(tokens.shape[0]),
                "stage_weight": stage_weight,
            }
        return {
            "loss_total": torch.stack(weighted_losses).sum(),
            "stage_loss_tensors": unweighted_losses,
            "stages": stages,
        }

    @staticmethod
    def _cell_balanced_legacy_multiscale(
        records: Sequence[tuple[MaskSupervisionGroup, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Valid-pair normalize within cells, then equally weight nonempty cells."""

        cells = tuple(dict.fromkeys(group.cell for group, _ in records))
        by_cell = {
            cell: [result for group, result in records if group.cell == cell]
            for cell in cells
        }

        reference = records[0][1]["loss_total"]
        stage_cell_losses: dict[str, dict[str, torch.Tensor]] = {
            stage: {} for stage in MULTISCALE_LEVEL_NAMES
        }
        stage_cell_denominators: dict[str, dict[str, int]] = {
            stage: {} for stage in MULTISCALE_LEVEL_NAMES
        }
        active_cells: list[str] = []
        for cell in cells:
            cell_has_pairs = False
            for stage in MULTISCALE_LEVEL_NAMES:
                denominator = sum(
                    int(result["stages"][stage]["kept_pairs"])
                    for result in by_cell[cell]
                )
                numerator = sum(
                    (
                        result["stage_loss_tensors"][stage]
                        * float(result["stages"][stage]["kept_pairs"])
                        for result in by_cell[cell]
                    ),
                    reference.new_zeros(()),
                )
                stage_cell_denominators[stage][cell] = denominator
                stage_cell_losses[stage][cell] = numerator / max(denominator, 1)
                cell_has_pairs = cell_has_pairs or denominator > 0
            if cell_has_pairs:
                active_cells.append(cell)
        if not active_cells:
            stage_tensors = {
                stage: reference * 0.0 for stage in MULTISCALE_LEVEL_NAMES
            }
        else:
            stage_tensors = {
                stage: torch.stack(
                    [stage_cell_losses[stage][cell] for cell in active_cells]
                ).mean()
                for stage in MULTISCALE_LEVEL_NAMES
            }
        stages: dict[str, dict[str, Any]] = {}
        sum_keys = {
            "input_pairs",
            "input_proposals",
            "valid_pairs",
            "kept_pairs",
            "kept_proposals",
            "dropped_proposals_insufficient_pure_positive",
            "dropped_proposals_no_pure_negative",
            "pure_positive_tokens",
            "pure_negative_tokens",
            "mixed_or_unknown_tokens_ignored",
            "zero_negative_pairs",
        }
        for stage in MULTISCALE_LEVEL_NAMES:
            stage_records = [
                record["stages"][stage] for _, record in records
            ]
            summary = dict(stage_records[0])
            for key, value in tuple(summary.items()):
                if key == "proposal_diagnostics":
                    summary.pop(key)
                elif isinstance(value, (int, float)):
                    if key in sum_keys:
                        summary[key] = sum(
                            float(record.get(key, 0.0)) for record in stage_records
                        )
                    else:
                        summary[key] = sum(
                            float(record.get(key, 0.0)) for record in stage_records
                        ) / len(stage_records)
            summary["loss"] = float(stage_tensors[stage].detach())
            summary["frame_group_count"] = len(records)
            summary["loss_denominator_valid_pairs_by_cell"] = {
                cell: stage_cell_denominators[stage][cell] for cell in cells
            }
            summary["cell_equalization_denominator"] = len(active_cells)
            stages[stage] = summary
        loss_total = sum(
            (
                float(MULTISCALE_LEVEL_LOSS_WEIGHTS[stage])
                * stage_tensors[stage]
                for stage in MULTISCALE_LEVEL_NAMES
            ),
            reference.new_zeros(()),
        )
        return {
            "loss_total": loss_total,
            "stage_loss_tensors": stage_tensors,
            "stages": stages,
            "normalization": (
                "valid_pair_mean_within_cell_then_equal_mean_nonempty_cells"
            ),
            "active_cells": active_cells,
            "empty_cells": [cell for cell in cells if cell not in active_cells],
        }

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        batch: ContrastiveTripletBatch | None = None,
        multiscale_loss_scale: float | None = None,
        return_loss_components: bool = False,
        return_multiscale_proposal_diagnostics: bool = False,
        *,
        source_cell: str | None = None,
        frame_groups: Sequence[Any] | None = None,
        supervision_step: int | None = None,
        supervision_total_steps: int | None = None,
        hierarchy_ramp_fraction: float | None = None,
    ) -> dict[str, Any]:
        groups = normalize_hierarchy_frame_groups(
            batch=batch,
            source_cell=source_cell,
            frame_groups=frame_groups,
        )
        (
            backbone_output,
            native,
            projected,
            level_embeddings_by_name,
        ) = self.encode_with_multiscale(
            points, features, token_space=self.token_pure_dec0
        )
        token_pure_metrics: dict[str, Any] | None = None
        if self.token_pure_dec0:
            token_records: list[
                tuple[MaskSupervisionGroup, dict[str, Any]]
            ] = []
            token_group_metrics: list[dict[str, Any]] = []
            for group in groups:
                token_batch, group_metrics = (
                    build_token_pure_dec0_contrastive_batch(
                        group.batch,
                        backbone_output.inverse_map,
                        backbone_output.scene_xyz,
                        max_positive_pairs_per_proposal=(
                            TOKEN_PURE_DEC0_MAX_PAIRS_PER_PROPOSAL
                        ),
                    )
                )
                token_group_metrics.append(
                    {
                        "cell": group.cell,
                        "group_id": group.group_id,
                        **group_metrics,
                    }
                )
                if token_batch is not None:
                    token_records.append(
                        (
                            group,
                            self.criterion(projected.float(), token_batch),
                        )
                    )
            final_result = (
                _cell_balanced_contrastive_results(token_records)
                if token_records
                else _zero_contrastive_result(projected.float(), self.criterion)
            )
            token_pure_metrics = _token_pure_dec0_optimization_metrics(
                groups=groups,
                token_records=token_records,
                group_metrics=token_group_metrics,
                expected_cells=self.token_pure_cells,
            )
        elif frame_groups is None:
            # Preserve the exact legacy arithmetic and result dictionary for
            # the single-batch path.
            final_result = self.criterion(projected.float(), groups[0].batch)
        else:
            final_result = _cell_balanced_contrastive_results(
                [
                    (group, self.criterion(projected.float(), group.batch))
                    for group in groups
                ]
            )
        result: dict[str, Any] = {
            **final_result,
            "loss_final_dec0": float(final_result["loss_total"].detach()),
            "loss_multiscale": 0.0,
            "multiscale_loss_scale": 0.0,
            "multiscale": {},
        }
        if token_pure_metrics is not None:
            result["token_pure_dec0"] = token_pure_metrics
            result["optimization_valid"] = bool(
                token_pure_metrics["optimization_valid"]
            )
        if self.multiscale_supervision:
            if len(backbone_output.multi_scale_tokens) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError(
                    "LitePT multi-scale output count drift: "
                    f"{len(backbone_output.multi_scale_tokens)} != "
                    f"{len(MULTISCALE_LEVEL_NAMES)}"
                )
            if len(backbone_output.finest_to_multi_scale) != len(
                MULTISCALE_LEVEL_NAMES
            ):
                raise RuntimeError("LitePT multi-scale parent-map count drift")

            multiscale_records = [
                (
                    group,
                    self._legacy_multiscale_for_batch(
                        backbone_output=backbone_output,
                        level_embeddings_by_name=level_embeddings_by_name,
                        batch=group.batch,
                        return_proposal_diagnostics=(
                            return_multiscale_proposal_diagnostics
                        ),
                    ),
                )
                for group in groups
            ]
            if frame_groups is None:
                multiscale_result = multiscale_records[0][1]
            else:
                multiscale_result = self._cell_balanced_legacy_multiscale(
                    multiscale_records
                )
            effective_scale = (
                self.multiscale_loss_weight
                if multiscale_loss_scale is None
                else float(multiscale_loss_scale)
            )
            if not 0.0 <= effective_scale <= self.multiscale_loss_weight + 1e-8:
                raise ValueError(
                    "multiscale_loss_scale must lie in "
                    "[0, multiscale_loss_weight], "
                    f"got {effective_scale} with max {self.multiscale_loss_weight}"
                )
            result["multiscale"] = multiscale_result["stages"]
            for stage_metrics in result["multiscale"].values():
                stage_metrics["effective_loss_weight"] = (
                    effective_scale * float(stage_metrics["stage_weight"])
                )
            auxiliary_loss = multiscale_result["loss_total"]
            unweighted_auxiliary_mean = torch.stack(
                [
                    multiscale_result["stage_loss_tensors"][stage]
                    for stage in MULTISCALE_LEVEL_NAMES
                ]
            ).mean()
            result["loss_total"] = final_result["loss_total"] + (
                effective_scale * auxiliary_loss
            )
            result["loss_multiscale"] = float(auxiliary_loss.detach())
            result["loss_multiscale_unweighted_mean"] = float(
                unweighted_auxiliary_mean.detach()
            )
            result["multiscale_loss_scale"] = effective_scale
            result["multiscale_level_loss_weights"] = dict(
                MULTISCALE_LEVEL_LOSS_WEIGHTS
            )
            if frame_groups is not None:
                result["multiscale_normalization"] = multiscale_result[
                    "normalization"
                ]
                result["multiscale_active_cells"] = multiscale_result[
                    "active_cells"
                ]
                result["multiscale_empty_cells"] = multiscale_result[
                    "empty_cells"
                ]
            result["multiscale_valid_pairs"] = int(
                sum(
                    int(level["kept_pairs"])
                    for level in result["multiscale"].values()
                )
            )
            if return_loss_components:
                result["loss_component_tensors"] = {
                    "final_dec0": final_result["loss_total"],
                    "legacy_auxiliary_scaled": effective_scale
                    * multiscale_result["loss_total"],
                    **multiscale_result["stage_loss_tensors"],
                    **{
                        f"legacy_{stage}_scaled": effective_scale
                        * float(MULTISCALE_LEVEL_LOSS_WEIGHTS[stage])
                        * multiscale_result["stage_loss_tensors"][stage]
                        for stage in MULTISCALE_LEVEL_NAMES
                    },
                }
                result["hierarchy_junction_tensors"] = (
                    self._hierarchy_stage_features(backbone_output)
                )
                result["hierarchy_junction_availability"] = {
                    "available": list(HIERARCHY_COARSE_TO_FINE),
                    "unavailable": [],
                }
            return result

        if not self.hierarchy_supervision:
            return result

        stage_features = self._hierarchy_stage_features(backbone_output)
        point_maps = hierarchy_point_maps(
            backbone_output.inverse_map,
            backbone_output.hierarchy_parent_maps,
        )
        assert self.hierarchy_mask_supervisor is not None
        mask_result = self.hierarchy_mask_supervisor(
            stage_features=stage_features,
            point_maps=point_maps,
            groups=groups,
        )
        active_mask_stage_weights, mask_stage_schedule_telemetry = (
            hierarchy_mask_stage_weights_for_step(
                static_weights=self.hierarchy_mask_stage_weights,
                schedule=getattr(
                    self,
                    "hierarchy_mask_stage_weight_schedule",
                    None,
                ),
                supervision_step=supervision_step,
                supervision_total_steps=supervision_total_steps,
            )
        )
        weighted_mask_loss, weighted_mask_stage_tensors = (
            weighted_hierarchy_mask_loss(
                loss_total=mask_result["loss_total"],
                stage_loss_tensors=mask_result["stage_loss_tensors"],
                stage_weights=active_mask_stage_weights,
            )
        )
        vicreg_result = native_hierarchy_vicreg_loss(
            stage_features,
            variance_target=self.hierarchy_variance_target,
            covariance_weight=self.hierarchy_covariance_weight,
            max_tokens=self.hierarchy_vicreg_max_tokens,
        )
        ramp_fraction = self.hierarchy_ramp_for_step(
            supervision_step=supervision_step,
            supervision_total_steps=supervision_total_steps,
            hierarchy_ramp_fraction=hierarchy_ramp_fraction,
        )
        mask_scale = ramp_fraction * self.hierarchy_mask_loss_weight
        vicreg_scale = self.hierarchy_vicreg_loss_weight
        scaled_mask_loss = mask_scale * weighted_mask_loss
        scaled_vicreg_loss = vicreg_scale * vicreg_result["loss_total"]
        result["loss_total"] = (
            final_result["loss_total"]
            + scaled_mask_loss
            + scaled_vicreg_loss
        )

        hierarchy_stages: dict[str, dict[str, Any]] = {}
        for stage in HIERARCHY_COARSE_TO_FINE:
            mask_stage = mask_result["metrics"]["stages"].get(
                stage,
                {
                    "active": False,
                    "route_weight": 0.0,
                    "loss": 0.0,
                },
            )
            vicreg_stage = vicreg_result["metrics"]["stages"][stage]
            stage_weight = float(
                active_mask_stage_weights.get(stage, 1.0)
            )
            hierarchy_stages[stage] = {
                "token_count": int(stage_features[stage].shape[0]),
                "mask": mask_stage,
                "vicreg": vicreg_stage,
                "mask_loss": float(mask_stage.get("loss", 0.0)),
                "mask_weighted_loss": float(
                    weighted_mask_stage_tensors.get(
                        stage,
                        weighted_mask_loss * 0.0,
                    ).detach()
                ),
                "mask_stage_weight": stage_weight,
                "mask_effective_weight": mask_scale
                * stage_weight
                * float(mask_stage.get("effective_weight_realized", 0.0)),
                "vicreg_loss": float(vicreg_stage["loss"]),
                "vicreg_effective_weight": vicreg_scale
                * float(vicreg_stage["stage_weight"]),
            }
        source_cells = list(dict.fromkeys(group.cell for group in groups))
        mask_optimization_valid = bool(
            mask_result["metrics"]["optimization_valid"]
        )
        vicreg_stage_validity = {
            stage: bool(vicreg_result["metrics"]["stages"][stage]["valid"])
            for stage in HIERARCHY_COARSE_TO_FINE
        }
        vicreg_all_stages_valid = all(vicreg_stage_validity.values())
        vicreg_required = self.hierarchy_vicreg_loss_weight > 0.0
        hierarchy_optimization_valid = bool(
            mask_optimization_valid
            and (not vicreg_required or vicreg_all_stages_valid)
        )
        result["hierarchy_supervision"] = {
            "enabled": True,
            "recipe": HIERARCHY_RECIPE_NAME,
            "optimization_valid": hierarchy_optimization_valid,
            "mask_optimization_valid": mask_optimization_valid,
            "expected_three_cell_valid": bool(
                mask_result["metrics"]["expected_three_cell_valid"]
            ),
            "route_stage_valid": bool(
                mask_result["metrics"]["route_stage_valid"]
            ),
            "fold_denominators_by_cell": mask_result["metrics"][
                "fold_denominators_by_cell"
            ],
            "fold_denominators_by_route_stage": mask_result["metrics"][
                "fold_denominators_by_route_stage"
            ],
            "route_stage_validity": mask_result["metrics"][
                "route_stage_validity"
            ],
            "vicreg_required_for_optimization": vicreg_required,
            "vicreg_all_stages_valid": vicreg_all_stages_valid,
            "vicreg_stage_validity": vicreg_stage_validity,
            "ramp_fraction": ramp_fraction,
            "mask_loss_weight": self.hierarchy_mask_loss_weight,
            "mask_stage_weights": dict(active_mask_stage_weights),
            "query_projection_policy": (
                # Small synthetic/checkpoint-compatibility constructions used
                # by diagnostics predate this optional C8 field.  The
                # supervisor itself still defaults to the exact C3--C7 shared
                # policy, so report that default rather than making telemetry
                # introduce a new required attribute outside __init__.
                getattr(self, "hierarchy_query_projection_policy", "shared_dec0")
            ),
            "mask_stage_weight_schedule": {
                "configured": getattr(
                    self,
                    "hierarchy_mask_stage_weight_schedule",
                    None,
                ),
                **mask_stage_schedule_telemetry,
            },
            "mask_effective_loss_weight": mask_scale,
            "vicreg_loss_weight": vicreg_scale,
            "loss_mask": float(weighted_mask_loss.detach()),
            "loss_mask_unweighted": float(mask_result["loss_total"].detach()),
            "loss_mask_scaled": float(scaled_mask_loss.detach()),
            "loss_vicreg": float(vicreg_result["loss_total"].detach()),
            "loss_vicreg_scaled": float(scaled_vicreg_loss.detach()),
            "source_cells": source_cells,
            "frame_group_count": len(groups),
            "normalization": (
                "dec0_pair_mean_within_cell_equal_across_nonempty_cells;"
                "mask_realized_fold_mean_within_cell_equal_across_nonempty_cells"
            ),
            "stages": hierarchy_stages,
            "mask": mask_result["metrics"],
            "vicreg": vicreg_result["metrics"],
        }
        if token_pure_metrics is not None:
            result["optimization_valid"] = bool(
                token_pure_metrics["optimization_valid"]
                and hierarchy_optimization_valid
            )
        result["loss_hierarchy_mask"] = float(weighted_mask_loss.detach())
        result["loss_hierarchy_vicreg"] = float(
            vicreg_result["loss_total"].detach()
        )
        if return_loss_components:
            result["loss_component_tensors"] = {
                "final_dec0": final_result["loss_total"],
                "hierarchy_mask": weighted_mask_loss,
                "hierarchy_mask_scaled": scaled_mask_loss,
                "hierarchy_vicreg": vicreg_result["loss_total"],
                "hierarchy_vicreg_scaled": scaled_vicreg_loss,
                **{
                    f"hierarchy_mask_{stage}": loss
                    for stage, loss in mask_result[
                        "stage_loss_tensors"
                    ].items()
                },
                **{
                    f"hierarchy_mask_{stage}_weighted": loss
                    for stage, loss in weighted_mask_stage_tensors.items()
                },
                **{
                    f"hierarchy_vicreg_{stage}": loss
                    for stage, loss in vicreg_result[
                        "stage_loss_tensors"
                    ].items()
                },
            }
            result["hierarchy_junction_tensors"] = stage_features
        return result
