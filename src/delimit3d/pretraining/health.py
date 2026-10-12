"""Pretraining health extracted from the executed trainer."""

from __future__ import annotations


import math


from typing import Any, Mapping, Sequence


from delimit3d.data.contrastive_sampler_v2 import DEFAULT_2D_CELLS as V2_DEFAULT_2D_CELLS


from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule


from .engine_settings import (
    CUMULATIVE_HEALTH_SCHEMA,
    GRADIENT_PANEL_CUMULATIVE_SCHEMA,
    HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
    HIERARCHY_HEALTH_STAGES,
    HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
    HIERARCHY_ROUTE_STAGES,
    LEGACY_MULTISCALE_CUMULATIVE_KEYS,
    LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
    LEGACY_MULTISCALE_CUMULATIVE_STAGES,
    LEGACY_MULTISCALE_FLOAT_KEYS,
    LEGACY_MULTISCALE_INTEGER_KEYS,
    RANK_CUMULATIVE_HEALTH_KEYS,
    RANK_CUMULATIVE_HEALTH_SCHEMA,
    TOKEN_CUMULATIVE_COUNT_KEYS,
    TOKEN_CUMULATIVE_HEALTH_SCHEMA,
)
from .records import (
    _json_copy,
)


def _merge_count_dicts(rows: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _strict_nonnegative_counter(value: Any, *, name: str) -> int:
    """Read a JSON counter without accepting booleans or negative values."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid cumulative counter: {name}")
    return int(value)


def _strict_finite_real(value: Any, *, name: str) -> float:
    """Read a JSON metric without accepting booleans or non-finite values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid cumulative metric: {name}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid cumulative metric: {name}")
    return parsed


def _strict_nonnegative_real(value: Any, *, name: str) -> float:
    parsed = _strict_finite_real(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"Invalid cumulative metric: {name}")
    return parsed


def _canonical_string_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"Invalid cumulative string set: {name}")
    parsed = tuple(value)
    if any(not isinstance(item, str) or not item for item in parsed):
        raise ValueError(f"Invalid cumulative string set: {name}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate cumulative string in {name}")
    return parsed


def _canonical_panel_schedule(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"Invalid cumulative gradient-panel schedule: {name}")
    parsed: list[int] = []
    for item in value:
        parsed.append(_strict_nonnegative_counter(item, name=f"{name}[]"))
    result = tuple(parsed)
    if result != tuple(sorted(set(result))) or any(item <= 0 for item in result):
        raise ValueError(f"Invalid cumulative gradient-panel schedule: {name}")
    return result


def _canonical_source_metadata(
    cells: Sequence[str], *, hierarchy_enabled: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered_cells = _canonical_string_sequence(cells, name="source_cells")
    granularities = tuple(
        dict.fromkeys(cell.rsplit("/", 1)[-1] for cell in ordered_cells)
    )
    allowed_granularities = (*HIERARCHY_ROUTE_STAGES, "sam2") if not hierarchy_enabled else tuple(HIERARCHY_ROUTE_STAGES)
    if any(granularity not in allowed_granularities for granularity in granularities):
        raise ValueError("Cumulative source cells contain an unknown granularity")
    if hierarchy_enabled and set(granularities) != set(HIERARCHY_ROUTE_STAGES):
        raise ValueError(
            "Hierarchy cumulative state requires exactly g02/g05/g08 granularities"
        )
    if hierarchy_enabled and tuple(ordered_cells) != tuple(V2_DEFAULT_2D_CELLS):
        raise ValueError("Hierarchy cumulative state requires canonical 2D source cells")
    if hierarchy_enabled and len(granularities) != len(HIERARCHY_ROUTE_STAGES):
        raise ValueError("Hierarchy cumulative granularities are duplicated")
    return ordered_cells, granularities


def _legacy_multiscale_stage_metrics(
    value: Any, *, stage: str
) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Legacy multiscale stage {stage} is not a mapping")
    missing = set(LEGACY_MULTISCALE_CUMULATIVE_KEYS) - set(value)
    if missing:
        raise ValueError(
            f"Legacy multiscale stage {stage} is missing canonical metrics: "
            f"{sorted(missing)}"
        )
    parsed: dict[str, int | float] = {}
    for key in LEGACY_MULTISCALE_INTEGER_KEYS:
        raw = value.get(key, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Invalid legacy multiscale counter: {stage}.{key}")
        raw_float = float(raw)
        # The legacy mapper uses -1 for an unmeasured feature-candidate
        # duplicate count when feature-hard mining requested zero candidates.
        # Canonical cumulative state has no unknown sentinel: no candidate
        # pool means zero candidate duplicates; a negative value otherwise is
        # rejected below.
        if (
            key == "feature_candidate_duplicates"
            and raw_float == -1.0
            and value.get("feature_hard_negatives", 0) == 0
            and value.get("feature_candidate_pool", 0) == 0
        ):
            raw_float = 0.0
        if not math.isfinite(raw_float) or raw_float < 0.0 or not raw_float.is_integer():
            raise ValueError(f"Invalid legacy multiscale counter: {stage}.{key}")
        parsed[key] = int(raw_float)
    for key in LEGACY_MULTISCALE_FLOAT_KEYS:
        parsed[key] = _strict_finite_real(
            value.get(key, 0.0), name=f"legacy_multiscale.{stage}.{key}"
        )
    if parsed["valid_pairs"] > parsed["input_pairs"]:
        raise ValueError(f"Legacy multiscale pair denominator drift: {stage}")
    if parsed["kept_pairs"] > parsed["valid_pairs"]:
        raise ValueError(f"Legacy multiscale kept-pair drift: {stage}")
    if parsed["kept_proposals"] > parsed["input_proposals"]:
        raise ValueError(f"Legacy multiscale proposal denominator drift: {stage}")
    return parsed


def _new_rank_cumulative_health(
    *,
    rank: int,
    cells: Sequence[str],
    hierarchy_enabled: bool,
    scheduled_gradient_panel_updates: Sequence[int],
    expected_microbatches_per_update: int = 1,
    legacy_multiscale_enabled: bool = False,
    logical_batch_schedule: LogicalBatchSchedule | None = None,
) -> dict[str, Any]:
    """Create rank-local, checkpointable counters with no sampled-row proxies."""

    if not isinstance(hierarchy_enabled, bool) or not isinstance(
        legacy_multiscale_enabled, bool
    ):
        raise ValueError("Cumulative-health enablement flags must be boolean")
    if hierarchy_enabled and legacy_multiscale_enabled:
        raise ValueError("Hierarchy and legacy multiscale delivery are exclusive")
    ordered_cells, granularities = _canonical_source_metadata(
        cells, hierarchy_enabled=hierarchy_enabled
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    expected_microbatches = _strict_nonnegative_counter(
        expected_microbatches_per_update,
        name="expected_microbatches_per_update",
    )
    if expected_microbatches <= 0:
        raise ValueError("expected_microbatches_per_update must be positive")
    schedule = _canonical_panel_schedule(
        scheduled_gradient_panel_updates, name="scheduled_updates"
    )
    hierarchy_per_granularity_stage = {
        granularity: {
            stage: {
                "visit_count": 0,
                "route_active_count": 0,
                "route_required_count": 0,
                "optimization_valid_count": 0,
                "optimization_invalid_count": 0,
                **{key: 0 for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS},
            }
            for stage in HIERARCHY_ROUTE_STAGES[granularity]
        }
        for granularity in HIERARCHY_ROUTE_STAGES
    }
    legacy_stage_template = {
        key: (0 if key in LEGACY_MULTISCALE_INTEGER_KEYS else 0.0)
        for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
    }
    return {
        "schema_version": RANK_CUMULATIVE_HEALTH_SCHEMA,
        **({"logical_batch_schedule": logical_batch_schedule.as_dict()}
           if logical_batch_schedule is not None else {}),
        "rank": int(rank),
        "through_update": 0,
        "optimizer_update_count": 0,
        "source_cells": list(ordered_cells),
        "granularities": list(granularities),
        "expected_microbatches_per_update": expected_microbatches,
        "clipping": {
            "definition": "any_rank_was_clipped_per_optimizer_update",
            "local_rank_update_event_count": 0,
            "local_rank_clipped_update_count": 0,
            "any_rank_clipped_update_count": 0,
            "all_ranks_unclipped_update_count": 0,
        },
        "token_pure_dec0": {
            "schema_version": TOKEN_CUMULATIVE_HEALTH_SCHEMA,
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "per_cell": {
                cell: {key: 0 for key in TOKEN_CUMULATIVE_COUNT_KEYS}
                for cell in ordered_cells
            },
        },
        "hierarchy_supervision": {
            "schema_version": HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
            "enabled": bool(hierarchy_enabled),
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "vicreg_valid_count": 0,
            "vicreg_invalid_count": 0,
            "per_cell": {
                granularity: {
                    "visit_count": 0,
                    "optimization_valid_count": 0,
                    "optimization_invalid_count": 0,
                    "input_groups": 0,
                    "input_proposals": 0,
                    "assigned_proposals": 0,
                    **{
                        key: 0
                        for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS
                    },
                }
                for granularity in granularities
            },
            "per_stage": {
                stage: {
                    "visit_count": 0,
                    **{
                        key: 0
                        for key in HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS
                    },
                    "vicreg_valid_count": 0,
                    "vicreg_invalid_count": 0,
                }
                for stage in HIERARCHY_HEALTH_STAGES
            },
            "per_granularity_stage": hierarchy_per_granularity_stage,
        },
        "legacy_multiscale_delivery": {
            "schema_version": LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
            "enabled": bool(legacy_multiscale_enabled),
            "microbatch_count": 0,
            "recorded_microbatch_count": 0,
            "optimization_valid_count": 0,
            "optimization_invalid_count": 0,
            "per_stage": {
                stage: dict(legacy_stage_template)
                for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES
            },
        },
        "gradient_panels": {
            "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
            "scheduled_updates": list(schedule),
            "panels": [],
        },
    }


def _accumulate_rank_cumulative_health_update(
    state: dict[str, Any],
    *,
    update: int,
    local_was_clipped: bool,
    clipped_rank_count: int,
    world_size: int,
    attempted_microbatch_count: int,
    optimization_valid_count: int,
    token_records: Sequence[Mapping[str, Any]],
    hierarchy_records: Sequence[Mapping[str, Any]],
    legacy_multiscale_records: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Commit one completed optimizer update into exact rank-local counters."""

    if state.get("schema_version") != RANK_CUMULATIVE_HEALTH_SCHEMA:
        raise ValueError("Rank cumulative-health schema drift")
    observed_update = _strict_nonnegative_counter(update, name="update")
    expected_update = _strict_nonnegative_counter(
        state.get("through_update"), name="through_update"
    ) + 1
    if observed_update != expected_update:
        raise ValueError(
            "Cumulative-health update sequence drift: "
            f"observed={observed_update} expected={expected_update}"
        )
    if (
        not isinstance(local_was_clipped, bool)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(clipped_rank_count, bool)
        or not isinstance(clipped_rank_count, int)
        or not 0 <= clipped_rank_count <= world_size
    ):
        raise ValueError("Invalid clipping event types")
    attempted = _strict_nonnegative_counter(
        attempted_microbatch_count, name="attempted_microbatch_count"
    )
    valid = _strict_nonnegative_counter(
        optimization_valid_count, name="optimization_valid_count"
    )
    if attempted <= 0 or not 0 <= valid <= attempted:
        raise ValueError("Invalid cumulative microbatch validity counts")
    expected_microbatches = _strict_nonnegative_counter(
        state.get("expected_microbatches_per_update"),
        name="expected_microbatches_per_update",
    )
    if "logical_batch_schedule" in state:
        schedule = LogicalBatchSchedule.from_dict(state["logical_batch_schedule"])
        if schedule.nominal_batch_size != expected_microbatches:
            raise ValueError("Cumulative nominal logical-batch denominator drift")
        expected_microbatches = schedule.batch_size(observed_update)
    if attempted != expected_microbatches:
        raise ValueError(
            "Cumulative microbatch denominator drift: "
            f"observed={attempted} expected={expected_microbatches}"
        )
    if (
        len(token_records) > attempted
        or len(hierarchy_records) > attempted
        or len(legacy_multiscale_records) > attempted
    ):
        raise ValueError("More delivery records than attempted microbatches")
    for records, name in (
        (token_records, "token"),
        (hierarchy_records, "hierarchy"),
        (legacy_multiscale_records, "legacy multiscale"),
    ):
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError(f"Malformed {name} cumulative delivery record")

    legacy = state["legacy_multiscale_delivery"]
    if not isinstance(legacy.get("enabled"), bool):
        raise ValueError("Legacy multiscale enablement is invalid")
    if legacy["enabled"]:
        if len(legacy_multiscale_records) != valid:
            raise ValueError(
                "Legacy multiscale record denominator does not match valid microbatches"
            )
    elif legacy_multiscale_records:
        raise ValueError(
            "Legacy multiscale records present while legacy delivery is disabled"
        )

    clipping = state["clipping"]
    clipping["local_rank_update_event_count"] += 1
    clipping["local_rank_clipped_update_count"] += int(bool(local_was_clipped))
    clipping["any_rank_clipped_update_count"] += int(clipped_rank_count > 0)
    clipping["all_ranks_unclipped_update_count"] += int(clipped_rank_count == 0)

    token = state["token_pure_dec0"]
    token["microbatch_count"] += attempted
    token["recorded_microbatch_count"] += len(token_records)
    token["optimization_valid_count"] += valid
    token["optimization_invalid_count"] += attempted - valid
    for record in token_records:
        for group in record.get("groups", ()):
            cell = str(group.get("cell"))
            if cell not in token["per_cell"]:
                continue
            summary = token["per_cell"][cell]
            realized_pairs = int(group.get("realized_positive_pairs", 0))
            summary["input_groups"] += 1
            summary["retained_groups"] += int(realized_pairs > 0)
            summary["input_proposals"] += int(group.get("input_proposals", 0))
            summary["retained_proposals"] += int(
                group.get("retained_proposals", 0)
            )
            summary["input_point_pairs"] += int(group.get("input_point_pairs", 0))
            summary["requested_token_pairs"] += sum(
                int(proposal.get("requested_token_pairs", 0))
                for proposal in group.get("proposals", ())
            )
            summary["realized_positive_pairs"] += realized_pairs
            summary["zero_denominator_groups"] += int(
                bool(group.get("zero_denominator", realized_pairs <= 0))
            )

    if legacy["enabled"]:
        legacy["microbatch_count"] += attempted
        legacy["recorded_microbatch_count"] += len(legacy_multiscale_records)
        legacy["optimization_valid_count"] += valid
        legacy["optimization_invalid_count"] += attempted - valid
        for record in legacy_multiscale_records:
            levels = record.get("levels")
            if not isinstance(levels, Mapping) or set(levels) != set(
                LEGACY_MULTISCALE_CUMULATIVE_STAGES
            ):
                raise ValueError("Legacy multiscale stage set drift")
            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
                parsed = _legacy_multiscale_stage_metrics(
                    levels[stage], stage=stage
                )
                stage_summary = legacy["per_stage"][stage]
                for key in LEGACY_MULTISCALE_INTEGER_KEYS:
                    stage_summary[key] += int(parsed[key])
                for key in LEGACY_MULTISCALE_FLOAT_KEYS:
                    stage_summary[key] += float(parsed[key])

    hierarchy = state["hierarchy_supervision"]
    if bool(hierarchy["enabled"]):
        hierarchy["microbatch_count"] += attempted
        hierarchy["recorded_microbatch_count"] += len(hierarchy_records)
        hierarchy["optimization_valid_count"] += valid
        hierarchy["optimization_invalid_count"] += attempted - valid
        vicreg_valid_records = 0
        per_cell_optimization_valid_counts = {
            granularity: 0 for granularity in hierarchy["per_cell"]
        }
        per_edge_optimization_valid_counts = {
            (granularity, stage): 0
            for granularity, stages in HIERARCHY_ROUTE_STAGES.items()
            for stage in stages
        }
        stage_vicreg_valid_counts = {
            stage: 0 for stage in HIERARCHY_HEALTH_STAGES
        }
        for record in hierarchy_records:
            if not isinstance(record, Mapping):
                continue
            stage_records = record.get("stages", {})
            if not isinstance(stage_records, Mapping):
                stage_records = {}
            record_vicreg_flag = record.get("vicreg_all_stages_valid")
            if isinstance(record_vicreg_flag, bool):
                record_vicreg_valid = record_vicreg_flag
            else:
                record_vicreg_valid = all(
                    isinstance(stage_record, Mapping)
                    and isinstance(stage_record.get("vicreg"), Mapping)
                    and bool(stage_record["vicreg"].get("valid", False))
                    for stage in HIERARCHY_HEALTH_STAGES
                    for stage_record in (stage_records.get(stage),)
                )
            vicreg_valid_records += int(record_vicreg_valid)
            route_records = record.get("route_stage_validity", {})
            if not isinstance(route_records, Mapping) or set(route_records) != set(
                HIERARCHY_ROUTE_STAGES
            ):
                raise ValueError("Hierarchy route granularity set drift")
            for granularity, cell_summary in hierarchy["per_cell"].items():
                route = route_records.get(granularity)
                if not isinstance(route, Mapping):
                    raise ValueError(
                        f"Hierarchy route record missing for {granularity}"
                    )
                route_valid = bool(route.get("optimization_valid", False))
                per_cell_optimization_valid_counts[granularity] += int(
                    route_valid
                )
                cell_summary["input_groups"] += int(
                    route.get("input_group_count", 0)
                )
                cell_summary["input_proposals"] += int(
                    route.get("input_proposal_count", 0)
                )
                cell_summary["assigned_proposals"] += int(
                    route.get("assigned_proposal_count", 0)
                )
                route_stages = route.get("stages", {})
                expected_route_stages = HIERARCHY_ROUTE_STAGES[granularity]
                if not isinstance(route_stages, Mapping) or set(route_stages) != set(
                    expected_route_stages
                ):
                    raise ValueError(
                        f"Hierarchy route stage set drift for {granularity}"
                    )
                for stage in expected_route_stages:
                    stage_route = route_stages[stage]
                    if not isinstance(stage_route, Mapping):
                        raise ValueError(
                            f"Hierarchy route stage record missing for "
                            f"{granularity}/{stage}"
                        )
                    cell_summary["mask_valid_fold_denominator"] += int(
                        stage_route.get("loss_denominator_valid_folds", 0)
                    )
                    for source, target in (
                        ("proposals_attempted", "mask_proposals_attempted"),
                        ("proposals_valid", "mask_proposals_valid"),
                        ("folds_attempted", "mask_folds_attempted"),
                        ("folds_valid", "mask_folds_valid"),
                    ):
                        cell_summary[target] += int(stage_route.get(source, 0))
                    edge_summary = hierarchy["per_granularity_stage"][granularity][
                        stage
                    ]
                    edge_summary["route_active_count"] += int(
                        bool(stage_route.get("route_active", False))
                    )
                    edge_summary["route_required_count"] += int(
                        bool(stage_route.get("route_required", False))
                    )
                    edge_valid = bool(stage_route.get("optimization_valid", False))
                    per_edge_optimization_valid_counts[(granularity, stage)] += int(
                        edge_valid
                    )
                    edge_summary["mask_valid_fold_denominator"] += int(
                        stage_route.get("loss_denominator_valid_folds", 0)
                    )
                    for source, target in (
                        ("proposals_attempted", "mask_proposals_attempted"),
                        ("proposals_valid", "mask_proposals_valid"),
                        ("folds_attempted", "mask_folds_attempted"),
                        ("folds_valid", "mask_folds_valid"),
                    ):
                        edge_summary[target] += int(stage_route.get(source, 0))
            for stage, stage_summary in hierarchy["per_stage"].items():
                stage_record = stage_records.get(stage, {})
                if not isinstance(stage_record, Mapping):
                    stage_record = {}
                mask = stage_record.get("mask", {})
                if not isinstance(mask, Mapping):
                    mask = {}
                vicreg = stage_record.get("vicreg", {})
                if not isinstance(vicreg, Mapping):
                    vicreg = {}
                stage_summary["mask_valid_fold_denominator"] += int(
                    mask.get("loss_denominator_valid_folds", 0)
                )
                for source, target in (
                    ("proposals_attempted", "mask_proposals_attempted"),
                    ("proposals_valid", "mask_proposals_valid"),
                    ("folds_attempted", "mask_folds_attempted"),
                    ("folds_valid", "mask_folds_valid"),
                ):
                    stage_summary[target] += int(mask.get(source, 0))
                stage_vicreg_valid_counts[stage] += int(
                    bool(vicreg.get("valid", False))
                )
        for (granularity, stage), valid_count in (
            per_edge_optimization_valid_counts.items()
        ):
            edge_summary = hierarchy["per_granularity_stage"][granularity][stage]
            edge_summary["visit_count"] += attempted
            edge_summary["optimization_valid_count"] += valid_count
            edge_summary["optimization_invalid_count"] += attempted - valid_count
        hierarchy["vicreg_valid_count"] += vicreg_valid_records
        hierarchy["vicreg_invalid_count"] += attempted - vicreg_valid_records
        for granularity, cell_summary in hierarchy["per_cell"].items():
            optimization_valid_count = per_cell_optimization_valid_counts[
                granularity
            ]
            cell_summary["optimization_valid_count"] += optimization_valid_count
            cell_summary["visit_count"] += attempted
            cell_summary["optimization_invalid_count"] += (
                attempted - optimization_valid_count
            )
        for stage, stage_summary in hierarchy["per_stage"].items():
            stage_summary["visit_count"] += attempted
            stage_summary["vicreg_valid_count"] += stage_vicreg_valid_counts[stage]
            stage_summary["vicreg_invalid_count"] += (
                attempted - stage_vicreg_valid_counts[stage]
            )
    elif hierarchy_records:
        raise ValueError("Hierarchy records present while hierarchy health is disabled")

    state["optimizer_update_count"] += 1
    state["through_update"] = observed_update


def _gradient_objective_stage_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count finite/nonzero stage gradients instead of averaging them away."""

    objectives = sorted(
        {
            str(objective)
            for record in records
            for objective in record.get("objectives", {})
        }
    )
    output: dict[str, Any] = {
        "schema_version": "litept_gradient_objective_stage_counts/v1",
        "record_count": len(records),
        "objectives": {},
    }
    for objective in objectives:
        objective_records = [
            record.get("objectives", {}).get(objective, {})
            for record in records
            if objective in record.get("objectives", {})
        ]
        stages = sorted(
            {
                str(stage)
                for objective_record in objective_records
                for stage in objective_record.get("stages", {})
            }
        )
        stage_counts: dict[str, Any] = {}
        for stage in stages:
            metrics_rows = [
                objective_record.get("stages", {}).get(stage)
                for objective_record in objective_records
            ]
            metrics_rows = [
                metrics for metrics in metrics_rows if isinstance(metrics, Mapping)
            ]
            finite_count = 0
            nonzero_count = 0
            for metrics in metrics_rows:
                norm = metrics.get("grad_norm")
                norm_finite = isinstance(norm, (int, float)) and math.isfinite(
                    float(norm)
                )
                finite = bool(metrics.get("finite", False)) and norm_finite
                finite_count += int(finite)
                nonzero_count += int(finite and float(norm) > 0.0)
            stage_counts[stage] = {
                "observation_count": len(metrics_rows),
                "finite_count": finite_count,
                "nonzero_count": nonzero_count,
                "cosine_valid_count": sum(
                    int(bool(metrics.get("cosine_valid", False)))
                    for metrics in metrics_rows
                ),
                "norm_ratio_valid_count": sum(
                    int(bool(metrics.get("norm_ratio_valid", False)))
                    for metrics in metrics_rows
                ),
            }
        output["objectives"][objective] = {
            "record_count": len(objective_records),
            "finite_record_count": sum(
                int(
                    bool(objective_record.get("stages"))
                    and all(
                        bool(metrics.get("finite", False))
                        and isinstance(metrics.get("grad_norm"), (int, float))
                        and math.isfinite(float(metrics["grad_norm"]))
                        for metrics in objective_record.get("stages", {}).values()
                    )
                )
                for objective_record in objective_records
            ),
            "nonzero_record_count": sum(
                int(
                    any(
                        bool(metrics.get("finite", False))
                        and isinstance(metrics.get("grad_norm"), (int, float))
                        and math.isfinite(float(metrics["grad_norm"]))
                        and float(metrics["grad_norm"]) > 0.0
                        for metrics in objective_record.get("stages", {}).values()
                    )
                )
                for objective_record in objective_records
            ),
            "stages": stage_counts,
        }
    return output


def _append_rank_gradient_panel(
    state: dict[str, Any], panel: Mapping[str, Any]
) -> None:
    if not isinstance(panel, Mapping):
        raise ValueError("Incomplete cumulative gradient-panel record")
    update = _strict_nonnegative_counter(
        panel.get("update"), name="gradient_panel.update"
    )
    gradient_panels = state["gradient_panels"]
    if update not in set(gradient_panels["scheduled_updates"]):
        raise ValueError(f"Unexpected gradient panel update {update}")
    if any(int(row.get("update", -1)) == update for row in gradient_panels["panels"]):
        raise ValueError(f"Duplicate gradient panel update {update}")
    required = {
        "kind",
        "update",
        "global_shared_gradient",
        "controller",
        "conflict_threshold",
        "objective_stage_counts",
    }
    if panel.get("kind") != "gradient_health" or not required.issubset(panel):
        raise ValueError("Incomplete cumulative gradient-panel record")
    if not isinstance(panel.get("global_shared_gradient"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel gradient report")
    if not isinstance(panel.get("controller"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel controller")
    if not isinstance(panel.get("objective_stage_counts"), Mapping):
        raise ValueError("Invalid cumulative gradient-panel stage counts")
    _strict_finite_real(
        panel.get("conflict_threshold"), name="gradient_panel.conflict_threshold"
    )
    gradient_panels["panels"].append(_json_copy(dict(panel)))


def _merge_rank_cumulative_health(
    rank_states: Sequence[Mapping[str, Any]],
    *,
    through_update: int,
    world_size: int,
    gradient_conflict_controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge exact per-rank state into the checkpoint-native health record."""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(through_update, bool)
        or not isinstance(through_update, int)
        or through_update < 0
    ):
        raise ValueError("Invalid cumulative-health merge boundary")
    if not isinstance(gradient_conflict_controller, Mapping):
        raise ValueError("Gradient conflict-controller state is not a mapping")
    if len(rank_states) != int(world_size):
        raise ValueError("Incomplete rank cumulative-health state")
    if any(not isinstance(row, Mapping) for row in rank_states):
        raise ValueError("Incomplete rank cumulative-health state")
    schedule_payloads = [row.get("logical_batch_schedule") for row in rank_states]
    if any(value != schedule_payloads[0] for value in schedule_payloads):
        raise ValueError("Ranks disagree on the logical-batch schedule")
    logical_batch_schedule = (LogicalBatchSchedule.from_dict(schedule_payloads[0])
                              if schedule_payloads[0] is not None else None)
    expected_keys = (RANK_CUMULATIVE_HEALTH_KEYS | {"logical_batch_schedule"}
                     if logical_batch_schedule is not None else RANK_CUMULATIVE_HEALTH_KEYS)
    if any(set(row) != expected_keys for row in rank_states):
        raise ValueError("Rank cumulative-health canonical key set drift")
    try:
        ordered = sorted(rank_states, key=lambda row: int(row.get("rank", -1)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cumulative-health rank identities are invalid") from exc
    if any(
        isinstance(row.get("rank"), bool) or not isinstance(row.get("rank"), int)
        for row in ordered
    ) or [int(row.get("rank", -1)) for row in ordered] != list(range(world_size)):
        raise ValueError("Cumulative-health rank identities are incomplete")
    for row in ordered:
        try:
            row_through_update = _strict_nonnegative_counter(
                row.get("through_update"), name="through_update"
            )
        except ValueError as exc:
            raise ValueError("Rank cumulative-health boundary drift") from exc
        if (
            row.get("schema_version") != RANK_CUMULATIVE_HEALTH_SCHEMA
            or row_through_update != int(through_update)
        ):
            raise ValueError("Rank cumulative-health boundary drift")
    try:
        optimizer_counts = {
            _strict_nonnegative_counter(
                row["optimizer_update_count"],
                name="optimizer_update_count",
            )
            for row in ordered
        }
    except (KeyError, ValueError) as exc:
        raise ValueError("Cumulative optimizer-update count drift") from exc
    if optimizer_counts != {int(through_update)}:
        raise ValueError("Cumulative optimizer-update count drift")
    metadata: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []
    for row in ordered:
        hierarchy_state_for_metadata = row.get("hierarchy_supervision")
        if not isinstance(hierarchy_state_for_metadata, Mapping):
            raise ValueError("Rank hierarchy-delivery state is not a mapping")
        hierarchy_enabled_for_metadata = hierarchy_state_for_metadata.get(
            "enabled"
        )
        if not isinstance(hierarchy_enabled_for_metadata, bool):
            raise ValueError("Rank hierarchy-delivery enablement is invalid")
        try:
            source_cells, granularities = _canonical_source_metadata(
                row.get("source_cells"),
                hierarchy_enabled=hierarchy_enabled_for_metadata,
            )
            expected_microbatches = _strict_nonnegative_counter(
                row.get("expected_microbatches_per_update"),
                name="expected_microbatches_per_update",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Rank cumulative-health source metadata is invalid") from exc
        if expected_microbatches <= 0:
            raise ValueError("Rank cumulative-health microbatch denominator is invalid")
        if list(source_cells) != list(row.get("source_cells")) or list(
            granularities
        ) != list(row.get("granularities")):
            raise ValueError("Rank cumulative-health source metadata order drift")
        metadata.append((source_cells, granularities, expected_microbatches))
    if len(set(metadata)) != 1:
        raise ValueError("Ranks disagree on cumulative source metadata")
    source_cells, granularities, expected_microbatches_per_update = metadata[0]
    expected_rank_microbatches = int(through_update) * expected_microbatches_per_update
    if logical_batch_schedule is not None:
        if logical_batch_schedule.nominal_batch_size != expected_microbatches_per_update:
            raise ValueError("Cumulative nominal logical-batch denominator drift")
        expected_rank_microbatches = logical_batch_schedule.consumed_visits(through_update)
    schedule_values: list[tuple[int, ...]] = []
    for row in ordered:
        panel_state = row.get("gradient_panels")
        if not isinstance(panel_state, Mapping):
            raise ValueError("Rank gradient-panel state is missing")
        if set(panel_state) != {"schema_version", "scheduled_updates", "panels"}:
            raise ValueError("Rank gradient-panel canonical key set drift")
        if panel_state.get("schema_version") != GRADIENT_PANEL_CUMULATIVE_SCHEMA:
            raise ValueError("Rank gradient-panel schema drift")
        try:
            schedule_values.append(
                _canonical_panel_schedule(
                    panel_state.get("scheduled_updates"),
                    name="scheduled_updates",
                )
            )
        except ValueError as exc:
            raise ValueError("Rank gradient-panel schedule is invalid") from exc
    schedules = set(schedule_values)
    if len(schedules) != 1:
        raise ValueError("Ranks disagree on the gradient-panel schedule")
    scheduled_updates = list(next(iter(schedules)))

    clipping_states: list[Mapping[str, Any]] = []
    for row in ordered:
        clipping_state = row.get("clipping")
        if not isinstance(clipping_state, Mapping):
            raise ValueError("Rank clipping state is missing")
        if set(clipping_state) != {
            "definition",
            "local_rank_update_event_count",
            "local_rank_clipped_update_count",
            "any_rank_clipped_update_count",
            "all_ranks_unclipped_update_count",
        }:
            raise ValueError("Rank clipping canonical key set drift")
        if clipping_state.get("definition") != (
            "any_rank_was_clipped_per_optimizer_update"
        ):
            raise ValueError("Rank clipping definition drift")
        clipping_states.append(clipping_state)
    try:
        any_rank_counts = {
            _strict_nonnegative_counter(
                state["any_rank_clipped_update_count"],
                name="clipping.any_rank_clipped_update_count",
            )
            for state in clipping_states
        }
        all_unclipped_counts = {
            _strict_nonnegative_counter(
                state["all_ranks_unclipped_update_count"],
                name="clipping.all_ranks_unclipped_update_count",
            )
            for state in clipping_states
        }
    except (KeyError, ValueError) as exc:
        raise ValueError("Rank clipping counters are invalid") from exc
    if len(any_rank_counts) != 1 or len(all_unclipped_counts) != 1:
        raise ValueError("Ranks disagree on global clipping events")
    clipped_update_count = next(iter(any_rank_counts))
    unclipped_update_count = next(iter(all_unclipped_counts))
    if (
        clipped_update_count > int(through_update)
        or unclipped_update_count > int(through_update)
    ):
        raise ValueError("Cumulative clipping counts are invalid")
    if clipped_update_count + unclipped_update_count != int(through_update):
        raise ValueError("Cumulative clipping denominator drift")
    try:
        per_rank_clipped = [
            _strict_nonnegative_counter(
                state["local_rank_clipped_update_count"],
                name="clipping.local_rank_clipped_update_count",
            )
            for state in clipping_states
        ]
        per_rank_events = [
            _strict_nonnegative_counter(
                state["local_rank_update_event_count"],
                name="clipping.local_rank_update_event_count",
            )
            for state in clipping_states
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError("Rank clipping counters are invalid") from exc
    if any(
        events != int(through_update)
        or clipped < 0
        or clipped > events
        for events, clipped in zip(per_rank_events, per_rank_clipped)
    ):
        raise ValueError("Rank clipping event denominator drift")
    if sum(per_rank_clipped) < clipped_update_count:
        raise ValueError("Rank clipping events undercount any-rank events")

    token_states: list[Mapping[str, Any]] = []
    token_cell_sets: list[frozenset[str]] = []
    for rank, row in enumerate(ordered):
        token_state = row.get("token_pure_dec0")
        if not isinstance(token_state, Mapping):
            raise ValueError(f"Rank {rank} token-delivery state is missing")
        if set(token_state) != {
            "schema_version",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "per_cell",
        }:
            raise ValueError("Rank token-delivery canonical key set drift")
        if token_state.get("schema_version") != TOKEN_CUMULATIVE_HEALTH_SCHEMA:
            raise ValueError("Rank token-delivery schema drift")
        try:
            microbatch_count = _strict_nonnegative_counter(
                token_state.get("microbatch_count"),
                name="token.microbatch_count",
            )
            recorded_microbatch_count = _strict_nonnegative_counter(
                token_state.get("recorded_microbatch_count"),
                name="token.recorded_microbatch_count",
            )
            optimization_valid_count = _strict_nonnegative_counter(
                token_state.get("optimization_valid_count"),
                name="token.optimization_valid_count",
            )
            optimization_invalid_count = _strict_nonnegative_counter(
                token_state.get("optimization_invalid_count"),
                name="token.optimization_invalid_count",
            )
        except ValueError as exc:
            raise ValueError("Rank token-delivery counters are invalid") from exc
        if (
            recorded_microbatch_count > microbatch_count
            or optimization_valid_count + optimization_invalid_count
            != microbatch_count
            or microbatch_count != expected_rank_microbatches
        ):
            raise ValueError("Rank token-delivery count partition drift")
        per_cell = token_state.get("per_cell")
        if not isinstance(per_cell, Mapping) or tuple(per_cell) != source_cells:
            raise ValueError("Rank token-delivery cell set drift")
        token_cell_sets.append(frozenset(str(cell) for cell in per_cell))
        for cell in per_cell:
            cell_state = per_cell.get(cell)
            if not isinstance(cell_state, Mapping):
                raise ValueError("Rank token-delivery cell state is missing")
            if set(cell_state) != set(TOKEN_CUMULATIVE_COUNT_KEYS):
                raise ValueError("Rank token-delivery counter key set drift")
            try:
                cell_counts = {
                    key: _strict_nonnegative_counter(
                        cell_state.get(key), name=f"token.{cell}.{key}"
                    )
                    for key in TOKEN_CUMULATIVE_COUNT_KEYS
                }
            except ValueError as exc:
                raise ValueError("Rank token-delivery cell counters are invalid") from exc
            if (
                cell_counts["retained_groups"] > cell_counts["input_groups"]
                or cell_counts["retained_proposals"]
                > cell_counts["input_proposals"]
                or cell_counts["realized_positive_pairs"]
                > cell_counts["requested_token_pairs"]
            ):
                raise ValueError("Rank token-delivery cell counter order drift")
        token_states.append(token_state)

    if len(set(token_cell_sets)) != 1:
        raise ValueError("Ranks disagree on token-delivery cell set")
    token_template = token_states[0]
    token_cells = tuple(token_template["per_cell"])
    token = {
        "schema_version": TOKEN_CUMULATIVE_HEALTH_SCHEMA,
        "expected_microbatch_count": expected_rank_microbatches * int(world_size),
        **{
            key: sum(int(state[key]) for state in token_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        },
        "per_cell": {
            cell: {
                key: sum(
                    int(state["per_cell"][cell][key]) for state in token_states
                )
                for key in TOKEN_CUMULATIVE_COUNT_KEYS
            }
            for cell in token_cells
        },
    }
    token["optimization_valid_fraction"] = int(
        token["optimization_valid_count"]
    ) / max(int(token["microbatch_count"]), 1)
    for cell_summary in token["per_cell"].values():
        cell_summary["group_survival_fraction"] = int(
            cell_summary["retained_groups"]
        ) / max(int(cell_summary["input_groups"]), 1)
        cell_summary["proposal_survival_fraction"] = int(
            cell_summary["retained_proposals"]
        ) / max(int(cell_summary["input_proposals"]), 1)
        cell_summary["realized_pair_yield"] = int(
            cell_summary["realized_positive_pairs"]
        ) / max(int(cell_summary["requested_token_pairs"]), 1)

    hierarchy_states: list[Mapping[str, Any]] = []
    hierarchy_cell_sets: list[frozenset[str]] = []
    hierarchy_stage_sets: list[frozenset[str]] = []
    for rank, row in enumerate(ordered):
        hierarchy_state = row.get("hierarchy_supervision")
        if not isinstance(hierarchy_state, Mapping):
            raise ValueError(f"Rank {rank} hierarchy-delivery state is missing")
        if hierarchy_state.get("schema_version") != HIERARCHY_CUMULATIVE_HEALTH_SCHEMA:
            raise ValueError("Rank hierarchy-delivery schema drift")
        if set(hierarchy_state) != {
            "schema_version",
            "enabled",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "vicreg_valid_count",
            "vicreg_invalid_count",
            "per_cell",
            "per_stage",
            "per_granularity_stage",
        }:
            raise ValueError("Rank hierarchy-delivery canonical key set drift")
        if not isinstance(hierarchy_state.get("enabled"), bool):
            raise ValueError("Rank hierarchy-delivery enablement is invalid")
        for key in (
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "vicreg_valid_count",
            "vicreg_invalid_count",
        ):
            _strict_nonnegative_counter(
                hierarchy_state.get(key), name=f"hierarchy.{key}"
            )
        if (
            int(hierarchy_state["recorded_microbatch_count"])
            > int(hierarchy_state["microbatch_count"])
            or int(hierarchy_state["optimization_valid_count"])
            + int(hierarchy_state["optimization_invalid_count"])
            != int(hierarchy_state["microbatch_count"])
            or int(hierarchy_state["vicreg_valid_count"])
            + int(hierarchy_state["vicreg_invalid_count"])
            != int(hierarchy_state["microbatch_count"])
            or (
                bool(hierarchy_state["enabled"])
                and int(hierarchy_state["microbatch_count"])
                != expected_rank_microbatches
            )
            or (
                not bool(hierarchy_state["enabled"])
                and any(
                    int(hierarchy_state[key]) != 0
                    for key in (
                        "microbatch_count",
                        "recorded_microbatch_count",
                        "optimization_valid_count",
                        "optimization_invalid_count",
                        "vicreg_valid_count",
                        "vicreg_invalid_count",
                    )
                )
            )
        ):
            raise ValueError("Rank hierarchy-delivery count partition drift")
        per_cell = hierarchy_state.get("per_cell")
        per_stage = hierarchy_state.get("per_stage")
        per_granularity_stage = hierarchy_state.get("per_granularity_stage")
        if not isinstance(per_cell, Mapping) or tuple(per_cell) != granularities:
            raise ValueError("Rank hierarchy-delivery cell set drift")
        if not isinstance(per_stage, Mapping) or tuple(per_stage) != HIERARCHY_HEALTH_STAGES:
            raise ValueError("Rank hierarchy-delivery stage set drift")
        if not isinstance(per_granularity_stage, Mapping) or tuple(
            per_granularity_stage
        ) != tuple(HIERARCHY_ROUTE_STAGES):
            raise ValueError("Rank hierarchy-delivery route set drift")
        hierarchy_cell_sets.append(frozenset(str(cell) for cell in per_cell))
        hierarchy_stage_sets.append(frozenset(str(stage) for stage in per_stage))
        for cell, cell_state in per_cell.items():
            if not isinstance(cell_state, Mapping):
                raise ValueError("Rank hierarchy-delivery cell state is missing")
            if set(cell_state) != {
                "visit_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "input_groups",
                "input_proposals",
                "assigned_proposals",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
            }:
                raise ValueError("Rank hierarchy-delivery cell key set drift")
            for key in (
                "visit_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "input_groups",
                "input_proposals",
                "assigned_proposals",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
            ):
                _strict_nonnegative_counter(
                    cell_state.get(key), name=f"hierarchy.{cell}.{key}"
                )
            if (
                bool(hierarchy_state["enabled"])
                and int(cell_state["visit_count"]) != expected_rank_microbatches
            ):
                raise ValueError("Rank hierarchy-delivery cell visit denominator drift")
            if (
                int(cell_state["optimization_valid_count"])
                + int(cell_state["optimization_invalid_count"])
                != int(cell_state["visit_count"])
                or int(cell_state["mask_proposals_valid"])
                > int(cell_state["mask_proposals_attempted"])
                or int(cell_state["mask_folds_valid"])
                > int(cell_state["mask_folds_attempted"])
                or int(cell_state["mask_valid_fold_denominator"])
                > int(cell_state["mask_folds_valid"])
            ):
                raise ValueError("Rank hierarchy-delivery cell counter drift")
        for granularity in HIERARCHY_ROUTE_STAGES:
            edge_states = per_granularity_stage[granularity]
            expected_edges = HIERARCHY_ROUTE_STAGES[granularity]
            if not isinstance(edge_states, Mapping) or tuple(edge_states) != expected_edges:
                raise ValueError(
                    f"Rank hierarchy-delivery route stage set drift for {granularity}"
                )
            for stage in expected_edges:
                edge_state = edge_states[stage]
                if not isinstance(edge_state, Mapping):
                    raise ValueError("Rank hierarchy-delivery route state is missing")
                for key in (
                    "visit_count",
                    "route_active_count",
                    "route_required_count",
                    "optimization_valid_count",
                    "optimization_invalid_count",
                    *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                ):
                    _strict_nonnegative_counter(
                        edge_state.get(key),
                        name=f"hierarchy.{granularity}.{stage}.{key}",
                    )
                if (
                    bool(hierarchy_state["enabled"])
                    and int(edge_state["visit_count"]) != expected_rank_microbatches
                ):
                    raise ValueError(
                        "Rank hierarchy-delivery route visit denominator drift"
                    )
                if (
                    int(edge_state["optimization_valid_count"])
                    + int(edge_state["optimization_invalid_count"])
                    != int(edge_state["visit_count"])
                    or int(edge_state["route_active_count"])
                    > int(edge_state["visit_count"])
                    or int(edge_state["route_required_count"])
                    > int(edge_state["visit_count"])
                    or int(edge_state["mask_proposals_valid"])
                    > int(edge_state["mask_proposals_attempted"])
                    or int(edge_state["mask_folds_valid"])
                    > int(edge_state["mask_folds_attempted"])
                    or int(edge_state["mask_valid_fold_denominator"])
                    > int(edge_state["mask_folds_valid"])
                ):
                    raise ValueError("Rank hierarchy-delivery route counter drift")
        for stage, stage_state in per_stage.items():
            if not isinstance(stage_state, Mapping):
                raise ValueError("Rank hierarchy-delivery stage state is missing")
            if set(stage_state) != {
                "visit_count",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                "vicreg_valid_count",
                "vicreg_invalid_count",
            }:
                raise ValueError("Rank hierarchy-delivery stage key set drift")
            for key in (
                "visit_count",
                *HIERARCHY_MASK_CUMULATIVE_COUNT_KEYS,
                "vicreg_valid_count",
                "vicreg_invalid_count",
            ):
                _strict_nonnegative_counter(
                    stage_state.get(key), name=f"hierarchy.{stage}.{key}"
                )
            if (
                bool(hierarchy_state["enabled"])
                and int(stage_state["visit_count"]) != expected_rank_microbatches
            ):
                raise ValueError(
                    "Rank hierarchy-delivery stage visit denominator drift"
                )
            if (
                int(stage_state["vicreg_valid_count"])
                + int(stage_state["vicreg_invalid_count"])
                != int(stage_state["visit_count"])
                or int(stage_state["mask_proposals_valid"])
                > int(stage_state["mask_proposals_attempted"])
                or int(stage_state["mask_folds_valid"])
                > int(stage_state["mask_folds_attempted"])
                or int(stage_state["mask_valid_fold_denominator"])
                > int(stage_state["mask_folds_valid"])
            ):
                raise ValueError("Rank hierarchy-delivery stage counter drift")
        hierarchy_states.append(hierarchy_state)

    if len(set(hierarchy_cell_sets)) != 1:
        raise ValueError("Ranks disagree on hierarchy-delivery cell set")
    if len(set(hierarchy_stage_sets)) != 1:
        raise ValueError("Ranks disagree on hierarchy-delivery stage set")
    hierarchy_template = hierarchy_states[0]
    hierarchy_enabled = bool(hierarchy_template["enabled"])
    if any(bool(state["enabled"]) != hierarchy_enabled for state in hierarchy_states):
        raise ValueError("Ranks disagree on hierarchy-health enablement")
    hierarchy = {
        "schema_version": HIERARCHY_CUMULATIVE_HEALTH_SCHEMA,
        "enabled": hierarchy_enabled,
        "expected_microbatch_count": expected_rank_microbatches * int(world_size),
        **{
            key: sum(int(state[key]) for state in hierarchy_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
                "vicreg_valid_count",
                "vicreg_invalid_count",
            )
        },
        "per_cell": {
            cell: {
                key: sum(
                    int(state["per_cell"][cell][key])
                    for state in hierarchy_states
                )
                for key in cell_summary
            }
            for cell, cell_summary in hierarchy_template["per_cell"].items()
        },
        "per_stage": {
            stage: {
                key: sum(
                    int(state["per_stage"][stage][key])
                    for state in hierarchy_states
                )
                for key in stage_summary
            }
            for stage, stage_summary in hierarchy_template["per_stage"].items()
        },
        "per_granularity_stage": {
            granularity: {
                stage: {
                    key: sum(
                        int(
                            state["per_granularity_stage"][granularity][stage][key]
                        )
                        for state in hierarchy_states
                    )
                    for key in hierarchy_template["per_granularity_stage"][
                        granularity
                    ][stage]
                }
                for stage in HIERARCHY_ROUTE_STAGES[granularity]
            }
            for granularity in HIERARCHY_ROUTE_STAGES
        },
    }
    hierarchy["optimization_valid_fraction"] = int(
        hierarchy["optimization_valid_count"]
    ) / max(int(hierarchy["microbatch_count"]), 1)
    hierarchy["vicreg_valid_fraction"] = int(
        hierarchy["vicreg_valid_count"]
    ) / max(int(hierarchy["microbatch_count"]), 1)
    for cell_summary in hierarchy["per_cell"].values():
        cell_summary["optimization_valid_fraction"] = int(
            cell_summary["optimization_valid_count"]
        ) / max(int(cell_summary["visit_count"]), 1)
        cell_summary["mask_proposal_survival_fraction"] = int(
            cell_summary["mask_proposals_valid"]
        ) / max(int(cell_summary["mask_proposals_attempted"]), 1)
        cell_summary["mask_fold_survival_fraction"] = int(
            cell_summary["mask_folds_valid"]
        ) / max(int(cell_summary["mask_folds_attempted"]), 1)
    for stage_summary in hierarchy["per_stage"].values():
        stage_summary["mask_proposal_survival_fraction"] = int(
            stage_summary["mask_proposals_valid"]
        ) / max(int(stage_summary["mask_proposals_attempted"]), 1)
        stage_summary["mask_fold_survival_fraction"] = int(
            stage_summary["mask_folds_valid"]
        ) / max(int(stage_summary["mask_folds_attempted"]), 1)
        stage_summary["vicreg_valid_fraction"] = int(
            stage_summary["vicreg_valid_count"]
        ) / max(int(stage_summary["visit_count"]), 1)
    for edge_stages in hierarchy["per_granularity_stage"].values():
        for edge_summary in edge_stages.values():
            edge_summary["optimization_valid_fraction"] = int(
                edge_summary["optimization_valid_count"]
            ) / max(int(edge_summary["visit_count"]), 1)
            edge_summary["mask_proposal_survival_fraction"] = int(
                edge_summary["mask_proposals_valid"]
            ) / max(int(edge_summary["mask_proposals_attempted"]), 1)
            edge_summary["mask_fold_survival_fraction"] = int(
                edge_summary["mask_folds_valid"]
            ) / max(int(edge_summary["mask_folds_attempted"]), 1)

    legacy_states: list[Mapping[str, Any]] = []
    legacy_enabled_values: list[bool] = []
    for rank, row in enumerate(ordered):
        legacy_state = row.get("legacy_multiscale_delivery")
        if not isinstance(legacy_state, Mapping):
            raise ValueError(f"Rank {rank} legacy multiscale state is missing")
        if set(legacy_state) != {
            "schema_version",
            "enabled",
            "microbatch_count",
            "recorded_microbatch_count",
            "optimization_valid_count",
            "optimization_invalid_count",
            "per_stage",
        }:
            raise ValueError("Rank legacy multiscale canonical key set drift")
        if legacy_state.get("schema_version") != LEGACY_MULTISCALE_CUMULATIVE_SCHEMA:
            raise ValueError("Rank legacy multiscale schema drift")
        enabled = legacy_state.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Rank legacy multiscale enablement is invalid")
        legacy_enabled_values.append(enabled)
        counters = {
            key: _strict_nonnegative_counter(
                legacy_state.get(key), name=f"legacy.{key}"
            )
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        }
        if enabled:
            if counters["microbatch_count"] != expected_rank_microbatches:
                raise ValueError("Rank legacy multiscale denominator drift")
        elif any(counters.values()):
            raise ValueError("Disabled legacy multiscale state is non-empty")
        if (
            counters["recorded_microbatch_count"] > counters["microbatch_count"]
            or counters["optimization_valid_count"]
            + counters["optimization_invalid_count"]
            != counters["microbatch_count"]
            or (
                enabled
                and counters["recorded_microbatch_count"]
                != counters["optimization_valid_count"]
            )
        ):
            raise ValueError("Rank legacy multiscale count partition drift")
        per_stage = legacy_state.get("per_stage")
        if not isinstance(per_stage, Mapping) or tuple(per_stage) != (
            *LEGACY_MULTISCALE_CUMULATIVE_STAGES,
        ):
            raise ValueError("Rank legacy multiscale stage set drift")
        for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES:
            stage_state = per_stage[stage]
            if not isinstance(stage_state, Mapping) or set(stage_state) != set(
                LEGACY_MULTISCALE_CUMULATIVE_KEYS
            ):
                raise ValueError("Rank legacy multiscale metric key set drift")
            for key in LEGACY_MULTISCALE_INTEGER_KEYS:
                count = _strict_nonnegative_counter(
                    stage_state.get(key), name=f"legacy.{stage}.{key}"
                )
                if not enabled and count != 0:
                    raise ValueError("Disabled legacy multiscale metrics are non-empty")
            for key in LEGACY_MULTISCALE_FLOAT_KEYS:
                metric = _strict_finite_real(
                    stage_state.get(key), name=f"legacy.{stage}.{key}"
                )
                if not enabled and metric != 0.0:
                    raise ValueError("Disabled legacy multiscale metrics are non-empty")
            if (
                int(stage_state["valid_pairs"]) > int(stage_state["input_pairs"])
                or int(stage_state["kept_pairs"]) > int(stage_state["valid_pairs"])
                or int(stage_state["kept_proposals"])
                > int(stage_state["input_proposals"])
            ):
                raise ValueError("Rank legacy multiscale metric denominator drift")
        legacy_states.append(legacy_state)
    if len(set(legacy_enabled_values)) != 1:
        raise ValueError("Ranks disagree on legacy multiscale enablement")
    legacy_enabled = legacy_enabled_values[0]
    if hierarchy_enabled and legacy_enabled:
        raise ValueError(
            "Hierarchy and legacy multiscale delivery cannot both be enabled"
        )
    legacy = {
        "schema_version": LEGACY_MULTISCALE_CUMULATIVE_SCHEMA,
        "enabled": legacy_enabled,
        "expected_microbatch_count": (
            expected_rank_microbatches * int(world_size) if legacy_enabled else 0
        ),
        **{
            key: sum(int(state[key]) for state in legacy_states)
            for key in (
                "microbatch_count",
                "recorded_microbatch_count",
                "optimization_valid_count",
                "optimization_invalid_count",
            )
        },
        "per_stage": {
            stage: {
                key: (
                    sum(int(state["per_stage"][stage][key]) for state in legacy_states)
                    if key in LEGACY_MULTISCALE_INTEGER_KEYS
                    else sum(
                        float(state["per_stage"][stage][key])
                        for state in legacy_states
                    )
                )
                for key in LEGACY_MULTISCALE_CUMULATIVE_KEYS
            }
            for stage in LEGACY_MULTISCALE_CUMULATIVE_STAGES
        },
    }
    legacy["optimization_valid_fraction"] = int(
        legacy["optimization_valid_count"]
    ) / max(int(legacy["microbatch_count"]), 1)
    for stage_summary in legacy["per_stage"].values():
        stage_summary["pair_survival_fraction"] = int(
            stage_summary["kept_pairs"]
        ) / max(int(stage_summary["input_pairs"]), 1)
        stage_summary["proposal_survival_fraction"] = int(
            stage_summary["kept_proposals"]
        ) / max(int(stage_summary["input_proposals"]), 1)

    panels_by_update: dict[int, dict[str, Any]] = {}
    for row in ordered:
        panel_state = row.get("gradient_panels")
        if not isinstance(panel_state, Mapping):
            raise ValueError("Rank gradient-panel state is missing")
        panel_records = panel_state.get("panels")
        if not isinstance(panel_records, list):
            raise ValueError("Rank gradient-panel records are invalid")
        rank_panel_updates: set[int] = set()
        for panel in panel_records:
            if not isinstance(panel, Mapping):
                raise ValueError("Rank gradient-panel record is invalid")
            try:
                panel_update = _strict_nonnegative_counter(
                    panel.get("update"), name="gradient_panel.update"
                )
            except ValueError as exc:
                raise ValueError("Rank gradient-panel record update is invalid") from exc
            if panel_update in rank_panel_updates:
                raise ValueError("Duplicate gradient panel update in rank state")
            rank_panel_updates.add(panel_update)
            if panel_update not in scheduled_updates or panel_update > int(
                through_update
            ):
                raise ValueError("Gradient panel lies outside cumulative schedule")
            required = {
                "kind",
                "update",
                "global_shared_gradient",
                "controller",
                "conflict_threshold",
                "objective_stage_counts",
            }
            if panel.get("kind") != "gradient_health" or not required.issubset(
                panel
            ):
                raise ValueError("Incomplete cumulative gradient-panel record")
            if not isinstance(panel.get("global_shared_gradient"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel gradient report")
            if not isinstance(panel.get("controller"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel controller")
            if not isinstance(panel.get("objective_stage_counts"), Mapping):
                raise ValueError("Invalid cumulative gradient-panel stage counts")
            _strict_finite_real(
                panel.get("conflict_threshold"),
                name="gradient_panel.conflict_threshold",
            )
            copied = _json_copy(dict(panel))
            if panel_update in panels_by_update and panels_by_update[panel_update] != copied:
                raise ValueError("Ranks disagree on a cumulative gradient panel")
            panels_by_update[panel_update] = copied
    observed_updates = sorted(panels_by_update)
    expected_updates = [
        value for value in scheduled_updates if value <= int(through_update)
    ]
    missing_updates = sorted(set(expected_updates) - set(observed_updates))
    unexpected_updates = sorted(set(observed_updates) - set(expected_updates))
    controller = _json_copy(dict(gradient_conflict_controller))
    rank_event_count = int(through_update) * int(world_size)
    rank_clipped_event_count = sum(per_rank_clipped)
    return {
        "schema_version": CUMULATIVE_HEALTH_SCHEMA,
        **({"logical_batch_schedule": logical_batch_schedule.as_dict()}
           if logical_batch_schedule is not None else {}),
        "through_update": int(through_update),
        "optimizer_update_count": int(through_update),
        "world_size": int(world_size),
        "source_cells": list(source_cells),
        "granularities": list(granularities),
        "expected_microbatches_per_update": expected_microbatches_per_update,
        "clipping": {
            "definition": "any_rank_was_clipped_per_optimizer_update",
            "clipped_update_count": clipped_update_count,
            "unclipped_update_count": unclipped_update_count,
            "clipped_update_fraction": clipped_update_count
            / max(int(through_update), 1),
            "rank_update_event_count": rank_event_count,
            "rank_clipped_event_count": rank_clipped_event_count,
            "rank_update_clipping_fraction": rank_clipped_event_count
            / max(rank_event_count, 1),
            "per_rank_clipped_update_counts": per_rank_clipped,
        },
        # Keep the checkpoint-native names aligned with the evaluator's
        # cumulative_health/v1 fallback.  Rank-local state intentionally keeps
        # the model-facing names above because it is restored by the runner.
        "token_delivery": token,
        "hierarchy_delivery": hierarchy,
        "legacy_multiscale_delivery": legacy,
        "gradient_panels": {
            "schema_version": GRADIENT_PANEL_CUMULATIVE_SCHEMA,
            "scheduled_updates": scheduled_updates,
            "expected_updates": expected_updates,
            "observed_updates": observed_updates,
            "missing_updates": missing_updates,
            "unexpected_updates": unexpected_updates,
            "complete_through_update": not missing_updates and not unexpected_updates,
            "panels": [panels_by_update[value] for value in observed_updates],
        },
        "gradient_conflict_controller": controller,
        "auxiliary_budget": _json_copy(controller.get("auxiliary_budget")),
        "rank_states": [_json_copy(dict(row)) for row in ordered],
    }


def _restore_rank_cumulative_health(
    payload: Mapping[str, Any],
    *,
    rank: int,
    checkpoint_update: int,
    world_size: int,
    scheduled_gradient_panel_updates: Sequence[int],
    allow_schedule_extension: bool,
) -> dict[str, Any]:
    """Validate the aggregate record and restore this rank's exact counters."""

    if not isinstance(payload, Mapping):
        raise ValueError("Resume cumulative-health payload is not a mapping")
    if payload.get("schema_version") != CUMULATIVE_HEALTH_SCHEMA:
        raise ValueError("Resume cumulative-health schema drift")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or isinstance(checkpoint_update, bool)
        or not isinstance(checkpoint_update, int)
        or checkpoint_update < 0
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or rank >= world_size
        or not isinstance(allow_schedule_extension, bool)
    ):
        raise ValueError("Resume rank identity is invalid")
    rank_states = payload.get("rank_states")
    controller = payload.get("gradient_conflict_controller")
    if not isinstance(rank_states, list) or not isinstance(controller, Mapping):
        raise ValueError("Resume lacks exact cumulative rank/controller state")
    reconstructed = _merge_rank_cumulative_health(
        rank_states,
        through_update=checkpoint_update,
        world_size=world_size,
        gradient_conflict_controller=controller,
    )
    if reconstructed != payload:
        raise ValueError("Resume cumulative-health aggregate is corrupt")
    current_schedule = list(
        _canonical_panel_schedule(
            scheduled_gradient_panel_updates, name="current_schedule"
        )
    )
    previous_schedule = list(
        _canonical_panel_schedule(
            reconstructed["gradient_panels"]["scheduled_updates"],
            name="previous_schedule",
        )
    )
    observed_updates = list(reconstructed["gradient_panels"]["observed_updates"])
    expected_previous_updates = [
        value for value in previous_schedule if value <= int(checkpoint_update)
    ]
    if observed_updates != expected_previous_updates:
        raise ValueError(
            "Resume gradient-panel history is incomplete or non-canonical"
        )
    current_past_updates = [
        value for value in current_schedule if value <= int(checkpoint_update)
    ]
    if current_past_updates != expected_previous_updates:
        raise ValueError(
            "Resume gradient-panel history is not a subset of the current schedule"
        )
    if previous_schedule != current_schedule and not allow_schedule_extension:
        raise ValueError("Resume gradient-panel schedule differs")
    if not set(observed_updates).issubset(set(current_schedule)):
        raise ValueError("Resume has panels outside the current schedule")
    restored = _json_copy(rank_states[int(rank)])
    restored["gradient_panels"]["scheduled_updates"] = current_schedule
    return restored
