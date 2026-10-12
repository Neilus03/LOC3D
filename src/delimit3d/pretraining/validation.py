"""Pretraining validation extracted from the executed trainer."""

from __future__ import annotations


import math


from typing import Any, Mapping, Sequence

import numpy as np


from delimit3d.data.contrastive_sampler_v2 import DEFAULT_2D_CELLS as V2_DEFAULT_2D_CELLS
from delimit3d.data.partfield_contrastive import negative_policy_name


def _native_fixed_pack_cosine_gap(evaluation: Mapping[str, Any]) -> float:
    """Return legacy dense-point dec0 gap as a diagnostic-only value."""

    if evaluation.get("selection_view") != "fixed_comparable":
        raise ValueError("Feature-hard gating requires a fixed-comparable panel")
    native = evaluation.get("cell_metrics", {}).get("native", {})
    gaps = [
        float(metrics["cosine_gap"])
        for cell, metrics in native.items()
        if str(cell).startswith("2d/")
    ]
    if not gaps or not all(math.isfinite(value) for value in gaps):
        raise ValueError("Fixed panel lacks finite native 2D cosine gaps")
    return float(np.mean(gaps))


def _native_token_pure_fixed_pack_cosine_gap(
    evaluation: Mapping[str, Any],
    *,
    expected_cells: Sequence[str] = V2_DEFAULT_2D_CELLS,
) -> float:
    """Return the exact native dec0 token-space gap or fail closed."""

    if evaluation.get("selection_view") != "fixed_comparable":
        raise ValueError("Token-pure feature-hard gating requires a fixed panel")
    pack = evaluation.get("fixed_eval_pack")
    if not isinstance(pack, Mapping) or not isinstance(
        pack.get("token_pure_dec0_fixed_pack"), Mapping
    ):
        raise ValueError("Fixed panel is not a V2 token-pure pack")
    report = evaluation.get("native_fixed_comparable_dec0")
    expected_cells = tuple(expected_cells)
    if (
        not isinstance(report, Mapping)
        or report.get("valid") is not True
        or tuple(report.get("expected_cells", ())) != expected_cells
        or set(report.get("cells", {})) != set(expected_cells)
        or any(
            not bool(report["cells"][cell].get("valid", False))
            or int(report["cells"][cell].get("denominator", 0)) <= 0
            for cell in expected_cells
        )
    ):
        raise ValueError("Fixed panel lacks exact valid selected-cell token metrics")
    value = float(report.get("macro_cosine_gap", float("nan")))
    if not math.isfinite(value):
        raise ValueError("Fixed token-space cosine gap is non-finite")
    return value


def _summarize_v2_sampling(
    records: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[str],
    allow_shared_proposal_negatives: bool = False,
) -> dict[str, Any]:
    """Compact per-update delivery report without serializing coverage queues."""
    policy = negative_policy_name(allow_shared_proposal_negatives)
    expected_veto = not allow_shared_proposal_negatives
    for record in records:
        for group in record.get("groups", []):
            actual = group["sampler"].get("teacher_co_membership_filtered", True)
            if actual is not expected_veto:
                raise ValueError("Sampling telemetry disagrees with the configured negative policy")

    per_cell = {
        cell: {
            "requested_proposals": 0,
            "realized_unique_proposals": 0,
            "contrastive_retained_proposals": 0,
            "duplicate_proposals": 0,
            "frame_groups": 0,
            "realized_positive_pairs": 0,
            "requested_positive_pairs": 0,
            "visible_points": 0,
            "known_points": 0,
            "unknown_points": 0,
            "minimum_safe_negatives_per_pair": None,
        }
        for cell in cells
    }
    for record in records:
        coverage_by_cell = record["coverage_by_cell"]
        for cell in cells:
            coverage = coverage_by_cell[cell]
            row = per_cell[cell]
            row["requested_proposals"] += int(coverage["requested_count"])
            row["realized_unique_proposals"] += int(coverage["realized_count"])
            row["contrastive_retained_proposals"] += int(
                coverage["contrastive_retained_count"]
            )
            row["duplicate_proposals"] += int(coverage["duplicate_count"])
            row["frame_groups"] += int(coverage["frame_group_count"])
        for group in record.get("groups", []):
            row = per_cell[str(group["cell"])]
            sampler = group["sampler"]
            row["realized_positive_pairs"] += int(
                sampler["realized_positive_pairs"]
            )
            row["requested_positive_pairs"] += int(
                sampler["requested_positive_pairs_per_proposal"]
            ) * len(sampler["retained_proposal_indices"])
            row["visible_points"] += int(sampler["visible_count"])
            row["known_points"] += int(sampler["known_count"])
            row["unknown_points"] += int(sampler["unknown_count"])
            safe_min = int(sampler["safe_negatives_per_pair_min"])
            previous = row["minimum_safe_negatives_per_pair"]
            row["minimum_safe_negatives_per_pair"] = (
                safe_min if previous is None else min(int(previous), safe_min)
            )
    totals = {
        key: sum(int(row[key]) for row in per_cell.values())
        for key in (
            "requested_proposals",
            "realized_unique_proposals",
            "contrastive_retained_proposals",
            "duplicate_proposals",
            "frame_groups",
            "realized_positive_pairs",
            "requested_positive_pairs",
        )
    }
    totals["proposal_survival_fraction"] = totals[
        "contrastive_retained_proposals"
    ] / max(totals["realized_unique_proposals"], 1)
    totals["positive_pair_yield"] = totals["realized_positive_pairs"] / max(
        totals["requested_positive_pairs"], 1
    )
    realized_by_cell = [
        int(per_cell[cell]["realized_unique_proposals"]) for cell in cells
    ]
    total_realized = sum(realized_by_cell)
    per_cell_share = {
        cell: int(per_cell[cell]["realized_unique_proposals"])
        / max(total_realized, 1)
        for cell in cells
    }
    report = {
        "schema_version": "litept_multigranular_sampling_health/v2",
        "scene_visits": len(records),
        "optimization_valid_scene_visits": sum(
            int(bool(record.get("optimization_valid", False)))
            for record in records
        ),
        "optimization_invalid_scene_visits": sum(
            int(not bool(record.get("optimization_valid", False)))
            for record in records
        ),
        "optimization_valid_fraction": sum(
            int(bool(record.get("optimization_valid", False)))
            for record in records
        )
        / max(len(records), 1),
        "optimization_drop_reasons": {
            reason: sum(
                int(record.get("optimization_drop_reason") == reason)
                for record in records
            )
            for reason in sorted(
                {
                    str(record["optimization_drop_reason"])
                    for record in records
                    if record.get("optimization_drop_reason")
                }
            )
        },
        "feature_hard_enabled": None,
        "totals": totals,
        "per_cell_optimization_share": per_cell_share,
        "per_cell": per_cell,
        "hard_invariants": {
            "zero_duplicate_proposals": totals["duplicate_proposals"] == 0,
            "frame_local_groups": True,
            "teacher_comembership_filtered": True,
            "unknown_background_excluded": True,
        },
    }
    if allow_shared_proposal_negatives:
        del report["hard_invariants"]["teacher_comembership_filtered"]
        report["negative_policy"] = policy
        report["teacher_comembership_filter_enabled"] = False
        if any(group["sampler"].get("teacher_co_membership_filtered") is not False
               for record in records for group in record.get("groups", [])):
            raise ValueError("Sampling telemetry disagrees with the selected-proposal ablation")
    report["negative_policy"] = policy
    report["allow_shared_proposal_negatives"] = allow_shared_proposal_negatives
    report["hard_invariants"]["negative_policy_consistent"] = True
    return report


def _mean_numeric_tree(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average common scalar telemetry leaves while retaining stable contracts."""

    if not records:
        return {}
    output: dict[str, Any] = {}
    common_keys = set(records[0])
    for record in records[1:]:
        common_keys &= set(record)
    for key in sorted(common_keys):
        values = [record[key] for record in records]
        if all(isinstance(value, bool) for value in values):
            output[key] = all(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            output[key] = float(np.mean([float(value) for value in values]))
        elif all(isinstance(value, Mapping) for value in values):
            output[key] = _mean_numeric_tree(values)
        elif all(value == values[0] for value in values):
            output[key] = values[0]
    output["microbatch_count"] = len(records)
    return output


def _summarize_token_pure_records(
    records: Sequence[Mapping[str, Any]], *, cells: Sequence[str]
) -> dict[str, Any]:
    """Reduce token-space delivery with explicit denominators and worst cases."""

    ordered_cells = tuple(str(cell) for cell in cells)
    per_cell: dict[str, dict[str, Any]] = {
        cell: {
            "input_groups": 0,
            "retained_groups": 0,
            "input_proposals": 0,
            "retained_proposals": 0,
            "input_point_pairs": 0,
            "requested_token_pairs": 0,
            "realized_positive_pairs": 0,
            "zero_denominator_groups": 0,
            "minimum_pure_positive_survival_fraction": None,
            "minimum_safe_negative_tokens": None,
            "drop_reasons": {},
        }
        for cell in ordered_cells
    }
    worst_groups: list[dict[str, Any]] = []
    valid_visits = 0
    exact_three_cell_visits = 0
    for visit_index, record in enumerate(records):
        retained_cells = {str(cell) for cell in record.get("retained_cells", ())}
        optimization_valid = bool(
            record.get("optimization_valid", retained_cells == set(ordered_cells))
        )
        valid_visits += int(optimization_valid)
        exact_three_cell_visits += int(retained_cells == set(ordered_cells))
        for group_index, group in enumerate(record.get("groups", ())):
            cell = str(group.get("cell"))
            if cell not in per_cell:
                continue
            summary = per_cell[cell]
            summary["input_groups"] += 1
            realized_pairs = int(group.get("realized_positive_pairs", 0))
            summary["retained_groups"] += int(realized_pairs > 0)
            summary["input_proposals"] += int(group.get("input_proposals", 0))
            summary["retained_proposals"] += int(
                group.get("retained_proposals", 0)
            )
            summary["input_point_pairs"] += int(group.get("input_point_pairs", 0))
            summary["realized_positive_pairs"] += realized_pairs
            summary["zero_denominator_groups"] += int(
                bool(group.get("zero_denominator", realized_pairs <= 0))
            )
            requested_token_pairs = 0
            group_min_survival: float | None = None
            group_min_safe: int | None = None
            for proposal in group.get("proposals", ()):
                requested_token_pairs += int(
                    proposal.get("requested_token_pairs", 0)
                )
                survival = proposal.get("pure_positive_survival_fraction")
                if isinstance(survival, (int, float)) and math.isfinite(
                    float(survival)
                ):
                    group_min_survival = (
                        float(survival)
                        if group_min_survival is None
                        else min(group_min_survival, float(survival))
                    )
                safe = proposal.get("safe_negative_tokens_min")
                if isinstance(safe, (int, float)):
                    group_min_safe = (
                        int(safe)
                        if group_min_safe is None
                        else min(group_min_safe, int(safe))
                    )
                reason = proposal.get("drop_reason")
                if reason:
                    reasons = summary["drop_reasons"]
                    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1
            summary["requested_token_pairs"] += requested_token_pairs
            if group_min_survival is not None:
                previous = summary["minimum_pure_positive_survival_fraction"]
                summary["minimum_pure_positive_survival_fraction"] = (
                    group_min_survival
                    if previous is None
                    else min(float(previous), group_min_survival)
                )
            if group_min_safe is not None:
                previous_safe = summary["minimum_safe_negative_tokens"]
                summary["minimum_safe_negative_tokens"] = (
                    group_min_safe
                    if previous_safe is None
                    else min(int(previous_safe), group_min_safe)
                )
            worst_groups.append(
                {
                    "visit_index": visit_index,
                    "group_index": group_index,
                    "cell": cell,
                    "group_id": group.get("group_id"),
                    "input_proposals": int(group.get("input_proposals", 0)),
                    "retained_proposals": int(group.get("retained_proposals", 0)),
                    "requested_token_pairs": requested_token_pairs,
                    "realized_positive_pairs": realized_pairs,
                    "zero_denominator": bool(
                        group.get("zero_denominator", realized_pairs <= 0)
                    ),
                }
            )
    for summary in per_cell.values():
        summary["group_survival_fraction"] = int(
            summary["retained_groups"]
        ) / max(int(summary["input_groups"]), 1)
        summary["proposal_survival_fraction"] = int(
            summary["retained_proposals"]
        ) / max(int(summary["input_proposals"]), 1)
        summary["realized_pair_yield"] = int(
            summary["realized_positive_pairs"]
        ) / max(int(summary["requested_token_pairs"]), 1)
    worst_groups.sort(
        key=lambda row: (
            row["realized_positive_pairs"] / max(row["requested_token_pairs"], 1),
            row["retained_proposals"] / max(row["input_proposals"], 1),
            row["visit_index"],
            row["group_index"],
        )
    )
    return {
        "schema_version": "litept_token_pure_delivery_summary/v2",
        "microbatch_count": len(records),
        "optimization_valid_count": valid_visits,
        "optimization_valid_fraction": valid_visits / max(len(records), 1),
        "exact_three_cell_count": exact_three_cell_visits,
        "exact_three_cell_fraction": exact_three_cell_visits
        / max(len(records), 1),
        "per_cell": per_cell,
        "worst_groups": worst_groups[:12],
    }


def _summarize_hierarchy_records(
    records: Sequence[Mapping[str, Any]], *, cells: Sequence[str]
) -> dict[str, Any]:
    """Reduce mask/VICReg delivery without discarding variable rank evidence."""

    granularities = tuple(str(cell).rsplit("/", 1)[-1] for cell in cells)
    per_cell: dict[str, dict[str, Any]] = {
        granularity: {
            "frame_groups": 0,
            "valid_fold_denominator": 0,
            "active_visits": 0,
            "g02_routed_dec1_proposals": 0,
            "g02_routed_dec2_proposals": 0,
            "route_valid_visits": 0,
            "route_input_proposals": 0,
            "route_assigned_proposals": 0,
        }
        for granularity in granularities
    }
    stage_names = ("enc4", "dec3", "dec2", "dec1", "dec0")
    per_stage: dict[str, dict[str, Any]] = {
        stage: {
            "minimum_token_count": None,
            "mask_valid_fold_denominator": 0,
            "mask_proposals_attempted": 0,
            "mask_proposals_valid": 0,
            "mask_folds_attempted": 0,
            "mask_folds_valid": 0,
            "vicreg_all_valid": True,
            "vicreg_minimum_std": None,
            "vicreg_minimum_participation_rank": None,
            "vicreg_local_sample_count": 0,
            "vicreg_global_sample_count": 0,
            "vicreg_discarded_token_count": 0,
        }
        for stage in stage_names
    }
    optimization_valid_count = 0
    for record in records:
        optimization_valid_count += int(record.get("optimization_valid", True))
        mask = record.get("mask", {})
        for granularity, route_record in record.get(
            "route_stage_validity", {}
        ).items():
            if granularity not in per_cell:
                continue
            per_cell[granularity]["route_valid_visits"] += int(
                bool(route_record.get("optimization_valid", False))
            )
            per_cell[granularity]["route_input_proposals"] += int(
                route_record.get("input_proposal_count", 0)
            )
            per_cell[granularity]["route_assigned_proposals"] += int(
                route_record.get("assigned_proposal_count", 0)
            )
        for granularity, cell_record in mask.get("cells", {}).items():
            if granularity not in per_cell:
                continue
            summary = per_cell[granularity]
            summary["frame_groups"] += int(cell_record.get("group_count", 0))
            denominator = int(
                cell_record.get("loss_denominator_valid_folds", 0)
            )
            summary["valid_fold_denominator"] += denominator
            summary["active_visits"] += int(denominator > 0)
            if granularity == "g02":
                for route in cell_record.get("route", ()):
                    summary["g02_routed_dec1_proposals"] += int(
                        route.get("routed_dec1_proposals", 0)
                    )
                    summary["g02_routed_dec2_proposals"] += int(
                        route.get("routed_dec2_proposals", 0)
                    )
        for stage, stage_record in record.get("stages", {}).items():
            if stage not in per_stage:
                continue
            summary = per_stage[stage]
            token_count = int(stage_record.get("token_count", 0))
            summary["minimum_token_count"] = (
                token_count
                if summary["minimum_token_count"] is None
                else min(int(summary["minimum_token_count"]), token_count)
            )
            mask_stage = stage_record.get("mask", {})
            summary["mask_valid_fold_denominator"] += int(
                mask_stage.get("loss_denominator_valid_folds", 0)
            )
            for source, target in (
                ("proposals_attempted", "mask_proposals_attempted"),
                ("proposals_valid", "mask_proposals_valid"),
                ("folds_attempted", "mask_folds_attempted"),
                ("folds_valid", "mask_folds_valid"),
            ):
                summary[target] += int(mask_stage.get(source, 0))
            vicreg = stage_record.get("vicreg", {})
            summary["vicreg_all_valid"] = bool(
                summary["vicreg_all_valid"] and vicreg.get("valid", True)
            )
            for source, target in (
                ("local_sampled_token_count", "vicreg_local_sample_count"),
                ("global_sampled_token_count", "vicreg_global_sample_count"),
            ):
                summary[target] += int(vicreg.get(source, 0))
            summary["vicreg_discarded_token_count"] += sum(
                int(value)
                for value in vicreg.get("per_rank_discarded_token_counts", ())
            )
            for source, target in (
                ("component_std_min", "vicreg_minimum_std"),
                ("participation_rank", "vicreg_minimum_participation_rank"),
            ):
                value = vicreg.get(source)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    summary[target] = (
                        float(value)
                        if summary[target] is None
                        else min(float(summary[target]), float(value))
                    )
    for summary in per_stage.values():
        summary["mask_proposal_survival_fraction"] = int(
            summary["mask_proposals_valid"]
        ) / max(int(summary["mask_proposals_attempted"]), 1)
        summary["mask_fold_survival_fraction"] = int(
            summary["mask_folds_valid"]
        ) / max(int(summary["mask_folds_attempted"]), 1)
    for summary in per_cell.values():
        summary["route_valid_fraction"] = int(
            summary["route_valid_visits"]
        ) / max(len(records), 1)
        summary["route_assignment_fraction"] = int(
            summary["route_assigned_proposals"]
        ) / max(int(summary["route_input_proposals"]), 1)
    return {
        "schema_version": "litept_hierarchy_delivery_summary/v2",
        "microbatch_count": len(records),
        "optimization_valid_count": optimization_valid_count,
        "optimization_valid_fraction": optimization_valid_count
        / max(len(records), 1),
        "per_cell": per_cell,
        "per_stage": per_stage,
    }
