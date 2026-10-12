"""Native feature health and held-out evaluation."""

from __future__ import annotations
from delimit3d.data.partfield_contrastive import negative_policy_name


import random


from pathlib import Path


from typing import Any, Sequence

import numpy as np

import torch


from delimit3d.training.contracts import SOURCE_CELLS, TWO_D_CELLS, normalize_source_cells, stable_seed


from delimit3d.losses.litept_hierarchy_supervision import build_token_pure_dec0_contrastive_batch


from .contrastive import (
    build_pure_token_contrastive_batch,
)
from .fixed_eval import (
    _fixed_scene_batch_views,
)
from .model import (
    LitePTContrastiveModel,
)
from .sampling import (
    _holdout_batches,
)
from .settings import (
    MULTISCALE_LEVEL_NAMES,
    SamplingConfig,
)
from .sources import (
    SceneSource,
)


def feature_health(features: torch.Tensor, *, max_samples: int = 5000) -> dict[str, float]:
    with torch.no_grad():
        sample = features[:max_samples].float()
        centered = sample - sample.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False)
        covariance = centered.T @ centered / max(int(centered.shape[0]), 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        norms = torch.linalg.vector_norm(sample, dim=1)
        return {
            "feature_std_mean": float(std.mean()),
            "feature_std_min": float(std.min()),
            "effective_rank": float(entropy.exp()),
            "feature_norm_mean": float(norms.mean()),
            "feature_norm_std": float(norms.std(unbiased=False)),
            "feature_norm_min": float(norms.min()),
            "feature_norm_max": float(norms.max()),
        }


def evaluate_holdout(
    *,
    model: LitePTContrastiveModel,
    scenes: list[SceneSource],
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
    fixed_eval_pack: Path | None = None,
    fixed_eval_pack_report: dict[str, Any] | None = None,
    feature_spaces: tuple[str, ...] = ("native", "projected"),
    cells: Sequence[str] | None = None,
    allow_shared_proposal_negatives: bool | None = None,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    recorded_flag = (fixed_eval_pack_report or {}).get(
        "allow_shared_proposal_negatives",
        ((fixed_eval_pack_report or {}).get("token_pure_dec0_fixed_pack") or {}).get(
            "allow_shared_proposal_negatives", False
        ),
    )
    flag = recorded_flag if allow_shared_proposal_negatives is None else allow_shared_proposal_negatives
    negative_policy_name(flag)
    if fixed_eval_pack_report is not None and recorded_flag != flag:
        raise ValueError("Fixed evaluation negative policy differs from the training recipe")
    if flag and (fixed_eval_pack is None or fixed_eval_pack_report is None):
        raise ValueError("Selected-proposal fixed evaluation requires its validated, recorded pack")
    if not feature_spaces or any(
        name not in ("native", "projected") for name in feature_spaces
    ):
        raise ValueError(f"Invalid evaluation feature spaces: {feature_spaces}")
    model.eval()
    metric_names = (
        "loss_total",
        "triplet_ranking_accuracy",
        "hardest_negative_ranking_accuracy",
        "cosine_gap",
    )
    records: list[dict[str, Any]] = []
    health_rows: list[dict[str, float]] = []
    multiscale_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_mapping_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_native_health: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    multiscale_projected_health: dict[str, list[dict[str, float]]] = {
        name: [] for name in MULTISCALE_LEVEL_NAMES
    }
    token_pure_mapping_records: list[dict[str, Any]] = []
    token_pure_evaluation = bool(
        fixed_eval_pack_report is not None
        and isinstance(
            fixed_eval_pack_report.get("token_pure_dec0_fixed_pack"), dict
        )
    )
    selection_view = (
        "fixed_comparable" if fixed_eval_pack is not None else "adaptive_feature_hard"
    )
    with torch.no_grad():
        for scene in scenes:
            forward_seed = stable_seed(seed, scene.scene_id, "eval-forward")
            random.seed(forward_seed)
            np.random.seed(forward_seed % (2**32))
            torch.manual_seed(forward_seed)
            points = torch.from_numpy(scene.points).to(device)
            features = torch.from_numpy(scene.features).to(device)
            (
                backbone_output,
                native,
                projected,
                level_embeddings_by_name,
            ) = model.encode_with_multiscale(
                points, features, token_space=token_pure_evaluation
            )
            native = (
                backbone_output.scene_tokens
                if token_pure_evaluation
                else native
            ).float()
            projected = projected.float()
            health_rows.append(feature_health(native))
            if model.multiscale_supervision:
                for name, tokens in zip(
                    MULTISCALE_LEVEL_NAMES,
                    backbone_output.multi_scale_tokens,
                    strict=True,
                ):
                    multiscale_native_health[name].append(
                        feature_health(tokens.float())
                    )
                    multiscale_projected_health[name].append(
                        feature_health(level_embeddings_by_name[name].float())
                    )
            if fixed_eval_pack is None:
                batch_views = {
                    "adaptive_feature_hard": _holdout_batches(
                        scene=scene,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed=seed,
                        cells=active_cells,
                    )
                }
            else:
                batch_views = _fixed_scene_batch_views(
                    pack_dir=fixed_eval_pack,
                    scene=scene,
                    device=device,
                    sampling=sampling,
                )
            for view, batches in batch_views.items():
                for cell, group, batch in batches:
                    if batch.allow_shared_proposal_negatives != flag:
                        raise ValueError("Fixed-eval batch negative policy differs from its recorded pack")
                    evaluation_batch = batch
                    token_mapping: dict[str, Any] | None = None
                    if token_pure_evaluation:
                        evaluation_batch, token_mapping = (
                            build_token_pure_dec0_contrastive_batch(
                                batch,
                                backbone_output.inverse_map.long(),
                                backbone_output.scene_xyz,
                                max_positive_pairs_per_proposal=int(
                                    sampling.positive_pairs_per_proposal
                                ),
                            )
                        )
                        if view == "fixed_comparable":
                            token_pure_mapping_records.append(
                                {
                                    "scene_id": scene.scene_id,
                                    "cell": cell,
                                    "group": group,
                                    **token_mapping,
                                }
                            )
                        if evaluation_batch is None:
                            continue
                    embeddings_by_space = {
                        "native": native,
                        "projected": projected,
                    }
                    for feature_space in feature_spaces:
                        embeddings = embeddings_by_space[feature_space]
                        result = model.criterion(embeddings, evaluation_batch)
                        records.append(
                            {
                                "scene_id": scene.scene_id,
                                "cell": cell,
                                "group": group,
                                "view": view,
                                "feature_space": feature_space,
                                # Keep the denominator beside every metric:
                                # fixed V2 groups can contain different numbers
                                # of dec0-valid pairs after token remapping.
                                "positive_pair_denominator": int(
                                    evaluation_batch.num_pairs
                                ),
                                **{
                                    name: float(result[name])
                                    for name in metric_names
                                },
                            }
                        )
                    if model.multiscale_supervision:
                        for name, level_xyz, parent_map in zip(
                            MULTISCALE_LEVEL_NAMES,
                            backbone_output.multi_scale_xyz,
                            backbone_output.finest_to_multi_scale,
                            strict=True,
                        ):
                            level_batch, mapping_stats = (
                                build_pure_token_contrastive_batch(
                                    batch,
                                    parent_map[backbone_output.inverse_map.long()],
                                    level_xyz,
                                    level_name=name,
                                )
                            )
                            multiscale_mapping_records[name].append(
                                {
                                    key: float(value)
                                    for key, value in mapping_stats.items()
                                }
                            )
                            if level_batch is None:
                                continue
                            level_result = model.multiscale_criteria[name](
                                level_embeddings_by_name[name].float(),
                                level_batch,
                            )
                            multiscale_records[name].append(
                                {
                                    "loss": float(level_result["loss_total"]),
                                    "hardest_negative_ranking_accuracy": float(
                                        level_result[
                                            "hardest_negative_ranking_accuracy"
                                        ]
                                    ),
                                    "cosine_gap": float(level_result["cosine_gap"]),
                                    "temperature": float(
                                        level_result["temperature"]
                                    ),
                                }
                            )
            del (
                points,
                features,
                native,
                projected,
                backbone_output,
                level_embeddings_by_name,
            )

    view_metrics: dict[str, dict[str, Any]] = {}
    for view in sorted({str(row["view"]) for row in records}):
        cell_metrics: dict[str, dict[str, dict[str, float]]] = {}
        scene_cell_metrics: dict[
            str, dict[str, dict[str, dict[str, float]]]
        ] = {}
        macro: dict[str, dict[str, float]] = {}
        for feature_space in feature_spaces:
            cell_metrics[feature_space] = {}
            scene_cell_metrics[feature_space] = {
                scene.scene_id: {} for scene in scenes
            }
            for cell in active_cells:
                selected = [
                    row
                    for row in records
                    if (
                        row["view"] == view
                        and row["feature_space"] == feature_space
                        and row["cell"] == cell
                    )
                ]
                if not selected:
                    raise RuntimeError(
                        f"No holdout records for {view}/{feature_space}/{cell}"
                    )
                per_scene = {
                    scene.scene_id: [
                        row
                        for row in selected
                        if row["scene_id"] == scene.scene_id
                    ]
                    for scene in scenes
                }
                if any(not rows for rows in per_scene.values()) and not token_pure_evaluation:
                    missing = [
                        scene_id
                        for scene_id, rows in per_scene.items()
                        if not rows
                    ]
                    raise RuntimeError(
                        f"No holdout records for {view}/{feature_space}/{cell}: "
                        f"{missing}"
                    )
                for scene_id, rows in per_scene.items():
                    if not rows:
                        continue
                    scene_cell_metrics[feature_space][scene_id][cell] = {
                        name: float(np.mean([row[name] for row in rows]))
                        for name in metric_names
                    }
                cell_metrics[feature_space][cell] = {
                    name: float(
                        np.mean(
                            [
                                scene_cell_metrics[feature_space][scene.scene_id][cell][name]
                                for scene in scenes
                                if cell
                                in scene_cell_metrics[feature_space][scene.scene_id]
                            ]
                        )
                    )
                    for name in metric_names
                }
                cell_metrics[feature_space][cell][
                    "min_groups_per_scene"
                ] = float(min(len(rows) for rows in per_scene.values()))
                cell_metrics[feature_space][cell][
                    "max_groups_per_scene"
                ] = float(max(len(rows) for rows in per_scene.values()))
            hardest = {
                cell: cell_metrics[feature_space][cell][
                    "hardest_negative_ranking_accuracy"
                ]
                for cell in active_cells
            }
            two_d_values = [
                value
                for cell, value in hardest.items()
                if cell.startswith("2d/")
            ]
            three_d_values = [
                value
                for cell, value in hardest.items()
                if cell.startswith("3d/")
            ]
            family_macro: dict[str, float] = {
                "overall": float(np.mean(list(hardest.values()))),
            }
            if two_d_values:
                family_macro["2d"] = float(np.mean(two_d_values))
            if three_d_values:
                family_macro["3d"] = float(np.mean(three_d_values))
            macro[feature_space] = family_macro
        view_metrics[view] = {
            "native_selection_macro": macro["native"],
            "projected_diagnostic_macro": macro.get("projected"),
            "cell_metrics": cell_metrics,
            "scene_cell_metrics": scene_cell_metrics,
        }
    health = {
        key: float(
            np.min([row[key] for row in health_rows])
            if key in ("feature_std_min", "feature_norm_min")
            else np.max([row[key] for row in health_rows])
            if key == "feature_norm_max"
            else np.mean([row[key] for row in health_rows])
        )
        for key in health_rows[0]
    }
    model.train()
    cell_noun = (
        "six-cell"
        if tuple(active_cells) == SOURCE_CELLS
        else "2d-three-cell"
        if tuple(active_cells) == TWO_D_CELLS
        else f"{len(active_cells)}-cell"
    )
    selected_metrics = view_metrics[selection_view]
    fixed_dec0_cells = tuple(cell for cell in active_cells if cell in (*TWO_D_CELLS, "2d/sam2"))
    fixed_dec0_cell_metrics: dict[str, dict[str, Any]] = {}
    for cell in fixed_dec0_cells:
        mapping_rows = [
            row for row in token_pure_mapping_records if row["cell"] == cell
        ]
        rows = [
            row
            for row in records
            if row["view"] == "fixed_comparable"
            and row["feature_space"] == "native"
            and row["cell"] == cell
        ]
        denominator = sum(int(row["positive_pair_denominator"]) for row in rows)
        requested_pairs = sum(
            sum(
                int(proposal.get("requested_token_pairs", 0))
                for proposal in row.get("proposals", ())
            )
            for row in mapping_rows
        )
        realized_pairs = sum(
            int(row.get("realized_positive_pairs", 0)) for row in mapping_rows
        )
        valid_groups = sum(
            int(int(row.get("realized_positive_pairs", 0)) > 0)
            for row in mapping_rows
        )
        attempted_groups = len(mapping_rows)
        valid_group_fraction = valid_groups / max(attempted_groups, 1)
        fixed_dec0_cell_metrics[cell] = {
            "cosine_gap": (
                float(
                    sum(
                        float(row["cosine_gap"])
                        * int(row["positive_pair_denominator"])
                        for row in rows
                    )
                    / denominator
                )
                if denominator
                else 0.0
            ),
            "denominator": int(denominator),
            "requested_positive_pairs": int(requested_pairs),
            "realized_positive_pairs": int(realized_pairs),
            "pair_yield_fraction": realized_pairs / max(requested_pairs, 1),
            "attempted_groups": attempted_groups,
            "valid_groups": valid_groups,
            "valid_group_fraction": valid_group_fraction,
            "valid": bool(
                token_pure_evaluation
                and rows
                and denominator > 0
                and attempted_groups > 0
                and valid_group_fraction >= 0.95
            ),
        }
    fixed_dec0_valid = all(
        fixed_dec0_cell_metrics[cell]["valid"] for cell in fixed_dec0_cells
    )
    fixed_dec0_macro_cosine_gap = (
        float(
            np.mean(
                [
                    fixed_dec0_cell_metrics[cell]["cosine_gap"]
                    for cell in fixed_dec0_cells
                ]
            )
        )
        if fixed_dec0_valid
        else 0.0
    )
    fixed_dec0_report = {
        "schema_version": "litept_token_pure_dec0_fixed_eval/v1",
        "exact_native_dec0_token_space": token_pure_evaluation,
        "expected_cells": list(fixed_dec0_cells),
        "valid": fixed_dec0_valid,
        "macro_cosine_gap": fixed_dec0_macro_cosine_gap,
        "denominator": int(
            sum(
                fixed_dec0_cell_metrics[cell]["denominator"]
                for cell in fixed_dec0_cells
            )
        ),
        "denominators_by_cell": {
            cell: fixed_dec0_cell_metrics[cell]["denominator"]
            for cell in fixed_dec0_cells
        },
        "cells": fixed_dec0_cell_metrics,
    }
    multiscale_diagnostics: dict[str, dict[str, Any]] = {}
    for name in MULTISCALE_LEVEL_NAMES:
        mapping_rows = multiscale_mapping_records[name]
        valid_rows = multiscale_records[name]
        if not mapping_rows:
            continue

        def aggregate_health(rows: list[dict[str, float]]) -> dict[str, float]:
            if not rows:
                return {}
            return {
                key: float(
                    np.min([row[key] for row in rows])
                    if key in ("feature_std_min", "feature_norm_min")
                    else np.max([row[key] for row in rows])
                    if key == "feature_norm_max"
                    else np.mean([row[key] for row in rows])
                )
                for key in rows[0]
            }

        multiscale_diagnostics[name] = {
            "attempted_batches": len(mapping_rows),
            "valid_batches": len(valid_rows),
            "valid_batch_fraction": len(valid_rows) / len(mapping_rows),
            "conditional_metrics": (
                {
                    key: float(np.mean([row[key] for row in valid_rows]))
                    for key in valid_rows[0]
                }
                if valid_rows
                else {}
            ),
            "mapping": {
                key: float(np.mean([row[key] for row in mapping_rows]))
                for key in mapping_rows[0]
            },
            "native_health": aggregate_health(multiscale_native_health[name]),
            "projected_health": aggregate_health(
                multiscale_projected_health[name]
            ),
        }
    return {
        "selection_view": selection_view,
        "selection_metric": (
            f"native 72D fixed-comparable {cell_noun} macro hardest-negative ranking"
            if selection_view == "fixed_comparable"
            else f"native 72D adaptive-hard {cell_noun} macro hardest-negative ranking"
        ),
        "native_selection_macro": selected_metrics["native_selection_macro"],
        "native_fixed_comparable_dec0_macro_cosine_gap": fixed_dec0_macro_cosine_gap,
        "native_fixed_comparable_dec0_macro_cosine_gap_valid": fixed_dec0_valid,
        "native_fixed_comparable_dec0_macro_cosine_gap_denominator": fixed_dec0_report[
            "denominator"
        ],
        "native_fixed_comparable_dec0": fixed_dec0_report,
        "projected_diagnostic_macro": selected_metrics.get(
            "projected_diagnostic_macro"
        ),
        "cell_metrics": selected_metrics["cell_metrics"],
        "scene_cell_metrics": selected_metrics["scene_cell_metrics"],
        "view_metrics": view_metrics,
        "native_health": health,
        "multiscale_diagnostics": multiscale_diagnostics,
        "fixed_eval_pack": fixed_eval_pack_report,
        "records": records,
    }
