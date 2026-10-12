"""Mask3D proposal scoring and ScanNet AP helpers.

The legacy class-agnostic functions remain available for the geometry gate.
The class-aware helpers use the upstream convention: softmax over foreground
classes plus no-object, select the best foreground class per query, and
multiply that class probability by mask quality.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from delimit3d.metrics.official_instance_ap import (
    SCANNET_IGNORE_MODE,
    SCANNET_MIN_REGION_SIZE,
    build_instance_ap_records,
    evaluate_official_instance_ap,
    merge_ap_record_sets,
)


def expand_segment_logits_to_vertices(
    segment_mask_logits: torch.Tensor,
    point_to_segment: torch.Tensor,
) -> torch.Tensor:
    """Expand query-by-segment logits to the original mesh vertex order."""
    if segment_mask_logits.ndim != 2:
        raise ValueError("segment_mask_logits must have shape [Q,S]")
    if point_to_segment.ndim != 1:
        raise ValueError("point_to_segment must have shape [N]")
    if point_to_segment.numel() and (
        int(point_to_segment.min().item()) < 0
        or int(point_to_segment.max().item()) >= int(segment_mask_logits.shape[1])
    ):
        raise ValueError("point_to_segment contains an out-of-range segment index")
    return segment_mask_logits[:, point_to_segment.long()]


def _mask3d_proposal_scores(
    mask_logits: torch.Tensor,
    positive_masks: torch.Tensor,
    object_logits: torch.Tensor,
) -> torch.Tensor:
    """Reproduce Mask3D's class-probability times mask-quality score."""
    if mask_logits.shape != positive_masks.shape:
        raise ValueError("mask logits and positive masks must have identical shapes")
    mask_probabilities = mask_logits.float().sigmoid()
    positive = positive_masks.to(dtype=mask_probabilities.dtype)
    mask_quality = (mask_probabilities * positive).sum(dim=1) / (
        positive.sum(dim=1) + 1.0e-6
    )
    class_probabilities = object_logits.float().softmax(dim=-1)
    foreground_probabilities = class_probabilities[:, :-1]
    if foreground_probabilities.shape[1] == 0:
        raise ValueError("object_logits must contain at least one foreground class")
    object_probability = foreground_probabilities.max(dim=1).values
    return object_probability * mask_quality


def _proposal_scores_and_classes(
    mask_logits: torch.Tensor,
    positive_masks: torch.Tensor,
    object_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Mask3D scores and the winning foreground class per query."""
    if object_logits.ndim != 2 or object_logits.shape[1] < 2:
        raise ValueError("object_logits must have shape [Q, foreground_classes+1]")
    if mask_logits.shape != positive_masks.shape:
        raise ValueError("mask logits and positive masks must have identical shapes")
    mask_probabilities = mask_logits.float().sigmoid()
    positive = positive_masks.to(dtype=mask_probabilities.dtype)
    mask_quality = (mask_probabilities * positive).sum(dim=1) / (
        positive.sum(dim=1) + 1.0e-6
    )
    class_probabilities = object_logits.float().softmax(dim=-1)
    foreground_probabilities = class_probabilities[:, :-1]
    class_probability, class_ids = foreground_probabilities.max(dim=1)
    return class_probability * mask_quality, class_ids


def _query_proposals_impl(
    prediction: dict[str, Any],
    *,
    topk: int | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    mask_logits = prediction.get("mask_logits")
    if mask_logits is None:
        mask_logits = expand_segment_logits_to_vertices(
            prediction["segment_mask_logits"], prediction["point_to_segment"]
        )
    mask_logits = mask_logits.detach().cpu().float()
    positive_masks = mask_logits > 0.0
    scores, class_ids = _proposal_scores_and_classes(
        mask_logits,
        positive_masks,
        prediction["object_logits"].detach().cpu(),
    )
    masks = positive_masks.numpy()
    scores_np = scores.numpy()
    class_ids_np = class_ids.numpy().astype(np.int64, copy=False)
    query_indices = np.arange(masks.shape[0], dtype=np.int64)
    if topk is not None:
        keep = np.argsort(-scores_np, kind="stable")[: min(int(topk), len(scores_np))]
        masks = masks[keep]
        scores_np = scores_np[keep]
        class_ids_np = class_ids_np[keep]
        query_indices = query_indices[keep]
    return (
        [masks[query_index] for query_index in range(masks.shape[0])],
        scores_np.astype(np.float64, copy=False),
        class_ids_np,
        query_indices,
    )


def raw_query_proposals(
    prediction: dict[str, Any],
    *,
    topk: int | None = 100,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if "object_logits" not in prediction:
        logits = prediction.get("mask_logits")
        if logits is None:
            logits = expand_segment_logits_to_vertices(
                prediction["segment_mask_logits"], prediction["point_to_segment"])
        logits = logits.detach().cpu().float()
        masks = logits > 0
        scores = (logits.sigmoid()*masks).sum(1)/(masks.sum(1)+1e-6)
        order = torch.argsort(scores, descending=True, stable=True)
        if topk is not None: order = order[:topk]
        return [row.numpy() for row in masks[order]], scores[order].numpy(), order.numpy()
    proposals, scores, _class_ids, query_indices = _query_proposals_impl(
        prediction, topk=topk
    )
    return proposals, scores, query_indices


def raw_query_proposals_with_classes(
    prediction: dict[str, Any],
    *,
    topk: int | None = 100,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return raw query masks, scores, foreground classes, and query ids."""
    return _query_proposals_impl(prediction, topk=topk)


def _dbscan_split_query_proposals_impl(
    prediction: dict[str, Any],
    xyz: np.ndarray | torch.Tensor,
    *,
    eps: float,
    min_samples: int,
    topk: int | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise RuntimeError("DBSCAN postprocessing requires scikit-learn") from exc

    raw_masks, _raw_scores, raw_class_ids, raw_indices = _query_proposals_impl(
        prediction, topk=topk
    )
    mask_logits = prediction.get("mask_logits")
    if mask_logits is None:
        mask_logits = expand_segment_logits_to_vertices(
            prediction["segment_mask_logits"], prediction["point_to_segment"]
        )
    mask_probabilities = mask_logits.detach().cpu().float().sigmoid().numpy()
    class_probabilities = (
        prediction["object_logits"].detach().cpu().float().softmax(dim=-1).numpy()
    )
    coordinates = np.asarray(
        xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz,
        dtype=np.float32,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {coordinates.shape}")

    proposals: list[np.ndarray] = []
    scores: list[float] = []
    class_ids: list[int] = []
    query_indices: list[int] = []
    for mask, class_id, query_index in zip(raw_masks, raw_class_ids, raw_indices):
        if mask.shape != (coordinates.shape[0],):
            raise ValueError("Prediction and XYZ vertex counts differ")
        positive_indices = np.flatnonzero(mask)
        if positive_indices.size == 0:
            continue
        labels = DBSCAN(
            eps=float(eps), min_samples=int(min_samples), n_jobs=-1
        ).fit(coordinates[positive_indices]).labels_
        for cluster_id in np.unique(labels):
            if int(cluster_id) < 0:
                continue
            component = np.zeros(coordinates.shape[0], dtype=bool)
            component[positive_indices[labels == cluster_id]] = True
            proposals.append(component)
            component_quality = float(
                mask_probabilities[int(query_index), component].mean()
            )
            scores.append(
                float(class_probabilities[int(query_index), int(class_id)])
                * component_quality
            )
            class_ids.append(int(class_id))
            query_indices.append(int(query_index))
    score_array = np.asarray(scores, dtype=np.float64)
    class_array = np.asarray(class_ids, dtype=np.int64)
    query_array = np.asarray(query_indices, dtype=np.int64)
    if topk is None:
        keep = np.argsort(-score_array, kind="stable")
    else:
        keep = np.argsort(-score_array, kind="stable")[: min(int(topk), len(score_array))]
    return (
        [proposals[index] for index in keep],
        score_array[keep],
        class_array[keep],
        query_array[keep],
    )


def dbscan_split_query_proposals(
    prediction: dict[str, Any],
    xyz: np.ndarray | torch.Tensor,
    *,
    eps: float = 0.95,
    min_samples: int = 1,
    topk: int | None = 100,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Split each positive raw query into spatial DBSCAN components.

    Each component is scored by its parent query's class probability multiplied
    by the mean sigmoid mask confidence inside that component, matching
    Mask3D's postprocessing. Noise points are discarded.
    """
    proposals, scores, _class_ids, query_indices = _dbscan_split_query_proposals_impl(
        prediction, xyz, eps=eps, min_samples=min_samples, topk=topk
    )
    return proposals, scores, query_indices


def dbscan_split_query_proposals_with_classes(
    prediction: dict[str, Any],
    xyz: np.ndarray | torch.Tensor,
    *,
    eps: float = 0.95,
    min_samples: int = 1,
    topk: int | None = 100,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Class-aware DBSCAN proposal splitting with Mask3D scores."""
    return _dbscan_split_query_proposals_impl(
        prediction, xyz, eps=eps, min_samples=min_samples, topk=topk
    )


def _named_ca_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "CA-AP": metrics["AP"],
        "CA-AP50": metrics["AP50"],
        "CA-AP25": metrics["AP25"],
        "protocol": "custom_class_agnostic_scannet20_things",
        "class_agnostic": True,
        "canonical_scannet_leaderboard_metric": False,
        **metrics,
    }


def _named_class_aware_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Name the canonical semantic-instance AP fields without overclaiming scope."""
    return {
        **metrics,
        "mAP": metrics["AP"],
        "mAP50": metrics["AP50"],
        "mAP25": metrics["AP25"],
        "protocol": "mask3d_class_aware_scannet20",
        "class_agnostic": False,
        "num_foreground_classes": 18,
        "canonical_scannet_leaderboard_metric": True,
    }


def evaluate_litept_mask3d_ca_ap(
    *,
    scene_ids: Sequence[str],
    predictions: Sequence[dict[str, Any]],
    gt_instance_ids: Sequence[np.ndarray],
    xyz: Sequence[np.ndarray | torch.Tensor] | None = None,
    report_dbscan: bool = True,
    dbscan_eps: float = 0.95,
    dbscan_min_samples: int = 1,
) -> dict[str, Any]:
    """Evaluate raw queries primarily and DBSCAN components secondarily."""
    if not (len(scene_ids) == len(predictions) == len(gt_instance_ids)):
        raise ValueError("scene_ids, predictions, and gt_instance_ids must align")
    if xyz is not None and len(xyz) != len(predictions):
        raise ValueError("xyz must align with predictions")

    raw_records: list[dict[str, Any]] = []
    dbscan_records: list[dict[str, Any]] = []
    for scene_index, (scene_id, prediction, gt_ids) in enumerate(
        zip(scene_ids, predictions, gt_instance_ids)
    ):
        proposals, scores, query_indices = raw_query_proposals(prediction)
        raw_records.append(
            build_instance_ap_records(
                scene_id=str(scene_id),
                gt_ids=np.asarray(gt_ids, dtype=np.int64),
                proposals=proposals,
                scores=scores,
                query_indices=query_indices,
                class_agnostic=True,
                min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
            )
        )
        if report_dbscan:
            if xyz is None:
                raise ValueError("xyz is required when report_dbscan=True")
            split_masks, split_scores, split_indices = dbscan_split_query_proposals(
                prediction,
                xyz[scene_index],
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
            )
            dbscan_records.append(
                build_instance_ap_records(
                    scene_id=str(scene_id),
                    gt_ids=np.asarray(gt_ids, dtype=np.int64),
                    proposals=split_masks,
                    scores=split_scores,
                    query_indices=split_indices,
                    class_agnostic=True,
                    min_valid_gt_points=SCANNET_MIN_REGION_SIZE,
                    min_valid_pred_points=SCANNET_MIN_REGION_SIZE,
                )
            )

    raw_metrics = evaluate_official_instance_ap(
        merge_ap_record_sets(raw_records),
        threshold_preset="scannet_official",
        ignore_mode=SCANNET_IGNORE_MODE,
    )
    result: dict[str, Any] = {
        "primary": "raw_queries",
        "raw_queries": _named_ca_metrics(raw_metrics),
    }
    if report_dbscan:
        dbscan_metrics = evaluate_official_instance_ap(
            merge_ap_record_sets(dbscan_records),
            threshold_preset="scannet_official",
            ignore_mode=SCANNET_IGNORE_MODE,
        )
        result["dbscan_secondary"] = {
            **_named_ca_metrics(dbscan_metrics),
            "eps": float(dbscan_eps),
            "min_samples": int(dbscan_min_samples),
        }
    return result


__all__ = [
    "dbscan_split_query_proposals",
    "dbscan_split_query_proposals_with_classes",
    "evaluate_litept_mask3d_ca_ap",
    "expand_segment_logits_to_vertices",
    "raw_query_proposals",
    "raw_query_proposals_with_classes",
]
