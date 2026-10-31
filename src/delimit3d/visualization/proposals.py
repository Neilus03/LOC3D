"""Preserved Mask3D proposals and explicitly separate display diagnostics.

Mask-only exports need masks and stable query IDs, never a classification head.
Every original proposal, overlap, score and evaluated setting survives ingestion.
Display assignment does not participate in benchmark matching or AP.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import Observation, Panel, Scene
from .export_contracts import (PROPOSAL_SCHEMA, geometry_hash, pairing_status, read_bundle,
                               validate_geometry, validate_inverse)


@dataclass
class Proposals:
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    masks: np.ndarray  # Q x original N; independent masks, never a display segmentation
    probabilities: np.ndarray | None
    query_ids: np.ndarray
    scores: np.ndarray | None

    def display_segmentation(self, *, policy: str = "highest_score", threshold: float | None = None
                             ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resolve overlaps for display only; stable export order breaks exact score ties."""
        if policy not in {"highest_score", "first_query"}:
            raise ValueError("Display policy must be highest_score or first_query")
        if policy == "highest_score" and self.scores is None:
            raise ValueError("highest_score display needs exported scores; use explicit first_query policy")
        if threshold is not None and (not np.isfinite(threshold) or self.scores is None):
            raise ValueError("Display score threshold needs finite value and exported scores")
        accepted = np.ones(len(self.masks), dtype=bool)
        if threshold is not None:
            accepted &= self.scores >= threshold
        order = np.argsort(-self.scores, kind="stable") if policy == "highest_score" else np.arange(len(self.masks))
        labels = np.zeros(self.masks.shape[1], dtype=np.int64)
        for query in order:
            if accepted[query]:
                labels[self.masks[query] & (labels == 0)] = query + 1
        return labels, {"scope": "visualization_only", "official_benchmark_processing": False,
                        "policy": policy, "score_threshold": threshold,
                        "ties": "stable original export order", "background_id": 0,
                        "display_id_to_query_id": {str(i + 1): str(value) for i, value in enumerate(self.query_ids)}}

    def to_observation(self, name: str, *, policy: str | None = None) -> Observation:
        policy = policy or ("highest_score" if self.scores is not None else "first_query")
        labels, display = self.display_segmentation(policy=policy)
        confidence = None
        if self.scores is not None:
            confidence = np.zeros(len(labels), dtype=np.float64)
            selected = labels > 0
            confidence[selected] = self.scores[labels[selected] - 1]
        return Observation(name=name, prediction=labels, confidence=confidence, metadata={
            "adapter": "mask3d_proposals", "prediction_kind": "instance", "prediction_background_ids": [0],
            "proposals": self.record, "proposal_masks": self.masks, "query_ids": self.query_ids,
            "proposal_scores": self.scores, "display_processing": display,
            "confidence_semantics": self.record.get("score_semantics"), "confidence_calibrated": False})


def proposals_from_arrays(record: Mapping[str, Any], arrays: Mapping[str, Any], scene: Scene) -> Proposals:
    scene.validate()
    record, arrays = deepcopy(dict(record)), {key: np.asarray(value) for key, value in arrays.items()}
    if record.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError(f"Expected schema {PROPOSAL_SCHEMA}")
    validate_geometry(record, arrays, scene)
    if record.get("output_contract") not in {"mask_only", "class_aware"}:
        raise ValueError("output_contract must explicitly be mask_only or class_aware")
    if "masks" not in arrays or "query_ids" not in arrays:
        raise ValueError("Proposal exports require individual masks and persistent query_ids")
    native = arrays["masks"]
    if native.ndim != 2 or not (np.issubdtype(native.dtype, np.floating) or np.issubdtype(native.dtype, np.integer) or native.dtype == np.bool_) or not np.isfinite(native).all():
        raise ValueError("Proposal masks must be finite numeric/binary 2D arrays")
    if record.get("mask_layout") == "NQ":
        native = native.T
    elif record.get("mask_layout") != "QN":
        raise ValueError("mask_layout must explicitly be QN or NQ")
    query_ids = arrays["query_ids"]
    if query_ids.shape != (len(native),) or query_ids.dtype.kind not in "iuUS":
        raise ValueError("query_ids must be an integer/string vector matching mask query count")
    if len(set(map(str, query_ids.tolist()))) != len(query_ids):
        raise ValueError("Persistent query_ids must be unique; duplicates are separate queries")
    domain = record.get("prediction_domain")
    if domain in {"tokens", "segments"}:
        inverse = validate_inverse(arrays, len(scene.points), native.shape[1])
        expanded = native[:, inverse]
    elif domain == "raw_points" and native.shape[1] == len(scene.points):
        expanded = native
    else:
        raise ValueError("prediction_domain/count invalid; use raw_points or exact token/segment inverse map")
    semantics = record.get("mask_values")
    if semantics == "binary":
        if not np.isin(expanded, [0, 1]).all():
            raise ValueError("Binary masks contain values other than zero/one")
        masks, probabilities = expanded.astype(bool), None
    elif semantics in {"logits", "probabilities"}:
        threshold = record.get("mask_threshold")
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Nonbinary masks require recorded mask_threshold in probability space")
        if semantics == "probabilities" and (np.any(expanded < 0) or np.any(expanded > 1)):
            raise ValueError("Mask probabilities must lie in [0,1]")
        probabilities = (1 / (1 + np.exp(-np.clip(expanded.astype(np.float64), -80, 80)))) if semantics == "logits" else expanded
        comparator = record.get("threshold_comparator", "ge")
        if comparator not in {"ge", "gt"}:
            raise ValueError("threshold_comparator must be ge or gt")
        masks = probabilities >= threshold if comparator == "ge" else probabilities > threshold
    else:
        raise ValueError("mask_values must explicitly be binary, probabilities, or logits")
    scores = arrays.get("scores")
    if scores is not None:
        if scores.shape != (len(native),) or not (np.issubdtype(scores.dtype, np.floating) or np.issubdtype(scores.dtype, np.integer)) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite per-proposal values")
        if not record.get("score_semantics"):
            raise ValueError("Exported scores require recorded score_semantics")
    if record["output_contract"] == "mask_only" and ("class_logits" in arrays or "class_probabilities" in arrays):
        raise ValueError("mask_only export cannot claim classification-head outputs")
    for key in ("class_logits", "class_probabilities"):
        if key in arrays and (arrays[key].ndim != 2 or arrays[key].shape[0] != len(native)
                              or not np.isfinite(arrays[key]).all()):
            raise ValueError(f"{key} must retain finite [Q,C] query arrays")
    if "class_probabilities" in arrays and (np.any(arrays["class_probabilities"] < 0)
                                              or np.any(arrays["class_probabilities"] > 1)):
        raise ValueError("class_probabilities must lie in [0,1]")
    for key in ("class_ids", "selected_by_evaluator"):
        if key in arrays and arrays[key].shape != (len(native),):
            raise ValueError(f"{key} must have one entry per preserved proposal")
    if "class_ids" in arrays and not np.issubdtype(arrays["class_ids"].dtype, np.integer):
        raise ValueError("class_ids must contain integer category identities")
    if "selected_by_evaluator" in arrays and not np.isin(arrays["selected_by_evaluator"], [0, 1]).all():
        raise ValueError("selected_by_evaluator must be boolean")
    matching_missing = []
    matching = record.get("recorded_matching")
    if matching is not None:
        if not isinstance(matching, list):
            raise ValueError("recorded_matching must be a list of evaluator matching rows")
        known_queries = set(map(str, query_ids.tolist()))
        for row in matching:
            if not isinstance(row, Mapping) or str(row.get("query_id")) not in known_queries:
                raise ValueError("Recorded matching references an unknown query_id")
            if row.get("gt_id") is not None and scene.labels is not None and row["gt_id"] not in scene.labels:
                raise ValueError("Recorded matching references an unknown gt_id")
            for field in ("domain", "unit", "source"):
                if not row.get(field):
                    matching_missing.append(f"query {row['query_id']}: {field}")
            if row.get("domain") and row["domain"] not in {"raw_points", "tokens", "segments"}:
                raise ValueError("Recorded matching has unknown metric domain")
    record["matching_record_status"] = {"status": "unavailable" if matching is None else
                                        "incomplete" if matching_missing else "record_complete",
                                        "missing": matching_missing}
    missing = []
    if scores is None:
        missing.append("scores: ranking/score inspection unavailable")
    if not record.get("evaluated_postprocessing"):
        missing.append("evaluated_postprocessing: benchmark settings were not exported")
    record["missing_artifacts"] = missing
    record["pairing_status"] = pairing_status(record.get("model", {}))
    record["proposal_count"] = len(native)
    record["overlapping_point_count"] = int(np.count_nonzero(masks.sum(axis=0) > 1))
    return Proposals(record, arrays, masks, probabilities, query_ids, scores)


def load_proposals(path: str | Path, scene: Scene, *, expected_pairing: Mapping[str, Any] | None = None) -> Proposals:
    record, arrays = read_bundle(path)
    result = proposals_from_arrays(record, arrays, scene)
    result.record["pairing_status"] = pairing_status(result.record.get("model",{}), expected_pairing)
    return result


def proposals_from_mask3d_output(scene: Scene, output: Mapping[str, Any], config: Mapping[str, Any]) -> Proposals:
    """Bridge existing native/class-aware or Delimit3D mask-only outputs, without inference.

    Specify layout/value semantics. A native array's implicit row number is stored
    explicitly as its query identity. Score derivation is opt-in and labeled derived;
    supplied evaluator scores always take precedence.
    """
    from .adapters import _array
    mask_key = config.get("mask_key", "mask_logits" if config.get("output_contract") == "mask_only" else "pred_masks")
    value = output[mask_key]
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError("Mask3D output bridge supports exactly one scene")
        value = value[0]
    masks = _array(value)
    layout = config.get("mask_layout", "QN" if mask_key == "mask_logits" else "NQ")
    if masks.ndim != 2:
        raise ValueError("Mask3D output requires two-dimensional masks")
    count = masks.shape[0] if layout == "QN" else masks.shape[1]
    arrays = {"masks": masks, "query_ids": _array(output.get("query_ids", np.arange(count)))}
    for key in ("inverse_map", "representative_indices", "selected_by_evaluator", "class_ids"):
        if key in output:
            arrays[key] = _array(output[key])
    score_key = config.get("score_key", "scores")
    if score_key in output:
        arrays["scores"] = _array(output[score_key])
    contract = config.get("output_contract")
    if contract is None:
        contract = "class_aware" if config.get("class_key", "pred_logits") in output else "mask_only"
    class_key = config.get("class_key", "pred_logits")
    if contract == "class_aware" and class_key in output:
        classes = _array(output[class_key])
        if classes.ndim == 3 and classes.shape[0] == 1:
            classes = classes[0]
        arrays["class_logits"] = classes
    record = {"schema": PROPOSAL_SCHEMA, "scene_id": scene.scene_id,
              "source_geometry_sha256": geometry_hash(scene.points), "output_contract": contract,
              "mask_values": config.get("mask_values", "logits"), "mask_layout": layout,
              "mask_threshold": config.get("mask_threshold", .5),
              "threshold_comparator": config.get("threshold_comparator", "gt" if contract == "mask_only" else "ge"),
              "prediction_domain": config.get("prediction_domain", "tokens" if "inverse_map" in arrays else "raw_points"),
              "model": config.get("model", {}), "score_semantics": config.get("score_semantics"),
              "evaluated_postprocessing": config.get("evaluated_postprocessing"),
              "recorded_matching": config.get("recorded_matching"),
              "query_identity_source": "supplied query_ids" if "query_ids" in output else "original native query axis"}
    if "scores" not in arrays and config.get("derive_scores") == "mean_positive_mask_probability":
        if record["mask_values"] != "logits":
            raise ValueError("mean_positive_mask_probability derivation currently requires logits")
        native = masks if layout == "QN" else masks.T
        positive = native > 0
        probability = 1 / (1 + np.exp(-np.clip(native.astype(float), -80, 80)))
        arrays["scores"] = (positive * probability).sum(axis=1) / np.maximum(positive.sum(axis=1), 1e-6)
        record["score_semantics"] = "derived mean sigmoid(logit) over positive native mask logits; empty masks score zero"
        record["score_source"] = "CPU derivation from saved native masks; not asserted to be the evaluated score"
    elif config.get("derive_scores") not in {None, "mean_positive_mask_probability"}:
        raise ValueError("Unknown score derivation")
    return proposals_from_arrays(record, arrays, scene)


def proposal_diagnostics(scene: Scene, proposals: Proposals, *, iou_threshold: float = .5,
                         coverage_threshold: float = .2, duplicate_iou: float = .8) -> dict[str, Any]:
    """Display diagnostics on every preserved mask. No AP is calculated here."""
    for value in (iou_threshold, coverage_threshold, duplicate_iou):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Diagnostic thresholds must lie in (0,1]")
    report = {"scope": "illustrative saved-proposal diagnostic; not official AP or benchmark matching",
              "official_metric": False, "iou_threshold": iou_threshold, "coverage_threshold": coverage_threshold,
              "duplicate_iou": duplicate_iou, "proposal_count": len(proposals.masks),
              "recorded_matching": proposals.record.get("recorded_matching"),
              "evaluated_postprocessing": proposals.record.get("evaluated_postprocessing"),
              "missing_artifacts": proposals.record["missing_artifacts"]}
    if scene.labels is None:
        report["skip_reason"] = "Ground truth unavailable; individual proposals still inspectable"
        return report
    ignored = list(scene.metadata.get("ignore_label_ids", []))
    valid = (scene.labels >= 0) & ~np.isin(scene.labels, ignored)
    gt_ids = np.unique(scene.labels[valid & ~np.isin(scene.labels, scene.metadata.get("background_ids", []))])
    truth = np.asarray([scene.labels == label for label in gt_ids], dtype=bool).reshape(len(gt_ids), len(scene.points))
    masks = proposals.masks & valid
    pred_sizes, gt_sizes = masks.sum(axis=1), truth.sum(axis=1)
    intersection = np.zeros((len(masks), len(truth)), dtype=np.int64)
    for target in range(len(truth)):
        intersection[:, target] = masks[:, truth[target]].sum(axis=1)
    union = pred_sizes[:, None] + gt_sizes[None, :] - intersection
    ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    eligible = ious >= iou_threshold
    weights = eligible * (min(ious.shape, default=0) + 1) + np.where(eligible, ious, 0)
    rows, cols = linear_sum_assignment(-weights)
    report["matching_policy"] = "maximum eligible-match cardinality, then summed IoU; fixed threshold"
    matches = {int(row): int(col) for row, col in zip(rows, cols) if ious[row, col] >= iou_threshold}
    gt_coverage = np.divide(intersection, gt_sizes[None, :], out=np.zeros_like(ious), where=gt_sizes[None, :] > 0)
    pred_coverage = np.divide(intersection, pred_sizes[:, None], out=np.zeros_like(ious), where=pred_sizes[:, None] > 0)
    strong = (gt_coverage >= coverage_threshold) & (pred_coverage >= coverage_threshold)
    duplicates = []
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            shared_gt = np.flatnonzero((ious[first] >= iou_threshold) & (ious[second] >= iou_threshold))
            if not len(shared_gt):
                continue
            overlap = np.count_nonzero(masks[first] & masks[second])
            denom = pred_sizes[first] + pred_sizes[second] - overlap
            pair_iou = float(overlap / denom) if denom else 0.
            if pair_iou >= duplicate_iou:
                duplicates.append({"query_ids": [str(proposals.query_ids[first]), str(proposals.query_ids[second])],
                                   "pair_iou": pair_iou, "gt_ids": gt_ids[shared_gt].astype(int).tolist()})
    split_candidates = []
    for target in range(len(truth)):
        relevant = np.flatnonzero(strong[:, target])
        if len(relevant) > 1:
            fragmented = any(np.count_nonzero(masks[a] & masks[b]) / max(1, np.count_nonzero(masks[a] | masks[b])) < duplicate_iou
                             for index, a in enumerate(relevant) for b in relevant[index + 1:])
            if fragmented:
                split_candidates.append({"gt_id": int(gt_ids[target]),
                    "query_ids": [str(proposals.query_ids[q]) for q in relevant]})
    entries = []
    for query, query_id in enumerate(proposals.query_ids):
        best = int(np.argmax(ious[query])) if len(gt_ids) else None
        entries.append({"query_id": str(query_id), "query_index": query, "point_count": int(pred_sizes[query]),
                        "score": None if proposals.scores is None else float(proposals.scores[query]),
                        "best_gt_id": None if best is None else int(gt_ids[best]),
                        "best_iou": None if best is None else float(ious[query, best]),
                        "display_diagnostic_match_gt_id": int(gt_ids[matches[query]]) if query in matches else None,
                        "selected_by_evaluator": bool(proposals.arrays["selected_by_evaluator"][query])
                        if "selected_by_evaluator" in proposals.arrays else None})
    report.update(proposals=entries, missed_gt_ids=[int(value) for index, value in enumerate(gt_ids) if index not in matches.values()],
                  unmatched_query_ids=[str(value) for index, value in enumerate(proposals.query_ids) if index not in matches],
                  split_candidates=split_candidates,
                  merge_candidates=[{"query_id": str(proposals.query_ids[q]), "gt_ids": gt_ids[strong[q]].astype(int).tolist()}
                                    for q in range(len(masks)) if strong[q].sum() > 1], duplicates=duplicates)
    return report


def proposal_panels(scene: Scene, proposals: Proposals, name: str, *, max_proposals: int = 8,
                    query_ids: list[str] | None = None) -> tuple[list[Panel], dict[str, Any]]:
    from .diagnostics import _GRAY, _label_colors
    report = proposal_diagnostics(scene, proposals)
    display, processing = proposals.display_segmentation(policy="highest_score" if proposals.scores is not None else "first_query")
    stable_ids = np.zeros(len(display), dtype=np.int64)
    for index, query_id in enumerate(proposals.query_ids):
        identity = int.from_bytes(hashlib.sha256(str(query_id).encode()).digest()[:7], "little")
        stable_ids[display == index + 1] = identity
    panels = [Panel(key=f"{name}/proposal_display", title=f"{name} · display segmentation",
                    colors=_label_colors(stable_ids, display > 0), values=display,
                    caption=f"Independent saved proposals resolved for display by {processing['policy']}. "
                            "Colors follow original query IDs; this display is not benchmark postprocessing.",
                    metadata={"display_processing": processing, "failure_type": "proposal_display"})]
    report["display_processing"] = processing
    if scene.labels is not None:
        valid_domain = (scene.labels >= 0) & ~np.isin(scene.labels, scene.metadata.get("ignore_label_ids", []))
        valid_gt = valid_domain & ~np.isin(scene.labels, scene.metadata.get("background_ids", []))
        panels.insert(0, Panel(key=f"{name}/proposal_gt", title="Ground truth instances",
                               colors=_label_colors(scene.labels, valid_gt), values=scene.labels,
                               caption="Ground truth colors identify original instances."))
        errors = np.zeros(len(scene.points), dtype=np.int64)
        errors[np.isin(scene.labels, report["missed_gt_ids"])] = 1
        false_indices = [i for i, item in enumerate(proposals.query_ids) if str(item) in report["unmatched_query_ids"]]
        if false_indices:
            errors[np.any(proposals.masks[false_indices], axis=0)] += 2
        colors = np.tile(_GRAY, (len(scene.points), 1))
        colors[errors == 1], colors[errors == 2], colors[errors == 3] = [.34, .56, .9], [.84, .37, 0], [.65, .35, .65]
        colors[~valid_domain] = _GRAY
        errors[~valid_domain] = -1
        panels.append(Panel(key=f"{name}/proposal_errors", title="Missed objects / unmatched proposals",
            colors=colors, values=errors,
            caption=f"Blue: GT without a diagnostic IoU ≥{report['iou_threshold']:g} match; orange: unmatched proposal; "
                    "purple: both. Matching uses preserved masks and is not official AP.",
            metadata={"failure_type": "missed_or_unmatched", "diagnostic": report,
                      "category_legend": {"1": {"label": "Missed GT", "rgb": [.34, .56, .9]},
                                          "2": {"label": "Unmatched proposal", "rgb": [.84, .37, 0]},
                                          "3": {"label": "Both", "rgb": [.65, .35, .65]},
                                          "0": {"label": "Other / ignored", "rgb": _GRAY.tolist()}}}))
    if max_proposals < 0:
        raise ValueError("max_proposals must be nonnegative")
    if query_ids is not None:
        lookup = {str(value): i for i, value in enumerate(proposals.query_ids)}
        unknown = set(map(str, query_ids)) - lookup.keys()
        if unknown:
            raise ValueError(f"Requested unknown proposal query IDs: {sorted(unknown)}")
        selected = [lookup[str(value)] for value in query_ids]
        selection = "explicit persistent query_ids"
    else:
        order = np.argsort(-proposals.scores, kind="stable") if proposals.scores is not None else np.arange(len(proposals.masks))
        selected = order[:max_proposals].tolist()
        selection = "highest exported score, stable export-order ties" if proposals.scores is not None else "original export order"
    report["inspection_selection"] = {"method": selection, "candidate_count": len(proposals.masks),
                                      "selected_query_ids": [str(proposals.query_ids[q]) for q in selected]}
    entries = report.get("proposals", [])
    for query in selected:
        colors = np.tile(_GRAY, (len(scene.points), 1))
        colors[proposals.masks[query]] = [0, .62, .45]
        info = entries[query] if entries else {"query_id": str(proposals.query_ids[query])}
        score_text = "score not exported" if proposals.scores is None else f"score {proposals.scores[query]:.4g}"
        panels.append(Panel(key=f"{name}/proposal_{query}", title=f"Query {proposals.query_ids[query]} · {score_text}",
                            colors=colors, values=proposals.masks[query].astype(np.int8),
                            caption=f"Independent proposal before display overlap resolution; {score_text}. "
                                    f"Score semantics: {proposals.record.get('score_semantics') or 'unavailable'}. "
                                    "Scores are not calibrated uncertainty.",
                            metadata={"proposal": info, "selection": report["inspection_selection"],
                                      "failure_type": "proposal_inspection",
                                      "category_legend": {"1": {"label": "Proposal mask", "rgb": [0, .62, .45]},
                                                          "0": {"label": "Scene context", "rgb": _GRAY.tolist()}}}))
        if scene.labels is not None and info.get("best_gt_id") is not None:
            valid = (scene.labels >= 0) & ~np.isin(scene.labels, scene.metadata.get("ignore_label_ids", []))
            target = scene.labels == info["best_gt_id"]
            positive = proposals.masks[query]
            error = np.zeros(len(scene.points), dtype=np.int8)
            error[valid & target & positive] = 1
            error[valid & ~target & positive] = 2
            error[valid & target & ~positive] = 3
            error[~valid] = -1
            colors = np.tile(_GRAY, (len(scene.points), 1))
            colors[error == 1], colors[error == 2], colors[error == 3] = [0, .62, .45], [.84, .37, 0], [.34, .56, .9]
            panels.append(Panel(key=f"{name}/proposal_{query}_error",
                title=f"Query {proposals.query_ids[query]} versus GT {info['best_gt_id']}", colors=colors, values=error,
                caption="Green: true positive; orange: false positive; blue: false negative; gray: other/ignored. "
                        "Compared with the best-overlap GT for this independent proposal; this is not official matching.",
                metadata={"proposal": info, "failure_type": "proposal_boundary_or_extent",
                          "tp": int((error == 1).sum()), "fp": int((error == 2).sum()), "fn": int((error == 3).sum()),
                          "category_legend": {"1": {"label": "True positive", "rgb": [0, .62, .45]},
                                              "2": {"label": "False positive", "rgb": [.84, .37, 0]},
                                              "3": {"label": "False negative", "rgb": [.34, .56, .9]},
                                              "0": {"label": "Other / ignored", "rgb": _GRAY.tolist()}}}))
    for panel in panels:
        panel.metadata["pairing"] = proposals.record["pairing_status"]
        if proposals.record["pairing_status"]["missing"]:
            panel.caption += " Encoder/decoder/preprocessing record incomplete."
    return panels, report
