"""Model-independent, bounded-memory diagnostics for point-aligned observations.

These are illustrative diagnostics on a selected scene and shared sampled points,
not official segmentation benchmarks. PCA groups share both a fitted basis and
color limits. Cosine maps have fixed scales, including between-model differences.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .contracts import Observation, Panel, Scene


_KINDS = {
    "rgb", "labels", "pca", "similarity", "delta", "boundaries", "norm",
    "prediction", "confidence", "clicks",
}
_GRAY = np.array([0.53, 0.55, 0.58], dtype=np.float32)
_SEQUENTIAL = np.array(
    [[0.10, 0.13, 0.32], [0.11, 0.36, 0.55], [0.10, 0.63, 0.62],
     [0.59, 0.81, 0.39], [0.99, 0.88, 0.23]], dtype=np.float32,
)
_DIVERGING = np.array(
    [[0.17, 0.34, 0.69], [0.52, 0.72, 0.86], [0.94, 0.94, 0.92],
     [0.94, 0.60, 0.37], [0.71, 0.18, 0.16]], dtype=np.float32,
)


def _scalar_colors(values: np.ndarray, limits: tuple[float, float], *, diverging: bool = False
                   ) -> np.ndarray:
    palette = _DIVERGING if diverging else _SEQUENTIAL
    valid = np.isfinite(values)
    width = limits[1] - limits[0]
    unit = np.full(values.shape, 0.5, dtype=np.float64)
    if width > 0:
        unit[valid] = np.clip((values[valid] - limits[0]) / width, 0, 1)
    grid = np.linspace(0, 1, len(palette))
    colors = np.column_stack([np.interp(unit, grid, palette[:, c]) for c in range(3)])
    colors[~valid] = _GRAY
    return colors.astype(np.float32)


def _scalar_panel(key: str, title: str, values: np.ndarray, *, limits: tuple[float, float],
                  caption: str, diverging: bool = False, query_index: int | None = None,
                  metadata: dict[str, Any] | None = None) -> Panel:
    invalid = ~np.isfinite(values)
    extra = dict(metadata or {})
    extra["undefined_count"] = int(invalid.sum())
    extra["color_stops"] = (_DIVERGING if diverging else _SEQUENTIAL).tolist()
    extra.setdefault("value_label", "cosine similarity difference" if "/delta/" in key else
                     "cosine similarity" if "/similarity/" in key else
                     "mean neighbor cosine distance" if key.endswith("/boundaries") else
                     "feature L2 norm" if key.endswith("/norm") else title.rsplit(" · ", 1)[-1])
    if invalid.any():
        extra["undefined_mask"] = invalid
        caption += " Gray points are undefined; scalar storage uses zero placeholders."
    return Panel(
        key=key, title=title, colors=_scalar_colors(values, limits, diverging=diverging),
        values=np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        cmap="coolwarm" if diverging else "viridis", limits=limits, caption=caption,
        query_index=query_index, metadata=extra,
    )


def _valid_labels(labels: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    valid = labels >= 0
    ignored = list(metadata.get("ignore_label_ids", [])) + list(metadata.get("background_ids", []))
    if ignored:
        valid &= ~np.isin(labels, ignored)
    return valid


def _label_colors(labels: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = labels >= 0
    colors = np.tile(_GRAY, (len(labels), 1))
    ids, inverse = np.unique(labels[valid], return_inverse=True)
    # Integer mixing makes colors stable across scenes and subsets, even for sparse IDs.
    palette = [colorsys.hsv_to_rgb(((int(i) * 2654435761) % 2**32) / 2**32, 0.64, 0.93)
               for i in ids]
    if len(ids):
        colors[valid] = np.asarray(palette, dtype=np.float32)[inverse]
    return colors


def _normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(features, axis=1)
    valid = norms > 1e-12
    normalized = np.zeros_like(features, dtype=np.float32)
    np.divide(features, norms[:, None], out=normalized, where=valid[:, None])
    return normalized, norms


def _cosine(normalized: np.ndarray, norms: np.ndarray, query: int) -> np.ndarray:
    if norms[query] <= 1e-12:
        return np.full(len(normalized), np.nan, dtype=np.float32)
    values = np.clip(normalized @ normalized[query], -1, 1)
    values[norms <= 1e-12] = np.nan
    return values


def _average_precision(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Threshold-integrated AP; tied scores enter together, independent of row order."""
    total = int(positive.sum())
    if not len(scores) or not total:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.cumsum(positive[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    true_positive = cumulative[ends]
    recall_increase = np.diff(np.r_[0, true_positive]) / total
    precision = true_positive / (ends + 1)
    return float(np.sum(recall_increase * precision))


def _precision_at(scores: np.ndarray, positive: np.ndarray, k: int) -> float | None:
    """Expected precision under uniform ordering within the cutoff score tie."""
    k = min(k, len(scores))
    if not k:
        return None
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > cutoff, scores == cutoff
    expected = positive[above].sum() + (k - above.sum()) * positive[tied].mean()
    return float(expected / k)


def _token_ids(observation: Observation, layer: str, size: int) -> np.ndarray | None:
    ids = observation.metadata.get("token_ids")
    if isinstance(ids, dict):
        ids = ids.get(layer)
    if ids is None:
        return None
    ids = np.asarray(ids)
    if ids.shape != (size,):
        raise ValueError(f"{observation.name}/{layer}: token_ids must have shape ({size},)")
    return ids


def _query_stats(scene: Scene, observation: Observation, layer: str, query: int,
                 scores: np.ndarray, sample: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_index": query, "scope": "illustrative shared sampled raw-point retrieval",
        "average_precision": None, "precision_10": None, "precision_100": None,
        "eligible_points": 0, "positive_points": 0, "self_exclusion": "query point",
    }
    if scene.labels is None:
        result["skip_reason"] = "Scene has no instance ground truth."
        return result
    labels = np.asarray(scene.labels)
    valid_labels = _valid_labels(labels, scene.metadata)
    if not valid_labels[query]:
        result["skip_reason"] = "Query ground-truth label is ignored or background."
        return result
    keep = valid_labels[sample] & np.isfinite(scores[sample]) & (sample != query)
    tokens = _token_ids(observation, layer, len(labels))
    if tokens is not None and tokens[query] >= 0:
        keep &= tokens[sample] != tokens[query]
        result["self_exclusion"] = "query point and all points sharing its encoder token"
    eligible = sample[keep]
    positive = labels[eligible] == labels[query]
    result.update(
        query_instance_id=int(labels[query]), eligible_points=int(len(eligible)),
        positive_points=int(positive.sum()),
    )
    if not len(eligible) or not positive.any():
        result["skip_reason"] = "No eligible positive points remain in the shared sample."
        return result
    values = scores[eligible]
    result.update(
        average_precision=_average_precision(values, positive),
        precision_10=_precision_at(values, positive, 10),
        precision_100=_precision_at(values, positive, 100),
        precision_10_effective_k=min(10, len(values)),
        precision_100_effective_k=min(100, len(values)),
        positive_fraction=float(positive.mean()),
        ap_ties="equal scores enter at one threshold",
        precision_ties="expected value under uniform ordering within cutoff ties",
    )
    return result


def _shared_pca(entries: list[tuple[Observation, str, np.ndarray]], sample: np.ndarray,
                max_points: int, *, include_pca: bool = True
                ) -> tuple[list[Panel], dict[str, Any], tuple[float, float]]:
    # Bound total fit size across all compared models; choose the same point prefix per model.
    per_model = max(1, max_points // len(entries))
    fit_sample = sample[:per_model]
    pooled_norms = np.concatenate([np.linalg.norm(features[fit_sample], axis=1)
                                   for _, _, features in entries])
    norm_limits = (0.0, max(float(np.quantile(pooled_norms, 0.99)), 1e-12))
    info = {
        "members": [f"{observation.name}/{layer}" for observation, layer, _ in entries],
        "feature_dimension": int(entries[0][2].shape[1]),
        "fit_points_per_model": len(fit_sample),
        "total_fit_points": len(fit_sample) * len(entries),
        "shared_norm_limits": list(norm_limits),
    }
    if not include_pca:
        return [], info, norm_limits
    fit = np.concatenate([features[fit_sample] for _, _, features in entries]).astype(np.float64)
    mean = fit.mean(axis=0)
    centered = fit - mean
    q = min(3, *centered.shape)
    if not np.any(centered):
        components = np.eye(centered.shape[1], q)
        explained = np.zeros(q)
    else:
        # d x d covariance avoids full sampled-point x d SVD work for typical encoder widths.
        covariance = centered.T @ centered
        eigenvalues, vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:q]
        components = vectors[:, order]
        for col in range(q):
            pivot = np.argmax(np.abs(components[:, col]))
            if components[pivot, col] < 0:
                components[:, col] *= -1
        explained = np.maximum(eigenvalues[order], 0) / max(float(np.trace(covariance)), 1e-12)
    fit_projection = centered @ components
    lo, hi = np.quantile(fit_projection, [0.01, 0.99], axis=0)
    width = hi - lo
    info.update({
        "explained_variance_ratio": explained.tolist(),
        "lower_limits": lo.tolist(), "upper_limits": hi.tolist(),
        "scaling": "shared component-wise 1st to 99th percentile; constant components gray",
    })
    panels = []
    for observation, layer, features in entries:
        colors = np.full((len(features), 3), 0.5, dtype=np.float32)
        for start in range(0, len(features), 8192):
            stop = min(len(features), start + 8192)
            projected = (features[start:stop] - mean) @ components
            for col in range(q):
                if width[col] > 1e-12:
                    colors[start:stop, col] = np.clip((projected[:, col] - lo[col]) / width[col], 0, 1)
        panels.append(Panel(
            key=f"{observation.name}/{layer}/pca", title=f"{observation.name} · {layer} · PCA",
            colors=colors,
            caption="Shared PCA basis and 1–99% color limits within matching layer/dimension group. "
                    "Colors are a projection aid, not instance predictions.", metadata=info,
        ))
    return panels, info, norm_limits


def _boundaries(points: np.ndarray, normalized: np.ndarray, norms: np.ndarray,
                tree: cKDTree) -> np.ndarray:
    """Mean cosine distance to eight geometry neighbors, with O(chunk*k*d) work memory."""
    n, dimension = normalized.shape
    values = np.full(n, np.nan, dtype=np.float32)
    k = min(8, n - 1)
    if k < 1:
        return values
    chunk_size = max(16, min(2048, 1_000_000 // max(k * dimension, 1)))
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        _, neighbors = tree.query(points[start:end], k=k + 1, workers=1)
        valid = neighbors != np.arange(start, end)[:, None]
        valid &= np.cumsum(valid, axis=1) <= k
        valid &= norms[neighbors] > 1e-12
        valid &= norms[start:end, None] > 1e-12
        similarities = np.einsum("id,ikd->ik", normalized[start:end], normalized[neighbors])
        distances = np.clip(1 - similarities, 0, 2)
        counts = valid.sum(axis=1)
        np.divide((distances * valid).sum(axis=1), counts, out=values[start:end], where=counts > 0)
    return values


def _align_instances(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray
                     ) -> tuple[np.ndarray, dict[int, int]]:
    """Sparse maximum-IoU one-to-one instance matching, independent of numeric IDs."""
    aligned = np.full(len(prediction), -1, dtype=np.int64)
    pred_ids, pred_inverse, pred_counts = np.unique(prediction[valid & (prediction >= 0)],
                                                   return_inverse=True, return_counts=True)
    gt_ids, gt_counts = np.unique(truth[valid], return_counts=True)
    if not len(pred_ids) or not len(gt_ids):
        return aligned, {}
    gt_inverse = np.searchsorted(gt_ids, truth[valid & (prediction >= 0)])
    pairs, intersections = np.unique(np.column_stack([pred_inverse, gt_inverse]),
                                     axis=0, return_counts=True)
    unions = pred_counts[pairs[:, 0]] + gt_counts[pairs[:, 1]] - intersections
    weights = intersections / unions
    # Private dummy column per prediction lets unmatched segments remain unmatched.
    rows = np.r_[pairs[:, 0], np.arange(len(pred_ids))]
    cols = np.r_[pairs[:, 1], len(gt_ids) + np.arange(len(pred_ids))]
    data = np.r_[weights, np.full(len(pred_ids), np.finfo(np.float64).eps)]
    graph = coo_matrix((data, (rows, cols)), shape=(len(pred_ids), len(gt_ids) + len(pred_ids))).tocsr()
    match_rows, match_cols = min_weight_full_bipartite_matching(graph, maximize=True)
    mapping = {int(pred_ids[row]): int(gt_ids[col])
               for row, col in zip(match_rows, match_cols) if col < len(gt_ids)}
    if mapping:
        chosen = prediction >= 0
        lookup = np.full(len(pred_ids), -1, dtype=np.int64)
        for row, col in zip(match_rows, match_cols):
            if col < len(gt_ids):
                lookup[row] = gt_ids[col]
        positions = np.searchsorted(pred_ids, prediction[chosen])
        in_range = positions < len(pred_ids)
        found = np.zeros_like(in_range)
        found[in_range] = pred_ids[positions[in_range]] == prediction[chosen][in_range]
        output = np.flatnonzero(chosen)[found]
        aligned[output] = lookup[positions[found]]
    return aligned, mapping


def _prediction_panels(scene: Scene, observation: Observation, prediction: np.ndarray,
                       *, key: str, title: str, include_error: bool = True
                       ) -> tuple[list[Panel], dict[str, Any]]:
    kind = observation.metadata.get("prediction_kind", "instance")
    truth = scene.semantic_labels if kind == "semantic" else scene.labels
    prediction_background = observation.metadata.get("prediction_background_ids", [-1])
    available = (prediction >= 0) & ~np.isin(prediction, prediction_background)
    info: dict[str, Any] = {
        "prediction_kind": kind, "scope": "illustrative selected-scene diagnostic",
        "prediction_background_ids": [int(value) for value in prediction_background],
        "negative_prediction_ids": "always treated as no prediction",
    }
    if truth is None:
        info["skip_reason"] = "No matching ground-truth labels; showing unaligned prediction IDs."
        return [Panel(key=key, title=title, colors=_label_colors(prediction, available),
                      caption=info["skip_reason"], metadata=info)], info
    truth = np.asarray(truth)
    valid = _valid_labels(truth, scene.metadata)
    if kind == "instance":
        foreground_prediction = np.where(available, prediction, -1)
        aligned, mapping = _align_instances(foreground_prediction, truth, valid)
        colors = _label_colors(aligned, valid & (aligned >= 0))
        correct = (aligned == truth) & available
        info["prediction_to_gt_id"] = {str(k): v for k, v in mapping.items()}
        caption = "Prediction colors aligned to GT by one-to-one maximum-IoU matching. "
        caption += "Gray denotes ignored, missed, or unmatched segments."
    elif kind in ("semantic", "binary"):
        if kind == "binary":
            target = observation.metadata.get("target_instance_id")
            if target is None:
                info["skip_reason"] = "Binary prediction has no target_instance_id metadata."
                return [Panel(key=key, title=title, colors=_label_colors(prediction, available),
                              caption=info["skip_reason"], metadata=info)], info
            truth = (truth == target).astype(np.int64)
        colors = _label_colors(prediction, valid & available)
        correct = (prediction == truth) & available
        caption = f"{kind.capitalize()} labels compared by explicit label identity. Gray is ignored."
    else:
        raise ValueError(f"Unknown prediction_kind {kind!r} for {observation.name}")
    correct &= valid
    info.update(
        evaluated_points=int(valid.sum()),
        point_agreement=float(correct.sum() / valid.sum()) if valid.any() else None,
        missed_points=int((valid & ~available).sum()),
    )
    panels = [Panel(key=key, title=title, colors=colors, caption=caption, metadata=info)]
    if include_error:
        error_colors = np.tile(_GRAY, (len(truth), 1))
        error_colors[valid & ~correct] = [0.87, 0.23, 0.23]
        error_colors[valid & ~available] = [0.97, 0.66, 0.16]
        error_colors[correct] = [0.16, 0.69, 0.48]
        panels.append(Panel(
            key=f"{key}_error", title=f"{title} · disagreement", colors=error_colors,
            caption="Green: matched correct; red: disagreement/unmatched; amber: no prediction; "
                    "gray: ignored GT. Instance IDs are matched before comparison.", metadata=info,
        ))
    return panels, info


def build_panels(scene: Scene, observations: list[Observation], *, query_indices: list[int],
                 seed: int = 0, max_analysis_points: int = 12000,
                 reference_name: str | None = None, requested: list[str] | None = None
                 ) -> tuple[list[Panel], dict[str, Any]]:
    """Build reusable display panels and JSON-serializable illustrative diagnostics.

    ``token_ids`` metadata may be a point array or a layer-to-array mapping. Negative
    truth IDs, plus explicit ``ignore_label_ids`` and ``background_ids`` in scene
    metadata, are excluded from retrieval and prediction diagnostics. Confidence
    is named as such only when metadata explicitly declares calibration and semantics.
    Observation ``prediction_background_ids`` denotes no prediction (for example,
    AGILE3D instance ID zero); it is separate from ground-truth background IDs.
    """
    wanted = set(_KINDS if requested is None else requested)
    unknown = wanted - _KINDS
    if unknown:
        raise ValueError(f"Unknown visualization kinds: {', '.join(sorted(unknown))}")
    if max_analysis_points < 1:
        raise ValueError("max_analysis_points must be positive")
    points = np.asarray(scene.points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Scene points must be finite Nx3 coordinates")
    n = len(points)
    queries = list(dict.fromkeys(int(q) for q in query_indices))
    if any(q < 0 or q >= n for q in queries):
        raise ValueError("Query indices must refer to scene points")
    if len({obs.name for obs in observations}) != len(observations):
        raise ValueError("Observation names must be unique")
    if reference_name is not None and reference_name not in {obs.name for obs in observations}:
        raise ValueError(f"Unknown reference observation {reference_name!r}")
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n, max_analysis_points), replace=False) if n else np.array([], dtype=int)
    report: dict[str, Any] = {
        "scene_id": scene.scene_id, "scope": "illustrative diagnostics, not benchmark scores",
        "seed": int(seed), "point_count": n, "analysis_point_count": len(sample),
        "analysis_indices": sample.tolist(), "query_indices": queries,
        "reference_name": reference_name, "requested": sorted(wanted),
        "pca_groups": {}, "queries": {}, "predictions": {}, "skipped": [],
        "sampling": "One fixed raw-point sample shared by every observation; no token averaging.",
    }
    panels: list[Panel] = []

    def skip(kind: str, reason: str, observation: str | None = None) -> None:
        report["skipped"].append({"kind": kind, "observation": observation, "reason": reason})

    if not n:
        skip("all", "Scene contains no points.")
        return panels, report
    if "rgb" in wanted:
        if scene.colors is None:
            skip("rgb", "Scene has no RGB colors.")
        else:
            colors = np.asarray(scene.colors, dtype=np.float32)
            if colors.max(initial=0) > 1:
                colors = colors / 255
            panels.append(Panel(key="rgb", title="RGB", colors=np.clip(colors, 0, 1),
                                caption="Input scene appearance."))
    if "labels" in wanted:
        if scene.labels is None:
            skip("labels", "Scene has no instance ground truth.")
        else:
            labels = np.asarray(scene.labels)
            label_noun = "parts" if scene.metadata.get("label_kind") == "part" else "instances"
            panels.append(Panel(key="labels", title=f"GT {label_noun}",
                                colors=_label_colors(labels, _valid_labels(labels, scene.metadata)),
                                caption=f"Stable colors per {label_noun[:-1]} ID. Ignored/background IDs are gray."))
        if scene.semantic_labels is not None:
            labels = np.asarray(scene.semantic_labels)
            panels.append(Panel(key="semantic_labels", title="GT semantic labels",
                                colors=_label_colors(labels, _valid_labels(labels, scene.metadata)),
                                caption="Stable colors per semantic ID. Ignored/background IDs are gray."))
    groups: dict[tuple[str, int], list[tuple[Observation, str, np.ndarray]]] = defaultdict(list)
    features_by_key: dict[tuple[str, str], np.ndarray] = {}
    for obs in observations:
        if not obs.features:
            for kind in sorted(wanted & {"pca", "similarity", "delta", "boundaries", "norm"}):
                skip(kind, "Observation provides no encoder features.", obs.name)
        for layer, raw_features in obs.features.items():
            features = np.asarray(raw_features, dtype=np.float32)
            if features.ndim != 2 or features.shape[0] != n or features.shape[1] < 1:
                raise ValueError(f"{obs.name}/{layer}: features must be NxD with D >= 1")
            if not np.isfinite(features).all():
                raise ValueError(f"{obs.name}/{layer}: features contain nonfinite values")
            features_by_key[obs.name, layer] = features
            groups[layer, features.shape[1]].append((obs, layer, features))
    norm_limits_by_key: dict[tuple[str, str], tuple[float, float]] = {}
    if wanted & {"pca", "norm"}:
        for (layer, dimension), entries in groups.items():
            pca_panels, info, norm_limits = _shared_pca(
                entries, sample, max_analysis_points, include_pca="pca" in wanted,
            )
            report["pca_groups"][f"{layer}:d{dimension}"] = info
            if "pca" in wanted:
                panels.extend(pca_panels)
            for obs, name, _ in entries:
                norm_limits_by_key[obs.name, name] = norm_limits
    tree = cKDTree(points) if "boundaries" in wanted and n > 1 and groups else None
    ref_features = {layer: features for (name, layer), features in features_by_key.items()
                    if name == reference_name}
    ref_maps: dict[tuple[str, int], np.ndarray] = {}
    if "delta" in wanted and reference_name is not None:
        for layer, features in ref_features.items():
            normalized, norms = _normalize(features)
            for query in queries:
                ref_maps[layer, query] = _cosine(normalized, norms, query)
    elif "delta" in wanted:
        skip("delta", "No reference observation was selected.")
    if not queries:
        for kind in sorted(wanted & {"similarity", "delta"}):
            skip(kind, "No query points were selected.")
    for obs in observations:
        for layer in obs.features:
            features = features_by_key[obs.name, layer]
            if not wanted & {"norm", "boundaries", "similarity", "delta"}:
                continue
            normalized, norms = _normalize(features)
            stem = f"{obs.name}/{layer}"
            if "norm" in wanted:
                panels.append(_scalar_panel(
                    f"{stem}/norm", f"{obs.name} · {layer} · feature norm", norms,
                    limits=norm_limits_by_key[obs.name, layer],
                    caption="Feature L2 magnitude; shared 99% upper limit within PCA comparison group. "
                            "Magnitude is not a calibrated confidence score.",
                ))
            if "boundaries" in wanted:
                if tree is None:
                    skip("boundaries", "At least two scene points are required.", obs.name)
                else:
                    values = _boundaries(points, normalized, norms, tree)
                    panels.append(_scalar_panel(
                        f"{stem}/boundaries", f"{obs.name} · {layer} · feature boundaries", values,
                        limits=(0.0, 2.0),
                        caption="Mean cosine distance to eight nearest 3D neighbors (or all available). "
                                "Geometry, sampling density, and token boundaries can affect this diagnostic.",
                        metadata={"geometry_neighbors": min(8, n - 1)},
                    ))
            if wanted & {"similarity", "delta"}:
                for query in queries:
                    values = _cosine(normalized, norms, query)
                    if "similarity" in wanted:
                        stats = _query_stats(scene, obs, layer, query, values, sample)
                        report["queries"][f"{stem}/q{query}"] = stats
                        panels.append(_scalar_panel(
                            f"{stem}/similarity/q{query}",
                            f"{obs.name} · {layer} · query {query}", values, limits=(-1.0, 1.0),
                            caption="Cosine similarity to the same selected raw point; fixed [-1, 1] scale. "
                                    "Retrieval excludes the query and its token when available.",
                            diverging=True, query_index=query, metadata={"retrieval": stats},
                        ))
                    if "delta" in wanted and reference_name is not None and obs.name != reference_name:
                        if (layer, query) not in ref_maps:
                            skip("delta", f"Reference has no matching layer {layer!r}.", obs.name)
                        else:
                            panels.append(_scalar_panel(
                                f"{stem}/delta/q{query}",
                                f"{obs.name} − {reference_name} · {layer} · query {query}",
                                values - ref_maps[layer, query], limits=(-2.0, 2.0),
                                caption="Difference between scalar cosine-similarity maps at the same query. "
                                        "Red increases similarity; blue decreases it. This is not a feature-vector subtraction.",
                                diverging=True, query_index=query,
                                metadata={"reference_name": reference_name,
                                          "reference_dimension": ref_features[layer].shape[1],
                                          "feature_dimension": features.shape[1]},
                            ))
        if "prediction" in wanted:
            if obs.prediction is None:
                skip("prediction", "Observation provides no prediction IDs.", obs.name)
            else:
                prediction_panels, info = _prediction_panels(
                    scene, obs, np.asarray(obs.prediction), key=f"{obs.name}/prediction",
                    title=f"{obs.name} · prediction",
                )
                panels.extend(prediction_panels)
                report["predictions"][obs.name] = info
        if "confidence" in wanted:
            if obs.confidence is None:
                skip("confidence", "Observation provides no confidence or score array.", obs.name)
            else:
                values = np.asarray(obs.confidence, dtype=np.float32)
                semantics = obs.metadata.get("confidence_semantics")
                calibrated = obs.metadata.get("confidence_calibrated") is True and bool(semantics)
                if calibrated and (values.min() < 0 or values.max() > 1):
                    raise ValueError(f"{obs.name}: declared calibrated confidence must lie in [0, 1]")
                limits = (0.0, 1.0) if calibrated else tuple(float(v) for v in np.quantile(values, [0.01, 0.99]))
                panels.append(_scalar_panel(
                    f"{obs.name}/confidence", f"{obs.name} · {'confidence' if calibrated else 'score'}",
                    values, limits=limits,
                    caption=f"Declared calibrated confidence: {semantics}." if calibrated else
                            "Model-supplied score; calibration is not established. Robust per-model color limits.",
                    metadata={"confidence_calibrated": calibrated, "confidence_semantics": semantics},
                ))
        if "clicks" in wanted:
            if obs.steps is None:
                skip("clicks", "Observation provides no stepwise prediction history.", obs.name)
            else:
                for step, prediction in enumerate(np.asarray(obs.steps)):
                    # Legacy arrays have no target map. Keep native IDs across all
                    # frames; independently matching each frame can hide switches.
                    info = {"temporal_identity": "persistent native prediction IDs",
                            "pairing_status": "incomplete legacy sequence",
                            "metric_domain": "unknown; no target IoU computed"}
                    step_panels = [Panel(
                        key=f"{obs.name}/step_{step}",
                        title=f"{obs.name} · interaction step {step}",
                        colors=_label_colors(prediction, (prediction >= 0) & ~np.isin(
                            prediction, obs.metadata.get("prediction_background_ids", [-1]))),
                        caption="Persistent native IDs; no per-frame GT matching. Legacy sequence: "
                                "target mapping, evaluation domain and checkpoint pairing may be incomplete.",
                        metadata=info)]
                    if obs.metadata.get("prediction_kind") in {"binary", "semantic"}:
                        # These domains already have explicit fixed label identities.
                        step_panels, info = _prediction_panels(scene, obs, prediction,
                            key=f"{obs.name}/step_{step}", title=f"{obs.name} · interaction step {step}",
                            include_error=False)
                    clicks = [click for i, click in enumerate(obs.clicks)
                              if int(click.get("step", i)) <= step]
                    for panel in step_panels:
                        panel.metadata = {**panel.metadata, "clicks": clicks, "step": step}
                    panels.extend(step_panels)
                    report["predictions"][f"{obs.name}/step_{step}"] = info
    return panels, report
