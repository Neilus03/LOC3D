"""CPU research diagnostics with immutable PCA and explicit point identities.

This module operates only on saved arrays. It never loads a model or runs inference.
PCA coordinates are a descriptive projection, not semantic correspondence. By default
one PCA state belongs to one source/layer and is fitted independently of other models.
Persisting its actual numerical transform is essential: eigenvectors inside a repeated
or nearly repeated eigenspace need not be stable across numerical libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


PCA_SCHEMA = "delimit3d.frozen-feature-pca.v1"
PCA_CAVEAT = (
    "PCA colors are a projection aid, not semantic correspondence or segmentation. "
    "Equal feature dimensions do not establish aligned coordinates."
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    sha = hashlib.sha256()
    sha.update(str(value.dtype).encode())
    sha.update(str(value.shape).encode())
    sha.update(value.tobytes())
    return sha.hexdigest()


def _matrix(features: Any) -> np.ndarray:
    result = np.asarray(features)
    if (result.ndim != 2 or min(result.shape) < 1 or
            not np.issubdtype(result.dtype, np.number) or
            np.issubdtype(result.dtype, np.complexfloating) or not np.isfinite(result).all()):
        raise ValueError("features must be a nonempty finite real N by D matrix")
    return result


def _ids(value: Any, n: int, name: str, *, unique: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (n,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain {n} integer identities")
    if unique and (np.any(result < 0) or len(np.unique(result)) != n):
        raise ValueError(f"{name} must be unique nonnegative original point identities")
    return result.astype(np.int64, copy=False)


def _source(record: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(_json(dict(record)))
    for key in ("source_id", "feature_level"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"source_record needs a stable nonempty {key}")
    return value


def validate_requested_identity(requested: Mapping[str, Any], recorded: Mapping[str, Any],
                                *, prefix: str = "identity") -> None:
    """Require every explicitly requested field, including exact epoch and query.

    Use actual artifact metadata for ``recorded``. A display name or a checkpoint
    filename does not establish its epoch. Missing evidence is an error only for
    fields the caller explicitly requires; it must not trigger a substitute.
    """
    for key, expected in requested.items():
        if key not in recorded:
            raise ValueError(f"{prefix}.{key}: requested identity is missing from artifact record")
        actual = recorded[key]
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                raise ValueError(f"{prefix}.{key}: requested identity is not a mapping")
            validate_requested_identity(expected, actual, prefix=f"{prefix}.{key}")
        elif _json(expected) != _json(actual):
            raise ValueError(f"{prefix}.{key}: requested {expected!r}, recorded {actual!r}; no substitution")


@dataclass(frozen=True)
class FrozenPCA:
    """Serializable fit including sample identities, centering, basis, and scaling.

    ``save`` refuses to overwrite a previous file. ``rgb`` uses only these stored
    quantities and never rescales to the points or models being displayed.
    """
    mean: np.ndarray
    basis: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fit_original_point_ids: np.ndarray
    explained_variance_ratio: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for key in ("mean", "basis", "lower", "upper", "fit_original_point_ids",
                    "explained_variance_ratio"):
            array = np.array(getattr(self, key), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, key, array)
        object.__setattr__(self, "metadata", json.loads(_json(self.metadata)))
        d, q = self.basis.shape
        if (self.mean.shape != (d,) or self.lower.shape != (q,) or
                self.upper.shape != (q,) or self.explained_variance_ratio.shape != (q,) or
                q < 1 or q > 3 or np.any(self.upper < self.lower)):
            raise ValueError("Invalid frozen PCA state dimensions/scales")
        for key in ("mean", "basis", "lower", "upper", "explained_variance_ratio"):
            if not np.isfinite(getattr(self, key)).all():
                raise ValueError("Frozen PCA state contains nonfinite values")
        _ids(self.fit_original_point_ids, len(self.fit_original_point_ids),
             "fit_original_point_ids", unique=True)
        if self.metadata.get("schema") != PCA_SCHEMA:
            raise ValueError("Unsupported frozen PCA schema")
        _source(self.metadata["source_record"])

    def record(self) -> dict[str, Any]:
        value = json.loads(_json(self.metadata))
        value["arrays"] = {
            key: {"sha256": _digest(getattr(self, key)), "shape": list(getattr(self, key).shape)}
            for key in ("mean", "basis", "lower", "upper", "fit_original_point_ids",
                        "explained_variance_ratio")
        }
        value["state_sha256"] = hashlib.sha256(_json(value).encode()).hexdigest()
        return value

    def save(self, path: str | Path) -> None:
        """Write one safe NPZ containing numerical state and JSON (no pickle)."""
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, mean=self.mean, basis=self.basis,
                                lower=self.lower, upper=self.upper,
                                fit_original_point_ids=self.fit_original_point_ids,
                                explained_variance_ratio=self.explained_variance_ratio,
                                metadata=np.asarray(_json(self.metadata)),
                                state_sha256=np.asarray(self.record()["state_sha256"]))

    @classmethod
    def load(cls, path: str | Path) -> "FrozenPCA":
        with np.load(path, allow_pickle=False) as data:
            state = cls(*(data[key] for key in (
                "mean", "basis", "lower", "upper", "fit_original_point_ids",
                "explained_variance_ratio")), metadata=json.loads(str(data["metadata"].item())))
            if state.record()["state_sha256"] != str(data["state_sha256"].item()):
                raise ValueError("Frozen PCA state hash mismatch")
        return state

    def compatibility(self, source_record: Mapping[str, Any]) -> dict[str, str]:
        source = _source(source_record)
        fitted = self.metadata["source_record"]
        for field in ("feature_level", "preprocessing", "feature_stages"):
            if _json(source.get(field)) != _json(fitted.get(field)):
                raise ValueError(f"Frozen PCA {field} differs from fitted source")
        if source["source_id"] == fitted["source_id"]:
            for field in ("checkpoint_sha256", "encoder_checkpoint_sha256", "epoch"):
                if _json(source.get(field)) != _json(fitted.get(field)):
                    raise ValueError(f"Frozen PCA {field} differs for the same claimed source")
            return {"mode": "same_source", "interpretation": PCA_CAVEAT}
        coordinate_system = source.get("feature_coordinate_system")
        if (not coordinate_system or coordinate_system != fitted.get("feature_coordinate_system")
                or not source.get("coordinate_compatibility_reason")
                or not fitted.get("coordinate_compatibility_reason")):
            raise ValueError("Different PCA source: equal dimensions are insufficient; use its own "
                             "frozen state or explicitly justify a shared feature_coordinate_system")
        return {"mode": "explicit_common_coordinate_system",
                "justification": str(source["coordinate_compatibility_reason"]),
                "interpretation": PCA_CAVEAT}

    def rgb(self, features: np.ndarray, *, source_record: Mapping[str, Any],
            chunk_size: int = 8192) -> np.ndarray:
        value = _matrix(features)
        self.compatibility(source_record)
        if value.shape[1] != len(self.mean):
            raise ValueError("Feature dimension differs from frozen PCA state")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        colors = np.full((len(value), 3), 0.5, dtype=np.float32)
        width = self.upper - self.lower
        active = width > 1e-12
        q = len(width)
        for start in range(0, len(value), chunk_size):
            stop = min(start + chunk_size, len(value))
            projection = (value[start:stop].astype(np.float64) - self.mean) @ self.basis
            scaled = np.full_like(projection, 0.5)
            np.divide(projection - self.lower, width, out=scaled, where=active)
            colors[start:stop, :q] = np.clip(scaled, 0, 1)
        return colors


def fit_frozen_pca(features: np.ndarray, *, original_point_ids: np.ndarray,
                   source_record: Mapping[str, Any], max_fit_points: int = 12000,
                   seed: int = 0, quantiles: tuple[float, float] = (0.01, 0.99)) -> FrozenPCA:
    """Fit one immutable source; adding other checkpoints cannot affect this fit.

    Sampling is uniform without replacement over sorted original identities.
    Both fit identities and selected-feature hash are persisted. Colors use fixed
    fit-sample quantiles; displayed crops and later checkpoints never refit them.
    """
    value = _matrix(features)
    ids = _ids(original_point_ids, len(value), "original_point_ids", unique=True)
    source = _source(source_record)
    if (isinstance(max_fit_points, bool) or not isinstance(max_fit_points, int)
            or max_fit_points < 1):
        raise ValueError("max_fit_points must be a positive integer")
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    ordered = np.argsort(ids, kind="stable")
    if len(ordered) > max_fit_points:
        positions = np.sort(np.random.default_rng(seed).choice(
            len(ordered), max_fit_points, replace=False))
        ordered = ordered[positions]
    fit = value[ordered].astype(np.float64)
    mean = fit.mean(axis=0)
    centered = fit - mean
    q = min(3, *fit.shape)
    covariance = centered.T @ centered
    eigenvalues, vectors = np.linalg.eigh(covariance)
    ordering = np.argsort(eigenvalues, kind="stable")[::-1]
    basis = vectors[:, ordering[:q]]
    for column in range(q):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    explained = np.maximum(eigenvalues[ordering[:q]], 0) / max(float(np.trace(covariance)), 1e-12)
    lo, hi = np.quantile(centered @ basis, quantiles, axis=0)
    return FrozenPCA(mean, basis, lo, hi, ids[ordered], explained, {
        "schema": PCA_SCHEMA, "source_record": source, "seed": int(seed),
        "sample_method": "uniform without replacement over sorted original point identities",
        "population_point_count": len(value), "fit_point_count": len(fit),
        "fit_features_sha256": _digest(value[ordered]), "centering": "fit sample arithmetic mean",
        "scaling": "fixed per-component fit sample quantiles; constant components gray",
        "quantiles": list(quantiles), "feature_dimension": value.shape[1],
        "normalization": "none before PCA", "numpy_version": np.__version__,
        "norm_upper_limit": max(float(np.quantile(np.linalg.norm(fit, axis=1), quantiles[1])), 1e-12),
        "eigensolver": "numpy.linalg.eigh on float64 centered scatter matrix",
        "eigenvector_sign": "largest absolute coefficient positive; first coefficient on ties",
        "repeated_eigenvalue_limit": "refitting may rotate degenerate eigenspaces across libraries; reuse saved basis",
        "coordinate_policy": "independent frozen source by default; explicit justification required to share",
        "interpretation": PCA_CAVEAT,
    })


def query_cosine(features: np.ndarray, query_index: int) -> np.ndarray:
    """Within-model scalar cosine; zero vectors are undefined (NaN), not zero."""
    value = _matrix(features)
    if (isinstance(query_index, (bool, np.bool_)) or
            not isinstance(query_index, (int, np.integer)) or not 0 <= query_index < len(value)):
        raise ValueError("query_index must be an in-range integer original-array index")
    norm = np.linalg.norm(value.astype(np.float64), axis=1)
    if norm[query_index] <= 1e-12:
        return np.full(len(value), np.nan, dtype=np.float32)
    result = np.full(len(value), np.nan, dtype=np.float64)
    numerator = np.asarray(value, dtype=np.float64) @ value[query_index].astype(np.float64)
    np.divide(numerator, norm * norm[query_index], out=result, where=norm > 1e-12)
    return np.clip(result, -1, 1).astype(np.float32)


def spherical_kmeans(features: np.ndarray, n_clusters: int, *, seed: int = 0,
                     max_iter: int = 100, tol: float = 1e-6
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster rows with query-free spherical k-means.

    The input rows are L2-normalized internally, so the assignment maximizes
    cosine similarity to a unit cluster center.  The returned labels are in
    the original row order; rows with an undefined (near-zero) feature receive
    ``-1`` and do not participate in fitting.  Initialization and empty-cluster
    recovery are deterministic for a fixed ``seed``.

    This is intentionally a small NumPy implementation rather than a dependency
    on scikit-learn: feature research runs in the saved-artifact environment and
    must record the exact cosine-space procedure used for a comparison.
    """
    value = _matrix(features).astype(np.float64, copy=False)
    if (isinstance(n_clusters, (bool, np.bool_)) or not isinstance(n_clusters, (int, np.integer))
            or n_clusters < 1):
        raise ValueError("n_clusters must be a positive integer")
    if (isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1):
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")

    norms = np.linalg.norm(value, axis=1)
    valid = norms > 1e-12
    if int(valid.sum()) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of nonzero feature rows")
    points = value[valid] / norms[valid, None]
    count, dimension = points.shape
    rng = np.random.default_rng(seed)

    # Deterministic k-means++ in cosine distance.  The first center is sampled
    # from a seeded generator; subsequent centers favor points far from the
    # already selected centers.
    centers = np.empty((n_clusters, dimension), dtype=np.float64)
    selected = np.empty(n_clusters, dtype=np.int64)
    selected[0] = int(rng.integers(count))
    centers[0] = points[selected[0]]
    min_distance = np.maximum(0.0, 1.0 - points @ centers[0])
    for index in range(1, n_clusters):
        weights = min_distance * min_distance
        total = float(weights.sum())
        if total <= 1e-15:
            remaining = np.setdiff1d(np.arange(count), selected[:index], assume_unique=False)
            selected[index] = int(rng.choice(remaining))
        else:
            selected[index] = int(rng.choice(count, p=weights / total))
        centers[index] = points[selected[index]]
        min_distance = np.minimum(min_distance, np.maximum(0.0, 1.0 - points @ centers[index]))

    labels = np.full(count, -1, dtype=np.int64)
    reinitializations = 0
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        similarities = points @ centers.T
        new_labels = similarities.argmax(axis=1).astype(np.int64)
        updated = np.empty_like(centers)
        current_best = similarities[np.arange(count), new_labels]
        for cluster in range(n_clusters):
            members = points[new_labels == cluster]
            if len(members):
                candidate = members.sum(axis=0)
                length = float(np.linalg.norm(candidate))
                if length > 1e-12:
                    updated[cluster] = candidate / length
                    continue
            # Keep K clusters even when a degenerate update empties one.  A
            # farthest point is preferable to an arbitrary center because it
            # addresses the current worst represented region.
            order = np.argsort(current_best, kind="stable")
            replacement = next((int(row) for row in order
                                if not np.any(np.all(np.isclose(updated[:cluster], points[row]), axis=1))),
                               int(rng.integers(count)))
            updated[cluster] = points[replacement]
            reinitializations += 1
        center_distance = np.max(np.maximum(0.0, 1.0 - np.sum(centers * updated, axis=1)))
        iterations = iteration
        centers = updated
        if np.array_equal(new_labels, labels) or center_distance <= tol:
            labels = new_labels
            break
        labels = new_labels

    # Reassign once against the final centers so labels and centers describe
    # the same partition when convergence was triggered by a small center move.
    labels = np.argmax(points @ centers.T, axis=1).astype(np.int64)
    final_similarity = np.max(points @ centers.T, axis=1)
    full_labels = np.full(len(value), -1, dtype=np.int64)
    full_labels[valid] = labels
    record = {
        "algorithm": "spherical_kmeans",
        "distance": "1 - cosine similarity between L2-normalized feature and unit center",
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "feature_rows": int(len(value)),
        "valid_rows": int(valid.sum()),
        "zero_feature_rows": int((~valid).sum()),
        "dimension": int(dimension),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "iterations": int(iterations),
        "reinitializations": int(reinitializations),
        "mean_cosine_to_center": float(final_similarity.mean()),
        "mean_cosine_distance": float((1.0 - final_similarity).mean()),
        "fit_uses_queries": False,
    }
    return full_labels, centers.astype(np.float32), record


def matched_cosine_delta(features_a: np.ndarray, features_b: np.ndarray, *,
                         points_a: np.ndarray, points_b: np.ndarray,
                         original_point_ids_a: np.ndarray, original_point_ids_b: np.ndarray,
                         query_original_point_id: int, scene_id_a: str, scene_id_b: str,
                         geometry_atol: float = 0) -> np.ndarray:
    """Return B minus A scalar cosine on the exact same scene and query identity.

    Different feature dimensions are supported; feature coordinates are not
    subtracted. Array order and geometry must match; no nearest-neighbor repair.
    """
    a, b = _matrix(features_a), _matrix(features_b)
    if len(a) != len(b) or scene_id_a != scene_id_b:
        raise ValueError("Matched cosine requires the same scene and point population")
    ids_a = _ids(original_point_ids_a, len(a), "original_point_ids_a", unique=True)
    ids_b = _ids(original_point_ids_b, len(b), "original_point_ids_b", unique=True)
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Matched cosine original point identities/order differ")
    pa, pb = np.asarray(points_a), np.asarray(points_b)
    if (pa.shape != (len(a), 3) or pb.shape != pa.shape or not np.isfinite(pa).all()
            or not np.isfinite(pb).all() or not np.isfinite(geometry_atol) or geometry_atol < 0
            or not np.allclose(pa, pb, atol=geometry_atol, rtol=0)):
        raise ValueError("Matched cosine requires explicitly corresponding finite geometry")
    if isinstance(query_original_point_id, (bool, np.bool_)) or not isinstance(
            query_original_point_id, (int, np.integer)):
        raise ValueError("Query original point identity must be an integer")
    position = np.flatnonzero(ids_a == query_original_point_id)
    if len(position) != 1:
        raise ValueError("Requested query original point identity is absent")
    query = int(position[0])
    return query_cosine(b, query) - query_cosine(a, query)


def boundary_regions(points: np.ndarray, labels: np.ndarray, *, valid_mask: np.ndarray | None = None,
                     k: int = 8, radius: float = 0.1) -> np.ndarray:
    """GT transition among up to k nearest other points inside a declared radius.

    This is a geometry-dependent diagnostic region, not an official boundary
    metric. Duplicate coordinates are allowed; self identities are excluded.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite N by 3 geometry")
    labels = _ids(labels, len(points), "labels")
    if (not isinstance(k, int) or isinstance(k, bool) or k < 1 or
            not np.isfinite(radius) or radius <= 0):
        raise ValueError("Boundary k and radius must be positive")
    valid = labels >= 0 if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (len(points),):
        raise ValueError("valid_mask must match point order")
    result = np.zeros(len(points), dtype=bool)
    if len(points) < 2:
        return result
    tree = cKDTree(points)
    for start in range(0, len(points), 8192):
        end = min(start + 8192, len(points))
        distances, neighbors = tree.query(points[start:end], k=min(k + 1, len(points)), workers=1)
        keep = ((neighbors != np.arange(start, end)[:, None]) & (distances <= radius)
                & valid[start:end, None] & valid[neighbors])
        keep &= np.cumsum(keep, axis=1) <= k
        result[start:end] = np.any(keep & (labels[start:end, None] != labels[neighbors]), axis=1)
    return result


def _expected_topk_fraction(scores: np.ndarray, region: np.ndarray, k: int) -> float | None:
    if k < 1 or not len(scores):
        return None
    k = min(k, len(scores))
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > cutoff, scores == cutoff
    return float((region[above].sum() + (k - above.sum()) * region[tied].mean()) / k)


def query_diagnostics(scores: np.ndarray, *, points: np.ndarray, labels: np.ndarray,
                      original_point_ids: np.ndarray, query_original_point_id: int,
                      semantic_labels: np.ndarray | None = None,
                      parent_labels: np.ndarray | None = None, token_ids: np.ndarray | None = None,
                      analysis_point_ids: np.ndarray | None = None,
                      common_eligible_mask: np.ndarray | None = None,
                      ignore_label_ids: tuple[int, ...] = (-1,), background_ids: tuple[int, ...] = (),
                      nearby_radius: float = 0.5, boundary_radius: float = 0.1,
                      boundary_k: int = 8, label_kind: str = "instance"
                      ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Self-excluded raw-point leakage and object/part region diagnostics.

    ``parent_labels`` denotes parent object identity for a part study. Region
    masks overlap intentionally (e.g. same-category and nearby). Retrieval depth
    is the eligible target size, explicitly GT-assisted. A caller comparing
    tokenizations should supply the intersection in ``common_eligible_mask``.
    """
    xyz, values = np.asarray(points), np.asarray(scores)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be finite N by 3 geometry")
    n = len(xyz)
    if values.shape != (n,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError("scores must match raw-point order")
    if np.any(np.isinf(values)) or np.any(np.abs(values[np.isfinite(values)]) > 1 + 1e-6):
        raise ValueError("scores must be cosine similarities in [-1, 1] or undefined NaN")
    if not np.isfinite(nearby_radius) or nearby_radius <= 0:
        raise ValueError("nearby_radius must be finite and positive in scene coordinate units")
    ids = _ids(original_point_ids, n, "original_point_ids", unique=True)
    labels = _ids(labels, n, "labels")
    if isinstance(query_original_point_id, (bool, np.bool_)) or not isinstance(
            query_original_point_id, (int, np.integer)):
        raise ValueError("Query original point identity must be an integer")
    query = np.flatnonzero(ids == query_original_point_id)
    if len(query) != 1:
        raise ValueError("Requested query original point identity is absent")
    q = int(query[0])
    if label_kind not in {"instance", "part"}:
        raise ValueError("label_kind must be instance or part")
    valid = (labels >= 0) & ~np.isin(labels, tuple(ignore_label_ids) + tuple(background_ids))
    if not valid[q]:
        raise ValueError("Query target is ignored or background")
    target = valid & (labels == labels[q])
    eligible = valid & np.isfinite(values) & (ids != query_original_point_id)
    exclusion = "query original point"
    if token_ids is not None:
        tokens = _ids(token_ids, n, "token_ids")
        if tokens[q] >= 0:
            eligible &= tokens != tokens[q]
            exclusion = "query original point and every point sharing its encoder token"
    if analysis_point_ids is not None:
        analysis_ids = _ids(analysis_point_ids, len(analysis_point_ids), "analysis_point_ids", unique=True)
        if not np.isin(analysis_ids, ids).all():
            raise ValueError("analysis_point_ids contains an absent original point identity")
        eligible &= np.isin(ids, analysis_ids)
    if common_eligible_mask is not None:
        common = np.asarray(common_eligible_mask)
        if common.dtype != np.bool_ or common.shape != (n,):
            raise ValueError("common_eligible_mask must be a boolean raw-point mask")
        eligible &= common
    distances, _ = cKDTree(xyz[target]).query(xyz, workers=1)
    boundary = boundary_regions(xyz, labels, valid_mask=valid, k=boundary_k, radius=boundary_radius)
    regions = {
        "target": target,
        "other_instances": valid & ~target,
        "nearby_other_instances": valid & ~target & (distances <= nearby_radius),
        "target_boundary": target & boundary,
        "target_interior": target & ~boundary,
    }
    absent = []
    if semantic_labels is not None:
        semantics = _ids(semantic_labels, n, "semantic_labels")
        if semantics[q] >= 0:
            regions["same_category_other_instances"] = valid & ~target & (semantics == semantics[q])
        else:
            absent.append("same-category leakage: query semantic label missing")
    else:
        absent.append("same-category leakage: semantic labels unavailable")
    if parent_labels is not None:
        parents = _ids(parent_labels, n, "parent_labels")
        if parents[q] >= 0:
            regions["sibling_parts"] = valid & ~target & (parents == parents[q])
        else:
            absent.append("sibling-part leakage: query parent label missing")
    elif label_kind == "part":
        absent.append("sibling-part leakage: parent object labels unavailable")
    k = int((target & eligible).sum())
    eligible_scores = values[eligible]
    region_stats = {}
    for name, mask in regions.items():
        observed = mask & eligible
        region_stats[name] = {
            "eligible_points": int(observed.sum()),
            "mean_cosine": float(values[observed].mean()) if observed.any() else None,
            "fraction_of_top_target_size": _expected_topk_fraction(eligible_scores, mask[eligible], k),
        }
    record = {
        "scope": "illustrative raw-point feature diagnostic; GT-assisted retrieval depth; not downstream AP",
        "label_kind": label_kind, "query_original_point_id": int(query_original_point_id),
        "query_position": xyz[q].tolist(), "target_label_id": int(labels[q]),
        "self_exclusion": exclusion, "eligible_points": int(eligible.sum()),
        "eligible_point_ids_sha256": _digest(ids[eligible]),
        "retrieval_depth": k, "retrieval_depth_definition": "eligible raw points of the query GT target",
        "tie_policy": "expected fraction under uniform ordering within cutoff score ties",
        "nearby_definition": {"distance": "Euclidean distance to nearest GT target point",
                              "radius": float(nearby_radius), "units": "scene coordinates"},
        "boundary_definition": {"kind": "GT label transition among nearest geometry neighbors",
                                "k": boundary_k, "radius": float(boundary_radius),
                                "units": "scene coordinates", "official_metric": False},
        "common_eligibility_supplied": common_eligible_mask is not None,
        "regions_overlap": True, "regions": region_stats, "unavailable": absent,
        "status": "available" if k > 0 else "no eligible target points",
        "cosine_limits": [-1, 1], "interpretation": "Scores measure query affinity, not calibrated uncertainty",
    }
    return record, {"eligible": eligible, **regions}


def build_feature_research_panels(scene: Any, observations: list[Any], *,
                                  query_indices: list[int], output_directory: str | Path,
                                  seed: int = 0, max_fit_points: int = 12000,
                                  reference_name: str | None = None,
                                  settings: Mapping[str, Any] | None = None
                                  ) -> tuple[list[Any], dict[str, Any]]:
    """Suite bridge: persist independent PCA and add self-excluded query context.

    The caller replaces legacy PCA panels with these and retains RGB, GT, cosine,
    scalar-delta, and downstream panels. ``settings.pca_states`` maps ``model/layer``
    to previously saved states. ``settings.requested_identities`` maps model names
    to required fields; these are verified against cache metadata, never filenames.
    Output paths are fresh; each saved state is linked from panel and report.
    """
    from .contracts import Panel
    from .diagnostics import _scalar_panel

    cfg = dict(settings or {})
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(scene.points), dtype=np.int64)
    report: dict[str, Any] = {"pca": {}, "queries": {}, "skipped": [],
                              "interpretation": PCA_CAVEAT,
                              "pca_mode": "independent frozen per source and feature level",
                              "norm_mode": "independent fixed fit-sample scales per source"}
    panels = []
    query_maps: dict[tuple[str, str, int], np.ndarray] = {}
    tokens: dict[tuple[str, str], np.ndarray | None] = {}
    for observation in observations:
        metadata = observation.metadata
        # Manifests may add descriptive metadata, but that cannot establish an
        # exact requested epoch/checkpoint/preprocessing identity. Cache adapters
        # expose a separate identity measured from payload/source-record evidence.
        cached = metadata.get("adapter") in {"cache", "npz"} or "cache_sha256" in metadata
        measured = metadata.get("artifact_identity", {}) if cached else metadata
        actual = {key: measured[key] for key in ("checkpoint_sha256", "encoder_checkpoint_sha256", "epoch",
                                                 "preprocessing") if measured.get(key) is not None}
        actual["scene_id"] = scene.scene_id  # established by correspondence validation
        actual["query_original_point_ids"] = list(query_indices)
        validate_requested_identity(cfg.get("requested_identities", {}).get(observation.name, {}), actual,
                                    prefix=observation.name)
        for level, features in observation.features.items():
            key = f"{observation.name}/{level}"
            checkpoint = metadata.get("checkpoint_sha256") or metadata.get("encoder_checkpoint_sha256")
            cache = metadata.get("cache_sha256")
            source = {"source_id": str(checkpoint or cache or _digest(features)), "feature_level": level,
                      "fit_scene_id": scene.scene_id, "model_name": observation.name,
                      "checkpoint_sha256": checkpoint, "cache_sha256": cache,
                      "preprocessing": metadata.get("preprocessing", metadata.get("wrapper_settings")),
                      "feature_stages": metadata.get("feature_stages"),
                      "geometry_sha256": _digest(np.asarray(scene.points)),
                      "identity_basis": "checkpoint digest" if checkpoint else
                                        "cache digest; checkpoint identity incomplete" if cache else
                                        "feature-array digest; checkpoint identity incomplete"}
            if metadata.get("epoch") is not None:
                source["epoch"] = metadata["epoch"]
            for field in ("feature_coordinate_system", "coordinate_compatibility_reason"):
                if field in metadata:
                    source[field] = metadata[field]
            configured = cfg.get("pca_states", {}).get(key)
            if configured:
                state = FrozenPCA.load(configured)
            else:
                state = fit_frozen_pca(features, original_point_ids=ids, source_record=source,
                                       max_fit_points=max_fit_points, seed=seed)
            colors = state.rgb(features, source_record=source)
            suffix = hashlib.sha256(key.encode()).hexdigest()[:12]
            path = output / f"pca-{suffix}.npz"
            state.save(path)
            item = {**state.record(), "path": str(path), "reused_state": str(configured) if configured else None,
                    "compatibility": state.compatibility(source)}
            report["pca"][key] = item
            low_percent, high_percent = [100 * float(v) for v in state.metadata["quantiles"]]
            scale_text = f"{low_percent:g}–{high_percent:g}%"
            compatibility_mode = item["compatibility"]["mode"]
            pca_caption = ("Independent frozen PCA" if compatibility_mode == "same_source" else
                           "Explicitly shared frozen PCA projection")
            panels.append(Panel(
                key=f"{key}/pca", title=f"{observation.name} · {level} · frozen PCA", colors=colors,
                caption=f"{pca_caption} and fixed fit-sample {scale_text} scales. "
                        "Colors across independent fits are not aligned. " + PCA_CAVEAT,
                metadata={"kind": "pca", "model_name": observation.name, "feature_level": level,
                          "pca_state_path": str(path), "pca_state_sha256": item["state_sha256"],
                          "scale_semantics": "independent per PCA component, fitted once", "source_record": source},
            ))
            norms = np.linalg.norm(np.asarray(features, dtype=np.float64), axis=1)
            panels.append(_scalar_panel(
                f"{key}/norm", f"{observation.name} · {level} · feature norm", norms,
                limits=(0.0, float(state.metadata["norm_upper_limit"])),
                caption=f"Feature L2 magnitude; independent fixed {high_percent:g}% fit-sample upper limit per source. "
                        "Scale is not shared across models; magnitude is not calibrated uncertainty.",
                metadata={"kind": "norm", "model_name": observation.name, "feature_level": level,
                          "scale_semantics": "independent fixed fit-sample scale",
                          "pca_state_sha256": item["state_sha256"]},
            ))
            token_value = metadata.get("token_ids")
            tokens[observation.name, level] = token_value.get(level) if isinstance(token_value, dict) else token_value
            for query in query_indices:
                query_maps[observation.name, level, query] = query_cosine(features, query)
    if scene.labels is None:
        report["skipped"].append({"kind": "leakage", "reason": "GT target labels unavailable"})
        return panels, report
    for observation in observations:
        for level in observation.features:
            for query in query_indices:
                key = f"{observation.name}/{level}/q{query}"
                scores = query_maps[observation.name, level, query]
                common = None
                # Comparing leakage must exclude the query token of BOTH models,
                # even if their inverse maps differ. Cosine NaNs also use intersection.
                eligibility_members = []
                if reference_name and (reference_name, level, query) in query_maps:
                    common = np.ones(len(ids), dtype=bool)
                    for candidate in observations:
                        model = candidate.name
                        if (model, level, query) not in query_maps:
                            continue
                        eligibility_members.append(model)
                        common &= np.isfinite(query_maps[model, level, query])
                        token_value = tokens.get((model, level))
                        if token_value is not None and token_value[query] >= 0:
                            common &= np.asarray(token_value) != token_value[query]
                try:
                    item, regions = query_diagnostics(
                        scores, points=scene.points, labels=scene.labels, original_point_ids=ids,
                        query_original_point_id=query, semantic_labels=scene.semantic_labels,
                        parent_labels=scene.metadata.get("part_parent_labels"),
                        token_ids=tokens[observation.name, level], common_eligible_mask=common,
                        ignore_label_ids=tuple(scene.metadata.get("ignore_label_ids", [-1])),
                        background_ids=tuple(scene.metadata.get("background_ids", [])),
                        nearby_radius=float(cfg.get("nearby_radius", 0.5)),
                        boundary_radius=float(cfg.get("boundary_radius", 0.1)),
                        boundary_k=int(cfg.get("boundary_k", 8)),
                        label_kind=str(scene.metadata.get("label_kind", "instance")),
                    )
                except ValueError as error:
                    if "Query target is ignored or background" not in str(error):
                        raise
                    report["skipped"].append({"kind": "leakage", "key": key, "reason": str(error)})
                    continue
                item["eligibility_members"] = eligibility_members or [observation.name]
                report["queries"][key] = item
                # Context is categorical: it is never mistaken for a score map.
                categories = np.full(len(ids), -1, dtype=np.int64)
                categories[regions["nearby_other_instances"]] = 2
                if "same_category_other_instances" in regions:
                    categories[regions["same_category_other_instances"]] = 3
                if "sibling_parts" in regions:
                    categories[regions["sibling_parts"]] = 4
                categories[regions["target"]] = 1
                labels = {1: "query GT target", 2: "nearby other object", 3: "same-category other object",
                          4: "sibling part"}
                # Fixed Okabe-Ito category roles; these are regions, not target-ID colors.
                role_colors = {1: [0.0, 0.62, 0.45], 2: [0.90, 0.62, 0.0],
                               3: [0.80, 0.47, 0.65], 4: [0.0, 0.45, 0.70]}
                colors = np.tile(np.asarray([0.53, 0.55, 0.58], dtype=np.float32), (len(ids), 1))
                for category, color in role_colors.items():
                    colors[categories == category] = color
                palette = {str(i): {"label": label, "rgb": role_colors[i]}
                           for i, label in labels.items() if np.any(categories == i)}
                panels.append(Panel(
                    key=f"{observation.name}/{level}/leakage_context/q{query}",
                    title=f"{observation.name} · {level} · query {query} · target context", colors=colors,
                    query_index=query,
                    caption="GT-defined regions for feature leakage diagnostics. Regions overlap in statistics; "
                            "display priority: target, sibling part, same-category object, nearby object. "
                            "GT target size sets retrieval depth; these are illustrative raw-point statistics.",
                    metadata={"kind": "leakage_context", "model_name": observation.name,
                              "feature_level": level, "region_legend": palette, "diagnostics": item},
                ))
    return panels, report


# Short public name retained for suite callers.
feature_research_panels = build_feature_research_panels
