"""Full-scene frozen-feature diagnostics; GT-K/query readouts, never instance AP.

The numerical scoring follows the sealed ScanNet full-validation evaluators.
Labels determine K, query anchors, and post-fit Hungarian scoring only. Every
original point, including ignored/background points, contributes to scoring.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

def pool_features(features, native_inverse, inverse, counts):
    """Preserve original-point weighting when native and common cells differ."""
    import numpy as np
    from scipy.sparse import csr_matrix
    features = np.asarray(features)
    native_inverse, inverse, counts = map(np.asarray, (native_inverse, inverse, counts))
    if features.ndim != 2 or not all(features.shape) or features.dtype.kind not in 'fiu' or not np.isfinite(features).all():
        raise ValueError('Expected finite nonempty native features [tokens, channels]')
    for name, mapping in (('native_inverse', native_inverse), ('point_to_cell', inverse)):
        if mapping.ndim != 1 or not len(mapping) or mapping.dtype.kind not in 'iu' or mapping.min() < 0:
            raise ValueError(f'Invalid {name} mapping')
    if counts.ndim != 1 or not len(counts) or counts.dtype.kind not in 'iu' or counts.min() <= 0:
        raise ValueError('Every common cell must contain original points')
    if native_inverse.shape != inverse.shape or native_inverse.max() >= len(features) or inverse.max() >= len(counts):
        raise ValueError('Mappings do not cover the same original point support')
    if not np.array_equal(np.bincount(inverse, minlength=len(counts)), counts):
        raise ValueError('Common cell counts do not match the original point mapping')
    ntokens, ncells = len(features), len(counts)
    pairs, multiplicity = np.unique(inverse.astype(np.int64) * ntokens + native_inverse, return_counts=True)
    rows, columns = pairs // ntokens, pairs % ntokens
    bijective = (ntokens == ncells and len(pairs) == ncells
                 and np.array_equal(rows, np.arange(ncells)) and len(np.unique(columns)) == ntokens)
    if bijective:
        return np.asarray(features[columns], dtype=np.float32), True
    weights = multiplicity.astype(np.float32) / counts[rows].astype(np.float32)
    matrix = csr_matrix((weights, (rows, columns)), shape=(ncells, ntokens), dtype=np.float32)
    return np.asarray(matrix @ features, dtype=np.float32), False


def select_restart(records):
    if not records or not all(np.isfinite(r['mean_cosine_to_center']) for r in records):
        raise ValueError('Restarts require finite feature objectives')
    return max(range(len(records)), key=lambda i: records[i]['mean_cosine_to_center'])



def confusion(prediction, truth, k):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if k < 1 or prediction.shape != truth.shape:
        raise ValueError('Prediction does not cover the common original points')
    for values in (prediction, truth):
        if values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu' or values.min() < -1 or values.max() >= k:
            raise ValueError('Labels must be integer original-point IDs in [-1, K)')
    prediction, truth = prediction.astype(np.int64), truth.astype(np.int64)
    gt = np.where(truth >= 0, truth, k)
    pred = np.where(prediction >= 0, prediction, k)
    return np.bincount(gt * (k + 1) + pred, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)


def metrics(prediction, truth, objects):
    k = len(objects)
    matrix = confusion(prediction, truth, k)
    size = matrix[:k].sum(axis=1).astype(np.float64)
    predicted = matrix[:, :k].sum(axis=0).astype(np.float64)
    tp = np.diag(matrix)[:k].astype(np.float64)
    union = size + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(k), where=union > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, size, out=np.zeros(k), where=size > 0)
    contamination = np.divide(matrix[k, :k], predicted, out=np.zeros(k), where=predicted > 0)
    cover = matrix[:k, :k] >= size[:, None] * 0.5
    np.fill_diagonal(cover, False)
    split = (matrix[:k, :k] >= size[:, None] * 0.2).sum(axis=1) >= 2
    rows = [{**obj, "iou": float(iou[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "true_positive_points": int(tp[i]),
             "false_positive_points": int(predicted[i] - tp[i]),
             "false_negative_points": int(size[i] - tp[i]),
             "predicted_points": int(predicted[i]), "background_contamination": float(contamination[i]),
             "other_objects_half_covered": int(cover[:, i].sum()), "split_at_20_percent": bool(split[i])}
            for i, obj in enumerate(objects)]
    return {"macro_iou": float(iou.mean()), "micro_iou": float(tp.sum() / max(union.sum(), 1)),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "objects_iou25": float((iou >= 0.25).mean()),
            "objects_iou50": float((iou >= 0.5).mean()), "objects_iou75": float((iou >= 0.75).mean()),
            "foreground_assigned": float(matrix[:k, :k].sum() / max(size.sum(), 1)),
            "background_rejection": float(matrix[k, k] / max(matrix[k].sum(), 1)),
            "background_assigned_points": int(matrix[k, :k].sum()),
            "masks_covering_half_another_object": float((cover.sum(axis=0) > 0).mean()),
            "mean_other_objects_half_covered": float(cover.sum(axis=0).mean()),
            "objects_split_at_20_percent": float(split.mean()), "object_count": k,
            "unassigned_points": int((np.asarray(prediction) < 0).sum()), "per_object": rows}


def match_clusters(clusters, truth, objects):
    clusters = np.asarray(clusters)
    k = len(objects)
    matrix = confusion(clusters, truth, k)
    intersection = matrix[:k, :k].astype(np.float64)
    union = matrix[:k].sum(axis=1)[:, None] + matrix[:, :k].sum(axis=0)[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    rows, cols = linear_sum_assignment(-iou)
    mapping = np.full(k, -1, dtype=np.int32)
    mapping[cols] = rows
    aligned = np.full(len(clusters), -1, dtype=np.int32)
    valid = clusters >= 0
    aligned[valid] = mapping[clusters[valid]]
    return aligned, mapping, iou



def aggregate(records):
    import numpy as np
    if not records or any(not r['per_object'] for r in records):
        raise ValueError('Aggregation requires nonempty scene and object records')
    per_object = [o for r in records for o in r['per_object']]
    result = {'scenes': len(records), 'objects': len(per_object)}
    for key in ('macro_iou', 'macro_precision', 'macro_recall', 'micro_iou'):
        result['scene_mean_' + key] = float(np.mean([r[key] for r in records]))
    for key in ('iou', 'precision', 'recall'):
        result['pooled_object_mean_' + key] = float(np.mean([o[key] for o in per_object]))
    for threshold in (25, 50, 75):
        result['pooled_objects_iou' + str(threshold)] = float(np.mean([o['iou'] >= threshold / 100 for o in per_object]))
    result['by_class'] = {}
    for class_index in sorted({o['class_index'] for o in per_object}):
        group = [o for o in per_object if o['class_index'] == class_index]
        result['by_class'][str(class_index)] = {'class_id': group[0]['class_id'], 'class_name': group[0]['class_name'],
            'objects': len(group), 'mean_iou': float(np.mean([o['iou'] for o in group]))}
    result['represented_class_mean_iou'] = float(np.mean([c['mean_iou'] for c in result['by_class'].values()]))
    large = [o for o in per_object if o['vertices'] >= 100]
    result['objects_at_least_100_vertices'] = {
        'objects': len(large),
        'mean_iou': float(np.mean([o['iou'] for o in large])) if large else None,
        'objects_iou50': float(np.mean([o['iou'] >= .5 for o in large])) if large else None,
        'interpretation': 'subset of the same fitted and matched clustering; K and assignment unchanged; not AP'}
    return result



DEFAULT_SEEDS = (20261010, 20261011, 20261012, 20261013, 20261014)


def shared_geometry(left, right):
    """Reject comparisons with different ordered inputs or common support."""
    for key in ('xyz', 'rgb255', 'normal', 'shift', 'common_grid', 'point_to_cell'):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f'Taxonomies have different {key} input/support')


def cosine_probe(features, point_to_cell, queries):
    """Competing GT-query cosine; retain collisions and all background cells."""
    import torch
    from delimit3d.probes.segmentation import normalize_features
    features = normalize_features(features)
    inverse, queries = np.asarray(point_to_cell), np.asarray(queries)
    for values, upper, name in ((inverse, len(features), 'point_to_cell'),
                                (queries, len(inverse), 'queries')):
        if values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu' or values.min() < 0 or values.max() >= upper:
            raise ValueError(f'Invalid {name}')
    query_cells = inverse[queries]
    anchor = torch.as_tensor(query_cells, device=features.device, dtype=torch.long)
    scores = (features @ features[anchor].T).clamp(-1, 1)
    return (scores.cpu().numpy(), scores.argmax(dim=1).cpu().numpy().astype(np.int32),
            query_cells)


def evaluate_features(features, native_inverse, point_to_cell, labels, queries,
                      objects, *, seeds=DEFAULT_SEEDS, device='cpu',
                      max_iter=100, tol=1e-6, grid_size=0.05):
    """Return reusable common-cell arrays and full-original-point metrics.

    ``features`` is an unnormalized native token array. Both maps cover every
    original point in the same order. GT IDs must be contiguous and queries
    must select one existing vertex of each object, in object index order.
    Different grid sizes are different protocols, recorded in the result.
    This function neither extracts features nor verifies checkpoint freezing.
    """
    import torch
    from delimit3d.probes.segmentation import normalize_features, spherical_kmeans
    labels, queries, inverse = map(np.asarray, (labels, queries, point_to_cell))
    k = len(objects)
    confusion(labels, labels, k)
    if labels.shape != inverse.shape or not np.array_equal(np.unique(labels[labels >= 0]), np.arange(k)):
        raise ValueError('Each target must occur in the full original point support')
    if (queries.shape != (k,) or queries.dtype.kind not in 'iu' or queries.min() < 0
            or queries.max() >= len(labels) or not np.array_equal(labels[queries], np.arange(k))):
        raise ValueError('Queries must select one original GT vertex per target in order')
    for index, obj in enumerate(objects):
        if obj.get('index', index) != index or obj.get('vertices', int((labels == index).sum())) != int((labels == index).sum()):
            raise ValueError('Object metadata does not match original-point labels')
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in seeds):
        raise ValueError('Seeds must be a nonempty list of distinct integers')
    if isinstance(max_iter, (bool, np.bool_)) or max_iter < 1 or int(max_iter) != max_iter or not np.isfinite(tol) or tol < 0:
        raise ValueError('Invalid convergence parameters')
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError('Grid size must be finite and positive')
    if inverse.ndim != 1 or inverse.dtype.kind not in 'iu' or inverse.min() < 0:
        raise ValueError('Invalid point_to_cell mapping')
    counts = np.bincount(inverse)
    pooled, bijective = pool_features(features, native_inverse, inverse, counts)
    # Match the research sequence: normalize after pooling on CPU, then once
    # on the requested device to account for float32 norm arithmetic.
    norms = np.linalg.norm(pooled, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError('Pooling produced a zero feature vector')
    pooled /= norms[:, None]
    tensor = normalize_features(torch.as_tensor(np.ascontiguousarray(pooled), device=device))
    if len(tensor) < k:
        raise ValueError('GT instance count exceeds occupied common cells')
    fits, clusters, centers = [], [], []
    scope = 'all common occupied cells, including background; original-point-weighted pooling'
    with torch.inference_mode():
        for seed in seeds:
            assignment, center, fit = spherical_kmeans(tensor, k, int(seed), max_iter=int(max_iter), tol=tol)
            fit['fit_scope'] = scope
            fits.append(fit)
            clusters.append(assignment)
            centers.append(center)
        selected = select_restart(fits)
        aligned, mapping, _ = match_clusters(clusters[selected][inverse], labels, objects)
        scores, cosine_labels, query_cells = cosine_probe(tensor, inverse, queries)
    arrays = {
        'features': tensor.cpu().numpy(), 'inverse_map': inverse,
        'point_to_cell': inverse, 'restart_cell_clusters': np.stack(clusters),
        'restart_centers': np.stack(centers), 'selected_restart': np.asarray(selected),
        'cluster_to_gt': mapping, 'kmeans_cell_labels': mapping[clusters[selected]],
        'cosine_cell_labels': cosine_labels, 'cosine_query_cells': query_cells,
        'cosine_cell_scores': scores, 'prediction': mapping[clusters[selected]],
        'point_prediction': aligned,
    }
    record = {
        'schema': 'loc3d.frozen_feature_diagnostics/v1', 'status': 'COMPLETE',
        'claim_boundary': 'GT-K and GT-query frozen-feature diagnostics; not instance AP',
        'original_points': len(labels), 'common_cells': len(tensor), 'objects': k,
        'native_tokens': len(features), 'native_feature_dim': tensor.shape[1],
        'bijective_native_common_support': bijective, 'grid_size_m': grid_size,
        'fit_scope': scope, 'k_source': 'eligible GT raw-instance count',
        'restart_selection': 'maximum mean cosine to center; first restart wins ties',
        'hungarian_scope': 'post-fit maximum-total-IoU on all original points',
        'cosine_query_source': 'one supplied original GT vertex per target',
        'cosine_background_rejection': False, 'cosine_query_collisions_retained': True,
        'seeds': list(map(int, seeds)), 'max_iterations': int(max_iter), 'tolerance': tol,
        'selected_restart': selected, 'selected_seed': int(seeds[selected]),
        'kmeans_restarts': fits, 'device': str(device), 'torch_version': torch.__version__,
        'metrics': {'kmeans': metrics(aligned, labels, objects),
                    'cosine': metrics(cosine_labels[inverse], labels, objects)},
    }
    return arrays, record
