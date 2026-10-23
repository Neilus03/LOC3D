"""Closed-vocabulary labeling of fixed instances with a linear classifier.

The module takes object descriptors, never point labels, as classifier inputs.
Geometric matching and the official-style semantic AP records remain separate
from fitting so evaluation labels cannot influence predictions.
"""
from __future__ import annotations

import numpy as np

CLASS_IDS = np.asarray([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39], dtype=np.int64)
CLASS_NAMES = ["cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"]
MIN_VERTICES = 100


def pool_reference(point_features, masks):
    """Independent CPU definition: vertex-weighted mean and channelwise max."""
    point_features = np.asarray(point_features, dtype=np.float32)
    rows = []
    for mask in masks:
        selected = point_features[np.asarray(mask, dtype=bool)]
        if not len(selected):
            raise ValueError("An empty instance has no descriptor")
        rows.append(np.concatenate([selected.mean(0, dtype=np.float64), selected.max(0)]))
    return np.asarray(rows, dtype=np.float32).reshape(-1, point_features.shape[1] * 2)


def probabilities(descriptors, weights, intercept, mean, scale):
    x = (np.asarray(descriptors, dtype=np.float64) - mean) / scale
    logits = x @ np.asarray(weights, dtype=np.float64).T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def accuracy(rows):
    result = {}
    for cid, name in zip(CLASS_IDS, CLASS_NAMES):
        selected = [r for r in rows if r["gt_class"] == int(cid)]
        correct = sum(r["predicted_class"] == int(cid) for r in selected)
        result[name] = {"class_id": int(cid), "objects": len(selected), "correct": correct,
                        "accuracy": correct / len(selected) if selected else None}
    present = [v["accuracy"] for v in result.values() if v["accuracy"] is not None]
    n = len(rows)
    correct = sum(r["predicted_class"] == r["gt_class"] for r in rows)
    return {"objects": n, "correct": correct, "accuracy": correct / n if n else None,
            "class_mean_accuracy": float(np.mean(present)) if present else None,
            "per_class": result}


def geometry_matches(intersections, pred_sizes, gt_sizes, eligible, candidates):
    """Class-blind maximum-total-IoU matching before label examination."""
    from scipy.optimize import linear_sum_assignment
    pi = np.flatnonzero(candidates)
    gi = np.flatnonzero(eligible)
    inter = np.asarray(intersections, dtype=np.float64)[np.ix_(pi, gi)]
    union = np.asarray(pred_sizes)[pi, None] + np.asarray(gt_sizes)[None, gi] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    if not iou.size:
        return []
    pp, gg = linear_sum_assignment(-iou)
    return [(int(pi[p]), int(gi[g]), float(iou[p, g])) for p, g in zip(pp, gg)]


def official_match_record(scene, gt_ids, gt_classes, gt_sizes, pred_classes,
                          pred_sizes, scores, intersections, void_intersections):
    """Build the ScanNet evaluator's count records without materializing text masks.

    Ground-truth IDs are encoded as category*1000 + a scene-local instance ID.
    Small GT instances remain in the record for the reference ignore logic.
    Overlap counts include all original vertices; background is never cropped
    out of predicted masks before computing their sizes or intersections.
    """
    gt = {name: [] for name in CLASS_NAMES}
    pred = {name: [] for name in CLASS_NAMES}
    by_class = {int(c): n for c, n in zip(CLASS_IDS, CLASS_NAMES)}
    for gid, cid, size in zip(gt_ids, gt_classes, gt_sizes):
        gt[by_class[int(cid)]].append({"instance_id": int(gid), "label_id": int(cid),
            "vert_count": int(size), "med_dist": -1.0, "dist_conf": 0.0, "matched_pred": []})
    locations = {v["instance_id"]: v for values in gt.values() for v in values}
    for i, (cid, size, score, ignored) in enumerate(zip(pred_classes, pred_sizes, scores, void_intersections)):
        if int(size) < MIN_VERTICES:
            continue
        item = {"uuid": f"{scene}:p{i}", "pred_id": i, "label_id": int(cid),
                "vert_count": int(size), "confidence": float(score),
                "void_intersection": int(ignored), "matched_gt": []}
        for j, (gid, gc) in enumerate(zip(gt_ids, gt_classes)):
            inter = int(intersections[i, j])
            if int(gc) != int(cid) or inter == 0:
                continue
            original = locations[int(gid)]
            gcopy = {k: v for k, v in original.items() if k != "matched_pred"}
            gcopy["intersection"] = inter
            pcopy = {k: v for k, v in item.items() if k != "matched_gt"}
            pcopy["intersection"] = inter
            original["matched_pred"].append(pcopy)
            item["matched_gt"].append(gcopy)
        pred[by_class[int(cid)]].append(item)
    return {"gt": gt, "pred": pred}


def reference_ap(reference, matches):
    """Run the sealed ScanNet18 reference's original matching and AP integration."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        values = reference.evaluate_matches(matches)
    aps = values[0] if isinstance(values, tuple) else values
    result = reference.compute_averages(aps)
    return {"AP": float(result["all_ap"]), "AP50": float(result["all_ap_50%"]),
            "AP25": float(result["all_ap_25%"]), "per_class": {
                name: {"AP": float(row["ap"]), "AP50": float(row["ap50%"]),
                       "AP25": float(row["ap25%"])} for name, row in result["classes"].items()}}
