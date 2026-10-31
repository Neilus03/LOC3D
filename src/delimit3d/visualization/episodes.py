"""Faithful saved AGILE3D episodes: persistent labels, actual clicks, recorded metrics.

No per-frame Hungarian matching is performed. Predictions use one immutable
``prediction_target_map`` from decoder query/label IDs to scene object/part IDs.
Missing masks leave a usable trace record and target/click panels.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .contracts import Observation, Panel, Scene
from .export_contracts import (EPISODE_SCHEMA, integer_array, pairing_status, read_bundle,
                               validate_geometry, validate_inverse)

MODES = {
    "recorded_benchmark": "Recorded benchmark evaluation; actual evaluator-selected clicks",
    "controlled_replay": "Controlled replay; identical clicks require cross-model identity verification",
    "exploratory_replay": "Exploratory user-selected replay; not a benchmark evaluation",
}

SCANNET40_SOURCE_IMPLEMENTATION = (
    "frozen run_scannet40_matched.py:425-439 at 4a1888213f0507eae86b51d16df76b7d8809ba7b"
)
TARGET_LABEL_POLICIES = {"preserve_negative_ignored", "all_non_targets_are_background"}


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class Episode:
    record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    predictions: np.ndarray | None
    target_labels: np.ndarray
    pairing: dict[str, Any]

    @property
    def clicks(self) -> list[dict[str, Any]]:
        return self.record["clicks"]

    def to_observation(self, name: str) -> Observation:
        if self.predictions is None:
            raise ValueError("Episode has recorded clicks/metrics but no saved prediction masks")
        counts = [frame["click_count"] for frame in self.record["frames"]]
        clicks = []
        for click in self.clicks:
            active = next((i for i, count in enumerate(counts) if click["order"] < count), len(counts))
            clicks.append({**click, "step": active, "object_id": click.get("assigned_label_id")})
        return Observation(name=name, prediction=self.predictions[-1], steps=self.predictions,
                           clicks=clicks, metadata={
                               "adapter": "agile3d_recorded", "episode": self.record,
                               "pairing": self.pairing, "prediction_kind": "persistent_instance",
                               "prediction_background_ids": [self.record["background_id"]],
                               "target_labels": self.target_labels,
                               "temporal_identity": "fixed recorded decoder-ID to target-ID mapping",
                               "preprocessing": self.record.get("model", {}).get("preprocessing"),
                           })


def episode_from_arrays(record: Mapping[str, Any], arrays: Mapping[str, Any], scene: Scene, *,
                        expected_pairing: Mapping[str, Any] | None = None) -> Episode:
    scene.validate()
    record = deepcopy(dict(record))
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if record.get("schema") != EPISODE_SCHEMA:
        raise ValueError(f"Expected schema {EPISODE_SCHEMA}")
    validate_geometry(record, arrays, scene)
    if record.get("mode") not in MODES:
        raise ValueError(f"Episode mode must be one of {sorted(MODES)}")
    if not record.get("episode_id"):
        raise ValueError("Episode needs a stable episode_id")
    targets = record.get("targets", [])
    if not targets or any(not target.get("target_id") for target in targets):
        raise ValueError("Episode needs target_id and label_id for each target")
    target_ids = [str(target["target_id"]) for target in targets]
    labels = [_int(target.get("label_id"), "target.label_id") for target in targets]
    background = _int(record.get("background_id"), "background_id")
    if len(set(target_ids)) != len(target_ids) or len(set(labels)) != len(labels) or background in labels:
        raise ValueError("Target IDs/label IDs must be unique and distinct from background_id")
    target_lookup = dict(zip(target_ids, labels))
    domain = record.get("metric_domain")
    if domain not in {"raw_points", "tokens"}:
        raise ValueError("metric_domain must be raw_points or tokens")
    budget = record.get("click_budget", {})
    for key in ("definition", "maximum", "counts_background", "denominator"):
        if key not in budget:
            raise ValueError(f"click_budget.{key} must preserve evaluator semantics")
    if not isinstance(budget["definition"], str) or not budget["definition"]:
        raise ValueError("click_budget.definition must be explicit text")
    _int(budget["maximum"], "click_budget.maximum", 1)
    if not isinstance(budget["counts_background"], bool) or not budget["denominator"]:
        raise ValueError("click_budget needs explicit counts_background and denominator")
    if not record.get("click_policy"):
        raise ValueError("click_policy must name the recorded click-selection rule")
    frame_ids, previous_count = set(), -1
    frames = record.get("frames", [])
    if not frames:
        raise ValueError("Episode must contain recorded frames (masks can be absent)")
    clicks = record.get("clicks", [])
    click_ids = set()
    token_count = None
    if "inverse_map" in arrays:
        inverse = integer_array(arrays["inverse_map"], "inverse_map", (len(scene.points),))
        token_count = int(inverse.max(initial=-1)) + 1
        if token_count == 0:
            raise ValueError("inverse_map cannot be empty")
        validate_inverse(arrays, len(scene.points), token_count)
    for order, click in enumerate(clicks):
        if not isinstance(click.get("click_id"), str) or not click["click_id"] or click["click_id"] in click_ids:
            raise ValueError("Each click requires a unique persistent click_id")
        click_ids.add(click["click_id"])
        if _int(click.get("order"), "click.order") != order:
            raise ValueError("click.order must be contiguous recorded event order starting at zero")
        index = _int(click.get("point_index"), "click.point_index")
        if index >= len(scene.points):
            raise ValueError("Click point_index outside original scene point domain")
        position = np.asarray(click.get("position"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.allclose(
                position, scene.points[index], atol=0, rtol=0):
            raise ValueError("Click position does not match its exact original point identity")
        polarity = click.get("polarity")
        if polarity not in {"positive", "negative", "background"}:
            raise ValueError("click.polarity must be positive, negative, or background")
        target = click.get("target_id")
        if polarity != "background" and str(target) not in target_lookup:
            raise ValueError("Positive/negative clicks require a recorded target_id")
        if polarity == "background" and target is not None and str(target) not in target_lookup:
            raise ValueError("Unknown background-click target_id")
        assigned = _int(click.get("assigned_label_id"), "click.assigned_label_id")
        if assigned not in {background, *labels}:
            raise ValueError("Click assigned_label_id is outside the persistent target universe")
        if polarity == "positive" and assigned != target_lookup[str(target)]:
            raise ValueError("Positive click assigned label differs from its target identity")
        if polarity in {"negative", "background"} and assigned != background:
            raise ValueError("Negative/background clicks must retain the recorded background assignment")
        if "token_index" in click:
            token = _int(click["token_index"], "click.token_index")
            if token_count is None or token >= token_count or arrays["inverse_map"][index] != token:
                raise ValueError("Click token_index does not correspond to original point_index")
    for frame in frames:
        frame_id = frame.get("frame_id")
        if frame_id is None or str(frame_id) in frame_ids:
            raise ValueError("Frames require unique persistent frame_id values")
        frame_ids.add(str(frame_id))
        count = _int(frame.get("click_count"), "frame.click_count")
        if count < previous_count or count > len(clicks):
            raise ValueError("Frame click_count must be monotone and within recorded click history")
        previous_count = count
        for metric in frame.get("metrics", []):
            if not metric.get("name") or metric.get("domain") not in {"raw_points", "tokens"}:
                raise ValueError("Recorded metrics need name and explicit raw_points/tokens domain")
            if not metric.get("unit") or not np.isfinite(float(metric.get("value", np.nan))):
                raise ValueError("Recorded metrics need a finite value and explicit unit")
            if metric.get("target_id") is not None and str(metric["target_id"]) not in target_lookup:
                raise ValueError("Metric refers to an unknown target identity")
            if "iou" in str(metric["name"]).lower():
                maximum = 1 if metric["unit"] == "fraction" else 100 if metric["unit"] == "percent" else None
                if maximum is None or not 0 <= float(metric["value"]) <= maximum:
                    raise ValueError("IoU metric value is inconsistent with its declared unit")
    target_labels = arrays.get("target_labels", scene.labels)
    if target_labels is None:
        raise ValueError("Episode requires saved target_labels or scene ground truth")
    target_labels = integer_array(target_labels, "target_labels", (len(scene.points),))
    policy = record.get("target_label_policy")
    if policy is None and record.get("source_implementation") == SCANNET40_SOURCE_IMPLEMENTATION:
        # Early exports of this exact frozen evaluator omitted target_labels. Its
        # raw truth starts at zero and only requested objects are assigned labels:
        # even original negative/unlabelled points count in false-positive unions.
        # This compatibility rule must not leak into arbitrary recorded episodes.
        policy = "all_non_targets_are_background"
        record["target_label_policy_compatibility"] = {
            "inferred_from_exact_source_implementation": SCANNET40_SOURCE_IMPLEMENTATION,
            "reason": "Frozen evaluator assigns every non-target raw point to background, including negative scene labels",
        }
    policy = policy or "preserve_negative_ignored"
    if policy not in TARGET_LABEL_POLICIES:
        raise ValueError(f"target_label_policy must be one of {sorted(TARGET_LABEL_POLICIES)}")
    record["target_label_policy"] = policy
    retain = np.isin(target_labels, labels)
    if policy == "preserve_negative_ignored":
        retain |= target_labels < 0
    target_labels = np.where(retain, target_labels, background)
    for label in labels:
        if not np.any(target_labels == label):
            raise ValueError(f"Recorded target label {label} has no points in target ground truth")
    predictions = None
    if "predictions" in arrays:
        native = integer_array(arrays["predictions"], "predictions", (len(frames), None))
        prediction_domain = record.get("prediction_domain")
        if prediction_domain not in {"raw_points", "tokens"}:
            raise ValueError("Saved predictions require prediction_domain raw_points or tokens")
        if prediction_domain == "tokens":
            inverse = validate_inverse(arrays, len(scene.points), native.shape[1])
            raw = native[:, inverse]
        elif native.shape[1] != len(scene.points):
            raise ValueError("Raw predictions do not match original point count")
        else:
            raw = native
        mapping = record.get("prediction_target_map")
        if not isinstance(mapping, Mapping):
            raise ValueError("Predictions need one fixed prediction_target_map for the whole sequence")
        parsed_mapping = {int(key): target_lookup[str(value)] for key, value in mapping.items()
                          if str(value) in target_lookup}
        if len(parsed_mapping) != len(mapping) or background in parsed_mapping:
            raise ValueError("prediction_target_map contains an unknown target or maps background")
        if len(set(parsed_mapping.values())) != len(parsed_mapping):
            raise ValueError("Multiple decoder IDs cannot map to the same interactive target")
        unknown = set(np.unique(raw).tolist()) - {background, *parsed_mapping}
        if unknown:
            raise ValueError(f"Predictions contain unmapped persistent decoder IDs: {sorted(unknown)}")
        predictions = np.full(raw.shape, background, dtype=np.int64)
        for pred_id, label in parsed_mapping.items():
            predictions[raw == pred_id] = label
    status = pairing_status(record.get("model", {}), expected_pairing)
    record["pairing_status"] = status
    record["missing_artifacts"] = [] if predictions is not None else ["predictions: per-frame masks were not exported"]
    return Episode(record, arrays, predictions, target_labels, status)


def load_episode(path: str | Path, scene: Scene, *, expected_pairing: Mapping[str, Any] | None = None
                 ) -> Episode:
    record, arrays = read_bundle(path)
    return episode_from_arrays(record, arrays, scene, expected_pairing=expected_pairing)


def compare_click_sequences(episodes: list[Episode]) -> dict[str, Any]:
    """Compare scene, target universe and full event identities; adaptive clicks may differ."""
    if not episodes:
        raise ValueError("At least one episode is required")
    def identity(episode: Episode) -> tuple[Any, ...]:
        return (episode.record["scene_id"], episode.record["source_geometry_sha256"],
                tuple((str(t["target_id"]), t["label_id"]) for t in episode.record["targets"]),
                tuple((c["click_id"], c["order"], c["point_index"], tuple(c["position"]),
                       c.get("target_id"), c["polarity"], c["assigned_label_id"], c.get("token_index"))
                      for c in episode.clicks))
    identical = all(identity(item) == identity(episodes[0]) for item in episodes[1:])
    frame_counts_match = all([f["click_count"] for f in item.record["frames"]] ==
                            [f["click_count"] for f in episodes[0].record["frames"]] for item in episodes[1:])
    return {"identical_click_identities": identical, "identical_frame_click_counts": frame_counts_match,
            "comparison_count": len(episodes),
            "label": "Identical recorded clicks" if identical and len(episodes) > 1 else
                     "Actual recorded clicks; identical-click comparison not established",
            "modes": [item.record["mode"] for item in episodes]}


def _error_values(prediction: np.ndarray, truth: np.ndarray, label: int) -> np.ndarray:
    valid = truth >= 0
    positive, target = prediction == label, truth == label
    values = np.zeros(len(truth), dtype=np.int64)
    values[valid & positive & target] = 1
    values[valid & positive & ~target] = 2
    values[valid & ~positive & target] = 3
    values[~valid] = -1
    return values


def episode_panels(scene: Scene, episode: Episode, name: str, *, boundary_radius: float | None = None
                   ) -> tuple[list[Panel], dict[str, Any]]:
    from .diagnostics import _GRAY, _label_colors
    record, truth = episode.record, episode.target_labels
    valid = (truth >= 0) & (truth != record["background_id"])
    target_legend = {str(target["label_id"]): {"label": str(target["target_id"]),
        "rgb": _label_colors(np.array([target["label_id"]]))[0].tolist()} for target in record["targets"]}
    target_legend[str(record["background_id"])] = {"label": "Background", "rgb": _GRAY.tolist()}
    error_legend = {"1": {"label": "True positive", "rgb": [0, .62, .45]},
                    "2": {"label": "False positive", "rgb": [.84, .37, 0]},
                    "3": {"label": "False negative", "rgb": [.34, .56, .9]},
                    "0": {"label": "Other / ignored", "rgb": _GRAY.tolist()}}
    panels = [Panel(key=f"{name}/episode_target", title=f"{name} · target ground truth",
                    colors=_label_colors(truth, valid), values=truth,
                    caption="Persistent target colors shared across all recorded frames.",
                    metadata={"targets": record["targets"], "episode_id": record["episode_id"],
                              "category_legend": target_legend})]
    report = {"episode_id": record["episode_id"], "mode": record["mode"],
              "mode_description": MODES[record["mode"]], "pairing": episode.pairing,
              "metric_domain": record["metric_domain"], "click_budget": record["click_budget"],
              "target_label_policy": record["target_label_policy"],
              "target_label_policy_compatibility": record.get("target_label_policy_compatibility"),
              "missing_artifacts": record["missing_artifacts"], "frames": []}
    if episode.predictions is None:
        panels[0].caption += " Prediction masks unavailable; recorded clicks/metrics retained in the report."
        panels[0].metadata["clicks"] = episode.clicks
        report["recorded_frames"] = record["frames"]
        return panels, report
    boundary = None
    if boundary_radius is not None:
        if not np.isfinite(boundary_radius) or boundary_radius <= 0:
            raise ValueError("boundary_radius must be positive in scene coordinate units")
        tree = cKDTree(scene.points)
        boundary = np.zeros(len(truth), dtype=bool)
        for start in range(0, len(truth), 2048):
            groups = tree.query_ball_point(scene.points[start:start + 2048], boundary_radius, workers=1)
            for offset, neighbors in enumerate(groups):
                i = start + offset
                boundary[i] = truth[i] >= 0 and np.any((truth[neighbors] >= 0) & (truth[neighbors] != truth[i]))
    previous_ious: dict[str, float] = {}
    for step, (frame, prediction) in enumerate(zip(record["frames"], episode.predictions)):
        clicks = episode.clicks[:frame["click_count"]]
        metadata = {"step": step, "frame_id": frame["frame_id"], "clicks": clicks,
                    "click_count": frame["click_count"], "mode": record["mode"],
                    "metric_domain": record["metric_domain"], "click_budget": record["click_budget"],
                    "target_label_policy": record["target_label_policy"],
                    "recorded_metrics": frame.get("metrics", []), "pairing": episode.pairing,
                    "targets": record["targets"], "identity_matching": "none; fixed recorded IDs",
                    "category_legend": target_legend}
        metric_text = "; ".join(f"{m['name']}[{m.get('target_id', 'mean')}]={m['value']:g} {m['unit']} ({m['domain']})"
                                for m in frame.get("metrics", []))
        caption = (f"{MODES[record['mode']]}. {frame['click_count']} recorded click events. "
                   f"Budget: {record['click_budget']['definition']}. Persistent target IDs; no frame matching.")
        if metric_text:
            caption += " Recorded: " + metric_text + "."
        if episode.pairing["missing"]:
            caption += " Encoder/decoder/preprocessing record incomplete."
        panels.append(Panel(key=f"{name}/episode_step_{step}",
                            title=f"{name} · {frame['click_count']} events · "
                                  + (f"budget {frame['clicks_per_object_budget']}/object" if "clicks_per_object_budget" in frame
                                     else str(frame["frame_id"])),
                            colors=_label_colors(prediction, prediction != record["background_id"]),
                            values=prediction, caption=caption, metadata=metadata))
        target_reports = []
        for target in record["targets"]:
            label, target_id = target["label_id"], str(target["target_id"])
            errors = _error_values(prediction, truth, label)
            colors = np.tile(_GRAY, (len(truth), 1))
            colors[errors == 1], colors[errors == 2], colors[errors == 3] = [0, .62, .45], [.84, .37, 0], [.34, .56, .9]
            tp, fp, fn = (int(np.count_nonzero(errors == i)) for i in (1, 2, 3))
            iou = tp / (tp + fp + fn) if tp + fp + fn else 1.
            target_info = {"target_id": target_id, "tp": tp, "fp": fp, "fn": fn,
                           "illustrative_raw_point_iou": iou, "official_metric": False,
                           "regression_from_previous": target_id in previous_ious and iou < previous_ious[target_id]}
            # Only raw-point IoU with the same denominator can be checked from displayed arrays.
            for metric in frame.get("metrics", []):
                if (metric["name"] in {"iou", "raw_point_iou"} and metric["domain"] == "raw_points"
                        and str(metric.get("target_id")) == target_id):
                    expected = float(metric["value"]) / (100 if metric["unit"] == "percent" else 1)
                    target_info["recorded_iou_matches_display_domain"] = bool(np.isclose(expected, iou, atol=1e-6, rtol=0))
                    if not target_info["recorded_iou_matches_display_domain"]:
                        raise ValueError(f"Recorded raw-point IoU disagrees with saved mask for {target_id}, frame {frame['frame_id']}")
            previous_ious[target_id] = iou
            other = [item for item in record["targets"] if item["label_id"] != label]
            target_info["interference_points"] = {str(item["target_id"]): int(np.count_nonzero(
                (prediction == label) & (truth == item["label_id"]))) for item in other}
            panels.append(Panel(key=f"{name}/episode_step_{step}/target_{label}_error",
                                title=f"{target_id} · errors · {frame['click_count']} clicks",
                                colors=colors, values=errors,
                                caption="Green: true positive; orange: false positive; blue: false negative; "
                                        "gray: other/ignored. Original raw-point diagnostic, not a benchmark aggregate.",
                                metadata={**metadata, **target_info, "failure_type": "object_or_part_error",
                                          "category_legend": error_legend}))
            if boundary is not None:
                error_boundary = boundary & np.isin(errors, [2, 3])
                target_info["boundary_error_points"] = int(error_boundary.sum())
                boundary_colors = np.tile(_GRAY, (len(truth), 1))
                boundary_colors[error_boundary] = colors[error_boundary]
                panels.append(Panel(key=f"{name}/episode_step_{step}/target_{label}_boundary",
                    title=f"{target_id} · boundary errors", colors=boundary_colors,
                    caption=f"False positives/negatives within {boundary_radius:g} scene units of a GT label boundary; "
                            "geometry-based display diagnostic.", metadata={**metadata, "boundary_radius": boundary_radius, "category_legend": error_legend}))
            target_reports.append(target_info)
        for metric in frame.get("metrics", []):
            if metric["name"] == "mean_iou" and metric["domain"] == "raw_points":
                expected = float(metric["value"]) / (100 if metric["unit"] == "percent" else 1)
                measured = float(np.mean([row["illustrative_raw_point_iou"] for row in target_reports]))
                if not np.isclose(expected, measured, atol=1e-6, rtol=0):
                    raise ValueError(f"Recorded raw-point mean IoU disagrees with saved masks, frame {frame['frame_id']}")
        report["frames"].append({**frame, "targets": target_reports})
    return panels, report


def import_scannet40_episode(trace_path: str | Path, prediction_path: str | Path,
                             cache_path: str | Path, scene: Scene, *, model: Mapping[str, Any],
                             source_snapshot: str | Path | None = None,
                             geometry_atol: float = 0.0) -> Episode:
    """Import the frozen short-adaptation ScanNet40 evaluator's saved MO/SO snapshots.

    This specific format was verified against frozen ``run_scannet40_matched.py``
    lines 425–439 (source 4a1888213f0507eae86b51d16df76b7d8809ba7b):
    ``click_n`` stores token predictions after ``n * object_count`` accounted
    events; metrics are raw-point IoU after inverse-map expansion. Perfect
    prediction padding accounts virtual events but adds no physical clicks.
    Original traces and source hashes remain in the imported record.
    """
    import json
    from .adapters import _array, _ordered_geometry, _torch_load
    from .export_contracts import file_hash, geometry_hash
    trace_path, prediction_path, cache_path = map(Path, (trace_path, prediction_path, cache_path))
    trace = json.loads(trace_path.read_text())
    if trace.get("scene") != scene.scene_id:
        raise ValueError("Historical trace scene differs from original scene")
    count, object_ids = int(trace["object_count"]), [int(value) for value in trace["object_ids"]]
    if count != len(object_ids) or not count:
        raise ValueError("Historical trace object_count differs from target identities")
    cache = _torch_load(cache_path)
    _ordered_geometry(cache, {"geometry_atol": geometry_atol}, scene, cache_path.parent)
    arrays = {key: _array(cache[key]) for key in ("inverse_map", "representative_indices") if key in cache}
    if "representative_indices" not in arrays:
        raise ValueError("Historical token clicks require saved representative_indices")
    representatives = arrays["representative_indices"]
    if "scene_xyz" in cache:
        arrays["token_points"] = _array(cache["scene_xyz"])
    if "shift" in cache:
        arrays["coordinate_shift"] = _array(cache["shift"])
    inverse = validate_inverse(arrays, len(scene.points), len(representatives))
    model = deepcopy(dict(model))
    decoder = model.setdefault("decoder", {})
    recorded_decoder = trace.get("checkpoint_sha256")
    if decoder.get("sha256") and decoder["sha256"] != recorded_decoder:
        raise ValueError("Historical trace decoder checkpoint hash differs from supplied model")
    decoder["sha256"] = recorded_decoder
    model.setdefault("encoder", {})
    if cache.get("wrapper_settings"):
        import hashlib
        preprocessing = dict(cache["wrapper_settings"])
        if model.get("preprocessing") and model["preprocessing"] != preprocessing:
            raise ValueError("Historical cache preprocessing differs from supplied model record")
        model["preprocessing"] = preprocessing
        model["preprocessing_sha256"] = hashlib.sha256(json.dumps(preprocessing, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    # Historical caches retain the encoder hash; any supplied value must agree.
    encoder_hash = cache.get("checkpoint_sha256", cache.get("encoder_checkpoint_sha256"))
    if encoder_hash is not None:
        encoder_hash = str(encoder_hash)
        if model["encoder"].get("sha256") and model["encoder"]["sha256"] != encoder_hash:
            raise ValueError("Historical cache encoder checkpoint hash differs from supplied model")
        model["encoder"]["sha256"] = encoder_hash
    events = {}
    states = trace["states"]
    for state in states:
        for label, tokens in state.get("clicks", {}).items():
            times = state.get("click_times", {}).get(str(label), [])
            if len(times) != len(tokens):
                raise ValueError("Historical clicks and click_times lengths differ")
            for token, time in zip(tokens, times):
                token, time, label = int(token), int(time), int(label)
                if not 0 <= token < len(representatives) or not 0 <= label <= count:
                    raise ValueError("Historical click token/target outside recorded domain")
                identity = (token, label)
                if time in events and events[time] != identity:
                    raise ValueError("Historical click identity changed at the same recorded event time")
                events[time] = identity
        actual_count = sum(map(len, state.get("clicks", {}).values()))
        if int(state["total_clicks"]) != actual_count and not state.get("perfect_prediction_padded"):
            raise ValueError("Historical accounted clicks differ from events without declared perfect padding")
    if sorted(events) != list(range(len(events))):
        raise ValueError("Historical global click times must be contiguous from zero")
    targets = [{"target_id": str(obj), "label_id": obj,
                "native_decoder_id": index + 1,
                "object_record": trace.get("objects", [{}] * count)[index]}
               for index, obj in enumerate(object_ids)]
    clicks = []
    for order, (token, label) in sorted(events.items()):
        index = int(representatives[token])
        target = str(object_ids[label - 1]) if label else None
        clicks.append({"click_id": f"{scene.scene_id}/event-{order}", "order": order,
                       "point_index": index, "position": scene.points[index].tolist(),
                       "token_index": token, "target_id": target,
                       "assigned_label_id": object_ids[label - 1] if label else 0,
                       "polarity": "positive" if label else "background",
                       "native_assigned_label_id": label})
        if "token_points" in arrays:
            clicks[-1]["native_position"] = arrays["token_points"][token].tolist()
            clicks[-1]["native_coordinate_frame"] = "saved preprocessed scene_xyz token positions"
    frames, snapshots = [], []
    with np.load(prediction_path, allow_pickle=False) as saved:
        keys = sorted(saved.files, key=lambda key: int(key.removeprefix("click_")))
        for key in keys:
            if not key.startswith("click_"):
                raise ValueError("Historical snapshot NPZ must contain only click_n prediction arrays")
            budget = int(key.removeprefix("click_"))
            accounted = budget * count
            candidates = [state for state in states if int(state["total_clicks"]) == accounted]
            if len(candidates) != 1:
                raise ValueError(f"Saved {key} does not identify one trace state at {accounted} total events")
            state = candidates[0]
            snapshot = integer_array(saved[key], key, (len(representatives),))
            native_truth = np.zeros(len(scene.points), dtype=np.int64)
            if scene.labels is None:
                raise ValueError("Historical import requires original scene instance ground truth")
            for dense_id, obj in enumerate(object_ids, 1):
                native_truth[scene.labels == obj] = dense_id
            raw_prediction = snapshot[inverse]
            metric_records = []
            for dense_id, obj in enumerate(object_ids, 1):
                truth_mask, pred_mask = native_truth == dense_id, raw_prediction == dense_id
                union = np.count_nonzero(truth_mask | pred_mask)
                iou = np.count_nonzero(truth_mask & pred_mask) / union if union else 1.
                expected = float(state["object_ious"][str(dense_id)])
                if not np.isclose(iou, expected, atol=1e-8, rtol=0):
                    raise ValueError(f"Saved {key} raw-point IoU for object {obj} differs from its trace; "
                                     "check instance labels, inverse map and snapshot identity")
                metric_records.append({"name": "raw_point_iou", "value": expected, "unit": "fraction",
                                       "domain": "raw_points", "target_id": str(obj)})
            if not np.isclose(float(state["mean_iou"]), np.mean([m["value"] for m in metric_records]), atol=1e-8, rtol=0):
                raise ValueError(f"Saved {key} mean IoU differs from the recorded target mean")
            metric_records.append({"name": "mean_iou", "value": float(state["mean_iou"]),
                                   "unit": "fraction", "domain": "raw_points"})
            frames.append({"frame_id": key, "click_count": sum(map(len, state["clicks"].values())),
                           "accounted_total_clicks": accounted, "clicks_per_object_budget": budget,
                           "perfect_prediction_padded": bool(state.get("perfect_prediction_padded", False)),
                           "metrics": metric_records})
            snapshots.append(snapshot)
    arrays["predictions"] = np.stack(snapshots)
    arrays["target_labels"] = np.where(np.isin(scene.labels, object_ids), scene.labels, 0)
    source_files = {str(path.resolve()): file_hash(path) for path in (trace_path, prediction_path, cache_path)}
    if source_snapshot is not None:
        source_files[str(Path(source_snapshot).resolve())] = file_hash(Path(source_snapshot))
    record = {"schema": EPISODE_SCHEMA, "episode_id": f"scannet40/{scene.scene_id}/" + "-".join(map(str, object_ids)),
              "scene_id": scene.scene_id, "source_geometry_sha256": geometry_hash(scene.points),
              "mode": "recorded_benchmark", "targets": targets, "background_id": 0,
              "prediction_domain": "tokens", "metric_domain": "raw_points",
              "target_label_policy": "all_non_targets_are_background",
              "prediction_target_map": {str(i + 1): str(obj) for i, obj in enumerate(object_ids)},
              "clicks": clicks, "frames": frames, "model": model,
              "click_budget": {"definition": "total accounted click events divided by requested object count; "
                                "includes background corrections and declared perfect-prediction virtual padding",
                               "maximum": 20, "counts_background": True, "denominator": count},
              "click_policy": "recorded AGILE3D deepest-error simulation with kdtree; model-adaptive",
              "source_files": source_files, "source_trace": trace,
              "source_implementation": SCANNET40_SOURCE_IMPLEMENTATION,
              "experiment_family": "public_vs_256_update_adaptation",
              "snapshot_coverage": "only exported click budgets; intermediate predictions unavailable"}
    return episode_from_arrays(record, arrays, scene)
