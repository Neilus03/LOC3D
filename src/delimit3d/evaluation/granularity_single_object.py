"""Full-scene, single-object click evaluation of the matched AGILE3D adapter.

Uses the released dataset, click simulator and evaluator. The quantizer is the
same explicit NumPy utility implementation used during matched MO training.
Each object has a paired seed and an atomic episode, so a restart cannot mix
checkpoints or omit hard objects from later-click averages.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from .backbone_batchnorm import evaluation_metadata


SO_DATASET = "scannet40"
# This exact predecessor wrote valid predictions, then failed before aggregation
# because it selected the nonexistent "scannet" label set. No other old version
# is eligible for this recovery, and no missing predictions may be generated.
AGGREGATION_RECOVERY_SOURCE_SHA256 = "bffebf468762bd6ac0538acc198625e3773d0bb7dac86176a1e71ac2f63b5817"


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_episode(record, index, scene, object_id):
    if (record["index"], record["scene"], record["object_id"]) != (index, scene, object_id):
        raise ValueError("Single-object episode identity mismatch")
    values = record["iou"]
    if len(values) != 21 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Single-object evaluation requires finite IoU at all 21 click counts")


def reconcile_prediction_identity(stored, current, source_path):
    """Preserve provenance of complete legacy predictions; never relabel them."""
    if stored == current:
        return stored, None
    original_source = stored.get("source_sha256", {}).get(str(source_path))
    candidate = copy.deepcopy(current)
    if str(source_path) in candidate.get("source_sha256", {}):
        candidate["source_sha256"][str(source_path)] = original_source
    if (original_source != AGGREGATION_RECOVERY_SOURCE_SHA256 or candidate != stored
            or not stored.get("full_benchmark") or stored.get("objects") != 10357
            or stored.get("max_clicks") != 20):
        raise ValueError("Cannot resume SO episodes with different checkpoint/data/code")
    return stored, {
        "reason": "Correct aggregation dataset selector from scannet to scannet40",
        "prediction_source_sha256": original_source,
        "aggregation_source_sha256": current["source_sha256"][str(source_path)],
        "original_prediction_identity_preserved": True,
    }


def aggregate_single_results(object_list, object_classes, result):
    evaluator = importlib.import_module("delimit3d.tasks.agile3d.protocol.evaluation.evaluator_SO").EvaluatorSO(
        SO_DATASET, str(object_list), str(object_classes), str(result), [.5, .65, .8, .85, .9])
    metrics = {key: float(value) for key, value in evaluator.eval_results().items()}
    if not metrics or any(not math.isfinite(v) for v in metrics.values()):
        raise ValueError("Missing or nonfinite final SO metric")
    return metrics


@torch.no_grad()
def evaluate_single(model, config, paths, official, base, device, *,
                    object_list, object_classes, output, identity, limit=None, voxel_size=.05):
    if float(voxel_size) not in (.02, .05):
        raise ValueError("SO evaluation requires an explicit supported voxel size")
    module = importlib.import_module("delimit3d.tasks.agile3d.protocol.datasets.InterSingleObj3DSegDataset")
    module.ME = SimpleNamespace(utils=SimpleNamespace(
        sparse_quantize=base._sparse_quantize, batched_coordinates=base._batched_coordinates))
    dataset = module.InterSingleObj3DSegDataset(
        str(paths["scan_folder"]), str(object_list), float(voxel_size), crop=False, transforms=False,
        use_normals=True, normal_root=str(paths["normal_root"]))
    if len(dataset) != 10357 or len(np.loadtxt(object_classes, dtype=str)) != 10357:
        raise ValueError("The full ScanNet40 SO list must contain 10357 objects")
    count = len(dataset) if limit is None else limit
    if not 0 < count <= len(dataset):
        raise ValueError("Invalid evaluation object count")
    state_before_evaluation = base.state_sha256(model.state_dict())
    identity = {**identity, **evaluation_metadata(model), "schema": "fixed9_agile3d_single_object/v1",
        "objects": count, "full_benchmark": limit is None, "max_clicks": 20,
        "seed_rule": "training_seed + 20000033 + object_list_index",
        "object_list_sha256": base.sha256_file(object_list),
        "object_classes_sha256": base.sha256_file(object_classes),
        "quantizer": base.QUANTIZATION_BACKEND,
        "source_sha256": {str(p): base.sha256_file(p) for p in (
            Path(__file__), Path(module.__file__),
            paths["agile3d_root"] / "utils/seg.py",
            paths["agile3d_root"] / "evaluation/evaluator_SO.py")}}
    if float(voxel_size) != .05:
        identity["voxel_size_m"] = float(voxel_size)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    recovery = None
    if identity_path.exists():
        identity, recovery = reconcile_prediction_identity(
            json.loads(identity_path.read_text()), identity, Path(__file__))
    else:
        write_json(identity_path, identity)
    episodes = output / "episodes"
    episodes.mkdir(exist_ok=True)
    pending = []
    for index in range(count):
        scene, object_id = map(str, dataset.dataset_list[index])
        path = episodes / f"{index:05d}.json"
        if path.exists():
            validate_episode(json.loads(path.read_text()), index, scene, object_id)
        else:
            pending.append(index)
    if recovery is not None:
        expected = {f"{index:05d}.json" for index in range(count)}
        if pending or {path.name for path in episodes.glob("*.json")} != expected:
            raise ValueError("Aggregation-only recovery requires all original SO episodes")
        recovery.update(objects_verified=count, rows_verified=count * 21,
                        prediction_identity_sha256=base.sha256_file(identity_path))
        print(json.dumps({"event": "single_object_aggregation_recovery", **recovery}), flush=True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, pending), batch_size=1, shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        collate_fn=module.collation_fn, pin_memory=True)
    model.eval()
    for index, batch in zip(pending, loader, strict=True):
        base.set_seed(int(config["seed"]) + 20_000_033 + index)
        unpacked = base.unpack_batch(batch, device)
        tokens, xyz = model.encode_batch(unpacked["grid_coord"], unpacked["raw_coords"],
            unpacked["features"], unpacked["point_offsets"])
        if len(tokens) != 1:
            raise ValueError("SO evaluation must use one full scene per object")
        tokens, xyz = tokens[0], xyz[0]
        labels, labels_full = unpacked["labels"][0], unpacked["labels_full"][0]
        inverse = torch.as_tensor(unpacked["inverse_maps"][0], device=device, dtype=torch.long)
        clicks = {"0": [], "1": []}
        times = copy.deepcopy(clicks)
        ious = []
        for click in range(21):
            prediction = (torch.zeros_like(labels) if click == 0 else
                          model.decode(tokens, xyz, clicks, times)["pred_masks"].argmax(dim=1).clone())
            for object_id, indices in clicks.items():
                if indices:
                    prediction[torch.as_tensor(indices, device=device, dtype=torch.long)] = int(object_id)
            iou, _ = official["mean_iou_scene"](prediction[inverse], labels_full)
            ious.append(float(iou))
            if click < 20:
                new, _, _, new_times = official["get_simulated_clicks"](
                    prediction, labels, xyz, click, training=False)
                if new is not None:
                    clicks, times = official["extend_clicks"](clicks, times, new, new_times)
        scene, object_id = map(str, dataset.dataset_list[index])
        record = {"index": index, "scene": scene, "object_id": object_id, "iou": ious}
        validate_episode(record, index, scene, object_id)
        write_json(episodes / f"{index:05d}.json", record)
        print(json.dumps({"event": "single_object_complete", "index": index,
                          "scene": scene, "object_id": object_id, "iou_at_5": ious[5]}), flush=True)
    result = output / "val_results_single.csv"
    original_result_sha256 = base.sha256_file(result) if recovery and result.exists() else None
    temporary = output / f"val_results_single.{os.getpid()}.tmp"
    with temporary.open("x") as stream:
        for index in range(count):
            scene, object_id = map(str, dataset.dataset_list[index])
            record = json.loads((episodes / f"{index:05d}.json").read_text())
            validate_episode(record, index, scene, object_id)
            for click, iou in enumerate(record["iou"]):
                stream.write(f"{index} {scene.replace('scene', '')} {object_id} {click} {iou:.10f}\n")
    if original_result_sha256 and base.sha256_file(temporary) != original_result_sha256:
        raise ValueError("Rebuilt SO result differs from the saved prediction CSV")
    os.replace(temporary, result)
    metrics = {}
    if limit is None:
        metrics = aggregate_single_results(object_list, object_classes, result)
    if base.state_sha256(model.state_dict()) != state_before_evaluation:
        raise RuntimeError("Single-object evaluation changed model weights or buffers")
    report = {"passed": True, "full_benchmark": limit is None, "objects": count,
              "state_unchanged": True, "optimizer_steps_performed": 0,
              "rows": count * 21, "result_file": str(result), "result_sha256": base.sha256_file(result),
              "metrics": metrics, "identity": identity,
              "aggregation_dataset": SO_DATASET,
              "aggregation_source_sha256": base.sha256_file(Path(__file__))}
    if recovery is not None:
        recovery.update(result_sha256=report["result_sha256"], prediction_rows_unchanged=True)
        write_json(output / "AGGREGATION_RECOVERY.json", recovery)
        report["aggregation_recovery"] = recovery
    write_json(output / "COMPLETE.json", report)
    return report
