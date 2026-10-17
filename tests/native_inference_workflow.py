r"""Opt-in CUDA validation using your own registered checkpoints and scene.

Run explicitly inside a GPU allocation with the documented native environment::

    python tests/native_inference_workflow.py --registry /models/registry.yaml \
        --scene /data/scene.npz --output /results/a-new-validation-directory

At least two available model registrations are needed to check model switching.
No checkpoints or datasets are downloaded. Clicks are synthetic interface checks,
not ground-truth prompts; this workflow makes no segmentation-accuracy claim.
The output directory must not already exist. Unavailable registrations are listed
as skipped, and only a completed run writes status="passed" in verification.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import weakref


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _require(condition, message):
    # Keep validation active even when Python is invoked with optimization.
    if not condition:
        raise AssertionError(message)


def _write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _rng_snapshot(torch, np):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state().clone(),
    }


def _check_rng(before, torch, np):
    after = _rng_snapshot(torch, np)
    _require(before["python"] == after["python"], "Inference changed Python RNG state")
    a, b = before["numpy"], after["numpy"]
    _require(a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:],
             "Inference changed NumPy RNG state")
    for key in ("torch_cpu", "torch_cuda"):
        _require(torch.equal(before[key], after[key]), f"Inference changed {key} RNG state")


def _normalization_buffers(model):
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return {name: value.detach().cpu().clone()
            for name, value in model.named_buffers() if name.endswith(suffixes)}


def _check_normalization(model, before, torch):
    after = _normalization_buffers(model)
    _require(before.keys() == after.keys(), "Normalization buffer names changed")
    for name, value in before.items():
        _require(torch.equal(value, after[name]), f"Normalization buffer changed: {name}")


def _load_export(response, service, preview, scene, task, np):
    """Read the actual downloadable artifact and check full/display alignment."""
    path = service.output / Path(response["download_url"]).name
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "provenance_json"}
        provenance = json.loads(str(archive["provenance_json"]))
    labels = arrays["labels"]
    point_count = len(scene.points)
    _require(labels.shape == (point_count,) and np.issubdtype(labels.dtype, np.integer),
             "Exported labels must cover every original input point")
    _require(np.all(labels >= 0), "Exported labels must be nonnegative")
    np.testing.assert_array_equal(np.asarray(response["labels"]), labels[preview["point_ids"]])
    _require(provenance["scene_sha256"] == scene.sha256, "Export refers to a different scene")
    _require(provenance["point_count"] == point_count and provenance["point_order"] == "original_input_rows",
             "Export is missing the original input-order contract")
    _require(provenance == response["provenance"], "Download and response provenance differ")
    scores, ids = arrays["scores"], arrays["instance_ids"]
    _require(scores.ndim == ids.ndim == 1 and len(scores) == len(ids),
             "Instance IDs and confidence scores do not align")
    _require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)),
             "Model returned invalid confidence scores")
    _require(np.issubdtype(ids.dtype, np.integer) and len(np.unique(ids)) == len(ids) and np.all(ids > 0),
             "Invalid instance IDs")
    _require(set(np.unique(labels)).issubset({0, *map(int, ids)}), "Labels refer to missing instance IDs")
    _require(response["instance_count"] == len(ids), "Response instance count differs from the export")
    np.testing.assert_array_equal(response["instance_ids"], ids)
    np.testing.assert_allclose(response["scores"], scores, rtol=0, atol=0)
    if task == "mask3d":
        packed = arrays["packed_masks"]
        _require(packed.dtype == np.uint8 and packed.shape == ((point_count + 7) // 8, len(ids)),
                 "Packed Mask3D masks do not retain every original point and proposal")
        _require(int(arrays["mask_point_count"]) == point_count, "Packed-mask point count differs")
        _require(provenance["packed_masks_bitorder"] == "big", "Unknown packed-mask bit ordering")
        _require(np.all(scores[:-1] >= scores[1:]), "Mask3D proposals are not confidence ordered")
        _require(np.all(scores >= provenance["score_threshold"]), "Mask3D retained a mask below the requested score")
        # Unpack one proposal at a time, retaining overlaps without allocating
        # an additional N-points x N-proposals array for a large input scene.
        expected = np.zeros(point_count, dtype=labels.dtype)
        for column, instance_id in enumerate(ids):
            mask = np.unpackbits(packed[:, column], count=point_count, bitorder="big").astype(bool)
            expected[mask & (expected == 0)] = instance_id
        np.testing.assert_array_equal(labels, expected,
                                      err_msg="Exclusive labels do not match the full overlapping proposals")
    return arrays, provenance, str(path.relative_to(service.output.parent))


def _interactive_clicks(scene, runner, np):
    from delimit3d.inference.backends import quantize_interactive

    _, _, representatives, inverse = quantize_interactive(scene.points, runner.grid_size)
    _require(len(representatives) >= 2, "Interactive validation needs at least two occupied voxels")
    _require(runner.model.decoder.max_click_events >= 4,
             "Interactive validation needs a decoder supporting at least four click events")
    first = len(scene.points) // 2
    # A distant occupied voxel makes the second object independent of the first
    # prompt even when adjacent input rows belong to the same quantized point.
    distances = np.sum((scene.points[representatives] - scene.points[first]) ** 2, axis=1)
    distances[inverse[representatives] == inverse[first]] = -1
    second = int(representatives[int(np.argmax(distances))])
    same_voxel = np.flatnonzero(inverse == inverse[first])
    alias = int(same_voxel[-1] if same_voxel[-1] != first else same_voxel[0])
    initial = [{"point_index": first, "object_id": 7}]
    two_objects = initial + [{"point_index": second, "object_id": 2}]
    conflicting = two_objects + [{"point_index": alias, "object_id": 2},
                                 {"point_index": first, "object_id": 0}]
    return two_objects, conflicting, inverse, alias != first


def _check_click_constraints(labels, clicks, inverse, np):
    latest = {}
    for click in clicks:
        latest[int(inverse[click["point_index"]])] = click["object_id"]
    for token, object_id in latest.items():
        _require(np.all(labels[inverse == token] == object_id),
                 f"Latest click was not honored for all original points in voxel {token}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, required=True, help="Your local model registry; at least two available entries.")
    parser.add_argument("--scene", type=Path, required=True, help="Your NPZ/PLY scene, including the selected models' required inputs.")
    parser.add_argument("--output", type=Path, required=True, help="A fresh directory for verification JSON and complete predictions.")
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from delimit3d.inference.backends import ModelRunner
    from delimit3d.inference.registry import load_registry
    from delimit3d.inference.scene import load_scene
    from delimit3d.inference.service import InferenceService

    if not torch.cuda.is_available():
        raise RuntimeError("This opt-in workflow requires a CUDA GPU and the documented native inference environment")
    entries = load_registry(args.registry)
    available = [entry for entry in entries if entry["available"]]
    if len(available) < 2:
        raise ValueError("Register at least two available trained models to validate real model switching")
    scene = load_scene(args.scene)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "verification.json"
    report = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "native inference mechanics; synthetic clicks; no benchmark accuracy measurement",
        "registry": str(args.registry.resolve()), "scene": str(args.scene.resolve()),
        "scene_sha256": scene.sha256, "point_count": len(scene.points),
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": sys.version.split()[0],
        "rng_scope": ["Python", "NumPy", "Torch CPU", "Torch current CUDA device"],
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("tests/native_inference_workflow.py", "src/delimit3d/inference/backends.py",
             "src/delimit3d/inference/service.py", "src/delimit3d/inference/scene.py")},
        "skipped_unavailable_models": [{"model_id": entry["id"], "reason": entry["reason"]}
                                       for entry in entries if not entry["available"]],
        "model_visits": [], "model_loads": [],
    }
    _write_report(report_path, report)
    previous_refs = {}
    normalization_before = {}

    def factory(entry, device):
        nonlocal previous_refs, normalization_before
        # Do not call gc.collect() here: this is specifically a witness that
        # InferenceService broke hook cycles before allocating the next model.
        alive = [name for name, reference in previous_refs.items() if reference() is not None]
        _require(not alive, f"Model switch retained previous GPU objects: {alive}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record = {"model_id": entry["id"], "previous_model_released_before_load": bool(previous_refs),
                  "cuda_allocated_before_load_bytes": torch.cuda.memory_allocated()}
        print(json.dumps({"event": "loading_model", **record}), flush=True)
        runner = ModelRunner(entry, device=device)
        normalization_before = _normalization_buffers(runner.model)
        _require(bool(normalization_before), "Expected LitePT normalization buffers to validate")
        previous_refs = {"runner": weakref.ref(runner), "model": weakref.ref(runner.model),
                         "backbone": weakref.ref(runner.model.backbone),
                         "parameter": weakref.ref(next(runner.model.parameters()))}
        report["model_loads"].append(record)
        return runner

    service = InferenceService(args.registry, output / "results", runner_factory=factory)
    preview = service.add_scene(scene)
    report["display_point_count"] = len(preview["points"])
    reference_labels = None

    def visit(entry, revisit=False):
        start = time.monotonic()
        task = entry["task"]
        clicks = [] if task == "mask3d" else [{"point_index": len(scene.points) // 2, "object_id": 7}]
        request = {"model_id": entry["id"], "scene_id": preview["scene_id"],
                   "clicks": clicks, "score_threshold": .3}
        artifacts = {}
        click_fixture = None

        def predict(stage, current):
            print(json.dumps({"event": "predict", "model_id": entry["id"], "stage": stage}), flush=True)
            report["active_check"] = {"model_id": entry["id"], "stage": stage}
            _write_report(report_path, report)
            rng = _rng_snapshot(torch, np)
            response = service.predict(current)
            _check_rng(rng, torch, np)
            _check_normalization(service.runner.model, normalization_before, torch)
            arrays, provenance, artifact = _load_export(response, service, preview, scene, task, np)
            artifacts[stage] = artifact
            return arrays, provenance

        first, provenance = predict("initial", request)
        runner = service.runner
        cache = runner._automatic if task == "mask3d" else runner._encoded
        _require(cache is not None, "Inference did not retain its scene cache")
        repeated, _ = predict("same_scene_cached", request)
        _require((runner._automatic if task == "mask3d" else runner._encoded) is cache,
                 "An unchanged scene was re-encoded")
        np.testing.assert_array_equal(first["labels"], repeated["labels"])
        np.testing.assert_allclose(first["scores"], repeated["scores"], rtol=1e-5, atol=1e-6)
        del cache
        runner._scene_key = None
        reencoded, _ = predict("forced_reencode", request)
        np.testing.assert_array_equal(first["labels"], reencoded["labels"],
                                      err_msg="Repeated scene encoding changed labels")
        checks = {"full_original_point_export": True, "preview_matches_original_indices": True,
                  "normalization_buffers_unchanged": True, "rng_states_unchanged": True,
                  "same_scene_cache_reused": True, "reencoded_labels_identical": True}
        if task == "mask3d":
            cache = runner._automatic
            predict("changed_score_threshold", {**request, "score_threshold": .6})
            _require(runner._automatic is cache, "Changing only the score threshold recomputed Mask3D proposals")
            del cache
            checks.update(packed_mask_shape_and_assignment=True, score_threshold_reuses_proposals=True)
        else:
            two_objects, conflicting, inverse, distinct_alias = _interactive_clicks(scene, runner, np)
            cache = runner._encoded
            multiple, _ = predict("two_sparse_object_ids", {**request, "clicks": two_objects})
            _require(runner._encoded is cache, "Adding an object re-encoded the interactive scene")
            np.testing.assert_array_equal(multiple["instance_ids"], [2, 7])
            _check_click_constraints(multiple["labels"], two_objects, inverse, np)
            conflict, _ = predict("latest_click_wins", {**request, "clicks": conflicting})
            _require(runner._encoded is cache, "Refining prompts re-encoded the interactive scene")
            _check_click_constraints(conflict["labels"], conflicting, inverse, np)
            del cache
            checks.update(sparse_object_ids_preserved=True, latest_click_wins_at_voxel_resolution=True,
                          prompt_edits_reuse_encoded_scene=True)
            click_fixture = {"final_clicks": conflicting, "same_voxel_conflict_uses_distinct_input_rows": distinct_alias}
        torch.cuda.synchronize()
        record = {"model_id": entry["id"], "task": task, "revisit_after_switching": revisit,
                  "elapsed_seconds": round(time.monotonic() - start, 3), "checks": checks,
                  "normalization_buffer_count": len(normalization_before),
                  "initial_instance_count": len(first["instance_ids"]),
                  "initial_assigned_point_count": int(np.count_nonzero(first["labels"])),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "provenance": provenance, "artifacts": artifacts}
        if click_fixture is not None:
            record["synthetic_click_fixture"] = click_fixture
        report["model_visits"].append(record)
        print(json.dumps({"event": "model_verified", "model_id": entry["id"], "checks": checks}), flush=True)
        _write_report(report_path, report)
        # Only CPU arrays leave this scope; no strong reference to the runner,
        # its CUDA tensors or its cache survives the next service model switch.
        return first["labels"]

    try:
        for index, entry in enumerate(available):
            labels = visit(entry)
            if index == 0:
                reference_labels = labels
        revisited = visit(available[0], revisit=True)
        np.testing.assert_array_equal(reference_labels, revisited,
                                      err_msg="Reloading the first model changed its initial labels")
        _require(len(report["model_loads"]) == len(available) + 1,
                 "The service did not load each model and then reload the first")
        report.pop("active_check", None)
        report["status"] = "passed"
        report["first_model_reload_labels_identical"] = True
        report["validated_model_count"] = len(available)
        report["validated_switch_count"] = len(report["model_loads"]) - 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)
        # Final teardown is separate from the before-load GC checks above.
        service.runner = None
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": "passed", "verification": str(report_path),
                      "models": len(available), "model_switches": len(available)}), flush=True)


if __name__ == "__main__":
    main()
