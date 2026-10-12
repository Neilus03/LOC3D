#!/usr/bin/env python3
"""Frozen native-feature instance diagnostics, with sealed inputs and outputs.

Each worker is restricted to one CUDA device. Clustering sees normalized native
tokens and a scalar GT-K; query points and GT labels enter only retrieval and
post-fit scoring. The primary retrieval result has no rejection threshold.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import ctypes
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

SCENES = [
    "scene0651_02", "scene0378_00", "scene0578_01", "scene0207_02",
    "scene0616_00", "scene0256_01", "scene0670_00", "scene0633_01",
    "scene0231_02", "scene0488_01",
]
ARMS = ("sonata", "concerto_small", "litept_public", "litept_delimit3d")
THRESHOLDS = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SEEDS = [20261010 + i for i in range(5)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_manifest(root):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = (root / "manifest.sha256").read_text().strip()
    if sha(path) != expected or [row["scene"] for row in manifest["scenes"]] != SCENES:
        raise ValueError("Sealed evaluation manifest changed")
    if list(manifest["arms"]) != sorted(ARMS):
        raise ValueError("Unexpected model arms")
    return manifest, expected


def read_ply(path, mesh=False):
    from plyfile import PlyData
    options = {"known_list_len": {"face": {"vertex_indices": 3}}} if mesh else {}
    ply = PlyData.read(str(path), **options)
    vertices = ply["vertex"].data
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    if mesh:
        faces = np.asarray(ply["face"].data["vertex_indices"].tolist(), dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("Invalid mesh topology")
        return points, faces
    names = vertices.dtype.names
    name = next((key for key in ("label", "instance", "instance_id") if key in names), None)
    if name is None:
        raise ValueError("Source has no instance labels")
    return points, np.asarray(vertices[name], dtype=np.int64)


def prepare(root):
    draft = json.loads((root / "draft.json").read_text())
    if (root / "manifest.json").exists():
        load_manifest(root)
        print("Inputs already sealed", flush=True)
        return
    split_path = Path(draft["validation_split"])
    validation = {line.strip() for line in split_path.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")}
    if len(validation) != 312 or not set(SCENES) <= validation:
        raise ValueError("The selected scenes are not in the 312-scene ScanNet validation split")
    prepared = []
    for scene in SCENES:
        destination = root / "inputs" / scene
        destination.mkdir(parents=True, exist_ok=True)
        paths = draft["scene_sources"][scene]
        for item in paths.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Input source changed")
        xyz = np.load(paths["coord"]["path"]).astype(np.float32)
        rgb = np.load(paths["color"]["path"]).astype(np.float32)
        normal = np.load(paths["normal"]["path"]).astype(np.float32)
        source_xyz, instance = read_ply(paths["instance_ply"]["path"])
        mesh_xyz, faces = read_ply(paths["mesh"]["path"], mesh=True)
        if (xyz.shape != rgb.shape or xyz.shape != normal.shape or
                not np.allclose(xyz, source_xyz, atol=1e-5, rtol=0) or
                not np.allclose(xyz, mesh_xyz, atol=1e-5, rtol=0)):
            raise ValueError(f"{scene}: original point identity mismatch")
        if not all(np.isfinite(value).all() for value in (xyz, rgb, normal)):
            raise ValueError("Nonfinite geometry")
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("Unknown RGB range")
        if rgb.max() <= 1.5:
            rgb *= 255.0
        lengths = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(lengths, 1e-8)
        gt_ids = np.unique(instance[instance > 0])
        labels = np.full(len(xyz), -1, dtype=np.int32)
        queries, objects = [], []
        for index, gt_id in enumerate(gt_ids):
            ids = np.flatnonzero(instance == gt_id)
            center = xyz[ids].mean(axis=0)
            query = int(ids[np.argmin(np.square(xyz[ids] - center).sum(axis=1))])
            labels[ids] = index
            queries.append(query)
            objects.append({"index": index, "gt_instance_id": int(gt_id),
                            "vertices": len(ids), "query_vertex": query})
        if not objects:
            raise ValueError("Empty GT foreground")
        shift = np.array([(xyz[:, 0].min() + xyz[:, 0].max()) / 2,
                          (xyz[:, 1].min() + xyz[:, 1].max()) / 2, xyz[:, 2].min()], np.float32)
        packed = destination / "scene.npz"
        atomic_npz(packed, xyz=xyz, rgb255=rgb, normal=normal, labels=labels,
                   queries=np.asarray(queries, np.int64), faces=faces, shift=shift)
        npy_header_identity = hashlib.sha256(xyz.tobytes()).hexdigest()
        prepared.append({"scene": scene, "input": str(packed), "sha256": sha(packed),
                         "original_points": len(xyz), "faces": len(faces),
                         "coordinate_array_sha256": npy_header_identity,
                         "objects": objects, "background_points": int((labels < 0).sum()),
                         "source_files": paths})
        print(f"PREPARED {scene}: {len(xyz)} vertices, {len(objects)} objects", flush=True)
    manifest = {"schema": "delimit3d.feature_segmentation_scannet10/v1", "scenes": prepared,
                "arms": draft["arms"], "source": draft["source"],
                "validation_split": {"path": str(split_path), "sha256": sha(split_path), "scenes": 312},
                "protocol": {"feature_normalization": "L2 of entire native descriptor",
                    "cosine": "one shared centroid-nearest original GT vertex per instance; highest similarity wins",
                    "cosine_primary_threshold": -1.0, "cosine_threshold_sweep": THRESHOLDS,
                    "threshold_sweep_usage": "diagnostic only; no threshold selected using evaluation GT",
                    "clustering": "query-free spherical k-means on every native token including background",
                    "k_source": "GT foreground instance count oracle",
                    "clustering_seeds": SEEDS, "max_iterations": 100, "center_cosine_tolerance": 1e-6,
                    "restart_selection": "largest mean cosine to assigned center, never GT metrics",
                    "matching": "Hungarian maximum full-mask IoU after clustering; GT matching used only for scoring and colors",
                    "background": "included in clustering and assignments; mask leakage counts as false positives",
                    "metric_points": "all original scene vertices; no GT foreground-only filtering",
                    "normalization_state": "model.eval and frozen parameters; state tensors unchanged",
                    "claim_boundary": "seeded and GT-K feature diagnostics; not automatic instance AP or a matched training ablation"}}
    atomic_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(sha(root / "manifest.json") + "\n")
    print("INPUTS_SEALED", flush=True)


def confusion(prediction, truth, k):
    prediction = np.asarray(prediction, np.int64)
    truth = np.asarray(truth, np.int64)
    if prediction.shape != truth.shape or prediction.min() < -1 or prediction.max() >= k:
        raise ValueError("Prediction does not cover the common original points")
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


def adjusted_rand(left, right):
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    columns = int(b.max()) + 1
    joint = np.bincount(a * columns + b).astype(np.float64)
    choose2 = lambda value: (value * (value - 1) / 2).sum()
    rows = np.bincount(a).astype(np.float64)
    cols = np.bincount(b).astype(np.float64)
    pairs = len(a) * (len(a) - 1) / 2
    expected = choose2(rows) * choose2(cols) / max(pairs, 1)
    maximum = (choose2(rows) + choose2(cols)) / 2
    return float((choose2(joint) - expected) / (maximum - expected)) if maximum > expected else 1.0


def spherical_kmeans(features, k, seed, max_iter=100, tol=1e-6):
    """GPU fitting, with no labels, spatial coordinates, or queries accepted."""
    import torch
    if features.ndim != 2 or len(features) < k or k < 1:
        raise ValueError("Invalid cluster count")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    n, d = features.shape
    first = int(torch.randint(n, (1,), generator=generator, device=features.device).item())
    centers = [features[first].clone()]
    selected = {first}
    minimum = (1 - features @ centers[0]).clamp_min(0)
    for _ in range(1, k):
        weights = minimum.square()
        if float(weights.sum().item()) <= 1e-12:
            index = next(i for i in range(n) if i not in selected)
        else:
            index = int(torch.multinomial(weights, 1, generator=generator).item())
        selected.add(index)
        centers.append(features[index].clone())
        minimum = torch.minimum(minimum, (1 - features @ centers[-1]).clamp_min(0))
    centers = torch.stack(centers)
    previous = None
    reinitializations = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        scores = features @ centers.T
        best, labels = scores.max(dim=1)
        updated = torch.zeros((k, d), device=features.device, dtype=features.dtype)
        updated.index_add_(0, labels, features)
        norms = updated.norm(dim=1)
        empty = torch.nonzero(norms <= 1e-8).flatten()
        if len(empty):
            farthest = torch.argsort(best, stable=True)
            for position, cluster in enumerate(empty.tolist()):
                updated[cluster] = features[farthest[position]]
                reinitializations += 1
            norms = updated.norm(dim=1)
        updated /= norms[:, None]
        movement = float((1 - (centers * updated).sum(dim=1)).clamp_min(0).max().item())
        unchanged = previous is not None and torch.equal(labels, previous)
        centers = updated
        if unchanged or movement <= tol:
            converged = True
            break
        previous = labels
    best, labels = (features @ centers.T).max(dim=1)
    record = {"seed": seed, "iterations": iteration, "converged": converged,
              "mean_cosine_to_center": float(best.double().mean().item()),
              "mean_cosine_distance": float((1 - best.double()).mean().item()),
              "empty_cluster_reinitializations": reinitializations,
              "cluster_token_counts": torch.bincount(labels, minlength=k).cpu().tolist(),
              "fit_uses_queries": False, "fit_uses_labels": False,
              "fit_scope": "all native tokens, including background",
              "update_precision": "float32 CUDA sums", "k": k}
    return labels.cpu().numpy().astype(np.int32), centers.cpu().numpy(), record


def tensor_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_model(root, arm, config):
    import torch
    if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match manifest")
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    if arm.startswith("litept"):
        from student.models.litept_wrapper import LitePTBackbone
        model = LitePTBackbone(litept_root=os.environ["DELIMIT3D_LITEPT_ROOT"], in_channels=6,
            grid_size=config["grid_size_m"], litept_variant="litept_s_star",
            voxel_reduce="representative", representative_sampling="first",
            cache_training_voxelization=False)
        if arm == "litept_public":
            state = {"model." + key.removeprefix("module.").removeprefix("backbone."): value
                     for key, value in payload["state_dict"].items()
                     if key.removeprefix("module.").startswith("backbone.")}
        else:
            if int(payload["epoch"]) != config["epoch"]:
                raise ValueError("Pretraining epoch mismatch")
            state = {key.removeprefix("backbone."): value
                     for key, value in payload["model_state_dict"].items() if key.startswith("backbone.")}
        report = model.load_state_dict(state, strict=True)
        module = None
        architecture = model.resolved_architecture if hasattr(model, "resolved_architecture") else {}
    else:
        module = importlib.import_module("sonata" if arm == "sonata" else "concerto")
        architecture = payload["config"]
        if (architecture["in_channels"] != 9 or not architecture["enc_mode"] or
                list(architecture["enc_channels"]) != config["stage_channels"]):
            raise ValueError("PTv3 checkpoint feature contract mismatch")
        model = module.model.PointTransformerV3(**architecture)
        report = model.load_state_dict(payload["state_dict"], strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise ValueError("Strict state load failed")
    del payload
    model.eval().requires_grad_(False).cuda()
    if any(child.training for child in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Model is not fully frozen in evaluation mode")
    return model, module, {"architecture": architecture, "loaded_state_keys": len(state) if module is None else len(model.state_dict()),
                           "parameters": sum(p.numel() for p in model.parameters()), "strict_load": True}


def extract_features(model, module, data, config, scene):
    import torch
    seed = int(hashlib.sha256(("feature-segmentation-input:" + scene).encode()).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    xyz = data["xyz"].astype(np.float32)
    if module is None:
        centered = xyz - data["shift"]
        inputs = np.concatenate((data["rgb255"] / 255, data["normal"]), axis=1).astype(np.float32)
        output = model(torch.from_numpy(centered).cuda(), torch.from_numpy(inputs).cuda())
        features = output.scene_tokens.float()
        inverse = output.inverse_map.long().cpu().numpy()
        token_ids, representatives = np.unique(inverse, return_index=True)
        if not np.array_equal(token_ids, np.arange(len(features))):
            raise ValueError("Native tokens do not all have original descendants")
        coordinates = output.scene_xyz.float().cpu().numpy()
        grids = np.floor(coordinates / config["grid_size_m"]).astype(np.int64)
        grids -= grids.min(axis=0)
        if not np.allclose(centered[representatives], coordinates, atol=1e-5, rtol=0):
            raise ValueError("LitePT representative point identity failed")
        hierarchy = {"native_stage_shapes": [[len(features), features.shape[1]]], "parent_maps": []}
    else:
        transform = module.transform.Compose([
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=config["grid_size_m"], hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"), dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse", "original_index"),
                 feat_keys=("coord", "color", "normal")),
        ])
        point = transform({"coord": xyz.copy(), "color": data["rgb255"].copy(),
                           "normal": data["normal"].copy(), "original_index": np.arange(len(xyz), dtype=np.int64),
                           "index_valid_keys": ["coord", "color", "normal", "original_index"]})
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                point[key] = value.cuda(non_blocking=True)
        point = model(point)
        native_shapes, parent_maps = [], []
        native_shapes.append(list(point.feat.shape))
        while "pooling_parent" in point:
            if "pooling_inverse" not in point:
                raise ValueError("Missing exact pooling inverse")
            parent, inverse_parent = point.pop("pooling_parent"), point.pop("pooling_inverse")
            if (inverse_parent.shape != (len(parent.feat),) or inverse_parent.min() < 0 or
                    inverse_parent.max() >= len(point.feat)):
                raise ValueError("Invalid native hierarchy mapping")
            native_shapes.append(list(parent.feat.shape))
            parent_maps.append(inverse_parent.long().cpu().numpy())
            parent.feat = torch.cat((parent.feat, point.feat[inverse_parent]), dim=1)
            point = parent
        native_shapes.reverse()
        parent_maps.reverse()
        if len(native_shapes) != 5 or [value[1] for value in native_shapes] != config["stage_channels"]:
            raise ValueError("All five native feature stages were not retained")
        features = point.feat.float()
        inverse = point.inverse.long().cpu().numpy()
        representatives = point.original_index.long().cpu().numpy()
        coordinates = point.coord.float().cpu().numpy()
        grids = point.grid_coord.long().cpu().numpy()
        if not np.allclose((xyz - data["shift"])[representatives], coordinates, atol=1e-5, rtol=0):
            raise ValueError("PTv3 representative point identity failed")
        hierarchy = {"native_stage_shapes": native_shapes, "parent_maps": parent_maps}
    if (features.shape[1] != config["feature_dim"] or not torch.isfinite(features).all() or
            inverse.shape != (len(xyz),) or inverse.min() < 0 or inverse.max() >= len(features) or
            len(representatives) != len(features) or
            not np.array_equal(inverse[representatives], np.arange(len(features)))):
        raise ValueError("Feature width, finite state, or original-point mapping contract failed")
    norms = features.norm(dim=1)
    if not torch.isfinite(norms).all() or norms.min() <= 1e-8:
        raise ValueError("Undefined cosine feature row")
    metadata = {"input_seed": seed, "tokens": len(features), "original_points": len(xyz),
                "feature_dim": features.shape[1], "native_stage_shapes": hierarchy["native_stage_shapes"],
                "raw_feature_norm_min": float(norms.min().item()),
                "raw_feature_norm_mean": float(norms.mean().item()),
                "raw_feature_norm_max": float(norms.max().item()),
                "query_token_collisions": len(data["queries"]) - len(np.unique(inverse[data["queries"]]))}
    return features, inverse, representatives, coordinates, grids, hierarchy, metadata


def worker(root, arm):
    import torch
    manifest, manifest_sha = load_manifest(root)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each model must see exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if not any(name in gpu for name in ("4090", "3090")):
        raise RuntimeError(f"Unexpected GPU: {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    self_test()
    with torch.inference_mode():
        sample = torch.tensor([[1., 0., 0.], [.99, .05, 0.], [0., 1., 0.], [.05, .99, 0.]], device="cuda")
        sample /= sample.norm(dim=1, keepdim=True)
        labels, centers, _ = spherical_kmeans(sample, 2, SEEDS[0])
        assert adjusted_rand(labels, [0, 0, 1, 1]) == 1
        assert np.allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-6)
    print("GPU_CLUSTER_CONTRACT_PASSED", flush=True)
    directory = root / "arms" / arm
    directory.mkdir(parents=True, exist_ok=True)
    model, module, load_record = load_model(root, arm, manifest["arms"][arm])
    state_before = tensor_state_hash(model)
    runtime = {"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
               "gpu": gpu, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "manifest_sha256": manifest_sha, "state_before": state_before, **load_record}
    atomic_json(directory / "runtime.json", runtime)
    scenes = {}
    for scene_record in manifest["scenes"]:
        scene = scene_record["scene"]
        output = directory / scene
        output.mkdir(parents=True, exist_ok=True)
        if (output / "COMPLETE.json").exists():
            complete = json.loads((output / "COMPLETE.json").read_text())
            if complete["manifest_sha256"] != manifest_sha:
                raise ValueError("Cannot resume outputs from another input manifest")
            for filename, digest in complete["artifacts"].items():
                if sha(output / filename) != digest:
                    raise ValueError("Completed scene artifact changed")
            scenes[scene] = json.loads((output / "metrics.json").read_text())
            continue
        if sha(scene_record["input"]) != scene_record["sha256"]:
            raise ValueError("Prepared scene changed")
        start = time.monotonic()
        with np.load(scene_record["input"], allow_pickle=False) as packed:
            data = {key: packed[key] for key in packed.files if key != "faces"}
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            features, inverse, representatives, coords, grids, hierarchy, metadata = extract_features(
                model, module, data, manifest["arms"][arm], scene)
            cpu_features = features.cpu().numpy()
            temporary = output / f"features.npy.tmp.{os.getpid()}"
            with temporary.open("wb") as stream:
                np.save(stream, cpu_features, allow_pickle=False)
            os.replace(temporary, output / "features.npy")
            atomic_npz(output / "point_maps.npz", inverse=inverse, representatives=representatives,
                       scene_xyz=coords, grid_coord=grids,
                       **{f"parent_map_{i}": value for i, value in enumerate(hierarchy["parent_maps"])})
            normalized = features / features.norm(dim=1, keepdim=True)
            del features, cpu_features
            query_tokens = inverse[data["queries"]]
            scores = (normalized @ normalized[torch.as_tensor(query_tokens, device="cuda")].T).clamp(-1, 1)
            best, winner = scores.max(dim=1)
            best = best.cpu().numpy()
            winner = winner.cpu().numpy().astype(np.int32)
            atomic_npz(output / "cosine.npz", query_vertices=data["queries"], query_tokens=query_tokens,
                       token_scores=scores.cpu().numpy(), token_winner=winner, token_best=best,
                       original_point_winner=winner[inverse])
            del scores
            cosine_rows = []
            for tau in THRESHOLDS:
                prediction = np.where(best[inverse] >= tau, winner[inverse], -1)
                cosine_rows.append({"threshold": tau, **metrics(prediction, data["labels"], scene_record["objects"])})
            restarts, token_labels, centers, aligned_labels, mappings, matching_ious = [], [], [], [], [], []
            for seed in SEEDS:
                labels, center, record = spherical_kmeans(normalized, len(scene_record["objects"]), seed)
                aligned, mapping, matching_iou = match_clusters(labels[inverse], data["labels"], scene_record["objects"])
                record["metrics_after_fit"] = metrics(aligned, data["labels"], scene_record["objects"])
                restarts.append(record)
                token_labels.append(labels)
                centers.append(center)
                aligned_labels.append(aligned)
                mappings.append(mapping)
                matching_ious.append(matching_iou)
                print(f"CLUSTER {arm} {scene} seed={seed} iter={record['iterations']} objective={record['mean_cosine_to_center']:.6f}", flush=True)
            selected = max(range(len(restarts)), key=lambda i: restarts[i]["mean_cosine_to_center"])
            stability = [[adjusted_rand(a, b) for b in token_labels] for a in token_labels]
            atomic_npz(output / "kmeans.npz", token_labels=np.stack(token_labels), centers=np.stack(centers),
                       cluster_to_gt=np.stack(mappings), matching_iou=np.stack(matching_ious),
                       selected_restart=np.asarray(selected), selected_original_labels=aligned_labels[selected],
                       selected_original_clusters=token_labels[selected][inverse])
            atomic_npz(output / "predictions.npz", cosine=winner[inverse], kmeans=aligned_labels[selected])
            record = {"scene": scene, "arm": arm, "manifest_sha256": manifest_sha,
                      "features": metadata, "cosine_primary": cosine_rows[0], "cosine_threshold_sweep": cosine_rows,
                      "kmeans_primary": restarts[selected]["metrics_after_fit"],
                      "kmeans_restarts": restarts, "selected_restart": selected,
                      "restart_selection_uses_gt": False, "kmeans_token_stability_ari": stability,
                      "elapsed_seconds": time.monotonic() - start,
                      "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated()}
            atomic_json(output / "metrics.json", record)
            if tensor_state_hash(model) != state_before:
                raise ValueError("Encoder or normalization state changed during inference")
            complete = {"manifest_sha256": manifest_sha, "arm": arm, "scene": scene,
                        "state_unchanged": True, "artifacts": {
                            name: sha(output / name) for name in ("features.npy", "point_maps.npz", "cosine.npz",
                                                                  "kmeans.npz", "predictions.npz", "metrics.json")}}
            atomic_json(output / "COMPLETE.json", complete)
            scenes[scene] = record
            print(f"SCENE_COMPLETE {arm} {scene}: cosine={record['cosine_primary']['macro_iou']:.4f} kmeans={record['kmeans_primary']['macro_iou']:.4f}", flush=True)
            del normalized, data, restarts, token_labels, centers, aligned_labels, hierarchy
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(directory / "summary.json", {"arm": arm, "manifest_sha256": manifest_sha, "scenes": scenes, "runtime": runtime})
    atomic_json(directory / "COMPLETE.json", {"arm": arm, "manifest_sha256": manifest_sha,
                "scenes": SCENES, "state_before": state_before, "state_after": tensor_state_hash(model),
                "summary_sha256": sha(directory / "summary.json")})
    print(f"ARM_COMPLETE {arm}", flush=True)


def raster_library(root):
    source = root / "source" / "raster.c"
    library = root / "visuals" / "raster.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(source), "-o", str(library), "-lm"], check=True)
    lib = ctypes.CDLL(str(library))
    lib.raster.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def visible_faces(lib, xyz, faces, size=850):
    az, el = np.radians(-60), np.radians(65)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    right = np.array([-np.sin(az), np.cos(az), 0], np.float32)
    up = np.cross(view, right)
    projection = xyz @ np.stack([right, up, view], axis=1)
    low, high = projection[:, :2].min(0), projection[:, :2].max(0)
    scale = (size - 65) / max(float((high - low).max()), 1e-6)
    projection[:, :2] = (projection[:, :2] - (low + high) / 2) * scale + size / 2
    projection[:, 1] = size - projection[:, 1]
    projection = np.ascontiguousarray(projection, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    ids = np.full((size, size), -1, np.int64)
    depth = np.full((size, size), -np.inf, np.float32)
    lib.raster(projection.ctypes.data, faces.ctypes.data, len(faces), size, size, ids.ctypes.data, depth.ctypes.data)
    return ids


def label_canvas(values, faces, visible, colors):
    face_values = values[faces]
    face_label = np.where(face_values[:, 1] == face_values[:, 2], face_values[:, 1], face_values[:, 0])
    face_colors = np.full((len(faces), 3), 0.80, np.float32)
    valid = face_label >= 0
    face_colors[valid] = colors[face_label[valid]]
    canvas = np.full((*visible.shape, 3), 0.975, np.float32)
    drawn = visible >= 0
    canvas[drawn] = face_colors[visible[drawn]]
    return canvas


def finalize(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    manifest, manifest_sha = load_manifest(root)
    summaries = {}
    for arm in ARMS:
        folder = root / "arms" / arm
        complete = json.loads((folder / "COMPLETE.json").read_text())
        if complete["manifest_sha256"] != manifest_sha or complete["scenes"] != SCENES or complete["state_before"] != complete["state_after"]:
            raise ValueError("Model completion contract failed")
        if sha(folder / "summary.json") != complete["summary_sha256"]:
            raise ValueError("Model summary hash mismatch")
        summaries[arm] = json.loads((folder / "summary.json").read_text())
    visuals = root / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    lib = raster_library(root)
    aggregate, per_scene_rows, per_object_rows = {}, [], []
    for arm in ARMS:
        aggregate[arm] = {}
        for method in ("cosine", "kmeans"):
            scene_metrics = [summaries[arm]["scenes"][scene][method + "_primary"] for scene in SCENES]
            objects = [obj for row in scene_metrics for obj in row["per_object"]]
            iou = np.array([obj["iou"] for obj in objects])
            aggregate[arm][method] = {"scene_mean_macro_iou": float(np.mean([row["macro_iou"] for row in scene_metrics])),
                "pooled_object_mean_iou": float(iou.mean()), "pooled_objects_iou25": float((iou >= 0.25).mean()),
                "pooled_objects_iou50": float((iou >= 0.5).mean()), "pooled_objects_iou75": float((iou >= 0.75).mean()),
                "scene_mean_background_rejection": float(np.mean([row["background_rejection"] for row in scene_metrics])),
                "scene_mean_foreign_mask_rate": float(np.mean([row["masks_covering_half_another_object"] for row in scene_metrics])),
                "scene_mean_micro_iou": float(np.mean([row["micro_iou"] for row in scene_metrics])),
                "scene_mean_macro_precision": float(np.mean([row["macro_precision"] for row in scene_metrics])),
                "scene_mean_macro_recall": float(np.mean([row["macro_recall"] for row in scene_metrics])),
                "scenes": len(SCENES), "objects": len(objects)}
            for scene, row in zip(SCENES, scene_metrics):
                per_scene_rows.append({"arm": arm, "method": method, "scene": scene,
                                       **{key: value for key, value in row.items() if key != "per_object"}})
                per_object_rows.extend({"arm": arm, "method": method, "scene": scene, **obj} for obj in row["per_object"])
    with PdfPages(visuals / "scannet10_gallery.pdf") as gallery:
        for scene_record in manifest["scenes"]:
            scene = scene_record["scene"]
            with np.load(scene_record["input"], allow_pickle=False) as packed:
                xyz, faces, truth, rgb = packed["xyz"], packed["faces"], packed["labels"], packed["rgb255"] / 255
            k = len(scene_record["objects"])
            colors = np.array([colorsys.hsv_to_rgb((i * 0.61803398875) % 1, 0.58 + 0.15 * (i % 3) / 2, 0.80 + 0.15 * (i % 2)) for i in range(k)], np.float32)
            drawn_faces = faces[xyz[:, 2][faces].mean(axis=1) < xyz[:, 2].max() - 0.10]
            visible = visible_faces(lib, xyz, drawn_faces)
            fig, axes = plt.subplots(5, 2, figsize=(12, 23), facecolor="#F7F8FA")
            rgb_faces = rgb[drawn_faces].mean(axis=1)
            rgb_canvas = np.full((*visible.shape, 3), 0.975, np.float32)
            mask = visible >= 0
            rgb_canvas[mask] = rgb_faces[visible[mask]]
            axes[0, 0].imshow(rgb_canvas)
            axes[0, 0].set_title("RGB", fontweight="bold")
            axes[0, 1].imshow(label_canvas(truth, drawn_faces, visible, colors))
            axes[0, 1].set_title(f"GT instances · {k} foreground objects", fontweight="bold")
            for row, arm in enumerate(ARMS, 1):
                output = root / "arms" / arm / scene
                complete = json.loads((output / "COMPLETE.json").read_text())
                for filename in ("metrics.json", "predictions.npz"):
                    if sha(output / filename) != complete["artifacts"][filename]:
                        raise ValueError("Visual source artifact changed")
                with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
                    for column, method in enumerate(("cosine", "kmeans")):
                        metric = summaries[arm]["scenes"][scene][method + "_primary"]
                        axes[row, column].imshow(label_canvas(predictions[method], drawn_faces, visible, colors))
                        label = manifest["arms"][arm]["label"]
                        recipe = "competing cosine · no rejection" if method == "cosine" else "spherical k-means · GT-K · no queries"
                        axes[row, column].set_title(f"{label}\n{recipe}\nmean IoU {metric['macro_iou']:.1%} · objects IoU ≥ .5 {metric['objects_iou50']:.1%}", fontsize=10, fontweight="bold")
            for ax in axes.flat:
                ax.set_axis_off()
            fig.suptitle(f"ScanNet validation · {scene}", fontsize=20, fontweight="bold", y=0.995)
            fig.text(0.5, 0.978, "Identical original vertices and cosine queries; model-native features and grids", ha="center", fontsize=10, color="#475569")
            fig.text(0.5, 0.007, "Feature diagnostic: seeded cosine and query-free GT-K clustering. Full-scene scoring; background leakage penalized.\nClustering restarts selected by objective; GT matching afterward. Same mesh, camera and instance colors in every panel.", ha="center", fontsize=8, color="#475569")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.957, bottom=0.028, wspace=0.025, hspace=0.13)
            fig.savefig(visuals / f"{scene}_comparison.png", dpi=145, facecolor=fig.get_facecolor())
            fig.savefig(visuals / f"{scene}_comparison.pdf", facecolor=fig.get_facecolor())
            gallery.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"VISUAL_COMPLETE {scene}", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="#F7F8FA")
    positions = np.arange(len(ARMS))
    labels = [manifest["arms"][arm]["label"].replace(" · ", "\n") for arm in ARMS]
    for ax, key, title in zip(axes, ("scene_mean_macro_iou", "pooled_object_mean_iou", "pooled_objects_iou50"),
                              ("Mean scene object IoU", "Mean IoU across all objects", "Objects with IoU ≥ 0.5")):
        ax.bar(positions - 0.18, [aggregate[arm]["cosine"][key] for arm in ARMS], 0.36, label="Competing cosine", color="#2677C7")
        ax.bar(positions + 0.18, [aggregate[arm]["kmeans"][key] for arm in ARMS], 0.36, label="Spherical k-means · GT-K", color="#229B79")
        ax.set_xticks(positions, labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen feature segmentation · 10 ScanNet validation scenes", fontweight="bold", fontsize=17)
    fig.text(0.5, 0.01, "Native descriptors and resolutions; oracle object seeds or oracle K. This is not automatic instance AP.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(visuals / "aggregate_comparison.png", dpi=180)
    fig.savefig(visuals / "aggregate_comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#F7F8FA")
    for arm in ARMS:
        values = [np.mean([summaries[arm]["scenes"][scene]["cosine_threshold_sweep"][i]["macro_iou"] for scene in SCENES]) for i in range(len(THRESHOLDS))]
        ax.plot(THRESHOLDS, values, marker="o", label=manifest["arms"][arm]["label"])
    ax.set(xlabel="Cosine rejection threshold", ylabel="Mean scene object IoU", ylim=(0, 1),
           title="Threshold sweep · diagnostic, no evaluation-set threshold selected")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(visuals / "cosine_threshold_sweep.png", dpi=180)
    fig.savefig(visuals / "cosine_threshold_sweep.pdf")
    plt.close(fig)
    for name, rows in (("per_scene_metrics.csv", per_scene_rows), ("per_object_metrics.csv", per_object_rows)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(root / "summary.json", {"status": "COMPLETE", "manifest_sha256": manifest_sha,
        "protocol": manifest["protocol"], "arms": manifest["arms"], "aggregate": aggregate,
        "scene_count": len(SCENES), "object_count": sum(len(row["objects"]) for row in manifest["scenes"]),
        "per_scene": {arm: summaries[arm]["scenes"] for arm in ARMS},
        "runtimes": {arm: summaries[arm]["runtime"] for arm in ARMS}})
    report = ["# Frozen feature segmentation on ScanNet validation", "",
        "Completed four-model, two-method evaluation on 10 fixed validation scenes.", "",
        "Competing cosine uses one identical centroid-nearest GT vertex per object and no rejection. "
        "Spherical k-means uses all native scene tokens and oracle GT foreground K, without queries. "
        "Five seeded restarts are selected by the feature-space objective. Hungarian maximum IoU matching "
        "is applied afterward. Background leakage contributes false positives.", "",
        "| Model | Features / grid | Cosine mean scene IoU | Clustering mean scene IoU | Cosine objects IoU >= 0.5 | Clustering objects IoU >= 0.5 |",
        "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        spec = manifest["arms"][arm]
        a, b = aggregate[arm]["cosine"], aggregate[arm]["kmeans"]
        report.append(f"| {spec['label']} | {spec['feature_dim']}D / {spec['grid_size_m'] * 100:g} cm | {a['scene_mean_macro_iou']:.2%} | {b['scene_mean_macro_iou']:.2%} | {a['pooled_objects_iou50']:.2%} | {b['pooled_objects_iou50']:.2%} |")
    report += ["", "These are feature diagnostics with oracle seeds or oracle K. Models retain different native "
               "feature dimensions, grids, pretraining data and training budgets. These results are not automatic "
               "instance AP and do not isolate the effect of a single pretraining objective.", "",
               "All source/checkpoint/input identities are in manifest.json. Per-scene and per-object metrics, "
               "all threshold and clustering restart records, full native features and original-point predictions "
               "are retained. The visuals folder contains a 10-scene gallery, scene PNG/PDFs and aggregate plots."]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    atomic_json(root / "COMPLETE.json", {"manifest_sha256": manifest_sha, "scenes": SCENES,
        "arms": list(ARMS), "summary_sha256": sha(root / "summary.json"),
        "artifacts": {str(path.relative_to(root)): sha(path) for path in sorted(visuals.iterdir()) if path.suffix in (".png", ".pdf")}})
    print("EVALUATION_COMPLETE", flush=True)


def self_test():
    objects = [{"index": i, "gt_instance_id": i + 1, "vertices": 2, "query_vertex": i * 2} for i in range(2)]
    truth = np.array([0, 0, 1, 1, -1])
    perfect = np.array([0, 0, 1, 1, -1])
    result = metrics(perfect, truth, objects)
    assert result["macro_iou"] == 1 and result["background_rejection"] == 1
    contaminated = np.array([0, 0, 1, 1, 0])
    result = metrics(contaminated, truth, objects)
    assert np.isclose(result["macro_iou"], (2 / 3 + 1) / 2)
    aligned, mapping, _ = match_clusters(np.array([1, 1, 0, 0, -1]), truth, objects)
    assert np.array_equal(aligned, perfect) and np.array_equal(mapping, [1, 0])
    assert adjusted_rand([0, 0, 1, 1], [1, 1, 0, 0]) == 1
    missed = metrics(np.full(5, -1), truth, objects)
    assert missed["macro_iou"] == 0 and len(missed["per_object"]) == 2
    print("CPU_METRIC_CONTRACTS_PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "finalize", "self-test"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "worker":
        worker(args.root, args.arm)
    elif args.mode == "finalize":
        finalize(args.root)


if __name__ == "__main__":
    main()
