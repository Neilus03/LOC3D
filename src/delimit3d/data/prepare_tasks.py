"""Prepare ordered ScanNet geometry and downstream targets from official data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from delimit3d.runtime.records import file_sha256, write_json
from delimit3d.training.contracts import load_scene_ids


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area
    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]
    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    return nv / length


def mesh_arrays(raw, scene):
    from plyfile import PlyData
    path = raw / f"{scene}_vh_clean_2.ply"
    ply = PlyData.read(path)
    vertices = ply["vertex"].data
    points = np.column_stack([vertices[key] for key in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertices[key] for key in ("red", "green", "blue")]).astype(np.uint8)
    faces = np.stack(ply["face"].data["vertex_indices"], axis=0)
    normals = vertex_normal(points.astype(np.float64), faces).astype(np.float32)
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError(f"Invalid mesh normals: {scene}")
    return points, colors, normals, path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare ScanNet inputs without a CHORUS checkout.")
    parser.add_argument("--task", choices=("mask3d", "agile3d"), required=True)
    parser.add_argument("--raw-scans-root", type=Path, required=True)
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gt-cache-root", type=Path, help="Mask3D target cache directory.")
    parser.add_argument("--processed-scan-folder", type=Path, help="Official AGILE3D PLYs, for row alignment.")
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError("Use a fresh output root for prepared data")
    if args.task == "mask3d" and (args.gt_cache_root is None or args.gt_cache_root.exists()):
        parser.error("Mask3D requires a fresh --gt-cache-root")
    if args.task == "agile3d" and args.processed_scan_folder is None:
        parser.error("AGILE3D requires --processed-scan-folder")
    scenes, records, cache_rows = load_scene_ids(args.scene_ids_file), {}, {}
    for scene in scenes:
        raw = args.raw_scans_root / scene
        points, colors, normals, mesh_path = mesh_arrays(raw, scene)
        destination = args.output_root / scene
        destination.mkdir(parents=True, exist_ok=False)
        if args.task == "agile3d":
            from plyfile import PlyData
            ply_path = args.processed_scan_folder / f"{scene}.ply"
            vertex = PlyData.read(ply_path)["vertex"].data
            processed = np.column_stack([vertex[key] for key in ("x", "y", "z")]).astype(np.float32)
            if processed.shape != points.shape or not np.allclose(processed, points, rtol=0, atol=1e-5):
                raise ValueError(f"AGILE3D vertices differ from the raw mesh: {scene}")
            np.save(destination / "normal.npy", normals)
            np.save(destination / "coord.npy", points)
            records[scene] = {"mesh_sha256": file_sha256(mesh_path), "ply_sha256": file_sha256(ply_path),
                             "normal_sha256": file_sha256(destination / "normal.npy"), "num_points": len(points)}
            continue
        from delimit3d.source.datasets.scannet.gt import load_scannet_gt_instances
        from delimit3d.tasks.mask3d.data.scannet_gt_cache import cache_file, _atomic_write_npz
        gt = load_scannet_gt_instances(raw, scene, eval_benchmark="scannet20")
        if gt.segment_ids is None or gt.vertex_xyz is None or not np.allclose(gt.vertex_xyz, points, rtol=0, atol=1e-5):
            raise ValueError(f"ScanNet geometry/label rows differ: {scene}")
        for name, array in (("points", points), ("colors", colors), ("normals", normals)):
            np.save(destination / f"{name}.npy", array)
        artifacts = {name: file_sha256(destination / f"{name}.npy") for name in ("points", "colors", "normals")}
        write_json(destination / "scene_meta.json", {
            "schema": "delimit3d_scannet_geometry/v1", "scene_id": scene, "num_points": len(points),
            "granularities": [0.5], "label_files": {}, "artifacts_sha256": artifacts,
            "normal_policy": "litept_area_weighted_vertex_normals_eps_1e8_v1", "mesh_sha256": file_sha256(mesh_path)})
        pairs = np.asarray(sorted(gt.instance_class_ids.items()), dtype=np.int64).reshape(-1, 2)
        target = cache_file(args.gt_cache_root, scene)
        _atomic_write_npz(target, instance_ids=np.where(gt.instance_ids == 0, -1, gt.instance_ids).astype(np.int64),
                          segment_ids=np.asarray(gt.segment_ids, dtype=np.int64), instance_class_pairs=pairs)
        cache_rows[scene] = {"cache_file": target.name, "num_points": len(points),
                             "sha256": file_sha256(target)}
        records[scene] = {"geometry": artifacts, "targets": cache_rows[scene],
                         "mesh_sha256": file_sha256(mesh_path), "num_points": len(points)}
    if args.task == "mask3d":
        write_json(args.gt_cache_root / "manifest.json", {"cache_version": 1, "eval_benchmark": "scannet20", "scenes": cache_rows})
    write_json(args.output_root / "preparation.json", {"schema": "delimit3d_task_inputs/v1", "task": args.task,
        "split_sha256": file_sha256(args.scene_ids_file), "normal_policy": "litept_area_weighted_vertex_normals_eps_1e8_v1", "scenes": records})
    print(json.dumps({"task": args.task, "scenes": len(records), "output_root": str(args.output_root)}, indent=2))
