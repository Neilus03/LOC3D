#!/usr/bin/env python3
"""Sealed, original-vertex ScanNet200 targets for frozen-feature diagnostics.

The processed dataset uses zero-based instance IDs (-1 is ignored) and
contiguous ScanNet200 semantic indices. Wall and floor are unscored; ceiling is
one of the 198 instance classes. Every source vertex remains in the input,
common grid, and metric support, including vertices of unscored objects.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


# Official ScanNet200 semantic order, also used by Pointcept's segment200.npy.
CLASS_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
    22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41,
    42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122,
    125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
    155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185,
    188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242,
    250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356,
    370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609, 748,
    776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171,
    1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183,
    1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191,
)
CLASS_NAMES = (
    "wall", "chair", "floor", "table", "door", "couch", "cabinet", "shelf",
    "desk", "office chair", "bed", "pillow", "sink", "picture", "window",
    "toilet", "bookshelf", "monitor", "curtain", "book", "armchair",
    "coffee table", "box", "refrigerator", "lamp", "kitchen cabinet",
    "towel", "clothes", "tv", "nightstand", "counter", "dresser", "stool",
    "cushion", "plant", "ceiling", "bathtub", "end table", "dining table",
    "keyboard", "bag", "backpack", "toilet paper", "printer", "tv stand",
    "whiteboard", "blanket", "shower curtain", "trash can", "closet",
    "stairs", "microwave", "stove", "shoe", "computer tower", "bottle",
    "bin", "ottoman", "bench", "board", "washing machine", "mirror",
    "copier", "basket", "sofa chair", "file cabinet", "fan", "laptop",
    "shower", "paper", "person", "paper towel dispenser", "oven", "blinds",
    "rack", "plate", "blackboard", "piano", "suitcase", "rail", "radiator",
    "recycling bin", "container", "wardrobe", "soap dispenser", "telephone",
    "bucket", "clock", "stand", "light", "laundry basket", "pipe",
    "clothes dryer", "guitar", "toilet paper holder", "seat", "speaker",
    "column", "bicycle", "ladder", "bathroom stall", "shower wall", "cup",
    "jacket", "storage bin", "coffee maker", "dishwasher", "paper towel roll",
    "machine", "mat", "windowsill", "bar", "toaster", "bulletin board",
    "ironing board", "fireplace", "soap dish", "kitchen counter", "doorframe",
    "toilet paper dispenser", "mini fridge", "fire extinguisher", "ball",
    "hat", "shower curtain rod", "water cooler", "paper cutter", "tray",
    "shower door", "pillar", "ledge", "toaster oven", "mouse",
    "toilet seat cover dispenser", "furniture", "cart", "storage container",
    "scale", "tissue box", "light switch", "crate", "power outlet",
    "decoration", "sign", "projector", "closet door", "vacuum cleaner",
    "candle", "plunger", "stuffed animal", "headphones", "dish rack", "broom",
    "guitar case", "range hood", "dustpan", "hair dryer", "water bottle",
    "handicap bar", "purse", "vent", "shower floor", "water pitcher",
    "mailbox", "bowl", "paper bag", "alarm clock", "music stand",
    "projector screen", "divider", "laundry detergent", "bathroom counter",
    "object", "bathroom vanity", "closet wall", "laundry hamper",
    "bathroom stall door", "ceiling light", "trash bin", "dumbbell",
    "stair rail", "tube", "bathroom cabinet", "cd case", "closet rod",
    "coffee kettle", "structure", "shower head", "keyboard piano",
    "case of water bottles", "coat rack", "storage organizer", "folded chair",
    "fire alarm", "power strip", "calendar", "poster", "potted plant",
    "luggage", "mattress",
)
EXCLUDED_CLASS_IDS = (1, 3)
SOURCE_ASSETS = ("coord", "color", "normal", "segment200", "instance")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_mesh_xyz(path):
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertices[key] for key in ("x", "y", "z")]).astype(np.float32)


def _taxonomy(class_ids=None, class_names=None):
    ids = tuple(int(value) for value in (CLASS_IDS if class_ids is None else class_ids))
    names = tuple(CLASS_NAMES if class_names is None else class_names)
    if len(ids) != 200 or len(names) != 200 or len(set(ids)) != 200:
        raise ValueError("ScanNet200 taxonomy must have 200 unique IDs and 200 names")
    if any(value <= 0 for value in ids) or not all(str(name).strip() for name in names):
        raise ValueError("Invalid ScanNet200 taxonomy")
    return ids, names


def prepare_scene(scene, source_dir, out_dir, *, grid_size=0.05,
                  min_instance_points=1, class_ids=None, class_names=None,
                  mesh_path=None):
    """Prepare one scene, returning its manifest-ready record.

    ``source_dir`` is the processed scene folder; ``out_dir`` is the destination
    scene folder. The raw mesh is inferred from the adjacent scannet_raw root,
    or supplied explicitly. It is required to verify original vertex identity.
    Instances with fewer than ``min_instance_points`` remain input/background
    vertices. The default keeps all recognized nonempty thing instances; the
    official 100-vertex eligibility flag is recorded for every kept object.
    """
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    ids, names = _taxonomy(class_ids, class_names)
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("Grid size must be finite and positive")
    if isinstance(min_instance_points, bool) or int(min_instance_points) != min_instance_points or min_instance_points < 1:
        raise ValueError("Minimum instance size must be a positive integer")
    min_instance_points = int(min_instance_points)
    if mesh_path is None:
        mesh_path = source_dir.parents[2] / "scannet_raw" / "scans" / scene / f"{scene}_vh_clean_2.ply"
    mesh_path = Path(mesh_path)
    paths = {key: source_dir / f"{key}.npy" for key in SOURCE_ASSETS}
    paths["mesh"] = mesh_path
    sources = {key: {"path": str(path.resolve()), "sha256": sha(path),
                     "bytes": path.stat().st_size} for key, path in paths.items()}
    metadata_path = out_dir / "target.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if (existing["source_files"] != sources or existing["common_grid_m"] != grid_size
                or existing["eligibility"]["min_instance_points"] != min_instance_points
                or existing["taxonomy"]["class_ids"] != list(ids)
                or existing["taxonomy"]["class_names"] != list(names)):
            raise ValueError(f"{scene}: refusing to replace different sealed targets")
        load_scene(existing, verify=True)
        return existing

    data = {key: np.load(path, allow_pickle=False) for key, path in paths.items() if key != "mesh"}
    xyz = np.asarray(data["coord"], dtype=np.float32)
    rgb = np.asarray(data["color"], dtype=np.float32)
    normal = np.asarray(data["normal"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError(f"{scene}: invalid or empty XYZ array")
    if rgb.shape != xyz.shape or normal.shape != xyz.shape:
        raise ValueError(f"{scene}: geometry/color/normal shape mismatch")
    if not all(np.isfinite(array).all() for array in (xyz, rgb, normal)):
        raise ValueError(f"{scene}: nonfinite inputs")
    if rgb.min() < 0 or rgb.max() > 255:
        raise ValueError(f"{scene}: RGB outside [0,255]")
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    semantic, instance = data["segment200"], data["instance"]
    for key, array in (("segment200", semantic), ("instance", instance)):
        if array.shape != (len(xyz),) or array.dtype.kind not in "iu" or array.min() < -1:
            raise ValueError(f"{scene}: invalid {key} labels")
    if semantic.max() >= len(ids):
        raise ValueError(f"{scene}: segment200 is not a contiguous ScanNet200 label")
    raw_xyz = _read_mesh_xyz(mesh_path)
    if xyz.shape != raw_xyz.shape or not np.allclose(xyz, raw_xyz, rtol=0, atol=1e-5):
        raise ValueError(f"{scene}: processed coordinates differ from original mesh vertex order")

    labels = np.full(len(xyz), -1, dtype=np.int32)
    objects, queries, excluded = [], [], []
    for raw_id in np.unique(instance[instance >= 0]):
        vertices = np.flatnonzero(instance == raw_id)
        classes = np.unique(semantic[vertices])
        if len(classes) != 1:
            raise ValueError(f"{scene}: instance {raw_id} mixes semantic classes {classes.tolist()}")
        semantic_index = int(classes[0])
        class_id = None if semantic_index < 0 else ids[semantic_index]
        reason = ("unrecognized_semantic_class" if class_id is None else
                  "wall_or_floor" if class_id in EXCLUDED_CLASS_IDS else
                  "below_min_instance_points" if len(vertices) < min_instance_points else None)
        if reason:
            excluded.append({"raw_instance_id": int(raw_id), "semantic_index": semantic_index,
                             "class_id": class_id, "vertices": len(vertices), "reason": reason})
            continue
        index = len(objects)
        center = xyz[vertices].mean(axis=0)
        query = int(vertices[np.square(xyz[vertices] - center).sum(axis=1).argmin()])
        labels[vertices] = index
        queries.append(query)
        objects.append({"index": index, "gt_instance_id": int(raw_id),
                        "raw_instance_id": int(raw_id), "semantic_index": semantic_index,
                        "class_index": semantic_index,
                        "class_id": int(class_id), "class_name": names[semantic_index],
                        "vertices": len(vertices), "query_vertex": query,
                        "official_min_region_100_eligible": bool(len(vertices) >= 100)})
    if not objects:
        raise ValueError(f"{scene}: no eligible foreground instances")
    shift = np.array([(xyz[:, 0].min() + xyz[:, 0].max()) / 2,
                      (xyz[:, 1].min() + xyz[:, 1].max()) / 2,
                      xyz[:, 2].min()], dtype=np.float32)
    voxels = np.floor((xyz - shift) / grid_size).astype(np.int64)
    common_grid, point_to_cell = np.unique(voxels, axis=0, return_inverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    packed_path = out_dir / "scene.npz"
    temporary = packed_path.with_name(packed_path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, xyz=xyz, rgb255=rgb, normal=normal, labels=labels,
                            queries=np.asarray(queries, dtype=np.int64), shift=shift,
                            common_grid=common_grid, point_to_cell=point_to_cell.astype(np.int32))
    os.replace(temporary, packed_path)
    # Recheck hashes after reading to reject concurrent source replacement.
    if any(sha(paths[key]) != source["sha256"] for key, source in sources.items()):
        raise ValueError(f"{scene}: input source changed during preparation")
    record = {"schema": "delimit3d.scannet200_feature_targets/v1", "scene": scene,
              "input": str(packed_path.resolve()), "sha256": sha(packed_path),
              "original_points": len(xyz), "common_cells": len(common_grid),
              "common_grid_m": float(grid_size), "objects": objects,
              "background_points": int((labels < 0).sum()),
              "coordinate_array_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
              "source_files": sources,
              "taxonomy": {"class_ids": list(ids), "class_names": list(names),
                           "semantic_encoding": "zero-based contiguous index; -1 ignored",
                           "instance_encoding": "zero-based raw instance id; -1 ignored"},
              "eligibility": {"excluded_class_ids": list(EXCLUDED_CLASS_IDS),
                              "excluded_class_names": ["wall", "floor"],
                              "min_instance_points": min_instance_points,
                              "official_min_region_points": 100,
                              "official_min_region_100_objects": sum(obj["official_min_region_100_eligible"] for obj in objects),
                              "excluded_instances": excluded,
                              "ignored_instance_points": int((instance < 0).sum()),
                              "ignored_vertices_remain_in_input_and_scoring": True,
                              "claim_boundary": "GT-K feature diagnostic, not official instance AP; all original vertices contribute false-positive leakage"}}
    _write_json(metadata_path, record)
    return record


def load_scene(record, *, verify=True):
    """Return ``(arrays, objects)`` from a scene record or target.json path."""
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text())
    path = Path(record["input"])
    if verify and sha(path) != record["sha256"]:
        raise ValueError(f"{record['scene']}: sealed input hash mismatch")
    with np.load(path, allow_pickle=False) as packed:
        arrays = {key: packed[key] for key in packed.files}
    n, objects = record["original_points"], record["objects"]
    if arrays["xyz"].shape != (n, 3) or arrays["labels"].shape != (n,):
        raise ValueError(f"{record['scene']}: target point support changed")
    if arrays["point_to_cell"].shape != (n,) or arrays["common_grid"].shape != (record["common_cells"], 3):
        raise ValueError(f"{record['scene']}: common-grid shape mismatch")
    if not np.array_equal(arrays["queries"], [obj["query_vertex"] for obj in objects]):
        raise ValueError(f"{record['scene']}: query anchors changed")
    if verify:
        grid, inverse = np.unique(np.floor((arrays["xyz"] - arrays["shift"]) / record["common_grid_m"]).astype(np.int64), axis=0, return_inverse=True)
        if not np.array_equal(grid, arrays["common_grid"]) or not np.array_equal(inverse, arrays["point_to_cell"]):
            raise ValueError(f"{record['scene']}: common-grid identity changed")
        labels = arrays["labels"]
        if labels.min() < -1 or labels.max() >= len(objects):
            raise ValueError(f"{record['scene']}: invalid object labels")
        for index, obj in enumerate(objects):
            if obj["index"] != index or int((labels == index).sum()) != obj["vertices"] or labels[obj["query_vertex"]] != index:
                raise ValueError(f"{record['scene']}: object target metadata mismatch")
        if hashlib.sha256(arrays["xyz"].tobytes()).hexdigest() != record["coordinate_array_sha256"]:
            raise ValueError(f"{record['scene']}: coordinate array identity mismatch")
    return arrays, objects
