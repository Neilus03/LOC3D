"""Verify and hash the actual geometry, normals and downstream targets."""
import hashlib
import json
from pathlib import Path
from delimit3d.runtime.records import file_sha256
from delimit3d.training.contracts import load_scene_ids


def agile3d_data_identity(paths, *, training=True):
    import numpy as np
    from plyfile import PlyData
    from .agile3d.protocol.datasets.normal_utils import _normal_candidates
    from delimit3d.evaluation.agile3d_rgbn6_contract import validate_normal_alignment
    keys = ("train_list", "val_list") if training else ("val_list",)
    splits, scenes = {}, set()
    for key in keys:
        entries = json.loads(paths[key].read_text())
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Expected the official AGILE3D sample mapping: {key}")
        splits[key] = set()
        for entry in entries:
            scene, separator, _ = entry.partition("_obj_")
            if not separator:
                raise ValueError(f"Malformed AGILE3D sample: {entry}")
            splits[key].add(scene)
            scenes.add(scene)
    if splits.get("train_list", set()) & splits["val_list"]:
        raise ValueError("AGILE3D train and validation scenes overlap")
    hashes, records = {}, []
    for scene in sorted(scenes):
        normal = next((p for p in _normal_candidates(paths["normal_root"], scene) if p.is_file()), None)
        if normal is None:
            raise FileNotFoundError(f"Missing ordered normals for {scene}")
        witness = normal.with_name("coord.npy")
        if not witness.is_file():
            raise FileNotFoundError(f"Missing normal coordinate witness: {witness}")
        ply = paths["scan_folder"] / (scene + ".ply")
        vertex = PlyData.read(ply)["vertex"]
        xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")])
        alignment = validate_normal_alignment(xyz, np.load(witness, mmap_mode="r"), np.load(normal, mmap_mode="r"))
        hashes[scene] = {"ply": file_sha256(ply), "normal": file_sha256(normal), "coordinate": file_sha256(witness)}
        records.append({"scene": scene, **alignment, "ply_sha256": hashes[scene]["ply"],
                        "normal_sha256": hashes[scene]["normal"], "coordinate_sha256": hashes[scene]["coordinate"]})
    return {"train_split": file_sha256(paths["train_list"]) if training else None,
            "val_split": file_sha256(paths["val_list"]), "scenes": hashes,
            "normal_contract_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def mask3d_data_identity(config, *, evaluation=False):
    from .mask3d.data.training_pack import _resolve_pack_dir
    from .mask3d.data.scannet_gt_cache import cache_file
    data = config["data"]
    splits = {key: load_scene_ids(Path(data[key])) for key in (("val_split",) if evaluation else ("train_split", "val_split"))}
    if set(splits.get("train_split", [])) & set(splits["val_split"]):
        raise ValueError("Mask3D train and validation splits overlap")
    records = {}
    for scene in sorted(set(splits.get("train_split", []) + splits["val_split"])):
        folder = _resolve_pack_dir(Path(data["scans_root"]) / scene)
        records[scene] = {name: file_sha256(folder / name) for name in
            ("points.npy", "colors.npy", "normals.npy", "scene_meta.json", "valid_points.npy", "seen_points.npy", "supervision_mask.npy")
            if (folder / name).is_file()}
        if data.get("scannet_gt_cache_root"):
            records[scene]["targets"] = file_sha256(cache_file(data["scannet_gt_cache_root"], scene))
        else:
            for path in sorted((Path(data["scans_root"]) / scene).glob("*")):
                if path.is_file() and path.name.endswith((".labels.ply", ".segs.json", ".aggregation.json")):
                    records[scene][path.name] = file_sha256(path)
    return {"splits": {key: file_sha256(data[key]) for key in splits}, "scenes": records}
