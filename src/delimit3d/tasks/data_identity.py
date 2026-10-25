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
    if config["data"].get("limited_annotation") is not None:
        return low_annotation_data_identity(config, evaluation=evaluation)
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


def low_annotation_data_identity(config, *, evaluation=False):
    """Hash consumed sanitized assets and verify their sealed preparation record."""
    data = config["data"]
    settings = data["limited_annotation"]
    root = Path(settings["prepared_root"]).resolve(strict=True)
    manifest_path = root / "data_manifest.json"
    manifest_hash = file_sha256(manifest_path)
    if (root / "data_manifest.sha256").read_text().strip() != manifest_hash:
        raise ValueError("LA data preparation manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("hidden_training_annotations_exported") is not False:
        raise ValueError("LA data must contain only selected training annotations")
    budget = int(settings["budget_points"])
    if budget not in manifest["budgets"]:
        raise ValueError("Annotation budget was not prepared")
    keys = ("val_split",) if evaluation else ("train_split", "val_split")
    splits = {key: load_scene_ids(Path(data[key])) for key in keys}
    if set(splits.get("train_split", [])) & set(splits["val_split"]):
        raise ValueError("LA train and validation splits overlap")
    records = {}
    for key, scene_ids in splits.items():
        allowed = set(manifest["scene_ids"]["train" if key == "train_split" else "validation"])
        if len(set(scene_ids)) != len(scene_ids) or not set(scene_ids) <= allowed:
            raise ValueError(f"LA {key} contains duplicate or foreign scenes")
        for scene in scene_ids:
            relative_paths = [f"geometry/{scene}.npz",
                              f"train_annotations/{budget}/{scene}.npz" if key == "train_split"
                              else f"validation/{scene}.npz"]
            records[scene] = {}
            for relative in relative_paths:
                path = (root / relative).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError("LA asset escapes its prepared data root")
                expected = manifest["files"][relative]
                digest = file_sha256(path)
                if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValueError(f"Sealed LA asset changed: {relative}")
                records[scene][relative] = digest
    return {"protocol": "scannet_official_limited_annotation", "budget_points": budget,
            "preparation_manifest_sha256": manifest_hash,
            "splits": {key: file_sha256(data[key]) for key in splits}, "scenes": records}
