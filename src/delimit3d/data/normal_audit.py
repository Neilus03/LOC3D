"""Verify the exact ordered RGBN6 scenes selected for BN recalibration."""
import argparse
import json
from pathlib import Path
import numpy as np
from delimit3d.runtime.records import file_sha256, write_json
from delimit3d.cli.recalibration_contract import build_bn_recalibration_scene_selection

SCHEMA = "delimit3d_rgbn6_normal_audit/v1"


def validate(payload, *, source_root, train_split, scene_count, seed, expected_train_split_sha256=None):
    selection = build_bn_recalibration_scene_selection(train_split, scene_count=scene_count, seed=seed)
    stored = payload.get("scene_selection", {})
    if (payload.get("schema_version") != SCHEMA or payload.get("passed") is not True
            or stored.get("selected_scene_ids") != selection["selected_scene_ids"]
            or stored.get("source_split", {}).get("sha256") != selection["source_split"]["sha256"]
            or stored.get("seed") != seed or stored.get("scene_count") != scene_count):
        raise ValueError("Normal audit selection changed")
    if expected_train_split_sha256 and selection["source_split"]["sha256"] != expected_train_split_sha256:
        raise ValueError("Normal audit split changed")
    rows = payload.get("scenes", [])
    if [row.get("scene_id") for row in rows] != selection["selected_scene_ids"]:
        raise ValueError("Normal audit scene order changed")
    for row in rows:
        path = Path(source_root) / row["scene_id"] / "source_manifest.json"
        if row.get("passed") is not True or file_sha256(path) != row["manifest_sha256"]:
            raise ValueError("Normal audit source changed")
    return {"schema_version": SCHEMA, "selection": selection,
            "source_manifest_sha256": {row["scene_id"]: row["manifest_sha256"] for row in rows}, "passed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    from delimit3d.pretraining.sources import load_scene_source
    selection = build_bn_recalibration_scene_selection(args.train_split, scene_count=args.scenes, seed=args.seed)
    rows = []
    for scene_id in selection["selected_scene_ids"]:
        scene = load_scene_source(args.source_root / scene_id / "source_manifest.json",
            coordinate_normalization_policy="structured3d_local_y_up_to_scannet_z_up_v1", verify_hashes=True)
        normals = scene.normals
        if normals is None or normals.shape != scene.points.shape or not np.isfinite(normals).all():
            raise ValueError(f"Missing, nonfinite or misaligned normals: {scene_id}")
        lengths = np.linalg.norm(normals, axis=1)
        nonzero = lengths[lengths > 1e-6]
        if not len(nonzero) or (nonzero < .9).any() or (nonzero > 1.002).any():
            raise ValueError(f"Invalid normal magnitudes: {scene_id}")
        if not np.array_equal(scene.features[:, 3:], normals):
            raise ValueError("RGBN6 feature normals differ from the canonicalized geometry normals")
        rows.append({"scene_id": scene_id, "manifest_sha256": scene.manifest_sha256,
                     "num_points": len(normals), "zero_normals": int((lengths <= 1e-6).sum()), "passed": True})
    report = {"schema_version": SCHEMA, "scene_selection": selection,
        "coordinate_policy": "structured3d_local_y_up_to_scannet_z_up_v1", "scenes": rows, "passed": True}
    validate(report, source_root=args.source_root, train_split=args.train_split, scene_count=args.scenes, seed=args.seed)
    write_json(args.output, report)
    print(json.dumps({"scenes": len(rows), "passed": True}, indent=2))
