"""Preparation commands for geometry, 2D teachers, sources and fixed evaluation."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

from delimit3d.runtime.config import expand_environment, load_config, METADATA_KEYS
from delimit3d.runtime.records import file_sha256, write_json


def geometry_main(argv):
    parser = argparse.ArgumentParser(description="Prepare Structured3D geometry without reading GT or bbox archives.")
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--raw-zips-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    from delimit3d.source.datasets.structured3d.prepare import prepare_structured3d_scene
    from delimit3d.cli.prepare_structured3d_raw import Structured3DReader
    from delimit3d.training.contracts import load_scene_ids
    from .prepare_2d import geometry_reference
    reader = Structured3DReader([str(path) for path in sorted(args.raw_zips_dir.glob("*perspective_full*.zip"))])
    try:
        for scene in load_scene_ids(args.scene_ids_file):
            prepare_structured3d_scene(scene, str(args.raw_zips_dir), str(args.output_root), include_gt=False)
            geometry_reference(args.output_root / scene, reader)
    finally:
        for archive in reader.readers:
            archive.close()


def sources_main(argv):
    parser = argparse.ArgumentParser(description="Project UnSAMv2 masks into a new 2D-only source pack.")
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--raw-zips-dir", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-mode", choices=("full_scene", "sphere_50k"), default="sphere_50k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from delimit3d.cli.prepare_structured3d_raw import Structured3DReader
    from delimit3d.training.contracts import load_scene_ids
    from .prepare_2d import build_2d_scene
    reader = Structured3DReader([str(path) for path in sorted(args.raw_zips_dir.glob("*perspective_full*.zip"))])
    try:
        for scene in load_scene_ids(args.scene_ids_file):
            build_2d_scene(scene_id=scene, geometry_root=args.geometry_root, proposal_root=args.proposal_root,
                output_root=args.output_root, reader=reader, crop_mode=args.crop_mode, seed=args.seed)
    finally:
        for archive in reader.readers:
            archive.close()


def seal_main(argv):
    parser = argparse.ArgumentParser(description="Hash and validate every prepared training/validation source.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from delimit3d.training.contracts import load_scene_ids, audit_scene_source_manifest, EXPERIMENT_ID
    train, val = load_scene_ids(args.train_split), load_scene_ids(args.holdout_split)
    if not train or not val:
        raise ValueError("Training and validation scene splits must both be nonempty")
    if set(train) & set(val):
        raise ValueError("Training and validation scene splits overlap")
    audits = [audit_scene_source_manifest(args.source_root / scene / "source_manifest.json", expected_scene_id=scene) for scene in train + val]
    cell_sets = {tuple(sorted(audit.trainable_proposals)) for audit in audits}
    if len(cell_sets) != 1 or any(not cell.startswith("2d/") for cell in next(iter(cell_sets))):
        raise ValueError("Seal a uniform 2D-only source pack")
    source_cells = list(cell_sets.pop())
    crop_modes = {audit.crop_mode for audit in audits}
    if len(crop_modes) != 1:
        raise ValueError("Do not combine different crop preparation protocols")
    total_bytes = sum(path.stat().st_size for scene in train + val for path in (args.source_root / scene).iterdir() if path.is_file())
    value = {"schema_version": "delimit3d_2d_source_audit/v1", "experiment_id": EXPERIMENT_ID,
        "dataset": "structured3d", "source_root": str(args.source_root.resolve()), "crop_mode": crop_modes.pop(),
        "source_cells": source_cells,
        "artifact_hashes_verified": True, "passed": True, "checks": {"disjoint_splits": True, "all_scene_hashes_verified": True},
        "scene_count": len(audits), "train_scene_count": len(train), "validation_scene_count": len(val),
        "total_artifact_bytes": total_bytes,
        "splits": {key: {"sha256": file_sha256(path), "scene_count": len(scenes)}
                   for key, path, scenes in (("optimization", args.train_split, train), ("validation", args.holdout_split, val))},
        "scenes": [{"scene_id": audit.scene_id, "manifest_sha256": audit.manifest_sha256,
                    "split": "optimization" if audit.scene_id in set(train) else "validation",
                    "num_points": audit.crop_points,
                    "trainable_proposals": audit.trainable_proposals} for audit in audits]}
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, value)


def fixed_eval_main(argv):
    parser = argparse.ArgumentParser(description="Create an immutable fixed native-token heldout evaluation pack.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("sam2", "unsamv2"), required=True)
    parser.add_argument("--grid-size", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretraining-contract", type=Path,
        help="Read the negative policy from the SAM2 pretraining recipe.")
    parser.add_argument("--allow-shared-proposal-negatives", action=argparse.BooleanOptionalAction,
        default=None, help="Use selected-proposal negatives; otherwise use overlap veto.")
    args = parser.parse_args(argv)
    policy = args.allow_shared_proposal_negatives
    if args.pretraining_contract is not None:
        from delimit3d.runtime.config import load_config
        recipe = load_config(args.pretraining_contract, expand_env=False)
        configured_policy = recipe.get("allow_shared_proposal_negatives", False)
        if not isinstance(configured_policy, bool):
            parser.error("allow_shared_proposal_negatives must be a boolean")
        if policy is not None and policy != configured_policy:
            parser.error("Negative policy differs from the supplied pretraining contract")
        if args.teacher == "sam2" and recipe.get("source_granularity") != "sam2":
            parser.error("SAM2 fixed evaluation requires a SAM2 pretraining contract")
        policy = configured_policy
    policy = False if policy is None else policy
    if policy and args.teacher != "sam2":
        parser.error("Selected-proposal negatives are restricted to SAM2 fixed evaluation")
    from delimit3d.training.contracts import load_scene_ids, TWO_D_CELLS
    from delimit3d.pretraining.sources import load_scene_source
    from delimit3d.pretraining.settings import sampling_config_for_profile
    from delimit3d.pretraining.fixed_eval import (
        prepare_fixed_holdout_eval_pack, validate_fixed_eval_negative_policy,
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file():
        validate_fixed_eval_negative_policy(json.loads(manifest_path.read_text()),
            allow_shared_proposal_negatives=policy)
    scenes = [load_scene_source(args.source_root / scene / "source_manifest.json",
        coordinate_normalization_policy="structured3d_local_y_up_to_scannet_z_up_v1") for scene in load_scene_ids(args.holdout_split)]
    sampling = sampling_config_for_profile("pf_hnm_512_v1")
    plans = None
    if args.teacher == "sam2":
        from delimit3d.training.sam2_pretraining import admit_sam2_holdout
        scenes, plans = admit_sam2_holdout(scenes, sampling=sampling, seed=args.seed,
            grid_size=args.grid_size, report_path=args.output.with_name(args.output.name + "_admission.json"),
            allow_shared_proposal_negatives=policy)
    report = prepare_fixed_holdout_eval_pack(pack_dir=args.output, scenes=scenes,
        sampling=sampling, sampling_profile="pf_hnm_512_v1",
        seed=args.seed, cells=("2d/sam2",) if args.teacher == "sam2" else TWO_D_CELLS,
        grid_size=args.grid_size, precomputed_token_plans=plans,
        allow_shared_proposal_negatives=policy)
    print(json.dumps(report, indent=2))


def code_record_main(argv):
    parser = argparse.ArgumentParser(description="Record this checkout's executable source bytes.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError("Run code-record from the release checkout or supply --repo-root")
    files = [path for folder in (root / "src", root / "configs", root / "scripts", root / "environment") for path in folder.rglob("*")
             if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".patch", ".txt", ".list"} and "__pycache__" not in path.parts]
    files.append(root / "pyproject.toml")
    value = {"schema_version": "delimit3d_code_manifest/v1", "passed": True,
        "snapshot_root": str(root), "files": {str(path.relative_to(root)): file_sha256(path) for path in sorted(files)}}
    if args.output.exists():
        if json.loads(args.output.read_text()) != value:
            raise FileExistsError("Existing code record differs; use a fresh output filename")
        return
    write_json(args.output, value)


def decoder_main(argv):
    parser = argparse.ArgumentParser(description="Create the same fresh decoder template for all backbone comparisons.")
    parser.add_argument("--task", choices=("mask3d", "agile3d"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    seed = args.seed if args.seed is not None else (42 if args.task == "mask3d" else 20260912)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if args.task == "mask3d":
            from delimit3d.tasks.mask3d.model import LitePTMask3DDecoder
            from delimit3d.tasks.mask3d.initialization import save_decoder_template
            report = save_decoder_template(LitePTMask3DDecoder(eval_queries=100), args.output, downstream_seed=seed)
        else:
            from delimit3d.evaluation.agile3d_decoder import save_initialization
            report = save_initialization(args.output, seed=seed, decoder_kwargs={
                "feature_dim": 72, "hidden_dim": 128, "num_heads": 8, "dim_feedforward": 1024,
                "num_decoders": 3, "num_bg_queries": 10, "dropout": 0.0, "pre_norm": False,
                "max_click_events": 200, "normalize_pos_enc": True, "gauss_scale": 1.0, "aux": True})
    print(json.dumps(report, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Prepare Delimit3D geometry, 2D masks and reproducible source packs.", add_help=False)
    actions = {"geometry": geometry_main, "sources": sources_main, "seal": seal_main,
               "fixed-eval": fixed_eval_main, "code-record": code_record_main, "decoder-init": decoder_main}
    from delimit3d.runtime.setup import source_main, weight_main
    actions.update(dependencies=source_main, weights=weight_main)
    routes = {"sam2": "delimit3d.cli.export_structured3d_sam2", "sam2-seal": "delimit3d.cli.finalize_structured3d_sam2",
              "unsamv2": "delimit3d.cli.prepare_structured3d_raw",
              "backbone-init": "delimit3d.cli.initialize_scratch",
              "task-inputs": "delimit3d.data.prepare_tasks",
              "normal-audit": "delimit3d.data.normal_audit",
              "calibration": "delimit3d.pretraining.calibration"}
    parser.add_argument("--action", choices=(*actions, *routes))
    parser.add_argument("--config", type=Path)
    known, remaining = parser.parse_known_args(argv)
    if known.config:
        config = load_config(known.config, expand_env=False)
        action = known.action or config.pop("action", None)
        defaults = []
        explicit = {item.split("=", 1)[0] for item in remaining if item.startswith("--")}
        for key, value in config.items():
            option = "--" + key.replace("_", "-")
            if key in METADATA_KEYS or option in explicit or value is False or value is None:
                continue
            value = expand_environment(value)
            defaults.append(option)
            if value is not True:
                defaults.extend(map(str, value if isinstance(value, list) else [value]))
        remaining = defaults + remaining
    else:
        action = known.action
    if action is None:
        parser.print_help()
        if "--help" in argv or "-h" in argv:
            return
        parser.error("supply --action or a preparation --config")
    if action in actions:
        return actions[action](remaining)
    return importlib.import_module(routes[action]).main(remaining)
