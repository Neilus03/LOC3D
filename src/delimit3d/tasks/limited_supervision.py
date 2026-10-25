"""Portable, sealed Mask3D comparisons under two distinct label budgets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

from delimit3d.runtime.config import expand_environment, load_config
from delimit3d.runtime.records import file_sha256, config_sha256

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "loc3d_limited_supervision/v1"
ARMS = ("scratch", "pretrained")


def scenes(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Scene lists must be nonempty and contain no duplicates")
    if any(not re.fullmatch(r"scene\d{4}_\d{2}", scene) for scene in ids):
        raise ValueError("Expected ScanNet scene identifiers")
    return ids


def select_subset(pool, percent):
    if percent not in (1, 5, 10, 20) or len(pool) != len(set(pool)):
        raise ValueError("Invalid subset percentage or duplicate training scenes")
    order = sorted(pool)
    random.Random(42).shuffle(order)
    return order[:len(pool) * percent // 100]


def record(path):
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def code_identity():
    """Bind generated configurations to the implementation actually executed."""
    digest = hashlib.sha256()
    for folder in ("src", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".json", ".txt") and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
                digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def data_identity(config):
    from .data_identity import mask3d_data_identity
    return mask3d_data_identity(config)


def dependency_identity(root):
    root = Path(root).expanduser().resolve(strict=True)
    sources = sorted(root.rglob("*.py"))
    if not sources:
        raise ValueError("LitePT dependency checkout has no Python sources")
    return {str(path.relative_to(root)): file_sha256(path) for path in sources}


def validate_checkpoint(payload, config, manifest):
    from .contracts import model_contract
    if payload.get("baseline_config_sha256") != config_sha256(config):
        raise ValueError("Checkpoint belongs to a different arm, budget or training recipe")
    if payload.get("data_identity") != manifest["data_identity"]:
        raise ValueError("Checkpoint training data differs from the sealed experiment")
    if payload.get("model_contract") != model_contract(config):
        raise ValueError("Checkpoint model/readout contract differs from the sealed experiment")


def prepare(args):
    import yaml
    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError("Choose a fresh experiment directory; sealed experiments are immutable")
    train, val = scenes(args.train_split), scenes(args.val_split)
    # The bundled official scene universe defines these named protocols.
    for observed, name, count in ((train, "train", 1201), (val, "val", 312)):
        if len(observed) != count or set(observed) != set(scenes(ROOT / f"configs/splits/scannet_{name}.txt")):
            raise ValueError(f"Expected the official {count}-scene ScanNet {name} universe")
    if set(train) & set(val):
        raise ValueError("Training and validation scenes overlap")
    low_data = args.protocol == "low-data"
    if low_data:
        if args.percent is None or args.budget is not None or args.prepared_root is not None:
            raise ValueError("low-data requires --percent and dense inputs, without --budget/--prepared-root")
        if args.scans_root is None or args.gt_cache_root is None:
            raise ValueError("low-data requires --scans-root and --gt-cache-root")
        selected = select_subset(train, args.percent)
        recipe = "mask3d_low_data.yaml"
    else:
        if args.budget is None or args.percent is not None or args.scans_root is not None or args.gt_cache_root is not None:
            raise ValueError("low-annotation requires --budget and --prepared-root, without dense training inputs")
        if args.prepared_root is None:
            raise ValueError("low-annotation requires --prepared-root")
        selected, recipe = train, "mask3d_low_annotation.yaml"
    inputs = {"pretrained_encoder": record(args.backbone_checkpoint), "decoder": record(args.decoder_template),
              "source_train_split": record(args.train_split), "source_val_split": record(args.val_split)}
    if args.scratch_checkpoint is not None:
        inputs["scratch_encoder"] = record(args.scratch_checkpoint)
    litept_root = args.litept_root or os.environ.get("DELIMIT3D_LITEPT_ROOT")
    if not litept_root or not Path(litept_root).expanduser().is_dir():
        raise ValueError("Provide --litept-root or DELIMIT3D_LITEPT_ROOT pointing to the prepared LitePT checkout")
    config = load_config(ROOT / "configs/tasks" / recipe, expand_env=False)
    config["experiment"].update(output_root=str(root / "runs"), seed=42)
    config["model"]["litept_root"] = str(Path(litept_root).expanduser().resolve())
    config["initialization"].update(decoder_template=inputs["decoder"]["path"], scratch_checkpoint=None)
    config["data"].update(train_split=str(root / "train.txt"), val_split=str(root / "validation.txt"))
    if low_data:
        config["data"].update(scans_root=str(args.scans_root.resolve(strict=True)),
                             scannet_gt_cache_root=str(args.gt_cache_root.resolve(strict=True)))
    else:
        prepared = args.prepared_root.expanduser().resolve(strict=True)
        inputs["prepared_data_manifest"] = record(prepared / "data_manifest.json")
        config["data"]["limited_annotation"].update(prepared_root=str(prepared), budget_points=args.budget)
    config["initialization"]["backbone_checkpoint"] = inputs["pretrained_encoder"]["path"]
    config = expand_environment(config)
    dependency = dependency_identity(config["model"]["litept_root"])
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "train.txt").write_text("\n".join(selected) + "\n")
    (root / "validation.txt").write_text("\n".join(val) + "\n")
    files = {"train.txt": file_sha256(root / "train.txt"), "validation.txt": file_sha256(root / "validation.txt")}
    for arm in ARMS:
        arm_config = copy.deepcopy(config)
        arm_config["experiment"]["name"] = arm
        arm_config["initialization"].update(
            backbone_source="scratch" if arm == "scratch" else "structured3d",
            backbone_checkpoint=None if arm == "scratch" else inputs["pretrained_encoder"]["path"],
            scratch_checkpoint=inputs.get("scratch_encoder", {}).get("path") if arm == "scratch" else None)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(arm_config, sort_keys=False))
        files[str(path.relative_to(root))] = file_sha256(path)
    observed_data = data_identity(config)
    steps = math.ceil(len(selected) / 5)
    manifest = {"schema": SCHEMA, "protocol": args.protocol, "seed": 42,
        "percent": args.percent, "budget_points": args.budget, "checkpoint_label": args.checkpoint_label,
        "inputs": inputs, "files": files, "code_sha256": code_identity(), "data_identity": observed_data,
        "litept_source_sha256": dependency,
        "selected_scenes": selected, "validation_scenes": val,
        "exposure": {"train_scenes": len(selected), "validation_scenes": 312, "epochs": 600,
                     "batch_scenes": 5, "updates_per_epoch": steps, "updates_total": steps * 600,
                     "scene_visits": len(selected) * 600, "repeat_to_full_training_size": False},
        "metric": "custom class-agnostic raw-query CA-AP on ScanNet20 objects",
        "scratch_initialization": "explicit_saved_initial_encoder" if args.scratch_checkpoint else "seed42_fresh_initialization",
        "comparison": "same data, task, decoder, seed and schedule; only encoder initialization differs",
        "historical_reproduction": "requires the original checkpoint identities and prepared data; a new checkpoint is a new comparison"}
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(file_sha256(root / "manifest.json") + "\n")
    return verify(root)


def verify(root):
    root = Path(root).expanduser().resolve(strict=True)
    if file_sha256(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Experiment manifest changed")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unknown experiment schema")
    for relative, expected in manifest["files"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Sealed file changed: {relative}")
    for item in manifest["inputs"].values():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Input changed: {item['path']}")
    if code_identity() != manifest["code_sha256"]:
        raise ValueError("Implementation changed; keep the sealed checkout or prepare a new experiment")
    configs = [load_config(root / "configs" / f"{arm}.yaml") for arm in ARMS]
    if dependency_identity(configs[0]["model"]["litept_root"]) != manifest["litept_source_sha256"]:
        raise ValueError("LitePT dependency sources changed")
    if data_identity(configs[0]) != manifest["data_identity"]:
        raise ValueError("Prepared training or validation data changed")
    for config in configs:
        config["experiment"].pop("name")
        for key in ("backbone_source", "backbone_checkpoint", "scratch_checkpoint"):
            config["initialization"].pop(key)
    if configs[0] != configs[1]:
        raise ValueError("Matched arms differ beyond encoder initialization")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare", help="Seal matched scratch/pretrained recipes without launching jobs")
    prep.add_argument("--protocol", choices=("low-data", "low-annotation"), required=True)
    prep.add_argument("--percent", type=int, choices=(1, 5, 10, 20))
    prep.add_argument("--budget", type=int, choices=(20, 50, 100, 200))
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--train-split", type=Path, default=ROOT / "configs/splits/scannet_train.txt")
    prep.add_argument("--val-split", type=Path, default=ROOT / "configs/splits/scannet_val.txt")
    prep.add_argument("--backbone-checkpoint", type=Path, required=True)
    prep.add_argument("--checkpoint-label", required=True, help="Explicit pretraining recipe/epoch identity; recorded verbatim")
    prep.add_argument("--scratch-checkpoint", type=Path, help="Optional saved random initial encoder for historical reproduction")
    prep.add_argument("--decoder-template", type=Path, required=True)
    prep.add_argument("--litept-root", type=Path)
    prep.add_argument("--scans-root", type=Path)
    prep.add_argument("--gt-cache-root", type=Path)
    prep.add_argument("--prepared-root", type=Path)
    for action in ("verify", "train", "evaluate"):
        sub = commands.add_parser(action)
        sub.add_argument("--experiment", type=Path, required=True)
        if action != "verify":
            sub.add_argument("--arm", choices=ARMS, required=True)
            sub.add_argument("--device", default="cuda")
        if action == "train":
            sub.add_argument("--resume", type=Path)
        if action == "evaluate":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args)
    else:
        result = verify(args.experiment)
    if args.action in ("prepare", "verify"):
        print(json.dumps({"verified": True, "protocol": result["protocol"], "exposure": result["exposure"]}, indent=2))
        return
    # The reference experiment uses one real joint batch on one GPU.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The sealed limited-supervision workflow requires a single GPU process")
    from .cli import run_task
    root = args.experiment.expanduser().resolve()
    checkpoint = args.checkpoint if args.action == "evaluate" else args.resume
    if checkpoint is not None:
        import torch
        validate_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False),
                            load_config(root / "configs" / f"{args.arm}.yaml"), result)
    forwarded = ["--config", str(root / "configs" / f"{args.arm}.yaml"), "--device", args.device]
    if args.action == "train" and args.resume:
        forwarded += ["--resume", str(args.resume)]
    if args.action == "evaluate":
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Choose a fresh evaluation output directory")
        forwarded += ["--checkpoint", str(args.checkpoint), "--output", str(args.output)]
    return run_task(args.action, forwarded)
