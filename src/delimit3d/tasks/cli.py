"""Public task training and evaluation commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from delimit3d.runtime.config import expand_environment, load_config
from delimit3d.runtime.records import config_sha256, file_sha256, write_json


def resolved_task_config(args, action=None):
    config = load_config(args.config, expand_env=False)
    if args.output is not None:
        if config["task"] == "mask3d":
            config["experiment"]["output_root"] = str(args.output.parent)
            config["experiment"]["name"] = args.output.name
        else:
            config["paths"]["output_root"] = str(args.output)
    if args.backbone_checkpoint is not None:
        name = config["model"].get("backbone_name", "litept_s_star")
        if name in {"concerto", "sonata"}:
            config[name]["checkpoint"] = str(args.backbone_checkpoint)
        elif config["task"] == "mask3d":
            config["initialization"].update(backbone_source="structured3d", backbone_checkpoint=str(args.backbone_checkpoint))
        else:
            config["encoder_initialization"].update(kind="pretrained_checkpoint", checkpoint=str(args.backbone_checkpoint))
    if args.set:
        import yaml
        for assignment in args.set:
            key, separator, value = assignment.partition("=")
            if not separator:
                raise ValueError("--set expects section.key=value")
            keys, parent = key.split("."), config
            for part in keys[:-1]:
                if part not in parent or not isinstance(parent[part], dict):
                    raise ValueError(f"Unknown config section: {key}")
                parent = parent[part]
            if keys[-1] not in parent:
                raise ValueError(f"Unknown config key: {key}")
            parent[keys[-1]] = yaml.safe_load(value)
    if action == "evaluate":
        if config["task"] == "mask3d":
            config["data"]["train_split"] = None
        else:
            config["paths"]["train_list"] = None
    if config["task"] == "agile3d" and (action == "train" or args.protocol == "mo"):
        config["paths"]["single_object_root"] = None
    if not args.show_config:
        config = expand_environment(config)
    config["_config_sha256"] = config_sha256(config)
    config["_config_path"] = str(args.config.resolve())
    return config


def agile_paths(config, *, training=True):
    from .agile3d import protocol
    paths = {key: Path(value).expanduser().resolve() for key, value in config["paths"].items() if value is not None}
    paths["agile3d_root"] = Path(protocol.__file__).parent
    for key in ("scan_folder", "val_list", "normal_root", "decoder_initialization") + (("train_list",) if training else ()):
        if not paths[key].exists():
            raise FileNotFoundError(f"{key}: {paths[key]}")
    policy = config.get("evaluation", {}).get("backbone_batchnorm_policy")
    if policy not in {"eval", "train_no_update"}:
        raise ValueError("Specify the AGILE3D BatchNorm inference policy explicitly")
    if config["protocol"].get("quantization_backend") != "numpy_utility_compat_v1":
        raise ValueError("AGILE3D uses the pinned NumPy representative ordering")
    if config["protocol"].get("coordinate_contract") != "official_agile3d_per_axis_minimum":
        raise ValueError("Unknown AGILE3D coordinate convention")
    from .data_identity import agile3d_data_identity
    config["_data_identity"] = agile3d_data_identity(paths, training=training)
    return paths


def run_task(action, argv=None):
    parser = argparse.ArgumentParser(description=f"{action.title()} a Delimit3D Mask3D or AGILE3D task.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Full task checkpoint for evaluation.")
    parser.add_argument("--resume", type=Path, help="Full task checkpoint including optimizer/RNG for continuation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", choices=("all", "mo", "so"), default="all", help="AGILE3D evaluation coverage.")
    args = parser.parse_args(argv)
    config = resolved_task_config(args, action)
    if args.show_config:
        print(json.dumps(config, indent=2))
        return
    if action == "evaluate" and args.checkpoint is None:
        parser.error("evaluation requires --checkpoint")
    if action == "train" and args.checkpoint is not None:
        parser.error("use --resume to continue training")
    if action == "train":
        output = (Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
                  if config["task"] == "mask3d" else Path(config["paths"]["output_root"]))
        if output.exists() and any(output.iterdir()):
            if args.resume is None:
                raise FileExistsError("Choose a fresh training output directory or supply --resume")
            import torch
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            latest = output / ("checkpoints/latest.pt" if config["task"] == "mask3d" else "checkpoints/checkpoint.pth")
            if latest.exists() and file_sha256(latest) != file_sha256(args.resume):
                raise ValueError("Resume would overwrite a newer checkpoint; use a fresh output directory")
            if config["task"] == "mask3d" and (output / "metrics.jsonl").exists():
                for line in (output / "metrics.jsonl").read_text().splitlines():
                    if json.loads(line).get("epoch", 0) > payload["epoch"]:
                        raise ValueError("Metrics extend beyond the resume checkpoint")
    if config["task"] == "mask3d":
        if action == "evaluate":
            from .mask3d.evaluate import evaluate
            return evaluate(config, args.checkpoint, args.device)
        from .mask3d.engine import main
        # Pass the resolved mapping directly; the engine retains its DDP,
        # target mapping, losses, scheduler and checkpoint recovery behavior.
        forwarded = ["--config", str(args.config), "--device", args.device]
        if args.resume:
            forwarded += ["--resume", str(args.resume)]
        main(forwarded, config=config)
        return
    if config["task"] != "agile3d":
        raise ValueError("task must be mask3d or agile3d")
    import torch
    from .agile3d import engine, trainer
    from .agile3d.model import Agile3DModel
    paths = agile_paths(config, training=action == "train")
    engine.set_seed(int(config["seed"]))
    device = torch.device(args.device)
    model = Agile3DModel(config, paths).to(device)
    if action == "train":
        trainer.train(model, config, paths, device, resume=args.resume)
        return
    payload = trainer.load_checkpoint(args.checkpoint, config, model=model)
    model.load_state_dict(payload["model"], strict=True)
    model.initial_encoder_state_sha256 = engine.state_sha256(model.backbone.state_dict())
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    official = engine.protocol_functions()
    result = {"checkpoint_sha256": file_sha256(args.checkpoint), "epoch": payload["epoch"] + 1,
              "evaluation_config_sha256": config["_config_sha256"], "evaluation_data": config["_data_identity"],
              "training_data": payload.get("data_identity"), "historical_checkpoint": payload.get("historical_checkpoint")}
    if args.protocol in {"all", "mo"}:
        dataset, collate = engine.make_dataset(config, paths, official, "val")
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(config["protocol"]["val_batch_size"]),
            shuffle=False, drop_last=False, collate_fn=collate, num_workers=int(config["runtime"]["num_workers"]),
            pin_memory=True, generator=torch.Generator().manual_seed(int(config["seed"]) + 1))
        result["multi_object"] = engine.evaluate(model, loader, config, paths, official, device, payload["epoch"] + 1)
    if args.protocol in {"all", "so"}:
        from delimit3d.evaluation.granularity_single_object import evaluate_single
        if "single_object_root" not in paths:
            raise ValueError("SO evaluation requires paths.single_object_root")
        root = paths["single_object_root"]
        result["single_object"] = evaluate_single(
            model, config, paths, official, engine, device,
            object_list=root / "object_ids.npy", object_classes=root / "object_classes.txt",
            output=paths["output_root"] / "single_object",
            voxel_size=float(config["protocol"]["voxel_size_m"]),
            identity={"checkpoint_sha256": result["checkpoint_sha256"], "config_sha256": config["_config_sha256"],
                      "data_identity": config["_data_identity"]},
        )
    write_json(paths["output_root"] / f"evaluation_epoch_{payload['epoch'] + 1:04d}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def train_main(argv=None):
    run_task("train", argv)


def evaluate_main(argv=None):
    run_task("evaluate", argv)
