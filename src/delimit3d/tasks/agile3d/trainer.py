"""One training/resume loop for every AGILE3D backbone."""
from __future__ import annotations
import json
from pathlib import Path
import time
import torch

from delimit3d.runtime.records import save_torch, write_json
from . import engine
from delimit3d.tasks.contracts import model_contract

SCHEMA = "delimit3d_agile3d_checkpoint/v1"


def optimizer_for(model, config):
    settings = config["optimizer"]
    lr = float(settings.get("decoder_learning_rate", settings.get("learning_rate", 1e-4)))
    groups = [{"name": "adapters_decoder", "params": [*model.adapters.parameters(), *model.decoder.parameters()], "lr": lr}]
    if not model.frozen_encoder:
        groups.insert(0, {"name": "encoder", "params": list(model.backbone.parameters()),
                          "lr": lr * float(settings.get("encoder_learning_rate_multiplier", 0.1))})
    return torch.optim.AdamW(groups, weight_decay=float(settings["weight_decay"]))


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, config, *, validation_completed=False):
    if model.frozen_encoder and engine.state_sha256(model.backbone.state_dict()) != model.initial_encoder_state_sha256:
        raise RuntimeError("Frozen encoder parameters or buffers changed")
    save_torch(path, {
        "schema": SCHEMA, "config_sha256": config["_config_sha256"],
        "epoch": epoch, "step": step, "model": engine.cpu_state_dict(model),
        "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
        "rng_state": engine.capture_rng_state(), "validation_completed": validation_completed,
        "initialization": model.initialization, "data_identity": config.get("_data_identity"),
        "model_contract": model_contract(config),
    })


def load_checkpoint(path, config, *, resume=False, model=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        if resume or model is None:
            raise ValueError("Convert historical checkpoints explicitly before resuming")
        from .legacy import evaluation_checkpoint
        return evaluation_checkpoint(payload, model, path)
    if payload.get("model_contract") != model_contract(config):
        raise ValueError("Checkpoint backbone/readout/decoder contract differs")
    if resume and payload.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("Checkpoint recipe differs; convert historical checkpoints explicitly")
    if resume and payload.get("data_identity") != config.get("_data_identity"):
        raise ValueError("Checkpoint data changed")
    required = {"model", "epoch", "step"}
    if resume:
        required.update({"optimizer", "lr_scheduler", "rng_state", "validation_completed"})
    if missing := required.difference(payload):
        raise ValueError(f"Incomplete checkpoint state: {sorted(missing)}")
    return payload


def validate_resume_frontier(output, resume, payload, *, steps_per_epoch):
    completed = int(payload["epoch"]) + 1
    if payload["step"] != completed * steps_per_epoch:
        raise ValueError("Resume step count differs from the dataset/batch contract")
    if payload["lr_scheduler"].get("last_epoch") != completed:
        raise ValueError("Resume scheduler epoch differs from the completed epoch")
    latest = output / "checkpoints/checkpoint.pth"
    if latest.exists() and engine.sha256_file(latest) != engine.sha256_file(resume):
        raise ValueError("A stale resume would overwrite a newer checkpoint; use a fresh output directory")
    metrics = output / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            if row.get("epoch", 0) > completed or row.get("step", 0) > payload["step"]:
                raise ValueError("Training metrics extend beyond the resume checkpoint")
    return completed


def train(model, config, paths, device, *, resume=None):
    official = engine.protocol_functions()
    output = paths["output_root"]
    latest = output / "checkpoints/checkpoint.pth"
    if latest.exists() and resume is None:
        raise FileExistsError("A checkpoint exists; supply --resume explicitly")
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["protocol"]["lr_drop"])
    train_loader, val_loader = engine.make_loaders(config, paths, official)
    start, step, payload = 0, 0, None
    if resume:
        payload = load_checkpoint(resume, config, resume=True)
        if payload.get("initialization") != model.initialization:
            raise ValueError("Resume initialization source or decoder template changed")
        expected_epochs = validate_resume_frontier(output, resume, payload, steps_per_epoch=len(train_loader))
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["lr_scheduler"])
        engine.restore_rng_state(payload["rng_state"])
        start, step = expected_epochs, int(payload["step"])
    elif (output / "metrics.jsonl").exists():
        raise FileExistsError("Training metrics exist without an explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)

    def log(row):
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row), flush=True)

    def validate(epoch):
        metrics = engine.evaluate(model, val_loader, config, paths, official, device, epoch + 1)
        # The saved RNG frontier includes validation. An interrupted validation
        # is replayed from the checkpoint published at the end of the epoch.
        save_checkpoint(latest, model, optimizer, scheduler, epoch, step, config, validation_completed=True)
        save_checkpoint(output / f"checkpoints/checkpoint{epoch + 1:04d}.pth", model, optimizer, scheduler, epoch, step, config, validation_completed=True)
        log({"split": "val", **metrics})

    every = int(config["protocol"]["val_epochs"])
    epochs = int(config["protocol"]["epochs"])
    if payload and (start % every == 0 or start == epochs) and not payload["validation_completed"]:
        validate(start - 1)
    if start >= epochs:
        write_json(output / "status.json", {"completed_epochs": start, "steps": step, "complete": True})
    for epoch in range(start, epochs):
        model.train()
        train_loader.generator.manual_seed(engine.paired_training_seed(config["seed"], epoch, 0))
        rows, started = [], time.monotonic()
        for index, batch in enumerate(train_loader, 1):
            engine.set_seed(engine.paired_training_seed(config["seed"], epoch, index))
            rows.append(engine.train_batch(model, optimizer, batch, config, official, device))
            step += 1
            if index == 1 or index % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "batch": index, "step": step, **rows[-1]}), flush=True)
        scheduler.step()
        save_checkpoint(latest, model, optimizer, scheduler, epoch, step, config)
        if (epoch + 1) % 20 == 0 or epoch + 1 in config["protocol"]["lr_drop"]:
            save_checkpoint(output / f"checkpoints/checkpoint{epoch + 1:04d}.pth", model, optimizer, scheduler, epoch, step, config)
        log({"split": "train", "epoch": epoch + 1, "step": step,
             **{key: sum(row[key] for row in rows) / len(rows) for key in ("loss", "loss_bce", "loss_dice")},
             "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
             "seconds": time.monotonic() - started})
        if (epoch + 1) % every == 0 or epoch + 1 == epochs:
            validate(epoch)
        write_json(output / "status.json", {"completed_epochs": epoch + 1, "steps": step, "complete": epoch + 1 == epochs})
