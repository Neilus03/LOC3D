"""Full ScanNet Mask3D evaluation, independent of training loaders/optimizers."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from delimit3d.runtime.records import file_sha256, write_json
from delimit3d.tasks.contracts import model_contract
from delimit3d.tasks.data_identity import mask3d_data_identity
from .config import parse_granularities, set_seed
from .data.batched_scene_batch import collate_multi_scene_samples
from .engine import _evaluate, _make_dataset
from .factory import build_model


def evaluate(config, checkpoint, device):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Evaluate Mask3D with one worker; its validation protocol uses one scene at a time")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native Mask3D backbones require a CUDA device")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    path = output / f"evaluation_epoch_{payload['epoch']:04d}.json"
    if path.exists():
        raise FileExistsError("Choose a fresh evaluation output directory")
    set_seed(int(config["experiment"]["seed"]))
    config["_data_identity"] = mask3d_data_identity(config, evaluation=True)
    model, initialization = build_model(config, device)
    if payload.get("model_contract") is not None and payload["model_contract"] != model_contract(config):
        raise ValueError("Checkpoint model/readout contract differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    dataset = _make_dataset(config, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
        num_workers=int(config["train"]["num_workers"]), pin_memory=True,
        collate_fn=collate_multi_scene_samples)
    settings = config["eval"]
    metrics = _evaluate(model, loader, granularity=parse_granularities(config["data"])[0],
        class_aware=config["model"]["architecture"] == "litept_mask3d_class_aware_scannet20", device=device,
        report_dbscan=bool(settings["report_dbscan"]), dbscan_eps=float(settings["dbscan_eps"]),
        dbscan_min_samples=int(settings["dbscan_min_samples"]), topk_per_image=settings["topk_per_image"],
        backbone_batchnorm_policy=settings["backbone_batchnorm_policy"])
    result = {"epoch": payload["epoch"], "checkpoint_sha256": file_sha256(checkpoint),
        "evaluation_config_sha256": config["_config_sha256"], "model_contract": model_contract(config),
        "data_identity": config["_data_identity"], "initialization": initialization,
        "normalization_policy": settings["backbone_batchnorm_policy"], "optimizer_steps": 0, "metrics": metrics}
    write_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"data_identity", "initialization"}}, indent=2))
