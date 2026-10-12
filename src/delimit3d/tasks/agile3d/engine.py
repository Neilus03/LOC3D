"""Shared AGILE3D training episodes and official SO/MO evaluation policy."""
from __future__ import annotations
import copy
import faulthandler
import hashlib
import json
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from delimit3d.evaluation.agile3d_decoder import compute_agile3d_losses, state_sha256
from delimit3d.evaluation.agile3d_minkowski_compat import _sparse_quantize, _batched_coordinates
QUANTIZATION_BACKEND = "numpy_utility_compat_v1"
from delimit3d.evaluation.backbone_batchnorm import evaluation_metadata

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_state(payload: Any, path: Path) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: checkpoint must contain a mapping")
    if isinstance(payload.get("backbone_state_dict"), Mapping):
        return payload["backbone_state_dict"]
    if isinstance(payload.get("model_state_dict"), Mapping):
        full = {
            str(key).removeprefix("module."): value
            for key, value in payload["model_state_dict"].items()
        }
        backbone = {
            key: value
            for key, value in full.items()
            if key.startswith(("backbone.", "student.backbone."))
        }
        if not backbone:
            raise ValueError(f"{path}: model_state_dict contains no backbone state")
        return backbone
    for key in ("model", "state_dict"):
        if isinstance(payload.get(key), Mapping):
            return payload[key]
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError(
        f"{path}: expected backbone_state_dict, model_state_dict, model, "
        "state_dict, or a plain tensor state dict"
    )


def _wrapper_key_candidates(key: str) -> list[str]:
    normalized = str(key).removeprefix("module.")
    candidates = [normalized]
    for prefix in ("student.backbone.", "backbone."):
        if normalized.startswith(prefix):
            candidates.append(normalized.removeprefix(prefix))
    candidates.extend(
        candidate if candidate.startswith("model.") else f"model.{candidate}"
        for candidate in list(candidates)
    )
    return list(dict.fromkeys(candidates))


def load_frozen_backbone(model: LitePTBackbone, checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = _checkpoint_state(payload, checkpoint)
    expected = model.state_dict()
    mapped: dict[str, Tensor] = {}
    source_for_target: dict[str, str] = {}
    for source_key, value in source.items():
        if not torch.is_tensor(value):
            continue
        target_key = next(
            (candidate for candidate in _wrapper_key_candidates(str(source_key)) if candidate in expected),
            None,
        )
        if target_key is None:
            continue
        if target_key in mapped and source_for_target[target_key] != str(source_key):
            raise RuntimeError(
                f"{checkpoint}: multiple source tensors map to {target_key}: "
                f"{source_for_target[target_key]} and {source_key}"
            )
        mapped[target_key] = value.detach().cpu()
        source_for_target[target_key] = str(source_key)
    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise RuntimeError(
            f"{checkpoint}: strict LitePT-S* RGBN6 load failed; "
            f"missing {len(missing)} tensors, first={missing[:8]}"
        )
    mismatched = sorted(
        key for key in expected
        if tuple(expected[key].shape) != tuple(mapped[key].shape) or expected[key].dtype != mapped[key].dtype
    )
    if mismatched:
        raise RuntimeError(
            f"{checkpoint}: LitePT tensor shape/dtype mismatch; first={mismatched[:8]}"
        )
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in mapped.values()):
        raise ValueError(f"{checkpoint}: non-finite LitePT checkpoint tensors")
    model.load_state_dict(mapped, strict=True)
    return {
        "path": str(checkpoint),
        "file_sha256": sha256_file(checkpoint),
        "payload_epoch": payload.get("epoch") if isinstance(payload, Mapping) else None,
        "source_tensor_count": len(mapped),
        "state_sha256": state_sha256(model.state_dict()),
    }


def make_dataset(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    split: str,
) -> tuple[Any, Any]:
    args = SimpleNamespace(
        scan_folder=str(paths["scan_folder"]),
        train_list=str(paths.get("train_list", "")),
        val_list=str(paths["val_list"]),
        voxel_size=float(config["protocol"]["voxel_size_m"]),
        use_normals=True,
        normal_root=str(paths["normal_root"]),
        external_feature_root=None,
        external_feature_dim=None,
    )
    return official["build_dataset"](split=split, args=args)


def unpack_batch(batch: Any, device: torch.device) -> dict[str, Any]:
    (
        coords_batch,
        raw_coords,
        features,
        labels,
        labels_full,
        inverse_maps,
        click_idx,
        scene_names,
        num_objects,
    ) = batch
    coords_batch = coords_batch.to(device=device, non_blocking=True)
    raw_coords = raw_coords.to(device=device, dtype=torch.float32, non_blocking=True)
    features = features.to(device=device, dtype=torch.float32, non_blocking=True)
    labels_device = [value.to(device=device, dtype=torch.long, non_blocking=True) for value in labels]
    lengths = torch.tensor(
        [int(value.shape[0]) for value in labels_device], device=device, dtype=torch.long
    )
    point_offsets = torch.cumsum(lengths, dim=0)
    if int(point_offsets[-1].item()) != int(raw_coords.shape[0]):
        raise RuntimeError("official AGILE3D collate lengths do not match raw coordinates")
    return {
        "grid_coord": coords_batch[:, 1:].to(dtype=torch.int32),
        "raw_coords": raw_coords,
        "features": features,
        "point_offsets": point_offsets,
        "labels": labels_device,
        "labels_full": [value.to(device=device, dtype=torch.long) for value in labels_full],
        "inverse_maps": inverse_maps,
        "click_idx": copy.deepcopy(list(click_idx)),
        "scene_names": list(scene_names),
        "num_objects": [int(value) for value in num_objects],
    }


def split_by_offsets(value: Tensor, offsets: Tensor) -> list[Tensor]:
    ends = [int(item) for item in offsets.detach().to("cpu").tolist()]
    starts = [0] + ends[:-1]
    return [value[start:end] for start, end in zip(starts, ends)]


def initialize_training_episode(
    labels: Sequence[Tensor],
    click_idx: list[dict[Any, Any]],
) -> tuple[list[Tensor], list[dict[Any, Any]], list[dict[Any, Any]]]:
    """Copy the released AGILE3D random-object training setup."""
    labels_new: list[Tensor] = []
    clicks = copy.deepcopy(click_idx)
    for index, sample_labels in enumerate(labels):
        valid_object_ids = torch.unique(sample_labels)
        valid_object_ids = valid_object_ids[valid_object_ids != -1]
        max_num_objects = int(valid_object_ids.numel())
        if max_num_objects <= 0:
            raise RuntimeError("a training scene has no valid object labels")
        object_count = int(np.random.randint(1, min(10, max_num_objects) + 1))
        selected = valid_object_ids[torch.randperm(max_num_objects, device=sample_labels.device)[:object_count]]
        remapped = torch.zeros(sample_labels.shape, device=sample_labels.device, dtype=torch.long)
        for new_id, object_id in enumerate(selected.tolist(), start=1):
            remapped[sample_labels == int(object_id)] = int(new_id)
            clicks[index][str(new_id)] = []
        clicks[index]["0"] = []
        labels_new.append(remapped)
    return labels_new, clicks, copy.deepcopy(clicks)


def initialize_evaluation_clicks(click_idx, object_count):
    """Start the official click protocol with an empty slot for every object."""
    if object_count <= 0:
        raise ValueError("AGILE3D evaluation requires a positive object count")
    clicks = {str(key): [] for key in click_idx}
    for object_id in range(object_count + 1):
        clicks.setdefault(str(object_id), [])
    return clicks, copy.deepcopy(clicks)


def run_training_click_prefix(
    model: LitePTAgile3D,
    scene_tokens: Sequence[Tensor],
    scene_xyz: Sequence[Tensor],
    labels: Sequence[Tensor],
    clicks: list[dict[Any, Any]],
    click_times: list[dict[Any, Any]],
    official: Mapping[str, Any],
) -> tuple[list[dict[Any, Any]], list[dict[Any, Any]]]:
    model.eval()
    rounds = random.randint(0, 19)
    current_round = 0
    with torch.no_grad():
        while current_round <= rounds:
            if current_round == 0:
                predictions = [torch.zeros_like(target) for target in labels]
            else:
                predictions = [
                    model.decode(tokens, xyz, clicks[index], click_times[index])["pred_masks"].argmax(dim=1)
                    for index, (tokens, xyz) in enumerate(zip(scene_tokens, scene_xyz))
                ]
            for index, (prediction, target, xyz) in enumerate(zip(predictions, labels, scene_xyz)):
                prediction = prediction.clone()
                for object_id, click_ids in clicks[index].items():
                    if click_ids:
                        click_tensor = torch.as_tensor(click_ids, device=prediction.device, dtype=torch.long)
                        prediction[click_tensor] = int(object_id)
                new_clicks, _count, _positions, new_times = official["get_simulated_clicks"](
                    prediction,
                    target,
                    xyz,
                    current_round,
                    training=True,
                )
                if new_clicks is not None:
                    clicks[index], click_times[index] = official["extend_clicks"](
                        clicks[index], click_times[index], new_clicks, new_times
                    )
            current_round += 1
    model.train()
    return clicks, click_times


def train_batch(
    model: LitePTAgile3D,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    config: Mapping[str, Any],
    official: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    unpacked = unpack_batch(batch, device)
    scene_tokens, scene_xyz = model.encode_batch(
        unpacked["grid_coord"],
        unpacked["raw_coords"],
        unpacked["features"],
        unpacked["point_offsets"],
    )
    labels, clicks, click_times = initialize_training_episode(
        unpacked["labels"], unpacked["click_idx"]
    )
    clicks, click_times = run_training_click_prefix(
        model, scene_tokens, scene_xyz, labels, clicks, click_times, official
    )

    losses: list[Tensor] = []
    loss_bce: list[Tensor] = []
    loss_dice: list[Tensor] = []
    for tokens, xyz, target, scene_clicks, scene_times in zip(
        scene_tokens, scene_xyz, labels, clicks, click_times
    ):
        output = model.decode(tokens, xyz, scene_clicks, scene_times)
        loss, details = compute_agile3d_losses(
            output,
            target,
            scene_xyz=xyz,
            clicks=scene_clicks,
            bce_weight=float(config["loss"]["bce_weight"]),
            dice_weight=float(config["loss"]["dice_weight"]),
            alpha=float(config["loss"]["alpha"]),
            beta=float(config["loss"]["beta"]),
            radius=float(config["loss"]["radius"]),
        )
        losses.append(loss)
        loss_bce.append(details["loss_bce"])
        loss_dice.append(details["loss_dice"])
    batch_loss = torch.stack(losses).mean()
    if not torch.isfinite(batch_loss):
        raise FloatingPointError(f"non-finite training loss: {batch_loss}")
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["optimizer"]["clip_norm"]), error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "loss": float(batch_loss.detach().cpu()),
        "loss_bce": float(torch.stack(loss_bce).mean().detach().cpu()),
        "loss_dice": float(torch.stack(loss_dice).mean().detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        "scenes": len(scene_tokens),
        "tokens": int(sum(int(value.shape[0]) for value in scene_tokens)),
    }


@torch.no_grad()
def _evaluate(
    model: LitePTAgile3D,
    loader: Iterable[Any],
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    output_root = paths["output_root"]
    validation_root = output_root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    result_file = validation_root / f"val_results_epoch_{int(epoch):04d}.csv"
    if result_file.exists():
        result_file = result_file.with_name(f"{result_file.stem}_attempt_{time.time_ns()}.csv")
    partial_file = result_file.with_name(f"{result_file.name}.{time.time_ns()}.partial")
    instance_counter = 0
    validation_started = time.monotonic()
    print(json.dumps({"event": "validation_start", "epoch": epoch,
                      "result_file": str(result_file)}), flush=True)
    faulthandler.dump_traceback_later(600, repeat=True)
    with partial_file.open("x") as handle:
        for batch in loader:
            set_seed(int(config["seed"]) + 10_000_019 + instance_counter)
            scene_started = time.monotonic()
            unpacked = unpack_batch(batch, device)
            scene_tokens, scene_xyz = model.encode_batch(
                unpacked["grid_coord"],
                unpacked["raw_coords"],
                unpacked["features"],
                unpacked["point_offsets"],
            )
            if len(scene_tokens) != 1:
                raise RuntimeError("official AGILE3D validation must use val_batch_size=1")
            tokens, xyz = scene_tokens[0], scene_xyz[0]
            labels = unpacked["labels"][0]
            labels_full = unpacked["labels_full"][0]
            inverse_map = torch.as_tensor(
                unpacked["inverse_maps"][0], device=device, dtype=torch.long
            )
            object_count = int(unpacked["num_objects"][0])
            clicks, click_times = initialize_evaluation_clicks(unpacked["click_idx"][0], object_count)
            scene_name = str(unpacked["scene_names"][0])
            print(json.dumps({"event": "validation_scene_start", "epoch": epoch,
                              "scene_index": instance_counter + 1, "scene": scene_name,
                              "objects": object_count, "tokens": len(tokens)}), flush=True)
            current_clicks = 0
            max_clicks = object_count * int(config["protocol"]["max_num_clicks"])
            while current_clicks <= max_clicks:
                if current_clicks == 0:
                    prediction = torch.zeros_like(labels)
                else:
                    output = model.decode(tokens, xyz, clicks, click_times)
                    prediction = output["pred_masks"].argmax(dim=1)
                    prediction = prediction.clone()
                    for object_id, click_ids in clicks.items():
                        if click_ids:
                            click_tensor = torch.as_tensor(click_ids, device=device, dtype=torch.long)
                            prediction[click_tensor] = int(object_id)

                prediction_full = prediction[inverse_map]
                iou, _ = official["mean_iou_scene"](prediction_full, labels_full)
                handle.write(
                    f"{instance_counter} "
                    f"{str(unpacked['scene_names'][0]).replace('scene', '')} "
                    f"{object_count} {current_clicks / object_count} {float(iou):.10f}\n"
                )
                # Publish every completed interaction, including a partial
                # scene if a later CUDA/runtime failure interrupts evaluation.
                handle.flush()
                faulthandler.dump_traceback_later(600, repeat=True)
                if current_clicks == 0 or current_clicks % 10 == 0:
                    print(json.dumps({"event": "validation_click", "epoch": epoch,
                                      "scene": scene_name, "clicks": current_clicks,
                                      "max_clicks": max_clicks, "iou": float(iou),
                                      "seconds": time.monotonic() - scene_started}), flush=True)
                new_clicks, _new_count, _positions, new_times = official["get_simulated_clicks"](
                    prediction,
                    labels,
                    xyz,
                    current_clicks,
                    training=False,
                )
                # Released evaluation keeps every scene in later-click
                # averages even after the voxel prediction has no errors.
                # With no new click, repeat the unchanged click state.
                if new_clicks is not None:
                    clicks, click_times = official["extend_clicks"](
                        clicks, click_times, new_clicks, new_times
                    )
                # Match the released engine: the first correction round adds
                # one click per selected object; later rounds add one click.
                current_clicks += int(object_count if current_clicks == 0 else 1)
            instance_counter += 1
            print(json.dumps({"event": "validation_scene_complete", "epoch": epoch,
                              "completed_scenes": instance_counter, "scene": scene_name,
                              "seconds": time.monotonic() - scene_started}), flush=True)
    evaluator = official["EvaluatorMO"](
        str(paths["val_list"]),
        str(partial_file),
        [0.5, 0.65, 0.8, 0.85, 0.9],
    )
    metrics = {key: float(value) for key, value in evaluator.eval_results().items()}
    metrics.update({"epoch": int(epoch), "result_file": str(result_file),
                    **evaluation_metadata(model)})
    if (getattr(model, "frozen_encoder", True)
            and state_sha256(model.backbone.state_dict()) != model.initial_encoder_state_sha256):
        raise RuntimeError("evaluation changed a frozen encoder parameter or buffer")
    os.replace(partial_file, result_file)
    print(json.dumps({"event": "validation_complete", "epoch": epoch,
                      "scenes": instance_counter,
                      "seconds": time.monotonic() - validation_started}), flush=True)
    metric_file = result_file.with_name(result_file.stem.replace("val_results", "metrics") + ".json")
    with metric_file.open("x") as handle:
        handle.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def make_loaders(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    official: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset, collate_fn = make_dataset(config, paths, official, "train")
    val_dataset, val_collate_fn = make_dataset(config, paths, official, "val")
    protocol = config["protocol"]
    train_batch_size = 1 if smoke else int(protocol["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=not smoke,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0 if smoke else int(config["runtime"]["num_workers"]),
        pin_memory=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(protocol["val_batch_size"]),
        shuffle=False,
        drop_last=False,
        collate_fn=val_collate_fn,
        num_workers=0 if smoke else int(config["runtime"]["num_workers"]),
        pin_memory=True,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 1),
    )
    return train_loader, val_loader


def paired_training_seed(seed: int, epoch: int, batch_index: int) -> int:
    """Model-dependent click paths must not shift later scene/episode draws."""
    return int(seed) + int(epoch) * 1_000_003 + int(batch_index)


def evaluate(*args, **kwargs):
    try:
        return _evaluate(*args, **kwargs)
    finally:
        faulthandler.cancel_dump_traceback_later()


def protocol_functions():
    from .protocol.datasets.InterMultiObj3DSegDataset import build
    from .protocol.evaluation.evaluator_MO import EvaluatorMO
    from .protocol.utils.seg import extend_clicks, get_simulated_clicks, mean_iou_scene
    return {"build_dataset": build, "EvaluatorMO": EvaluatorMO,
            "extend_clicks": extend_clicks, "get_simulated_clicks": get_simulated_clicks,
            "mean_iou_scene": mean_iou_scene, "minkowski_backend": "numpy_utility_compat_v1"}
