"""Resumable one-GPU fixed-nine training, with immutable shared visit plans."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from delimit3d.data.granularity_ablation import load_scene
from delimit3d.data.granularity_pf69 import atomic_json, build_visit, cached_plan
from delimit3d.training.contracts import stable_seed
from delimit3d.training.granularity_ablation import ARMS, GRANULARITIES, digest, file_record, lr_factor
from delimit3d.training.granularity_pf69_contract import require_geometry, validate_contract
from delimit3d.training.granularity_pf69_model import GranularityModel, tensor_state_hash


def identity(value):
    return value


def initialize_worker(_):
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")


class SharedVisits(Dataset):
    def __init__(self, *, scenes, source_root, cache_root, contract_hash, arm, start_cursor=0):
        self.scenes = list(scenes)
        self.source_root, self.cache_root = Path(source_root), Path(cache_root)
        self.contract_hash, self.arm, self.start_cursor = contract_hash, arm, start_cursor
        self.permutations = {}

    def __len__(self):
        # Bounded starvation guard, far above the ~58,596 admitted visits needed.
        return max(0, 4883 * 12 * 10 - self.start_cursor)

    def __getitem__(self, offset):
        cursor = self.start_cursor + offset
        epoch, position = divmod(cursor, len(self.scenes))
        if epoch not in self.permutations:
            self.permutations = {epoch: np.random.default_rng(
                stable_seed(42, epoch, "fixed9-common-scene-order")).permutation(len(self.scenes))}
        scene_id = self.scenes[int(self.permutations[epoch][position])]
        scene = load_scene(self.source_root, scene_id)
        plan = cached_plan(scene, epoch=epoch, cache_root=self.cache_root, contract_hash=self.contract_hash)
        base = {"candidate_cursor": cursor, "admitted": plan["admitted"], "pool_counts": plan["counts"]}
        return base | (build_visit(scene, plan, arm=self.arm) if plan["admitted"] else
                       {"scene_id": scene_id, "epoch": epoch, "plan_sha256": digest(plan)})


def atomic_checkpoint(path, payload):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def code_manifest(root):
    # Keep a runtime inventory of the deployed source used by the model, loss
    # and sampler. Training never modifies this isolated snapshot.
    return [file_record(path) for path in sorted((root / "src/delimit3d").rglob("*.py"))]


def checkpoint_payload(*, model, optimizer, update, cursor, contract_hash, geometry_hash,
                       code_hash, initial_hashes, pool_totals, trace_hash):
    backbone = model.backbone.state_dict()
    return {"schema_version": "delimit3d_fixed9_pretrain_checkpoint/v1",
            "arm": model.arm, "contract_sha256": contract_hash, "geometry_sha256": geometry_hash,
            "code_sha256": code_hash, "update": update, "nominal_epoch": update * 12 / 2930,
            "candidate_cursor": cursor, "model_state_dict": model.state_dict(),
            "backbone_state_dict": backbone,
            "state_dict": {"backbone." + key.removeprefix("model."): value for key, value in backbone.items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state": {"update": update, "total_updates": 4883, "warmup_updates": 244,
                                "factor": lr_factor(update) if update else 0.},
            "initialization_hashes": initial_hashes, "pool_totals": pool_totals,
            "visit_trace_sha256": trace_hash, "public_checkpoint_used": False,
            "normalization": "local_batchnorm", "bn_policy": "raw_training_checkpoint",
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}


def verify_resume(payload, *, contract_hash, geometry_hash, code_hash, arm):
    expected = {"contract_sha256": contract_hash, "geometry_sha256": geometry_hash,
                "code_sha256": code_hash, "arm": arm}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Resume contract, code, geometry or arm mismatch")
    update = payload["update"]
    if not isinstance(update, int) or not 0 <= update <= 4883:
        raise ValueError("Invalid resume update")
    if payload["scheduler_state"] != {"update": update, "total_updates": 4883, "warmup_updates": 244,
                                     "factor": lr_factor(update) if update else 0.}:
        raise ValueError("Resume scheduler horizon/factor mismatch")
    if payload["candidate_cursor"] < update * 12:
        raise ValueError("Resume cursor predates consumed scene visits")
    for group, rate in zip(payload["optimizer_state_dict"]["param_groups"], (1e-4, 3e-3), strict=True):
        expected_rate = rate * lr_factor(update) if update else rate
        if not math.isclose(group["lr"], expected_rate, rel_tol=1e-12):
            raise ValueError("Resume optimizer learning-rate mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "geometry", "source-root", "cache", "litept-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.workers <= 4:
        raise ValueError("Use zero to four input workers per arm")
    contract, geometry = (json.loads(path.read_text()) for path in (args.contract, args.geometry))
    validate_contract(contract)
    require_geometry(contract, geometry)
    if socket.gethostname().split(".")[0] != "pf-pc69":
        raise RuntimeError("This execution contract is restricted to pf-pc69")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU to each arm")
    if "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected the approved RTX 4090")
    contract_hash, geometry_hash = digest(contract), file_record(args.geometry)["sha256"]
    sources = code_manifest(Path(__file__).resolve().parents[3])
    code_hash = digest(sources)
    if args.mode == "train":
        if args.smoke_summary is None:
            raise ValueError("All-six-arm CUDA smoke summary is mandatory")
        summary = json.loads(args.smoke_summary.read_text())
        if (summary.get("passed") is not True or summary.get("contract_sha256") != contract_hash
                or summary.get("geometry_sha256") != geometry_hash or summary.get("code_sha256") != code_hash
                or set(summary.get("arms", {})) != set(ARMS)):
            raise ValueError("Missing/mismatched six-arm GPU smoke gate")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise FileExistsError("Refusing to overwrite an existing arm directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    # Packets contain many independent CSR tensors; avoid one inherited file
    # descriptor per storage exhausting the host's per-process descriptor cap.
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = GranularityModel(litept_root=args.litept_root, arm=args.arm).cuda().train()
    initial_hashes = model.initialization_hashes()
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4}, {"params": head_params, "lr": 3e-3}], weight_decay=.05)
    update, cursor = 0, 0
    pool_totals = {g: {"common": 0, "contrastive_multi": 0} for g in GRANULARITIES}
    trace_hash = hashlib.sha256(b"fixed9-visit-trace/v1").hexdigest()
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        verify_resume(payload, contract_hash=contract_hash, geometry_hash=geometry_hash, code_hash=code_hash, arm=args.arm)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        update, cursor = payload["update"], payload["candidate_cursor"]
        pool_totals, trace_hash = payload["pool_totals"], payload["visit_trace_sha256"]
        if payload["initialization_hashes"] != initial_hashes:
            raise ValueError("Fresh initialization does not match resumed experiment")
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    atomic_json(args.output / "execution.json", {
        "contract_sha256": contract_hash, "geometry_sha256": geometry_hash, "code_sha256": code_hash,
        "code": sources, "arm": args.arm, "mode": args.mode, "pid": os.getpid(),
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "initialization_hashes": initial_hashes, "precision": "fp32",
        "normalization": "local_batchnorm", "scenes_per_update": 12,
        "started_utc": datetime.now(timezone.utc).isoformat()})
    dataset = SharedVisits(scenes=geometry["admitted_scene_ids"], source_root=args.source_root,
                           cache_root=args.cache, contract_hash=contract_hash, arm=args.arm, start_cursor=cursor)
    loader_args = {"num_workers": args.workers, "batch_size": None, "collate_fn": identity,
                   "worker_init_fn": initialize_worker}
    if args.workers:
        loader_args.update(multiprocessing_context="spawn", prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(dataset, **loader_args)
    iterator = iter(loader)
    final_update = 8 if args.mode == "smoke" else 4883
    snapshots = {0, 1220, 2441, 4883, *(math.ceil(epoch * 2930 / 12) for epoch in range(1, 21))}
    started = time.monotonic()
    all_gradients_finite = True
    initial_backbone_hash = initial_hashes["backbone"]
    first_trace = []
    hard_packet = None

    def save_checkpoint(tag):
        payload = checkpoint_payload(model=model, optimizer=optimizer, update=update, cursor=cursor,
            contract_hash=contract_hash, geometry_hash=geometry_hash, code_hash=code_hash,
            initial_hashes=initial_hashes, pool_totals=pool_totals, trace_hash=trace_hash)
        atomic_checkpoint(args.output / tag, payload)
        atomic_checkpoint(args.output / "latest.pt", payload)

    if update == 0:
        save_checkpoint("checkpoint_u000000.pt")
    with (args.output / "metrics.jsonl").open("a", buffering=1) as metrics_file:
        while update < final_update:
            next_update = update + 1
            for group, rate in zip(optimizer.param_groups, (1e-4, 3e-3), strict=True):
                group["lr"] = rate * lr_factor(next_update)
            optimizer.zero_grad(set_to_none=True)
            records, visits, rejected = [], [], 0
            while len(records) < 12:
                packet = next(iterator)
                cursor = packet["candidate_cursor"] + 1
                for g in GRANULARITIES:
                    for key in ("common", "contrastive_multi"):
                        pool_totals[g][key] += packet["pool_counts"][g][key]
                if not packet["admitted"]:
                    rejected += 1
                    continue
                if hard_packet is None and any(group["tokens"].num_feature_hard_negatives > 0
                                               for group in packet["groups"]):
                    hard_packet = packet
                torch.manual_seed(stable_seed(42, packet["scene_id"], packet["epoch"], "backbone-forward"))
                loss, record = model(packet, update=next_update,
                    telemetry=len(records) == 0 and next_update in (1, 8, 1220, 2441, 4883))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite loss; no update or arm-specific recovery applied")
                (loss / 12.).backward()
                records.append(record)
                visit = {"scene": packet["scene_id"], "epoch": packet["epoch"], "plan": packet["plan_sha256"]}
                visits.append(visit)
                trace_hash = digest({"previous": trace_hash, "visit": visit})
                if next_update == 1:
                    first_trace.append(visit)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            all_gradients_finite &= math.isfinite(float(gradient_norm))
            optimizer.step()
            update = next_update
            retention = {g: pool_totals[g]["common"] / max(1, pool_totals[g]["contrastive_multi"])
                         for g in GRANULARITIES}
            if update >= 244 and min(retention.values()) < .9:
                raise RuntimeError("Cumulative common proposal retention fell below the unchanged 90% gate")
            row = {"update": update, "nominal_epoch": update * 12 / 2930, "candidate_cursor": cursor,
                   "arm": args.arm, "visits": visits, "visit_trace_sha256": trace_hash,
                   "rejected_common_visits": rejected, "retention": retention,
                   "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
                   "gradient_norm_before_clip": float(gradient_norm),
                   "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                   "elapsed_seconds": time.monotonic() - started,
                   "loss": sum(r["loss"] for r in records) / 12,
                   "contrastive": sum(r["contrastive"] for r in records) / 12,
                   "mask": sum(r["mask"] for r in records) / 12,
                   "vicreg": sum(r["vicreg"] for r in records) / 12,
                   "scene_records": records}
            metrics_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps({key: value for key, value in row.items()
                              if key not in ("scene_records", "visits")}), flush=True)
            if (args.mode == "train" and update in snapshots) or update == final_update:
                save_checkpoint(f"checkpoint_u{update:06d}.pt")
    final_hash = tensor_state_hash(model.backbone.state_dict())
    final_parameter_hash = tensor_state_hash(dict(model.backbone.named_parameters()))
    if (final_hash == initial_backbone_hash or not all_gradients_finite
            or final_parameter_hash == initial_hashes["backbone_parameters"]):
        raise RuntimeError("Backbone did not update with finite gradients")
    if args.mode == "smoke":
        # Exercise the post-warmup miner too; this extra diagnostic update is
        # discarded. Production always starts again from the shared random init.
        optimizer.zero_grad(set_to_none=True)
        if hard_packet is None:
            raise RuntimeError("Smoke did not contain any feature-hard-negative candidate")
        loss, hard_record = model(hard_packet, update=245)
        if hard_record["feature_hard_negatives_max"] <= 0:
            raise RuntimeError("The post-warmup miner was not exercised")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Post-warmup feature-hard smoke failed")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        marker = "SMOKE_COMPLETE.json"
        extra = {"feature_hard_check": hard_record, "first_update_visits": first_trace,
                 "first_update_visits_sha256": digest(first_trace), "mechanics_passed": True}
    else:
        marker, extra = "COMPLETE.json", {}
    atomic_json(args.output / marker, {
        "passed": True, "arm": args.arm, "update": update, "contract_sha256": contract_hash,
        "geometry_sha256": geometry_hash, "code_sha256": code_hash, "initialization_hashes": initial_hashes,
        "final_backbone_sha256": final_hash, "visit_trace_sha256": trace_hash,
        "final_backbone_parameter_sha256": final_parameter_hash,
        "checkpoint": file_record(args.output / f"checkpoint_u{update:06d}.pt"),
        "finished_utc": datetime.now(timezone.utc).isoformat(), **extra})


if __name__ == "__main__":
    main()
