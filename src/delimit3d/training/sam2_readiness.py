"""Validate the retained historical SAM2 short-tail recovery contract."""
import json
import math
from delimit3d.data.sam2_source import SOURCE_CELL
from delimit3d.runtime.records import write_json
from delimit3d.training.contracts import sha256_file
from .sam2_rank_balance import ROTATING, padding_ranks


def verify_smoke(root, *, world_size=3):
    import torch
    if world_size not in (3, 4):
        raise ValueError("Euler S* requires three or four GPU ranks")
    batch_size = 12 // world_size
    epoch_global_visits = world_size * (batch_size + 1)
    expected_padding = 2 * len(padding_ranks(2930, world_size))
    smoke = root / "smoke"
    config = json.loads((smoke / "resolved_config.json").read_text())
    summary = json.loads((smoke / "run_summary.json").read_text())
    if (summary.get("mechanics_passed") is not True or not summary["checks"] or not all(summary["checks"].values())
            or summary["epochs"] != 2 or summary["updates"] != 4 or config["world_size"] != world_size
            or config["litept_variant"] != "litept_s_star" or config["source_cells"] != [SOURCE_CELL]
            or config["supervised_stages"] != ["dec0"] or config["grid_size"] != 0.02
            or config["rank_balance_policy"] != ROTATING or config["rank_balance"]["padding_visits_total"] != expected_padding):
        raise RuntimeError(f"{world_size}-4090 S* smoke failed its frozen contract")
    checkpoints = [smoke / f"checkpoints/epoch_{e:04d}.pt" for e in (0, 1, 2)]
    initial, middle, final = [torch.load(p, map_location="cpu", weights_only=False) for p in checkpoints]
    paused = json.loads((smoke / "run_paused.json").read_text())
    for checkpoint, update, count in ((middle, 2, epoch_global_visits), (final, 4, 2 * epoch_global_visits)):
        if (checkpoint.get("resumable") is not True or checkpoint["update"] != update
                or len(checkpoint["rank_rng_states"]) != world_size or len(checkpoint["rank_coverage_states"]) != world_size
                or not checkpoint["optimizer_state_dict"]["state"]
                or checkpoint["cumulative_health"]["token_delivery"]["microbatch_count"] != count):
            raise RuntimeError(f"{world_size}-rank checkpoint replay/counter state is incomplete")
    if (paused.get("passed") is not True or paused["epoch"] != 1 or paused["update"] != 2
            or paused["checkpoint_sha256"] != sha256_file(checkpoints[1])
            or config["resume_parent"]["sha256"] != paused["checkpoint_sha256"]):
        raise RuntimeError(f"{world_size}-rank partial-tail checkpoint was not exactly resumed")
    if not all(torch.isfinite(t).all() for t in final["model_state_dict"].values()):
        raise RuntimeError("Non-finite S* smoke state")
    changed = [k for k, t in final["model_state_dict"].items() if not torch.equal(t, initial["model_state_dict"][k])]
    if not any(k.startswith("backbone.") and k.endswith((".weight", ".bias")) for k in changed):
        raise RuntimeError("S* smoke did not update backbone parameters")
    rows = [json.loads(line) for line in (smoke / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["kind"] == "train"]
    if [r["scenes_per_rank_per_update"] for r in train] != [batch_size, 1, batch_size, 1] or not all(math.isfinite(r["loss_total"]) for r in train):
        raise RuntimeError("Smoke lacks finite full/short-tail training updates")
    report = {"passed": True, "scope": f"{world_size}_gpu_training_mechanics_not_transfer_evidence",
              "world_size": world_size, "epochs": 2, "updates": 4, "global_scene_visits": 2 * epoch_global_visits,
              "padding_visits": expected_padding, "partial_tail_resume_verified": True, "checks": summary["checks"],
              "final_checkpoint_sha256": sha256_file(checkpoints[-1]), "changed_tensor_count": len(changed)}
    write_json(smoke / "SMOKE_VERIFIED.json", report)
    return report
