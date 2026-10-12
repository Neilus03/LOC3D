import json
from pathlib import Path

import pytest

from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule
from delimit3d.training.runner import parse_args
from delimit3d.training.sam2_rank_balance import (
    STRICT, ROTATING, balance_contract, partition_scene_ids, epoch_scene_order, padding_ranks,
)


def test_three_ranks_keep_all_canonical_scenes_and_only_one_rotating_extra_visit():
    scenes = [f"scene_{i:05d}" for i in range(2930)]
    parts = [partition_scene_ids(scenes, rank=r, world_size=3, policy=ROTATING) for r in range(3)]
    assert [len(p[0]) for p in parts] == [977, 977, 976]
    assert [p[1] for p in parts] == [977] * 3
    assert [s for part, _ in parts for s in part] == scenes
    padding = []
    for epoch in range(150):
        epoch_ids = []
        for rank, (local, visits) in enumerate(parts):
            order = epoch_scene_order(scene_count=len(local), visits=visits,
                                      rank=rank, epoch=epoch, seed=42, policy=ROTATING)
            assert sorted(order[:len(local)]) == list(range(len(local)))
            epoch_ids.extend(local[i] for i in order)
            if rank == 2:
                padding.append(local[order[-1]])
        assert len(epoch_ids) == 2931 and set(epoch_ids) == set(scenes)
    assert len(set(padding)) == 150
    contract = balance_contract(scenes, world_size=3, epochs=150, policy=ROTATING)
    assert contract["canonical_scene_visits_total"] == 439500
    assert contract["scene_visits_including_padding_total"] == 439650
    assert contract["padding_visits_total"] == 150


def test_schedule_and_resume_preserve_every_visit_including_explicit_padding():
    schedule = LogicalBatchSchedule(977, 4, 150, (25, 50, 75, 100, 125, 150))
    assert schedule.total_updates == 36642
    for epoch in schedule.flush_epochs:
        assert schedule.update_at_epoch(epoch) == epoch // 25 * 6107
        assert schedule.consumed_visits(schedule.update_at_epoch(epoch)) == epoch * 977
    full = [epoch_scene_order(scene_count=976, visits=977, rank=2, epoch=e, seed=42, policy=ROTATING)
            for e in range(150)]
    restarted = full[:25] + [epoch_scene_order(scene_count=976, visits=977, rank=2,
                            epoch=e, seed=42, policy=ROTATING) for e in range(25, 150)]
    assert restarted == full
    with pytest.raises(ValueError, match="divide evenly"):
        partition_scene_ids(list(range(2930)), rank=0, world_size=3, policy=STRICT)


def test_four_ranks_keep_all_sources_with_two_independent_rotating_padding_cycles():
    scenes = [f"scene_{i:05d}" for i in range(2930)]
    parts = [partition_scene_ids(scenes, rank=r, world_size=4, policy=ROTATING) for r in range(4)]
    assert [len(p[0]) for p in parts] == [733, 733, 732, 732]
    assert [p[1] for p in parts] == [733] * 4
    assert [s for part, _ in parts for s in part] == scenes
    assert padding_ranks(2930, 4) == (2, 3)
    cycles = {2: [], 3: []}
    for epoch in range(150):
        epoch_ids = []
        for rank, (local, visits) in enumerate(parts):
            order = epoch_scene_order(scene_count=len(local), visits=visits,
                                      rank=rank, epoch=epoch, seed=42, policy=ROTATING)
            assert sorted(order[:len(local)]) == list(range(len(local)))
            epoch_ids.extend(local[i] for i in order)
            if rank in cycles:
                cycles[rank].append(local[order[-1]])
        assert len(epoch_ids) == 2932 and set(epoch_ids) == set(scenes)
    assert all(len(set(cycle)) == 150 for cycle in cycles.values())
    assert set(cycles[2]).isdisjoint(cycles[3])
    contract = balance_contract(scenes, world_size=4, epochs=150, policy=ROTATING)
    assert contract["canonical_scene_visits_total"] == 439500
    assert contract["scene_visits_including_padding_total"] == 439800
    assert contract["padding_visits_total"] == 300
    schedule = LogicalBatchSchedule(733, 3, 150, (25, 50, 75, 100, 125, 150))
    assert schedule.total_updates == 36654
    for epoch in schedule.flush_epochs:
        assert schedule.update_at_epoch(epoch) == epoch // 25 * 6109
        assert schedule.consumed_visits(schedule.update_at_epoch(epoch)) == epoch * 733
        assert schedule.batch_size(schedule.update_at_epoch(epoch)) == 1


def test_three_rank_and_strict_partitions_keep_their_existing_exact_order():
    scenes = list(range(2930))
    for rank in range(3):
        assert partition_scene_ids(scenes, rank=rank, world_size=3, policy=ROTATING)[0] == scenes[rank * 977:(rank + 1) * 977]
    for rank in range(2):
        assert partition_scene_ids(scenes, rank=rank, world_size=2, policy=STRICT)[0] == scenes[rank * 1465:(rank + 1) * 1465]
    assert padding_ranks(2930, 3) == (2,)
    assert padding_ranks(2930, 2) == ()






def test_same_epoch_padding_advances_persisted_catalog_and_resumes():
    from test_sam2_pretraining import _persistent_visit, _scene
    from delimit3d.training.sam2_sampling import validate_saved_coverage_states
    scene, states = _scene("sam2"), {}
    _persistent_visit(scene, 0, states)
    first = json.loads(json.dumps(states))
    _persistent_visit(scene, 0, states)
    assert next(iter(states.values()))["cursor"] > next(iter(first.values()))["cursor"]
    validate_saved_coverage_states(states, start_epoch=1, seed=42)
    resumed = json.loads(json.dumps(states))
    _persistent_visit(scene, 1, states)
    _persistent_visit(scene, 1, resumed)
    assert states == resumed
    validate_saved_coverage_states(states, start_epoch=2, seed=42)


@pytest.mark.parametrize("world_size", [3, 4])
def test_gpu_smoke_verifier_requires_exact_resume_and_all_ranks(tmp_path, world_size):
    import torch
    from delimit3d.training.sam2_readiness import verify_smoke
    from delimit3d.training.contracts import sha256_file
    smoke = tmp_path / "smoke"
    (smoke / "checkpoints").mkdir(parents=True)
    base = {"model_state_dict": {"backbone.layer.weight": torch.ones(2, 2)}}
    torch.save(base, smoke / "checkpoints/epoch_0000.pt")
    epoch_visits = world_size * (12 // world_size + 1)
    final = {"model_state_dict": {"backbone.layer.weight": torch.ones(2, 2) * 2},
             "resumable": True, "update": 4, "rank_rng_states": [{} for _ in range(world_size)],
             "rank_coverage_states": [{} for _ in range(world_size)], "optimizer_state_dict": {"state": {0: {"step": 4}}},
             "cumulative_health": {"token_delivery": {"microbatch_count": 2 * epoch_visits}}}
    mid = {**final, "epoch": 1, "update": 2, "cumulative_health": {"token_delivery": {"microbatch_count": epoch_visits}}}
    torch.save(mid, smoke / "checkpoints/epoch_0001.pt")
    torch.save(final, smoke / "checkpoints/epoch_0002.pt")
    digest = sha256_file(smoke / "checkpoints/epoch_0001.pt")
    config = {"world_size": world_size, "source_cells": ["2d/sam2"], "litept_variant": "litept_s_star",
              "supervised_stages": ["dec0"], "grid_size": 0.02, "rank_balance_policy": ROTATING,
              "rank_balance": {"padding_visits_total": 2 * len(padding_ranks(2930, world_size))}, "resume_parent": {"sha256": digest}}
    (smoke / "resolved_config.json").write_text(json.dumps(config))
    (smoke / "run_summary.json").write_text(json.dumps({"mechanics_passed": True,
        "checks": {"exact": True}, "epochs": 2, "updates": 4}))
    (smoke / "run_paused.json").write_text(json.dumps({"passed": True, "epoch": 1,
                                                    "update": 2, "checkpoint_sha256": digest}))
    (smoke / "metrics.jsonl").write_text("\n".join(json.dumps({"kind": "train", "loss_total": 1,
        "scenes_per_rank_per_update": n}) for n in (12 // world_size, 1, 12 // world_size, 1)))
    assert verify_smoke(tmp_path, world_size=world_size)["partial_tail_resume_verified"] is True
    final["rank_rng_states"] = [{}, {}]
    torch.save(final, smoke / "checkpoints/epoch_0002.pt")
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_smoke(tmp_path, world_size=world_size)
