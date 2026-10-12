"""Explicit, replayable rank balancing; canonical splits never change."""

from __future__ import annotations

import math

import numpy as np

from delimit3d.training.contracts import stable_seed

STRICT = "strict_even"
ROTATING = "rotating_local_padding_v1"


def partition_scene_ids(scene_ids, *, rank, world_size, policy=STRICT):
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Canonical training scenes must be nonempty and unique")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid rank/world size")
    if policy not in (STRICT, ROTATING):
        raise ValueError("Unknown rank-balancing policy")
    if policy == STRICT and len(scene_ids) % world_size:
        raise ValueError("Training split must divide evenly across ranks")
    base, remainder = divmod(len(scene_ids), world_size)
    visits = math.ceil(len(scene_ids) / world_size)
    start = rank * base + min(rank, remainder)
    count = base + int(rank < remainder)
    local = list(scene_ids[start:start + count])
    if not local or visits - len(local) > 1:
        raise ValueError("Rotating padding supports at most one missing visit per rank")
    return local, visits


def padding_ranks(scene_count, world_size):
    """Ranks needing one extra visit under the balanced contiguous partition."""
    if world_size < 1 or scene_count < world_size:
        raise ValueError("Each rank must contain a canonical training scene")
    remainder = scene_count % world_size
    return tuple(range(remainder, world_size)) if remainder else ()


def epoch_scene_order(*, scene_count, visits, rank, epoch, seed, policy=STRICT):
    if scene_count < 1 or epoch < 0 or not scene_count <= visits <= scene_count + 1:
        raise ValueError("Invalid balanced epoch dimensions")
    if policy not in (STRICT, ROTATING) or policy == STRICT and scene_count != visits:
        raise ValueError("Unbalanced epoch requires explicit rotating padding")
    order = np.random.default_rng(stable_seed(seed, rank, epoch, "scene-order")).permutation(scene_count).tolist()
    if visits > scene_count:
        # Each base scene occurs once. Only the deficit rank gets an extra
        # visit, rotating without replacement across its own training scenes.
        cycle = np.random.default_rng(stable_seed(seed, rank, "rank-padding-cycle")).permutation(scene_count)
        order.append(int(cycle[epoch % scene_count]))
    return order


def balance_contract(scene_ids, *, world_size, epochs, policy):
    parts = [partition_scene_ids(scene_ids, rank=r, world_size=world_size, policy=policy)
             for r in range(world_size)]
    visits = parts[0][1]
    return {
        "policy": policy,
        "canonical_training_scene_count": len(scene_ids),
        "rank_unique_scene_counts": [len(p[0]) for p in parts],
        "scene_visits_per_epoch_per_rank": visits,
        "padding_visits_per_epoch": visits * world_size - len(scene_ids),
        "padding_visits_total": (visits * world_size - len(scene_ids)) * epochs,
        "canonical_scene_visits_total": len(scene_ids) * epochs,
        "scene_visits_including_padding_total": visits * world_size * epochs,
        "padding_selection": "seeded_rank_local_permutation_epoch_cycle",
        "canonical_split_unchanged": True,
        "no_scene_dropped": True,
        "validation_never_used_for_padding": True,
    }
