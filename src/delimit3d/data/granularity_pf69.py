"""Shared, label-free eligibility and exact fixed-nine proposal plans.

Every arm gets the same admitted scene visits. A's losses/negative exclusions
see only its own granularity; B/C/D share a multigranular proposal/negative plan.
Eligibility is independent of learned features and checks both routing maps.
"""
from __future__ import annotations

from dataclasses import fields, replace
import fcntl
import json
from pathlib import Path
import os

import numpy as np
import torch

from delimit3d.data.granularity_ablation import AblationScene, augmented_scene, load_scene
from delimit3d.data.contrastive_sampler_v2 import (
    FrameProposalCatalog, _catalog_preselection_context, _frame_relation_stage_support,
    _proposal_preselection_evidence, derive_litept_hierarchy_ancestry,
)
from delimit3d.data.partfield_contrastive import (
    ContrastiveTripletBatch, sample_safe_overlapping_proposal_triplets,
    token_positive_pair_payload_digest,
)
from delimit3d.losses.litept_hierarchy_supervision import (
    _prepare_hierarchy_membership, _prepare_hierarchy_scene,
    build_token_pure_dec0_contrastive_batch,
)
from delimit3d.training.contracts import stable_seed
from delimit3d.training.granularity_ablation import GRANULARITIES, ROUTES, STAGES, digest


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_catalogs(scene: AblationScene):
    return tuple(FrameProposalCatalog(
        "2d/" + item.granularity, item.frame_id, torch.from_numpy(item.visible).long(),
        torch.from_numpy(item.offsets).long(), torch.from_numpy(item.members).long())
        for item in scene.catalogs)


def group_seed(scene_id, epoch, catalog, proposal_index):
    return stable_seed(42, scene_id, epoch, catalog.cell, catalog.frame_id,
                       proposal_index, "fixed9-proposal")


def pair_generator(seed):
    return torch.Generator(device="cpu").manual_seed(stable_seed(seed, "positive-pairs"))


def raw_batch(catalog, index, pairs, *, points, relation_catalogs, seed):
    """Use the existing authoritative-native-pair fast path, one mask per batch."""
    sizes = torch.cat([item.proposal_offsets[1:] - item.proposal_offsets[:-1]
                       for item in relation_catalogs])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    members = torch.cat([item.proposal_point_indices for item in relation_catalogs])
    pair_offsets = torch.tensor([0, len(pairs)], dtype=torch.long)
    labels = torch.full((len(pairs),), index, dtype=torch.long)
    pair_occurrences = torch.zeros(len(pairs), dtype=torch.long)
    payload_digest = token_positive_pair_payload_digest(
        pair_offsets, pairs, proposal_labels=labels, pair_proposal_indices=pair_occurrences)
    batch, _ = sample_safe_overlapping_proposal_triplets(
        num_scene_points=len(points), points=points,
        visible_indices=catalog.visible_indices, proposal_offsets=catalog.proposal_offsets,
        proposal_point_indices=catalog.proposal_point_indices,
        selected_proposal_indices=torch.tensor([index]), positive_pairs_per_proposal=64,
        num_uniform_negatives=256, num_spatial_hard_negatives=128, spatial_candidate_pool=1024,
        num_feature_hard_negatives=128, feature_candidate_pool=1024,
        relation_proposal_offsets=offsets, relation_proposal_point_indices=members,
        require_negative_proposal_membership=True, defer_token_loss=True,
        generator=torch.Generator().manual_seed(stable_seed(seed, "negative-candidates")),
        token_positive_pair_offsets=pair_offsets, token_positive_pair_indices=pairs,
        token_positive_pair_digest=payload_digest)
    return batch


def two_fold_support(prepared, *, stages):
    """Exact soft-occupancy validity predicate used by the existing BCE/Dice loss.

Mixed coarse tokens are valid. Confidence-weighted foreground/background mass,
not the presence of pure foreground at a coarse level, determines usability.
"""
    if len(prepared.partitions) != 1:
        raise ValueError("Fixed-nine heads operate on one proposal at a time")
    partition = prepared.partitions[0]
    if partition["pure_dec0_ids"].numel() < 2:
        return False
    for stage in stages:
        data = prepared.stages[stage]
        denominator = data.all_counts[data.eligible_tokens].clamp_min(1.)
        negative = data.eligible_counts.float() - data.full_counts_by_proposal[0]
        if bool((negative < 0).any()):
            raise ValueError("Negative known support is inconsistent")
        negative_mass = (negative / denominator).sum()
        for fold in (0, 1):
            if not bool((partition["token_fold"] == fold).any()):
                return False
            foreground = data.pure_fold_counts_by_proposal[0][1 - fold]
            if not bool(((foreground / denominator).sum() > 1e-6) & (negative_mass > 1e-6)):
                return False
    return True


def geometry_fingerprint(ancestry):
    import hashlib
    value = hashlib.sha256()
    for stage in ("dec0", *STAGES):
        value.update(ancestry.point_to_stage[stage].numpy().tobytes())
    return value.hexdigest()


def plan_scene(scene, *, epoch=0):
    points_np, _, _ = augmented_scene(scene, epoch=epoch)
    points = torch.from_numpy(points_np)
    ancestry = derive_litept_hierarchy_ancestry(points, grid_size=.05)
    catalogs = cpu_catalogs(scene)
    by_frame = {}
    for catalog in catalogs:
        by_frame.setdefault(catalog.frame_id, []).append(catalog)
    hierarchy_scene = _prepare_hierarchy_scene(
        point_maps=ancestry.point_to_stage,
        dec0_token_count=len(ancestry.stage_grids["dec0"]),
        stage_token_counts={stage: len(ancestry.stage_grids[stage]) for stage in STAGES})
    pools = {g: [] for g in GRANULARITIES}
    counts = {g: {"raw": 0, "contrastive_multi": 0, "contrastive_single": 0,
                  "hierarchy_union": 0, "common": 0} for g in GRANULARITIES}
    for frame_id, siblings in by_frame.items():
        multi_support = _frame_relation_stage_support(siblings, ancestry, num_points=len(points))
        for catalog in siblings:
            granularity = catalog.cell.split("/")[1]
            support = _frame_relation_stage_support([catalog], ancestry, num_points=len(points))
            single_context = _catalog_preselection_context(
                catalog=catalog, ancestry=ancestry, frame_support=support, num_points=len(points))
            multi_context = _catalog_preselection_context(
                catalog=catalog, ancestry=ancestry, frame_support=multi_support, num_points=len(points))
            for index in range(len(catalog.proposal_offsets) - 1):
                seed = group_seed(scene.scene_id, epoch, catalog, index)
                evidence = []
                for context, frame_support in ((single_context, support), (multi_context, multi_support)):
                    evidence.append(_proposal_preselection_evidence(
                        catalog=catalog, proposal_index=index, granularity=granularity,
                        ancestry=ancestry, frame_support=frame_support,
                        require_negative_proposal_membership=True, require_hierarchy_routes=False,
                        catalog_context=context, sampled_positive_pair_count=64,
                        sampled_positive_pair_generator=pair_generator(seed)))
                single, multi = evidence
                count = counts[granularity]
                count["raw"] += 1
                count["contrastive_single"] += int(single["eligible"])
                count["contrastive_multi"] += int(multi["eligible"])
                if not multi["eligible"]:
                    continue
                pairs = multi["token_positive_pairs"]
                batch = raw_batch(catalog, index, pairs, points=points,
                                  relation_catalogs=siblings, seed=seed)
                prepared = _prepare_hierarchy_membership(
                    point_maps=ancestry.point_to_stage, batch=batch, group_seed=seed,
                    dec0_token_count=len(ancestry.stage_grids["dec0"]), scene=hierarchy_scene)
                stages = sorted(set(ROUTES["C"][granularity]) | set(ROUTES["D"][granularity]))
                hierarchy_valid = two_fold_support(prepared, stages=stages)
                count["hierarchy_union"] += int(hierarchy_valid)
                if not hierarchy_valid or not single["eligible"]:
                    continue
                if not torch.equal(pairs, single["token_positive_pairs"]):
                    raise RuntimeError("Single/multi contexts disagree on the pure positive-pair plan")
                count["common"] += 1
                pools[granularity].append({"frame": frame_id, "index": index,
                                           "seed": seed, "pairs": pairs.tolist()})
    admitted = all(len(pools[g]) >= 9 for g in GRANULARITIES)
    selected = {}
    if admitted:
        for g in GRANULARITIES:
            pool = sorted(pools[g], key=lambda row: (row["frame"], row["index"]))
            order = np.random.default_rng(stable_seed(42, scene.scene_id, epoch, g, "fixed9-selection")).permutation(len(pool))
            selected[g] = [pool[int(i)] for i in order[:9]]
    return {"schema": "fixed9_pf69_scene_plan/v1", "scene_id": scene.scene_id, "epoch": epoch,
            "admitted": admitted, "counts": counts, "selected": selected,
            "geometry_sha256": geometry_fingerprint(ancestry),
            "source_hashes": {Path(item["path"]).name: item["sha256"] for item in scene.artifacts}}


def cached_plan(scene, *, epoch, cache_root, contract_hash):
    directory = Path(cache_root) / contract_hash / f"epoch_{epoch:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (scene.scene_id + ".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = {**plan_scene(scene, epoch=epoch), "contract_sha256": contract_hash}
            atomic_json(path, result)
    expected_hashes = {Path(item["path"]).name: item["sha256"] for item in scene.artifacts}
    if (result["contract_sha256"] != contract_hash or result["source_hashes"] != expected_hashes
            or result["epoch"] != epoch or result["scene_id"] != scene.scene_id):
        raise ValueError("Cached visit plan/source drift")
    return result


def move_batch(batch, device):
    return replace(batch, **{field.name: getattr(batch, field.name).to(device)
                            for field in fields(batch) if torch.is_tensor(getattr(batch, field.name))})


def build_visit(scene, plan, *, arm):
    """CPU packet; candidate negatives are generated in one deterministic space."""
    if not plan["admitted"]:
        raise ValueError("Cannot build a rejected shared visit")
    points_np, colors, normals = augmented_scene(scene, epoch=plan["epoch"])
    points = torch.from_numpy(points_np)
    ancestry = derive_litept_hierarchy_ancestry(points, grid_size=.05)
    if geometry_fingerprint(ancestry) != plan["geometry_sha256"]:
        raise RuntimeError("Input augmentation/voxelization replay drift")
    catalogs = cpu_catalogs(scene)
    lookup = {(item.cell.split("/")[1], item.frame_id): item for item in catalogs}
    # The approved wrapper uses the first representative, not voxel centroids.
    # Spatial mining must see those exact same native-token coordinates.
    token_xyz = points[ancestry.representative_indices]
    granularities = ["g" + arm[1:]] if arm.startswith("A") else list(GRANULARITIES)
    quota = 9 if arm.startswith("A") else 3
    groups = []
    for granularity in granularities:
        for record in plan["selected"][granularity][:quota]:
            catalog = lookup[granularity, record["frame"]]
            relations = ([catalog] if arm.startswith("A") else
                         [item for item in catalogs if item.frame_id == record["frame"]])
            raw = raw_batch(catalog, record["index"], torch.tensor(record["pairs"], dtype=torch.long),
                            points=points, relation_catalogs=relations, seed=record["seed"])
            tokens, metadata = build_token_pure_dec0_contrastive_batch(
                raw, ancestry.point_to_stage["dec0"], token_xyz, max_positive_pairs_per_proposal=64)
            if tokens is None or tokens.num_pairs != len(record["pairs"]) or metadata["retained_proposals"] != 1:
                raise RuntimeError("Preselected contrastive proposal/pairs did not survive exact rebuild")
            groups.append({"granularity": granularity, "seed": record["seed"],
                           "identity": [granularity, record["frame"], record["index"]],
                           "raw": raw, "tokens": tokens})
    if len(groups) != 9 or len({tuple(group["identity"]) for group in groups}) != 9:
        raise RuntimeError("A fixed-nine visit must have nine distinct mask identities")
    return {"points": points, "features": torch.from_numpy(np.concatenate([colors, normals], axis=1)),
            "point_to_dec0": ancestry.point_to_stage["dec0"], "groups": groups,
            "voxel_grid": ancestry.stage_grids["dec0"],
            "representative_indices": ancestry.representative_indices,
            "point_maps": ancestry.point_to_stage,
            "scene_id": scene.scene_id, "epoch": plan["epoch"], "plan_sha256": digest(plan)}
