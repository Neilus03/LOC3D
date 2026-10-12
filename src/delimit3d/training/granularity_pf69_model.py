"""Balanced A/B/C/D objectives on one identical LitePT-S* backbone."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from delimit3d.data.granularity_pf69 import move_batch
from delimit3d.losses.litept_hierarchy_supervision import (
    ProposalConditionedHierarchyLoss, _prepare_hierarchy_membership, _prepare_hierarchy_scene,
    hierarchy_point_maps, native_hierarchy_vicreg_loss,
)
from delimit3d.losses.partfield_contrastive_loss import PartFieldContrastiveCriterion
from delimit3d.models.litept_wrapper import LitePTBackbone
from delimit3d.training.adaptation import ContrastiveProjectionHead
from delimit3d.training.granularity_ablation import STAGES, arm_spec, assignment_counts


def tensor_state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class BalancedHierarchyLoss(ProposalConditionedHierarchyLoss):
    """Two explicit stages per mask, no size-based fallback or dynamic routing."""

    def forward_balanced(self, *, features, point_maps, groups, routes):
        queries = self._queries_by_stage(features["dec0"])
        keys = {stage: F.normalize(self.key_projections[stage](self.key_norms[stage](features[stage])),
                                   dim=-1, eps=self.eps) for stage in STAGES}
        scene = _prepare_hierarchy_scene(
            point_maps=point_maps, dec0_token_count=len(features["dec0"]),
            stage_token_counts={stage: len(features[stage]) for stage in STAGES})
        by_granularity = {}
        assignments = Counter()
        for group in groups:
            granularity, batch, seed = group["granularity"], group["raw"], group["seed"]
            prepared = _prepare_hierarchy_membership(
                point_maps=point_maps, batch=batch, group_seed=seed,
                dec0_token_count=len(features["dec0"]), scene=scene)
            stage_losses = []
            for stage in routes[granularity]:
                record = self._one_group_stage(
                    query_tokens=queries[stage], stage_keys=keys[stage], point_maps=point_maps,
                    batch=batch, stage=stage, group_seed=seed, prepared=prepared)
                if record["loss_denominator"] != 2 or record["metrics"]["proposals_valid"] != 1:
                    raise RuntimeError("An admitted mask lost a required fold; refusing arm-specific fallback")
                stage_losses.append(record["loss_total"])
                assignments[stage] += 1
            by_granularity.setdefault(granularity, []).append(torch.stack(stage_losses).mean())
        loss = torch.stack([torch.stack(items).mean() for items in by_granularity.values()]).mean()
        return loss, dict(assignments)


class GranularityModel(nn.Module):
    def __init__(self, *, litept_root: Path, arm: str):
        super().__init__()
        self.arm = arm
        self.spec = arm_spec(arm)
        self.backbone = LitePTBackbone(
            litept_root=str(litept_root), in_channels=6, grid_size=.05,
            litept_variant="litept_s_star", voxel_reduce="representative",
            representative_sampling="first", capture_hierarchy=True,
            cache_training_voxelization=False)
        self.projector = ContrastiveProjectionHead(72, 128, 128)
        self.criterion = PartFieldContrastiveCriterion(
            temperature=.07, learnable_temperature=True, symmetric_feature_hard_mining=True)
        # Auxiliaries are constructed after all common parameters so every arm
        # has identical backbone/projector/temperature initialization.
        self.hierarchy_mask_supervisor = (
            BalancedHierarchyLoss(query_projection_policy="per_stage_dec0")
            if arm in ("C", "D") else None)

    def initialization_hashes(self):
        common = {key: value for key, value in self.state_dict().items()
                  if not key.startswith("hierarchy_mask_supervisor.")}
        return {"common": tensor_state_hash(common),
                "backbone": tensor_state_hash(self.backbone.state_dict()),
                "backbone_parameters": tensor_state_hash(dict(self.backbone.named_parameters())),
                "all": tensor_state_hash(self.state_dict())}

    def forward(self, packet, *, update, telemetry=False):
        device = next(self.parameters()).device
        # Quantize exactly once, in the audited CPU planner. FP32 division on
        # CPU/CUDA can straddle a voxel boundary even for identical coordinates.
        # The existing prequantized wrapper preserves this canonical token order.
        representatives = packet["representative_indices"]
        output = self.backbone(packet["points"][representatives].to(device),
                               packet["features"][representatives].to(device),
                               grid_coord=packet["voxel_grid"].to(device))
        if not torch.equal(output.inverse_map.cpu(), torch.arange(len(representatives))):
            raise RuntimeError("Prequantized backbone changed canonical voxel IDs")
        features = dict(zip(output.hierarchy_stage_names, output.hierarchy_tokens, strict=True))
        if set(features) != {"dec0", *STAGES}:
            raise RuntimeError("Expected every native LitePT hierarchy stage")
        point_maps = hierarchy_point_maps(packet["point_to_dec0"].to(device), output.hierarchy_parent_maps)
        if any(not torch.equal(point_maps[stage].cpu(), packet["point_maps"][stage])
               for stage in ("dec0", *STAGES)):
            raise RuntimeError("CUDA hierarchy differs from the audited CPU ancestry")
        embeddings = self.projector(features["dec0"])
        losses = {}
        groups = []
        realized_pairs = 0
        hard_counts = []
        for original in packet["groups"]:
            tokens = move_batch(original["tokens"], device)
            if update < 245:
                tokens = replace(tokens, num_feature_hard_negatives=0, feature_candidate_indices=None)
            record = self.criterion(embeddings, tokens)
            losses.setdefault(original["granularity"], []).append(record["loss_total"])
            realized_pairs += tokens.num_pairs
            hard_counts.append(tokens.num_feature_hard_negatives)
            if self.hierarchy_mask_supervisor is not None:
                groups.append({**original, "raw": move_batch(original["raw"], device)})
        contrastive = torch.stack([torch.stack(items).mean() for items in losses.values()]).mean()
        mask, vicreg = contrastive.new_zeros(()), contrastive.new_zeros(())
        assignments = {}
        if self.hierarchy_mask_supervisor is not None:
            mask, assignments = self.hierarchy_mask_supervisor.forward_balanced(
                features=features, point_maps=point_maps, groups=groups, routes=self.spec["mask_routes"])
            if assignments != assignment_counts(self.spec):
                raise RuntimeError(f"Routed assignment budget drift: {assignments}")
            vicreg = native_hierarchy_vicreg_loss(
                features, variance_target=1., covariance_weight=.04, max_tokens=4096)["loss_total"]
        auxiliary_factor = min(1., update / 244.)
        total = contrastive + auxiliary_factor * (mask + 2. * vicreg)
        metrics = {"loss": float(total.detach()), "contrastive": float(contrastive.detach()),
                   "mask": float(mask.detach()), "vicreg": float(vicreg.detach()),
                   "auxiliary_factor": auxiliary_factor, "assignments": assignments,
                   "positive_pairs": realized_pairs, "proposals": len(packet["groups"]),
                   "feature_hard_negatives_min": min(hard_counts),
                   "feature_hard_negatives_max": max(hard_counts)}
        if telemetry:
            norms = {}
            for name, objective in (("contrastive", contrastive), ("mask", mask), ("vicreg", vicreg)):
                if objective.requires_grad:
                    gradients = torch.autograd.grad(objective, list(features.values()),
                                                    retain_graph=True, allow_unused=True)
                    norms[name] = {stage: 0. if gradient is None else float(gradient.detach().norm())
                                   for stage, gradient in zip(features, gradients, strict=True)}
            metrics["native_activation_gradient_norms_not_parameter_cosines"] = norms
        return total, metrics
