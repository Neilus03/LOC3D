"""Set-prediction criterion for instance segmentation.

Pairs predicted queries to GT pseudo-instances via Hungarian matching,
then supervises matched masks (BCE + Dice) and all query scores.

All mask losses are computed **only over supervised points**.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from delimit3d.tasks.mask3d.data.target_builder import InstanceTargets

log = logging.getLogger(__name__)


# ── primitive losses ─────────────────────────────────────────────────────


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    balance_mode: str = "none",
    reduction_dim: int = -1,
) -> torch.Tensor:
    """BCE-with-logits reduced over supervised points.

    ``pos_neg_balanced`` gives foreground and background equal aggregate
    weight for each mask, regardless of the number of points in each group.
    This avoids the all-background shortcut for small instances in large
    scenes. The default preserves the legacy point-wise mean exactly.

    Parameters
    ----------
    logits  : (..., K) float — predicted mask logits for supervised points.
    targets : (..., K) float — 0/1 GT mask values for the same points.
    """
    balance_mode = str(balance_mode).strip().lower()
    if balance_mode not in {"none", "pos_neg_balanced"}:
        raise ValueError(
            "mask BCE balance_mode must be 'none' or 'pos_neg_balanced', "
            f"got {balance_mode!r}"
        )

    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if balance_mode == "none":
        return losses.mean(dim=reduction_dim)

    positive = targets > 0.5
    negative = ~positive
    positive_count = positive.sum(dim=reduction_dim)
    negative_count = negative.sum(dim=reduction_dim)
    positive_mean = (losses * positive).sum(dim=reduction_dim) / positive_count.clamp_min(1)
    negative_mean = (losses * negative).sum(dim=reduction_dim) / negative_count.clamp_min(1)
    group_count = (positive_count > 0).to(losses.dtype) + (
        negative_count > 0
    ).to(losses.dtype)
    balanced_sum = positive_mean * (positive_count > 0) + negative_mean * (
        negative_count > 0
    )
    return balanced_sum / group_count.clamp_min(1.0)


def sigmoid_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Dice loss computed from logits over supervised points.

    Parameters
    ----------
    logits  : (K,) float
    targets : (K,) float — 0/1
    eps     : smoothing to avoid 0/0.
    """
    probs = logits.sigmoid()
    intersection = (probs * targets).sum()
    cardinality = probs.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


def soft_iou_score_target(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft mask IoU target for score supervision.

    Inputs are already sliced to the supervised point universe.  The returned
    target is detached by callers before it is used in the score loss.
    """
    pred_prob = mask_logits.sigmoid()
    gt_float = gt_masks.float()
    intersection = (pred_prob * gt_float).sum(dim=1)
    union = (pred_prob + gt_float - pred_prob * gt_float).sum(dim=1)
    return intersection / union.clamp_min(eps)


@torch.no_grad()
def unmatched_soft_iou_score_targets(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    supervision_mask: torch.Tensor,
    unmatched_mask: torch.Tensor,
    *,
    weight: float = 0.25,
    cap: float = 0.25,
    gt_chunk_size: int = 32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Small soft score targets for unmatched queries that overlap any GT."""
    unmatched_idx = unmatched_mask.nonzero(as_tuple=False).flatten()
    if unmatched_idx.numel() == 0 or gt_masks.shape[0] == 0:
        return mask_logits.new_zeros((int(unmatched_idx.numel()),))
    supervised = supervision_mask.bool()
    if int(supervised.sum().item()) == 0:
        return mask_logits.new_zeros((int(unmatched_idx.numel()),))

    pred_prob = mask_logits[unmatched_idx][:, supervised].sigmoid().float()
    gt_sup = gt_masks[:, supervised].float()
    pred_sum = pred_prob.sum(dim=1, keepdim=True)
    max_iou = pred_prob.new_zeros((pred_prob.shape[0],))
    chunk = max(int(gt_chunk_size), 1)
    for start in range(0, int(gt_sup.shape[0]), chunk):
        gt_chunk = gt_sup[start:start + chunk]
        inter = pred_prob @ gt_chunk.T
        gt_sum = gt_chunk.sum(dim=1).view(1, -1)
        union = pred_sum + gt_sum - inter
        max_iou = torch.maximum(max_iou, (inter / union.clamp_min(eps)).max(dim=1).values)
    return (max_iou * float(weight)).clamp(max=float(cap)).to(dtype=mask_logits.dtype)


def pairwise_score_ranking_loss(
    score_logits: torch.Tensor,
    score_targets: torch.Tensor,
    *,
    margin: float = 0.0,
    target_delta: float = 1e-6,
    max_pairs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank query scores according to detached score targets.

    For every pair (i, j) where target_i > target_j + target_delta, penalize
    logits that do not place i above j by the requested margin.
    """
    zero = score_logits.sum() * 0.0
    if score_logits.numel() <= 1:
        return zero, score_logits.new_tensor(0.0)

    targets = score_targets.detach()
    target_diff = targets[:, None] - targets[None, :]
    pair_mask = target_diff > float(target_delta)
    num_pairs = pair_mask.sum()
    if not bool(num_pairs.item()):
        return zero, score_logits.new_tensor(0.0)

    logit_diff = score_logits[:, None] - score_logits[None, :]
    pair_losses = F.softplus(float(margin) - logit_diff[pair_mask])
    max_pairs = int(max_pairs)
    if max_pairs > 0 and pair_losses.numel() > max_pairs:
        pair_losses = torch.topk(pair_losses, k=max_pairs, largest=True).values
    return pair_losses.mean(), score_logits.new_tensor(float(num_pairs.item()))


# ── pairwise cost matrix ────────────────────────────────────────────────


@torch.no_grad()
def build_pairwise_cost_matrix(
    mask_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    class_logits: torch.Tensor | None = None,
    class_ids: torch.Tensor | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weight: float = 0.0,
    mask_bce_balance_mode: str = "none",
) -> torch.Tensor:
    """Build [Q, M] cost matrix for Hungarian matching.

    Iterates over GT masks to avoid materializing a [Q, M, N] tensor.
    All costs are computed only over supervised points.
    """
    supervised = supervision_mask.bool()
    pred_sup = mask_logits[:, supervised]    # [Q, K]
    Q, K = pred_sup.shape
    M = gt_masks.shape[0]

    cost = torch.zeros(Q, M, device=mask_logits.device)

    for m in range(M):
        gt_m = gt_masks[m, supervised].float()  # [K]

        if bce_weight > 0:
            bce = masked_bce_with_logits(
                pred_sup,
                gt_m.unsqueeze(0).expand_as(pred_sup),
                balance_mode=mask_bce_balance_mode,
                reduction_dim=1,
            )
            cost[:, m] += bce_weight * bce

        if dice_weight > 0:
            probs = pred_sup.sigmoid()          # [Q, K]
            inter = (probs * gt_m.unsqueeze(0)).sum(dim=1)
            card = probs.sum(dim=1) + gt_m.sum()
            dice = 1.0 - (2.0 * inter + 1.0) / (card + 1.0)
            cost[:, m] += dice_weight * dice

    if (
        class_weight > 0
        and class_logits is not None
        and class_ids is not None
        and M > 0
    ):
        probs = class_logits.softmax(dim=-1)
        class_ids = class_ids.to(device=mask_logits.device, dtype=torch.long)
        cost += class_weight * (-probs[:, class_ids])

    return cost


# ── Hungarian matching ───────────────────────────────────────────────────


@torch.no_grad()
def hungarian_match(
    cost_matrix: torch.Tensor,
    *,
    context: str = "",
    sanitize_non_finite: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Run scipy Hungarian on a [Q, M] cost matrix.

    Returns (pred_indices, gt_indices) — both int arrays of length M
    (assuming Q >= M, every GT gets a match).
    """
    if cost_matrix.numel() == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    if not torch.isfinite(cost_matrix).all():
        # Cost matrices can occasionally contain NaN/Inf if upstream logits blow
        # up or targets contain unexpected values. By default we sanitize to
        # "very bad" costs so training can continue, but we log loudly.
        non_finite = ~torch.isfinite(cost_matrix)
        nf_count = int(non_finite.sum().item())
        total = int(cost_matrix.numel())
        finite_vals = cost_matrix[~non_finite]
        finite_min = float(finite_vals.min().item()) if finite_vals.numel() else float("nan")
        finite_max = float(finite_vals.max().item()) if finite_vals.numel() else float("nan")
        log.error(
            "Non-finite entries in Hungarian cost matrix%s: shape=%s non_finite=%d/%d finite_min=%s finite_max=%s",
            f" ({context})" if context else "",
            tuple(cost_matrix.shape),
            nf_count,
            total,
            f"{finite_min:.6g}" if np.isfinite(finite_min) else str(finite_min),
            f"{finite_max:.6g}" if np.isfinite(finite_max) else str(finite_max),
        )

        if sanitize_non_finite:
            # Replace NaN/Inf with a large finite value so SciPy can proceed.
            # Keep relative scale by anchoring to max finite if available.
            if finite_vals.numel():
                replacement = float(finite_vals.max().item()) + 1e6
            else:
                replacement = 1e6
            cost_matrix = torch.where(non_finite, cost_matrix.new_full((), replacement), cost_matrix)
        else:
            raise ValueError(
                f"Hungarian cost matrix contains non-finite entries{f' ({context})' if context else ''}."
            )

    cost_np = cost_matrix.detach().cpu().numpy()
    try:
        pred_idx, gt_idx = linear_sum_assignment(cost_np)
    except ValueError as e:
        # Re-log with a bit more detail so the crash is actionable.
        cm = cost_matrix
        log.error(
            "Hungarian matching failed%s: shape=%s min=%s max=%s finite=%s",
            f" ({context})" if context else "",
            tuple(cm.shape),
            f"{float(cm.min().item()):.6g}" if cm.numel() else "n/a",
            f"{float(cm.max().item()):.6g}" if cm.numel() else "n/a",
            bool(torch.isfinite(cm).all().item()) if cm.numel() else True,
        )
        raise

    return pred_idx.astype(np.int64), gt_idx.astype(np.int64)


# ── criterion ────────────────────────────────────────────────────────────


class MaskSetCriterion(nn.Module):
    """Set-prediction loss: Hungarian matching + BCE + Dice + score BCE.

    Parameters
    ----------
    bce_weight:
        Weight on per-mask BCE loss.
    dice_weight:
        Weight on per-mask Dice loss.
    score_weight:
        Weight on the objectness score BCE loss.
    cost_bce_weight / cost_dice_weight:
        Weights used inside the matching cost matrix (can differ from
        the loss weights).
    mask_bce_balance_mode:
        ``"none"`` preserves the legacy point-wise mask BCE. With
        ``"pos_neg_balanced"``, foreground and background contribute equally
        to both Hungarian assignment and matched-mask supervision.
    score_target_mode:
        ``"binary"`` keeps the legacy objectness target: matched queries are 1,
        unmatched queries are 0.  ``"iou"`` trains matched query scores to
        predict detached soft mask IoU against their assigned target.
    score_loss_balance_mode:
        ``"none"`` keeps the legacy score BCE reduction over all queries.
        ``"pos_neg_balanced"`` averages matched and unmatched query BCE terms
        separately before applying ``score_pos_weight`` / ``score_neg_weight``.
    score_unmatched_target_mode:
        ``"zero"`` keeps unmatched query targets at 0. ``"soft_iou"`` assigns a
        small detached IoU-proportional target to unmatched duplicate/partial
        proposals, reducing the pressure to push every non-Hungarian query to 0.
    score_rank_weight:
        Optional pairwise ranking loss weight. When positive, query scores are
        trained to rank larger detached score targets above smaller ones.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        score_weight: float = 0.5,
        cost_bce_weight: float = 1.0,
        cost_dice_weight: float = 1.0,
        class_weight: float = 0.0,
        no_object_weight: float = 0.1,
        cost_class_weight: float = 0.0,
        mask_bce_balance_mode: str = "none",
        score_target_mode: str = "binary",
        score_loss_balance_mode: str = "none",
        score_pos_weight: float = 1.0,
        score_neg_weight: float = 1.0,
        score_unmatched_target_mode: str = "zero",
        score_unmatched_iou_weight: float = 0.25,
        score_unmatched_iou_cap: float = 0.25,
        score_rank_weight: float = 0.0,
        score_rank_margin: float = 0.0,
        score_rank_target_delta: float = 1e-6,
        score_rank_max_pairs: int = 0,
    ) -> None:
        super().__init__()
        score_target_mode = str(score_target_mode).strip().lower()
        if score_target_mode not in {"binary", "iou"}:
            raise ValueError(
                f"score_target_mode must be 'binary' or 'iou', got {score_target_mode!r}"
            )
        mask_bce_balance_mode = str(mask_bce_balance_mode).strip().lower()
        if mask_bce_balance_mode not in {"none", "pos_neg_balanced"}:
            raise ValueError(
                "mask_bce_balance_mode must be 'none' or 'pos_neg_balanced', "
                f"got {mask_bce_balance_mode!r}"
            )
        score_loss_balance_mode = str(score_loss_balance_mode).strip().lower()
        if score_loss_balance_mode not in {"none", "pos_neg_balanced"}:
            raise ValueError(
                "score_loss_balance_mode must be 'none' or 'pos_neg_balanced', "
                f"got {score_loss_balance_mode!r}"
            )
        score_unmatched_target_mode = str(score_unmatched_target_mode).strip().lower()
        if score_unmatched_target_mode not in {"zero", "soft_iou"}:
            raise ValueError(
                "score_unmatched_target_mode must be 'zero' or 'soft_iou', "
                f"got {score_unmatched_target_mode!r}"
            )
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.score_weight = score_weight
        self.cost_bce_weight = cost_bce_weight
        self.cost_dice_weight = cost_dice_weight
        self.class_weight = class_weight
        self.no_object_weight = no_object_weight
        self.cost_class_weight = cost_class_weight
        self.mask_bce_balance_mode = mask_bce_balance_mode
        self.score_target_mode = score_target_mode
        self.score_loss_balance_mode = score_loss_balance_mode
        self.score_pos_weight = float(score_pos_weight)
        self.score_neg_weight = float(score_neg_weight)
        self.score_unmatched_target_mode = score_unmatched_target_mode
        self.score_unmatched_iou_weight = float(score_unmatched_iou_weight)
        self.score_unmatched_iou_cap = float(score_unmatched_iou_cap)
        self.score_rank_weight = float(score_rank_weight)
        self.score_rank_margin = float(score_rank_margin)
        self.score_rank_target_delta = float(score_rank_target_delta)
        self.score_rank_max_pairs = int(score_rank_max_pairs)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        targets: InstanceTargets,
        *,
        context: str = "",
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        pred:
            Dict from the decoder with at least ``mask_logits`` [Q, N]
            and ``score_logits`` [Q].
        targets:
            :class:`InstanceTargets` with ``gt_masks`` [M, N] and
            ``supervision_mask`` [N].

        Returns
        -------
        Dict with individual losses, total loss, and matching info.
        """
        mask_logits = pred["mask_logits"]          # [Q, N]
        score_logits = pred["score_logits"]        # [Q]
        class_logits = pred.get("class_logits")    # [Q, C+1], optional
        gt_masks = targets.gt_masks.to(mask_logits.device)
        supervision_mask = targets.supervision_mask.to(mask_logits.device)
        class_ids = (
            targets.class_ids.to(mask_logits.device)
            if targets.class_ids is not None
            else None
        )

        Q = mask_logits.shape[0]
        M = gt_masks.shape[0]
        supervised = supervision_mask.bool()

        # ── matching ──
        if not torch.isfinite(mask_logits).all():
            nf = int((~torch.isfinite(mask_logits)).sum().item())
            log.error(
                "Non-finite values in pred mask_logits%s: shape=%s non_finite=%d/%d",
                f" ({context})" if context else "",
                tuple(mask_logits.shape),
                nf,
                int(mask_logits.numel()),
            )
        if not torch.isfinite(score_logits).all():
            nf = int((~torch.isfinite(score_logits)).sum().item())
            log.error(
                "Non-finite values in pred score_logits%s: shape=%s non_finite=%d/%d",
                f" ({context})" if context else "",
                tuple(score_logits.shape),
                nf,
                int(score_logits.numel()),
            )
        if not torch.isfinite(gt_masks).all():
            nf = int((~torch.isfinite(gt_masks)).sum().item())
            log.error(
                "Non-finite values in GT masks%s: shape=%s non_finite=%d/%d",
                f" ({context})" if context else "",
                tuple(gt_masks.shape),
                nf,
                int(gt_masks.numel()),
            )

        cost_matrix = build_pairwise_cost_matrix(
            mask_logits, gt_masks, supervision_mask,
            bce_weight=self.cost_bce_weight,
            dice_weight=self.cost_dice_weight,
            class_logits=class_logits,
            class_ids=class_ids,
            class_weight=self.cost_class_weight,
            mask_bce_balance_mode=self.mask_bce_balance_mode,
        )
        pred_idx, gt_idx = hungarian_match(
            cost_matrix,
            context=context,
            sanitize_non_finite=True,
        )
        pred_idx_t = torch.from_numpy(pred_idx).to(mask_logits.device)
        gt_idx_t = torch.from_numpy(gt_idx).to(mask_logits.device)

        # ── matched mask losses (only over supervised points) ──
        if len(pred_idx) > 0:
            matched_pred = mask_logits[pred_idx_t][:, supervised]   # [M', K]
            matched_gt = gt_masks[gt_idx_t][:, supervised].float()  # [M', K]
            matched_ious = soft_iou_score_target(
                matched_pred,
                matched_gt,
            ).detach().to(dtype=score_logits.dtype)

            bce_per_pair = torch.stack([
                masked_bce_with_logits(
                    matched_pred[i],
                    matched_gt[i],
                    balance_mode=self.mask_bce_balance_mode,
                )
                for i in range(len(pred_idx))
            ])
            dice_per_pair = torch.stack([
                sigmoid_dice_loss(matched_pred[i], matched_gt[i])
                for i in range(len(pred_idx))
            ])

            loss_mask_bce = bce_per_pair.mean()
            loss_mask_dice = dice_per_pair.mean()
        else:
            loss_mask_bce = mask_logits.sum() * 0.0
            loss_mask_dice = mask_logits.sum() * 0.0
            matched_ious = score_logits.new_zeros((0,))

        # ── score loss ──
        score_targets = torch.zeros(
            Q,
            device=score_logits.device,
            dtype=score_logits.dtype,
        )
        score_positive_mask = torch.zeros(Q, device=score_logits.device, dtype=torch.bool)
        if len(pred_idx) > 0:
            if self.score_target_mode == "binary":
                matched_score_targets = torch.ones(
                    len(pred_idx),
                    device=score_logits.device,
                    dtype=score_logits.dtype,
                )
            else:
                matched_score_targets = matched_ious
            score_targets[pred_idx_t] = matched_score_targets
            score_positive_mask[pred_idx_t] = True
        else:
            matched_score_targets = score_logits.new_zeros((0,))
        score_negative_mask = ~score_positive_mask
        if self.score_unmatched_target_mode == "soft_iou" and score_negative_mask.any():
            unmatched_score_targets = unmatched_soft_iou_score_targets(
                mask_logits,
                gt_masks,
                supervision_mask,
                score_negative_mask,
                weight=self.score_unmatched_iou_weight,
                cap=self.score_unmatched_iou_cap,
            )
            score_targets[score_negative_mask] = unmatched_score_targets
        elif score_negative_mask.any():
            unmatched_score_targets = score_targets[score_negative_mask]
        else:
            unmatched_score_targets = score_logits.new_zeros((0,))
        score_bce_per_query = F.binary_cross_entropy_with_logits(
            score_logits, score_targets, reduction="none",
        )
        zero_score_loss = score_logits.sum() * 0.0
        if score_positive_mask.any():
            score_loss_pos = score_bce_per_query[score_positive_mask].mean()
        else:
            score_loss_pos = zero_score_loss
        if score_negative_mask.any():
            score_loss_neg = score_bce_per_query[score_negative_mask].mean()
        else:
            score_loss_neg = zero_score_loss
        if self.score_loss_balance_mode == "none":
            loss_score = F.binary_cross_entropy_with_logits(
                score_logits, score_targets, reduction="mean",
            )
        else:
            loss_score = (
                self.score_pos_weight * score_loss_pos
                + self.score_neg_weight * score_loss_neg
            )
        if self.score_rank_weight > 0.0:
            loss_score_rank, score_rank_num_pairs = pairwise_score_ranking_loss(
                score_logits,
                score_targets,
                margin=self.score_rank_margin,
                target_delta=self.score_rank_target_delta,
                max_pairs=self.score_rank_max_pairs,
            )
        else:
            loss_score_rank = zero_score_loss
            score_rank_num_pairs = score_logits.new_tensor(0.0)
        if matched_score_targets.numel() > 0:
            score_target_mean_matched = matched_score_targets.mean()
            score_target_max_matched = matched_score_targets.max()
            score_target_min_matched = matched_score_targets.min()
            score_logits_mean_matched = score_logits[score_positive_mask].mean()
            score_prob_mean_matched = score_logits[score_positive_mask].sigmoid().mean()
        else:
            score_target_mean_matched = score_logits.new_tensor(0.0)
            score_target_max_matched = score_logits.new_tensor(0.0)
            score_target_min_matched = score_logits.new_tensor(0.0)
            score_logits_mean_matched = score_logits.new_tensor(0.0)
            score_prob_mean_matched = score_logits.new_tensor(0.0)
        if score_negative_mask.any():
            score_logits_mean_unmatched = score_logits[score_negative_mask].mean()
            score_prob_mean_unmatched = score_logits[score_negative_mask].sigmoid().mean()
            score_target_mean_unmatched = unmatched_score_targets.mean()
            score_target_max_unmatched = unmatched_score_targets.max()
            score_target_min_unmatched = unmatched_score_targets.min()
        else:
            score_logits_mean_unmatched = score_logits.new_tensor(0.0)
            score_prob_mean_unmatched = score_logits.new_tensor(0.0)
            score_target_mean_unmatched = score_logits.new_tensor(0.0)
            score_target_max_unmatched = score_logits.new_tensor(0.0)
            score_target_min_unmatched = score_logits.new_tensor(0.0)
        score_target_mean_all = (
            score_targets.mean() if score_targets.numel() else score_logits.new_tensor(0.0)
        )
        num_score_targets_positive = score_positive_mask.sum().to(
            dtype=score_logits.dtype,
        )
        num_score_targets_total = score_logits.new_tensor(float(Q))
        score_positive_query_ratio = (
            num_score_targets_positive / num_score_targets_total.clamp_min(1.0)
        )

        # ── class/no-object loss (Mask3D-style, optional) ──
        if class_logits is not None and class_ids is not None:
            num_classes_with_no_object = int(class_logits.shape[-1])
            no_object_id = num_classes_with_no_object - 1
            class_targets = torch.full(
                (Q,),
                no_object_id,
                dtype=torch.long,
                device=class_logits.device,
            )
            if len(pred_idx) > 0:
                class_targets[pred_idx_t] = class_ids[gt_idx_t]

            ce_weights = torch.ones(
                num_classes_with_no_object,
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            ce_weights[no_object_id] = float(self.no_object_weight)
            loss_class = F.cross_entropy(
                class_logits,
                class_targets,
                weight=ce_weights,
                reduction="mean",
            )
        else:
            loss_class = mask_logits.sum() * 0.0

        # ── total ──
        loss_total = (
            self.bce_weight * loss_mask_bce
            + self.dice_weight * loss_mask_dice
            + self.score_weight * loss_score
            + self.score_rank_weight * loss_score_rank
            + self.class_weight * loss_class
        )

        return {
            "loss_total": loss_total,
            "loss_mask_bce": loss_mask_bce.detach(),
            "loss_mask_dice": loss_mask_dice.detach(),
            "loss_score": loss_score.detach(),
            "loss_score_rank": loss_score_rank.detach(),
            "loss_class": loss_class.detach(),
            "score_target_mean_matched": score_target_mean_matched.detach(),
            "score_target_max_matched": score_target_max_matched.detach(),
            "score_target_min_matched": score_target_min_matched.detach(),
            "score_target_mean_unmatched": score_target_mean_unmatched.detach(),
            "score_target_max_unmatched": score_target_max_unmatched.detach(),
            "score_target_min_unmatched": score_target_min_unmatched.detach(),
            "score_target_mean_all": score_target_mean_all.detach(),
            "num_score_targets_positive": num_score_targets_positive.detach(),
            "num_score_targets_total": num_score_targets_total.detach(),
            "score_positive_query_ratio": score_positive_query_ratio.detach(),
            "score_loss_pos": score_loss_pos.detach(),
            "score_loss_neg": score_loss_neg.detach(),
            "score_loss_balance_mode": self.score_loss_balance_mode,
            "score_rank_num_pairs": score_rank_num_pairs.detach(),
            "score_logits_mean_matched": score_logits_mean_matched.detach(),
            "score_logits_mean_unmatched": score_logits_mean_unmatched.detach(),
            "score_prob_mean_matched": score_prob_mean_matched.detach(),
            "score_prob_mean_unmatched": score_prob_mean_unmatched.detach(),
            "score_target_mode": self.score_target_mode,
            "score_unmatched_target_mode": self.score_unmatched_target_mode,
            "num_matches": len(pred_idx),
            "matched_pred_indices": pred_idx,
            "matched_gt_indices": gt_idx,
            "matched_target_indices": gt_idx,
            "matched_ious": matched_ious.detach(),
            "cost_matrix_shape": tuple(cost_matrix.shape),
        }


class MultiGranCriterion(nn.Module):
    """Applies :class:`MaskSetCriterion` independently per granularity head
    and sums the losses.

    When the decoder returns ``aux_outputs`` (intermediate layer predictions),
    the same criterion is applied to each auxiliary layer with independent
    Hungarian matching, weighted by *aux_weight*.

    Parameters
    ----------
    criterion:
        A shared :class:`MaskSetCriterion` instance (same loss weights
        for every head).
    granularity_weights:
        Optional per-head loss multiplier.  Defaults to 1.0 for each.
    aux_weight:
        Multiplier on losses from intermediate decoder layers.
    """

    def __init__(
        self,
        criterion: MaskSetCriterion,
        granularity_weights: dict[str, float] | None = None,
        aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.granularity_weights = granularity_weights or {}
        self.aux_weight = aux_weight

    def _compute_heads_loss(
        self,
        heads_pred: dict,
        targets_by_granularity: dict[str, "InstanceTargets"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute per-head losses and sum them. Returns (total, per-head dict)."""
        total_loss = None
        heads_result: dict[str, Any] = {}

        for g, targets_g in targets_by_granularity.items():
            head_pred = heads_pred[g]
            ld = self.criterion(head_pred, targets_g, context=f"granularity={g}")

            w = self.granularity_weights.get(g, 1.0)
            weighted = w * ld["loss_total"]

            if total_loss is None:
                total_loss = weighted
            else:
                total_loss = total_loss + weighted

            heads_result[g] = ld

        assert total_loss is not None, "No granularity heads to compute loss for"
        return total_loss, heads_result

    def forward(
        self,
        pred: dict,
        targets_by_granularity: dict[str, "InstanceTargets"],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        pred:
            Nested dict from ``MultiHeadQueryInstanceDecoder`` with
            ``pred["heads"][g]`` containing ``mask_logits`` and
            ``score_logits`` for each granularity ``g``.
            Optionally ``pred["aux_outputs"]`` — list of intermediate-layer
            prediction dicts with the same structure.
        targets_by_granularity:
            Dict mapping granularity keys to :class:`InstanceTargets`.

        Returns
        -------
        Dict with ``loss_total`` (summed), per-head loss dicts keyed by
        granularity, per-head matching info, and optional ``loss_aux``.
        """
        final_loss, heads_result = self._compute_heads_loss(
            pred["heads"], targets_by_granularity,
        )
        result: dict[str, Any] = {"heads": heads_result, "loss_total": final_loss}

        if self.aux_weight > 0 and "aux_outputs" in pred:
            aux_loss_sum = None
            for aux_pred in pred["aux_outputs"]:
                layer_loss, _ = self._compute_heads_loss(
                    aux_pred["heads"], targets_by_granularity,
                )
                if aux_loss_sum is None:
                    aux_loss_sum = layer_loss
                else:
                    aux_loss_sum = aux_loss_sum + layer_loss

            if aux_loss_sum is not None:
                weighted_aux = self.aux_weight * aux_loss_sum
                result["loss_aux"] = weighted_aux.detach()
                result["loss_total"] = result["loss_total"] + weighted_aux

        return result


class SingleGranCriterion(nn.Module):
    """Criterion for :class:`ContinuousQueryInstanceDecoder` (single-head).

    Wraps :class:`MaskSetCriterion` and handles the flat output dict from the
    continuous decoder, including auxiliary layer losses.

    Parameters
    ----------
    criterion:
        A shared :class:`MaskSetCriterion` instance.
    aux_weight:
        Multiplier on losses from intermediate decoder layers.
    granularity_weights:
        Optional per-granularity multipliers applied to ``loss_total`` when
        *granularity_key* is passed to ``forward`` (training / eval at fixed g).
    """

    def __init__(
        self,
        criterion: MaskSetCriterion,
        aux_weight: float = 0.0,
        granularity_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.aux_weight = aux_weight
        self.granularity_weights = granularity_weights or {}

    def forward(
        self,
        pred: dict,
        targets: "InstanceTargets",
        *,
        context: str = "",
        granularity_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        pred:
            Flat dict from ``ContinuousQueryInstanceDecoder`` with
            ``mask_logits`` [Q, N], ``score_logits`` [Q], and optionally
            ``aux_outputs`` — list of intermediate-layer prediction dicts.
        targets:
            :class:`InstanceTargets` for the sampled granularity.
        granularity_key:
            If set (e.g. ``"g05"``), scales ``loss_total`` by
            ``granularity_weights[granularity_key]`` (default 1.0).

        Returns
        -------
        Dict with ``loss_total``, individual losses, matching info, and
        optional ``loss_aux``.  When granularity weighting is applied,
        ``loss_total_unweighted`` is the combined loss before scaling (for
        logging / per-head plots); ``loss_total`` is the scaled tensor used for
        backprop.
        """
        result = self.criterion(pred, targets, context=context)

        if self.aux_weight > 0 and "aux_outputs" in pred:
            aux_loss_sum = None
            for aux_idx, aux_pred in enumerate(pred["aux_outputs"]):
                aux_ctx = f"{context}/aux{aux_idx}" if context else f"aux{aux_idx}"
                aux_result = self.criterion(aux_pred, targets, context=aux_ctx)
                if aux_loss_sum is None:
                    aux_loss_sum = aux_result["loss_total"]
                else:
                    aux_loss_sum = aux_loss_sum + aux_result["loss_total"]

            if aux_loss_sum is not None:
                weighted_aux = self.aux_weight * aux_loss_sum
                result["loss_aux"] = weighted_aux.detach()
                result["loss_total"] = result["loss_total"] + weighted_aux

        unweighted_total = result["loss_total"]
        w = 1.0
        if granularity_key is not None:
            w = float(self.granularity_weights.get(granularity_key, 1.0))
            result["loss_total_unweighted"] = unweighted_total.detach()

        result["loss_total"] = unweighted_total * w

        return result
