"""Batch-normalized Mask3D set loss over ScanNet mesh segments."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from delimit3d.tasks.mask3d.data.segment_targets import SegmentInstanceTargets
from delimit3d.tasks.mask3d.losses.mask_set_loss import (
    build_pairwise_cost_matrix,
    hungarian_match,
    sigmoid_dice_loss,
)


class Mask3DSegmentSetCriterion(nn.Module):
    """Faithful class/mask/dice loss with independent auxiliary matching.

    Mask and dice terms are summed over all matched instances in the local
    batch and divided by the distributed target-instance count. This avoids
    giving a sparse scene the same mask-loss weight as a crowded scene.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 2.0,
        no_object_weight: float = 0.1,
        include_auxiliary: bool = True,
    ) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("Mask3D requires at least one foreground class plus no-object")
        self.num_classes = int(num_classes)
        self.no_object_class_id = self.num_classes - 1
        self.class_weight = float(class_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.no_object_weight = float(no_object_weight)
        self.include_auxiliary = bool(include_auxiliary)

    @staticmethod
    def _distributed_instance_denominator(
        targets: list[SegmentInstanceTargets],
        device: torch.device,
    ) -> torch.Tensor:
        count = torch.tensor(
            [sum(target.num_instances for target in targets)],
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count /= float(dist.get_world_size())
        return count.clamp_min(1.0)[0]

    def _validate_batch(
        self,
        predictions: list[dict[str, Any]],
        targets: list[SegmentInstanceTargets],
    ) -> None:
        if not predictions or len(predictions) != len(targets):
            raise ValueError(
                "predictions and targets must be non-empty lists of equal length"
            )
        for scene_index, (prediction, target) in enumerate(zip(predictions, targets)):
            logits = prediction["segment_mask_logits"]
            if logits.ndim != 2:
                raise ValueError("segment_mask_logits must have shape [Q,S]")
            if int(logits.shape[1]) != int(target.gt_masks.shape[1]):
                raise ValueError(
                    f"scene {scene_index}: prediction has {logits.shape[1]} segments "
                    f"but target has {target.gt_masks.shape[1]}"
                )
            if "segment_ids" in prediction and not torch.equal(
                prediction["segment_ids"].to(target.segment_ids.device),
                target.segment_ids,
            ):
                raise ValueError(
                    f"scene {scene_index}: prediction and target segment ordering differ"
                )
            object_logits = prediction["object_logits"]
            if object_logits.shape != (logits.shape[0], self.num_classes):
                raise ValueError(
                    "object_logits must have shape "
                    f"[Q,{self.num_classes}], got {tuple(object_logits.shape)}"
                )

    def _match_layer(
        self,
        predictions: list[dict[str, Any]],
        targets: list[SegmentInstanceTargets],
        *,
        context: str,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for scene_index, (prediction, target) in enumerate(zip(predictions, targets)):
            mask_logits = prediction["segment_mask_logits"]
            object_logits = prediction["object_logits"]
            gt_masks = target.gt_masks.to(mask_logits.device)
            domain = torch.ones(
                mask_logits.shape[1], device=mask_logits.device, dtype=torch.bool
            )
            if target.class_ids is None:
                if self.num_classes != 2 and gt_masks.shape[0] > 0:
                    raise ValueError(
                        "Class-aware Mask3D criterion requires target.class_ids "
                        f"when num_classes={self.num_classes}"
                    )
                class_ids = torch.zeros(
                    gt_masks.shape[0], device=mask_logits.device, dtype=torch.long
                )
            else:
                class_ids = target.class_ids.to(
                    device=mask_logits.device, dtype=torch.long
                )
                if class_ids.shape != (gt_masks.shape[0],):
                    raise ValueError(
                        "target.class_ids must align with target.gt_masks: "
                        f"{tuple(class_ids.shape)} != {tuple(gt_masks.shape[:1])}"
                    )
                if class_ids.numel() and (
                    int(class_ids.min().item()) < 0
                    or int(class_ids.max().item()) >= self.no_object_class_id
                ):
                    raise ValueError(
                        "target.class_ids must be contiguous foreground ids in "
                        f"[0,{self.no_object_class_id - 1}]"
                    )
            cost = build_pairwise_cost_matrix(
                mask_logits,
                gt_masks,
                domain,
                class_logits=object_logits,
                class_ids=class_ids,
                bce_weight=self.mask_weight,
                dice_weight=self.dice_weight,
                class_weight=self.class_weight,
            )
            pred_indices, target_indices = hungarian_match(
                cost, context=f"{context}/scene{scene_index}"
            )
            matches.append(
                (
                    torch.as_tensor(
                        pred_indices, device=mask_logits.device, dtype=torch.long
                    ),
                    torch.as_tensor(
                        target_indices, device=mask_logits.device, dtype=torch.long
                    ),
                )
            )
        return matches

    def _layer_loss(
        self,
        predictions: list[dict[str, Any]],
        targets: list[SegmentInstanceTargets],
        *,
        denominator: torch.Tensor,
        context: str,
    ) -> dict[str, Any]:
        self._validate_batch(predictions, targets)
        matches = self._match_layer(predictions, targets, context=context)

        classification_logits: list[torch.Tensor] = []
        classification_targets: list[torch.Tensor] = []
        mask_sum = denominator * 0.0
        dice_sum = denominator * 0.0
        match_count = 0
        for prediction, target, (pred_indices, target_indices) in zip(
            predictions, targets, matches
        ):
            object_logits = prediction["object_logits"]
            query_classes = torch.full(
                (object_logits.shape[0],),
                self.no_object_class_id,
                device=object_logits.device,
                dtype=torch.long,
            )
            if pred_indices.numel():
                if target.class_ids is None:
                    query_classes[pred_indices] = 0
                else:
                    query_classes[pred_indices] = target.class_ids.to(
                        device=object_logits.device, dtype=torch.long
                    )[target_indices]
                predicted_masks = prediction["segment_mask_logits"][pred_indices]
                target_masks = target.gt_masks.to(predicted_masks.device)[
                    target_indices
                ].float()
                per_mask_bce = F.binary_cross_entropy_with_logits(
                    predicted_masks, target_masks, reduction="none"
                ).mean(dim=1)
                mask_sum = mask_sum + per_mask_bce.sum()
                dice_sum = dice_sum + torch.stack(
                    [
                        sigmoid_dice_loss(predicted, expected)
                        for predicted, expected in zip(predicted_masks, target_masks)
                    ]
                ).sum()
                match_count += int(pred_indices.numel())
            classification_logits.append(object_logits)
            classification_targets.append(query_classes)

        class_weights = classification_logits[0].new_ones(self.num_classes)
        class_weights[self.no_object_class_id] = self.no_object_weight
        concatenated_logits = torch.cat(classification_logits, dim=0)
        concatenated_targets = torch.cat(classification_targets, dim=0)
        # Match torch.nn.functional.cross_entropy(..., reduction="mean",
        # weight=class_weights), while retaining the numerator and effective
        # denominator for exact gradient accumulation across scene microbatches.
        class_loss_terms = F.cross_entropy(
            concatenated_logits,
            concatenated_targets,
            weight=class_weights,
            reduction="none",
        )
        class_denominator = class_weights[concatenated_targets].sum().clamp_min(1.0)
        class_numerator = class_loss_terms.sum()
        loss_class = class_numerator / class_denominator
        loss_mask = mask_sum / denominator
        loss_dice = dice_sum / denominator
        total = (
            self.class_weight * loss_class
            + self.mask_weight * loss_mask
            + self.dice_weight * loss_dice
        )
        return {
            "loss_total": total,
            "loss_class": loss_class.detach(),
            "loss_score": loss_class.detach(),
            "loss_mask_bce": loss_mask.detach(),
            "loss_mask_dice": loss_dice.detach(),
            # These tensors intentionally retain their computation graphs.
            # They let the training loop reproduce Mask3D's full-batch
            # normalization when scenes must be forwarded as separate
            # microbatches because the joint sparse batch does not fit.
            "_class_numerator": class_numerator,
            "_class_denominator": class_denominator.detach(),
            "_mask_numerator": mask_sum,
            "_dice_numerator": dice_sum,
            "num_matches": match_count,
            "matches": matches,
        }

    def forward(
        self,
        predictions: list[dict[str, Any]],
        targets: list[SegmentInstanceTargets],
        *,
        context: str = "mask3d_segment",
        normalization_denominator: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_batch(predictions, targets)
        device = predictions[0]["segment_mask_logits"].device
        if normalization_denominator is None:
            denominator = self._distributed_instance_denominator(targets, device)
        else:
            denominator = torch.as_tensor(
                normalization_denominator,
                device=device,
                dtype=torch.float32,
            ).clamp_min(1.0)
        result = self._layer_loss(
            predictions, targets, denominator=denominator, context=f"{context}/final"
        )
        final_total = result["loss_total"]
        auxiliary_total = final_total * 0.0
        layer_components: list[dict[str, torch.Tensor]] = [
            {
                "class_numerator": result["_class_numerator"],
                "class_denominator": result["_class_denominator"],
                "mask_numerator": result["_mask_numerator"],
                "dice_numerator": result["_dice_numerator"],
            }
        ]

        if self.include_auxiliary:
            auxiliary_counts = {len(prediction.get("aux_outputs", [])) for prediction in predictions}
            if len(auxiliary_counts) != 1:
                raise ValueError("All scenes must expose the same number of auxiliary outputs")
            auxiliary_count = next(iter(auxiliary_counts))
            for layer_index in range(auxiliary_count):
                layer_predictions = [
                    prediction["aux_outputs"][layer_index]
                    for prediction in predictions
                ]
                layer_result = self._layer_loss(
                    layer_predictions,
                    targets,
                    denominator=denominator,
                    context=f"{context}/aux{layer_index}",
                )
                auxiliary_total = auxiliary_total + layer_result["loss_total"]
                layer_components.append(
                    {
                        "class_numerator": layer_result["_class_numerator"],
                        "class_denominator": layer_result["_class_denominator"],
                        "mask_numerator": layer_result["_mask_numerator"],
                        "dice_numerator": layer_result["_dice_numerator"],
                    }
                )
                result[f"loss_aux_layer_{layer_index}"] = layer_result[
                    "loss_total"
                ].detach()
        result["loss_aux"] = auxiliary_total.detach()
        result["loss_total"] = final_total + auxiliary_total
        result["loss_total_unweighted"] = result["loss_total"].detach()
        result["distributed_num_instances"] = denominator.detach()
        result["loss_components"] = layer_components
        result["scene_loss_totals"] = [
            float(result["loss_total"].detach().item())
        ] * len(predictions)
        return result

    def microbatch_loss(
        self,
        loss_result: dict[str, Any],
        *,
        total_target_instances: float | torch.Tensor,
        total_class_denominator: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compose one microbatch's contribution to a Mask3D batch loss.

        Mask3D receives all scenes in one sparse batch.  Its class loss is
        normalized by the weighted query count, while mask and dice losses are
        normalized by the total number of target instances.  Dividing one
        already-combined scalar by an accumulation length is therefore not
        equivalent: it underweights mask/dice terms whenever scenes are
        forwarded separately.  This method keeps each component's graph and
        applies the full logical-batch denominators before backpropagation.

        ``loss_result`` must be the result of :meth:`forward` with the same
        logical-batch target denominator.  The returned scalar can be
        backpropagated immediately, so graphs from earlier microbatches do not
        need to remain resident on the GPU.
        """
        components = loss_result.get("loss_components")
        if not components:
            raise ValueError(
                "loss_result does not contain differentiable loss_components; "
                "call the criterion forward before composing microbatch loss"
            )
        device = components[0]["class_numerator"].device
        target_denominator = torch.as_tensor(
            total_target_instances, device=device, dtype=torch.float32
        ).clamp_min(1.0)
        class_denominator = torch.as_tensor(
            total_class_denominator, device=device, dtype=torch.float32
        ).clamp_min(1.0)

        total = components[0]["class_numerator"] * 0.0
        for component in components:
            total = total + self.class_weight * (
                component["class_numerator"] / class_denominator
            )
            total = total + self.mask_weight * (
                component["mask_numerator"] / target_denominator
            )
            total = total + self.dice_weight * (
                component["dice_numerator"] / target_denominator
            )
        return total


__all__ = ["Mask3DSegmentSetCriterion"]
