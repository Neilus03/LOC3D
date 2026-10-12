"""Mask-only Hungarian training: no class head, class matching, or semantic loss."""
import torch
import torch.nn.functional as F
from delimit3d.tasks.mask3d.losses.mask3d_segment_loss import Mask3DSegmentSetCriterion
from delimit3d.tasks.mask3d.losses.mask_set_loss import build_pairwise_cost_matrix, hungarian_match, sigmoid_dice_loss

class Mask3DMaskOnlyCriterion(Mask3DSegmentSetCriterion):
    def __init__(self, **kwargs):
        kwargs['class_weight'] = 0.
        super().__init__(**kwargs)

    def _validate_batch(self, predictions, targets):
        if not predictions or len(predictions) != len(targets):
            raise ValueError('Nonempty aligned prediction and target batches required')
        for prediction, target in zip(predictions, targets):
            if 'object_logits' in prediction:
                raise ValueError('Mask-only prediction must not contain class logits')
            logits = prediction['segment_mask_logits']
            if logits.ndim != 2 or logits.shape[1] != target.gt_masks.shape[1]:
                raise ValueError('Mask and target segment dimensions differ')

    def _layer_loss(self, predictions, targets, *, denominator, context):
        if not predictions or len(predictions) != len(targets):
            raise ValueError('Nonempty aligned prediction and target batches required')
        zero = sum(p['segment_mask_logits'].sum()*0 for p in predictions)
        mask_sum, dice_sum = zero, zero
        matches = []
        for index, (prediction, target) in enumerate(zip(predictions, targets)):
            if 'object_logits' in prediction:
                raise ValueError('Mask-only prediction must not contain class logits')
            logits = prediction['segment_mask_logits']
            gt = target.gt_masks.to(logits.device).float()
            if logits.ndim != 2 or logits.shape[1] != gt.shape[1]:
                raise ValueError('Mask and target segment dimensions differ')
            if 'segment_ids' in prediction and not torch.equal(prediction['segment_ids'].to(target.segment_ids.device), target.segment_ids):
                raise ValueError('Segment ordering differs')
            cost = build_pairwise_cost_matrix(logits, gt,
                torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool),
                bce_weight=self.mask_weight, dice_weight=self.dice_weight, class_weight=0.)
            pi, ti = hungarian_match(cost, context=f'{context}/scene{index}')
            pi=torch.as_tensor(pi,device=logits.device,dtype=torch.long)
            ti=torch.as_tensor(ti,device=logits.device,dtype=torch.long)
            matches.append((pi,ti))
            if pi.numel():
                mask_sum = mask_sum + F.binary_cross_entropy_with_logits(logits[pi],gt[ti],reduction='none').mean(1).sum()
                dice_sum = dice_sum + torch.stack([sigmoid_dice_loss(p,t) for p,t in zip(logits[pi],gt[ti])]).sum()
        mask, dice = mask_sum/denominator, dice_sum/denominator
        return dict(loss_total=self.mask_weight*mask+self.dice_weight*dice,
            loss_class=zero.detach(), loss_score=zero.detach(), loss_mask_bce=mask.detach(),
            loss_mask_dice=dice.detach(), _class_numerator=zero,
            _class_denominator=denominator.new_ones(()), _mask_numerator=mask_sum,
            _dice_numerator=dice_sum, num_matches=sum(len(p) for p,t in matches), matches=matches)
