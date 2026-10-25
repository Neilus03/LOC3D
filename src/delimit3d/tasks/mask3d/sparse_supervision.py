"""ScanNet LA annotation boundary and partial-observation Mask3D objective.

This module never opens dense annotations. Unknown is -1; explicit, permitted
non-instance classes are 0; observed foreground identities are positive.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

THING_NYU40 = frozenset((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39))
VALID_NYU40 = THING_NYU40 | {1, 2}


def select_annotations(indices, vertex_count, segment_ids, groups, label_map):
    """Preparation-only boundary: export labels at EXACT official vertices.

    Do not build dense masks, count hidden instances, or derive extents. Only
    groups intersecting the selected segment IDs contribute to this result.
    Local identities are assigned by first occurrence in official index order.
    """
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('Official indices must be a one-dimensional integer array')
    indices = indices.astype(np.int64)
    # The official files contain duplicate indices (including 59 points200 scenes).
    # Preserve them exactly. The voxel policy coalesces identical observations.
    if np.any(indices < 0) or np.any(indices >= vertex_count):
        raise ValueError('Out-of-range official vertex index')
    if len(segment_ids) != vertex_count:
        raise ValueError('Segment/mesh vertex alignment changed')
    selected_segments = np.asarray(segment_ids, dtype=np.int64)[indices]
    permitted_segments = set(selected_segments.tolist())
    lookup = {}
    for group in groups:
        intersection = permitted_segments.intersection(group['segments'])
        if not intersection:
            continue
        semantic = int(label_map.get(group['label'], 0))
        identity = int(group['objectId'])
        for segment in intersection:
            value = (identity, semantic)
            if segment in lookup and lookup[segment] != value:
                raise ValueError('Contradictory annotation for a permitted vertex')
            lookup[segment] = value
    instance = np.full(len(indices), -1, dtype=np.int64)
    semantic = np.full(len(indices), -1, dtype=np.int64)
    local_ids = {}
    for i, segment in enumerate(selected_segments):
        pair = lookup.get(int(segment))
        if pair is None or pair[1] not in VALID_NYU40:
            continue
        identity, sem = pair
        semantic[i] = sem
        if sem in THING_NYU40:
            if identity not in local_ids:
                local_ids[identity] = len(local_ids) + 1
            instance[i] = local_ids[identity]
        else:
            instance[i] = 0
    return dict(vertex_indices=indices, instance_ids=instance, semantic_ids=semantic)


@dataclass
class SparseAnnotations:
    vertex_indices: torch.Tensor
    instance_ids: torch.Tensor
    semantic_ids: torch.Tensor
    num_points: int

    def to(self, device):
        return SparseAnnotations(self.vertex_indices.to(device), self.instance_ids.to(device),
                                 self.semantic_ids.to(device), self.num_points)


@dataclass
class ObservedTargets:
    columns: torch.Tensor
    instance_ids: torch.Tensor
    masks: torch.Tensor
    statistics: dict


def observed_targets(annotations, point_to_segment):
    """Keep unanimous annotated voxels; drop every label in a conflicting voxel.

    The representative point need not be annotated. Several same-identity
    annotations in a voxel are coalesced into ONE supervised logit. Hidden
    labels in that voxel are never read. No label is propagated to another voxel.
    """
    if point_to_segment.shape != (annotations.num_points,):
        raise ValueError('Original vertex domain changed')
    a = annotations.to(point_to_segment.device)
    if len(a.vertex_indices) != len(a.instance_ids) or len(a.instance_ids) != len(a.semantic_ids):
        raise ValueError('Annotation arrays are not aligned')
    if len(a.vertex_indices) and (a.vertex_indices.min() < 0 or a.vertex_indices.max() >= a.num_points):
        raise ValueError('Annotation escaped original vertex domain')
    valid = (a.instance_ids >= 0) & (a.semantic_ids >= 0)
    voxel = point_to_segment[a.vertex_indices[valid]]
    identities, semantics = a.instance_ids[valid], a.semantic_ids[valid]
    columns, values = [], []
    conflict_points = conflict_voxels = retained = 0
    for v in voxel.unique(sorted=True):
        at = voxel == v
        pairs = torch.stack((identities[at], semantics[at]), dim=1).unique(dim=0)
        if len(pairs) != 1:
            conflict_points += int(at.sum())
            conflict_voxels += 1
            continue
        columns.append(v)
        values.append(pairs[0, 0])
        retained += int(at.sum())
    columns = torch.stack(columns) if columns else voxel.new_empty(0)
    values = torch.stack(values) if values else identities.new_empty(0)
    instances = values[values > 0].unique(sorted=True)
    masks = (instances[:, None] == values[None, :]).float()
    stats = dict(official_annotations=len(a.vertex_indices), unique_annotated_vertices=len(a.vertex_indices.unique()),
                 duplicate_official_entries=len(a.vertex_indices)-len(a.vertex_indices.unique()),valid_annotations=int(valid.sum()),
                 invalid_annotations=int((~valid).sum()), after_augmentation=len(a.vertex_indices),
                 after_crop=len(a.vertex_indices), retained_annotations=retained,
                 supervised_voxels=len(columns), coalesced_annotations=retained-len(columns),
                 conflicting_annotations=conflict_points, conflicting_voxels=conflict_voxels,
                 observed_instances_before=len(a.instance_ids[a.instance_ids > 0].unique()),
                 observed_instances_after=len(instances), unknown_instances=None)
    return ObservedTargets(columns, instances, masks, stats)


class SparseMask3DCriterion(nn.Module):
    """Match observed instances only; unmatched queries receive ZERO supervision.

    Matched queries: foreground CE plus BCE/Dice on unanimously labeled voxels.
    Another observed instance and an explicitly annotated wall/floor are valid
    mask negatives. Unannotated vertices, invalid classes, and conflicts are
    unknown. Every auxiliary layer uses this same rule and independent matching.
    Empty supervision is a differentiable zero; no negative-only surrogate.
    """
    def __init__(self, *, num_classes=2, include_auxiliary=True, **kwargs):
        super().__init__()
        if num_classes != 2:
            raise ValueError('This protocol is class-agnostic')
        self.include_auxiliary = include_auxiliary
        self.no_object_weight = 0.0
        self.statistics = []
        self.all_matches = []

    @staticmethod
    def match(prediction, target):
        logits = prediction['segment_mask_logits'][:, target.columns]
        objects = prediction['object_logits']
        if len(target.instance_ids) > len(logits):
            raise ValueError('Observed instances exceed the sealed query count')
        if not len(target.instance_ids):
            return logits.new_empty(0, dtype=torch.long), logits.new_empty(0, dtype=torch.long)
        with torch.no_grad():
            truth = target.masks
            # Pairwise BCE without allocating [Q,K,A].
            bce = (F.softplus(logits).mean(1)[:, None]
                   - logits @ truth.T / logits.shape[1])
            prob = logits.sigmoid()
            dice = 1 - (2 * prob @ truth.T + 1) / (prob.sum(1)[:, None] + truth.sum(1)[None, :] + 1)
            cost = 5 * bce + 2 * dice - 2 * objects.softmax(-1)[:, :1]
            if not torch.isfinite(cost).all():
                raise ValueError('Nonfinite sparse matching cost')
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
        return (torch.as_tensor(rows, device=logits.device), torch.as_tensor(cols, device=logits.device))

    def layer(self, predictions, targets):
        # Include every output in the zero graph; empty scenes have zero gradient.
        zero = sum(p['segment_mask_logits'].sum()*0 + p['object_logits'].sum()*0 for p in predictions)
        cls = bce = dice = zero
        denominator = max(sum(len(t.instance_ids) for t in targets), 1)
        matches = []
        for p, t in zip(predictions, targets, strict=True):
            rows, cols = self.match(p, t)
            matches.append((rows, cols))
            if len(rows) == 0:
                continue
            logits = p['segment_mask_logits'][rows][:, t.columns]
            truth = t.masks[cols]
            cls = cls + F.cross_entropy(p['object_logits'][rows], torch.zeros_like(rows), reduction='sum')
            bce = bce + F.binary_cross_entropy_with_logits(logits, truth, reduction='none').mean(1).sum()
            prob = logits.sigmoid()
            dice = dice + (1-(2*(prob*truth).sum(1)+1)/(prob.sum(1)+truth.sum(1)+1)).sum()
        cls, bce, dice = cls/denominator, bce/denominator, dice/denominator
        return dict(loss_total=2*cls+5*bce+2*dice, loss_class=cls.detach(),
                    loss_mask_bce=bce.detach(), loss_mask_dice=dice.detach(), matches=matches)

    def forward(self, predictions, annotations, *, normalization_denominator=None, **kwargs):
        if normalization_denominator is not None:
            raise ValueError('LA v2 uses real scene batches, no dense denominator')
        if not predictions or len(predictions) != len(annotations):
            raise ValueError('Empty or mismatched batch')
        targets = [observed_targets(a, p['point_to_segment']) for p,a in zip(predictions, annotations, strict=True)]
        for p,t in zip(predictions,targets,strict=True):
            if len(t.columns) and t.columns.max() >= p['segment_mask_logits'].shape[1]:
                raise ValueError('Voxel target mapping differs from decoder mapping')
        self.statistics.extend(t.statistics for t in targets)
        result = self.layer(predictions, targets)
        self.all_matches = [result['matches']]
        aux = result['loss_total'] * 0
        if self.include_auxiliary:
            counts = {len(p.get('aux_outputs', [])) for p in predictions}
            if len(counts) != 1:
                raise ValueError('Auxiliary layer count changed across scenes')
            for index in range(next(iter(counts))):
                layer = self.layer([p['aux_outputs'][index] for p in predictions], targets)
                aux = aux + layer['loss_total']
                self.all_matches.append(layer['matches'])
        result['loss_aux'] = aux.detach()
        result['loss_total'] = result['loss_total'] + aux
        if not torch.isfinite(result['loss_total']):
            raise ValueError('Nonfinite sparse objective')
        return result


def geometry_voxel_segments(points, offsets, grid_size=.05):
    """Match LitePT's floor quantization; never use annotated mesh segments."""
    out, start = [], 0
    for end in offsets.tolist():
        grid = torch.floor(points[start:end] / grid_size).long()
        grid -= grid.min(0).values
        _, inverse = torch.unique(grid, dim=0, sorted=True, return_inverse=True)
        out.append(inverse)
        start = end
    if start != len(points):
        raise ValueError('Invalid batch offsets')
    return torch.cat(out)
