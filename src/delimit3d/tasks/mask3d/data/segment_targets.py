"""Segment-domain targets for LitePT -> Mask3D transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SegmentInstanceTargets:
    """Instance masks over all ScanNet mesh segments in one scene.

    ``class_ids`` is optional for backwards-compatible class-agnostic arms.
    When present, it is aligned with ``instance_ids`` and contains contiguous
    foreground class indices (the no-object class is owned by the criterion).
    """

    segment_ids: torch.Tensor
    point_to_segment: torch.Tensor
    instance_ids: np.ndarray
    gt_masks: torch.Tensor
    num_instances: int
    instance_sizes: np.ndarray
    class_ids: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "SegmentInstanceTargets":
        return SegmentInstanceTargets(
            segment_ids=self.segment_ids.to(device),
            point_to_segment=self.point_to_segment.to(device),
            instance_ids=self.instance_ids,
            gt_masks=self.gt_masks.to(device),
            num_instances=self.num_instances,
            instance_sizes=self.instance_sizes,
            class_ids=(self.class_ids.to(device) if self.class_ids is not None else None),
        )


def build_segment_instance_targets(
    instance_labels: torch.Tensor,
    scan_segments: torch.Tensor,
    *,
    ignore_label: int = -1,
    min_instance_points: int = 1,
    require_pure_segments: bool = True,
    instance_class_map: dict[int, int] | None = None,
) -> SegmentInstanceTargets:
    """Collapse valid instances into segment-domain binary masks.

    Every segment remains in the mask domain. Segments belonging to ignored
    wall/floor/background regions are therefore explicit negatives for every
    object mask. When ``instance_class_map`` is supplied, the returned class
    ids remain aligned with the filtered instance rows for class-aware
    Mask3D training.
    """
    if instance_labels.ndim != 1 or scan_segments.ndim != 1:
        raise ValueError("instance_labels and scan_segments must be one-dimensional")
    if instance_labels.shape != scan_segments.shape:
        raise ValueError(
            f"instance labels {instance_labels.shape} and segments "
            f"{scan_segments.shape} are not vertex-aligned"
        )
    if instance_labels.numel() == 0:
        raise ValueError("Cannot construct targets from an empty scene")

    segment_ids, point_to_segment = torch.unique(
        scan_segments.long(), sorted=True, return_inverse=True
    )
    num_segments = int(segment_ids.numel())
    segment_instance = torch.full(
        (num_segments,), int(ignore_label), device=instance_labels.device, dtype=torch.long
    )

    for segment_index in range(num_segments):
        labels = instance_labels[point_to_segment == segment_index].long()
        positive = labels[labels != int(ignore_label)].unique(sorted=True)
        if positive.numel() > 1:
            raise RuntimeError(
                f"ScanNet segment {int(segment_ids[segment_index].item())} spans "
                f"multiple target instances: {positive.tolist()}"
            )
        if positive.numel() == 1:
            if require_pure_segments and bool((labels == int(ignore_label)).any().item()):
                raise RuntimeError(
                    f"ScanNet segment {int(segment_ids[segment_index].item())} mixes "
                    "an object instance with ignored/background vertices"
                )
            segment_instance[segment_index] = positive[0]

    rows: list[torch.Tensor] = []
    kept_ids: list[int] = []
    sizes: list[int] = []
    class_ids: list[int] = []
    for instance_id_tensor in segment_instance[segment_instance != int(ignore_label)].unique(
        sorted=True
    ):
        instance_id = int(instance_id_tensor.item())
        point_count = int((instance_labels == instance_id).sum().item())
        if point_count < int(min_instance_points):
            continue
        rows.append(segment_instance == instance_id)
        kept_ids.append(instance_id)
        sizes.append(point_count)
        if instance_class_map is not None:
            if instance_id not in instance_class_map:
                raise KeyError(
                    f"Missing class id for kept instance {instance_id}; "
                    "class-aware Mask3D targets require every instance to be labeled"
                )
            class_ids.append(int(instance_class_map[instance_id]))

    if rows:
        masks = torch.stack(rows, dim=0)
    else:
        masks = torch.zeros(
            (0, num_segments), device=instance_labels.device, dtype=torch.bool
        )
    class_ids_t = (
        torch.tensor(class_ids, device=instance_labels.device, dtype=torch.long)
        if instance_class_map is not None
        else None
    )
    return SegmentInstanceTargets(
        segment_ids=segment_ids,
        point_to_segment=point_to_segment,
        instance_ids=np.asarray(kept_ids, dtype=np.int64),
        gt_masks=masks,
        num_instances=len(kept_ids),
        instance_sizes=np.asarray(sizes, dtype=np.int64),
        class_ids=class_ids_t,
    )


__all__ = ["SegmentInstanceTargets", "build_segment_instance_targets"]
