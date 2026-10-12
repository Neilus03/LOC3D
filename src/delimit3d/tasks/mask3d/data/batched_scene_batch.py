"""Batch contract for multi-scene training steps.

The training path uses flat concatenation plus offsets, similar to Pointcept:

- points/features are concatenated across scenes
- scene boundaries are preserved with cumulative point offsets
- labels and masks remain scene-local so loss matching semantics stay unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _offsets_from_counts(counts: list[int]) -> torch.Tensor:
    if not counts:
        return torch.zeros(0, dtype=torch.long)
    return torch.tensor(counts, dtype=torch.long).cumsum(dim=0)


def split_tensor_by_offsets(
    tensor: torch.Tensor,
    offsets: torch.Tensor,
) -> list[torch.Tensor]:
    """Split a flat tensor along dim 0 using cumulative end offsets."""
    if offsets.numel() == 0:
        return []

    parts: list[torch.Tensor] = []
    start = 0
    for end in offsets.tolist():
        parts.append(tensor[start:end])
        start = end
    return parts


@dataclass(frozen=True)
class BatchedMultiSceneSample:
    """A flat-concatenated batch of one or more scenes."""

    scene_ids: list[str]
    scene_dirs: list[str]
    source_names: list[str]
    source_indices: list[int]
    points: torch.Tensor
    features: torch.Tensor
    point_offsets: torch.Tensor
    labels_by_granularity: dict[str, list[torch.Tensor]]
    instance_classes_by_granularity: dict[str, list[dict[int, int] | None]]
    valid_points: list[torch.Tensor]
    seen_points: list[torch.Tensor]
    supervision_masks: list[torch.Tensor]
    scene_meta: list[Any]
    granularities: tuple[str, ...]
    vertex_indices: list[torch.Tensor | None]
    scan_segments: list[torch.Tensor | None]
    padding: list[bool]

    @property
    def num_scenes(self) -> int:
        return len(self.scene_ids)

    @property
    def point_counts(self) -> list[int]:
        counts: list[int] = []
        start = 0
        for end in self.point_offsets.tolist():
            counts.append(end - start)
            start = end
        return counts

    @property
    def total_points(self) -> int:
        if self.point_offsets.numel() == 0:
            return 0
        return int(self.point_offsets[-1].item())

    def split_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        return split_tensor_by_offsets(tensor, self.point_offsets)


def collate_multi_scene_samples(
    batch: list[dict[str, Any]],
) -> BatchedMultiSceneSample:
    """Concatenate a list of scene dicts into one flat batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    granularities = tuple(batch[0]["granularities"])
    for sample in batch[1:]:
        if tuple(sample["granularities"]) != granularities:
            raise ValueError("All samples in a batch must share the same granularities")

    points = torch.cat([sample["points"] for sample in batch], dim=0)
    features = torch.cat([sample["features"] for sample in batch], dim=0)
    point_offsets = _offsets_from_counts(
        [int(sample["points"].shape[0]) for sample in batch]
    )

    labels_by_granularity: dict[str, list[torch.Tensor]] = {
        g: [sample["labels_by_granularity"][g] for sample in batch]
        for g in granularities
    }
    instance_classes_by_granularity: dict[str, list[dict[int, int] | None]] = {
        g: [
            sample.get("instance_classes_by_granularity", {}).get(g)
            for sample in batch
        ]
        for g in granularities
    }

    vertex_indices: list[torch.Tensor | None] = [
        sample["vertex_indices"] if "vertex_indices" in sample else None
        for sample in batch
    ]

    return BatchedMultiSceneSample(
        scene_ids=[str(sample["scene_id"]) for sample in batch],
        scene_dirs=[str(sample["scene_dir"]) for sample in batch],
        source_names=[str(sample.get("source_name", "default")) for sample in batch],
        source_indices=[int(sample.get("source_index", 0)) for sample in batch],
        points=points,
        features=features,
        point_offsets=point_offsets,
        labels_by_granularity=labels_by_granularity,
        instance_classes_by_granularity=instance_classes_by_granularity,
        valid_points=[sample["valid_points"] for sample in batch],
        seen_points=[sample["seen_points"] for sample in batch],
        supervision_masks=[sample["supervision_mask"] for sample in batch],
        scene_meta=[sample["scene_meta"] for sample in batch],
        granularities=granularities,
        vertex_indices=vertex_indices,
        scan_segments=[sample.get("scan_segments") for sample in batch],
        padding=[bool(sample.get("is_padding", False)) for sample in batch],
    )
