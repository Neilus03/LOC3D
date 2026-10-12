"""Faithful Mask3D decoder on a LitePT-S* feature pyramid.

This is deliberately separate from ``mask3d_continuous_decoder.py``.  It has
no granularity prompt, no learned queries, and no MinkowskiEngine dependency.
LitePT replaces the original sparse-convolution backbone while the decoder
keeps Mask3D's four-level, three-cycle query update schedule.

The decoder structure follows Mask3D (MIT license; see
``student/THIRD_PARTY_NOTICES.md``).  Tensor layout adaptations are local to
this repository's LitePT feature-pyramid contract.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from delimit3d.models.litept_wrapper import (
    LitePTBackbone,
    LitePTFeaturePyramid,
)


MASK3D_MEMORY_CHANNELS = (504, 252, 144, 72)
MASK3D_DEC0_POOLED_MEMORY_CHANNELS = (72, 72, 72, 72)
# Four memory levels retained, but native enc4 features are intentionally not
# exposed to Mask3D: dec3 is mean-pooled onto the exact enc4 grid instead.
MASK3D_DECODER_ONLY_MEMORY_CHANNELS = (252, 252, 144, 72)
MASK3D_MEMORY_STRIDES = (16, 8, 4, 2)


def _offset_starts(offsets: torch.Tensor) -> torch.Tensor:
    return torch.cat([offsets.new_zeros(1), offsets[:-1]])


def _slice_by_offsets(
    tensor: torch.Tensor,
    offsets: torch.Tensor,
    scene_index: int,
) -> torch.Tensor:
    starts = _offset_starts(offsets)
    return tensor[int(starts[scene_index].item()) : int(offsets[scene_index].item())]


def _scatter_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Pure-tensor scatter mean over the first axis."""
    result = values.new_zeros((int(output_size), *values.shape[1:]))
    counts = values.new_zeros((int(output_size),))
    result.index_add_(0, index.long(), values)
    counts.index_add_(
        0,
        index.long(),
        torch.ones(index.shape[0], device=values.device, dtype=values.dtype),
    )
    view_shape = (int(output_size),) + (1,) * (values.ndim - 1)
    return result / counts.clamp_min(1.0).view(view_shape)


@torch.no_grad()
def farthest_point_indices_from_zero(
    grid: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Deterministic FPS whose first selected voxel is index zero.

    Mask3D's PointNet++ FPS kernel initializes its distance buffer and starts
    from the first point.  Repeating the selected sequence preserves the fixed
    query count for tiny smoke-test scenes.
    """
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(f"grid must have shape [V, 3], got {tuple(grid.shape)}")
    num_points = int(grid.shape[0])
    if num_points == 0:
        raise ValueError("Mask3D query initialization requires at least one voxel")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    unique_count = min(int(count), num_points)
    selected = torch.empty(unique_count, device=grid.device, dtype=torch.long)
    nearest = torch.full(
        (num_points,), float("inf"), device=grid.device, dtype=torch.float32
    )
    coordinates = grid.float()
    farthest = 0
    for index in range(unique_count):
        selected[index] = farthest
        squared_distance = ((coordinates - coordinates[farthest]) ** 2).sum(dim=-1)
        nearest = torch.minimum(nearest, squared_distance)
        farthest = int(nearest.argmax().item())

    if unique_count == int(count):
        return selected
    repetitions = (int(count) + unique_count - 1) // unique_count
    return selected.repeat(repetitions)[: int(count)]


class Mask3DFourierPositionEncoding(nn.Module):
    """Mask3D Gaussian Fourier coordinates normalized exactly to ``[0, 1]``."""

    def __init__(self, hidden_dim: int = 128, gauss_scale: float = 1.0) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even number")
        gaussian = torch.empty(3, hidden_dim // 2).normal_()
        gaussian.mul_(float(gauss_scale))
        self.register_buffer("gauss_B", gaussian)

    def normalize(
        self,
        xyz: torch.Tensor,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
    ) -> torch.Tensor:
        return (xyz - scene_min) / (scene_max - scene_min).clamp_min(1e-6)

    def forward(
        self,
        xyz: torch.Tensor,
        *,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.normalize(xyz.float(), scene_min.float(), scene_max.float())
        projected = (normalized * (2.0 * math.pi)) @ self.gauss_B.float()
        encoded = torch.cat([projected.sin(), projected.cos()], dim=-1)
        return encoded.to(dtype=xyz.dtype)


class _Mask3DQueryProjection(nn.Module):
    """Upstream GenericMLP query projection (two 1x1 convs plus ReLU)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=True),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=True),
            nn.ReLU(),
        )

    def forward(self, encoded_xyz: torch.Tensor) -> torch.Tensor:
        projected = self.layers(encoded_xyz.transpose(0, 1).unsqueeze(0))
        return projected[0].transpose(0, 1)


class _SelfAttentionLayer(nn.Module):
    """Pure-PyTorch port of Mask3D's self-attention block."""

    def __init__(self, dim: int, heads: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, target: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            normalized = self.norm(target)
            positioned = normalized + query_pos
            update = self.self_attn(positioned, positioned, normalized, need_weights=False)[0]
            return target + self.dropout(update)
        positioned = target + query_pos
        update = self.self_attn(positioned, positioned, target, need_weights=False)[0]
        return self.norm(target + self.dropout(update))


class _CrossAttentionLayer(nn.Module):
    """Pure-PyTorch port of Mask3D's cross-attention block."""

    def __init__(self, dim: int, heads: int, dropout: float, pre_norm: bool) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pre_norm:
            normalized = self.norm(target)
            query = normalized + query_pos
            update = self.multihead_attn(
                query,
                memory + memory_pos,
                memory,
                attn_mask=attention_mask,
                need_weights=False,
            )[0]
            return target + self.dropout(update)
        update = self.multihead_attn(
            target + query_pos,
            memory + memory_pos,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        return self.norm(target + self.dropout(update))


class _FeedForwardLayer(nn.Module):
    """Pure-PyTorch port of Mask3D's ReLU feed-forward block."""

    def __init__(
        self,
        dim: int,
        feedforward_dim: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.pre_norm = bool(pre_norm)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _update(self, target: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(target))))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            return target + self.dropout(self._update(self.norm(target)))
        return self.norm(target + self.dropout(self._update(target)))


def sequential_average_pool(
    finest_logits: torch.Tensor,
    parent_maps: Iterable[torch.Tensor],
    parent_sizes: Iterable[int],
) -> list[torch.Tensor]:
    """Average-pool query logits through each adjacent LitePT hierarchy edge."""
    current = finest_logits
    outputs: list[torch.Tensor] = []
    for parent_map, parent_size in zip(parent_maps, parent_sizes):
        if int(parent_map.shape[0]) != int(current.shape[1]):
            raise ValueError(
                "Adjacent parent map length does not match child logits: "
                f"{parent_map.shape[0]} != {current.shape[1]}"
            )
        pooled = current.new_zeros((current.shape[0], int(parent_size)))
        counts = current.new_zeros((int(parent_size),))
        pooled.index_add_(1, parent_map.long(), current)
        counts.index_add_(
            0,
            parent_map.long(),
            torch.ones(parent_map.shape[0], device=current.device, dtype=current.dtype),
        )
        current = pooled / counts.clamp_min(1.0).unsqueeze(0)
        outputs.append(current)
    return outputs


@dataclass(frozen=True)
class SceneLitePTPyramid:
    """One-scene view of :class:`LitePTFeaturePyramid`."""

    finest_features: torch.Tensor
    finest_grid: torch.Tensor
    finest_xyz: torch.Tensor
    memory_features: tuple[torch.Tensor, ...]
    memory_grids: tuple[torch.Tensor, ...]
    memory_xyz: tuple[torch.Tensor, ...]
    parent_maps: tuple[torch.Tensor, ...]
    original_to_finest: torch.Tensor
    representative_indices: torch.Tensor
    point_xyz: torch.Tensor


def split_feature_pyramid(
    pyramid: LitePTFeaturePyramid,
) -> list[SceneLitePTPyramid]:
    """Split a batched pyramid while rebasing every local sparse index."""
    num_scenes = int(pyramid.point_offsets.numel())
    if num_scenes != int(pyramid.finest_offsets.numel()):
        raise ValueError("Point and finest-token offsets disagree on batch size")
    if len(pyramid.memory_levels) != 4 or len(pyramid.parent_maps) != 4:
        raise ValueError("Mask3D requires four memory levels and four parent maps")

    fine_to_coarse_levels = (
        pyramid.memory_levels[3],
        pyramid.memory_levels[2],
        pyramid.memory_levels[1],
        pyramid.memory_levels[0],
    )
    child_offsets = (pyramid.finest_offsets,) + tuple(
        level.offsets for level in fine_to_coarse_levels[:-1]
    )
    parent_offsets = tuple(level.offsets for level in fine_to_coarse_levels)

    scenes: list[SceneLitePTPyramid] = []
    point_starts = _offset_starts(pyramid.point_offsets)
    finest_starts = _offset_starts(pyramid.finest_offsets)
    for scene_index in range(num_scenes):
        local_parent_maps: list[torch.Tensor] = []
        for mapping, child_offset, parent_offset in zip(
            pyramid.parent_maps, child_offsets, parent_offsets
        ):
            child_mapping = _slice_by_offsets(mapping, child_offset, scene_index)
            parent_start = _offset_starts(parent_offset)[scene_index]
            local_mapping = child_mapping - parent_start
            parent_count = int(
                (parent_offset[scene_index] - _offset_starts(parent_offset)[scene_index]).item()
            )
            if local_mapping.numel() and (
                int(local_mapping.min().item()) < 0
                or int(local_mapping.max().item()) >= parent_count
            ):
                raise RuntimeError("Cross-scene collision detected in LitePT parent map")
            local_parent_maps.append(local_mapping)

        original_to_finest = _slice_by_offsets(
            pyramid.original_to_finest, pyramid.point_offsets, scene_index
        ) - finest_starts[scene_index]
        representatives = _slice_by_offsets(
            pyramid.representative_indices, pyramid.finest_offsets, scene_index
        ) - point_starts[scene_index]

        scenes.append(
            SceneLitePTPyramid(
                finest_features=_slice_by_offsets(
                    pyramid.finest_features, pyramid.finest_offsets, scene_index
                ),
                finest_grid=_slice_by_offsets(
                    pyramid.finest_grid, pyramid.finest_offsets, scene_index
                ),
                finest_xyz=_slice_by_offsets(
                    pyramid.finest_xyz, pyramid.finest_offsets, scene_index
                ),
                memory_features=tuple(
                    _slice_by_offsets(level.features, level.offsets, scene_index)
                    for level in pyramid.memory_levels
                ),
                memory_grids=tuple(
                    _slice_by_offsets(level.grid, level.offsets, scene_index)
                    for level in pyramid.memory_levels
                ),
                memory_xyz=tuple(
                    _slice_by_offsets(level.xyz, level.offsets, scene_index)
                    for level in pyramid.memory_levels
                ),
                parent_maps=tuple(local_parent_maps),
                original_to_finest=original_to_finest,
                representative_indices=representatives,
                point_xyz=_slice_by_offsets(
                    pyramid.point_xyz, pyramid.point_offsets, scene_index
                ),
            )
        )
    return scenes


class LitePTMask3DDecoder(nn.Module):
    """Four-level, three-cycle Mask3D decoder for one LitePT scene."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_cycles: int = 3,
        train_queries: int = 100,
        eval_queries: int = 150,
        num_classes: int = 2,
        mask_only: bool = False,
        memory_channels: tuple[int, ...] = MASK3D_MEMORY_CHANNELS,
        memory_strides: tuple[int, ...] = MASK3D_MEMORY_STRIDES,
        sample_sizes: tuple[int, ...] = (200, 800, 3200, 12800),
        gauss_scale: float = 1.0,
        pre_norm: bool = False,
        attention_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim != 128:
            raise ValueError("Faithful Mask3D requires hidden_dim=128")
        if num_heads != 8:
            raise ValueError("Faithful Mask3D requires num_heads=8")
        if dim_feedforward != 1024:
            raise ValueError("Faithful Mask3D requires dim_feedforward=1024")
        if dropout != 0.0:
            raise ValueError("Faithful Mask3D requires dropout=0")
        if num_cycles != 3:
            raise ValueError("Faithful Mask3D requires three decoder cycles")
        if train_queries != 100:
            raise ValueError(
                "Faithful LitePT -> Mask3D uses 100 training queries"
            )
        if eval_queries not in (100, 150):
            raise ValueError(
                "Faithful LitePT -> Mask3D supports 100 or 150 evaluation "
                f"queries, got {eval_queries}"
            )
        if pre_norm:
            raise ValueError("Faithful upstream Mask3D configuration uses post-norm blocks")
        if tuple(sample_sizes) != (200, 800, 3200, 12800):
            raise ValueError(
                "Faithful four-level sample sizes are (200, 800, 3200, 12800)"
            )
        if gauss_scale != 1.0:
            raise ValueError("Faithful Mask3D requires gauss_scale=1.0")
        if attention_threshold != 0.5:
            raise ValueError("Faithful Mask3D requires attention_threshold=0.5")
        allowed_memory_channels = {
            MASK3D_MEMORY_CHANNELS,
            MASK3D_DEC0_POOLED_MEMORY_CHANNELS,
            MASK3D_DECODER_ONLY_MEMORY_CHANNELS,
        }
        if tuple(memory_channels) not in allowed_memory_channels:
            raise ValueError(
                "LitePT-S* memory channels must be the native pyramid "
                f"{MASK3D_MEMORY_CHANNELS} or the dec0-pooled mechanism probe "
                f"{MASK3D_DEC0_POOLED_MEMORY_CHANNELS}, or the decoder-only "
                f"FPN {MASK3D_DECODER_ONLY_MEMORY_CHANNELS}"
            )
        if tuple(memory_strides) != MASK3D_MEMORY_STRIDES:
            raise ValueError(
                f"Mask3D memory strides must be {MASK3D_MEMORY_STRIDES}"
            )
        if len(sample_sizes) != 4:
            raise ValueError("sample_sizes must contain one value per memory level")
        if int(num_classes) < 2:
            raise ValueError(
                "Mask3D requires at least one foreground class plus no-object"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_cycles = int(num_cycles)
        self.train_queries = int(train_queries)
        self.eval_queries = int(eval_queries)
        self.num_classes = int(num_classes)
        self.no_object_class_id = self.num_classes - 1
        self.memory_channels = tuple(int(value) for value in memory_channels)
        self.memory_strides = tuple(int(value) for value in memory_strides)
        self.sample_sizes = tuple(int(value) for value in sample_sizes)
        self.attention_threshold = float(attention_threshold)

        self.mask_features_head = nn.Linear(72, hidden_dim)
        self.query_projection = _Mask3DQueryProjection(hidden_dim)
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mask_only = bool(mask_only)
        self.class_embed_head = None if self.mask_only else nn.Linear(hidden_dim, self.num_classes)
        self.position_encoding = Mask3DFourierPositionEncoding(
            hidden_dim, gauss_scale=gauss_scale
        )

        # Match upstream construction/initialization order: for each level,
        # create cross attention, memory squeeze, self attention, then FFN.
        # The four resulting level blocks are reused in all three cycles.
        cross_attention: list[nn.Module] = []
        memory_projections: list[nn.Module] = []
        self_attention: list[nn.Module] = []
        ffn_attention: list[nn.Module] = []
        for channels in self.memory_channels:
            cross_attention.append(
                _CrossAttentionLayer(hidden_dim, num_heads, dropout, pre_norm)
            )
            memory_projections.append(nn.Linear(channels, hidden_dim))
            self_attention.append(
                _SelfAttentionLayer(hidden_dim, num_heads, dropout, pre_norm)
            )
            ffn_attention.append(
                _FeedForwardLayer(
                    hidden_dim, dim_feedforward, dropout, pre_norm
                )
            )
        self.cross_attention = nn.ModuleList(cross_attention)
        self.memory_projections = nn.ModuleList(memory_projections)
        self.self_attention = nn.ModuleList(self_attention)
        self.ffn_attention = nn.ModuleList(ffn_attention)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    @property
    def num_queries(self) -> int:
        return self.train_queries if self.training else self.eval_queries

    def _predict(
        self,
        query: torch.Tensor,
        segment_mask_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized = self.decoder_norm(query)
        mask_embedding = self.mask_embed_head(normalized)
        segment_logits = mask_embedding @ segment_mask_features.transpose(0, 1)
        if self.mask_only:
            return {"segment_mask_logits": segment_logits}
        object_logits = self.class_embed_head(normalized)
        # Mask3D ranks a query by its best foreground class probability times
        # mask quality. In logit space this is the foreground-vs-no-object
        # log-odds, which reduces exactly to object_logit-no_object_logit for
        # the legacy two-class arm.
        score_logits = torch.logsumexp(
            object_logits[:, : self.no_object_class_id], dim=1
        ) - object_logits[:, self.no_object_class_id]
        return {
            "segment_mask_logits": segment_logits,
            "object_logits": object_logits,
            "score_logits": score_logits,
            "query_embed": normalized,
        }

    def _sample_memory(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        level_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        limit = self.sample_sizes[level_index]
        if not self.training or int(features.shape[0]) <= limit:
            return features, positions, attention_mask
        indices = torch.randperm(features.shape[0], device=features.device)[:limit]
        return features[indices], positions[indices], attention_mask[:, indices]

    def forward(
        self,
        pyramid: SceneLitePTPyramid,
        *,
        point_to_segment: torch.Tensor,
        voxel_to_segment: torch.Tensor,
        num_segments: int,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        observed_memory_channels = tuple(
            int(features.shape[1]) for features in pyramid.memory_features
        )
        if observed_memory_channels != self.memory_channels:
            raise ValueError(
                "Scene pyramid memory channels do not match the decoder: "
                f"{observed_memory_channels} != {self.memory_channels}"
            )
        if point_to_segment.shape != (pyramid.point_xyz.shape[0],):
            raise ValueError("point_to_segment shape does not match original points")
        if voxel_to_segment.shape != (pyramid.finest_features.shape[0],):
            raise ValueError("voxel_to_segment shape does not match finest voxels")

        mask_features = self.mask_features_head(pyramid.finest_features)
        segment_mask_features = _scatter_mean(
            mask_features, voxel_to_segment, int(num_segments)
        )
        query_indices = farthest_point_indices_from_zero(
            pyramid.finest_grid, self.num_queries
        )
        scene_min = pyramid.finest_xyz.min(dim=0).values
        scene_max = pyramid.finest_xyz.max(dim=0).values
        query_position = self.query_projection(
            self.position_encoding(
                pyramid.finest_xyz[query_indices],
                scene_min=scene_min,
                scene_max=scene_max,
            )
        ).unsqueeze(1)
        query = mask_features.new_zeros(
            (self.num_queries, 1, self.hidden_dim)
        )

        memory_positions = tuple(
            self.position_encoding(
                xyz,
                # Upstream Mask3D normalizes each memory resolution with
                # that level's own metric-coordinate range. Query positions
                # separately use the finest-resolution range above.
                scene_min=xyz.min(dim=0).values,
                scene_max=xyz.max(dim=0).values,
            )
            for xyz in pyramid.memory_xyz
        )
        parent_sizes = tuple(
            int(features.shape[0])
            for features in (
                pyramid.memory_features[3],
                pyramid.memory_features[2],
                pyramid.memory_features[1],
                pyramid.memory_features[0],
            )
        )

        auxiliary: list[dict[str, torch.Tensor]] = []
        debug_attention_shapes: list[tuple[int, int]] = []
        for _cycle in range(self.num_cycles):
            for level_index in range(4):
                prediction = self._predict(query[:, 0], segment_mask_features)
                auxiliary.append(prediction)

                finest_attention_logits = prediction["segment_mask_logits"][
                    :, voxel_to_segment.long()
                ].detach()
                pooled_logits = sequential_average_pool(
                    finest_attention_logits,
                    pyramid.parent_maps,
                    parent_sizes,
                )
                # pooled_logits is [stride2, stride4, stride8, stride16].
                hierarchy_index = 3 - level_index
                attention_logits = pooled_logits[hierarchy_index]
                attention_mask = (
                    attention_logits.sigmoid() < self.attention_threshold
                )
                all_masked = attention_mask.all(dim=1)
                attention_mask[all_masked] = False

                memory, memory_position, attention_mask = self._sample_memory(
                    pyramid.memory_features[level_index],
                    memory_positions[level_index],
                    attention_mask,
                    level_index,
                )
                all_masked = attention_mask.all(dim=1)
                attention_mask[all_masked] = False
                debug_attention_shapes.append(
                    (int(attention_mask.shape[0]), int(attention_mask.shape[1]))
                )
                head_mask = attention_mask.unsqueeze(0).repeat(
                    self.num_heads, 1, 1
                )
                projected_memory = self.memory_projections[level_index](memory)
                query = self.cross_attention[level_index](
                    query,
                    projected_memory.unsqueeze(1),
                    query_pos=query_position,
                    memory_pos=memory_position.unsqueeze(1),
                    attention_mask=head_mask,
                )
                query = self.self_attention[level_index](query, query_position)
                query = self.ffn_attention[level_index](query)

        result: dict[str, Any] = self._predict(query[:, 0], segment_mask_features)
        result["aux_outputs"] = auxiliary
        result["sampled_query_indices"] = query_indices
        result["sampled_query_xyz"] = pyramid.finest_xyz[query_indices]
        result["voxel_mask_logits"] = result["segment_mask_logits"][
            :, voxel_to_segment.long()
        ]
        result["mask_logits"] = result["segment_mask_logits"][
            :, point_to_segment.long()
        ]
        if return_debug:
            result["debug"] = {
                "attention_shapes": debug_attention_shapes,
                "normalized_query_xyz": self.position_encoding.normalize(
                    pyramid.finest_xyz[query_indices], scene_min, scene_max
                ),
                "normalized_memory_xyz": tuple(
                    self.position_encoding.normalize(
                        xyz,
                        xyz.min(dim=0).values,
                        xyz.max(dim=0).values,
                    )
                    for xyz in pyramid.memory_xyz
                ),
            }
        return result


def _tensor_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LitePTMask3DModel(nn.Module):
    """Complete LitePT-S* to Mask3D instance model.

    ``decoder_kwargs['num_classes']`` includes the no-object class, matching
    the upstream Mask3D convention. The default of two preserves the earlier
    class-agnostic diagnostic arm; ScanNet20 training uses 19 (18 foreground
    classes plus no-object).
    """

    def __init__(
        self,
        *,
        litept_root: str,
        backbone_frozen: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        backbone_config = dict(backbone_kwargs or {})
        allowed_backbone_keys = {
            "in_channels",
            "grid_size",
            "litept_variant",
            "feature_pyramid_mode",
            "voxel_reduce",
            "multi_scale",
            "representative_sampling",
            "cache_training_voxelization",
        }
        unexpected = sorted(set(backbone_config) - allowed_backbone_keys)
        if unexpected:
            raise ValueError(
                "LitePTMask3DModel received unsupported backbone overrides: "
                f"{unexpected}"
            )

        backbone_defaults: dict[str, Any] = {
            "in_channels": 6,
            "grid_size": 0.02,
            "litept_variant": "litept_s_star",
            "feature_pyramid_mode": "mask3d_fpn",
            "voxel_reduce": "representative",
            "multi_scale": False,
            "representative_sampling": "random",
            "cache_training_voxelization": False,
        }
        backbone_defaults.update(backbone_config)
        self.backbone = LitePTBackbone(
            litept_root=litept_root,
            **backbone_defaults,
        )
        decoder_config = dict(decoder_kwargs or {})
        feature_pyramid_mode = str(backbone_defaults["feature_pyramid_mode"])
        expected_channels_by_mode = {
            "mask3d_fpn": MASK3D_MEMORY_CHANNELS,
            "mask3d_dec0_pooled_fpn": MASK3D_DEC0_POOLED_MEMORY_CHANNELS,
            "mask3d_decoder_only_fpn": MASK3D_DECODER_ONLY_MEMORY_CHANNELS,
        }
        try:
            expected_memory_channels = expected_channels_by_mode[feature_pyramid_mode]
        except KeyError as error:
            raise ValueError(
                "Unsupported LitePT feature_pyramid_mode for Mask3D: "
                f"{feature_pyramid_mode!r}"
            ) from error
        configured_memory_channels = tuple(
            decoder_config.get("memory_channels", expected_memory_channels)
        )
        if configured_memory_channels != expected_memory_channels:
            raise ValueError(
                "Decoder memory channels do not match feature_pyramid_mode: "
                f"mode={feature_pyramid_mode!r} channels={configured_memory_channels} "
                f"expected={expected_memory_channels}"
            )
        decoder_config["memory_channels"] = expected_memory_channels
        self.decoder = LitePTMask3DDecoder(**decoder_config)
        self._backbone_frozen = False
        self._shuffle_orders_before_freeze: dict[str, bool] = {}
        self.set_backbone_frozen(backbone_frozen)

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def set_backbone_frozen(self, frozen: bool) -> None:
        requested = bool(frozen)
        shuffle_state = getattr(self, "_shuffle_orders_before_freeze", {})
        if requested and not shuffle_state:
            self._shuffle_orders_before_freeze = {
                name: bool(module.shuffle_orders)
                for name, module in self.backbone.named_modules()
                if hasattr(module, "shuffle_orders")
            }
        self._backbone_frozen = requested
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self._backbone_frozen)
        if self._backbone_frozen:
            # LitePT shuffles serialization orders independently of
            # ``module.training``. Disable every such switch so a frozen
            # representation is deterministic, then restore it when unfrozen.
            for _name, module in self.backbone.named_modules():
                if hasattr(module, "shuffle_orders"):
                    module.shuffle_orders = False
            self.backbone.eval()
        elif getattr(self, "_shuffle_orders_before_freeze", {}):
            for name, module in self.backbone.named_modules():
                if name in self._shuffle_orders_before_freeze:
                    module.shuffle_orders = self._shuffle_orders_before_freeze[name]
            self._shuffle_orders_before_freeze = {}

    def train(self, mode: bool = True) -> "LitePTMask3DModel":
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def backbone_state_hash(self) -> str:
        """Hash every backbone parameter and persistent buffer."""
        return _tensor_state_hash(self.backbone)

    def decoder_state_hash(self) -> str:
        """Hash every decoder tensor, including the Gaussian Fourier buffer."""
        return _tensor_state_hash(self.decoder)

    def parameter_groups(
        self,
        *,
        backbone_lr: float,
        decoder_lr: float,
        include_frozen_backbone: bool = False,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if include_frozen_backbone:
            # A staged-unfreeze run needs the backbone group present from the
            # first optimizer step so OneCycleLR can schedule it consistently.
            # The model may still have requires_grad=False during the initial
            # freeze window; no optimizer state or parameter update is created
            # until the group is unfrozen.
            backbone_parameters = list(self.backbone.parameters())
        else:
            backbone_parameters = [
                parameter
                for parameter in self.backbone.parameters()
                if parameter.requires_grad
            ]
        if backbone_parameters:
            groups.append(
                {"params": backbone_parameters, "lr": float(backbone_lr), "name": "backbone"}
            )
        decoder_parameters = [
            parameter
            for parameter in self.decoder.parameters()
            if parameter.requires_grad
        ]
        if decoder_parameters:
            groups.append(
                {"params": decoder_parameters, "lr": float(decoder_lr), "name": "decoder"}
            )
        return groups

    @staticmethod
    def _validate_offsets(
        offsets: torch.Tensor | None,
        num_points: int,
        device: torch.device,
    ) -> torch.Tensor:
        if offsets is None:
            return torch.tensor([num_points], device=device, dtype=torch.long)
        result = offsets.to(device=device, dtype=torch.long).flatten()
        if result.numel() == 0 or int(result[-1].item()) != num_points:
            raise ValueError("scene offsets must end at the concatenated point count")
        if bool((result[1:] <= result[:-1]).any().item()):
            raise ValueError("scene offsets must be strictly increasing")
        return result

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        scan_segments: torch.Tensor,
        *,
        scene_offsets: torch.Tensor | None = None,
        return_debug: bool = False,
    ) -> list[dict[str, Any]]:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be [N,3], got {tuple(points.shape)}")
        expected_channels = int(self.backbone.in_channels)
        if (
            features.ndim != 2
            or features.shape[0] != points.shape[0]
            or features.shape[1] != expected_channels
        ):
            raise ValueError(
                "features must have shape "
                f"[N,{expected_channels}], got {tuple(features.shape)}"
            )
        if scan_segments.shape != (points.shape[0],):
            raise ValueError("scan_segments must contain one id per input point")
        offsets = self._validate_offsets(scene_offsets, points.shape[0], points.device)

        backbone_context = torch.no_grad() if self._backbone_frozen else nullcontext()
        with backbone_context:
            pyramid = self.backbone(points, features, point_offsets=offsets)
        if not isinstance(pyramid, LitePTFeaturePyramid):
            raise TypeError(
                "LitePT feature-pyramid backbone did not return "
                "LitePTFeaturePyramid"
            )
        scene_pyramids = split_feature_pyramid(pyramid)

        segment_parts = [
            _slice_by_offsets(scan_segments, offsets, scene_index)
            for scene_index in range(int(offsets.numel()))
        ]
        outputs: list[dict[str, Any]] = []
        for scene_index, (scene, raw_segments) in enumerate(
            zip(scene_pyramids, segment_parts)
        ):
            segment_ids, point_to_segment = torch.unique(
                raw_segments.long(), sorted=True, return_inverse=True
            )
            voxel_to_segment = point_to_segment[scene.representative_indices.long()]
            prediction = self.decoder(
                scene,
                point_to_segment=point_to_segment,
                voxel_to_segment=voxel_to_segment,
                num_segments=int(segment_ids.numel()),
                return_debug=return_debug,
            )
            prediction.update(
                {
                    "scene_index": scene_index,
                    "segment_ids": segment_ids,
                    "point_to_segment": point_to_segment,
                    "voxel_to_segment": voxel_to_segment,
                    "point_to_voxel": scene.original_to_finest,
                    "representative_indices": scene.representative_indices,
                }
            )
            for auxiliary in prediction["aux_outputs"]:
                auxiliary["mask_logits"] = auxiliary["segment_mask_logits"][
                    :, point_to_segment.long()
                ]
                auxiliary["voxel_mask_logits"] = auxiliary["segment_mask_logits"][
                    :, voxel_to_segment.long()
                ]
            outputs.append(prediction)
        return outputs


__all__ = [
    "LitePTMask3DDecoder",
    "LitePTMask3DModel",
    "Mask3DFourierPositionEncoding",
    "SceneLitePTPyramid",
    "farthest_point_indices_from_zero",
    "sequential_average_pool",
    "split_feature_pyramid",
]
