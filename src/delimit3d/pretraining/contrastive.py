"""Contrastive projectors and pure-token supervision."""

from __future__ import annotations


from typing import Any, Mapping, Sequence


import torch

import torch.nn as nn

from delimit3d.training.contracts import TWO_D_CELLS, stable_seed

from delimit3d.data.partfield_contrastive import ContrastiveTripletBatch


from delimit3d.losses.litept_hierarchy_supervision import MaskSupervisionGroup


class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self, in_channels: int = 72, hidden_dim: int = 128, out_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def zero_loss_touching_module_parameters(
    reference: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Return scalar zero while keeping a module in the synchronized graph."""
    loss = reference.sum() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.sum() * 0.0
    return loss


def remap_contrastive_batch_to_level(
    batch: ContrastiveTripletBatch,
    point_to_level: torch.Tensor,
) -> tuple[ContrastiveTripletBatch | None, dict[str, int]]:
    """Map point-indexed tuples to one sparse feature level.

    A coarse LitePT level can merge both positive endpoints or a negative into
    a token sampled as positive for the same proposal. Keeping such a tuple
    would manufacture a false/cosine-one negative, so those rows and colliding
    candidates are removed explicitly. Repeated mapped tokens are deduplicated
    in original sampling order, and rows with no surviving base negative are
    dropped without suppressing valid rows. The remaining rows are
    rectangularized to the largest common negative count and can therefore be
    passed to the existing criterion unchanged.
    """

    if point_to_level.ndim != 1:
        raise ValueError(
            f"point_to_level must have shape [N], got {tuple(point_to_level.shape)}"
        )
    if point_to_level.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"point_to_level must be integral, got {point_to_level.dtype}")

    point_to_level = point_to_level.long()
    positive_pairs_all = point_to_level[batch.positive_pairs.long()]
    negative_indices_all = point_to_level[batch.negative_indices.long()]
    feature_candidates_all = (
        point_to_level[batch.feature_candidate_indices.long()]
        if batch.feature_candidate_indices is not None
        else None
    )

    input_pairs = int(positive_pairs_all.shape[0])
    collapsed_mask = positive_pairs_all[:, 0] == positive_pairs_all[:, 1]
    collapsed_positive_pairs = int(collapsed_mask.sum().item())
    valid_row_indices = torch.where(~collapsed_mask)[0]

    empty_stats = {
        "input_pairs": input_pairs,
        "valid_pairs": int(valid_row_indices.numel()),
        "kept_pairs": 0,
        "collapsed_positive_pairs": collapsed_positive_pairs,
        "zero_negative_pairs": 0,
        "base_negatives_per_pair": 0,
        "base_negative_collisions": 0,
        "base_negative_duplicates": 0,
        "feature_candidate_pool": 0,
        "feature_hard_negatives": 0,
        "feature_candidate_collisions": 0,
        "feature_candidate_duplicates": 0,
    }
    if valid_row_indices.numel() == 0:
        return None, empty_stats

    positive_pairs = positive_pairs_all[valid_row_indices]
    proposal_labels = batch.proposal_labels[valid_row_indices]
    negative_indices = negative_indices_all[valid_row_indices]

    def proposal_positive_collision_mask(values: torch.Tensor) -> torch.Tensor:
        """Mark tokens sampled as positive anywhere for the row's proposal."""
        collisions = torch.zeros_like(values, dtype=torch.bool)
        for proposal_label in torch.unique(proposal_labels, sorted=True):
            rows = proposal_labels == proposal_label
            all_rows = batch.proposal_labels == proposal_label
            positive_tokens = torch.unique(
                positive_pairs_all[all_rows].reshape(-1), sorted=True
            )
            collisions[rows] = torch.isin(values[rows], positive_tokens)
        return collisions

    def stable_unique_positions(
        values: torch.Tensor,
        invalid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return first-occurrence positions, per-row counts and diagnostics."""
        if values.ndim != 2 or invalid.shape != values.shape:
            raise ValueError("Mapped negative tensors must be rectangular [P,M]")
        width = int(values.shape[1])
        if width == 0:
            positions = torch.empty_like(values, dtype=torch.long)
            counts = torch.zeros(
                values.shape[0], dtype=torch.long, device=values.device
            )
            return positions, counts, 0, 0

        sorted_values, sorted_positions = torch.sort(
            values, dim=1, stable=True
        )
        sorted_invalid = torch.gather(invalid, 1, sorted_positions)
        first_for_value = torch.ones_like(sorted_values, dtype=torch.bool)
        first_for_value[:, 1:] = (
            sorted_values[:, 1:] != sorted_values[:, :-1]
        )
        usable = first_for_value & ~sorted_invalid
        sentinel = torch.full_like(sorted_positions, width)
        first_positions = torch.where(usable, sorted_positions, sentinel)
        # Sorting the original positions restores the sampler's deterministic
        # order and avoids preferentially retaining low sparse-token IDs.
        stable_positions = torch.sort(first_positions, dim=1).values
        counts = (stable_positions < width).sum(dim=1)
        duplicates = int((~first_for_value).sum().item())
        collisions = int((first_for_value & sorted_invalid).sum().item())
        return stable_positions, counts, duplicates, collisions

    base_positions, base_counts, base_duplicates, base_collisions = (
        stable_unique_positions(
            negative_indices,
            proposal_positive_collision_mask(negative_indices),
        )
    )
    rows_with_negatives = base_counts > 0
    zero_negative_pairs = int((~rows_with_negatives).sum().item())
    if not bool(rows_with_negatives.any()):
        empty_stats.update(
            {
                "zero_negative_pairs": zero_negative_pairs,
                "base_negative_collisions": base_collisions,
                "base_negative_duplicates": base_duplicates,
            }
        )
        return None, empty_stats

    positive_pairs = positive_pairs[rows_with_negatives]
    proposal_labels = proposal_labels[rows_with_negatives]
    row_indices = valid_row_indices[rows_with_negatives]
    negative_indices = negative_indices[rows_with_negatives]
    base_positions = base_positions[rows_with_negatives]
    base_counts = base_counts[rows_with_negatives]
    base_count = int(base_counts.min().item())
    remapped_negatives = torch.gather(
        negative_indices, 1, base_positions[:, :base_count]
    )

    candidate_pool = 0
    feature_hard = 0
    remapped_candidates = None
    candidate_duplicates = 0
    candidate_collisions = 0
    requested_feature_hard = int(batch.num_feature_hard_negatives)
    if requested_feature_hard > 0 and feature_candidates_all is not None:
        feature_candidates = feature_candidates_all[row_indices]
        candidate_positions, candidate_counts, candidate_duplicates, candidate_collisions = (
            stable_unique_positions(
                feature_candidates,
                proposal_positive_collision_mask(feature_candidates),
            )
        )
        candidate_pool = int(candidate_counts.min().item())
        feature_hard = min(requested_feature_hard, candidate_pool)
        if feature_hard > 0:
            remapped_candidates = torch.gather(
                feature_candidates,
                1,
                candidate_positions[:, :candidate_pool],
            )

    remapped = ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=remapped_negatives,
        proposal_labels=batch.proposal_labels[row_indices],
        feature_candidate_indices=remapped_candidates,
        num_feature_hard_negatives=feature_hard,
    )
    return remapped, {
        "input_pairs": input_pairs,
        "valid_pairs": int(valid_row_indices.numel()),
        "kept_pairs": int(remapped.num_pairs),
        "collapsed_positive_pairs": collapsed_positive_pairs,
        "zero_negative_pairs": zero_negative_pairs,
        "base_negatives_per_pair": base_count,
        "base_negative_collisions": base_collisions,
        "base_negative_duplicates": base_duplicates,
        "feature_candidate_pool": candidate_pool,
        "feature_hard_negatives": feature_hard,
        "feature_candidate_collisions": candidate_collisions,
        "feature_candidate_duplicates": candidate_duplicates,
    }


def build_pure_token_contrastive_batch(
    batch: ContrastiveTripletBatch,
    point_to_level: torch.Tensor,
    level_xyz: torch.Tensor,
    *,
    level_name: str,
    include_proposal_diagnostics: bool = False,
) -> tuple[ContrastiveTripletBatch | None, dict[str, Any]]:
    """Sample one fixed-budget contrastive batch from strictly pure tokens.

    Point pairs cannot simply be mapped to a coarse hierarchy: a sparse token
    may contain both proposal and complement points, turning it into a false
    positive or false hard negative.  This routine instead aggregates the
    exact sparse proposal membership carried by ``ContrastiveTripletBatch``.
    A token is accepted as positive only when *all* source points mapped to it
    belong to the proposal, and as negative only when all source points are in
    the proposal's eligible complement.  Mixed and visibility-unknown tokens
    are ignored.  Every retained proposal is resampled to its original pair
    budget, avoiding the proposal-size bias caused by conditioning point pairs
    on coarse-level survival.
    """

    metadata = (
        batch.proposal_member_offsets,
        batch.proposal_member_indices,
        batch.pair_proposal_indices,
        batch.eligible_indices,
    )
    if any(value is None for value in metadata):
        raise ValueError(
            "Multi-scale pure-token sampling requires exact proposal membership"
        )
    assert batch.proposal_member_offsets is not None
    assert batch.proposal_member_indices is not None
    assert batch.pair_proposal_indices is not None
    assert batch.eligible_indices is not None
    if batch.multiscale_seed is None:
        raise ValueError("Multi-scale pure-token sampling requires a frozen seed")
    if point_to_level.ndim != 1 or point_to_level.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("point_to_level must be an integral [N] tensor")
    if level_xyz.ndim != 2 or level_xyz.shape[1] != 3:
        raise ValueError("level_xyz must have shape [V,3]")
    if point_to_level.device != level_xyz.device:
        raise ValueError("point_to_level and level_xyz must share a device")

    point_to_level = point_to_level.long()
    offsets = batch.proposal_member_offsets.long()
    members = batch.proposal_member_indices.long()
    pair_to_proposal = batch.pair_proposal_indices.long()
    eligible = torch.unique(batch.eligible_indices.long(), sorted=True)
    num_points = int(point_to_level.numel())
    num_tokens = int(level_xyz.shape[0])
    num_proposals = int(offsets.numel()) - 1
    if (
        num_tokens <= 0
        or num_proposals <= 0
        or offsets.ndim != 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(members.numel())
        or pair_to_proposal.shape != (batch.num_pairs,)
    ):
        raise ValueError("Malformed proposal-membership CSR contract")
    index_chunks = [members, eligible, pair_to_proposal]
    if any(tensor.device != point_to_level.device for tensor in index_chunks):
        raise ValueError("Multi-scale proposal metadata must share the feature device")
    if members.numel() == 0 or eligible.numel() == 0:
        raise ValueError("Proposal membership and eligibility must be non-empty")
    if (
        int(members.min()) < 0
        or int(members.max()) >= num_points
        or int(eligible.min()) < 0
        or int(eligible.max()) >= num_points
        or int(pair_to_proposal.min()) < 0
        or int(pair_to_proposal.max()) >= num_proposals
    ):
        raise ValueError("Proposal-membership metadata contains an invalid index")
    if not bool(torch.isin(members, eligible).all()):
        raise ValueError("Proposal members must be a subset of eligible points")
    if point_to_level.numel() and (
        int(point_to_level.min()) < 0 or int(point_to_level.max()) >= num_tokens
    ):
        raise ValueError("point_to_level contains an out-of-range token")

    all_counts = torch.bincount(point_to_level, minlength=num_tokens)
    eligible_counts = torch.bincount(
        point_to_level[eligible], minlength=num_tokens
    )
    generator = torch.Generator(device=point_to_level.device).manual_seed(
        stable_seed(
            int(batch.multiscale_seed),
            "pure-token-multiscale",
            str(level_name),
        )
    )

    requested_uniform = int(batch.num_uniform_negatives)
    requested_spatial = int(batch.num_spatial_hard_negatives)
    requested_feature_hard = int(batch.num_feature_hard_negatives)
    requested_spatial_pool = int(batch.spatial_candidate_pool)
    requested_feature_pool = int(batch.feature_candidate_pool)
    if requested_uniform < 0 or requested_spatial < 0:
        raise ValueError("Negative-category counts must be non-negative")
    if requested_uniform + requested_spatial <= 0:
        raise ValueError("Pure-token supervision requires base negatives")
    if requested_spatial > 0 and requested_spatial_pool <= 0:
        raise ValueError("Spatial hard negatives require a candidate pool")
    if requested_feature_hard > 0 and requested_feature_pool <= 0:
        raise ValueError("Feature hard negatives require a candidate pool")

    records: list[dict[str, torch.Tensor]] = []
    total_pure_positive = 0
    total_pure_negative = 0
    total_mixed_or_unknown = 0
    dropped_insufficient_positive = 0
    dropped_no_negative = 0
    input_pair_budget = 0
    retained_pair_budget = 0
    proposal_diagnostics: list[dict[str, int | str | bool]] = []

    with torch.no_grad():
        for proposal_index in range(num_proposals):
            pair_rows = torch.where(pair_to_proposal == proposal_index)[0]
            pair_budget = int(pair_rows.numel())
            if pair_budget <= 0:
                raise ValueError("Every sampled proposal occurrence needs pair rows")
            input_pair_budget += pair_budget
            start = int(offsets[proposal_index])
            end = int(offsets[proposal_index + 1])
            proposal_members = torch.unique(members[start:end], sorted=True)
            positive_counts = torch.bincount(
                point_to_level[proposal_members], minlength=num_tokens
            )

            # Strict means strict over every original scene point mapped to the
            # token, not only points visible in a 2D frame.  Visibility-unknown
            # support therefore cannot masquerade as pure proposal support.
            pure_positive = torch.where(
                (positive_counts > 0) & (positive_counts == all_counts)
            )[0]
            pure_negative = torch.where(
                (positive_counts == 0)
                & (eligible_counts > 0)
                & (eligible_counts == all_counts)
            )[0]
            mixed_or_unknown = torch.where(
                (all_counts > 0)
                & ~(
                    ((positive_counts > 0) & (positive_counts == all_counts))
                    | (
                        (positive_counts == 0)
                        & (eligible_counts > 0)
                        & (eligible_counts == all_counts)
                    )
                )
            )[0]
            total_pure_positive += int(pure_positive.numel())
            total_pure_negative += int(pure_negative.numel())
            total_mixed_or_unknown += int(mixed_or_unknown.numel())

            proposal_row: dict[str, int | str | bool] = {
                "proposal_index": proposal_index,
                "pair_budget": pair_budget,
                "member_point_count": int(proposal_members.numel()),
                "pure_positive_token_count": int(pure_positive.numel()),
                "pure_negative_token_count": int(pure_negative.numel()),
                "mixed_or_unknown_token_count": int(mixed_or_unknown.numel()),
                "occupied_token_count": int((all_counts > 0).sum()),
                "retained": False,
                "drop_reason": "",
            }
            if pure_positive.numel() < 2:
                dropped_insufficient_positive += 1
                proposal_row["drop_reason"] = "insufficient_pure_positive"
                proposal_diagnostics.append(proposal_row)
                continue
            if pure_negative.numel() == 0:
                dropped_no_negative += 1
                proposal_row["drop_reason"] = "no_pure_negative"
                proposal_diagnostics.append(proposal_row)
                continue

            positive_a_local = torch.randint(
                int(pure_positive.numel()),
                (pair_budget,),
                generator=generator,
                device=point_to_level.device,
            )
            positive_offset = torch.randint(
                int(pure_positive.numel()) - 1,
                (pair_budget,),
                generator=generator,
                device=point_to_level.device,
            )
            positive_b_local = (
                positive_a_local + 1 + positive_offset
            ) % int(pure_positive.numel())
            positive_pairs = torch.stack(
                [
                    pure_positive[positive_a_local],
                    pure_positive[positive_b_local],
                ],
                dim=1,
            )

            uniform = point_to_level.new_empty((pair_budget, 0))
            if requested_uniform > 0:
                uniform_draw = torch.randint(
                    int(pure_negative.numel()),
                    (pair_budget, requested_uniform),
                    generator=generator,
                    device=point_to_level.device,
                )
                uniform = pure_negative[uniform_draw]

            spatial = point_to_level.new_empty((pair_budget, 0))
            if requested_spatial > 0:
                spatial_pool_count = min(
                    requested_spatial_pool, int(pure_negative.numel())
                )
                spatial_pool = pure_negative[
                    torch.randperm(
                        int(pure_negative.numel()),
                        generator=generator,
                        device=point_to_level.device,
                    )[:spatial_pool_count]
                ]
                spatial_count = min(requested_spatial, spatial_pool_count)
                distances = torch.linalg.vector_norm(
                    level_xyz[spatial_pool][None, :, :]
                    - level_xyz[positive_pairs[:, 0]][:, None, :],
                    dim=-1,
                )
                nearest = torch.topk(
                    distances,
                    k=spatial_count,
                    dim=1,
                    largest=False,
                ).indices
                spatial = spatial_pool[nearest]

            candidates = point_to_level.new_empty((pair_budget, 0))
            if requested_feature_hard > 0:
                candidate_count = min(
                    requested_feature_pool, int(pure_negative.numel())
                )
                candidate_pool = pure_negative[
                    torch.randperm(
                        int(pure_negative.numel()),
                        generator=generator,
                        device=point_to_level.device,
                    )[:candidate_count]
                ]
                candidates = candidate_pool[None, :].expand(pair_budget, -1)

            proposal_label = batch.proposal_labels[pair_rows[0]].expand(
                pair_budget
            )
            records.append(
                {
                    "positive_pairs": positive_pairs,
                    "uniform": uniform,
                    "spatial": spatial,
                    "candidates": candidates,
                    "proposal_labels": proposal_label,
                }
            )
            proposal_row["retained"] = True
            proposal_diagnostics.append(proposal_row)
            retained_pair_budget += pair_budget

    base_stats: dict[str, int | float] = {
        "input_pairs": int(batch.num_pairs),
        "input_proposals": num_proposals,
        "valid_pairs": retained_pair_budget,
        "kept_pairs": retained_pair_budget,
        "kept_proposals": len(records),
        "dropped_proposals_insufficient_pure_positive": (
            dropped_insufficient_positive
        ),
        "dropped_proposals_no_pure_negative": dropped_no_negative,
        "pure_positive_tokens": total_pure_positive,
        "pure_negative_tokens": total_pure_negative,
        "mixed_or_unknown_tokens_ignored": total_mixed_or_unknown,
        "accepted_mixed_tokens": 0,
        "collapsed_positive_pairs": 0,
        "zero_negative_pairs": input_pair_budget - retained_pair_budget,
        "base_negative_collisions": 0,
        "base_negative_duplicates": -1,
        "feature_candidate_collisions": 0,
        "feature_candidate_duplicates": -1,
        "strict_token_purity": 1.0,
        "negative_diversity_measured": 0,
    }
    if include_proposal_diagnostics:
        base_stats["proposal_diagnostics"] = proposal_diagnostics
    if not records:
        base_stats.update(
            {
                "base_negatives_per_pair": 0,
                "uniform_negatives_per_pair": 0,
                "spatial_hard_negatives_per_pair": 0,
                "feature_candidate_pool": 0,
                "feature_hard_negatives": 0,
            }
        )
        return None, base_stats

    spatial_count = min(int(record["spatial"].shape[1]) for record in records)
    candidate_pool = min(
        int(record["candidates"].shape[1]) for record in records
    )
    feature_hard = min(requested_feature_hard, candidate_pool)
    negative_indices = torch.cat(
        [
            torch.cat(
                [
                    record["uniform"],
                    record["spatial"][:, :spatial_count],
                ],
                dim=1,
            )
            for record in records
        ],
        dim=0,
    )
    remapped = ContrastiveTripletBatch(
        positive_pairs=torch.cat(
            [record["positive_pairs"] for record in records], dim=0
        ),
        negative_indices=negative_indices,
        proposal_labels=torch.cat(
            [record["proposal_labels"] for record in records], dim=0
        ),
        feature_candidate_indices=(
            torch.cat(
                [
                    record["candidates"][:, :candidate_pool]
                    for record in records
                ],
                dim=0,
            )
            if feature_hard > 0
            else None
        ),
        num_feature_hard_negatives=feature_hard,
    )
    if include_proposal_diagnostics:
        sorted_base = torch.sort(negative_indices, dim=1).values
        base_unique = 1 + (
            sorted_base[:, 1:] != sorted_base[:, :-1]
        ).sum(dim=1)
        base_duplicates = int(
            (negative_indices.shape[1] - base_unique).sum()
        )
        base_stats.update(
            {
                "base_negative_duplicates": base_duplicates,
                "base_negative_duplicate_fraction": (
                    base_duplicates / float(negative_indices.numel())
                ),
                "base_negative_unique_per_pair_min": int(base_unique.min()),
                "base_negative_unique_per_pair_mean": float(
                    base_unique.float().mean()
                ),
                "negative_diversity_measured": 1,
            }
        )
        if feature_hard > 0:
            assert remapped.feature_candidate_indices is not None
            feature_candidates = remapped.feature_candidate_indices
            sorted_candidates = torch.sort(
                feature_candidates, dim=1
            ).values
            candidate_unique = 1 + (
                sorted_candidates[:, 1:] != sorted_candidates[:, :-1]
            ).sum(dim=1)
            candidate_duplicates = int(
                (feature_candidates.shape[1] - candidate_unique).sum()
            )
            combined = torch.cat(
                [negative_indices, feature_candidates], dim=1
            )
            sorted_combined = torch.sort(combined, dim=1).values
            combined_unique = 1 + (
                sorted_combined[:, 1:] != sorted_combined[:, :-1]
            ).sum(dim=1)
            overlap = base_unique + candidate_unique - combined_unique
            base_stats.update(
                {
                    "feature_candidate_duplicates": candidate_duplicates,
                    "feature_candidate_unique_per_pair_min": int(
                        candidate_unique.min()
                    ),
                    "feature_candidate_unique_per_pair_mean": float(
                        candidate_unique.float().mean()
                    ),
                    "feature_candidate_base_overlap_unique_per_pair_mean": float(
                        overlap.float().mean()
                    ),
                    "feature_candidate_base_overlap_unique_per_pair_max": int(
                        overlap.max()
                    ),
                }
            )
    base_stats.update(
        {
            "base_negatives_per_pair": int(negative_indices.shape[1]),
            "uniform_negatives_per_pair": requested_uniform,
            "spatial_hard_negatives_per_pair": spatial_count,
            "feature_candidate_pool": candidate_pool,
            "feature_hard_negatives": feature_hard,
        }
    )
    return remapped, base_stats


def normalize_hierarchy_frame_groups(
    *,
    batch: ContrastiveTripletBatch | None,
    source_cell: str | None,
    frame_groups: Sequence[Any] | None,
) -> list[MaskSupervisionGroup]:
    """Normalize legacy, tuple, and sampler-v2 frame-group inputs.

    Sampler-v2 objects are consumed structurally to avoid coupling this module
    to the sampler implementation: only ``.cell``, ``.batch``, and optional
    ``.frame_id`` are required.
    """

    if frame_groups is None:
        if batch is None:
            raise ValueError("Either batch or frame_groups must be provided")
        if not isinstance(batch, ContrastiveTripletBatch):
            raise TypeError("batch must be a ContrastiveTripletBatch")
        return [
            MaskSupervisionGroup(
                cell=str(source_cell) if source_cell is not None else "legacy",
                batch=batch,
                group_id=None,
            )
        ]
    if batch is not None:
        raise ValueError("Pass batch or frame_groups, not both")
    if source_cell is not None:
        raise ValueError("source_cell belongs to the single-batch API")
    normalized: list[MaskSupervisionGroup] = []
    for index, group in enumerate(frame_groups):
        if hasattr(group, "cell") and hasattr(group, "batch"):
            cell = str(group.cell)
            group_batch = group.batch
            group_id_value = getattr(group, "frame_id", None)
            if group_id_value is None:
                group_id_value = getattr(group, "group_id", None)
        elif isinstance(group, (tuple, list)) and len(group) in (2, 3):
            if len(group) == 2:
                cell, group_batch = group
                group_id_value = None
            else:
                cell, group_id_value, group_batch = group
            cell = str(cell)
        else:
            raise TypeError(
                "frame_groups entries must expose .cell/.batch or be "
                "(cell, batch)/(cell, group_id, batch) tuples"
            )
        if not isinstance(group_batch, ContrastiveTripletBatch):
            raise TypeError(
                f"frame_groups[{index}] does not contain a ContrastiveTripletBatch"
            )
        normalized.append(
            MaskSupervisionGroup(
                cell=cell,
                batch=group_batch,
                group_id=(
                    None if group_id_value is None else str(group_id_value)
                ),
            )
        )
    if not normalized:
        raise ValueError("frame_groups must not be empty")
    return normalized


def _cell_balanced_contrastive_results(
    records: Sequence[tuple[MaskSupervisionGroup, dict[str, Any]]],
) -> dict[str, Any]:
    """Pair-weight frame losses within cell, then equally average cells."""

    if not records:
        raise ValueError("At least one contrastive result is required")
    cells = tuple(dict.fromkeys(group.cell for group, _ in records))
    by_cell = {
        cell: [result for group, result in records if group.cell == cell]
        for cell in cells
    }
    cell_pair_counts = {
        cell: sum(int(result["num_positive_pairs"]) for result in results)
        for cell, results in by_cell.items()
    }
    if any(count <= 0 for count in cell_pair_counts.values()):
        raise ValueError("Every contrastive frame group must realize positive pairs")
    cell_loss_tensors = {}
    for cell, results in by_cell.items():
        denominator = float(cell_pair_counts[cell])
        zero = results[0]["loss_total"].new_zeros(())
        cell_loss_tensors[cell] = sum(
            (
                result["loss_total"] * float(result["num_positive_pairs"])
                for result in results
            ),
            zero,
        ) / denominator
    loss = torch.stack(list(cell_loss_tensors.values())).mean()
    average_keys = (
        "temperature",
        "positive_cosine_mean",
        "negative_cosine_mean",
        "cosine_gap",
        "triplet_ranking_accuracy",
        "hardest_negative_ranking_accuracy",
        "num_negatives_per_pair",
        "num_base_negatives_per_pair",
        "num_feature_hard_negatives_per_pair",
    )
    output: dict[str, Any] = {
        "loss_total": loss,
        "loss_contrastive": float(loss.detach()),
        "num_positive_pairs": sum(
            int(result["num_positive_pairs"]) for _, result in records
        ),
    }
    cells_metrics: dict[str, dict[str, Any]] = {}
    for cell, results in by_cell.items():
        cell_metrics: dict[str, Any] = {
            "group_count": len(results),
            "loss": float(cell_loss_tensors[cell].detach()),
            "num_positive_pairs": cell_pair_counts[cell],
            "loss_denominator_positive_pairs": cell_pair_counts[cell],
        }
        for key in average_keys:
            cell_metrics[key] = sum(
                float(result[key]) * int(result["num_positive_pairs"])
                for result in results
            ) / float(cell_pair_counts[cell])
        cells_metrics[cell] = cell_metrics
    for key in average_keys:
        output[key] = sum(float(cells_metrics[cell][key]) for cell in cells) / len(
            cells
        )
    output["frame_group_metrics"] = {
        "normalization": "pair_mean_within_cell_then_equal_mean_across_nonempty_cells",
        "group_count": len(records),
        "cell_count": len(cells),
        "cell_equalization_denominator": len(cells),
        "cells": cells_metrics,
        "zero_denominator": False,
    }
    return output


def _zero_contrastive_result(
    reference: torch.Tensor,
    criterion: nn.Module,
) -> dict[str, Any]:
    """Return the normal contrastive schema with a graph-connected zero loss.

    A V2 scene can legitimately lose every point-space pair after exact dec0
    token-purity and co-membership filtering.  That is an optimization-validity
    decision for the synchronized runner, not a rank-local exception.  Touching
    ``reference`` keeps the backbone/projector in the graph, while touching the
    criterion keeps its private learnable temperature safe with
    ``find_unused_parameters=False``.
    """

    loss = zero_loss_touching_module_parameters(reference, criterion)
    raw_temperature = getattr(criterion, "temperature", 0.0)
    if isinstance(raw_temperature, torch.Tensor):
        temperature = float(raw_temperature.detach())
    else:
        temperature = float(raw_temperature)
    return {
        "loss_total": loss,
        "loss_contrastive": 0.0,
        "temperature": temperature,
        "positive_cosine_mean": 0.0,
        "negative_cosine_mean": 0.0,
        "cosine_gap": 0.0,
        "triplet_ranking_accuracy": 0.0,
        "hardest_negative_ranking_accuracy": 0.0,
        "num_positive_pairs": 0,
        "num_negatives_per_pair": 0,
        "num_base_negatives_per_pair": 0,
        "num_feature_hard_negatives_per_pair": 0,
        "frame_group_metrics": {
            "normalization": (
                "pair_mean_within_cell_then_equal_mean_across_nonempty_cells"
            ),
            "group_count": 0,
            "cell_count": 0,
            "cell_equalization_denominator": 0,
            "cells": {},
            "zero_denominator": True,
        },
    }


def _token_pure_dec0_optimization_metrics(
    *,
    groups: Sequence[MaskSupervisionGroup],
    token_records: Sequence[tuple[MaskSupervisionGroup, Mapping[str, Any]]],
    group_metrics: Sequence[Mapping[str, Any]],
    expected_cells: Sequence[str] = TWO_D_CELLS,
) -> dict[str, Any]:
    """Build an exact, runner-facing V2 validity contract.

    Counts here are measured *after* point-to-dec0 remapping.  They must not be
    confused with the sampler's point-space pair yield.  Missing expected cells
    remain explicit zero-denominator rows so aggregation cannot silently erase
    a starved granularity.
    """

    expected_cells = tuple(expected_cells)
    input_cells = tuple(dict.fromkeys(group.cell for group in groups))
    retained_cells = tuple(dict.fromkeys(group.cell for group, _ in token_records))
    unexpected_input_cells = tuple(
        cell for cell in input_cells if cell not in expected_cells
    )
    report_cells = expected_cells + tuple(
        cell for cell in unexpected_input_cells if cell not in expected_cells
    )
    cells: dict[str, dict[str, Any]] = {}
    for cell in report_cells:
        retained = [
            result for group, result in token_records if group.cell == cell
        ]
        denominator = sum(
            int(result["num_positive_pairs"]) for result in retained
        )
        cells[cell] = {
            "input_group_count": sum(group.cell == cell for group in groups),
            "retained_group_count": len(retained),
            "realized_positive_pairs": denominator,
            "loss_denominator_positive_pairs": denominator,
            "zero_denominator": denominator == 0,
            "optimization_valid": denominator > 0,
        }

    missing_input_cells = [cell for cell in expected_cells if cell not in input_cells]
    missing_retained_cells = [
        cell for cell in expected_cells if cells[cell]["zero_denominator"]
    ]
    exact_input_cell_set = (
        set(input_cells) == set(expected_cells) and not unexpected_input_cells
    )
    exact_retained_cell_set = (
        set(retained_cells) == set(expected_cells)
        and not any(cell not in expected_cells for cell in retained_cells)
    )
    expected_three_cell_valid = bool(
        exact_input_cell_set
        and exact_retained_cell_set
        and all(cells[cell]["optimization_valid"] for cell in expected_cells)
    )
    realized_by_cell = {
        cell: int(cells[cell]["realized_positive_pairs"])
        for cell in report_cells
    }
    denominators_by_cell = {
        cell: int(cells[cell]["loss_denominator_positive_pairs"])
        for cell in report_cells
    }
    return {
        "schema_version": "litept_token_pure_dec0_optimization/v2",
        "enabled": True,
        "optimization_valid": expected_three_cell_valid,
        "primary_loss_valid": bool(token_records),
        (
            "expected_three_cell_valid" if len(expected_cells) == 3
            else "expected_single_cell_valid"
        ): expected_three_cell_valid,
        "expected_input_cell_set_valid": exact_input_cell_set,
        "expected_retained_cell_set_valid": exact_retained_cell_set,
        "normalization": (
            "pair_mean_within_cell_then_equal_mean_nonempty_cells"
        ),
        "expected_cells": list(expected_cells),
        "input_cells": list(input_cells),
        "retained_cells": list(retained_cells),
        "missing_input_cells": missing_input_cells,
        "missing_retained_cells": missing_retained_cells,
        "unexpected_input_cells": list(unexpected_input_cells),
        "empty_cells": [
            cell for cell in report_cells if cells[cell]["zero_denominator"]
        ],
        "cell_equalization_denominator": len(retained_cells),
        "input_group_count": len(groups),
        "retained_group_count": len(token_records),
        "realized_positive_pairs": sum(realized_by_cell.values()),
        "loss_denominator_positive_pairs": sum(denominators_by_cell.values()),
        "realized_positive_pairs_by_cell": realized_by_cell,
        "loss_denominators_by_cell": denominators_by_cell,
        "zero_denominator": not bool(token_records),
        "cells": cells,
        "groups": [dict(metrics) for metrics in group_metrics],
    }
