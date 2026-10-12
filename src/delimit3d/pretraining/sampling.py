"""Deterministic scene and proposal sampling."""

from __future__ import annotations


from dataclasses import replace


from typing import Any, Iterable, Sequence

import numpy as np

import torch


from delimit3d.training.contracts import GRANULARITY_KEYS, SOURCE_CELL_SCHEDULE_FIXED, normalize_source_cells, source_cell_for_scene_epoch, stable_seed

from delimit3d.data.partfield_contrastive import ContrastiveTripletBatch, concatenate_triplet_batches, sample_overlapping_proposal_triplets, sample_partition_triplets


from .settings import (
    SamplingConfig,
    TRAINING_SAMPLING_MODES,
)
from .sources import (
    RawFrame,
    SceneSource,
)


def _generator(device: torch.device, *parts: Any) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stable_seed(*parts))


def sample_raw_batch(
    *,
    scene: SceneSource,
    frame: RawFrame,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed_parts: Iterable[Any],
    selected_proposal_indices: torch.Tensor | None = None,
) -> ContrastiveTripletBatch:
    generator = _generator(device, *seed_parts)
    return sample_overlapping_proposal_triplets(
        num_scene_points=int(scene.points.shape[0]),
        visible_indices=torch.from_numpy(frame.visible_indices).to(device),
        proposal_offsets=torch.from_numpy(frame.proposal_offsets).to(device),
        proposal_point_indices=torch.from_numpy(frame.proposal_point_indices).to(device),
        points=points,
        num_proposals=sampling.proposals_per_forward,
        positive_pairs_per_proposal=sampling.positive_pairs_per_proposal,
        num_uniform_negatives=sampling.uniform_negatives,
        num_spatial_hard_negatives=sampling.spatial_hard_negatives,
        spatial_candidate_pool=sampling.spatial_candidate_pool,
        num_feature_hard_negatives=sampling.feature_hard_negatives,
        feature_candidate_pool=sampling.feature_candidate_pool,
        selected_proposal_indices=selected_proposal_indices,
        generator=generator,
    )


def sample_3d_batch(
    *,
    labels: torch.Tensor,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed_parts: Iterable[Any],
    selected_proposal_labels: torch.Tensor | None = None,
) -> ContrastiveTripletBatch:
    generator = _generator(device, *seed_parts)
    return sample_partition_triplets(
        labels,
        eligible_mask=labels >= 0,
        points=points,
        num_proposals=sampling.proposals_per_forward,
        positive_pairs_per_proposal=sampling.positive_pairs_per_proposal,
        num_uniform_negatives=sampling.uniform_negatives,
        num_spatial_hard_negatives=sampling.spatial_hard_negatives,
        spatial_candidate_pool=sampling.spatial_candidate_pool,
        num_feature_hard_negatives=sampling.feature_hard_negatives,
        feature_candidate_pool=sampling.feature_candidate_pool,
        ignored_proposal_labels=(-1,),
        exclude_ignored_from_negatives=True,
        selected_proposal_labels=selected_proposal_labels,
        generator=generator,
    )


def coverage_queue_indices(
    *,
    population_size: int,
    start: int,
    count: int,
    seed_parts: Iterable[Any],
) -> list[int]:
    """Draw deterministic queue positions without replacement within each cycle."""

    population_size = int(population_size)
    start = int(start)
    count = int(count)
    if population_size <= 0:
        raise ValueError("coverage queue population_size must be positive")
    if start < 0 or count <= 0:
        raise ValueError("coverage queue start/count must be non-negative/positive")
    parts = tuple(seed_parts)
    permutations: dict[int, np.ndarray] = {}
    selected: list[int] = []
    for queue_position in range(start, start + count):
        cycle, within_cycle = divmod(queue_position, population_size)
        if cycle not in permutations:
            permutations[cycle] = np.random.default_rng(
                stable_seed(*parts, cycle)
            ).permutation(population_size)
        selected.append(int(permutations[cycle][within_cycle]))
    return selected


def _valid_raw_proposal_indices(frame: RawFrame) -> np.ndarray:
    sizes = frame.proposal_offsets[1:] - frame.proposal_offsets[:-1]
    return np.where(
        (sizes >= 2) & (sizes < int(frame.visible_indices.size))
    )[0].astype(np.int64, copy=False)


def _valid_partition_labels(labels: torch.Tensor) -> torch.Tensor:
    eligible = labels >= 0
    proposal_labels, counts = torch.unique(
        labels[eligible],
        sorted=True,
        return_counts=True,
    )
    eligible_count = int(eligible.sum())
    valid = (counts >= 2) & ((eligible_count - counts) >= 1)
    selected = proposal_labels[valid]
    if selected.numel() == 0:
        raise ValueError("No valid 3D proposal label for coverage sampling")
    return selected


def _coverage_raw_training_batch(
    *,
    scene: SceneSource,
    granularity: str,
    cell: str,
    epoch: int,
    occurrence: int,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
) -> tuple[ContrastiveTripletBatch, dict[str, Any]]:
    frames = tuple(
        frame
        for frame in scene.frames(granularity, "train")
        if _valid_raw_proposal_indices(frame).size > 0
    )
    if not frames:
        raise ValueError(
            f"{scene.scene_id}/{granularity}: no proposal-bearing train frame"
        )
    proposal_count = int(sampling.proposals_per_forward)
    queue_start = int(occurrence) * proposal_count
    frame_history = coverage_queue_indices(
        population_size=len(frames),
        start=0,
        count=queue_start + proposal_count,
        seed_parts=(scene.scene_id, granularity, "coverage-frame"),
    )
    current_frame_indices = frame_history[queue_start:]
    one_proposal = replace(sampling, proposals_per_forward=1)
    batches: list[ContrastiveTripletBatch] = []
    frame_ids: list[str] = []
    proposal_ids: list[str] = []
    for local_draw, frame_index in enumerate(current_frame_indices):
        frame = frames[frame_index]
        history_position = queue_start + local_draw
        prior_uses = sum(
            int(previous == frame_index)
            for previous in frame_history[:history_position]
        )
        valid_proposals = _valid_raw_proposal_indices(frame)
        if valid_proposals.size == 0:
            raise ValueError(
                f"{scene.scene_id}/{granularity}/{frame.physical_frame_id}: "
                "coverage frame has no valid proposal"
            )
        proposal_queue_index = coverage_queue_indices(
            population_size=int(valid_proposals.size),
            start=prior_uses,
            count=1,
            seed_parts=(
                scene.scene_id,
                granularity,
                frame.physical_frame_id,
                "coverage-proposal",
            ),
        )[0]
        proposal_index = int(valid_proposals[proposal_queue_index])
        batch = sample_raw_batch(
            scene=scene,
            frame=frame,
            points=points,
            device=device,
            sampling=one_proposal,
            seed_parts=(
                scene.scene_id,
                epoch,
                cell,
                frame.physical_frame_id,
                proposal_index,
                "coverage-train",
            ),
            selected_proposal_indices=torch.tensor(
                [proposal_index],
                dtype=torch.long,
                device=device,
            ),
        )
        # Coverage intentionally draws proposals from different RGB-D frames,
        # so there is no single eligible-point universe or seed for the
        # concatenated batch.  Preserve the historical single-scale coverage
        # path by dropping only the optional multi-scale CSR.  A multi-scale
        # run that selects coverage mode then fails explicitly in the strict
        # pure-token sampler instead of consuming invalid metadata.
        batches.append(
            replace(
                batch,
                proposal_member_offsets=None,
                proposal_member_indices=None,
                pair_proposal_indices=None,
                eligible_indices=None,
                multiscale_seed=None,
            )
        )
        frame_ids.append(frame.physical_frame_id)
        proposal_ids.append(f"{frame.physical_frame_id}:{proposal_index}")
    return concatenate_triplet_batches(batches), {
        "cell": cell,
        "family": "2d",
        "granularity": granularity,
        "frame_ids": frame_ids,
        "proposal_ids": proposal_ids,
    }


def training_batch(
    *,
    scene: SceneSource,
    epoch: int,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    repeat_g08_for_all_3d_slots: bool = False,
    sampling_mode: str = "random_with_replacement",
    return_metadata: bool = False,
    cells: Sequence[str] | None = None,
    cell_schedule: str = SOURCE_CELL_SCHEDULE_FIXED,
    schedule_seed: int = 42,
) -> (
    tuple[str, ContrastiveTripletBatch]
    | tuple[str, ContrastiveTripletBatch, dict[str, Any]]
):
    if sampling_mode not in TRAINING_SAMPLING_MODES:
        raise ValueError(
            f"Unknown training sampling mode {sampling_mode!r}; "
            f"expected one of {TRAINING_SAMPLING_MODES}"
        )
    active_cells = normalize_source_cells(cells)
    cell = source_cell_for_scene_epoch(
        scene.scene_id,
        epoch,
        cells=active_cells,
        schedule=cell_schedule,
        seed=schedule_seed,
    )
    family, granularity = cell.split("/", 1)
    occurrence = int(epoch) // len(active_cells)
    if family == "3d":
        label_key = "g08" if repeat_g08_for_all_3d_slots else granularity
        labels = torch.from_numpy(scene.labels_by_key[label_key]).to(device)
        selected_labels = None
        if sampling_mode in (
            "coverage_without_replacement",
            "hybrid_2d_random_3d_coverage",
        ):
            valid_labels = _valid_partition_labels(labels)
            selected_indices = coverage_queue_indices(
                population_size=int(valid_labels.numel()),
                start=occurrence * int(sampling.proposals_per_forward),
                count=int(sampling.proposals_per_forward),
                seed_parts=(
                    scene.scene_id,
                    cell,
                    label_key,
                    "coverage-3d-label",
                ),
            )
            selected_labels = valid_labels[
                torch.tensor(selected_indices, dtype=torch.long, device=device)
            ]
        batch = sample_3d_batch(
            labels=labels,
            points=points,
            device=device,
            sampling=sampling,
            seed_parts=(scene.scene_id, epoch, cell, label_key, "train"),
            selected_proposal_labels=selected_labels,
        )
        metadata = {
            "cell": cell,
            "family": "3d",
            "granularity": granularity,
            "frame_ids": [],
            "proposal_ids": [
                str(int(value))
                for value in (
                    selected_labels
                    if selected_labels is not None
                    else torch.unique(batch.proposal_labels, sorted=True)
                ).tolist()
            ],
        }
        return (cell, batch, metadata) if return_metadata else (cell, batch)
    if sampling_mode == "coverage_without_replacement":
        batch, metadata = _coverage_raw_training_batch(
            scene=scene,
            granularity=granularity,
            cell=cell,
            epoch=epoch,
            occurrence=occurrence,
            points=points,
            device=device,
            sampling=sampling,
        )
        return (cell, batch, metadata) if return_metadata else (cell, batch)
    frames = scene.frames(granularity, "train")
    if not frames:
        raise ValueError(
            f"{scene.scene_id}/{granularity}: no proposal-bearing train frame"
        )
    offset = stable_seed(scene.scene_id, granularity, "train-frame") % len(frames)
    frame = frames[(offset + occurrence) % len(frames)]
    batch = sample_raw_batch(
        scene=scene,
        frame=frame,
        points=points,
        device=device,
        sampling=sampling,
        seed_parts=(
            scene.scene_id,
            epoch,
            cell,
            frame.physical_frame_id,
            "train",
        ),
    )
    metadata = {
        "cell": cell,
        "family": "2d",
        "granularity": granularity,
        "frame_ids": [frame.physical_frame_id],
        "proposal_ids": [
            f"{frame.physical_frame_id}:{int(value)}"
            for value in torch.unique(batch.proposal_labels, sorted=True).tolist()
        ],
    }
    return (cell, batch, metadata) if return_metadata else (cell, batch)


def _holdout_batches(
    *,
    scene: SceneSource,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
    cells: Sequence[str] | None = None,
) -> list[tuple[str, str, ContrastiveTripletBatch]]:
    active_cells = set(normalize_source_cells(cells))
    batches: list[tuple[str, str, ContrastiveTripletBatch]] = []
    for key in GRANULARITY_KEYS:
        if f"3d/{key}" in active_cells:
            labels = torch.from_numpy(scene.labels_by_key[key]).to(device)
            batches.append(
                (
                    f"3d/{key}",
                    "3d",
                    sample_3d_batch(
                        labels=labels,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed_parts=(seed, scene.scene_id, "eval", "3d", key),
                    ),
                )
            )
        if f"2d/{key}" not in active_cells:
            continue
        for frame in scene.frames(key, "heldout"):
            batches.append(
                (
                    f"2d/{key}",
                    f"2d/{frame.physical_frame_id}",
                    sample_raw_batch(
                        scene=scene,
                        frame=frame,
                        points=points,
                        device=device,
                        sampling=sampling,
                        seed_parts=(
                            seed,
                            scene.scene_id,
                            "eval",
                            "2d",
                            key,
                            frame.physical_frame_id,
                        ),
                    ),
                )
            )
    return batches
