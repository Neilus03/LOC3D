"""Persistent V2 coverage identities for whole scenes and rotating SAM2 fragments.

The sampler epoch counts visits to the *same catalog*. Geometry augmentation
still uses the global training epoch; fragment rotation does not reset coverage.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from delimit3d.data.contrastive_sampler_v2 import DeterministicCoverageState
from delimit3d.data.sam2_source import FRAGMENT_POLICY, SOURCE_CELL
from delimit3d.training.contracts import stable_seed

COVERAGE_POLICY = "per_fragment_persisted_cursor/v2"
SAMPLING_OPERATION = "rgbn6-pretraining-v2"
_FRAGMENT_ID = re.compile(r"sam2_fragment_(\d+)_of_(\d+)_([0-9a-f]{16})")


def _fragment_parts(fragment_id: str) -> tuple[int, int]:
    match = _FRAGMENT_ID.fullmatch(fragment_id)
    if match is None:
        raise ValueError("Invalid SAM2 coverage fragment identity")
    position, count = int(match[1]), int(match[2])
    if count < 2 or not 0 <= position < count:
        raise ValueError("Invalid SAM2 coverage fragment position/count")
    return position, count


@dataclass(frozen=True)
class SamplingVisit:
    epoch: int
    operation: str
    keys_by_cell: dict[str, str]
    states_by_cell: dict[str, DeterministicCoverageState | None]


def sampling_visit_for_scene(
    scene, *, global_epoch: int, seed: int, cells, saved_states: Mapping,
) -> SamplingVisit:
    """Rebind each persisted cursor to its next catalog visit, never reconstruct it."""
    if global_epoch < 0:
        raise ValueError("Training epoch must be nonnegative")
    sampler_epoch = int(global_epoch)
    operation = SAMPLING_OPERATION
    fragment_id = scene.crop_id
    suffix = ""
    if fragment_id is not None and fragment_id.startswith("sam2_fragment_"):
        if tuple(cells) != (SOURCE_CELL,) or scene.source_variant_id != FRAGMENT_POLICY:
            raise ValueError("SAM2 fragment coverage requires the frozen single-source geometry policy")
        position, count = _fragment_parts(fragment_id)
        expected = (stable_seed(seed, scene.scene_id, FRAGMENT_POLICY) + global_epoch) % count
        if position != expected:
            raise ValueError("SAM2 fragment rotation differs from the training visit")
        sampler_epoch = int(global_epoch) // count
        operation += f"/{fragment_id}"
        suffix = f"|{fragment_id}"
    keys = {cell: f"{scene.scene_id}|{cell}{suffix}" for cell in cells}
    states = {}
    for cell, key in keys.items():
        record = saved_states.get(key)
        if record is None:
            states[cell] = None
        elif isinstance(record, Mapping):
            states[cell] = DeterministicCoverageState.from_dict(record).for_epoch(sampler_epoch)
        else:
            raise ValueError("V2 coverage state must be a mapping or null")
    return SamplingVisit(sampler_epoch, operation, keys, states)


def validate_saved_coverage_states(records: Mapping, *, start_epoch: int, seed: int) -> None:
    """Validate each cursor at its last visit, including dormant fragment catalogs."""
    completed_epoch = int(start_epoch) - 1
    for key, record in records.items():
        parts = str(key).split("|")
        if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
            raise ValueError("Invalid V2 saved coverage key")
        scene_id, cell = parts[:2]
        expected_epoch = completed_epoch
        expected_operation = SAMPLING_OPERATION
        if len(parts) == 3:
            if cell != SOURCE_CELL:
                raise ValueError("Fragment coverage keys are restricted to SAM2")
            fragment_id = parts[2]
            position, count = _fragment_parts(fragment_id)
            offset = stable_seed(seed, scene_id, FRAGMENT_POLICY) % count
            first_global_epoch = (position - offset) % count
            if first_global_epoch > completed_epoch:
                raise ValueError("Saved SAM2 cursor refers to an unvisited fragment")
            last_global_epoch = completed_epoch - ((completed_epoch - first_global_epoch) % count)
            expected_epoch = last_global_epoch // count
            expected_operation += f"/{fragment_id}"
        if record is None:
            continue  # Empty raw catalogs have no sampler state.
        state = DeterministicCoverageState.from_dict(record)
        if (state.scene_id != scene_id or state.seed != seed
                or state.operation != f"{expected_operation}/{cell}"
                or state.epoch != expected_epoch or state.cursor <= 0):
            raise ValueError("V2 saved coverage audit cursor is inconsistent with the resumable epoch boundary")
