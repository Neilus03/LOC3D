"""Pretraining schedule extracted from the executed trainer."""

from __future__ import annotations


from pathlib import Path


from typing import Collection, Mapping, Sequence


from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule


def _update_for_completed_epoch(
    *,
    epoch: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
    logical_batch_schedule: LogicalBatchSchedule | None = None,
) -> int:
    if logical_batch_schedule is not None:
        if (logical_batch_schedule.scenes_per_epoch != scene_visits_per_epoch_per_rank
                or logical_batch_schedule.nominal_batch_size != scenes_per_rank_per_update):
            raise ValueError("Logical-batch schedule dimensions drift")
        return logical_batch_schedule.update_at_epoch(int(epoch))
    scene_visits = int(epoch) * int(scene_visits_per_epoch_per_rank)
    logical_batch = int(scenes_per_rank_per_update)
    if logical_batch <= 0:
        raise ValueError("scenes_per_rank_per_update must be positive")
    if scene_visits % logical_batch != 0:
        raise ValueError(
            "Epoch boundary does not align with a complete logical batch: "
            f"epoch={epoch} scene_visits={scene_visits} "
            f"scenes_per_rank_per_update={logical_batch}"
        )
    return scene_visits // logical_batch


def _final_evaluation_endpoint(
    *,
    nominal_epochs: int,
    total_updates: int,
    maximum_updates: int | None,
    snapshot_label_by_update: Mapping[int, int],
    checkpoint_epochs: Collection[int],
    checkpoint_updates: Collection[int],
    checkpoint_dir: Path,
) -> tuple[int, Path]:
    """Resolve the exact retained evaluation/checkpoint at a run endpoint.

    A ``--maximum-updates`` screen normally stops between true scene-epoch
    boundaries. Its endpoint is therefore an update-aligned audit snapshot,
    not the nominal ``--epochs`` checkpoint.
    """

    nominal_epochs = int(nominal_epochs)
    total_updates = int(total_updates)
    if nominal_epochs < 0 or total_updates <= 0:
        raise ValueError("Final evaluation endpoint is invalid")
    snapshot_labels = {
        int(update): int(label)
        for update, label in snapshot_label_by_update.items()
    }
    retained_epochs = {int(value) for value in checkpoint_epochs}
    retained_updates = {int(value) for value in checkpoint_updates}

    if maximum_updates is not None:
        if int(maximum_updates) != total_updates:
            raise ValueError("Maximum-update endpoint disagrees with total updates")
        if total_updates not in snapshot_labels:
            raise ValueError(
                "Maximum-update endpoint must be an exact audit snapshot"
            )
        if total_updates not in retained_updates:
            raise ValueError(
                "Maximum-update endpoint audit snapshot must retain a checkpoint"
            )
        label = snapshot_labels[total_updates]
        return (
            label,
            checkpoint_dir / f"audit_e{label:04d}_u{total_updates:08d}.pt",
        )

    if nominal_epochs not in retained_epochs:
        raise ValueError("Final scene epoch must retain a resumable checkpoint")
    return nominal_epochs, checkpoint_dir / f"epoch_{nominal_epochs:04d}.pt"


def _updates_for_scene_visit_duration(
    *,
    epochs: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
) -> int:
    """Convert a scene-epoch duration to complete updates without dropping visits."""

    if int(epochs) <= 0:
        raise ValueError("duration epochs must be positive")
    scene_visits = int(epochs) * int(scene_visits_per_epoch_per_rank)
    logical_batch = int(scenes_per_rank_per_update)
    if logical_batch <= 0:
        raise ValueError("scenes_per_rank_per_update must be positive")
    return (scene_visits + logical_batch - 1) // logical_batch


def _scheduled_gradient_panel_updates(
    *,
    total_updates: int,
    scene_visits_per_epoch_per_rank: int,
    scenes_per_rank_per_update: int,
    eval_epochs: Sequence[int],
    snapshot_updates: Sequence[int],
    auxiliary_enabled: bool,
) -> list[int]:
    """Return the exact update schedule used by ``collect_global_gradient_panel``."""

    if not auxiliary_enabled:
        return []
    if int(total_updates) <= 0:
        raise ValueError("total_updates must be positive")
    # The evaluator's checkpoint fallback always expects a panel at the
    # checkpoint boundary, even when the final update is not also an eval or
    # audit boundary.
    updates = {1, int(total_updates)}
    updates.update(int(value) for value in snapshot_updates)
    for epoch in eval_epochs:
        if int(epoch) <= 0:
            continue
        panel_update = _update_for_completed_epoch(
            epoch=int(epoch),
            scene_visits_per_epoch_per_rank=scene_visits_per_epoch_per_rank,
            scenes_per_rank_per_update=scenes_per_rank_per_update,
        )
        if panel_update <= int(total_updates):
            updates.add(panel_update)
    if any(value <= 0 or value > int(total_updates) for value in updates):
        raise ValueError("Gradient-panel update lies outside the training horizon")
    return sorted(updates)
