"""Exact scene-epoch checkpoints without dropping or padding scene visits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


POLICY = "flush_at_evaluation_epoch_boundary/v1"


@dataclass(frozen=True)
class LogicalBatchSchedule:
    scenes_per_epoch: int
    nominal_batch_size: int
    epochs: int
    flush_epochs: tuple[int, ...]

    def __post_init__(self):
        for value in (self.scenes_per_epoch, self.nominal_batch_size, self.epochs):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Logical-batch schedule dimensions must be positive integers")
        if (not self.flush_epochs or tuple(sorted(set(self.flush_epochs))) != self.flush_epochs
                or any(isinstance(e, bool) or not isinstance(e, int) or not 0 < e <= self.epochs
                       for e in self.flush_epochs)
                or self.flush_epochs[-1] != self.epochs):
            raise ValueError("Flush epochs must be sorted, unique and include the final epoch")

    def update_at_epoch(self, epoch: int, *, allow_partial: bool = False) -> int:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch <= self.epochs:
            raise ValueError("Epoch lies outside the logical-batch schedule")
        previous, updates = 0, 0
        for boundary in self.flush_epochs:
            if boundary > epoch:
                break
            visits = (boundary - previous) * self.scenes_per_epoch
            updates += (visits + self.nominal_batch_size - 1) // self.nominal_batch_size
            previous = boundary
        remaining = (epoch - previous) * self.scenes_per_epoch
        if remaining % self.nominal_batch_size and not allow_partial:
            raise ValueError("Epoch is not a flushed or complete logical-batch boundary")
        return updates + (remaining + self.nominal_batch_size - 1) // self.nominal_batch_size

    @property
    def total_updates(self) -> int:
        return self.update_at_epoch(self.epochs)

    @property
    def short_updates(self) -> dict[int, int]:
        previous, result = 0, {}
        for boundary in self.flush_epochs:
            remainder = ((boundary - previous) * self.scenes_per_epoch) % self.nominal_batch_size
            if remainder:
                result[self.update_at_epoch(boundary)] = remainder
            previous = boundary
        return result

    def batch_size(self, update: int) -> int:
        if isinstance(update, bool) or not isinstance(update, int) or not 0 < update <= self.total_updates:
            raise ValueError("Update lies outside the logical-batch schedule")
        return self.short_updates.get(update, self.nominal_batch_size)

    def consumed_visits(self, update: int) -> int:
        if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= self.total_updates:
            raise ValueError("Update lies outside the logical-batch schedule")
        return update * self.nominal_batch_size - sum(
            self.nominal_batch_size - size for u, size in self.short_updates.items() if u <= update
        )

    def as_dict(self) -> dict:
        return {"policy": POLICY, "scenes_per_epoch": self.scenes_per_epoch,
                "nominal_batch_size": self.nominal_batch_size, "epochs": self.epochs,
                "flush_epochs": list(self.flush_epochs)}

    @classmethod
    def from_dict(cls, value: Mapping) -> "LogicalBatchSchedule":
        if (not isinstance(value, Mapping)
                or set(value) != {"policy", "scenes_per_epoch", "nominal_batch_size", "epochs", "flush_epochs"}
                or value["policy"] != POLICY or not isinstance(value["flush_epochs"], (list, tuple))):
            raise ValueError("Logical-batch schedule contract drift")
        return cls(value["scenes_per_epoch"], value["nominal_batch_size"], value["epochs"],
                   tuple(value["flush_epochs"]))
