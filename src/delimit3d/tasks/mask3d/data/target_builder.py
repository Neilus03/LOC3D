"""Convert per-point pseudo-labels into per-instance binary masks.

Takes ``labels`` (N,) and ``supervision_mask`` (N,) and builds the target
tensors consumed by the instance decoder loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceTargets:
    """Ground-truth instance masks built from CHORUS pseudo-labels.

    Masks are defined over *all* N points.  Non-supervised points are
    always False in every row — the loss will mask them out.
    """

    instance_ids: np.ndarray          # (M,)   int — original label id per instance
    gt_masks: torch.Tensor            # (M, N) bool
    supervision_mask: torch.Tensor    # (N,)   bool — kept here for convenience
    num_instances: int
    instance_sizes: np.ndarray        # (M,)   int — points per instance
    class_ids: torch.Tensor | None = None  # (M,) long — optional contiguous class id


def build_instance_targets(
    labels: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    min_instance_points: int = 1,
    ignore_label: int = -1,
    dense_instance_ids: bool = False,
    instance_class_map: dict[int, int] | None = None,
) -> InstanceTargets:
    """Build per-instance binary masks from per-point pseudo-labels.

    Parameters
    ----------
    labels:
        (N,) long — per-point pseudo-instance id.
        *ignore_label* marks unlabeled / noise points.
    supervision_mask:
        (N,) bool — which points participate in supervision.
    min_instance_points:
        Drop instances with fewer supervised points than this.
    ignore_label:
        Sentinel value for unlabeled points.
    dense_instance_ids:
        If True, store contiguous instance ids ``0 .. M-1`` (one per kept
        instance) instead of raw pseudo-label values.  Metrics that add 1 to
        ``instance_ids`` then yield per-point labels ``1 .. M``.
    instance_class_map:
        Optional mapping from raw instance id in ``labels`` to contiguous class
        id. Used by real-GT finetuning; pseudo-label pretraining leaves it None.

    Returns
    -------
    :class:`InstanceTargets` with ``gt_masks`` of shape (M, N).
    """
    N = labels.shape[0]
    assert supervision_mask.shape == (N,), (
        f"supervision_mask shape {supervision_mask.shape} != labels shape ({N},)"
    )

    valid = (labels != ignore_label) & supervision_mask
    valid_labels = labels[valid]
    unique_ids = valid_labels.unique(sorted=True)

    keep_ids: list[int] = []
    rows: list[torch.Tensor] = []
    sizes: list[int] = []
    class_ids: list[int] = []

    for inst_id in unique_ids:
        mask_row = valid & (labels == inst_id)
        n = int(mask_row.sum())
        if n >= min_instance_points:
            keep_ids.append(int(inst_id))
            rows.append(mask_row)
            sizes.append(n)
            if instance_class_map is not None:
                raw_id = int(inst_id)
                if raw_id not in instance_class_map:
                    raise KeyError(
                        f"Missing class id for instance {raw_id}; "
                        "real-GT class-aware finetuning requires every kept "
                        "instance to have a class label."
                    )
                class_ids.append(int(instance_class_map[raw_id]))

    if rows:
        gt_masks = torch.stack(rows, dim=0)
        instance_ids = np.array(keep_ids, dtype=np.int64)
        instance_sizes = np.array(sizes, dtype=np.int64)
    else:
        gt_masks = torch.zeros(0, N, dtype=torch.bool)
        instance_ids = np.zeros(0, dtype=np.int64)
        instance_sizes = np.zeros(0, dtype=np.int64)

    class_ids_t: torch.Tensor | None = None
    if instance_class_map is not None:
        class_ids_t = torch.tensor(class_ids, dtype=torch.long, device=labels.device)

    return InstanceTargets(
        instance_ids=instance_ids,
        gt_masks=gt_masks,
        supervision_mask=supervision_mask,
        num_instances=len(keep_ids),
        instance_sizes=instance_sizes,
        class_ids=class_ids_t,
    )


def build_instance_targets_multi(
    labels_by_granularity: dict[str, torch.Tensor],
    supervision_mask: torch.Tensor,
    *,
    min_instance_points: int = 1,
    ignore_label: int = -1,
    dense_instance_ids: bool = False,
    instance_class_maps: dict[str, dict[int, int] | None] | None = None,
) -> dict[str, InstanceTargets]:
    """Build per-instance targets for each granularity.

    Calls :func:`build_instance_targets` once per granularity key.

    Returns
    -------
    Dict mapping granularity keys (e.g. ``"g02"``) to :class:`InstanceTargets`.
    """
    return {
        g: build_instance_targets(
            labels,
            supervision_mask,
            min_instance_points=min_instance_points,
            ignore_label=ignore_label,
            dense_instance_ids=dense_instance_ids,
            instance_class_map=(
                instance_class_maps.get(g)
                if instance_class_maps is not None
                else None
            ),
        )
        for g, labels in labels_by_granularity.items()
    }


def log_target_stats(
    targets: InstanceTargets,
    tag: str = "",
) -> dict[str, Any]:
    """Log and return summary statistics — catches broken pseudo-labels early."""
    prefix = f"[{tag}] " if tag else ""
    N = int(targets.supervision_mask.shape[0])

    stats: dict[str, Any] = {
        "num_instances": targets.num_instances,
        "num_points": N,
    }

    if targets.num_instances == 0:
        log.warning("%sNo instances after filtering — labels may be broken", prefix)
        return stats

    sz = targets.instance_sizes.astype(float)
    stats.update(
        {
            "total_labeled_points": int(sz.sum()),
            "labeled_fraction": float(sz.sum() / N),
            "min_instance_size": int(sz.min()),
            "max_instance_size": int(sz.max()),
            "mean_instance_size": float(sz.mean()),
            "median_instance_size": float(np.median(sz)),
            "std_instance_size": float(sz.std()),
        }
    )

    log.info(
        "%s%d instances | sizes: min=%d  mean=%.1f  median=%.0f  max=%d  "
        "std=%.1f | labeled %.1f%% of %d pts",
        prefix,
        stats["num_instances"],
        stats["min_instance_size"],
        stats["mean_instance_size"],
        stats["median_instance_size"],
        stats["max_instance_size"],
        stats["std_instance_size"],
        stats["labeled_fraction"] * 100,
        stats["num_points"],
    )

    return stats


# ── smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from delimit3d.tasks.mask3d.data.single_scene_dataset import SingleSceneTrainingPackDataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m delimit3d.tasks.mask3d.data.target_builder <scene_dir> [granularity]")
        raise SystemExit(1)

    scene_dir = sys.argv[1]
    granularity = float(sys.argv[2]) if len(sys.argv) == 3 else 0.5

    ds = SingleSceneTrainingPackDataset(scene_dir, granularity=granularity)
    sample = ds[0]

    targets = build_instance_targets(sample["labels"], sample["supervision_mask"])
    print(f"\ngt_masks            : {targets.gt_masks.shape}  {targets.gt_masks.dtype}")
    print(f"instance_ids        : {targets.instance_ids.shape}  {targets.instance_ids.dtype}")
    print(f"instance_sizes      : {targets.instance_sizes.shape}  {targets.instance_sizes.dtype}")

    stats = log_target_stats(targets, tag=f"{ds.scene_id}/g{granularity}")
    print("\nfull stats:")
    for k, v in stats.items():
        print(f"  {k:24s}: {v}")
