"""Pure helpers for the LitePT-Mask3D training contract.

This module deliberately contains no dataset or model imports.  The runner uses
these helpers for the parts of the training protocol that are easy to get
silently wrong: optimizer-step counting, OneCycleLR construction, and the
training-split acceptance gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def effective_optimizer_steps_per_epoch(
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return optimizer steps per epoch, including a possible final partial group."""
    batches = int(num_batches)
    accumulation = int(gradient_accumulation_steps)
    if batches <= 0:
        raise ValueError(f"num_batches must be positive, got {batches}")
    if accumulation <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, "
            f"got {accumulation}"
        )
    return (batches + accumulation - 1) // accumulation


def build_onecycle_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    num_batches: int,
    gradient_accumulation_steps: int,
    config: Mapping[str, Any] | None = None,
) -> tuple[torch.optim.lr_scheduler.OneCycleLR, int]:
    """Build the reference-style OneCycleLR over optimizer, not microbatch, steps.

    ``OneCycleLR`` is stepped exactly once after each ``optimizer.step()``.
    ``max_lr`` comes from, in order: an explicit config value, a group's
    ``max_lr`` (present after a OneCycle checkpoint load), or the group's
    current ``lr``. Never treat a live scheduled ``lr`` as the cycle peak
    when ``max_lr`` is already on the param group.
    """
    epochs = int(max_epochs)
    if epochs <= 0:
        raise ValueError(f"max_epochs must be positive, got {epochs}")
    steps_per_epoch = effective_optimizer_steps_per_epoch(
        num_batches, gradient_accumulation_steps
    )
    scheduler_config = dict(config or {})
    name = str(scheduler_config.pop("name", "onecyclelr")).lower()
    if name not in {"onecycle", "onecyclelr", "one_cycle"}:
        raise ValueError(
            "LitePT-Mask3D requires OneCycleLR; unsupported scheduler "
            f"{name!r}"
        )
    interval = str(scheduler_config.pop("interval", "optimizer_step")).lower()
    if interval not in {"step", "optimizer_step"}:
        raise ValueError(
            "OneCycleLR must be stepped per optimizer step, "
            f"got interval={interval!r}"
        )

    group_lrs = [
        float(group["max_lr"]) if "max_lr" in group else float(group["lr"])
        for group in optimizer.param_groups
    ]
    configured_max_lr = scheduler_config.pop("max_lr", None)
    if configured_max_lr is None:
        max_lr: float | list[float] = group_lrs
    elif isinstance(configured_max_lr, Sequence) and not isinstance(
        configured_max_lr, (str, bytes)
    ):
        max_lr = [float(value) for value in configured_max_lr]
        if len(max_lr) != len(group_lrs):
            raise ValueError(
                "scheduler.max_lr list must match optimizer parameter groups: "
                f"{len(max_lr)} != {len(group_lrs)}"
            )
    else:
        max_lr = float(configured_max_lr)

    allowed_kwargs = {
        "pct_start",
        "anneal_strategy",
        "cycle_momentum",
        "base_momentum",
        "max_momentum",
        "div_factor",
        "final_div_factor",
        "three_phase",
    }
    unexpected = sorted(set(scheduler_config) - allowed_kwargs)
    if unexpected:
        raise ValueError(
            "Unsupported OneCycleLR config keys: " + ", ".join(unexpected)
        )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        **scheduler_config,
    )
    return scheduler, steps_per_epoch


def build_train_acceptance_report(
    *,
    epoch: int,
    train_metrics: Mapping[str, Any],
    train_set_loss: float,
    first_train_set_loss: float | None,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the capacity gate against metrics from the training split.

    ``min_per_scene_CA_AP50`` is retained as a backwards-compatible alias, but
    the report names the source explicitly so a held-out validation score cannot
    accidentally satisfy a memorization gate.
    """
    raw_metrics = train_metrics.get("raw_queries")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("train_metrics.raw_queries is required for acceptance")

    class_agnostic = bool(raw_metrics.get("class_agnostic", True))
    metric_key = "CA-AP50" if class_agnostic else "AP50"
    min_scene_key = "min_scene_CA-AP50" if class_agnostic else "min_scene_AP50"
    aggregate_ap50 = float(raw_metrics[metric_key])
    min_scene_ap50 = float(raw_metrics[min_scene_key])
    per_scene = {
        str(scene_id): float(scene_metrics[metric_key])
        for scene_id, scene_metrics in dict(raw_metrics["per_scene"]).items()
    }

    if class_agnostic:
        aggregate_threshold = acceptance.get("min_train_CA_AP50")
        scene_threshold = acceptance.get(
            "min_train_scene_CA_AP50",
            acceptance.get("min_per_scene_CA_AP50"),
        )
        require_high_train = bool(
            acceptance.get("require_high_train_CA_AP50", False)
        )
        aggregate_check_name = "aggregate_train_CA_AP50"
        scene_check_name = "minimum_train_scene_CA_AP50"
        aggregate_report_key = "aggregate_CA-AP50"
        scene_report_key = "min_scene_CA-AP50"
    else:
        aggregate_threshold = acceptance.get("min_train_AP50")
        scene_threshold = acceptance.get(
            "min_train_scene_AP50",
            acceptance.get("min_per_scene_AP50"),
        )
        require_high_train = bool(
            acceptance.get("require_high_train_AP50", False)
        )
        aggregate_check_name = "aggregate_train_AP50"
        scene_check_name = "minimum_train_scene_AP50"
        aggregate_report_key = "aggregate_AP50"
        scene_report_key = "min_scene_AP50"
    if require_high_train and aggregate_threshold is None and scene_threshold is None:
        raise ValueError(
            "the configured high-train AP50 gate needs an aggregate or "
            "per-scene AP50 threshold"
        )

    checks: dict[str, bool] = {
        "train_metrics_present": True,
    }
    if aggregate_threshold is not None:
        checks[aggregate_check_name] = aggregate_ap50 >= float(
            aggregate_threshold
        )
    if scene_threshold is not None:
        checks[scene_check_name] = min_scene_ap50 >= float(
            scene_threshold
        )
    if bool(acceptance.get("require_decreasing_set_loss", False)):
        checks["set_loss_decreased"] = (
            first_train_set_loss is not None
            and float(train_set_loss) < float(first_train_set_loss)
        )

    report: dict[str, Any] = {
        "epoch": int(epoch),
        "source": "train_split",
        "passed": bool(all(checks.values())),
        "thresholds": {
            ("min_train_CA_AP50" if class_agnostic else "min_train_AP50"): (
                None if aggregate_threshold is None else float(aggregate_threshold)
            ),
            (
                "min_train_scene_CA_AP50"
                if class_agnostic
                else "min_train_scene_AP50"
            ): (None if scene_threshold is None else float(scene_threshold)),
        },
        "class_agnostic": class_agnostic,
        metric_key: aggregate_ap50,
        min_scene_key: min_scene_ap50,
        ("per_scene_CA-AP50" if class_agnostic else "per_scene_AP50"): per_scene,
        "train_set_loss": float(train_set_loss),
        "first_train_set_loss": (
            None
            if first_train_set_loss is None
            else float(first_train_set_loss)
        ),
        "checks": checks,
    }
    report[aggregate_report_key] = aggregate_ap50
    report[scene_report_key] = min_scene_ap50
    return report


__all__ = [
    "build_onecycle_scheduler",
    "build_train_acceptance_report",
    "effective_optimizer_steps_per_epoch",
]
