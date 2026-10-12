"""Optimizer groups and learning-rate updates."""

from __future__ import annotations


from typing import Any


import torch

import torch.nn as nn


from .model import (
    LitePTContrastiveModel,
)


def optimizer_parameter_groups(
    model: LitePTContrastiveModel,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    names: dict[tuple[str, bool], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "backbone" if name.startswith("backbone.") else "head"
        no_decay = bool(
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "log_temperature" in name
        )
        key = (family, no_decay)
        groups.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)
    output: list[dict[str, Any]] = []
    report: dict[str, list[str]] = {}
    for family in ("backbone", "head"):
        for no_decay in (False, True):
            key = (family, no_decay)
            if key not in groups:
                continue
            lr = float(backbone_lr if family == "backbone" else head_lr)
            group_name = f"{family}/{'no_decay' if no_decay else 'decay'}"
            output.append(
                {
                    "params": groups[key],
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": 0.0 if no_decay else float(weight_decay),
                    "group_name": group_name,
                }
            )
            report[group_name] = names[key]
    return output, report


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in optimizer.param_groups:
        lr = float(group["initial_lr"]) * float(factor)
        group["lr"] = lr
        values[str(group["group_name"])] = lr
    return values
