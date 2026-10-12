"""Shared YAML loading and model/dataset config helpers for student scripts."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    from delimit3d.runtime.config import load_config as read_config
    return read_config(config_path)


def parse_granularities(data_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Extract dot-free granularity keys from ``data`` config.

    Supports:
        granularities: [0.2, 0.5, 0.8]    ->  ("g02", "g05", "g08")
        granularity: 0.8                   ->  ("g08",)
    """

    def _to_key(g: float | str) -> str:
        s = str(g)
        if s.startswith("g"):
            return s.replace(".", "")
        return f"g{s}".replace(".", "")

    if "granularities" in data_cfg:
        return tuple(_to_key(g) for g in data_cfg["granularities"])
    if "granularity" in data_cfg:
        return (_to_key(data_cfg["granularity"]),)
    raise KeyError("Config must specify data.granularities or data.granularity")


def resolve_num_queries(
    model_cfg: dict[str, Any],
    bb_cfg: dict[str, Any],
) -> tuple[int, dict[str, int] | None]:
    """Match ``run_student`` query resolution.

    Prefer ``model.num_queries`` / ``model.num_queries_by_granularity``; if absent,
    fall back to the same keys under ``model.backbone`` (common in YAML layouts).
    """
    nq = model_cfg.get("num_queries")
    if nq is None:
        nq = bb_cfg.get("num_queries", 128)
    nqb = model_cfg.get("num_queries_by_granularity")
    if nqb is None:
        nqb = bb_cfg.get("num_queries_by_granularity")
    return int(nq), nqb
