"""Inspect concatenated saved feature stages without guessing their normalization.

The stage widths must come from the export's architecture/provenance. Slicing a
jointly normalized descriptor retains its stage magnitudes; normalizing each
slice discards those magnitudes. They are different diagnostic inputs.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def split_feature_stages(features: np.ndarray, widths: Mapping[str, int], *,
                         normalization: str = "as_saved") -> dict[str, np.ndarray]:
    """Split every channel once, preserving row order and the supplied stage order.

    ``as_saved`` preserves the stored numbers exactly. ``per_stage_l2`` requires
    finite nonzero vectors in every stage. Neither option changes cell/point
    identities or implicitly normalizes the concatenated descriptor.
    """
    value = np.asarray(features)
    if (value.ndim != 2 or min(value.shape) < 1 or
            not np.issubdtype(value.dtype, np.number) or
            np.issubdtype(value.dtype, np.complexfloating) or not np.isfinite(value).all()):
        raise ValueError("features must be a nonempty finite real matrix")
    if not isinstance(widths, Mapping) or not widths:
        raise ValueError("widths must map stage names to positive channel counts")
    for name, width in widths.items():
        if not isinstance(name, str) or not name:
            raise ValueError("stage names must be nonempty strings")
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("stage widths must be positive integers")
    if sum(widths.values()) != value.shape[1]:
        raise ValueError("stage widths must cover every feature channel exactly once")
    if normalization not in {"as_saved", "per_stage_l2"}:
        raise ValueError("normalization must be as_saved or per_stage_l2")
    stages = {}
    start = 0
    for name, width in widths.items():
        block = value[:, start:start + width].copy()
        if normalization == "per_stage_l2":
            block = block.astype(np.float64)
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                raise ValueError(f"stage {name!r} has zero or nonfinite feature norms")
            block = (block / norms[:, None]).astype(np.float32)
        stages[name] = block
        start += width
    return stages
