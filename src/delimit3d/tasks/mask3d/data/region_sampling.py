"""Compatibility re-export for the top-level region-sampling implementation."""

from delimit3d.tasks.mask3d.region_sampling import (
    grid_sample_indices,
    sphere_crop_indices,
    sphere_crop_indices_multi_center,
)

__all__ = [
    "grid_sample_indices",
    "sphere_crop_indices",
    "sphere_crop_indices_multi_center",
]
