"""Sidecar normal loading for modality-controlled AGILE3D experiments.

The released AGILE3D PLYs intentionally contain RGB and labels only.  This
module keeps normals as a separate encoder-side asset so the official RGB
protocol remains reproducible while RGB+normal experiments can use the same
point ordering.
"""

from pathlib import Path

import numpy as np


def _normal_candidates(normal_root: Path, scene_name: str):
    """Yield supported sidecar layouts in deterministic order."""
    # The current LitePT ScanNet cache is <root>/<split>/<scene>/normal.npy.
    # The first layout also supports a future flat cross-dataset cache.
    yield normal_root / scene_name / "normal.npy"
    for split in ("train", "val", "test"):
        yield normal_root / split / scene_name / "normal.npy"


def load_scene_normals(normal_root, scene_name: str, expected_points: int) -> np.ndarray:
    """Load and validate a row-aligned ``N x 3`` normal sidecar.

    Parameters
    ----------
    normal_root:
        Root containing either ``<scene>/normal.npy`` or
        ``<split>/<scene>/normal.npy``.
    scene_name:
        AGILE3D scene stem, for example ``scene0188_00``.
    expected_points:
        Number of vertices in the AGILE3D PLY being loaded.
    """
    if not normal_root:
        raise ValueError(
            "RGB+normal input requested but --normal_root was not provided"
        )

    root = Path(normal_root)
    candidates = list(_normal_candidates(root, scene_name))
    normal_path = next((path for path in candidates if path.is_file()), None)
    if normal_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No normal.npy found for {scene_name}; searched: {searched}"
        )

    normals = np.asarray(np.load(normal_path), dtype=np.float32)
    if normals.shape != (expected_points, 3):
        raise ValueError(
            f"Normal shape mismatch for {scene_name}: {normal_path} has "
            f"{normals.shape}, expected {(expected_points, 3)}"
        )
    if not np.isfinite(normals).all():
        raise ValueError(f"Non-finite normal value in {normal_path}")

    norms = np.linalg.norm(normals, axis=1)
    nonzero_norms = norms[norms > 1.0e-6]

    # LitePT's original ScanNet preprocessing adds 1e-8 to the normalization
    # denominator.  Vertices not covered by a valid mesh face therefore stay
    # exactly zero, and a small number of nearly cancelling face normals have
    # a nonzero magnitude slightly below one.  Preserve those values exactly
    # instead of renormalizing them: this is the representation LitePT saw
    # during its own training.
    if nonzero_norms.size and (
        np.any(nonzero_norms < 0.9) or np.any(nonzero_norms > 1.002)
    ):
        raise ValueError(
            f"Normal magnitudes in {normal_path} are outside the LitePT "
            f"ScanNet range: min_nonzero={nonzero_norms.min():.6f}, "
            f"max_nonzero={nonzero_norms.max():.6f}"
        )

    return normals
