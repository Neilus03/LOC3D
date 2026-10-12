"""CPU contracts for the matched frozen RGBN6 representation comparison.

These guards deliberately do not import LitePT or any CUDA extension.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

QUANTIZATION_BACKEND = "numpy_utility_compat_v1"


def validate_normal_alignment(
    ply_xyz: np.ndarray, sidecar_xyz: np.ndarray, normals: np.ndarray
) -> dict[str, Any]:
    """Require a coordinate witness in the same frame and point order.

    Equal vertex counts alone cannot establish row alignment. Translation or
    rotation is also rejected: sidecar normals must already use the PLY frame.
    Zero normals are retained as in LitePT's mesh preprocessing.
    """
    xyz = np.asarray(ply_xyz)
    witness = np.asarray(sidecar_xyz)
    normals = np.asarray(normals)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("PLY coordinates must be a nonempty N x 3 array")
    if witness.shape != xyz.shape or normals.shape != xyz.shape:
        raise ValueError("normal and coordinate sidecars must match the PLY N x 3 shape")
    if not all(np.isfinite(value).all() for value in (xyz, witness, normals)):
        raise ValueError("non-finite coordinates or normals")
    maximum_error = float(np.abs(xyz.astype(np.float64) - witness).max())
    if maximum_error > 1e-5:
        raise ValueError(
            "normal coordinate witness is not row-aligned in the PLY frame: "
            f"max coordinate error {maximum_error:.8g} m"
        )
    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths[lengths > 1e-6]
    if nonzero.size and (np.any(nonzero < 0.9) or np.any(nonzero > 1.002)):
        raise ValueError("normal magnitudes are outside the LitePT ScanNet range")
    return {
        "points": len(xyz),
        "max_coordinate_error_m": maximum_error,
        "zero_normals": int((lengths <= 1e-6).sum()),
        "alignment": "per_row_xyz_same_frame_atol_1e-5_m",
    }


def matched_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Only encoder initialization and arm/output labels may differ."""
    value = copy.deepcopy({key: item for key, item in config.items() if not key.startswith("_")})
    for key in ("experiment_id", "sources", "encoder_initialization"):
        value.pop(key, None)
    value["paths"].pop("output_root", None)
    return value


def validate_matched_pair(
    pretrained: Mapping[str, Any], random_control: Mapping[str, Any]
) -> dict[str, Any]:
    if pretrained.get("encoder_initialization", {}).get("kind") != "pretrained_checkpoint":
        raise ValueError("first arm must use pretrained_checkpoint initialization")
    if random_control.get("encoder_initialization", {}).get("kind") not in {"random", "scratch"}:
        raise ValueError("control must use random initialization")
    for config in (pretrained, random_control):
        if config.get("encoder_initialization", {}).get("seed") != config.get("seed"):
            raise ValueError("encoder initialization seed must equal the matched training seed")
    left, right = matched_settings(pretrained), matched_settings(random_control)
    if left != right:
        differences = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
        raise ValueError(f"matched RGBN6 arm settings differ: {differences}")
    if pretrained["paths"]["output_root"] == random_control["paths"]["output_root"]:
        raise ValueError("matched arms must have distinct output roots")
    digest = hashlib.sha256(json.dumps(left, sort_keys=True).encode()).hexdigest()
    return {"matched_settings_sha256": digest, "only_factor": "frozen_encoder_initialization"}


def validate_encoder_argument(config: Mapping[str, Any], checkpoint: Any, *, required: bool) -> None:
    kind = config.get("encoder_initialization", {}).get("kind")
    if kind not in {"pretrained_checkpoint", "random", "scratch"}:
        raise ValueError("encoder_initialization.kind must be pretrained_checkpoint or scratch")
    if kind in {"random", "scratch"} and checkpoint is not None:
        raise ValueError("random control rejects an encoder checkpoint")
    if kind == "pretrained_checkpoint" and required and checkpoint is None:
        raise ValueError("pretrained arm requires --encoder-checkpoint")


def validate_decoder_initialization(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    expected = config.get("decoder_initialization", {})
    for key in ("seed", "tensor_state_sha256"):
        if not expected.get(key) or report.get(key) != expected[key]:
            raise ValueError(f"shared fresh decoder initialization differs in {key}")


def validate_audit_lock(lock: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """A lock is generated on CPU before either representation arm runs."""
    if lock.get("mode") != "audit" or lock.get("gpu_used") is not False:
        raise ValueError("expected a CPU audit lock")
    for key in ("matched_pair", "list_sha256", "quantization_backend", "source_hashes"):
        if not current.get(key) or lock.get(key) != current[key]:
            raise ValueError(f"CPU audit lock mismatch: {key}")
    if lock.get("normal_contract", {}).get("dataset_content_sha256") != current["normal_contract"]["dataset_content_sha256"]:
        raise ValueError("CPU audit lock mismatch: dataset content")
    for key in ("file_sha256", "tensor_state_sha256"):
        if lock.get("decoder_initialization", {}).get(key) != current["decoder_initialization"][key]:
            raise ValueError(f"CPU audit lock mismatch: decoder {key}")
