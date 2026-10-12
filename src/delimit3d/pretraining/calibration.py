#!/usr/bin/env python3
"""Deterministic scientific calibration for RGBN6 multi-scale pretraining.

The executable path is deliberately fail-closed.  It reproduces the frozen
48-scene stratified panel from the immutable Structured3D source audit, visits
each scene/cell pair in two complete full-weight schedule cycles, and measures
the per-visit frozen-initialization primary-direction factor

    d_0.4 = <g_dec0, g_dec0 + 0.4 h_aux> / ||g_dec0||^2

at named LitePT encoder/decoder sites.  It then runs matched control (C,
auxiliary multiplier 0.0) and multi-scale (M, multiplier 0.4) branches from
independently reconstructed but identical model, optimizer, and RNG state.
Each branch executes exactly eight production-shaped DDP optimizer updates over
the same frozen 96-visit schedule.  The paired branch is a mechanical gate for
finite optimization, clipping, state, and input pairing; its short losses are
explicitly not treated as representation-quality evidence.

Only standard-library modules are imported at module load time so panel,
schedule, direction, and summary functions remain CPU-testable without torch.
The CUDA/model stack is imported lazily by :func:`run_distributed`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = (
    "delimit3d_multiscale_calibration/v1"
)
EXPERIMENT_ID = "litept_gt3_unsam3_scratch_v1"
PANEL_NAMESPACE = "multiscale-calibration-v1"
FROZEN_SEED = 42
SOURCE_CELLS = ("2d/g02", "2d/g05", "2d/g08")
LEVEL_NAMES = ("dec3", "dec2", "dec1")
LEVEL_WEIGHTS = {"dec3": 0.20, "dec2": 0.30, "dec1": 0.50}
AUXILIARY_WEIGHT = 0.40
AUXILIARY_WARMUP_EPOCHS = 20
FULL_WEIGHT_CYCLE_STARTS = (21, 24)
SAMPLES_PER_STRATUM = 2
EXPECTED_OPTIMIZATION_SCENES = 2930
EXPECTED_PANEL_SIZE = 48
EXPECTED_SCHEDULE_SIZE = 96
EXPECTED_GLOBAL_UPDATES = 8
EXPECTED_WORLD_SIZE = 2
SCENES_PER_RANK_PER_UPDATE = 6
PAIRED_BRANCH_MULTIPLIERS = {"C": 0.0, "M": AUXILIARY_WEIGHT}
BACKBONE_PEAK_LR = 0.001
HEAD_PEAK_LR = 0.003
WEIGHT_DECAY = 0.05
GRAD_CLIP_NORM = 1.0
CODE_MANIFEST_SCHEMA = "chorus_fullscene_rgbn6_multiscale_pretrain_code/v1"
PRETRAINING_CONTRACT_SCHEMA = (
    "structured3d_rgbn6_fullscene_multiscale_pretrain_contract/v1"
)
FROZEN_SOURCE_AUDIT_SHA256 = (
    "a556ec1df95a57732e85a93971653cffd0a638732b55176e7affd70d9b0a4421"
)
FROZEN_PANEL_SHA256 = (
    "2fe1bc00718598df4c501a1fe2ddceb11fed70307323945121356d07877e9934"
)
FROZEN_SCHEDULE_SHA256 = (
    "3dbb8795c316a5d224a5ac412c0f4f2a3e8476e48bb59ffd3ec288e4c2479872"
)

# These boundaries are derived by rank-partitioning the immutable 2,930-scene
# optimization split, not by fitting thresholds to pilot outcomes.
FROZEN_POINT_QUARTILES = (
    (733, 28689, 403638),
    (732, 404127, 608710),
    (733, 608972, 784894),
    (732, 785020, 3699315),
)

# Exact result of the frozen selection algorithm:
# (one-based point quartile, target cell, proposal-density half, two scene IDs).
FROZEN_PANEL_SCENES: tuple[tuple[int, str, str, tuple[str, str]], ...] = (
    (1, "2d/g02", "low", ("scene_01236", "scene_01362")),
    (1, "2d/g02", "high", ("scene_02580", "scene_02588")),
    (1, "2d/g05", "low", ("scene_01894", "scene_02682")),
    (1, "2d/g05", "high", ("scene_02803", "scene_01480")),
    (1, "2d/g08", "low", ("scene_02530", "scene_00735")),
    (1, "2d/g08", "high", ("scene_00119", "scene_03467")),
    (2, "2d/g02", "low", ("scene_00740", "scene_02542")),
    (2, "2d/g02", "high", ("scene_03352", "scene_02587")),
    (2, "2d/g05", "low", ("scene_02830", "scene_01907")),
    (2, "2d/g05", "high", ("scene_00189", "scene_03307")),
    (2, "2d/g08", "low", ("scene_03080", "scene_02304")),
    (2, "2d/g08", "high", ("scene_02827", "scene_02684")),
    (3, "2d/g02", "low", ("scene_02004", "scene_02161")),
    (3, "2d/g02", "high", ("scene_00213", "scene_01510")),
    (3, "2d/g05", "low", ("scene_01072", "scene_00563")),
    (3, "2d/g05", "high", ("scene_02872", "scene_02576")),
    (3, "2d/g08", "low", ("scene_00619", "scene_00979")),
    (3, "2d/g08", "high", ("scene_02601", "scene_00989")),
    (4, "2d/g02", "low", ("scene_00047", "scene_01801")),
    (4, "2d/g02", "high", ("scene_02979", "scene_03087")),
    (4, "2d/g05", "low", ("scene_03381", "scene_00596")),
    (4, "2d/g05", "high", ("scene_03422", "scene_00859")),
    (4, "2d/g08", "low", ("scene_02797", "scene_00071")),
    (4, "2d/g08", "high", ("scene_01386", "scene_02917")),
)

SITE_COMPONENT_PATHS: dict[str, tuple[str, ...]] = {
    "enc0": LEVEL_NAMES,
    "enc4": LEVEL_NAMES,
    "dec3": LEVEL_NAMES,
    "dec2": ("dec2", "dec1"),
    "dec1": ("dec1",),
    "dec0": (),
}
PRIMARY_DIRECTION_SITES = (
    "shared_union",
    "enc0",
    "enc4",
    "dec3",
    "dec2",
    "dec1",
)
LEVEL_NATIVE_SITES = {"dec3": "dec3", "dec2": "dec2", "dec1": "dec1"}

_BASE_RECORD_HARD_INVARIANTS = (
    "manifest_hash_exact",
    "full_scene_point_count_exact",
    "canonical_target_cell_exact",
    "observed_target_cell_exact",
    "auxiliary_scale_exact",
    "total_loss_finite",
)
_LEVEL_RECORD_HARD_INVARIANTS = (
    "proposal_diagnostic_cardinality_exact",
    "input_proposal_cardinality_exact",
    "retained_proposal_count_exact",
    "input_pair_budget_exact",
    "proposal_retention_logic_valid",
    "strict_token_purity_exact",
    "accepted_mixed_tokens_zero",
    "negative_diversity_valid_when_batch_survives",
    "stage_weight_exact",
)
_GRADIENT_SITE_HARD_INVARIANTS = (
    "final_gradient_present_finite_nonzero",
    "component_topology_exact",
    "component_gradients_finite",
)
REQUIRED_RECORD_HARD_INVARIANTS = frozenset(
    (*_BASE_RECORD_HARD_INVARIANTS, "dec0_negative_control_exact")
    + tuple(
        f"{level}_{check}"
        for level in LEVEL_NAMES
        for check in _LEVEL_RECORD_HARD_INVARIANTS
    )
    + tuple(
        f"{site}_{check}"
        for site in (*SITE_COMPONENT_PATHS, "shared_union")
        for check in _GRADIENT_SITE_HARD_INVARIANTS
    )
    + tuple(f"{site}_d_0_4_positive" for site in PRIMARY_DIRECTION_SITES)
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    try:
        return sha256_file(path.resolve(strict=True))
    except (FileNotFoundError, OSError):
        return None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_trust_artifacts(*, code_manifest_path, pretraining_contract_path,
                             initial_checkpoint_path, source_audit_path,
                             running_script_path=None):
    from delimit3d.runtime.config import expand_environment, load_config
    paths = {"code_manifest": code_manifest_path, "pretraining_contract": pretraining_contract_path,
             "initial_checkpoint": initial_checkpoint_path, "source_audit": source_audit_path}
    hashes = {name: sha256_file(Path(path).resolve(strict=True)) for name, path in paths.items()}
    code = json.loads(Path(code_manifest_path).read_text())
    if code.get("schema_version") != "delimit3d_code_manifest/v1" or not code.get("passed") or not code.get("files"):
        raise ValueError("Create a release code record with prepare_data --action code-record")
    root = Path(__file__).resolve().parents[3]
    for relative, expected in code["files"].items():
        target = root / relative
        if not target.resolve().is_relative_to(root) or sha256_file(target) != expected:
            raise ValueError(f"Calibration code changed: {relative}")
    config = load_config(pretraining_contract_path, expand_env=False)
    config.pop("recipe", None)
    config = expand_environment(config)
    expected = {"input_features": "rgbn6", "grid_size": 0.02, "seed": 42,
        "required_crop_mode": "full_scene", "multiscale_supervision": True,
        "multiscale_loss_weight": AUXILIARY_WEIGHT, "multiscale_warmup_epochs": 20,
        "normalization_policy": "batchnorm", "hierarchy_supervision": "legacy",
        "sampling_profile": "pf_hnm_512_p16_v1", "backbone_lr": BACKBONE_PEAK_LR,
        "head_lr": HEAD_PEAK_LR, "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM,
        "expected_world_size": 2, "scenes_per_rank_per_update": 6}
    if drift := {key: (value, config.get(key)) for key, value in expected.items() if config.get(key) != value}:
        raise ValueError(f"Multiscale calibration recipe changed: {drift}")
    if sha256_file(Path(config["source_audit"])) != hashes["source_audit"] or sha256_file(Path(config["code_manifest"])) != hashes["code_manifest"]:
        raise ValueError("Calibration arguments differ from the recipe records")
    return {**{key + "_sha256": value for key, value in hashes.items()},
            "code_manifest_schema_version": code["schema_version"],
            "pretraining_contract_schema_version": config["schema"],
            "running_calibration_script_sha256": sha256_file(Path(__file__)), "passed": True}


def stable_seed(*parts: Any) -> int:
    """Exact local copy of the experiment's frozen stable-seed function."""

    digest = hashlib.sha256("/".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one JSON object without exposing a partial report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_source_audit(
    audit: Mapping[str, Any],
    *,
    require_frozen_contract: bool,
) -> list[dict[str, Any]]:
    """Validate the fields needed to derive the panel and return train rows."""

    required_top_level = {
        "schema_version": audit.get("schema_version"),
        "dataset": "structured3d",
        "experiment_id": EXPERIMENT_ID,
        "passed": True,
        "artifact_hashes_verified": True,
        "crop_mode": "full_scene",
    }
    mismatches = {
        key: {"expected": expected, "observed": audit.get(key)}
        for key, expected in required_top_level.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Source-audit contract mismatch: {mismatches}")

    if audit.get("schema_version") not in {"structured3d_fullscene_source_audit/v1", "delimit3d_2d_source_audit/v1"}:
        raise ValueError("Unsupported calibration source audit")
    scenes = audit.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Source audit must contain a non-empty scenes list")
    seen: set[str] = set()
    optimization: list[dict[str, Any]] = []
    for raw in scenes:
        if not isinstance(raw, dict):
            raise TypeError("Every source-audit scene row must be an object")
        scene_id = str(raw.get("scene_id", ""))
        if not scene_id or scene_id in seen:
            raise ValueError(f"Missing or duplicate source-audit scene ID: {scene_id!r}")
        seen.add(scene_id)
        if raw.get("split") != "optimization":
            continue
        point_count = raw.get("num_points")
        proposals = raw.get("trainable_proposals")
        manifest_sha256 = raw.get("manifest_sha256")
        if not isinstance(point_count, int) or point_count <= 0:
            raise ValueError(f"{scene_id}: invalid num_points={point_count!r}")
        if not isinstance(proposals, dict):
            raise ValueError(f"{scene_id}: trainable_proposals is missing")
        for cell in SOURCE_CELLS:
            value = proposals.get(cell)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{scene_id}: invalid trainable proposal count for {cell}: {value!r}"
                )
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ValueError(f"{scene_id}: invalid manifest SHA-256")
        optimization.append(dict(raw))

    if require_frozen_contract:
        declared = audit.get("train_scene_count")
        if declared != EXPECTED_OPTIMIZATION_SCENES:
            raise ValueError(
                "Frozen audit train-scene count drift: "
                f"{declared!r} != {EXPECTED_OPTIMIZATION_SCENES}"
            )
        if len(optimization) != EXPECTED_OPTIMIZATION_SCENES:
            raise ValueError(
                "Frozen optimization-row count drift: "
                f"{len(optimization)} != {EXPECTED_OPTIMIZATION_SCENES}"
            )
    return optimization


def _point_rank_partition(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["num_points"]), str(row["scene_id"])),
    )
    total = len(ordered)
    if total < 16:
        raise ValueError("At least 16 optimization scenes are required")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        enriched = dict(row)
        enriched["point_quartile_index"] = min(3, (index * 4) // total)
        output.append(enriched)
    return output


def point_quartile_summary(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int, int], ...]:
    partitioned = _point_rank_partition(rows)
    summary: list[tuple[int, int, int]] = []
    for quartile in range(4):
        selected = [
            row
            for row in partitioned
            if int(row["point_quartile_index"]) == quartile
        ]
        summary.append(
            (
                len(selected),
                min(int(row["num_points"]) for row in selected),
                max(int(row["num_points"]) for row in selected),
            )
        )
    return tuple(summary)


def build_calibration_panel(
    audit: Mapping[str, Any],
    *,
    seed: int = FROZEN_SEED,
    require_frozen_contract: bool = False,
) -> list[dict[str, Any]]:
    """Build the deterministic point-quartile/proposal-density panel."""

    rows = validate_source_audit(
        audit,
        require_frozen_contract=require_frozen_contract,
    )
    partitioned = _point_rank_partition(rows)
    if require_frozen_contract:
        observed_quartiles = point_quartile_summary(rows)
        if observed_quartiles != FROZEN_POINT_QUARTILES:
            raise ValueError(
                "Frozen point-quartile boundary drift: "
                f"{observed_quartiles!r} != {FROZEN_POINT_QUARTILES!r}"
            )

    panel: list[dict[str, Any]] = []
    for quartile_index in range(4):
        quartile_rows = [
            row
            for row in partitioned
            if int(row["point_quartile_index"]) == quartile_index
        ]
        for cell in SOURCE_CELLS:
            by_density = sorted(
                quartile_rows,
                key=lambda row: (
                    Fraction(
                        int(row["trainable_proposals"][cell]),
                        int(row["num_points"]),
                    ),
                    str(row["scene_id"]),
                ),
            )
            split = len(by_density) // 2
            density_halves = {
                "low": by_density[:split],
                "high": by_density[split:],
            }
            for density_half, candidates in density_halves.items():
                if len(candidates) < SAMPLES_PER_STRATUM:
                    raise ValueError(
                        f"Insufficient scenes in q{quartile_index + 1}/{cell}/{density_half}"
                    )
                selected = sorted(
                    candidates,
                    key=lambda row: (
                        stable_seed(
                            seed,
                            PANEL_NAMESPACE,
                            quartile_index,
                            cell,
                            "L" if density_half == "low" else "H",
                            row["scene_id"],
                        ),
                        str(row["scene_id"]),
                    ),
                )[:SAMPLES_PER_STRATUM]
                for replicate, row in enumerate(selected):
                    proposal_count = int(row["trainable_proposals"][cell])
                    point_count = int(row["num_points"])
                    panel_key = (
                        f"q{quartile_index + 1}|{cell}|{density_half}|r{replicate}"
                    )
                    panel.append(
                        {
                            "panel_key": panel_key,
                            "point_quartile": quartile_index + 1,
                            "point_quartile_index": quartile_index,
                            "target_cell": cell,
                            "density_half": density_half,
                            "density_half_index": (
                                0 if density_half == "low" else 1
                            ),
                            "panel_replicate": replicate,
                            "scene_id": str(row["scene_id"]),
                            "num_points": point_count,
                            "trainable_proposals": proposal_count,
                            "proposal_density_numerator": proposal_count,
                            "proposal_density_denominator": point_count,
                            "manifest_sha256": str(row["manifest_sha256"]),
                        }
                    )

    expected_size = 4 * len(SOURCE_CELLS) * 2 * SAMPLES_PER_STRATUM
    if len(panel) != expected_size:
        raise AssertionError(f"Panel cardinality drift: {len(panel)} != {expected_size}")
    if len({row["panel_key"] for row in panel}) != len(panel):
        raise AssertionError("Panel keys are not unique")

    if require_frozen_contract:
        expected = {
            (quartile, cell, half): tuple(scene_ids)
            for quartile, cell, half, scene_ids in FROZEN_PANEL_SCENES
        }
        observed: dict[tuple[int, str, str], tuple[str, ...]] = {}
        for quartile, cell, half, _ in FROZEN_PANEL_SCENES:
            observed[(quartile, cell, half)] = tuple(
                str(row["scene_id"])
                for row in panel
                if row["point_quartile"] == quartile
                and row["target_cell"] == cell
                and row["density_half"] == half
            )
        if observed != expected:
            raise ValueError(
                "Frozen 48-scene panel reproduction failed: "
                f"observed={observed!r} expected={expected!r}"
            )
        if len(panel) != EXPECTED_PANEL_SIZE:
            raise AssertionError("Frozen panel does not contain exactly 48 rows")
        if len({str(row["scene_id"]) for row in panel}) != EXPECTED_PANEL_SIZE:
            raise ValueError("Frozen panel unexpectedly reuses a scene")
        observed_panel_sha256 = canonical_json_sha256(panel)
        if observed_panel_sha256 != FROZEN_PANEL_SHA256:
            raise ValueError(
                "Frozen panel content digest drift: "
                f"{observed_panel_sha256} != {FROZEN_PANEL_SHA256}"
            )
    return panel


def balanced_cell_for_scene_epoch(
    scene_id: str,
    epoch: int,
    *,
    seed: int = FROZEN_SEED,
    cells: Sequence[str] = SOURCE_CELLS,
) -> str:
    """Exact pure implementation of balanced_random_permutation_v1."""

    active = tuple(str(cell) for cell in cells)
    if not active or len(set(active)) != len(active):
        raise ValueError("Active source cells must be non-empty and unique")
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    cycle, position = divmod(int(epoch), len(active))
    active_key = ",".join(active)
    ordered = tuple(
        sorted(
            active,
            key=lambda cell: (
                sha256_bytes(
                    (
                        f"{scene_id}/{EXPERIMENT_ID}/balanced-cell/"
                        f"{active_key}/{int(seed)}/{cycle}/{cell}"
                    ).encode("utf-8")
                ),
                cell,
            ),
        )
    )
    return ordered[position]


def auxiliary_scale_for_epoch(
    epoch: int,
    *,
    maximum: float = AUXILIARY_WEIGHT,
    warmup_epochs: int = AUXILIARY_WARMUP_EPOCHS,
) -> float:
    if epoch < 0 or maximum < 0.0 or warmup_epochs < 0:
        raise ValueError("Invalid auxiliary schedule input")
    ramp = 1.0 if warmup_epochs == 0 else min(1.0, epoch / warmup_epochs)
    return maximum * ramp


def build_target_schedule(
    panel: Sequence[Mapping[str, Any]],
    *,
    seed: int = FROZEN_SEED,
    cycle_starts: Sequence[int] = FULL_WEIGHT_CYCLE_STARTS,
) -> list[dict[str, Any]]:
    """Expand every panel row into two exact full-weight target-cell visits."""

    if len(cycle_starts) != 2:
        raise ValueError("The frozen pilot requires exactly two schedule cycles")
    visits: list[dict[str, Any]] = []
    for cycle_index, cycle_start in enumerate(cycle_starts):
        epochs = tuple(range(int(cycle_start), int(cycle_start) + len(SOURCE_CELLS)))
        for panel_row in panel:
            scene_id = str(panel_row["scene_id"])
            target_cell = str(panel_row["target_cell"])
            matches = [
                epoch
                for epoch in epochs
                if balanced_cell_for_scene_epoch(
                    scene_id,
                    epoch,
                    seed=seed,
                    cells=SOURCE_CELLS,
                )
                == target_cell
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{scene_id}/{target_cell}: expected one target epoch in "
                    f"{epochs}, got {matches}"
                )
            epoch = matches[0]
            scale = auxiliary_scale_for_epoch(epoch)
            if scale != AUXILIARY_WEIGHT:
                raise ValueError(
                    f"Calibration visit is not at full auxiliary weight: epoch={epoch}"
                )
            half_index = int(panel_row["density_half_index"])
            replicate = int(panel_row["panel_replicate"])
            quartile_index = int(panel_row["point_quartile_index"])
            rank = (quartile_index + replicate + cycle_index + half_index) % 2
            global_update = cycle_index * 4 + half_index * 2 + replicate
            visit_id = f"{panel_row['panel_key']}|cycle{cycle_index}|epoch{epoch}"
            visits.append(
                {
                    **dict(panel_row),
                    "visit_id": visit_id,
                    "cycle_index": cycle_index,
                    "cycle_start": int(cycle_start),
                    "epoch": epoch,
                    "auxiliary_scale": scale,
                    "rank": rank,
                    "global_update": global_update,
                }
            )

    visits.sort(
        key=lambda row: (
            int(row["global_update"]),
            int(row["rank"]),
            SOURCE_CELLS.index(str(row["target_cell"])),
            int(row["point_quartile"]),
        )
    )
    checks = target_schedule_checks(visits, panel, seed=seed)
    if not all(checks.values()):
        raise ValueError(f"Target schedule failed invariants: {checks}")
    return visits


def target_schedule_checks(
    visits: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    *,
    seed: int = FROZEN_SEED,
) -> dict[str, bool]:
    expected_panel_keys = {str(row["panel_key"]) for row in panel}
    visit_ids = [str(row.get("visit_id", "")) for row in visits]
    checks: dict[str, bool] = {
        "schedule_cardinality_exact": len(visits) == 2 * len(panel),
        "visit_ids_unique": len(set(visit_ids)) == len(visit_ids),
        "cycle_indices_and_starts_exact": all(
            int(row["cycle_index"]) in range(len(FULL_WEIGHT_CYCLE_STARTS))
            and int(row["cycle_start"])
            == FULL_WEIGHT_CYCLE_STARTS[int(row["cycle_index"])]
            and int(row["epoch"])
            in range(
                int(row["cycle_start"]),
                int(row["cycle_start"]) + len(SOURCE_CELLS),
            )
            for row in visits
        ),
        "every_panel_row_has_two_cycles": all(
            {
                int(row["cycle_index"])
                for row in visits
                if str(row["panel_key"]) == panel_key
            }
            == {0, 1}
            for panel_key in expected_panel_keys
        ),
        "every_panel_row_visits_both_ranks": all(
            {
                int(row["rank"])
                for row in visits
                if str(row["panel_key"]) == panel_key
            }
            == set(range(EXPECTED_WORLD_SIZE))
            for panel_key in expected_panel_keys
        ),
        "every_visit_targets_declared_cell": all(
            balanced_cell_for_scene_epoch(
                str(row["scene_id"]),
                int(row["epoch"]),
                seed=seed,
                cells=SOURCE_CELLS,
            )
            == row["target_cell"]
            for row in visits
        ),
        "every_visit_uses_full_auxiliary_weight": all(
            auxiliary_scale_for_epoch(int(row["epoch"])) == AUXILIARY_WEIGHT
            and float(row["auxiliary_scale"]) == AUXILIARY_WEIGHT
            for row in visits
        ),
    }
    update_ids = sorted({int(row["global_update"]) for row in visits})
    checks["global_update_count_exact"] = update_ids == list(
        range(2 * 2 * SAMPLES_PER_STRATUM)
    )
    checks["global_updates_have_twelve_visits"] = all(
        sum(int(row["global_update"]) == update for row in visits) == 12
        for update in update_ids
    )
    checks["every_rank_has_six_visits_per_update"] = all(
        sum(
            int(row["global_update"]) == update and int(row["rank"]) == rank
            for row in visits
        )
        == SCENES_PER_RANK_PER_UPDATE
        for update in update_ids
        for rank in range(EXPECTED_WORLD_SIZE)
    )
    checks["every_update_covers_all_quartile_cell_pairs"] = all(
        {
            (int(row["point_quartile"]), str(row["target_cell"]))
            for row in visits
            if int(row["global_update"]) == update
        }
        == {(quartile, cell) for quartile in range(1, 5) for cell in SOURCE_CELLS}
        for update in update_ids
    )
    return checks


def primary_direction_factor(
    reference_norm_squared: float,
    reference_auxiliary_dot: float,
    *,
    auxiliary_weight: float = AUXILIARY_WEIGHT,
) -> float:
    """Compute d_alpha from scalar sufficient statistics."""

    values = (reference_norm_squared, reference_auxiliary_dot, auxiliary_weight)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Direction statistics must be finite")
    if reference_norm_squared <= 0.0:
        raise ValueError("Reference gradient norm squared must be positive")
    if auxiliary_weight < 0.0:
        raise ValueError("Auxiliary weight must be non-negative")
    return 1.0 + auxiliary_weight * reference_auxiliary_dot / reference_norm_squared


def direction_summary_from_statistics(
    reference_norm_squared: float,
    auxiliary_norm_squared: float,
    reference_auxiliary_dot: float,
    *,
    auxiliary_weight: float = AUXILIARY_WEIGHT,
) -> dict[str, Any]:
    if auxiliary_norm_squared < 0.0 or not math.isfinite(auxiliary_norm_squared):
        raise ValueError("Auxiliary gradient norm squared must be finite and non-negative")
    factor = primary_direction_factor(
        reference_norm_squared,
        reference_auxiliary_dot,
        auxiliary_weight=auxiliary_weight,
    )
    reversal = (
        reference_norm_squared / -reference_auxiliary_dot
        if reference_auxiliary_dot < 0.0
        else None
    )
    return {
        "reference_gradient_norm": math.sqrt(reference_norm_squared),
        "weighted_stage_auxiliary_gradient_norm": math.sqrt(
            auxiliary_norm_squared
        ),
        "reference_auxiliary_dot_product": reference_auxiliary_dot,
        "auxiliary_to_reference_norm_ratio": (
            math.sqrt(auxiliary_norm_squared / reference_norm_squared)
        ),
        "effective_auxiliary_to_reference_norm_ratio": (
            auxiliary_weight
            * math.sqrt(auxiliary_norm_squared / reference_norm_squared)
        ),
        "d_0_4": factor,
        "candidate_weight_at_primary_direction_reversal": reversal,
        "primary_direction_positive": factor > 0.0,
    }


def local_optimizer_update_checksum(
    *,
    rank: int,
    global_update: int,
    visits: Sequence[Mapping[str, Any]],
) -> str:
    """Digest the ordered, model-independent inputs to one rank update."""

    return canonical_json_sha256(
        {
            "rank": int(rank),
            "global_update": int(global_update),
            "visits": [
                {
                    "visit_id": str(row.get("visit_id", "")),
                    "input_sha256": str(row.get("input_sha256", "")),
                }
                for row in visits
            ],
        }
    )


def global_optimizer_update_checksum(
    *,
    global_update: int,
    rank_inputs: Sequence[Mapping[str, Any]],
) -> str:
    """Digest both ordered rank inputs to one synchronized DDP update."""

    ordered = sorted(rank_inputs, key=lambda row: int(row["rank"]))
    return canonical_json_sha256(
        {
            "global_update": int(global_update),
            "rank_inputs": [
                {
                    "rank": int(row["rank"]),
                    "local_update_input_sha256": str(
                        row["local_update_input_sha256"]
                    ),
                }
                for row in ordered
            ],
        }
    )


def _optimizer_group_contract_exact(optimizer: Mapping[str, Any]) -> bool:
    if (
        optimizer.get("class") != "AdamW"
        or float(optimizer.get("backbone_peak_lr", -1.0)) != BACKBONE_PEAK_LR
        or float(optimizer.get("head_peak_lr", -1.0)) != HEAD_PEAK_LR
        or float(optimizer.get("weight_decay", -1.0)) != WEIGHT_DECAY
        or float(optimizer.get("gradient_clip_norm", -1.0)) != GRAD_CLIP_NORM
        or optimizer.get("initial_state_empty") is not True
        or not _is_sha256(optimizer.get("parameter_group_membership_sha256"))
    ):
        return False
    groups = optimizer.get("parameter_groups")
    if not isinstance(groups, list) or not groups:
        return False
    observed_names: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            return False
        name = str(group.get("group_name", ""))
        if name in observed_names or name not in {
            "backbone/decay",
            "backbone/no_decay",
            "head/decay",
            "head/no_decay",
        }:
            return False
        observed_names.add(name)
        family = name.split("/", 1)[0]
        expected_lr = BACKBONE_PEAK_LR if family == "backbone" else HEAD_PEAK_LR
        expected_decay = 0.0 if name.endswith("/no_decay") else WEIGHT_DECAY
        if (
            float(group.get("lr", -1.0)) != expected_lr
            or float(group.get("initial_lr", -1.0)) != expected_lr
            or float(group.get("weight_decay", -1.0)) != expected_decay
            or int(group.get("parameter_tensors", 0)) <= 0
            or int(group.get("parameters", 0)) <= 0
        ):
            return False
    return observed_names == {
        "backbone/decay",
        "backbone/no_decay",
        "head/decay",
        "head/no_decay",
    }


def _clip_record_exact(record: Mapping[str, Any]) -> bool:
    keys = (
        "pre_clip_norm",
        "post_clip_norm",
        "clip_return_norm",
        "clip_coefficient",
    )
    try:
        values = {key: float(record[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values.values()):
        return False
    pre = values["pre_clip_norm"]
    post = values["post_clip_norm"]
    returned = values["clip_return_norm"]
    coefficient = values["clip_coefficient"]
    expected_coefficient = min(1.0, GRAD_CLIP_NORM / (returned + 1.0e-6))
    tolerance = max(1.0e-6, 1.0e-5 * max(1.0, pre, returned))
    return (
        pre > 0.0
        and returned > 0.0
        and abs(pre - returned) <= tolerance
        and 0.0 < coefficient <= 1.0
        and abs(coefficient - expected_coefficient) <= 1.0e-6
        and post <= GRAD_CLIP_NORM + 1.0e-5
        and abs(post - min(pre, GRAD_CLIP_NORM)) <= tolerance
    )


def _summarize_paired_optimizer_branches_impl(
    branch_reports: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate paired C/M optimizer evidence without interpreting short losses."""

    checks: dict[str, bool] = {}
    checks["branch_set_exact"] = set(branch_reports) == set(
        PAIRED_BRANCH_MULTIPLIERS
    )
    expected_by_rank_update = {
        (rank, update): [
            str(row["visit_id"])
            for row in schedule
            if int(row["rank"]) == rank
            and int(row["global_update"]) == update
        ]
        for rank in range(EXPECTED_WORLD_SIZE)
        for update in range(EXPECTED_GLOBAL_UPDATES)
    }
    reports_by_branch_rank: dict[str, dict[int, Mapping[str, Any]]] = {}
    structurally_valid = checks["branch_set_exact"]
    for branch in PAIRED_BRANCH_MULTIPLIERS:
        raw_reports = branch_reports.get(branch, ())
        rank_map: dict[int, Mapping[str, Any]] = {}
        for report in raw_reports:
            if not isinstance(report, Mapping):
                structurally_valid = False
                continue
            try:
                report_rank = int(report["rank"])
            except (KeyError, TypeError, ValueError):
                structurally_valid = False
                continue
            if report_rank in rank_map:
                structurally_valid = False
            rank_map[report_rank] = report
        reports_by_branch_rank[branch] = rank_map
        structurally_valid &= set(rank_map) == set(range(EXPECTED_WORLD_SIZE))
    checks["two_rank_reports_per_branch"] = structurally_valid

    all_reports = [
        report
        for rank_map in reports_by_branch_rank.values()
        for report in rank_map.values()
    ]
    checks["branch_multiplier_exact"] = structurally_valid and all(
        report.get("branch") == branch
        and float(report.get("auxiliary_multiplier", -1.0)) == multiplier
        for branch, multiplier in PAIRED_BRANCH_MULTIPLIERS.items()
        for report in reports_by_branch_rank[branch].values()
    )
    checks["actual_two_rank_ddp_used"] = structurally_valid and all(
        report.get("ddp_wrapped") is True
        and int(report.get("world_size", -1)) == EXPECTED_WORLD_SIZE
        for report in all_reports
    )
    checks["strict_identical_initial_model"] = structurally_valid and all(
        report.get("strict_initialization_passed") is True
        and _is_sha256(report.get("initial_model_state_sha256"))
        and _is_sha256(report.get("initial_checkpoint_sha256"))
        for report in all_reports
    ) and len(
        {str(report["initial_model_state_sha256"]) for report in all_reports}
    ) == 1 and len(
        {str(report["initial_checkpoint_sha256"]) for report in all_reports}
    ) == 1
    checks["production_adamw_contract_exact"] = structurally_valid and all(
        isinstance(report.get("optimizer"), Mapping)
        and _optimizer_group_contract_exact(report["optimizer"])
        for report in all_reports
    )
    optimizer_membership = {
        str(report.get("optimizer", {}).get("parameter_group_membership_sha256", ""))
        for report in all_reports
    }
    checks["optimizer_group_membership_identical"] = (
        structurally_valid
        and len(optimizer_membership) == 1
        and all(_is_sha256(value) for value in optimizer_membership)
    )
    checks["rng_reset_identical_between_paired_branches"] = structurally_valid and all(
        int(reports_by_branch_rank[branch][rank].get("rng_reset_seed", -1))
        == FROZEN_SEED
        and _is_sha256(
            reports_by_branch_rank[branch][rank].get(
                "rng_state_before_first_visit_sha256"
            )
        )
        for branch in PAIRED_BRANCH_MULTIPLIERS
        for rank in range(EXPECTED_WORLD_SIZE)
    ) and all(
        reports_by_branch_rank["C"][rank][
            "rng_state_before_first_visit_sha256"
        ]
        == reports_by_branch_rank["M"][rank][
            "rng_state_before_first_visit_sha256"
        ]
        for rank in range(EXPECTED_WORLD_SIZE)
    )

    updates_by_branch_rank: dict[
        str, dict[int, dict[int, Mapping[str, Any]]]
    ] = {branch: {} for branch in PAIRED_BRANCH_MULTIPLIERS}
    update_structure_exact = structurally_valid
    for branch in PAIRED_BRANCH_MULTIPLIERS:
        for rank in range(EXPECTED_WORLD_SIZE):
            report = reports_by_branch_rank[branch].get(rank, {})
            raw_updates = report.get("updates", [])
            update_map: dict[int, Mapping[str, Any]] = {}
            if not isinstance(raw_updates, list):
                update_structure_exact = False
                raw_updates = []
            for update_record in raw_updates:
                if not isinstance(update_record, Mapping):
                    update_structure_exact = False
                    continue
                try:
                    update = int(update_record["global_update"])
                except (KeyError, TypeError, ValueError):
                    update_structure_exact = False
                    continue
                if update in update_map:
                    update_structure_exact = False
                update_map[update] = update_record
            updates_by_branch_rank[branch][rank] = update_map
            update_structure_exact &= (
                int(report.get("optimizer_steps", -1)) == EXPECTED_GLOBAL_UPDATES
                and set(update_map) == set(range(EXPECTED_GLOBAL_UPDATES))
            )
    checks["exactly_eight_ddp_global_updates_per_branch"] = update_structure_exact

    schedule_exact = update_structure_exact
    local_checksums_exact = update_structure_exact
    global_checksums_exact = update_structure_exact
    finite_mechanics = update_structure_exact
    synchronized_gradients = update_structure_exact
    visit_checksum_maps: dict[str, dict[str, str]] = {
        branch: {} for branch in PAIRED_BRANCH_MULTIPLIERS
    }
    update_checksum_maps: dict[str, dict[int, str]] = {
        branch: {} for branch in PAIRED_BRANCH_MULTIPLIERS
    }
    for branch in PAIRED_BRANCH_MULTIPLIERS:
        for update in range(EXPECTED_GLOBAL_UPDATES):
            rank_inputs: list[dict[str, Any]] = []
            pre_clip_norms: list[float] = []
            observed_global_hashes: set[str] = set()
            for rank in range(EXPECTED_WORLD_SIZE):
                record = updates_by_branch_rank[branch].get(rank, {}).get(update, {})
                visits = record.get("visits", [])
                if not isinstance(visits, list):
                    visits = []
                    schedule_exact = False
                visit_ids = [str(row.get("visit_id", "")) for row in visits]
                schedule_exact &= (
                    int(record.get("rank", -1)) == rank
                    and len(visits) == SCENES_PER_RANK_PER_UPDATE
                    and visit_ids == expected_by_rank_update[(rank, update)]
                )
                for visit in visits:
                    visit_id = str(visit.get("visit_id", ""))
                    input_sha256 = visit.get("input_sha256")
                    if not _is_sha256(input_sha256) or visit_id in visit_checksum_maps[branch]:
                        schedule_exact = False
                    else:
                        visit_checksum_maps[branch][visit_id] = str(input_sha256)
                expected_local = local_optimizer_update_checksum(
                    rank=rank,
                    global_update=update,
                    visits=visits,
                )
                observed_local = str(record.get("local_update_input_sha256", ""))
                local_checksums_exact &= observed_local == expected_local
                rank_inputs.append(
                    {
                        "rank": rank,
                        "local_update_input_sha256": expected_local,
                    }
                )
                observed_global_hashes.add(
                    str(record.get("global_update_input_sha256", ""))
                )
                losses = record.get("losses", [])
                gradient = record.get("gradient", {})
                finite_mechanics &= (
                    isinstance(losses, list)
                    and len(losses) == SCENES_PER_RANK_PER_UPDATE
                    and all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in losses
                    )
                    and record.get("all_losses_finite") is True
                    and isinstance(gradient, Mapping)
                    and _clip_record_exact(gradient)
                    and record.get("optimizer_state_finite") is True
                    and int(record.get("optimizer_state_tensor_count", 0)) > 0
                    and record.get("parameters_finite_after_step") is True
                )
                try:
                    pre_clip_norms.append(float(gradient["pre_clip_norm"]))
                except (KeyError, TypeError, ValueError):
                    synchronized_gradients = False
            expected_global = global_optimizer_update_checksum(
                global_update=update,
                rank_inputs=rank_inputs,
            )
            global_checksums_exact &= observed_global_hashes == {expected_global}
            update_checksum_maps[branch][update] = expected_global
            if len(pre_clip_norms) != EXPECTED_WORLD_SIZE:
                synchronized_gradients = False
            else:
                spread = max(pre_clip_norms) - min(pre_clip_norms)
                tolerance = max(1.0e-7, 1.0e-6 * max(pre_clip_norms))
                synchronized_gradients &= spread <= tolerance
    expected_visit_ids = {str(row["visit_id"]) for row in schedule}
    schedule_exact &= all(
        set(visit_checksum_maps[branch]) == expected_visit_ids
        for branch in PAIRED_BRANCH_MULTIPLIERS
    )
    checks["six_frozen_visits_per_rank_update_exact"] = schedule_exact
    checks["local_update_checksums_recomputed_exact"] = local_checksums_exact
    checks["global_update_checksums_recomputed_exact"] = global_checksums_exact
    checks["loss_gradient_clip_and_optimizer_state_finite"] = finite_mechanics
    checks["ddp_gradient_norms_synchronized"] = synchronized_gradients
    checks["paired_visit_checksums_exact"] = (
        visit_checksum_maps.get("C") == visit_checksum_maps.get("M")
        and set(visit_checksum_maps.get("C", {})) == expected_visit_ids
    )
    checks["paired_update_checksums_exact"] = (
        update_checksum_maps.get("C") == update_checksum_maps.get("M")
        and set(update_checksum_maps.get("C", {}))
        == set(range(EXPECTED_GLOBAL_UPDATES))
    )

    passed = bool(checks) and all(checks.values())
    return {
        "completed": passed,
        "passed": passed,
        "status": "passed" if passed else "failed_closed",
        "required_before_336_epoch_launch": True,
        "hard_checks": checks,
        "branch_multipliers": dict(PAIRED_BRANCH_MULTIPLIERS),
        "optimizer_contract": {
            "class": "AdamW",
            "backbone_peak_lr": BACKBONE_PEAK_LR,
            "head_peak_lr": HEAD_PEAK_LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "global_updates_per_branch": EXPECTED_GLOBAL_UPDATES,
            "scenes_per_rank_per_update": SCENES_PER_RANK_PER_UPDATE,
        },
        "paired_visit_checksum_sha256": (
            canonical_json_sha256(visit_checksum_maps["C"]) if passed else None
        ),
        "paired_update_checksum_sha256": (
            canonical_json_sha256(update_checksum_maps["C"]) if passed else None
        ),
        "scientific_interpretation": (
            "Mechanical paired-optimization gate only. Short calibration losses "
            "are not evidence of representation quality."
        ),
    }


def summarize_paired_optimizer_branches(
    branch_reports: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an affirmative paired gate only for complete, well-formed evidence."""

    try:
        return _summarize_paired_optimizer_branches_impl(
            branch_reports,
            schedule,
        )
    except Exception as exc:
        return {
            "completed": False,
            "passed": False,
            "status": "failed_closed",
            "required_before_336_epoch_launch": True,
            "hard_checks": {"paired_evidence_well_formed": False},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "branch_multipliers": dict(PAIRED_BRANCH_MULTIPLIERS),
            "scientific_interpretation": (
                "Malformed paired evidence cannot authorize training. Short "
                "calibration losses are not evidence of representation quality."
            ),
        }


def summarize_calibration_records(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    paired_optimizer_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce gradient records and combine them with the paired mechanical gate."""

    expected_ids = {str(row["visit_id"]) for row in schedule}
    observed_ids = [str(row.get("visit_id", "")) for row in records]
    record_by_id = {str(row.get("visit_id", "")): row for row in records}

    level_cell_summary: dict[str, dict[str, Any]] = {}
    realized_medians_nonzero = True
    every_level_cell_retains = True
    for level in LEVEL_NAMES:
        level_cell_summary[level] = {}
        site = LEVEL_NATIVE_SITES[level]
        for cell in SOURCE_CELLS:
            selected = [row for row in records if row.get("target_cell") == cell]
            kept = [
                int(row.get("levels", {}).get(level, {}).get("kept_proposals", 0))
                for row in selected
            ]
            effective_norms = [
                AUXILIARY_WEIGHT
                * LEVEL_WEIGHTS[level]
                * float(
                    row.get("gradient_sites", {})
                    .get(site, {})
                    .get("components", {})
                    .get(level, {})
                    .get("gradient_norm", 0.0)
                )
                for row in selected
            ]
            median_effective_norm = (
                float(statistics.median(effective_norms))
                if effective_norms
                else 0.0
            )
            retained_visits = sum(value > 0 for value in kept)
            every_level_cell_retains &= sum(kept) > 0
            realized_medians_nonzero &= (
                math.isfinite(median_effective_norm) and median_effective_norm > 0.0
            )
            level_cell_summary[level][cell] = {
                "attempted_visits": len(selected),
                "visits_with_retained_proposals": retained_visits,
                "retained_proposals": sum(kept),
                "median_realized_effective_auxiliary_gradient_norm": (
                    median_effective_norm
                ),
            }

    direction_values: dict[str, list[float]] = {
        site: [] for site in PRIMARY_DIRECTION_SITES
    }
    for row in records:
        for site in PRIMARY_DIRECTION_SITES:
            value = row.get("gradient_sites", {}).get(site, {}).get("d_0_4")
            if isinstance(value, (int, float)):
                direction_values[site].append(float(value))
    direction_summary = {
        site: {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": float(statistics.median(values)) if values else None,
            "maximum": max(values) if values else None,
        }
        for site, values in direction_values.items()
    }

    schedule_checks = target_schedule_checks(
        schedule,
        [
            {
                "panel_key": key,
            }
            for key in sorted({str(row["panel_key"]) for row in schedule})
        ],
    )
    hard_checks = {
        "record_cardinality_exact": len(records) == len(schedule),
        "record_visit_ids_unique": len(set(observed_ids)) == len(observed_ids),
        "record_visit_set_exact": set(observed_ids) == expected_ids,
        "schedule_invariants_passed": all(schedule_checks.values()),
        "every_record_hard_invariant_passed": bool(records)
        and all(
            isinstance(row.get("hard_invariants"), Mapping)
            and REQUIRED_RECORD_HARD_INVARIANTS.issubset(
                row["hard_invariants"]
            )
            and all(
                value is True
                for value in row["hard_invariants"].values()
            )
            for row in records
        ),
        "every_level_cell_retains_a_proposal": every_level_cell_retains,
        "every_level_cell_has_nonzero_median_realized_auxiliary_gradient": (
            realized_medians_nonzero
        ),
        "every_primary_direction_factor_is_finite_positive": bool(records)
        and all(
            len(direction_values[site]) == len(records)
            and all(math.isfinite(value) and value > 0.0 for value in direction_values[site])
            for site in PRIMARY_DIRECTION_SITES
        ),
        "dec0_is_auxiliary_topology_negative_control": bool(records)
        and all(
            bool(row.get("gradient_sites", {}).get("dec0", {}).get("topology_passed"))
            and float(row["gradient_sites"]["dec0"].get("weighted_stage_auxiliary_gradient_norm", -1.0))
            == 0.0
            and float(row["gradient_sites"]["dec0"].get("d_0_4", float("nan")))
            == 1.0
            for row in records
        ),
    }
    gradient_gate_passed = all(hard_checks.values())
    paired_gate = dict(paired_optimizer_gate or {})
    paired_passed = bool(
        paired_gate.get("completed") is True
        and paired_gate.get("passed") is True
        and paired_gate.get("status") == "passed"
    )
    if not paired_gate:
        paired_gate = {
            "completed": False,
            "passed": False,
            "status": "unmet_fail_closed",
            "required_before_336_epoch_launch": True,
            "reason": "No paired optimizer evidence was supplied.",
        }
    training_authorized = gradient_gate_passed and paired_passed
    return {
        "records_expected": len(schedule),
        "records_observed": len(records),
        "record_digest_by_visit": {
            visit_id: canonical_json_sha256(record_by_id[visit_id])
            for visit_id in sorted(expected_ids & set(record_by_id))
        },
        "level_cell_summary": level_cell_summary,
        "primary_direction_summary": direction_summary,
        "hard_checks": hard_checks,
        "required_record_hard_invariants": sorted(
            REQUIRED_RECORD_HARD_INVARIANTS
        ),
        "gradient_calibration_gate_passed": gradient_gate_passed,
        "paired_optimizer_gate": paired_gate,
        "training_authorized": training_authorized,
        "launch_gate_passed": training_authorized,
        "passed": training_authorized,
    }


def _tensor_fingerprint(torch: Any, batch: Any, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    tensor_fields = (
        "positive_pairs",
        "negative_indices",
        "proposal_labels",
        "feature_candidate_indices",
        "proposal_member_offsets",
        "proposal_member_indices",
        "pair_proposal_indices",
        "eligible_indices",
    )
    for name in tensor_fields:
        value = getattr(batch, name)
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "multiscale_seed": getattr(batch, "multiscale_seed"),
                "num_feature_hard_negatives": int(
                    getattr(batch, "num_feature_hard_negatives")
                ),
                "metadata": dict(metadata),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _gradient_component_report(
    torch: Any,
    gradients: Sequence[Any | None],
    indices: Sequence[int],
) -> dict[str, Any]:
    present = [gradients[index] for index in indices if gradients[index] is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in present)
    norm_squared = sum(
        (gradient.detach().float().square().sum() for gradient in present),
        start=torch.zeros((), device=present[0].device)
        if present
        else torch.zeros(()),
    )
    norm_squared_value = float(norm_squared)
    return {
        "parameter_tensors": len(indices),
        "present_gradient_tensors": len(present),
        "missing_gradient_tensors": len(indices) - len(present),
        "all_present_gradients_finite": finite,
        "gradient_norm": math.sqrt(max(0.0, norm_squared_value)),
        "gradient_nonzero": norm_squared_value > 0.0,
    }


def _gradient_site_report(
    torch: Any,
    model: Any,
    components: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    enc = model.backbone.model.enc
    dec = model.backbone.model.dec
    if len(enc) != 5 or len(dec) != 4:
        raise RuntimeError(
            f"Unexpected LitePT hierarchy: enc={len(enc)} dec={len(dec)}"
        )
    modules = {
        "enc0": enc[0],
        "enc4": enc[4],
        "dec3": dec[0],
        "dec2": dec[1],
        "dec1": dec[2],
        "dec0": dec[3],
    }
    parameters: list[Any] = []
    site_indices: dict[str, list[int]] = {}
    seen_parameter_ids: set[int] = set()
    for site, module in modules.items():
        indices: list[int] = []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen_parameter_ids:
                raise RuntimeError(f"Gradient probe parameter is shared across sites: {site}")
            seen_parameter_ids.add(id(parameter))
            indices.append(len(parameters))
            parameters.append(parameter)
        if not indices:
            raise RuntimeError(f"Gradient probe site has no parameters: {site}")
        site_indices[site] = indices
    site_indices["shared_union"] = [
        index
        for site in ("enc0", "enc4", "dec3", "dec2", "dec1")
        for index in site_indices[site]
    ]

    required_components = ("final_dec0", *LEVEL_NAMES)
    if set(components) != set(required_components):
        raise RuntimeError(
            f"Loss-component contract drift: {sorted(components)} != "
            f"{sorted(required_components)}"
        )
    reference = torch.autograd.grad(
        components["final_dec0"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    reference = tuple(
        gradient.detach() if gradient is not None else None
        for gradient in reference
    )
    weighted_auxiliary = [torch.zeros_like(parameter) for parameter in parameters]
    component_gradients: dict[str, tuple[Any | None, ...]] = {}
    for level_index, level in enumerate(LEVEL_NAMES):
        gradients = torch.autograd.grad(
            components[level],
            parameters,
            retain_graph=level_index + 1 < len(LEVEL_NAMES),
            allow_unused=True,
        )
        detached = tuple(
            gradient.detach() if gradient is not None else None
            for gradient in gradients
        )
        component_gradients[level] = detached
        for index, gradient in enumerate(detached):
            if gradient is not None:
                weighted_auxiliary[index].add_(
                    gradient,
                    alpha=float(LEVEL_WEIGHTS[level]),
                )
        del gradients

    reports: dict[str, Any] = {}
    hard_checks: dict[str, bool] = {}
    for site, indices in site_indices.items():
        expected_paths = (
            LEVEL_NAMES if site == "shared_union" else SITE_COMPONENT_PATHS[site]
        )
        final_report = _gradient_component_report(torch, reference, indices)
        component_reports = {
            level: _gradient_component_report(
                torch,
                component_gradients[level],
                indices,
            )
            for level in LEVEL_NAMES
        }
        topology_passed = all(
            (
                component_reports[level]["present_gradient_tensors"] > 0
                if level in expected_paths
                else component_reports[level]["present_gradient_tensors"] == 0
            )
            for level in LEVEL_NAMES
        )
        finite_components = all(
            report["all_present_gradients_finite"]
            for report in component_reports.values()
        )
        reference_present = final_report["present_gradient_tensors"] > 0
        reference_finite_nonzero = (
            final_report["all_present_gradients_finite"]
            and final_report["gradient_nonzero"]
        )

        reference_norm_squared_tensor = torch.zeros(
            (), device=parameters[indices[0]].device
        )
        auxiliary_norm_squared_tensor = torch.zeros_like(
            reference_norm_squared_tensor
        )
        dot_tensor = torch.zeros_like(reference_norm_squared_tensor)
        for index in indices:
            reference_gradient = reference[index]
            auxiliary_gradient = weighted_auxiliary[index]
            if reference_gradient is not None:
                reference_float = reference_gradient.float()
                auxiliary_float = auxiliary_gradient.float()
                reference_norm_squared_tensor += reference_float.square().sum()
                auxiliary_norm_squared_tensor += auxiliary_float.square().sum()
                dot_tensor += (reference_float * auxiliary_float).sum()
            elif bool(auxiliary_gradient.detach().float().abs().sum() != 0):
                finite_components = False
        reference_norm_squared = float(reference_norm_squared_tensor)
        auxiliary_norm_squared = float(auxiliary_norm_squared_tensor)
        dot = float(dot_tensor)
        direction = direction_summary_from_statistics(
            reference_norm_squared,
            auxiliary_norm_squared,
            dot,
        )
        report = {
            "expected_auxiliary_components": list(expected_paths),
            "final_dec0": final_report,
            "components": component_reports,
            "topology_passed": topology_passed,
            "all_component_gradients_finite": finite_components,
            **direction,
        }
        reports[site] = report
        hard_checks[f"{site}_final_gradient_present_finite_nonzero"] = (
            reference_present and reference_finite_nonzero
        )
        hard_checks[f"{site}_component_topology_exact"] = topology_passed
        hard_checks[f"{site}_component_gradients_finite"] = finite_components
        if site in PRIMARY_DIRECTION_SITES:
            hard_checks[f"{site}_d_0_4_positive"] = bool(
                direction["primary_direction_positive"]
            )
        if site == "dec0":
            hard_checks["dec0_negative_control_exact"] = (
                auxiliary_norm_squared == 0.0 and direction["d_0_4"] == 1.0
            )
    return reports, hard_checks


def _level_record(
    level: str,
    metrics: Mapping[str, Any],
    *,
    proposals_per_forward: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    proposal_rows_raw = metrics.get("proposal_diagnostics")
    proposal_rows = (
        [dict(row) for row in proposal_rows_raw]
        if isinstance(proposal_rows_raw, list)
        else []
    )
    retained = sum(bool(row.get("retained")) for row in proposal_rows)
    pair_budget = sum(int(row.get("pair_budget", 0)) for row in proposal_rows)
    kept_proposals = int(metrics.get("kept_proposals", -1))
    kept_pairs = int(metrics.get("kept_pairs", -1))
    valid_batch = kept_pairs > 0
    duplicate_fraction = metrics.get("base_negative_duplicate_fraction")
    diversity_valid = (
        not valid_batch
        or (
            int(metrics.get("negative_diversity_measured", 0)) == 1
            and isinstance(duplicate_fraction, (int, float))
            and math.isfinite(float(duplicate_fraction))
            and 0.0 <= float(duplicate_fraction) <= 1.0
            and int(metrics.get("base_negative_unique_per_pair_min", 0)) >= 1
        )
    )
    proposal_logic = all(
        (
            bool(row.get("retained"))
            and int(row.get("pure_positive_token_count", 0)) >= 2
            and int(row.get("pure_negative_token_count", 0)) >= 1
            and row.get("drop_reason") == ""
        )
        or (
            not bool(row.get("retained"))
            and row.get("drop_reason")
            in {"insufficient_pure_positive", "no_pure_negative"}
        )
        for row in proposal_rows
    )
    checks = {
        "proposal_diagnostic_cardinality_exact": len(proposal_rows)
        == proposals_per_forward,
        "input_proposal_cardinality_exact": int(metrics.get("input_proposals", -1))
        == proposals_per_forward,
        "retained_proposal_count_exact": retained == kept_proposals,
        "input_pair_budget_exact": pair_budget == int(metrics.get("input_pairs", -1)),
        "proposal_retention_logic_valid": proposal_logic,
        "strict_token_purity_exact": float(metrics.get("strict_token_purity", 0.0))
        == 1.0,
        "accepted_mixed_tokens_zero": int(metrics.get("accepted_mixed_tokens", -1))
        == 0,
        "negative_diversity_valid_when_batch_survives": diversity_valid,
        "stage_weight_exact": float(metrics.get("stage_weight", -1.0))
        == LEVEL_WEIGHTS[level],
    }
    record = {
        "loss": float(metrics.get("loss", 0.0)),
        "token_count": int(metrics.get("token_count", 0)),
        "input_pairs": int(metrics.get("input_pairs", 0)),
        "input_proposals": int(metrics.get("input_proposals", 0)),
        "kept_pairs": kept_pairs,
        "kept_proposals": kept_proposals,
        "pure_positive_tokens": int(metrics.get("pure_positive_tokens", 0)),
        "pure_negative_tokens": int(metrics.get("pure_negative_tokens", 0)),
        "mixed_or_unknown_tokens_ignored": int(
            metrics.get("mixed_or_unknown_tokens_ignored", 0)
        ),
        "base_negative_duplicate_fraction": (
            float(duplicate_fraction)
            if isinstance(duplicate_fraction, (int, float))
            else None
        ),
        "hardest_negative_ranking_accuracy": float(
            metrics.get("hardest_negative_ranking_accuracy", 0.0)
        ),
        "cosine_gap": float(metrics.get("cosine_gap", 0.0)),
        "temperature": float(metrics.get("temperature", 0.0)),
        "proposal_diagnostics": proposal_rows,
        "hard_invariants": checks,
    }
    return record, checks


def _run_visit(
    *,
    torch: Any,
    np: Any,
    visit: Mapping[str, Any],
    rank: int,
    device: Any,
    source_root: Path,
    model: Any,
    frozen_buffers: Mapping[str, Any],
    sampling: Any,
    load_scene_source: Any,
    build_scene_visit_input: Any,
    training_batch: Any,
    canonical_source_cell_for_scene_epoch: Any,
    train_augmentation_profile: str,
    color_mean: Sequence[float],
    color_std: Sequence[float],
    canonical_stable_seed: Any,
) -> dict[str, Any]:
    scene_id = str(visit["scene_id"])
    epoch = int(visit["epoch"])
    target_cell = str(visit["target_cell"])
    manifest_path = source_root / scene_id / "source_manifest.json"
    manifest_path = manifest_path.resolve(strict=True)
    manifest_sha256 = sha256_file(manifest_path)
    manifest_hash_exact = manifest_sha256 == visit["manifest_sha256"]
    if not manifest_hash_exact:
        raise RuntimeError(f"{scene_id}: source-manifest hash drift")

    source = load_scene_source(
        manifest_path,
        input_features="rgbn6",
        color_mean=color_mean,
        color_std=color_std,
        verify_hashes=True,
    )
    if source.crop_mode != "full_scene":
        raise RuntimeError(f"{scene_id}: calibration source is not full-scene")
    if int(source.points.shape[0]) != int(visit["num_points"]):
        raise RuntimeError(f"{scene_id}: point-count drift")

    canonical_cell = canonical_source_cell_for_scene_epoch(
        scene_id,
        epoch,
        cells=SOURCE_CELLS,
        schedule="balanced_random_permutation_v1",
        seed=FROZEN_SEED,
    )
    local_cell = balanced_cell_for_scene_epoch(scene_id, epoch)
    canonical_schedule_exact = canonical_cell == local_cell == target_cell
    if not canonical_schedule_exact:
        raise RuntimeError(
            f"{scene_id}/{epoch}: canonical target-cell mismatch "
            f"canonical={canonical_cell} local={local_cell} target={target_cell}"
        )

    with torch.no_grad():
        named_buffers = dict(model.named_buffers())
        if set(named_buffers) != set(frozen_buffers):
            raise RuntimeError("Model buffer topology drifted during calibration")
        for name, frozen in frozen_buffers.items():
            named_buffers[name].copy_(frozen)

    visit_input = build_scene_visit_input(
        scene=source,
        epoch=epoch,
        global_seed=FROZEN_SEED,
        input_features="rgbn6",
        augmentation_profile=train_augmentation_profile,
        color_mean=color_mean,
        color_std=color_std,
    )
    points = torch.from_numpy(visit_input.points).to(device)
    features = torch.from_numpy(visit_input.features).to(device)
    observed_cell, batch, batch_metadata = training_batch(
        scene=source,
        epoch=epoch,
        points=points,
        device=device,
        sampling=sampling,
        return_metadata=True,
        cells=SOURCE_CELLS,
        cell_schedule="balanced_random_permutation_v1",
        schedule_seed=FROZEN_SEED,
    )
    if observed_cell != target_cell:
        raise RuntimeError(
            f"{scene_id}/{epoch}: training_batch returned {observed_cell}, "
            f"expected {target_cell}"
        )
    batch_sha256 = _tensor_fingerprint(torch, batch, batch_metadata)

    forward_seed = canonical_stable_seed(
        FROZEN_SEED,
        rank,
        scene_id,
        epoch,
        "train-forward",
    )
    if int(forward_seed) != stable_seed(
        FROZEN_SEED,
        rank,
        scene_id,
        epoch,
        "train-forward",
    ):
        raise RuntimeError("Local stable-seed implementation drift")
    random.seed(forward_seed)
    np.random.seed(forward_seed % (2**32))
    torch.manual_seed(forward_seed)
    torch.cuda.manual_seed_all(forward_seed)
    model.zero_grad(set_to_none=True)
    result = model(
        points,
        features,
        batch,
        multiscale_loss_scale=AUXILIARY_WEIGHT,
        return_loss_components=True,
        return_multiscale_proposal_diagnostics=True,
    )
    components = result.get("loss_component_tensors")
    if not isinstance(components, dict):
        raise RuntimeError("Model did not return loss-component tensors")
    if float(result.get("multiscale_loss_scale", -1.0)) != AUXILIARY_WEIGHT:
        raise RuntimeError("Model did not exercise full auxiliary weight")
    if not bool(torch.isfinite(result["loss_total"])):
        raise FloatingPointError(f"{scene_id}/{epoch}: non-finite total loss")

    level_records: dict[str, Any] = {}
    hard_invariants: dict[str, bool] = {
        "manifest_hash_exact": manifest_hash_exact,
        "full_scene_point_count_exact": int(source.points.shape[0])
        == int(visit["num_points"]),
        "canonical_target_cell_exact": canonical_schedule_exact,
        "observed_target_cell_exact": observed_cell == target_cell,
        "auxiliary_scale_exact": float(result["multiscale_loss_scale"])
        == AUXILIARY_WEIGHT,
        "total_loss_finite": bool(torch.isfinite(result["loss_total"])),
    }
    for level in LEVEL_NAMES:
        metrics = result.get("multiscale", {}).get(level)
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Missing multi-scale metrics for {level}")
        level_record, level_checks = _level_record(
            level,
            metrics,
            proposals_per_forward=int(sampling.proposals_per_forward),
        )
        level_records[level] = level_record
        hard_invariants.update(
            {f"{level}_{name}": value for name, value in level_checks.items()}
        )

    gradient_sites, gradient_checks = _gradient_site_report(
        torch,
        model,
        components,
    )
    hard_invariants.update(gradient_checks)
    torch.cuda.synchronize(device)
    memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    record = {
        **dict(visit),
        "observed_cell": observed_cell,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "augmentation_seed": visit_input.augmentation_seed,
        "forward_seed": int(forward_seed),
        "batch_sha256": batch_sha256,
        "batch_metadata": dict(batch_metadata),
        "loss_total": float(result["loss_total"].detach()),
        "loss_final_dec0": float(result["loss_final_dec0"]),
        "loss_multiscale_unscaled": float(result["loss_multiscale"]),
        "levels": level_records,
        "gradient_sites": gradient_sites,
        "hard_invariants": hard_invariants,
        "memory": memory,
    }
    del (
        result,
        components,
        points,
        features,
        batch,
        visit_input,
        source,
    )
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _reset_branch_rng(torch: Any, np: Any, seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _rng_state_fingerprint(torch: Any, np: Any) -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(numpy_state[1].tobytes(order="C"))
    digest.update(str((numpy_state[2], numpy_state[3], numpy_state[4])).encode("ascii"))
    digest.update(torch.get_rng_state().detach().cpu().numpy().tobytes(order="C"))
    for state in torch.cuda.get_rng_state_all():
        digest.update(state.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_state_fingerprint(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _gradient_norm(torch: Any, parameters: Sequence[Any]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return math.sqrt(max(0.0, float(total)))


def _clip_gradient_record(
    torch: Any,
    parameters: Sequence[Any],
) -> dict[str, Any]:
    pre_clip_norm = _gradient_norm(torch, parameters)
    clip_return_norm = float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=GRAD_CLIP_NORM)
    )
    post_clip_norm = _gradient_norm(torch, parameters)
    clip_coefficient = min(
        1.0,
        GRAD_CLIP_NORM / (clip_return_norm + 1.0e-6),
    )
    return {
        "pre_clip_norm": pre_clip_norm,
        "post_clip_norm": post_clip_norm,
        "clip_return_norm": clip_return_norm,
        "clip_coefficient": clip_coefficient,
        "was_clipped": clip_return_norm > GRAD_CLIP_NORM,
    }


def _nested_optimizer_state_finite(
    torch: Any,
    value: Any,
) -> tuple[bool, int]:
    if torch.is_tensor(value):
        finite = (
            bool(torch.isfinite(value).all())
            if value.is_floating_point() or value.is_complex()
            else True
        )
        return finite, 1
    if isinstance(value, Mapping):
        results = [
            _nested_optimizer_state_finite(torch, nested)
            for nested in value.values()
        ]
    elif isinstance(value, (list, tuple)):
        results = [
            _nested_optimizer_state_finite(torch, nested) for nested in value
        ]
    elif isinstance(value, float):
        return math.isfinite(value), 0
    else:
        return True, 0
    return all(result[0] for result in results), sum(result[1] for result in results)


def _all_parameters_finite(torch: Any, model: Any) -> bool:
    return all(
        bool(torch.isfinite(parameter.detach()).all())
        for parameter in model.parameters()
    )


def _optimizer_contract_report(
    optimizer: Any,
    group_membership: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        groups.append(
            {
                "group_name": str(group["group_name"]),
                "lr": float(group["lr"]),
                "initial_lr": float(group["initial_lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": len(parameters),
                "parameters": sum(int(parameter.numel()) for parameter in parameters),
            }
        )
    return {
        "class": "AdamW",
        "backbone_peak_lr": BACKBONE_PEAK_LR,
        "head_peak_lr": HEAD_PEAK_LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "initial_state_empty": len(optimizer.state) == 0,
        "parameter_group_membership_sha256": canonical_json_sha256(
            {name: list(values) for name, values in group_membership.items()}
        ),
        "parameter_groups": groups,
    }


def _run_paired_optimizer_branch(
    *,
    torch: Any,
    np: Any,
    dist: Any,
    distributed_data_parallel: Any,
    branch: str,
    auxiliary_multiplier: float,
    rank: int,
    local_rank: int,
    world_size: int,
    device: Any,
    schedule: Sequence[Mapping[str, Any]],
    source_root: Path,
    initial_checkpoint: Path,
    initial_checkpoint_sha256: str,
    litept_root: Path,
    sampling: Any,
    litept_contrastive_model: Any,
    strict_initial_load: Any,
    optimizer_parameter_groups: Any,
    load_scene_source: Any,
    build_scene_visit_input: Any,
    training_batch: Any,
    canonical_source_cell_for_scene_epoch: Any,
    train_augmentation_profile: str,
    color_mean: Sequence[float],
    color_std: Sequence[float],
    canonical_stable_seed: Any,
) -> dict[str, Any]:
    if branch not in PAIRED_BRANCH_MULTIPLIERS:
        raise ValueError(f"Unknown paired optimizer branch: {branch}")
    if float(auxiliary_multiplier) != PAIRED_BRANCH_MULTIPLIERS[branch]:
        raise ValueError(f"Paired optimizer multiplier drift for branch {branch}")
    dist.barrier()
    _reset_branch_rng(torch, np, FROZEN_SEED)
    model = litept_contrastive_model(
        litept_root=litept_root,
        input_features="rgbn6",
        multiscale_supervision=True,
        multiscale_loss_weight=AUXILIARY_WEIGHT,
        multiscale_warmup_epochs=AUXILIARY_WARMUP_EPOCHS,
    ).to(device)
    initialization = strict_initial_load(
        model=model,
        path=initial_checkpoint,
        input_features="rgbn6",
        multiscale_supervision=True,
    )
    if (
        initialization.get("strict") is not True
        or initialization.get("sha256") != initial_checkpoint_sha256
        or int(initialization.get("seed", -1)) != FROZEN_SEED
    ):
        raise RuntimeError(f"Branch {branch} did not strict-load frozen initialization")
    initial_model_state_sha256 = _model_state_fingerprint(torch, model)
    optimizer_groups, group_membership = optimizer_parameter_groups(
        model,
        backbone_lr=BACKBONE_PEAK_LR,
        head_lr=HEAD_PEAK_LR,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    optimizer_contract = _optimizer_contract_report(optimizer, group_membership)
    if not _optimizer_group_contract_exact(optimizer_contract):
        raise RuntimeError("Production AdamW parameter-group contract drift")
    ddp_model = distributed_data_parallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=False,
    )
    model.train()
    rng_state_before_first_visit_sha256 = _rng_state_fingerprint(torch, np)
    local_schedule = [row for row in schedule if int(row["rank"]) == rank]
    updates: list[dict[str, Any]] = []
    for global_update in range(EXPECTED_GLOBAL_UPDATES):
        update_visits = [
            row
            for row in local_schedule
            if int(row["global_update"]) == global_update
        ]
        if len(update_visits) != SCENES_PER_RANK_PER_UPDATE:
            raise RuntimeError(
                f"Branch {branch} rank {rank} update {global_update} has "
                f"{len(update_visits)} visits"
            )
        optimizer.zero_grad(set_to_none=True)
        visit_records: list[dict[str, Any]] = []
        losses: list[float] = []
        for microbatch_index, visit in enumerate(update_visits):
            scene_id = str(visit["scene_id"])
            epoch = int(visit["epoch"])
            target_cell = str(visit["target_cell"])
            input_seed = canonical_stable_seed(
                FROZEN_SEED,
                rank,
                str(visit["visit_id"]),
                "paired-optimizer-input",
            )
            _reset_branch_rng(torch, np, int(input_seed))
            manifest_path = (source_root / scene_id / "source_manifest.json").resolve(
                strict=True
            )
            manifest_sha256 = sha256_file(manifest_path)
            if manifest_sha256 != visit["manifest_sha256"]:
                raise RuntimeError(f"{scene_id}: source-manifest hash drift")
            source = load_scene_source(
                manifest_path,
                input_features="rgbn6",
                color_mean=color_mean,
                color_std=color_std,
                verify_hashes=True,
            )
            if (
                source.crop_mode != "full_scene"
                or int(source.points.shape[0]) != int(visit["num_points"])
            ):
                raise RuntimeError(f"{scene_id}: full-scene source contract drift")
            canonical_cell = canonical_source_cell_for_scene_epoch(
                scene_id,
                epoch,
                cells=SOURCE_CELLS,
                schedule="balanced_random_permutation_v1",
                seed=FROZEN_SEED,
            )
            if canonical_cell != target_cell:
                raise RuntimeError(f"{scene_id}/{epoch}: target-cell schedule drift")
            visit_input = build_scene_visit_input(
                scene=source,
                epoch=epoch,
                global_seed=FROZEN_SEED,
                input_features="rgbn6",
                augmentation_profile=train_augmentation_profile,
                color_mean=color_mean,
                color_std=color_std,
            )
            points = torch.from_numpy(visit_input.points).to(device)
            features = torch.from_numpy(visit_input.features).to(device)
            observed_cell, batch, batch_metadata = training_batch(
                scene=source,
                epoch=epoch,
                points=points,
                device=device,
                sampling=sampling,
                return_metadata=True,
                cells=SOURCE_CELLS,
                cell_schedule="balanced_random_permutation_v1",
                schedule_seed=FROZEN_SEED,
            )
            if observed_cell != target_cell:
                raise RuntimeError(f"{scene_id}/{epoch}: sampled source cell drift")
            batch_sha256 = _tensor_fingerprint(torch, batch, batch_metadata)
            forward_seed = canonical_stable_seed(
                FROZEN_SEED,
                rank,
                scene_id,
                epoch,
                "train-forward",
            )
            _reset_branch_rng(torch, np, int(forward_seed))
            sync_context = (
                nullcontext()
                if microbatch_index + 1 == SCENES_PER_RANK_PER_UPDATE
                else ddp_model.no_sync()
            )
            with sync_context:
                result = ddp_model(
                    points,
                    features,
                    batch,
                    multiscale_loss_scale=float(auxiliary_multiplier),
                )
                if (
                    float(result.get("multiscale_loss_scale", -1.0))
                    != float(auxiliary_multiplier)
                ):
                    raise RuntimeError(
                        f"Branch {branch} did not apply its auxiliary multiplier"
                    )
                loss = result["loss_total"]
                local_finite = torch.tensor(
                    int(bool(torch.isfinite(loss))),
                    dtype=torch.int32,
                    device=device,
                )
                dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
                if int(local_finite) != 1:
                    raise FloatingPointError(
                        f"Branch {branch} update {global_update} has non-finite loss"
                    )
                (loss / SCENES_PER_RANK_PER_UPDATE).backward()
            loss_value = float(loss.detach())
            input_sha256 = canonical_json_sha256(
                {
                    "visit_id": str(visit["visit_id"]),
                    "rank": rank,
                    "global_update": global_update,
                    "manifest_sha256": manifest_sha256,
                    "augmentation_seed": visit_input.augmentation_seed,
                    "forward_seed": int(forward_seed),
                    "batch_sha256": batch_sha256,
                    "observed_cell": observed_cell,
                }
            )
            visit_records.append(
                {
                    "visit_id": str(visit["visit_id"]),
                    "input_sha256": input_sha256,
                    "batch_sha256": batch_sha256,
                    "loss": loss_value,
                    "loss_finite": math.isfinite(loss_value),
                }
            )
            losses.append(loss_value)
            del result, loss, points, features, batch, visit_input, source

        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient = _clip_gradient_record(torch, parameters)
        gradient_finite = torch.tensor(
            int(_clip_record_exact(gradient)),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(gradient_finite, op=dist.ReduceOp.MIN)
        if int(gradient_finite) != 1:
            raise FloatingPointError(
                f"Branch {branch} update {global_update} gradient/clip gate failed"
            )
        optimizer.step()
        optimizer_state_finite, optimizer_state_tensor_count = (
            _nested_optimizer_state_finite(torch, optimizer.state)
        )
        parameters_finite = _all_parameters_finite(torch, model)
        state_finite = torch.tensor(
            int(optimizer_state_finite and parameters_finite),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(state_finite, op=dist.ReduceOp.MIN)
        if int(state_finite) != 1:
            raise FloatingPointError(
                f"Branch {branch} update {global_update} optimizer state is non-finite"
            )
        local_checksum = local_optimizer_update_checksum(
            rank=rank,
            global_update=global_update,
            visits=visit_records,
        )
        gathered_rank_inputs: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(
            gathered_rank_inputs,
            {
                "rank": rank,
                "local_update_input_sha256": local_checksum,
            },
        )
        if any(value is None for value in gathered_rank_inputs):
            raise RuntimeError("DDP update-input checksum gather was incomplete")
        rank_inputs = [
            value for value in gathered_rank_inputs if value is not None
        ]
        if {int(value["rank"]) for value in rank_inputs} != set(
            range(EXPECTED_WORLD_SIZE)
        ):
            raise RuntimeError("DDP update-input checksum ranks drifted")
        global_checksum = global_optimizer_update_checksum(
            global_update=global_update,
            rank_inputs=rank_inputs,
        )
        updates.append(
            {
                "global_update": global_update,
                "rank": rank,
                "visits": visit_records,
                "losses": losses,
                "all_losses_finite": all(math.isfinite(value) for value in losses),
                "gradient": gradient,
                "optimizer_state_finite": optimizer_state_finite,
                "optimizer_state_tensor_count": optimizer_state_tensor_count,
                "parameters_finite_after_step": parameters_finite,
                "local_update_input_sha256": local_checksum,
                "global_update_input_sha256": global_checksum,
            }
        )
        del parameters
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "branch": branch,
        "auxiliary_multiplier": float(auxiliary_multiplier),
        "rank": rank,
        "world_size": world_size,
        "ddp_wrapped": True,
        "rng_reset_seed": FROZEN_SEED,
        "rng_state_before_first_visit_sha256": (
            rng_state_before_first_visit_sha256
        ),
        "strict_initialization_passed": True,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_model_state_sha256": initial_model_state_sha256,
        "optimizer": optimizer_contract,
        "optimizer_steps": len(updates),
        "updates": updates,
        "short_loss_interpretation": "mechanical_finiteness_only",
    }
    del ddp_model, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--pretraining-contract", type=Path, required=True)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    return parser.parse_args(argv)


def run_distributed(args: argparse.Namespace) -> int:
    """Execute the complete gradient panel and atomically emit rank-0 JSON."""

    if int(args.seed) != FROZEN_SEED:
        raise ValueError(
            f"The frozen calibration panel requires seed {FROZEN_SEED}, got {args.seed}"
        )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration report: {args.output}")

    # This must be set before CUDA libraries initialize.  Unsupported
    # nondeterministic kernels subsequently fail under deterministic-algorithm
    # mode instead of silently weakening the contract.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from delimit3d.training.contracts import (
        SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
        source_cell_for_scene_epoch,
        stable_seed as canonical_stable_seed,
    )
    from delimit3d.pretraining.api import (
        COMMON_RGB_COLOR_MEAN,
        COMMON_RGB_COLOR_STD,
        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE,
        LitePTContrastiveModel,
        build_scene_visit_input,
        load_scene_source,
        optimizer_parameter_groups,
        sampling_config_for_profile,
        training_batch,
    )
    from delimit3d.pretraining.checkpoint import (
        _strict_initial_load,
    )

    # Freeze the sealed release panel independently on every rank. Historical
    # panels keep their original report and are never relabeled as this release.
    global FROZEN_SOURCE_AUDIT_SHA256, FROZEN_POINT_QUARTILES, FROZEN_PANEL_SCENES
    global FROZEN_PANEL_SHA256, FROZEN_SCHEDULE_SHA256, EXPECTED_OPTIMIZATION_SCENES
    audit = json.loads(args.source_audit.read_text())
    rows = validate_source_audit(audit, require_frozen_contract=False)
    EXPECTED_OPTIMIZATION_SCENES = len(rows)
    FROZEN_SOURCE_AUDIT_SHA256 = sha256_file(args.source_audit)
    FROZEN_POINT_QUARTILES = point_quartile_summary(_point_rank_partition(rows))
    panel = build_calibration_panel(audit, seed=FROZEN_SEED, require_frozen_contract=False)
    FROZEN_PANEL_SCENES = tuple((q, cell, half, tuple(str(row["scene_id"]) for row in panel
        if row["point_quartile"] == q and row["target_cell"] == cell and row["density_half"] == half))
        for q in range(1,5) for cell in SOURCE_CELLS for half in ("low", "high"))
    FROZEN_PANEL_SHA256 = canonical_json_sha256(panel)
    FROZEN_SCHEDULE_SHA256 = canonical_json_sha256(build_target_schedule(panel, seed=FROZEN_SEED))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"Calibration requires exactly two ranks, got WORLD_SIZE={world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "Calibration requires exactly two visible CUDA devices per torchrun job"
        )
    if local_rank not in (0, 1):
        raise RuntimeError(f"Unexpected LOCAL_RANK={local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl")
    started = time.perf_counter()
    gathered: list[dict[str, Any] | None] = [None] * world_size
    local_payload: dict[str, Any]
    try:
        trust_report = validate_trust_artifacts(
            code_manifest_path=args.code_manifest,
            pretraining_contract_path=args.pretraining_contract,
            initial_checkpoint_path=args.initial_checkpoint,
            source_audit_path=args.source_audit,
        )
        audit_path = args.source_audit.resolve(strict=True)
        source_root = args.source_root.resolve(strict=True)
        source_audit_sha256 = sha256_file(audit_path)
        if source_audit_sha256 != FROZEN_SOURCE_AUDIT_SHA256:
            raise RuntimeError(
                "Frozen source-audit digest drift: "
                f"{source_audit_sha256} != {FROZEN_SOURCE_AUDIT_SHA256}"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        panel = build_calibration_panel(
            audit,
            seed=FROZEN_SEED,
            require_frozen_contract=True,
        )
        schedule = build_target_schedule(panel, seed=FROZEN_SEED)
        if len(schedule) != EXPECTED_SCHEDULE_SIZE:
            raise AssertionError("Frozen target schedule must contain 96 visits")
        schedule_sha256 = canonical_json_sha256(schedule)
        if schedule_sha256 != FROZEN_SCHEDULE_SHA256:
            raise RuntimeError(
                "Frozen target-schedule digest drift: "
                f"{schedule_sha256} != {FROZEN_SCHEDULE_SHA256}"
            )
        canonical_schedule_verified = all(
            source_cell_for_scene_epoch(
                str(row["scene_id"]),
                int(row["epoch"]),
                cells=SOURCE_CELLS,
                schedule=SOURCE_CELL_SCHEDULE_BALANCED_RANDOM,
                seed=FROZEN_SEED,
            )
            == row["target_cell"]
            for row in schedule
        )
        if not canonical_schedule_verified:
            raise RuntimeError("Local and canonical target schedules disagree")

        random.seed(FROZEN_SEED)
        np.random.seed(FROZEN_SEED)
        torch.manual_seed(FROZEN_SEED)
        torch.cuda.manual_seed_all(FROZEN_SEED)
        model = LitePTContrastiveModel(
            litept_root=args.litept_root.resolve(strict=True),
            input_features="rgbn6",
            multiscale_supervision=True,
            multiscale_loss_weight=AUXILIARY_WEIGHT,
            multiscale_warmup_epochs=AUXILIARY_WARMUP_EPOCHS,
        ).to(device)
        initial_checkpoint = args.initial_checkpoint.resolve(strict=True)
        initial_sha256_before_load = sha256_file(initial_checkpoint)
        if initial_sha256_before_load != trust_report["initial_checkpoint_sha256"]:
            raise RuntimeError("Initial checkpoint changed after trust validation")
        initial_report = _strict_initial_load(
            model=model,
            path=initial_checkpoint,
            input_features="rgbn6",
            multiscale_supervision=True,
        )
        if initial_report.get("strict") is not True:
            raise RuntimeError("Initialization did not strict-load")
        if initial_report.get("sha256") != initial_sha256_before_load:
            raise RuntimeError("Initialization changed during strict loading")
        if int(initial_report.get("seed", -1)) != FROZEN_SEED:
            raise RuntimeError("Initialization seed drift")
        model.train()
        frozen_buffers = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }
        sampling = sampling_config_for_profile("pf_hnm_512_p16_v1")
        if int(sampling.proposals_per_forward) != 16:
            raise RuntimeError("Calibration requires the frozen p16 sampling profile")
        local_schedule = [row for row in schedule if int(row["rank"]) == rank]
        if len(local_schedule) != EXPECTED_SCHEDULE_SIZE // world_size:
            raise RuntimeError(
                f"Rank {rank} visit count drift: {len(local_schedule)} != 48"
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        records: list[dict[str, Any]] = []
        for visit in local_schedule:
            records.append(
                _run_visit(
                    torch=torch,
                    np=np,
                    visit=visit,
                    rank=rank,
                    device=device,
                    source_root=source_root,
                    model=model,
                    frozen_buffers=frozen_buffers,
                    sampling=sampling,
                    load_scene_source=load_scene_source,
                    build_scene_visit_input=build_scene_visit_input,
                    training_batch=training_batch,
                    canonical_source_cell_for_scene_epoch=(
                        source_cell_for_scene_epoch
                    ),
                    train_augmentation_profile=(
                        TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE
                    ),
                    color_mean=COMMON_RGB_COLOR_MEAN,
                    color_std=COMMON_RGB_COLOR_STD,
                    canonical_stable_seed=canonical_stable_seed,
                )
            )
        del model, frozen_buffers
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        paired_optimizer_branches: dict[str, Any] = {}
        for branch, auxiliary_multiplier in PAIRED_BRANCH_MULTIPLIERS.items():
            paired_optimizer_branches[branch] = _run_paired_optimizer_branch(
                torch=torch,
                np=np,
                dist=dist,
                distributed_data_parallel=DistributedDataParallel,
                branch=branch,
                auxiliary_multiplier=auxiliary_multiplier,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
                schedule=schedule,
                source_root=source_root,
                initial_checkpoint=initial_checkpoint,
                initial_checkpoint_sha256=initial_sha256_before_load,
                litept_root=args.litept_root.resolve(strict=True),
                sampling=sampling,
                litept_contrastive_model=LitePTContrastiveModel,
                strict_initial_load=_strict_initial_load,
                optimizer_parameter_groups=optimizer_parameter_groups,
                load_scene_source=load_scene_source,
                build_scene_visit_input=build_scene_visit_input,
                training_batch=training_batch,
                canonical_source_cell_for_scene_epoch=(
                    source_cell_for_scene_epoch
                ),
                train_augmentation_profile=(
                    TRAIN_AUGMENTATION_MASK3D_RGBN6_FULLSCENE
                ),
                color_mean=COMMON_RGB_COLOR_MEAN,
                color_std=COMMON_RGB_COLOR_STD,
                canonical_stable_seed=canonical_stable_seed,
            )
        torch.cuda.synchronize(device)
        dist.barrier()
        local_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "device": torch.cuda.get_device_name(device),
            "status": "completed",
            "panel_sha256": canonical_json_sha256(panel),
            "schedule_sha256": schedule_sha256,
            "source_audit_sha256": source_audit_sha256,
            "trust_report": trust_report,
            "canonical_schedule_verified": canonical_schedule_verified,
            "initialization": initial_report,
            "records": records,
            "paired_optimizer_branches": paired_optimizer_branches,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # synchronize failures into one atomic report
        local_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "trust_report": locals().get("trust_report"),
            "elapsed_seconds": time.perf_counter() - started,
        }

    try:
        dist.all_gather_object(gathered, local_payload)
        exit_code = 2
        if rank == 0:
            failures = [
                payload
                for payload in gathered
                if payload is None or payload.get("status") != "completed"
            ]
            if failures:
                report: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "measurement_completed": False,
                    "code_manifest_sha256": _sha256_if_file(args.code_manifest),
                    "pretraining_contract_sha256": _sha256_if_file(
                        args.pretraining_contract
                    ),
                    "initial_checkpoint_sha256": _sha256_if_file(
                        args.initial_checkpoint
                    ),
                    "source_audit_sha256": _sha256_if_file(args.source_audit),
                    "gradient_calibration_gate_passed": False,
                    "paired_optimizer_gate": {
                        "completed": False,
                        "passed": False,
                        "status": "failed_closed",
                        "required_before_336_epoch_launch": True,
                    },
                    "training_authorized": False,
                    "launch_gate_passed": False,
                    "passed": False,
                    "ranks": gathered,
                }
                exit_code = 2
            else:
                assert all(payload is not None for payload in gathered)
                completed = [payload for payload in gathered if payload is not None]
                panel_hashes = {str(payload["panel_sha256"]) for payload in completed}
                schedule_hashes = {
                    str(payload["schedule_sha256"]) for payload in completed
                }
                initialization_hashes = {
                    str(payload["initialization"]["sha256"])
                    for payload in completed
                }
                trust_reports = [payload["trust_report"] for payload in completed]
                records = [
                    record
                    for payload in completed
                    for record in payload["records"]
                ]
                # Re-read on rank 0 so the emitted panel/schedule are exactly the
                # objects whose digests were independently reproduced by both ranks.
                audit = json.loads(
                    args.source_audit.resolve(strict=True).read_text(encoding="utf-8")
                )
                source_audit_sha256 = sha256_file(
                    args.source_audit.resolve(strict=True)
                )
                if source_audit_sha256 != FROZEN_SOURCE_AUDIT_SHA256:
                    raise RuntimeError(
                        "Source audit changed before report publication"
                    )
                panel = build_calibration_panel(
                    audit,
                    seed=FROZEN_SEED,
                    require_frozen_contract=True,
                )
                schedule = build_target_schedule(panel, seed=FROZEN_SEED)
                if canonical_json_sha256(schedule) != FROZEN_SCHEDULE_SHA256:
                    raise RuntimeError(
                        "Target schedule changed before report publication"
                    )
                publication_trust = validate_trust_artifacts(
                    code_manifest_path=args.code_manifest,
                    pretraining_contract_path=args.pretraining_contract,
                    initial_checkpoint_path=args.initial_checkpoint,
                    source_audit_path=args.source_audit,
                )
                paired_rank_reports = {
                    branch: [
                        payload["paired_optimizer_branches"][branch]
                        for payload in completed
                    ]
                    for branch in PAIRED_BRANCH_MULTIPLIERS
                }
                paired_gate = summarize_paired_optimizer_branches(
                    paired_rank_reports,
                    schedule,
                )
                summary = summarize_calibration_records(
                    records,
                    schedule,
                    paired_optimizer_gate=paired_gate,
                )
                rank_contract_checks = {
                    "both_ranks_reproduced_same_panel": len(panel_hashes) == 1,
                    "both_ranks_reproduced_same_schedule": len(schedule_hashes) == 1,
                    "both_ranks_strict_loaded_same_initialization": len(
                        initialization_hashes
                    )
                    == 1,
                    "both_ranks_verified_canonical_schedule": all(
                        bool(payload["canonical_schedule_verified"])
                        for payload in completed
                    ),
                    "both_ranks_verified_same_trust_root": all(
                        trust_report == publication_trust
                        for trust_report in trust_reports
                    ),
                }
                gradient_gate = bool(
                    summary["gradient_calibration_gate_passed"]
                    and all(rank_contract_checks.values())
                )
                training_authorized = bool(
                    gradient_gate
                    and paired_gate.get("completed") is True
                    and paired_gate.get("passed") is True
                )
                summary["gradient_calibration_gate_passed"] = gradient_gate
                summary["training_authorized"] = training_authorized
                summary["launch_gate_passed"] = training_authorized
                summary["passed"] = training_authorized
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "measurement_mode": (
                        "frozen_gradient_panel_and_paired_ddp_optimizer_gate"
                    ),
                    "measurement_completed": True,
                    "code_manifest_sha256": publication_trust[
                        "code_manifest_sha256"
                    ],
                    "pretraining_contract_sha256": publication_trust[
                        "pretraining_contract_sha256"
                    ],
                    "initial_checkpoint_sha256": publication_trust[
                        "initial_checkpoint_sha256"
                    ],
                    "source_audit_sha256": publication_trust[
                        "source_audit_sha256"
                    ],
                    "dataset": "structured3d",
                    "input_features": "rgbn6",
                    "crop_mode": "full_scene",
                    "source_family": "2d",
                    "source_cells": list(SOURCE_CELLS),
                    "seed": FROZEN_SEED,
                    "world_size": EXPECTED_WORLD_SIZE,
                    "sampling_profile": "pf_hnm_512_p16_v1",
                    "panel_contract": {
                        "optimization_scenes": EXPECTED_OPTIMIZATION_SCENES,
                        "point_quartiles": FROZEN_POINT_QUARTILES,
                        "density_partition": (
                            "within_quartile_and_cell_empirical_rank_halves"
                        ),
                        "samples_per_stratum": SAMPLES_PER_STRATUM,
                        "panel_rows": EXPECTED_PANEL_SIZE,
                        "panel_sha256": canonical_json_sha256(panel),
                        "panel": panel,
                    },
                    "schedule_contract": {
                        "cycle_starts": list(FULL_WEIGHT_CYCLE_STARTS),
                        "visits": EXPECTED_SCHEDULE_SIZE,
                        "global_updates_represented": EXPECTED_GLOBAL_UPDATES,
                        "scenes_per_rank_per_update": (
                            SCENES_PER_RANK_PER_UPDATE
                        ),
                        "schedule_sha256": canonical_json_sha256(schedule),
                        "schedule": schedule,
                    },
                    "objective_contract": {
                        "final_dec0_weight": 1.0,
                        "auxiliary_stage_weights": dict(LEVEL_WEIGHTS),
                        "auxiliary_weight": AUXILIARY_WEIGHT,
                        "auxiliary_warmup_epochs": AUXILIARY_WARMUP_EPOCHS,
                        "primary_direction_factor": (
                            "dot(g_dec0, g_dec0 + 0.4*h_aux) / "
                            "squared_norm(g_dec0)"
                        ),
                    },
                    "determinism_contract": {
                        "torch_deterministic_algorithms": True,
                        "cudnn_deterministic": True,
                        "cudnn_benchmark": False,
                        "cublas_workspace_config": os.environ.get(
                            "CUBLAS_WORKSPACE_CONFIG"
                        ),
                        "per_visit_batch_sha256": True,
                        "model_buffers_restored_before_each_visit": True,
                        "paired_branches_reconstructed_independently": True,
                        "paired_visit_and_update_checksums": True,
                    },
                    "optimizer_contract": {
                        "branches": dict(PAIRED_BRANCH_MULTIPLIERS),
                        "optimizer": "AdamW",
                        "backbone_peak_lr": BACKBONE_PEAK_LR,
                        "head_peak_lr": HEAD_PEAK_LR,
                        "weight_decay": WEIGHT_DECAY,
                        "gradient_clip_norm": GRAD_CLIP_NORM,
                        "global_updates_per_branch": EXPECTED_GLOBAL_UPDATES,
                        "scenes_per_rank_per_update": (
                            SCENES_PER_RANK_PER_UPDATE
                        ),
                    },
                    "rank_contract_checks": rank_contract_checks,
                    "summary": summary,
                    "gradient_calibration_gate_passed": gradient_gate,
                    "paired_optimizer_gate": paired_gate,
                    "training_authorized": training_authorized,
                    "launch_gate_passed": training_authorized,
                    "passed": training_authorized,
                    "scientific_limitations": [
                        (
                            "Direction factors are per-visit frozen-initialization "
                            "probes, not DDP-accumulated optimizer-update factors."
                        ),
                        (
                            "The eight-step C/M losses are recorded only for finite "
                            "mechanical validation and are not representation-quality "
                            "evidence."
                        ),
                        (
                            "The separate two-largest-scene memory and "
                            "rank-divergent zero-valid DDP sentinels remain "
                            "outside this runner."
                        ),
                    ],
                    "records": sorted(records, key=lambda row: str(row["visit_id"])),
                    "ranks": [
                        {key: value for key, value in payload.items() if key != "records"}
                        for payload in completed
                    ],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                exit_code = 0 if training_authorized else 2
            _atomic_json(args.output, report)

        exit_tensor = torch.tensor(exit_code, dtype=torch.int32, device=device)
        dist.broadcast(exit_tensor, src=0)
        return int(exit_tensor)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def validate_training_calibration(path, args):
    """Require a complete mechanical gate for these exact prepared inputs."""
    path = Path(path).resolve(strict=True)
    report = json.loads(path.read_text())
    paired = report.get("paired_optimizer_gate", {})
    if (report.get("schema_version") not in {SCHEMA_VERSION,
            "structured3d_rgbn6_fullscene_multiscale_scientific_calibration/v1"}
            or not report.get("measurement_completed")
            or report.get("passed") is not True or report.get("training_authorized") is not True
            or paired.get("completed") is not True or paired.get("passed") is not True):
        raise ValueError("Multiscale pretraining requires a completed gradient and paired-optimizer calibration")
    for key, current in (("initial_checkpoint", args.initial_checkpoint), ("source_audit", args.source_audit),
                         ("code_manifest", args.code_manifest)):
        if current is None or sha256_file(Path(current)) != report.get(key + "_sha256"):
            raise ValueError(f"Calibration {key} differs from this run")
    objective = report.get("objective_contract", {})
    optimizer = report.get("optimizer_contract", {})
    expected = {"backbone_peak_lr": args.backbone_lr, "head_peak_lr": args.head_lr,
                "weight_decay": args.weight_decay, "gradient_clip_norm": args.grad_clip_norm,
                "global_updates_per_branch": 8, "scenes_per_rank_per_update": args.scenes_per_rank_per_update}
    if (report.get("seed") != args.seed or report.get("world_size") != args.expected_world_size
            or objective.get("auxiliary_weight") != args.multiscale_loss_weight
            or objective.get("auxiliary_warmup_epochs") != args.multiscale_warmup_epochs
            or any(optimizer.get(key) != value for key, value in expected.items())):
        raise ValueError("Calibration objective/optimizer differs from this recipe")
    return {"path": str(path), "sha256": sha256_file(path), "schema_version": report["schema_version"], "passed": True}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(run_distributed(args))


if __name__ == "__main__":
    main()
