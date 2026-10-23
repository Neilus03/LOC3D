"""Pretraining dataset extracted from the executed trainer."""

from __future__ import annotations


import json


from concurrent.futures import ThreadPoolExecutor


from pathlib import Path


from typing import Any


import torch


from delimit3d.training.contracts import EXPERIMENT_ID, OFFICIAL_TRAIN_SCENES, audit_scene_source_manifest, load_scene_ids, sha256_file

from delimit3d.training.adaptation import load_scene_source, scene_source_for_visit

from delimit3d.data.contrastive_sampler_v2 import FrameProposalCatalog


def _preload_sources(
    *,
    sam2_maximum_raw_points: int = 500_000,
    scene_ids: list[str],
    source_root: Path,
    smoke_manifest: Path | None,
    expected_manifest_hashes: dict[str, str] | None,
    input_features: str,
    coordinate_normalization_policy: str = "native",
    color_mean: tuple[float, ...] | None = None,
    color_std: tuple[float, ...] | None = None,
    verify_hashes: bool = True,
    evaluation_visit_seed: int | None = None,
) -> list[Any]:
    scenes = []
    for scene_id in scene_ids:
        manifest_path = (
            smoke_manifest
            if smoke_manifest is not None
            else source_root / scene_id / "source_manifest.json"
        )
        scene = load_scene_source(
            manifest_path,
            sam2_maximum_raw_points=sam2_maximum_raw_points,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            color_mean=color_mean,
            color_std=color_std,
            verify_hashes=verify_hashes,
        )
        if smoke_manifest is None and scene.scene_id != scene_id:
            raise ValueError(
                f"Source scene {scene.scene_id} != split scene {scene_id}"
            )
        if expected_manifest_hashes is not None:
            expected_hash = expected_manifest_hashes.get(scene.scene_id)
            if expected_hash is None:
                raise ValueError(
                    f"{scene.scene_id}: absent from frozen dataset source audit"
                )
            if scene.manifest_sha256 != expected_hash:
                raise ValueError(
                    f"{scene.scene_id}: source manifest differs from frozen "
                    "dataset source audit"
                )
        if evaluation_visit_seed is not None:
            scene = scene_source_for_visit(scene, epoch=0, global_seed=evaluation_visit_seed)
        scenes.append(scene)
    return scenes


def _audit_lazy_sources(
    *,
    scene_ids: list[str],
    source_root: Path,
    expected_manifest_hashes: dict[str, str] | None,
) -> list[Any]:
    """Validate lazy source identity and structure without materializing arrays."""

    audits = []
    for scene_id in scene_ids:
        audit = audit_scene_source_manifest(
            source_root / scene_id / "source_manifest.json",
            expected_scene_id=scene_id,
            verify_hashes=False,
        )
        if expected_manifest_hashes is not None:
            expected_hash = expected_manifest_hashes.get(scene_id)
            if expected_hash is None:
                raise ValueError(
                    f"{scene_id}: absent from frozen dataset source audit"
                )
            if audit.manifest_sha256 != expected_hash:
                raise ValueError(
                    f"{scene_id}: source manifest differs from frozen "
                    "dataset source audit"
                )
        audits.append(audit)
    return audits


def _load_logical_scene_group(
    *,
    sam2_maximum_raw_points: int = 500_000,
    scene_indices: list[int],
    scene_ids: list[str],
    source_root: Path,
    expected_manifest_hashes: dict[str, str] | None,
    input_features: str,
    coordinate_normalization_policy: str,
    executor: ThreadPoolExecutor,
    color_mean: tuple[float, ...] | None = None,
    color_std: tuple[float, ...] | None = None,
) -> list[Any]:
    """Load one ordered logical scene group from the trusted local archive."""

    futures = [
        executor.submit(
            _preload_sources,
            sam2_maximum_raw_points=sam2_maximum_raw_points,
            scene_ids=[scene_ids[scene_index]],
            source_root=source_root,
            smoke_manifest=None,
            expected_manifest_hashes=expected_manifest_hashes,
            input_features=input_features,
            coordinate_normalization_policy=coordinate_normalization_policy,
            color_mean=color_mean,
            color_std=color_std,
            # The complete archive and every artifact were already verified by
            # the pack audit and extraction gate. Rehashing arrays on all 336
            # visits would change I/O cost, not integrity.
            verify_hashes=False,
        )
        for scene_index in scene_indices
    ]
    return [future.result()[0] for future in futures]


def _load_smoke_manifest_paths(path: Path) -> list[Path]:
    paths = [
        Path(line.strip()).resolve(strict=True)
        for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(paths) < 2:
        raise ValueError(
            "--smoke-manifest-list must contain at least two scene manifests"
        )
    if len(set(paths)) != len(paths):
        raise ValueError("--smoke-manifest-list contains duplicate paths")
    return paths


def _load_dataset_source_audit(
    *,
    path: Path,
    dataset: str,
    source_root: Path,
    train_split: Path,
    holdout_split: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    path = path.resolve(strict=True)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "structured3d":
        if (
            audit.get("schema_version")
            not in {"structured3d_fullscene_source_audit/v1", "structured3d_sam2_fullscene_source_audit/v1", "delimit3d_2d_source_audit/v1"}
            or audit.get("experiment_id") != EXPERIMENT_ID
            or audit.get("dataset") != "structured3d"
            or audit.get("crop_mode") not in ({"full_scene", "sphere_50k"}
                if audit.get("schema_version") == "delimit3d_2d_source_audit/v1" else {"full_scene"})
            or audit.get("artifact_hashes_verified") is not True
            or not audit.get("passed")
            or not all(audit.get("checks", {}).values())
        ):
            raise RuntimeError(
                f"Structured3D full-scene source audit failed or drifted: {path}"
            )
        if audit["schema_version"] == "structured3d_sam2_fullscene_source_audit/v1":
            record = audit["dataset_manifest"]
            dataset_path = Path(record["path"]).resolve(strict=True)
            if (sha256_file(dataset_path) != record["sha256"]
                    or audit.get("source_cells") != ["2d/sam2"]
                    or Path(audit["source_root"]).resolve(strict=True) != source_root.resolve(strict=True)):
                raise RuntimeError("SAM2 dataset seal or source identity drift")
        release_source = audit["schema_version"] == "delimit3d_2d_source_audit/v1"
        train_count = len(load_scene_ids(train_split)) if release_source else 2930
        val_count = len(load_scene_ids(holdout_split)) if release_source else 326
        split_contract = (("optimization", train_split, train_count),
                          ("validation", holdout_split, val_count))
        for split_name, current_path, expected_count in split_contract:
            record = audit.get("splits", {}).get(split_name, {})
            resolved_current = current_path.resolve(strict=True)
            if int(record.get("scene_count", -1)) != expected_count:
                raise RuntimeError(
                    f"Structured3D {split_name} scene-count drift in source audit"
                )
            if sha256_file(resolved_current) != record.get("sha256"):
                raise RuntimeError(
                    f"Structured3D {split_name} split hash differs from source audit"
                )
        rows = audit.get("scenes", [])
        expected_hashes = {
            str(row["scene_id"]): str(row["manifest_sha256"]) for row in rows
        }
        if (
            int(audit.get("scene_count", -1)) != train_count + val_count
            or int(audit.get("train_scene_count", -1)) != train_count
            or int(audit.get("validation_scene_count", -1)) != val_count
            or len(rows) != train_count + val_count
            or len(expected_hashes) != train_count + val_count
        ):
            raise RuntimeError(
                f"Structured3D source-audit scene map drift: {path}"
            )
        expected_ids = set(load_scene_ids(train_split)) | set(
            load_scene_ids(holdout_split)
        )
        if set(expected_hashes) != expected_ids:
            raise RuntimeError(
                f"Structured3D source-audit membership differs from current splits: {path}"
            )
        report = {
            "path": str(path),
            "sha256": sha256_file(path),
            "schema_version": audit["schema_version"],
            "scene_count": int(audit["scene_count"]),
            "source_root": str(source_root.resolve(strict=True)),
            "audited_source_root": str(audit["source_root"]),
            "crop_mode": audit["crop_mode"],
            "artifact_hashes_verified": True,
            "point_count_summary": audit.get("point_count_summary"),
            "total_artifact_bytes": int(audit["total_artifact_bytes"]),
            "passed": True,
        }
        return report, expected_hashes
    if dataset != "scannet":
        raise ValueError(f"Unsupported source-audit dataset: {dataset!r}")
    if (
        audit.get("schema_version")
        != "litept_gt3_unsam3_dataset_source_audit/v1"
        or audit.get("experiment_id") != EXPERIMENT_ID
        or not audit.get("passed")
        or not all(audit.get("checks", {}).values())
    ):
        raise RuntimeError(f"Dataset source audit failed or drifted: {path}")
    if int(audit.get("scene_count", -1)) != OFFICIAL_TRAIN_SCENES:
        raise RuntimeError(f"Dataset source audit scene-count drift: {path}")
    if Path(audit["source_root"]).resolve(strict=True) != source_root.resolve(
        strict=True
    ):
        raise RuntimeError(f"Dataset source audit root drift: {path}")
    dataset_record = audit.get("dataset_manifest", {})
    dataset_manifest_path = Path(dataset_record["path"]).resolve(strict=True)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    if dataset_manifest_sha256 != dataset_record["sha256"]:
        raise RuntimeError(f"Dataset manifest hash drift from source audit: {path}")
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    if (
        dataset_manifest.get("schema_version")
        != "litept_gt3_unsam3_dataset/v1"
        or dataset_manifest.get("experiment_id") != EXPERIMENT_ID
        or not all(dataset_manifest.get("checks", {}).values())
    ):
        raise RuntimeError(f"Dataset manifest failed or drifted: {dataset_manifest_path}")
    split_contract = (
        ("optimization", train_split),
        ("upstream_holdout", holdout_split),
    )
    for split_name, current_path in split_contract:
        record = dataset_manifest["splits"][split_name]
        resolved_current = current_path.resolve(strict=True)
        if Path(record["path"]).resolve(strict=True) != resolved_current:
            raise RuntimeError(f"{split_name} path differs from dataset manifest")
        current_hash = sha256_file(resolved_current)
        if (
            current_hash != record["file_sha256"]
            or current_hash != record["ordered_scene_digest"]
        ):
            raise RuntimeError(f"{split_name} hash differs from dataset manifest")
    rows = audit.get("scenes", [])
    expected_hashes = {
        str(row["scene_id"]): str(row["manifest_sha256"]) for row in rows
    }
    if (
        len(rows) != OFFICIAL_TRAIN_SCENES
        or len(expected_hashes) != OFFICIAL_TRAIN_SCENES
    ):
        raise RuntimeError(f"Dataset source audit scene map drift: {path}")
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": audit["schema_version"],
        "scene_count": int(audit["scene_count"]),
        "source_root": str(source_root.resolve(strict=True)),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "passed": True,
    }
    return report, expected_hashes


def _v2_frame_catalogs(
    *,
    scene: Any,
    device: torch.device,
    cells: tuple[str, ...],
) -> dict[str, tuple[FrameProposalCatalog, ...]]:
    """Move one scene's immutable frame-local proposal catalogs to a rank device."""

    output: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in cells:
        if not cell.startswith("2d/"):
            raise ValueError(f"V2 multigranular sampling is 2D-only, got {cell!r}")
        granularity = cell.split("/", 1)[1]
        output[cell] = tuple(
            FrameProposalCatalog(
                cell=cell,
                frame_id=frame.physical_frame_id,
                visible_indices=torch.as_tensor(
                    frame.visible_indices,
                    dtype=torch.long,
                    device=device,
                ),
                proposal_offsets=torch.as_tensor(
                    frame.proposal_offsets,
                    dtype=torch.long,
                    device=device,
                ),
                proposal_point_indices=torch.as_tensor(
                    frame.proposal_point_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
            for frame in scene.frames(granularity, "train")
        )
    return output
