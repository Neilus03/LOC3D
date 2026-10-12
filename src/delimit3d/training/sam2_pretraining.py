"""Fail-closed preparation and verification for the annotation-to-B* handoff."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from delimit3d.data.sam2_source import (
    SOURCE_CELL, MAX_VISIT_POINTS, FRAGMENT_POLICY, artifact_path, audit_sam2_source,
)
from delimit3d.training.contracts import EXPERIMENT_ID, sha256_file
from delimit3d.training.sam2_sampling import (
    COVERAGE_POLICY, sampling_visit_for_scene, validate_saved_coverage_states,
)
from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule


def write_json(path: Path, value: dict) -> None:
    from delimit3d.cli.finalize_structured3d_sam2 import _atomic_json
    _atomic_json(path, value)


def write_frozen_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError(f"Refusing to alter frozen handoff input {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def validate_dataset_seal(annotation_root: Path) -> dict:
    dataset_path = annotation_root / "manifests/dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text())
    if (
        dataset.get("schema_version") != "delimit3d_structured3d_sam2_dataset/v2"
        or dataset.get("complete") is not True
        or dataset.get("cache_hashes_verified") is not True
        or dataset.get("source_cells") != [SOURCE_CELL]
        or dataset.get("teacher_source_key") != SOURCE_CELL
        or dataset.get("full_scene") is not True
        or dataset.get("single_granularity") is not True
        or dataset.get("scene_count") != 3256
        or dataset.get("physical_frames") != 78009
        or dataset.get("automatic_mask_generator", {}).get("points_per_side") != 32
        or dataset.get("teacher_checkpoint_sha256") != "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        or Path(dataset["pack_root"]).resolve() != (annotation_root / "packs").resolve()
    ):
        raise RuntimeError("Full SAM2 dataset seal is incomplete or differs from the agreed teacher")
    scene_list = dataset["scene_ids_file"]
    if sha256_file(Path(scene_list["path"])) != scene_list["sha256"]:
        raise RuntimeError("Annotation scene-list hash drift")
    rows = dataset["scenes"]
    expected_ids = set(Path(scene_list["path"]).read_text().splitlines())
    if len(rows) != 3256 or {row["scene_id"] for row in rows} != expected_ids:
        raise RuntimeError("Annotation seal scene membership drift")
    for row in rows:
        manifest = annotation_root / "packs" / row["scene_id"] / "source_manifest.json"
        if sha256_file(manifest) != row["manifest_sha256"]:
            raise RuntimeError(f"{manifest}: manifest changed after annotation validation")
    return dataset


def verify_snapshot(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "structured3d_sam2_bstar_code_manifest/v1" or payload.get("passed") is not True:
        raise RuntimeError("Invalid training snapshot manifest")
    root = Path(payload["snapshot_root"])
    for row in payload["files"]:
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Training snapshot drift: {row['path']}")
    for row in payload.get("runtime_artifacts", []):
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"Training runtime drift: {row['path']}")
    return payload


def seal_snapshot(run_root: Path, *, code_root: Path | None = None) -> dict:
    import importlib.util
    root = code_root or run_root / "code"
    path = root / "code_manifest.json"
    files = []
    for candidate in sorted(root.rglob("*")):
        if (not candidate.is_file() or candidate == path
                or candidate.suffix not in (".py", ".json", ".sh", ".md")
                or any(part in ("__pycache__", ".pytest_cache") for part in candidate.parts)):
            continue
        files.append({"path": str(candidate.relative_to(root)), "sha256": sha256_file(candidate)})
    runtime = []
    for name in ("pointrope", "torch.version"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Required runtime dependency absent: {name}")
        native = Path(spec.origin).resolve(strict=True)
        runtime.append({"name": name, "path": str(native), "sha256": sha256_file(native)})
    result = {"schema_version": "structured3d_sam2_bstar_code_manifest/v1", "passed": True,
              "snapshot_root": str(root), "files": files, "runtime_artifacts": runtime,
              "litept_upstream_commit": "055c3fa4285a4feef679cffa69c9542e8b09e922",
              "source_checkout_has_uncommitted_changes": True,
              "integrity_identity": "exact_content_hashes_not_git_commit_alone"}
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError("Refusing to change an already sealed training snapshot")
    write_json(path, result)
    verify_snapshot(path)
    return {"passed": True, "file_count": len(files), "sha256": sha256_file(path)}


def probe_completed_packs(annotation_root: Path, run_root: Path, *, code_root: Path | None = None) -> dict:
    """Read-only real-pack loader/sampler/architecture check; no GPU allocation."""
    import torch
    from delimit3d.training.adaptation import (
        LitePTContrastiveModel, load_scene_source, sampling_config_for_profile,
        scene_source_for_visit, STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    )
    rows = []
    manifests = []
    for path in (annotation_root / "packs").glob("*/source_manifest.json"):
        value = json.loads(path.read_text())
        if value["counts"]["trainable_proposals"][SOURCE_CELL] >= 16 and value["counts"]["heldout_trainable_proposals"][SOURCE_CELL] >= 4:
            manifests.append((int(value["crop"]["num_points"]), path))
    if not manifests:
        raise RuntimeError("No completed packs available for CPU integration check")
    ordered = sorted(manifests)
    selected = [ordered[0], ordered[-1]]
    sampling = sampling_config_for_profile("pf_hnm_512_v1")
    for _, path in selected:
        full = load_scene_source(path, coordinate_normalization_policy=STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY)
        visit = scene_source_for_visit(full, epoch=0, global_seed=42)
        admitted, _ = admit_sam2_holdout([visit], sampling=sampling, seed=42, grid_size=0.02,
                                       report_path=run_root / "reports" / (full.scene_id + "_cpu_admission.json"))
        rows.append({"scene_id": full.scene_id, "full_scene_points": len(full.points),
                     "visit_points": len(visit.points), "fragment_id": visit.crop_id,
                     "frame_count": len(visit.raw_frames), "safe_heldout_plan": len(admitted) == 1,
                     "source_cells": [SOURCE_CELL], "unsam_label_arrays": len(visit.labels_by_key)})
    model = LitePTContrastiveModel(litept_root=(code_root or run_root / "code") / "deps/LitePT", litept_variant="litept_b_star",
                                  input_features="rgbn6", grid_size=0.02, token_pure_dec0=True,
                                  token_pure_cells=(SOURCE_CELL,)).cpu()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 53_189_973 or torch.cuda.is_initialized():
        raise RuntimeError("B* parameter contract drift or unexpected CUDA allocation during CPU check")
    report = {"passed": True, "scope": "real_pack_CPU_integration_not_GPU_training",
              "parameter_count": parameter_count, "gpu_allocated": False, "scenes": rows}
    write_json(run_root / "reports/CPU_REAL_PACK_CHECK.json", report)
    return report


def admit_sam2_holdout(scenes, *, sampling, seed, grid_size, report_path,
                       allow_shared_proposal_negatives=False):
    """Record inadmissible scenes, never relax token purity or safe negatives."""
    import torch
    from delimit3d.training.adaptation import _token_pure_fixed_holdout_plan

    admitted, plans, rows = [], {}, []
    for scene in scenes:
        try:
            plan = _token_pure_fixed_holdout_plan(
                scene=scene, points=torch.from_numpy(scene.points), device=torch.device("cpu"),
                sampling=sampling, seed=seed, cells=(SOURCE_CELL,), grid_size=grid_size,
                allow_shared_proposal_negatives=allow_shared_proposal_negatives,
            )
        except ValueError as error:
            # Only documented data-quality exhaustion is an admissible
            # exclusion. Corrupt caches/relations and programming errors stop.
            message = str(error)
            if "V2 fixed-eval catalog lacks a granularity" not in message and "V2 fixed-eval queue exhausted" not in message:
                raise
            rows.append({"scene_id": scene.scene_id, "manifest_sha256": scene.manifest_sha256,
                         "fragment_id": scene.crop_id, "admitted": False, "reason": message})
            continue
        admitted.append(scene)
        plans[scene.scene_id] = plan
        rows.append({"scene_id": scene.scene_id, "manifest_sha256": scene.manifest_sha256,
                     "fragment_id": scene.crop_id, "admitted": True})
    if not admitted:
        raise RuntimeError("No relation-safe SAM2 validation scenes; pretraining is not ready")
    report = {"schema_version": "structured3d_sam2_fixed_eval_admission/v1", "passed": True,
              "requested_scene_count": len(scenes), "admitted_scene_count": len(admitted),
              "excluded_scene_count": len(scenes) - len(admitted), "source_cells": [SOURCE_CELL],
              "policy": "fixed_epoch0_geometry_fragment_unknown_safe_token_pure",
              "downstream_transfer_evidence": False, "scenes": rows}
    if allow_shared_proposal_negatives:
        report["allow_shared_proposal_negatives"] = True
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        raise RuntimeError("Frozen SAM2 validation admission changed")
    write_json(report_path, report)
    return admitted, plans


def execution_provenance_path(run_root: Path, code_root: Path) -> Path:
    return run_root / "ready" / f"execution_provenance_{sha256_file(code_root / 'code_manifest.json')[:16]}.json"


def verify_artifact_unchanged_since_audit(manifest_path: Path, record: dict, *, audited_at_ns: int) -> None:
    artifact = artifact_path(manifest_path, record)
    metadata = artifact.stat()
    if (metadata.st_size != int(record["size_bytes"])
            or metadata.st_mtime_ns > audited_at_ns or metadata.st_ctime_ns > audited_at_ns):
        raise RuntimeError(f"{artifact}: changed since the pinned full hash audit; a fresh audit is required")


def prepare_handoff(annotation_root: Path, run_root: Path, *, code_root: Path | None = None,
                    reuse_source_audit_sha256: str | None = None) -> dict:
    dataset = validate_dataset_seal(annotation_root)
    code_manifest = (code_root or run_root / "code") / "code_manifest.json"
    verify_snapshot(code_manifest)
    shard_path = annotation_root / "manifests/shard_manifest.json"
    shard = json.loads(shard_path.read_text())
    by_id = {row["scene_id"]: row for row in shard["scenes"]}
    if len(by_id) != 3256 or set(by_id) != {row["scene_id"] for row in dataset["scenes"]}:
        raise RuntimeError("Frozen train/validation partition differs from annotation seal")
    train = sorted(scene_id for scene_id, row in by_id.items() if row["split"] == "train")
    validation = sorted(scene_id for scene_id, row in by_id.items() if row["split"] == "validation")
    if len(train) != 2930 or len(validation) != 326 or set(train) & set(validation):
        raise RuntimeError("Expected 2930 training and 326 disjoint validation scenes")
    ready = run_root / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    train_path, validation_path = ready / "train.txt", ready / "validation.txt"
    write_frozen_text(train_path, "\n".join(train) + "\n")
    write_frozen_text(validation_path, "\n".join(validation) + "\n")
    audit_path = ready / "source_audit.json"
    audited_at_ns = None
    if reuse_source_audit_sha256 is not None:
        if sha256_file(audit_path) != reuse_source_audit_sha256:
            raise RuntimeError("Reused source audit differs from its explicit launch trust root")
        previous_audit = json.loads(audit_path.read_text())
        if (previous_audit.get("passed") is not True
                or previous_audit.get("artifact_hashes_verified") is not True
                or previous_audit.get("checks", {}).get("all_artifact_hashes_verified") is not True):
            raise RuntimeError("Only a successful full artifact hash audit may be reused")
        audited_at_ns = audit_path.stat().st_mtime_ns
    scene_rows, total_bytes = [], 0
    for index, row in enumerate(dataset["scenes"]):
        path = annotation_root / "packs" / row["scene_id"] / "source_manifest.json"
        audit = audit_sam2_source(path, verify_hashes=reuse_source_audit_sha256 is None)
        manifest = json.loads(path.read_text())
        if audited_at_ns is not None:
            for artifact_record in manifest["artifacts"].values():
                verify_artifact_unchanged_since_audit(path, artifact_record, audited_at_ns=audited_at_ns)
        if audit.manifest_sha256 != row["manifest_sha256"] or audit.crop_points != int(by_id[audit.scene_id]["point_count"]):
            raise RuntimeError(f"{path}: geometry or seal drift")
        total_bytes += sum(int(record["size_bytes"]) for record in manifest["artifacts"].values())
        scene_rows.append({"scene_id": audit.scene_id, "manifest_sha256": audit.manifest_sha256,
                           "point_count": audit.crop_points, "split": by_id[audit.scene_id]["split"]})
        if index % 100 == 0:
            print(f"Verified SAM2 pack {index + 1}/3256", flush=True)
    dataset_path = annotation_root / "manifests/dataset_manifest.json"
    source_audit = {
        "schema_version": "structured3d_sam2_fullscene_source_audit/v1", "passed": True,
        "experiment_id": EXPERIMENT_ID, "dataset": "structured3d", "crop_mode": "full_scene",
        "source_cells": [SOURCE_CELL], "artifact_hashes_verified": True,
        "source_root": str(annotation_root / "packs"), "scene_count": 3256,
        "train_scene_count": 2930, "validation_scene_count": 326,
        "total_artifact_bytes": total_bytes,
        "checks": {"full_dataset_sealed": True, "teacher_frozen": True, "geometry_frozen": True,
                   "scene_splits_disjoint": True, "all_artifact_hashes_verified": True},
        "dataset_manifest": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "splits": {"optimization": {"path": str(train_path), "sha256": sha256_file(train_path), "scene_count": 2930},
                   "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path), "scene_count": 326}},
        "shard_manifest_sha256": sha256_file(shard_path), "scenes": scene_rows,
    }
    if audit_path.exists() and json.loads(audit_path.read_text()) != source_audit:
        raise RuntimeError("Frozen SAM2 source audit changed")
    if not audit_path.exists():
        write_json(audit_path, source_audit)
    # Seven scenes exercise full six-visit updates plus a one-visit checkpoint
    # tail, including near-ceiling whole geometry and oversized fragments.
    # Selection uses geometry/admission, never validation model performance.
    candidates = [row for row in dataset["scenes"] if row["scene_id"] in set(train)
                  and int(row["trainable_proposals"]) >= 16 and int(row["heldout_trainable_proposals"]) >= 4]
    smallest = sorted(candidates, key=lambda row: (int(row["point_count"]), row["scene_id"]))[:32]
    oversized = [row for row in candidates if int(row["point_count"]) > MAX_VISIT_POINTS]
    near_ceiling = sorted((row for row in candidates if 350000 <= int(row["point_count"]) <= MAX_VISIT_POINTS),
                          key=lambda row: -int(row["point_count"]))[:16]
    if len(smallest) < 5 or not oversized or not near_ceiling:
        raise RuntimeError("Cannot construct the required full-scene/fragment integration smoke")
    safe_small = select_smoke_scenes(smallest, count=5, annotation_root=annotation_root)
    safe_whole = select_smoke_scenes(near_ceiling, count=1, annotation_root=annotation_root)
    safe_large = select_smoke_scenes(sorted(oversized, key=lambda row: -int(row["point_count"]))[:16],
                                     count=1, annotation_root=annotation_root)
    smoke_ids = safe_small + safe_whole + safe_large
    write_frozen_text(ready / "smoke_manifests.txt", "\n".join(
        str(annotation_root / "packs" / scene_id / "source_manifest.json") for scene_id in smoke_ids) + "\n")
    preset = json.loads((code_manifest.parent / "configs/training/litept_bstar_sam2_fullscene.json").read_text())
    checkpoint_epochs = [int(e) for e in preset["checkpoint_epochs"].split(",")]
    if (preset["epochs"] != 150 or checkpoint_epochs != [0, 25, 50, 75, 100, 125, 150]
            or preset.get("flush_logical_batch_at_eval") is not True
            or preset["evaluation_epochs"] != preset["checkpoint_epochs"]):
        raise RuntimeError("Preset differs from the approved 150-epoch exact-checkpoint protocol")
    schedule = LogicalBatchSchedule(1465, 6, 150, tuple(checkpoint_epochs[1:]))
    provenance = {
        "schema_version": "structured3d_sam2_bstar_execution_provenance/v1", "passed": True,
        "trust_root": "annotation_success_seal_then_full_hash_audit_then_two_gpu_training_smoke",
        "annotation_root": str(annotation_root), "source_cells": [SOURCE_CELL],
        "dataset_manifest": source_audit["dataset_manifest"],
        "source_audit_sha256": sha256_file(audit_path), "code_manifest_sha256": sha256_file(code_manifest),
        "litept_variant": "litept_b_star", "precision": "fp32", "grid_size": 0.02,
        "maximum_visit_points": MAX_VISIT_POINTS, "fragment_policy": FRAGMENT_POLICY,
        "fragment_catalog_coverage": COVERAGE_POLICY,
        "artifact_verification": {"mode": ("pinned_full_hash_audit_with_identical_seal_and_unchanged_artifact_metadata"
                                               if reuse_source_audit_sha256 is not None else "fresh_full_hash_audit"),
                                  "reused_source_audit_sha256": reuse_source_audit_sha256},
        "epochs": 150, "checkpoint_epochs": checkpoint_epochs,
        "logical_batch_schedule": schedule.as_dict(), "optimizer_updates": schedule.total_updates,
        "world_size": 2, "logical_scenes_per_global_update": 12,
        "train_scene_count": 2930, "validation_scene_count": 326, "smoke_scene_ids": smoke_ids,
    }
    provenance_path = execution_provenance_path(run_root, code_manifest.parent)
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise RuntimeError("Frozen execution provenance changed")
    write_json(provenance_path, provenance)
    write_json(ready / "READY.json", {"passed": True, "scope": "CPU_data_ready_not_GPU_smoke_passed",
                                     "execution_provenance_sha256": sha256_file(provenance_path)})
    return provenance


def select_smoke_scenes(candidates, *, count: int, annotation_root: Path) -> list[str]:
    """Admit the actual two augmented training visits before spending GPU time."""
    from delimit3d.data.contrastive_sampler_v2 import NoRetainedFrameGroupsError
    from delimit3d.training.adaptation import (
        load_scene_source, scene_source_for_visit,
        STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
    )
    chosen = []
    for row in candidates:
        full = load_scene_source(annotation_root / "packs" / row["scene_id"] / "source_manifest.json",
                                 coordinate_normalization_policy=STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
                                 verify_hashes=False, color_mean=(0, 0, 0), color_std=(1, 1, 1))
        safe = True
        coverage_states = {}
        for epoch in (0, 1):
            scene = scene_source_for_visit(full, epoch=epoch, global_seed=42)
            try:
                plan, _ = sample_cpu_training_visit(scene, epoch=epoch, coverage_states=coverage_states)
            except NoRetainedFrameGroupsError:
                safe = False
                break
            realized = plan.coverage_by_cell[SOURCE_CELL]["contrastive_retained_count"]
            if realized < 16:
                safe = False
                break
        if safe:
            chosen.append(row["scene_id"])
            print(f"Admitted smoke scene {row['scene_id']} ({row['point_count']} full-scene points)", flush=True)
            if len(chosen) == count:
                return chosen
    raise RuntimeError(f"Only {len(chosen)}/{count} candidate scenes have safe two-epoch p16 SAM2 supervision")


def sample_cpu_training_visit(scene, *, epoch: int, coverage_states: dict,
                              allow_shared_proposal_negatives=False):
    """The production augmentation/quota/admission path, without a GPU model."""
    import torch
    from delimit3d.data.contrastive_sampler_v2 import FrameProposalCatalog, sample_multigranular_frame_group_plan
    from delimit3d.training.adaptation import build_scene_visit_input

    visit = build_scene_visit_input(scene=scene, epoch=epoch, global_seed=42, input_features="rgbn6",
                                   augmentation_profile="mask3d_scannet_rgbn6_fullscene_v1",
                                   color_mean=(0, 0, 0), color_std=(1, 1, 1))
    catalogs = tuple(FrameProposalCatalog(SOURCE_CELL, frame.physical_frame_id,
                                          torch.from_numpy(frame.visible_indices),
                                          torch.from_numpy(frame.proposal_offsets),
                                          torch.from_numpy(frame.proposal_point_indices))
                     for frame in scene.frames("sam2", "train"))
    sampling_visit = sampling_visit_for_scene(scene, global_epoch=epoch, seed=42,
                                              cells=(SOURCE_CELL,), saved_states=coverage_states)
    plan = sample_multigranular_frame_group_plan(
        scene_id=scene.scene_id, epoch=sampling_visit.epoch, seed=42, points=torch.from_numpy(visit.points),
        frames_by_cell={SOURCE_CELL: catalogs}, cells=(SOURCE_CELL,), total_proposal_quota=16,
        positive_pairs_per_proposal=64, num_uniform_negatives=256, num_spatial_hard_negatives=128,
        spatial_candidate_pool=1024, num_feature_hard_negatives=0, feature_candidate_pool=1024,
        require_negative_proposal_membership=True, require_hierarchy_routes=False, grid_size=0.02,
        allow_shared_proposal_negatives=allow_shared_proposal_negatives,
        operation=sampling_visit.operation, coverage_states=sampling_visit.states_by_cell,
    )
    for cell, record in plan.state_after_by_cell().items():
        coverage_states[sampling_visit.keys_by_cell[cell]] = record
    return plan, sampling_visit


def probe_sampling_visits(annotation_root: Path, run_root: Path) -> dict:
    """Check the failed whole-scene epoch and two real fragment revisits/replays."""
    import torch
    from delimit3d.training.adaptation import load_scene_source, scene_source_for_visit, STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY

    rows = []
    for scene_id in ("scene_00183", "scene_02051"):
        full = load_scene_source(annotation_root / "packs" / scene_id / "source_manifest.json",
                                 coordinate_normalization_policy=STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY,
                                 verify_hashes=True, color_mean=(0, 0, 0), color_std=(1, 1, 1))
        first = scene_source_for_visit(full, epoch=0, global_seed=42)
        fragment_count = int(first.crop_id.split("_of_")[1].split("_")[0]) if first.crop_id else 1
        epochs = (0, 1) if fragment_count == 1 else (0, 1, fragment_count, fragment_count + 1)
        states, visits = {}, []
        for epoch in epochs:
            scene = scene_source_for_visit(full, epoch=epoch, global_seed=42)
            before = json.loads(json.dumps(states))
            plan, sampling_visit = sample_cpu_training_visit(scene, epoch=epoch, coverage_states=states)
            replay_states = json.loads(json.dumps(before))
            replay, _ = sample_cpu_training_visit(scene, epoch=epoch, coverage_states=replay_states)
            if states != replay_states or plan.audit_metadata() != replay.audit_metadata():
                raise RuntimeError("SAM2 sampler restart replay changed")
            validate_saved_coverage_states(states, start_epoch=epoch + 1, seed=42)
            retained = plan.coverage_by_cell[SOURCE_CELL]["contrastive_retained_count"]
            if retained != 16:
                raise RuntimeError("Real SAM2 recovery visit lacks p16 safe supervision")
            key = sampling_visit.keys_by_cell[SOURCE_CELL]
            if before.get(key) is not None and states[key]["cursor"] <= before[key]["cursor"]:
                raise RuntimeError("SAM2 coverage cursor did not advance")
            visits.append({"global_epoch": epoch, "catalog_epoch": sampling_visit.epoch,
                           "fragment_id": scene.crop_id, "visit_points": len(scene.points),
                           "retained_proposals": retained, "coverage_key": key,
                           "cursor_before": before.get(key, {}).get("cursor", 0),
                           "cursor_after": states[key]["cursor"], "restart_replay_identical": True})
            print(f"Verified recovery visit {scene_id} global={epoch} catalog={sampling_visit.epoch}", flush=True)
        rows.append({"scene_id": scene_id, "full_scene_points": len(full.points),
                     "fragment_count": fragment_count, "visits": visits})
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA allocation during CPU sampler recovery check")
    report = {"passed": True, "scope": "real_pack_CPU_sampling_recovery_not_GPU_training",
              "fragment_catalog_coverage": COVERAGE_POLICY, "gpu_allocated": False, "scenes": rows}
    write_json(run_root / "reports/CPU_SAMPLING_RECOVERY_CHECK.json", report)
    return report


def verify_smoke(root: Path) -> dict:
    import torch
    summary = json.loads((root / "run_summary.json").read_text())
    config = json.loads((root / "resolved_config.json").read_text())
    if (summary.get("mechanics_passed") is not True or not all(summary["checks"].values())
            or summary.get("epochs") != 2 or summary.get("updates") != 4
            or config.get("world_size") != 2 or config.get("source_cells") != [SOURCE_CELL]
            or config.get("litept_variant") != "litept_b_star"
            or config.get("supervised_stages") != ["dec0"] or config.get("grid_size") != 0.02):
        raise RuntimeError("SAM2 B* two-GPU smoke failed its frozen training contract")
    initial = torch.load(root / "checkpoints/epoch_0000.pt", map_location="cpu", weights_only=False)
    resumed_from = torch.load(root / "checkpoints/epoch_0001.pt", map_location="cpu", weights_only=False)
    final = torch.load(root / "checkpoints/epoch_0002.pt", map_location="cpu", weights_only=False)
    states = final["model_state_dict"]
    if (final.get("resumable") is not True or final.get("update") != 4
            or len(final.get("rank_rng_states", [])) != 2
            or len(final.get("rank_coverage_states", [])) != 2
            or not final.get("optimizer_state_dict", {}).get("state")):
        raise RuntimeError("Smoke lacks complete optimizer/DDP replay state")
    if not all(torch.isfinite(value).all() for value in states.values()):
        raise RuntimeError("Non-finite smoke parameters")
    changed = [key for key, value in states.items() if key in initial["model_state_dict"]
               and not torch.equal(value, initial["model_state_dict"][key])]
    if not any(key.startswith("backbone.") and key.endswith((".weight", ".bias")) for key in changed):
        raise RuntimeError("Smoke did not update the LitePT-B* backbone")
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in rows if row["kind"] == "train"]
    if not train_rows or not all(math.isfinite(row["loss_total"]) for row in train_rows):
        raise RuntimeError("Smoke lacks finite training losses")
    if [row.get("scenes_per_rank_per_update") for row in train_rows] != [6, 1, 6, 1]:
        raise RuntimeError("Smoke did not exercise both full updates and exact short checkpoint tails")
    paused = json.loads((root / "run_paused.json").read_text())
    if (paused.get("passed") is not True or paused.get("epoch") != 1 or paused.get("update") != 2
            or resumed_from.get("epoch") != 1 or resumed_from.get("update") != 2
            or resumed_from.get("resumable") is not True
            or resumed_from["cumulative_health"]["token_delivery"]["microbatch_count"] != 14
            or config.get("resume_parent", {}).get("epoch") != 1
            or config["resume_parent"].get("sha256") != paused.get("checkpoint_sha256")
            or sha256_file(root / "checkpoints/epoch_0001.pt") != paused.get("checkpoint_sha256")
            or final["cumulative_health"]["token_delivery"]["microbatch_count"] != 28):
        raise RuntimeError("Smoke lacks successful partial-batch checkpoint/resume and exact visit accounting")
    report = {"passed": True, "scope": "real_two_gpu_training_mechanics_not_transfer_evidence",
              "world_size": 2, "updates": 4, "epochs": 2, "changed_tensor_count": len(changed),
              "short_batch_resume_verified": True, "global_scene_visits": 28,
              "final_checkpoint_sha256": sha256_file(root / "checkpoints/epoch_0002.pt"),
              "checks": summary["checks"], "source_cells": [SOURCE_CELL]}
    write_json(root / "SMOKE_VERIFIED.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify-code", "seal-code", "probe-packs", "probe-sampling", "verify-smoke", "verify-seal"))
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--reuse-source-audit-sha256")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_handoff(args.annotation_root, args.run_root, code_root=args.code_root,
                                 reuse_source_audit_sha256=args.reuse_source_audit_sha256)
    elif args.action == "verify-code":
        result = verify_snapshot((args.code_root or args.run_root / "code") / "code_manifest.json")
        result = {"passed": result["passed"], "file_count": len(result["files"])}
    elif args.action == "verify-seal":
        result = validate_dataset_seal(args.annotation_root)
        result = {"passed": True, "scene_count": result["scene_count"]}
    elif args.action == "seal-code":
        result = seal_snapshot(args.run_root, code_root=args.code_root)
    elif args.action == "probe-packs":
        result = probe_completed_packs(args.annotation_root, args.run_root, code_root=args.code_root)
    elif args.action == "probe-sampling":
        result = probe_sampling_visits(args.annotation_root, args.run_root)
    else:
        result = verify_smoke(args.run_root / "smoke")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
