import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from delimit3d.data import sam2_source
from delimit3d.data.contrastive_sampler_v2 import FrameProposalCatalog, sample_multigranular_frame_group_plan, source_cell_quotas
from delimit3d.training import adaptation
from delimit3d.training.contracts import SOURCE_CELLS, TWO_D_CELLS, sha256_file, source_cells_for_family
from delimit3d.training.runner import _canonical_source_metadata, _native_token_pure_fixed_pack_cosine_gap, parse_args
from delimit3d.training.sam2_pretraining import admit_sam2_holdout, select_smoke_scenes, validate_dataset_seal, verify_snapshot, verify_artifact_unchanged_since_audit, verify_smoke
from delimit3d.training.sam2_sampling import sampling_visit_for_scene, validate_saved_coverage_states
from test_single_granularity_pretraining import _CPUBackbone, _scene, _plan, REQUIRED

REPO = Path(__file__).resolve().parents[1]


def test_audit_reuse_rejects_artifacts_changed_after_verification(tmp_path):
    artifact = tmp_path / "points.npy"
    artifact.write_bytes(b"verified geometry")
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    record = {"path": str(artifact), "size_bytes": artifact.stat().st_size}
    verify_artifact_unchanged_since_audit(tmp_path / "source_manifest.json", record,
                                        audited_at_ns=audit.stat().st_mtime_ns)
    import os
    os.utime(artifact, ns=(audit.stat().st_mtime_ns + 1, audit.stat().st_mtime_ns + 1))
    with pytest.raises(RuntimeError, match="fresh audit is required"):
        verify_artifact_unchanged_since_audit(tmp_path / "source_manifest.json", record,
                                            audited_at_ns=audit.stat().st_mtime_ns)


def test_audit_reuse_rejects_changed_size_even_with_old_timestamps(tmp_path):
    artifact = tmp_path / "points.npy"
    artifact.write_bytes(b"geometry")
    with pytest.raises(RuntimeError, match="fresh audit is required"):
        verify_artifact_unchanged_since_audit(tmp_path / "source_manifest.json",
            {"path": str(artifact), "size_bytes": artifact.stat().st_size + 1}, audited_at_ns=2**63 - 1)


def test_full_smoke_verifier_accepts_explicit_grid_and_exact_partial_resume(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    initial = {"model_state_dict": {"backbone.layer.weight": torch.ones(2, 2)}}
    final = {"model_state_dict": {"backbone.layer.weight": torch.ones(2, 2) * 2},
             "resumable": True, "update": 4, "rank_rng_states": [{}, {}],
             "rank_coverage_states": [{}, {}], "optimizer_state_dict": {"state": {0: {"step": 4}}},
             "cumulative_health": {"token_delivery": {"microbatch_count": 28}}}
    torch.save(initial, tmp_path / "checkpoints/epoch_0000.pt")
    mid = {**final, "epoch": 1, "update": 2,
           "cumulative_health": {"token_delivery": {"microbatch_count": 14}}}
    torch.save(mid, tmp_path / "checkpoints/epoch_0001.pt")
    torch.save(final, tmp_path / "checkpoints/epoch_0002.pt")
    checkpoint_sha = sha256_file(tmp_path / "checkpoints/epoch_0001.pt")
    summary = {"mechanics_passed": True, "checks": {"exact": True}, "epochs": 2, "updates": 4}
    config = {"world_size": 2, "source_cells": ["2d/sam2"], "litept_variant": "litept_b_star",
              "supervised_stages": ["dec0"], "grid_size": 0.02,
              "resume_parent": {"epoch": 1, "sha256": checkpoint_sha}}
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    (tmp_path / "resolved_config.json").write_text(json.dumps(config))
    (tmp_path / "run_paused.json").write_text(json.dumps({"passed": True, "epoch": 1, "update": 2,
                                                         "checkpoint_sha256": checkpoint_sha}))
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps({"kind": "train", "loss_total": 1,
        "scenes_per_rank_per_update": n}) for n in (6, 1, 6, 1)))
    assert verify_smoke(tmp_path)["short_batch_resume_verified"] is True
    config.pop("grid_size")
    (tmp_path / "resolved_config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="frozen training contract"):
        verify_smoke(tmp_path)


def test_sam2_preset_keeps_legacy_defaults():
    args = parse_args(["--config", str(REPO / "configs/pretrain/sam2_dec0.yaml"), *REQUIRED, "--fixed-eval-pack", "/unused/fixed", "--source-audit", "/unused/audit", "--code-manifest", "/unused/code"])
    assert args.litept_variant == "litept_s_star" and args.required_crop_mode == "full_scene"
    assert args.grid_size == 0.02 and args.source_loading_mode == "logical_batch"
    assert args.epochs == 150 and args.scenes_per_rank_per_update == 6
    assert args.flush_logical_batch_at_eval is True
    assert args.checkpoint_epochs == args.evaluation_epochs == (0, 25, 50, 75, 100, 125, 150)
    assert source_cells_for_family("2d", granularity="sam2") == ("2d/sam2",)
    assert source_cell_quotas(7, cells=("2d/sam2",)) == {"2d/sam2": 16}
    assert source_cells_for_family("2d") == TWO_D_CELLS
    assert source_cells_for_family("2d3d") == SOURCE_CELLS
    assert _canonical_source_metadata(("2d/sam2",), hierarchy_enabled=False) == (("2d/sam2",), ("sam2",))
    with pytest.raises(ValueError):
        _canonical_source_metadata(("2d/sam2",), hierarchy_enabled=True)


def test_fragments_are_deterministic_disjoint_exhaustive_and_bounded():
    points = np.random.default_rng(42).normal(size=(53, 3)).astype(np.float32)
    fragments = sam2_source.geometry_fragments(points, 10)
    assert max(map(len, fragments)) <= 10
    joined = np.concatenate(fragments)
    assert len(joined) == len(np.unique(joined)) == len(points)
    assert np.array_equal(np.sort(joined), np.arange(len(points)))
    assert all(np.array_equal(a, b) for a, b in zip(fragments, sam2_source.geometry_fragments(points, 10)))


def test_fragment_rotation_keeps_logical_scene_and_overlap_membership(monkeypatch):
    scene = _scene("sam2", 128)
    scene = replace(scene, source_variant_id=sam2_source.FRAGMENT_POLICY,
                    colors=np.ones((128, 3)), normals=np.ones((128, 3)), crop_mode="full_scene")
    # Poison/shared proposal must survive remapping, rather than becoming an
    # invented negative against the original 4-point masks.
    frame = scene.raw_frames[0]
    frame = replace(frame, proposal_offsets=np.append(frame.proposal_offsets, 256),
                    proposal_point_indices=np.concatenate((frame.proposal_point_indices, np.arange(128))))
    scene = replace(scene, raw_frames=(frame,))
    monkeypatch.setattr(sam2_source, "MAX_VISIT_POINTS", 40)
    visits = [sam2_source.sam2_scene_for_visit(scene, epoch=e, global_seed=42) for e in range(4)]
    assert len({visit.crop_id for visit in visits}) == 4
    assert all(visit.scene_id == scene.scene_id and len(visit.points) == 32 and visit.crop_mode == "full_scene" for visit in visits)
    assert len(np.unique(np.concatenate([visit.points[:, 0] for visit in visits]))) == 128
    for visit in visits:
        local = visit.raw_frames[0]
        assert np.array_equal(local.proposal_point_indices[local.proposal_offsets[-2]:], np.arange(32))
        assert local.proposal_offsets[-1] == len(local.proposal_point_indices)
        assert set(local.proposal_point_indices) <= set(local.visible_indices)
    assert visits[0].crop_id == sam2_source.sam2_scene_for_visit(scene, epoch=4, global_seed=42).crop_id


def _persistent_visit(scene, epoch, states):
    cell = sam2_source.SOURCE_CELL
    identity = sampling_visit_for_scene(scene, global_epoch=epoch, seed=42,
                                        cells=(cell,), saved_states=states)
    frame = scene.raw_frames[0]
    catalog = FrameProposalCatalog(cell, frame.physical_frame_id, torch.as_tensor(frame.visible_indices),
                                  torch.as_tensor(frame.proposal_offsets), torch.as_tensor(frame.proposal_point_indices))
    plan = sample_multigranular_frame_group_plan(
        scene_id=scene.scene_id, epoch=identity.epoch, seed=42, points=torch.from_numpy(scene.points),
        frames_by_cell={cell: (catalog,)}, cells=(cell,), total_proposal_quota=4,
        positive_pairs_per_proposal=3, num_uniform_negatives=2, num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0, require_negative_proposal_membership=True,
        require_hierarchy_routes=False, grid_size=0.02,
        operation=identity.operation, coverage_states=identity.states_by_cell,
    )
    states[identity.keys_by_cell[cell]] = plan.state_after_by_cell()[cell]
    return plan, identity


def test_whole_scene_persistent_cursor_advances_and_missing_state_still_fails():
    scene, states = _scene("sam2"), {}
    first, _ = _persistent_visit(scene, 0, states)
    previous = json.loads(json.dumps(states))
    second, identity = _persistent_visit(scene, 1, states)
    assert identity.epoch == 1 and list(states) == ["scene_00000|2d/sam2"]
    assert states[list(states)[0]]["cursor"] > previous[list(states)[0]]["cursor"]
    replay, _ = _persistent_visit(scene, 1, previous)
    assert previous == states and replay.audit_metadata() == second.audit_metadata()
    assert first.state_after_by_cell() != second.state_after_by_cell()
    validate_saved_coverage_states(states, start_epoch=2, seed=42)
    with pytest.raises(ValueError, match="persisted coverage state is required"):
        _persistent_visit(scene, 1, {})


def test_fragment_rotation_revisit_and_checkpoint_replay_keep_separate_cursors(monkeypatch):
    full = replace(_scene("sam2"), source_variant_id=sam2_source.FRAGMENT_POLICY,
                   colors=np.ones((128, 3)), normals=np.ones((128, 3)), crop_mode="full_scene")
    monkeypatch.setattr(sam2_source, "MAX_VISIT_POINTS", 40)
    states, boundaries = {}, {}
    for epoch in range(12):
        scene = sam2_source.sam2_scene_for_visit(full, epoch=epoch, global_seed=42)
        before = json.loads(json.dumps(states))
        plan, identity = _persistent_visit(scene, epoch, states)
        assert identity.epoch == epoch // 4 and len(plan.groups) == 1
        key = identity.keys_by_cell[sam2_source.SOURCE_CELL]
        if epoch < 4:
            assert key not in before and identity.epoch == 0
        else:
            assert states[key]["cursor"] > before[key]["cursor"]
        replay, _ = _persistent_visit(scene, epoch, before)
        assert before == states and replay.audit_metadata() == plan.audit_metadata()
        validate_saved_coverage_states(states, start_epoch=epoch + 1, seed=42)
        boundaries[epoch + 1] = json.loads(json.dumps(states))
    assert len(states) == 4
    # Resume part-way through rotation: dormant cursors have older catalog
    # epochs, which must not be mistaken for a broken global epoch boundary.
    validate_saved_coverage_states(boundaries[5], start_epoch=5, seed=42)
    resumed = json.loads(json.dumps(boundaries[5]))
    for epoch in range(5, 12):
        _persistent_visit(sam2_source.sam2_scene_for_visit(full, epoch=epoch, global_seed=42), epoch, resumed)
    assert resumed == states
    revisited = sam2_source.sam2_scene_for_visit(full, epoch=4, global_seed=42)
    with pytest.raises(ValueError, match="persisted coverage state is required"):
        _persistent_visit(revisited, 4, {})
    corrupt = json.loads(json.dumps(boundaries[5]))
    dormant = next(key for key, value in corrupt.items() if value["epoch"] == 0)
    corrupt[dormant]["epoch"] = 1
    with pytest.raises(ValueError, match="resumable epoch boundary"):
        validate_saved_coverage_states(corrupt, start_epoch=5, seed=42)


def test_smoke_scene_admission_carries_epoch_zero_state(tmp_path, monkeypatch):
    full = replace(_scene("sam2"), colors=np.ones((128, 3)), normals=np.ones((128, 3)), crop_mode="full_scene")
    full = replace(full, raw_frames=(replace(full.raw_frames[0], split="train"),))
    monkeypatch.setattr(adaptation, "load_scene_source", lambda *args, **kwargs: full)
    assert select_smoke_scenes([{"scene_id": full.scene_id, "point_count": 128}],
                               count=1, annotation_root=tmp_path) == [full.scene_id]


def _pack(tmp_path):
    count = 12
    points = np.zeros((count, 3), np.float32)
    points[:, 0] = np.arange(count) * 0.06
    arrays = {"points": points, "colors": np.ones((count, 3), np.uint8),
              "normals": np.tile([0, 1, 0], (count, 1)).astype(np.float32),
              "crop_indices": np.arange(count)}
    artifacts = {}
    for key, value in arrays.items():
        path = tmp_path / (key + ".npy")
        np.save(path, value)
        artifacts[key] = {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    cache = tmp_path / "sam2_projection_cache.npz"
    np.savez(cache, schema_version=np.asarray(sam2_source.CACHE_SCHEMA), scene_id=np.asarray("scene_00000"),
             teacher_source_key=np.asarray("2d/sam2"), crop_indices_sha256=np.asarray(artifacts["crop_indices"]["sha256"]),
             physical_frame_ids=np.asarray(["0:0", "0:1"]), split_codes=np.asarray([0, 1]),
             frame_offsets=np.asarray([0, 8, 16]), eligible_point_indices=np.tile(np.arange(8), 2),
             # Visible-but-unmasked points 8..11 must NOT be loaded as known.
             visible_frame_offsets=np.asarray([0, 12, 24]), visible_point_indices=np.tile(np.arange(12), 2),
             proposal_frame_offsets=np.asarray([0, 2, 4]), proposal_point_offsets=np.asarray([0, 5, 10, 15, 20]),
             proposal_point_indices=np.tile(np.concatenate((np.arange(5), np.arange(3, 8))), 2))
    artifacts["sam2_projection_cache"] = {"path": cache.name, "sha256": sha256_file(cache), "size_bytes": cache.stat().st_size}
    manifest = {"schema_version": sam2_source.SCENE_SCHEMA, "scene_id": "scene_00000", "dataset": "structured3d",
                "complete": True, "source_cells": ["2d/sam2"], "teacher_source_key": "2d/sam2", "single_granularity": True,
                "crop": {"mode": "full_scene", "num_points": count}, "checks": {"unknown_safe": True},
                "semantics": {key: True for key in ("frame_local", "overlap_preserved", "unseen_points_are_unknown",
                    "visible_but_unmasked_points_are_unknown", "zero_proposal_frames_contribute_no_supervision")},
                "artifacts": artifacts, "frames": {"physical": ["0:0", "0:1"], "train": ["0:0"], "heldout": ["0:1"]},
                "counts": {"trainable_proposals": {"2d/sam2": 2}, "heldout_trainable_proposals": {"2d/sam2": 2}}}
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_versioned_loader_preserves_unknowns_and_overlaps(tmp_path):
    path = _pack(tmp_path)
    scene = adaptation.load_scene_source(path, coordinate_normalization_policy=adaptation.STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY)
    assert scene.labels_by_key == {} and scene.crop_mode == "full_scene"
    assert len(scene.points) == 12 and scene.features.shape == (12, 6)
    assert np.array_equal(scene.raw_frames[0].visible_indices, np.arange(8))
    assert set(scene.raw_frames[0].proposal_point_indices[:5]) & set(scene.raw_frames[0].proposal_point_indices[5:]) == {3, 4}
    assert len(scene.frames("sam2", "train")) == 1 and len(scene.frames("sam2", "heldout")) == 1
    assert not scene.frames("g05", "train")
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["points"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash drift"):
        adaptation.load_scene_source(path, coordinate_normalization_policy=adaptation.STRUCTURED3D_COORDINATE_NORMALIZATION_POLICY)


def test_sam2_token_pure_loss_backward_fixed_pack_and_admission(tmp_path, monkeypatch):
    monkeypatch.setattr("delimit3d.pretraining.model.LitePTBackbone", _CPUBackbone)
    scene = _scene("sam2")
    plan = _plan(scene, grid_size=0.02)
    model = adaptation.LitePTContrastiveModel(litept_root=tmp_path, token_pure_dec0=True,
                                             token_pure_cells=("2d/sam2",), grid_size=0.02)
    result = model(torch.from_numpy(scene.points), torch.from_numpy(scene.features), frame_groups=plan.groups)
    assert result["token_pure_dec0"]["optimization_valid"]
    result["loss_total"].backward()
    assert model.backbone.projection.weight.grad.norm() > 0
    sampling = adaptation.SamplingConfig(proposals_per_forward=4, positive_pairs_per_proposal=3, uniform_negatives=2,
                                        spatial_hard_negatives=0, feature_hard_negatives=0, feature_candidate_pool=2)
    empty = replace(scene, scene_id="scene_00001", raw_frames=())
    admitted, plans = admit_sam2_holdout([scene, empty], sampling=sampling, seed=42, grid_size=0.02,
                                      report_path=tmp_path / "admission.json")
    assert admitted == [scene] and json.loads((tmp_path / "admission.json").read_text())["excluded_scene_count"] == 1
    pack = tmp_path / "fixed"
    report = adaptation.prepare_fixed_holdout_eval_pack(pack_dir=pack, scenes=admitted, sampling=sampling,
        sampling_profile="test", seed=42, cells=("2d/sam2",), grid_size=0.02, precomputed_token_plans=plans)
    assert report["passed"] and report["token_pure_dec0_fixed_pack"]["source_cells"] == ["2d/sam2"]
    evaluation = adaptation.evaluate_holdout(model=model, scenes=admitted, device=torch.device("cpu"), sampling=sampling,
        seed=42, fixed_eval_pack=pack, fixed_eval_pack_report=report, cells=("2d/sam2",))
    assert np.isfinite(_native_token_pure_fixed_pack_cosine_gap(evaluation, expected_cells=("2d/sam2",)))


def test_incomplete_dataset_never_passes_handoff_gate(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/dataset_manifest.json").write_text(json.dumps({"complete": True, "scene_count": 3255}))
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_dataset_seal(tmp_path)


def test_snapshot_verification_detects_code_and_native_runtime_drift(tmp_path):
    code = tmp_path / "source.py"
    code.write_text("# frozen code\n")
    native = tmp_path / "pointrope.so"
    native.write_bytes(b"native runtime marker")
    path = tmp_path / "code_manifest.json"
    payload = {"schema_version": "structured3d_sam2_bstar_code_manifest/v1", "passed": True,
               "snapshot_root": str(tmp_path), "files": [{"path": code.name, "sha256": sha256_file(code)}],
               "runtime_artifacts": [{"path": str(native), "sha256": sha256_file(native)}]}
    path.write_text(json.dumps(payload))
    assert verify_snapshot(path)["passed"]
    native.write_bytes(b"changed native binary")
    with pytest.raises(RuntimeError, match="runtime drift"):
        verify_snapshot(path)
    native.write_bytes(b"native runtime marker")
    code.write_text("# changed code\n")
    with pytest.raises(RuntimeError, match="snapshot drift"):
        verify_snapshot(path)
