"""Opt-in two-GPU SAM2 policy, fixed evaluation and state recovery check."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b, *, rtol=1e-5, atol=1e-6):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y, rtol=rtol, atol=atol)
    else:
        assert a == b


def compare_recovery(a, b):
    # Policy verification requires exact replay of random and sampling state.
    # Floating-point model/optimizer differences are measured against a fresh
    # uninterrupted control and recorded, rather than asserting GPU identity.
    for key in ("rank_coverage_states", "rank_exposure_counts", "rank_rng_states",
                "rank_sampling_totals", "update", "epoch"):
        compare(a[key], b[key])


def max_tensor_difference(a, b):
    if isinstance(a, torch.Tensor):
        return (a - b).abs().max().item() if a.numel() else 0.0
    if isinstance(a, dict):
        return max((max_tensor_difference(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, (list, tuple)):
        return max((max_tensor_difference(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def stricter_model_comparison(a, b):
    try:
        compare(a["model_state_dict"], b["model_state_dict"])
    except AssertionError:
        return False
    return True


def make_sources(folder):
    from test_native_runtime import cloud
    from delimit3d.cli.prepare_sources import _array_record, _file_record
    from delimit3d.data.sam2_source import SCENE_SCHEMA, CACHE_SCHEMA, SOURCE_CELL
    from delimit3d.runtime.records import write_json
    from delimit3d.training.contracts import EXPERIMENT_ID

    source = folder / "sources"
    source.mkdir()
    points, features = cloud()
    points, features = (points * 4).numpy(), features.numpy()
    count = len(points)
    scenes = [f"scene_{index:05d}" for index in range(13)]
    # Nested masks exercise the policy difference. All points are known;
    # the CPU tests separately cover unknown points and mixed native tokens.
    proposals = (*np.split(np.arange(count), 16), np.arange(count // 2))
    sizes = [len(proposal) for proposal in proposals] * 2
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    for scene in scenes:
        destination = source / scene
        destination.mkdir()
        arrays = {"points": points, "colors": np.uint8(features[:, :3] * 255),
                  "normals": features[:, 3:], "crop_indices": np.arange(count, dtype=np.int64)}
        for key, array in arrays.items():
            np.save(destination / f"{key}.npy", array)
        artifacts = {key: _array_record(destination / f"{key}.npy", relative_to=destination)
                     for key in arrays}
        cache = destination / "sam2_projection_cache.npz"
        np.savez_compressed(cache,
            schema_version=np.array(CACHE_SCHEMA), scene_id=np.array(scene),
            teacher_source_key=np.array(SOURCE_CELL),
            crop_indices_sha256=np.array(artifacts["crop_indices"]["sha256"]),
            physical_frame_ids=np.array(["0:0", "0:1"]), split_codes=np.array([0, 1], dtype=np.int8),
            frame_offsets=np.array([0, count, 2 * count], dtype=np.int64),
            eligible_point_indices=np.tile(np.arange(count, dtype=np.int64), 2),
            proposal_frame_offsets=np.array([0, len(proposals), 2 * len(proposals)], dtype=np.int64),
            proposal_point_offsets=offsets,
            proposal_point_indices=np.tile(np.concatenate(proposals).astype(np.int64), 2))
        artifacts["sam2_projection_cache"] = _file_record(cache, relative_to=destination)
        write_json(destination / "source_manifest.json", {
            "schema_version": SCENE_SCHEMA, "scene_id": scene, "dataset": "structured3d",
            "experiment_id": EXPERIMENT_ID, "complete": True,
            "source_cells": [SOURCE_CELL], "teacher_source_key": SOURCE_CELL,
            "single_granularity": True, "labels": None,
            "crop": {"mode": "full_scene", "num_points": count}, "artifacts": artifacts,
            "checks": {"synthetic_artifacts_complete": True},
            "semantics": {key: True for key in (
                "frame_local", "overlap_preserved", "unseen_points_are_unknown",
                "visible_but_unmasked_points_are_unknown", "zero_proposal_frames_contribute_no_supervision")},
            "frames": {"physical": ["0:0", "0:1"], "train": ["0:0"], "heldout": ["0:1"]},
            "counts": {name: {SOURCE_CELL: len(proposals)} for name in (
                "trainable_proposals", "heldout_trainable_proposals")}})
    train, val = folder / "train.txt", folder / "val.txt"
    train.write_text("\n".join(scenes[:12]) + "\n")
    val.write_text(scenes[12] + "\n")
    return source, train, val


def main():
    if os.environ.get("DELIMIT3D_NATIVE_CHECKS") != "1" or torch.cuda.device_count() != 2:
        raise RuntimeError("This check needs an isolated allocation with exactly two CUDA GPUs")
    from delimit3d.data.cli import seal_main, fixed_eval_main, code_record_main
    from delimit3d.data.partfield_contrastive import negative_policy_name
    from delimit3d.runtime.config import load_config
    from delimit3d.runtime.records import file_sha256, write_json

    folder = Path(os.environ.get("DELIMIT3D_NATIVE_OUTPUT_DIR", ROOT / "runs/native_sam2_policies_check"))
    folder.mkdir(parents=True, exist_ok=False)
    source, train, val = make_sources(folder)
    audit, code, initial = (folder / name for name in ("audit.json", "code.json", "initial.pt"))
    seal_main(["--source-root", str(source), "--train-split", str(train),
               "--holdout-split", str(val), "--output", str(audit)])
    code_record_main(["--output", str(code), "--repo-root", str(ROOT)])
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--action", "backbone-init",
        "--litept-root", str(ROOT / ".deps/LitePT"), "--litept-variant", "litept_s_star",
        "--output", str(initial), "--seed", "42", "--grid-size", "0.02", "--input-features", "rgbn6"],
        check=True, cwd=ROOT)
    commands, checkpoints, results = {}, {}, {}
    for flag in (False, True):
        policy = negative_policy_name(flag)
        arm = folder / policy
        arm.mkdir()
        preset = ROOT / "configs/pretrain" / (
            "sam2_selected_proposal_negatives.yaml" if flag else "sam2_overlap_veto.yaml")
        fixed = arm / "fixed"
        fixed_eval_main(["--source-root", str(source), "--holdout-split", str(val),
            "--teacher", "sam2", "--grid-size", "0.02", "--pretraining-contract", str(preset),
            "--output", str(fixed)])
        config = load_config(preset, expand_env=False)
        config.update(litept_root=str(ROOT / ".deps/LitePT"), source_root=str(source), source_audit=str(audit),
            train_split=str(train), holdout_split=str(val), output_dir=str(arm / "continuous"),
            code_manifest=str(code), fixed_eval_pack=str(fixed), initial_checkpoint=str(initial),
            epochs=3, warmup_epochs=1, evaluation_epochs="0,1,3", checkpoint_epochs="0,1,3")
        recipe = arm / "recipe.json"
        write_json(recipe, config)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(ROOT / "scripts/pretrain.py"), "--config", str(recipe)]
        subprocess.run(command, check=True, cwd=ROOT)
        summary = json.loads((arm / "continuous/run_summary.json").read_text())
        assert summary["mechanics_passed"] and all(summary["checks"].values())
        assert summary["allow_shared_proposal_negatives"] is flag
        assert summary["negative_policy"] == policy
        resolved = json.loads((arm / "continuous/resolved_config.json").read_text())
        assert resolved["allow_shared_proposal_negatives"] is flag
        execution = json.loads((arm / "continuous_execution.json").read_text())
        assert execution["allow_shared_proposal_negatives"] is flag
        # Stage a true interruption at epoch one, including required history.
        parent = arm / "parent_at_epoch_0001/checkpoints"
        parent.mkdir(parents=True)
        for epoch in (0, 1):
            name = f"epoch_{epoch:04d}.pt"
            shutil.copy2(arm / "continuous/checkpoints" / name, parent / name)
        checkpoint = parent / "epoch_0001.pt"
        subprocess.run(command + ["--resume", str(checkpoint), "--resume-sha256", file_sha256(checkpoint),
            "--resume-into-fresh-attempt", "--output-dir", str(arm / "recovered")], check=True, cwd=ROOT)
        resumed_summary = json.loads((arm / "recovered/run_summary.json").read_text())
        assert resumed_summary["mechanics_passed"] and all(resumed_summary["checks"].values())
        continuous = torch.load(arm / "continuous/checkpoints/epoch_0003.pt", map_location="cpu", weights_only=False)
        recovered = torch.load(arm / "recovered/checkpoints/epoch_0003.pt", map_location="cpu", weights_only=False)
        for payload in (continuous, recovered):
            assert payload["allow_shared_proposal_negatives"] is flag
            assert payload["negative_policy"] == policy
            assert payload["resumable"] and payload["update"] == 3
            assert len(payload["rank_rng_states"]) == len(payload["rank_coverage_states"]) == 2
        # Measure repeated-training variation on the same GPUs, initialization
        # and data before comparing recovery state.
        subprocess.run(command + ["--output-dir", str(arm / "replica")], check=True, cwd=ROOT)
        replica_summary = json.loads((arm / "replica/run_summary.json").read_text())
        assert replica_summary["mechanics_passed"] and all(replica_summary["checks"].values())
        replica = torch.load(arm / "replica/checkpoints/epoch_0003.pt", map_location="cpu", weights_only=False)
        compare_recovery(continuous, recovered)
        compare_recovery(continuous, replica)
        results[policy] = {"passed": True, "updates": 3,
            "continuous_runtime_checks": len(summary["checks"]),
            "resumed_runtime_checks": len(resumed_summary["checks"]),
            "fixed_evaluation_verified": True, "checkpoint_policy_and_sampler_recovery_verified": True,
            "rng_and_discrete_state_exact": True, "uninterrupted_replication_control_verified": True,
            "stricter_model_tolerance_passed": {kind: stricter_model_comparison(continuous, other)
                for kind, other in (("resume", recovered), ("uninterrupted_replica", replica))},
            "maximum_absolute_difference": {kind: {key: max_tensor_difference(continuous[key], other[key])
                for key in ("model_state_dict", "optimizer_state_dict")}
                for kind, other in (("resume", recovered), ("uninterrupted_replica", replica))}}
        commands[flag], checkpoints[flag] = command, checkpoint
        write_json(folder / "progress.json", results)
        print(json.dumps({"policy": policy, "training_evaluation_resume_passed": True}), flush=True)
    # Use the other policy's correct fixed pack so these reach the checkpoint
    # guard rather than failing earlier on an evaluation-pack mismatch.
    for flag in (False, True):
        policy = negative_policy_name(flag)
        checkpoint = checkpoints[not flag]
        wrong = folder / policy / "wrong_policy_resume"
        log = folder / policy / "wrong_policy_resume.log"
        with log.open("w") as stream:
            result = subprocess.run(commands[flag] + ["--resume", str(checkpoint),
                "--resume-sha256", file_sha256(checkpoint), "--resume-into-fresh-attempt",
                "--output-dir", str(wrong)], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        assert result.returncode != 0
        assert "SAM2 negative policy differs on resume; start a separate run" in log.read_text()
        assert not (wrong / "checkpoints/epoch_0003.pt").exists()
        results[policy]["opposite_policy_resume_rejected"] = True
    write_json(folder / "COMPLETE.json", {"passed": True, "fixture": "synthetic_nested_sam2_masks",
        "world_size": 2, "global_batch": 12, "normalization": "sync_batchnorm", "precision": "fp32",
        "verification_scope": "negative policy and exact RNG/sampler/counter recovery",
        "floating_point_recovery_scope": "measured against uninterrupted repetition; GPU tensor identity is not asserted",
        "stricter_model_rtol": 1e-5, "stricter_model_atol": 1e-6,
        "initialization_sha256": file_sha256(initial), "policies": results,
        "production_teacher_export_verified": False, "full_benchmark_verified": False})
    print(json.dumps({"passed": True, "sam2_policies_public_training_evaluation_resume": True}), flush=True)


if __name__ == "__main__":
    main()
