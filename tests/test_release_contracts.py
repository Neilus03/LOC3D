"""Public release interfaces, portable configuration and label-free sources."""
import argparse
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from delimit3d.runtime.config import load_config
from delimit3d.runtime.records import config_sha256, file_sha256


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_configuration_inheritance_and_output_relocation(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text("paths: {output_root: old}\nmodel: {width: 72, stages: [1, 2]}\n")
    (tmp_path / "config.yaml").write_text("extends: base.yaml\nmodel: {stages: [3]}\n")
    original = load_config(tmp_path / "config.yaml")
    relocated = {**original, "paths": {"output_root": "elsewhere"}}
    assert original["model"] == {"width": 72, "stages": [3]}
    assert config_sha256(original) == config_sha256(relocated)
    (tmp_path / "base.yaml").write_text("extends: config.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "config.yaml")


def two_d_source(root, scene="scene_00000", crop="full_scene"):
    from delimit3d.cli.prepare_sources import ProjectedProposalFrame, _array_record, _file_record, _write_raw_cache
    from delimit3d.data.source_2d import SCHEMA
    from delimit3d.training.contracts import EXPERIMENT_ID, TWO_D_CELLS
    folder = root / scene
    folder.mkdir(parents=True)
    arrays = {"points": np.arange(18, dtype=np.float32).reshape(6, 3),
              "colors": np.full((6, 3), 128, dtype=np.uint8),
              "normals": np.tile([0., 0., 1.], (6, 1)).astype(np.float32),
              "crop_indices": np.arange(6, dtype=np.int64)}
    for key, array in arrays.items():
        np.save(folder / (key + ".npy"), array)
    projected = {key: tuple(ProjectedProposalFrame(
        frame_id=f"0:{index}", split=split, eligible_global_indices=np.arange(6),
        proposal_global_indices=(np.array([0, 1, 2]), np.array([2, 3, 4])),
        proposal_order_indices=np.arange(2), proposal_source_indices=np.arange(2),
        raw_stats={}, used_relaxed_fallback=False) for index, split in enumerate(("train", "heldout")))
        for key in ("g02", "g05", "g08")}
    _write_raw_cache(path=folder / "raw_projection_cache.npz", crop_indices=np.arange(6), projected=projected, num_points=6)
    artifacts = {key: _array_record(folder / (key + ".npy"), relative_to=folder) for key in arrays}
    artifacts["raw_projection_cache"] = _file_record(folder / "raw_projection_cache.npz", relative_to=folder)
    path = folder / "source_manifest.json"
    path.write_text(json.dumps({"schema_version": SCHEMA, "scene_id": scene, "experiment_id": EXPERIMENT_ID,
        "complete": True, "source_cells": list(TWO_D_CELLS), "labels": None,
        "crop": {"mode": crop, "num_points": 6}, "artifacts": artifacts,
        "frames": {"train": ["0:0"], "heldout": ["0:1"]},
        "counts": {key: {cell: 2 for cell in TWO_D_CELLS} for key in ("trainable_proposals", "heldout_trainable_proposals")}}))
    return path


@pytest.mark.parametrize("crop", ["full_scene", "sphere_50k"])
def test_two_d_sources_never_read_labels_and_preserve_overlap(tmp_path, monkeypatch, crop):
    from delimit3d.pretraining.sources import load_scene_source
    path = two_d_source(tmp_path, crop=crop)
    original_load = np.load
    def no_labels(path, *args, **kwargs):
        assert "labels" not in str(path) and "hierarchy" not in str(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", no_labels)
    scene = load_scene_source(path)
    assert scene.labels_by_key == {}
    assert scene.features.shape == (6, 6)
    assert scene.raw_frames[0].proposal_point_indices.tolist() == [0, 1, 2, 2, 3, 4]
    assert 5 in scene.raw_frames[0].visible_indices
    assert 5 not in scene.raw_frames[0].proposal_point_indices


def test_two_d_seal_supports_a_new_sphere_pack(tmp_path):
    from delimit3d.data.cli import seal_main
    from delimit3d.pretraining.dataset import _load_dataset_source_audit
    root = tmp_path / "sources"
    two_d_source(root, "scene_00000", "sphere_50k")
    two_d_source(root, "scene_00001", "sphere_50k")
    train, val, output = (tmp_path / name for name in ("train.txt", "val.txt", "audit.json"))
    train.write_text("scene_00000\n"); val.write_text("scene_00001\n")
    seal_main(["--source-root", str(root), "--train-split", str(train), "--holdout-split", str(val), "--output", str(output)])
    report, hashes = _load_dataset_source_audit(path=output, dataset="structured3d", source_root=root, train_split=train, holdout_split=val)
    assert report["crop_mode"] == "sphere_50k" and len(hashes) == 2
    assert json.loads(output.read_text())["source_cells"] == ["2d/g02", "2d/g05", "2d/g08"]


def test_source_seal_requires_nonempty_disjoint_splits(tmp_path):
    from delimit3d.data.cli import seal_main
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text(""); val.write_text("scene_00000\n")
    arguments = ["--source-root", str(tmp_path), "--train-split", str(train),
                 "--holdout-split", str(val), "--output", str(tmp_path / "audit.json")]
    with pytest.raises(ValueError, match="nonempty"):
        seal_main(arguments)
    train.write_text("scene_00000\n")
    with pytest.raises(ValueError, match="overlap"):
        seal_main(arguments)


def test_execution_record_binds_executed_package_and_input_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from delimit3d.pretraining import experiment_record as record
    monkeypatch.setenv("RANK", "0")
    package = Path(record.__file__).resolve().parents[1]
    config = tmp_path / "config.json"; config.write_text("{}")
    code = tmp_path / "code.json"
    payload = {"schema_version": "delimit3d_code_manifest/v1", "passed": True,
        "snapshot_root": str(tmp_path), "files": {
            "src/delimit3d/" + str(source.relative_to(package)): file_sha256(source)
            for source in package.rglob("*") if source.is_file() and "__pycache__" not in source.parts
            and source.suffix in {".py", ".json", ".patch", ".list"}}}
    payload["files"]["config.json"] = file_sha256(config)
    code.write_text(json.dumps(payload))
    assert record.verify_code_record(code)["passed"]
    artifacts = {"code_manifest": code}
    for name in ("initial_checkpoint", "train_split", "holdout_split"):
        artifacts[name] = tmp_path / name; artifacts[name].write_text(name)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"passed": True, "artifact_hashes_verified": True}))
    args = SimpleNamespace(**artifacts, source_audit=audit, output_dir=tmp_path / "run",
        execution_provenance=None, smoke_manifest=None, smoke_manifest_list=None)
    record.prepare_execution_record(args)
    saved = json.loads(args.execution_provenance.read_text())
    assert record.validate_execution_record(saved, args)["passed"]
    assert args.execution_provenance_sha256 == file_sha256(args.execution_provenance)
    args.output_dir = tmp_path / "relocated"; args.execution_provenance = None
    record.prepare_execution_record(args)
    assert json.loads(args.execution_provenance.read_text()) == saved
    artifacts["train_split"].write_text("changed")
    with pytest.raises(ValueError, match="inputs or code changed"):
        record.validate_execution_record(saved, args)
    payload["files"]["src/delimit3d/__init__.py"] = "0" * 64
    code.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Executed code differs"):
        record.verify_code_record(code)
    payload["files"] = {"../outside.py": "0" * 64}
    code.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid code record path"):
        record.verify_code_record(code)


def test_code_record_does_not_overwrite_a_changed_snapshot(tmp_path):
    from delimit3d.data.cli import code_record_main
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    source = tmp_path / "src"; source.mkdir()
    (source / "exclusions.list").write_text("scene_00000\n")
    output = tmp_path / "code.json"
    arguments = ["--repo-root", str(tmp_path), "--output", str(output)]
    code_record_main(arguments)
    saved = output.read_bytes()
    assert "src/exclusions.list" in json.loads(saved)["files"]
    code_record_main(arguments)
    assert output.read_bytes() == saved
    (source / "exclusions.list").write_text("scene_00001\n")
    with pytest.raises(FileExistsError, match="fresh output filename"):
        code_record_main(arguments)
    assert output.read_bytes() == saved


def test_stateless_batchnorm_restores_on_failure():
    from delimit3d.evaluation.backbone_batchnorm import current_batch_without_updates
    backbone = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5)).eval()
    state = {key: value.clone() for key, value in backbone.state_dict().items()}
    x = torch.randn(12, 4)
    with pytest.raises(RuntimeError, match="interrupted"):
        with current_batch_without_updates(backbone):
            actual = backbone(x)
            expected = torch.nn.functional.batch_norm(x, None, None, backbone[0].weight, backbone[0].bias, training=True)
            torch.testing.assert_close(actual, expected)
            raise RuntimeError("interrupted")
    assert not any(module.training for module in backbone.modules())
    assert "forward" not in backbone[0].__dict__
    assert all(torch.equal(value, state[key]) for key, value in backbone.state_dict().items())


def test_evaluation_starts_with_all_object_slots_without_mutating_input():
    from delimit3d.tasks.agile3d.engine import initialize_evaluation_clicks
    original = {"1": [4], "2": [9]}
    clicks, times = initialize_evaluation_clicks(original, 2)
    assert clicks == times == {"0": [], "1": [], "2": []}
    assert original == {"1": [4], "2": [9]}
    clicks["1"].append(5)
    assert times["1"] == []
    assert initialize_evaluation_clicks({}, 2)[0] == {"0": [], "1": [], "2": []}
    with pytest.raises(ValueError, match="positive object count"):
        initialize_evaluation_clicks({}, 0)


def test_probe_uses_original_query_rows_and_expands_predictions(tmp_path):
    from delimit3d.probes.cli import main
    features = np.array([[1., 0.], [.98, .02], [0., 1.]], dtype=np.float32)
    inverse = np.array([0, 0, 1, 2, 2])
    source, output = tmp_path / "features.npz", tmp_path / "cosine.npz"
    np.savez(source, features=features, original_to_token=inverse)
    main(["--features", str(source), "--method", "cosine", "--queries", "0,3", "--device", "cpu", "--output", str(output)])
    with np.load(output) as payload:
        assert payload["labels"].tolist() == [0, 0, 0, 1, 1]
    main(["--features", str(source), "--method", "kmeans", "--k", "2", "--restarts", "2", "--device", "cpu", "--output", str(tmp_path / "kmeans.npz")])


def test_prepared_mask3d_geometry_has_no_teacher_label_dependency(tmp_path):
    from delimit3d.tasks.mask3d.data.training_pack import load_training_pack_scene_multi
    folder = tmp_path / "scene0000_00"; folder.mkdir()
    for name, array in (("points", np.zeros((4, 3), dtype=np.float32)), ("colors", np.zeros((4, 3), dtype=np.uint8)), ("normals", np.tile([0.,0.,1.],(4,1)).astype(np.float32))):
        np.save(folder / f"{name}.npy", array)
    (folder / "scene_meta.json").write_text(json.dumps({"schema": "delimit3d_scannet_geometry/v1", "scene_id": folder.name,
        "num_points": 4, "granularities": [.5], "label_files": {},
        "artifacts_sha256": {name: file_sha256(folder / f"{name}.npy") for name in ("points","colors","normals")}}))
    scene = load_training_pack_scene_multi(folder, ("g05",))
    assert scene.num_points == 4 and scene.valid_points.all()


def test_agile3d_evaluation_accepts_new_data_but_resume_rejects_it(tmp_path):
    from delimit3d.tasks.agile3d.trainer import SCHEMA, load_checkpoint
    from delimit3d.tasks.contracts import model_contract
    config = {"task": "agile3d", "model": {"backbone_name": "litept_s_star", "feature_dim": 72},
              "protocol": {"voxel_size_m": .05, "input_features": "rgbn6"}, "_config_sha256": "recipe",
              "_data_identity": {"dataset": "s3dis"}}
    path = tmp_path / "checkpoint.pt"
    torch.save({"schema": SCHEMA, "model": {}, "epoch": 0, "step": 1, "config_sha256": "recipe",
                "data_identity": {"dataset": "scannet"}, "model_contract": model_contract(config)}, path)
    assert load_checkpoint(path, config)["epoch"] == 0
    with pytest.raises(ValueError, match="data changed"):
        load_checkpoint(path, config, resume=True)


def test_new_normal_audit_is_bound_to_scene_selection_and_source_bytes(tmp_path):
    from delimit3d.data.normal_audit import main, validate
    root = tmp_path / "sources"
    manifest = two_d_source(root)
    split, output = tmp_path / "train.txt", tmp_path / "audit.json"
    split.write_text("scene_00000\n")
    main(["--source-root", str(root), "--train-split", str(split), "--scenes", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert validate(report, source_root=root, train_split=split, scene_count=1, seed=42)["passed"]
    with pytest.raises(ValueError, match="selection changed"):
        validate(report, source_root=root, train_split=split, scene_count=1, seed=43)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        validate(report, source_root=root, train_split=split, scene_count=1, seed=42)


def test_multiscale_calibration_rejects_incomplete_or_changed_artifacts(tmp_path):
    from types import SimpleNamespace
    from delimit3d.pretraining.calibration import SCHEMA_VERSION, validate_training_calibration
    artifacts = {}
    for name in ("initial_checkpoint", "source_audit", "code_manifest"):
        artifacts[name] = tmp_path / name
        artifacts[name].write_text(name)
    args = SimpleNamespace(**artifacts, seed=42, expected_world_size=2, backbone_lr=.001, head_lr=.003,
        weight_decay=.05, grad_clip_norm=1., scenes_per_rank_per_update=6,
        multiscale_loss_weight=.4, multiscale_warmup_epochs=20)
    report = {"schema_version": SCHEMA_VERSION, "measurement_completed": True, "passed": True,
        "training_authorized": True, "seed": 42, "world_size": 2,
        "paired_optimizer_gate": {"completed": True, "passed": True},
        "objective_contract": {"auxiliary_weight": .4, "auxiliary_warmup_epochs": 20},
        "optimizer_contract": {"backbone_peak_lr": .001, "head_peak_lr": .003, "weight_decay": .05,
            "gradient_clip_norm": 1., "global_updates_per_branch": 8, "scenes_per_rank_per_update": 6},
        **{key + "_sha256": file_sha256(path) for key, path in artifacts.items()}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    assert validate_training_calibration(path, args)["passed"]
    report["paired_optimizer_gate"]["completed"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed gradient"):
        validate_training_calibration(path, args)
    report["paired_optimizer_gate"]["completed"] = True
    path.write_text(json.dumps(report))
    artifacts["code_manifest"].write_text("changed code")
    with pytest.raises(ValueError, match="code_manifest differs"):
        validate_training_calibration(path, args)


def test_task_preparation_preserves_mesh_and_normal_order(tmp_path, monkeypatch):
    from plyfile import PlyData, PlyElement
    from delimit3d.data.prepare_tasks import main
    from delimit3d.tasks.data_identity import agile3d_data_identity
    scene = "scene0000_00"
    raw, processed, normal_root = tmp_path / "raw", tmp_path / "processed", tmp_path / "normals"
    (raw / scene).mkdir(parents=True); processed.mkdir()
    xyz = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]], dtype=np.float32)
    vertex = np.zeros(4, dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
    for i, key in enumerate(("x","y","z")): vertex[key] = xyz[:,i]
    vertex["red"] = 128
    face = np.array([([0,1,2],),([0,2,3],)], dtype=[("vertex_indices","i4",(3,))])
    PlyData([PlyElement.describe(vertex,"vertex"), PlyElement.describe(face,"face")]).write(raw/scene/f"{scene}_vh_clean_2.ply")
    PlyData([PlyElement.describe(vertex,"vertex")]).write(processed/f"{scene}.ply")
    split = tmp_path / "scenes.txt"; split.write_text(scene + "\n")
    main(["--task","agile3d","--raw-scans-root",str(raw),"--scene-ids-file",str(split),
        "--processed-scan-folder",str(processed),"--output-root",str(normal_root)])
    np.testing.assert_array_equal(np.load(normal_root/scene/"coord.npy"),xyz)
    np.testing.assert_allclose(np.load(normal_root/scene/"normal.npy"),[[0.,0.,1.]]*4,atol=1e-6)
    val = tmp_path / "val.json"; val.write_text(json.dumps({scene+"_obj_1": None}))
    identity = agile3d_data_identity({"val_list":val,"normal_root":normal_root,"scan_folder":processed},training=False)
    assert identity["scenes"][scene]["coordinate"] == file_sha256(normal_root/scene/"coord.npy")
    np.save(normal_root/scene/"coord.npy",xyz[::-1])
    with pytest.raises(ValueError,match="not row-aligned"):
        agile3d_data_identity({"val_list":val,"normal_root":normal_root,"scan_folder":processed},training=False)
