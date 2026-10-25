"""Exercise the sparse loader boundary, real engine dispatch and update policy."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delimit3d.runtime.records import file_sha256
from delimit3d.tasks.data_identity import mask3d_data_identity
from delimit3d.tasks.mask3d.data.batched_scene_batch import collate_multi_scene_samples
from delimit3d.tasks.mask3d.low_annotation import (
    SupervisionAudit, suppress_empty_adamw_update, validate_config, validate_voxel_mapping,
)
from delimit3d.tasks.mask3d.low_annotation_data import LowAnnotationDataset, make_sparse_targets
from delimit3d.tasks.mask3d.sparse_supervision import SparseAnnotations, SparseMask3DCriterion


def prepared(tmp_path, monkeypatch):
    from delimit3d.tasks.mask3d import low_annotation_data as module

    def augmentation(points, colors, *, normals):
        return points + np.random.uniform(-.02, .02, size=(1, 3)).astype(np.float32), colors, normals

    monkeypatch.setattr(module, "augment_points_mask3d_scannet_rgbn6_aligned", augmentation)
    for folder in ("geometry", "train_annotations/20", "validation", "inputs"):
        (tmp_path / folder).mkdir(parents=True)
    points = np.arange(72, dtype=np.float32).reshape(24, 3) / 100
    for scene in ("scene0000_00", "scene0001_00"):
        np.savez(tmp_path / "geometry" / f"{scene}.npz", points=points,
                 colors=np.ones_like(points) * .5, normals=np.tile([0., 0., 1.], (24, 1)).astype(np.float32))
    indices = np.r_[np.arange(19), 0]
    np.savez(tmp_path / "train_annotations/20/scene0000_00.npz", vertex_indices=indices,
             instance_ids=np.ones(20, dtype=np.int64), semantic_ids=np.full(20, 5, dtype=np.int64))
    np.savez(tmp_path / "validation/scene0001_00.npz", instance_ids=np.ones(24, dtype=np.int64),
             instance_class_pairs=np.array([[1, 4]]))
    train, val = tmp_path / "inputs/train.txt", tmp_path / "inputs/validation.txt"
    train.write_text("scene0000_00\n")
    val.write_text("scene0001_00\n")
    manifest = dict(budgets=[20], hidden_training_annotations_exported=False,
                    scene_ids=dict(train=["scene0000_00"], validation=["scene0001_00"]), files={})
    for path in sorted(tmp_path.rglob("*.npz")):
        relative = str(path.relative_to(tmp_path))
        manifest["files"][relative] = dict(path=relative, bytes=path.stat().st_size, sha256=file_sha256(path))
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "data_manifest.sha256").write_text(file_sha256(manifest_path))
    return dict(experiment=dict(seed=42), data=dict(train_split=str(train), val_split=str(val),
                granularity=.5, limited_annotation=dict(budget_points=20, prepared_root=str(tmp_path))))


def test_training_only_opens_geometry_and_selected_annotations(tmp_path, monkeypatch):
    prepared(tmp_path, monkeypatch)
    opened = []
    original = np.load

    def tracking(path, *args, **kwargs):
        opened.append(Path(path).relative_to(tmp_path).as_posix())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracking)
    dataset = LowAnnotationDataset(tmp_path, ["scene0000_00"], train=True, budget=20)
    first = dataset[0]
    assert opened == ["geometry/scene0000_00.npz", "train_annotations/20/scene0000_00.npz"]
    assert first["labels_by_granularity"]["g05"][19:].tolist() == [-1] * 5
    assert first["scan_segments"].eq(0).all()
    assert first["scene_meta"]["la_indices"] == list(range(19)) + [0]
    targets = make_sparse_targets(collate_multi_scene_samples([first]), "g05", "cpu", class_aware=False)
    assert targets[0].vertex_indices.tolist() == list(range(19)) + [0]
    assert targets[0].num_points == 24


def test_augmentation_is_scene_epoch_deterministic_and_preserves_global_rng(tmp_path, monkeypatch):
    prepared(tmp_path, monkeypatch)
    dataset = LowAnnotationDataset(tmp_path, ["scene0000_00"], train=True, budget=20)
    np.random.seed(8)
    random.seed(8)
    torch.manual_seed(8)
    np_before, py_before, torch_before = np.random.get_state(), random.getstate(), torch.get_rng_state()
    first = dataset[0]
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert random.getstate() == py_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    torch.rand(100)
    second = dataset[0]
    assert torch.equal(first["points"], second["points"])
    dataset.epoch = 2
    third = dataset[0]
    assert not torch.equal(first["points"], third["points"])
    assert first["scene_meta"]["la_indices"] == third["scene_meta"]["la_indices"]


def test_engine_dispatch_and_validation_never_builds_sparse_training_targets(tmp_path, monkeypatch):
    config = prepared(tmp_path, monkeypatch)
    from delimit3d.tasks.mask3d.engine import _make_dataset, _make_targets, _concat_segments
    train = _make_dataset(config, train=True)
    batch = collate_multi_scene_samples([train[0]])
    target = _make_targets(batch, "g05", "cpu", class_aware=False, limited_annotation=True)
    assert len(target[0].vertex_indices) == 20
    segments = _concat_segments(batch, "cpu")
    assert segments.unique().numel() > 1
    with pytest.raises(ValueError, match="dense target"):
        _make_targets(batch, "g05", "cpu", class_aware=False)
    validation = _make_dataset(config, train=False)
    dense_batch = collate_multi_scene_samples([validation[0]])
    with pytest.raises(ValueError, match="Dense validation"):
        _make_targets(dense_batch, "g05", "cpu", class_aware=False, limited_annotation=True)
    with pytest.raises(ValueError, match="dense training-evaluation"):
        _make_dataset(config, train=False, split_key="train_split")
    with pytest.raises(ValueError, match="cannot share"):
        _concat_segments(SimpleNamespace(scene_meta=[{"limited_annotation": True}, {}]), "cpu")


def test_data_identity_rejects_mutation_and_wrong_train_membership(tmp_path, monkeypatch):
    config = prepared(tmp_path, monkeypatch)
    identity = mask3d_data_identity(config)
    assert identity["budget_points"] == 20
    assert len(identity["scenes"]) == 2
    assert len(mask3d_data_identity(config, evaluation=True)["scenes"]) == 1
    Path(config["data"]["train_split"]).write_text("scene0001_00\n")
    with pytest.raises(ValueError, match="overlap"):
        mask3d_data_identity(config)
    Path(config["data"]["train_split"]).write_text("scene0000_00\n")
    with (tmp_path / "train_annotations/20/scene0000_00.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Sealed LA asset changed"):
        mask3d_data_identity(config)
    # Evaluation is independent of all training annotation content.
    assert len(mask3d_data_identity(config, evaluation=True)["scenes"]) == 1


def test_empty_batch_preserves_adamw_parameters_moments_and_step_counter():
    parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=10)
    parameter.square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    state = copy.deepcopy(optimizer.state_dict()["state"])
    (parameter.sum() * 0).backward()
    criterion = SimpleNamespace(all_matches=[[(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]])
    assert not suppress_empty_adamw_update(optimizer, criterion)
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 2
    assert torch.equal(parameter, before)
    for index, row in state.items():
        for key, value in row.items():
            assert torch.equal(value, optimizer.state_dict()["state"][index][key])


def test_geometry_inverse_is_checked():
    valid = dict(point_to_segment=torch.tensor([0, 0, 1]), point_to_voxel=torch.tensor([0, 0, 1]))
    validate_voxel_mapping([valid])
    valid["point_to_segment"] = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="actual backbone"):
        validate_voxel_mapping([valid])


@pytest.mark.parametrize("section,key,value", [
    ("train", "backbone_lr", 1e-5), ("train", "gradient_accumulation_steps", 5),
    ("data", "max_points", 100000), ("eval", "report_train_metrics", True),
    ("model", "voxel_size", .02), ("scheduler", "max_lr", .001),
    ("loss", "include_auxiliary", False),
])
def test_recipe_rejects_dense_or_unmatched_changes(section, key, value):
    from delimit3d.runtime.config import load_config
    config = load_config(Path(__file__).parents[1] / "configs/tasks/mask3d_litept.yaml", expand_env=False)
    config["data"]["limited_annotation"] = dict(prepared_root="/external/prepared", budget_points=20)
    config["train"]["backbone_lr"] = 1e-4
    validate_config(config)
    config[section][key] = value
    with pytest.raises(ValueError, match="LA"):
        validate_config(config)


def test_supervision_audit_checks_exact_epoch_coverage():
    audit = SupervisionAudit()
    batch = SimpleNamespace(scene_ids=["scene0000_00"], points=torch.zeros(2, 3), features=torch.ones(2, 6))
    target = SparseAnnotations(torch.tensor([0]), torch.tensor([1]), torch.tensor([5]), 2)
    audit.observe(batch, [target])
    criterion = SimpleNamespace(statistics=[dict(official_annotations=1, unknown_instances=None)])
    with pytest.raises(ValueError, match="exactly once"):
        audit.finish(["scene0000_00", "scene0001_00"], criterion, 1)
    audit.effective_updates = 1
    row = audit.finish(["scene0000_00"], criterion, 1)
    assert row["scheduled_optimizer_slots"] == 1
    assert row["retention"]["official_annotations"] == 1
    assert row["unknown_instances"] is None


def test_engine_cpu_control_flow_commits_and_resumes_sparse_epochs(tmp_path, monkeypatch):
    """Run the real engine with a tiny CPU model, including optimizer/checkpoints.

    This verifies integration and resume; native CUDA backbone capacity is a
    separate hardware smoke, not established by this test.
    """
    from delimit3d.runtime.config import load_config
    from delimit3d.tasks.mask3d import engine, factory
    from delimit3d.tasks.mask3d.initialization import tensor_state_sha256
    data_root = tmp_path / "data"
    selected = prepared(data_root, monkeypatch)
    config = load_config(Path(__file__).parents[1] / "configs/tasks/mask3d_litept.yaml", expand_env=False)
    config["data"].update(selected["data"])
    config["experiment"].update(output_root=str(tmp_path / "runs"), name="sparse")
    config["train"].update(backbone_lr=1e-4, max_epochs=2, num_workers=0,
                           max_run_hours=0, evaluation_epochs=[2], checkpoint_epochs=[2])

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(6, 3)
            self.decoder = torch.nn.Linear(3, 4)
            self.decoder.num_classes = 2
            self.decoder.train_queries = self.decoder.eval_queries = 100
            self.backbone_frozen = False

        def backbone_state_hash(self):
            return tensor_state_sha256(self.backbone.state_dict())

        def decoder_state_hash(self):
            return tensor_state_sha256(self.decoder.state_dict())

        def parameter_groups(self, *, backbone_lr, decoder_lr, include_frozen_backbone):
            return [dict(params=list(self.backbone.parameters()), lr=backbone_lr, name="backbone"),
                    dict(params=list(self.decoder.parameters()), lr=decoder_lr, name="decoder")]

        def forward(self, points, features, segments, *, scene_offsets):
            values = self.decoder(self.backbone(features))
            output, start = [], 0
            for stop in scene_offsets.tolist():
                inverse = segments[start:stop]
                value = values[start:stop].mean(0)
                mask = value[:2, None].expand(2, int(inverse.max()) + 1)
                objects = value[2:].expand(2, 2)
                output.append(dict(segment_mask_logits=mask, object_logits=objects,
                                   point_to_segment=inverse, point_to_voxel=inverse,
                                   aux_outputs=[dict(segment_mask_logits=mask * .9, object_logits=objects * .9)]))
                start = stop
            return output

    def build_model(config, device):
        model = TinyModel()
        report = dict(backbone_source="scratch", backbone_checkpoint=None,
                      decoder_template=dict(downstream_seed=42), backbone_frozen=False)
        return model, report

    evaluations = []

    def evaluate(model, loader, **kwargs):
        for batch in loader:
            assert all("la_indices" not in meta for meta in batch.scene_meta)
            predictions = model(batch.points, batch.features, engine._concat_segments(batch, "cpu"),
                                scene_offsets=batch.point_offsets)
            validate_voxel_mapping(predictions)
        evaluations.append(True)
        return {"raw_queries": {"CA-AP": 0., "CA-AP50": 0., "CA-AP25": 0.,
                                "min_scene_CA-AP50": 0., "class_agnostic": True,
                                "per_scene": {"scene0001_00": {}}}}

    monkeypatch.setattr(factory, "build_model", build_model)
    monkeypatch.setattr(engine, "_distributed_runtime", lambda *a: (False, 0, 0, 1, torch.device("cpu"), None))
    monkeypatch.setattr(engine, "_evaluate", evaluate)
    arguments = ["--config", str(tmp_path / "unused.yaml"), "--device", "cpu"]
    engine.main(arguments, config=copy.deepcopy(config))
    output = tmp_path / "runs/sparse"
    latest = output / "checkpoints/latest.pt"
    first = torch.load(latest, weights_only=False)
    assert first["epoch"] == 1
    assert first["scheduler_state_dict"]["last_epoch"] == 1
    assert not evaluations
    engine.main([*arguments, "--resume", str(latest)], config=copy.deepcopy(config))
    final = torch.load(latest, weights_only=False)
    assert final["epoch"] == 2
    assert final["scheduler_state_dict"]["total_steps"] == final["scheduler_state_dict"]["last_epoch"] == 2
    assert len(evaluations) == 1
    assert (output / "eval_epoch_0002.json").is_file()
    assert not (output / "checkpoints/epoch_0001.pt").exists()
    audits = [json.loads(line) for line in (output / "supervision.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in audits] == [1, 2]
    assert all(row["effective_optimizer_updates"] == 1 for row in audits)
    assert all(int(state["step"]) == 2 for state in final["optimizer_state_dict"]["state"].values())
