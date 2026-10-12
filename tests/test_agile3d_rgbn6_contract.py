"""CPU tests for scientifically material RGBN6 transfer failure modes."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from delimit3d.runtime.config import load_config

from delimit3d.evaluation.agile3d_rgbn6_contract import (
    validate_audit_lock,
    validate_decoder_initialization,
    validate_encoder_argument,
    validate_matched_pair,
    validate_normal_alignment,
)

ROOT = Path(__file__).resolve().parents[1]


def pair():
    pretrained = load_config(ROOT / "configs/tasks/agile3d_litept.yaml", expand_env=False)
    pretrained["model"]["freeze_encoder"] = True
    pretrained["encoder_initialization"]["seed"] = pretrained["seed"]
    pretrained["decoder_initialization"]["tensor_state_sha256"] = "8dbc462b78b9e5d1e3450b7dcff1a7d33af86f31935a1444e9cf7ecbfb4da86a"
    control = copy.deepcopy(pretrained)
    control["encoder_initialization"].update(kind="scratch", checkpoint=None)
    control["paths"]["output_root"] = "random_control"
    return pretrained, control


@pytest.fixture
def runner():
    from delimit3d.tasks.agile3d import engine
    return engine


def test_prepared_configs_vary_only_frozen_encoder_initialization():
    result = validate_matched_pair(*pair())
    assert result["only_factor"] == "frozen_encoder_initialization"


@pytest.mark.parametrize("section,key,value", [
    ("model", "freeze_encoder", False),
    ("model", "in_channels", 3),
    ("protocol", "quantization_backend", "native"),
    ("protocol", "voxel_size_m", 0.02),
    ("optimizer", "decoder_learning_rate", 0.001),
    ("loss", "dice_weight", 1.0),
    ("paths", "train_list", "/some/other/manifest"),
    ("decoder_initialization", "seed", 42),
])
def test_pair_rejects_scientific_drift(section, key, value):
    pretrained, control = pair()
    control[section][key] = value
    with pytest.raises(ValueError, match="settings differ"):
        validate_matched_pair(pretrained, control)


def test_same_count_permuted_or_rotated_normal_sidecars_are_rejected():
    xyz = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    normals = np.eye(3)
    assert validate_normal_alignment(xyz, xyz, normals)["max_coordinate_error_m"] == 0
    for invalid in (xyz[[1, 0, 2]], xyz[:, [1, 0, 2]], xyz + 1):
        with pytest.raises(ValueError, match="not row-aligned"):
            validate_normal_alignment(xyz, invalid, normals)
    with pytest.raises(ValueError, match="non-finite"):
        validate_normal_alignment(xyz, xyz, normals * np.nan)
    with pytest.raises(ValueError, match="magnitudes"):
        validate_normal_alignment(xyz, xyz, normals * 2)
    assert validate_normal_alignment(xyz, xyz, np.zeros_like(normals))["zero_normals"] == 3


def test_random_control_rejects_pretrained_weights_and_pretrained_requires_them():
    pretrained, control = pair()
    validate_encoder_argument(control, None, required=True)
    with pytest.raises(ValueError, match="rejects"):
        validate_encoder_argument(control, "checkpoint.pt", required=True)
    with pytest.raises(ValueError, match="requires"):
        validate_encoder_argument(pretrained, None, required=True)


def test_decoder_initialization_is_pinned_independently_of_training_seed():
    config = pair()[0]
    report = dict(config["decoder_initialization"])
    validate_decoder_initialization(config, report)
    report["tensor_state_sha256"] = "changed"
    with pytest.raises(ValueError, match="tensor_state_sha256"):
        validate_decoder_initialization(config, report)


def test_audit_lock_rejects_changed_data_sources_and_decoder():
    lock = {
        "mode": "audit", "gpu_used": False,
        "matched_pair": {"x": 1}, "list_sha256": {"train": "a"},
        "quantization_backend": "numpy_utility_compat_v1", "source_hashes": {"a.py": "b"},
        "normal_contract": {"dataset_content_sha256": "data"},
        "decoder_initialization": {"file_sha256": "file", "tensor_state_sha256": "state"},
    }
    validate_audit_lock(lock, lock)
    for section, key in (
        ("source_hashes", "a.py"), ("normal_contract", "dataset_content_sha256"),
        ("decoder_initialization", "tensor_state_sha256"), ("list_sha256", "train"),
    ):
        current = copy.deepcopy(lock)
        current[section][key] = "changed"
        with pytest.raises(ValueError, match="mismatch"):
            validate_audit_lock(lock, current)


def fake_model(runner, monkeypatch, config):
    from delimit3d.tasks.agile3d import model as task_model
    class Backbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(6)
            self.linear = torch.nn.Linear(6, 72)

        def forward(self, xyz, features, *, point_offsets, grid_coord):
            tokens = self.linear(self.bn(features))
            return task_model.LitePTBackboneOutput(
                tokens, xyz, tokens, xyz, torch.arange(len(xyz)), torch.arange(len(xyz)),
                point_offsets, point_offsets,
            )

    monkeypatch.setattr(task_model, "LitePTBackbone", Backbone)
    monkeypatch.setattr(task_model, "load_initialization", lambda *a, **k: (
        torch.nn.Linear(72, 2), dict(config["decoder_initialization"]),
    ))
    return task_model.Agile3DModel(config, {"litept_root": Path("unused"), "decoder_initialization": Path("unused")})


def test_decoder_step_preserves_frozen_encoder_bn_and_has_no_encoder_graph(runner, monkeypatch):
    model = fake_model(runner, monkeypatch, pair()[1])
    model.train()
    before = runner.state_sha256(model.backbone.state_dict())
    assert model.decoder.training and not model.backbone.bn.training
    xyz, features = torch.rand(5, 3), torch.rand(5, 6, requires_grad=True)
    tokens, _ = model.encode_batch(torch.arange(15).reshape(5, 3), xyz, features, torch.tensor([5]))
    assert not tokens[0].requires_grad
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=0.01)
    before_decoder = runner.state_sha256(model.decoder.state_dict())
    model.decoder(tokens[0]).square().mean().backward()
    optimizer.step()
    assert runner.state_sha256(model.backbone.state_dict()) == before
    assert runner.state_sha256(model.decoder.state_dict()) != before_decoder
    assert features.grad is None
    model.backbone.bn.train()
    with pytest.raises(RuntimeError, match="normalization layer"):
        model.encode_batch(torch.zeros(5, 3), xyz, features, torch.tensor([5]))


def test_strict_encoder_load_rejects_missing_and_duplicate_mappings(runner, tmp_path):
    model = torch.nn.Linear(6, 72)
    good = tmp_path / "good.pt"
    torch.save({"backbone_state_dict": model.state_dict()}, good)
    assert runner.load_frozen_backbone(model, good)["source_tensor_count"] == 2
    bad = tmp_path / "bad.pt"
    torch.save({"backbone_state_dict": {"weight": model.weight}}, bad)
    with pytest.raises(RuntimeError, match="missing"):
        runner.load_frozen_backbone(model, bad)
    torch.save({"backbone_state_dict": {**model.state_dict(), "backbone.weight": model.weight}}, bad)
    with pytest.raises(RuntimeError, match="multiple source"):
        runner.load_frozen_backbone(model, bad)
    torch.save({"backbone_state_dict": {key: value.half() for key, value in model.state_dict().items()}}, bad)
    with pytest.raises(RuntimeError, match="dtype"):
        runner.load_frozen_backbone(model, bad)
    torch.save({"backbone_state_dict": {key: value * float("nan") for key, value in model.state_dict().items()}}, bad)
    with pytest.raises(ValueError, match="non-finite"):
        runner.load_frozen_backbone(model, bad)


def test_prequantized_normals_share_representative_and_inverse_indices(runner):
    xyz = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.07, 0, 0], [-0.01, 0, 0]])
    normals = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 0, 0]])
    grid, representatives, inverse = runner._sparse_quantize(
        coordinates=xyz, quantization_size=0.05, return_index=True, return_inverse=True,
    )
    assert representatives.tolist() == [3, 0, 2]
    assert np.array_equal(grid[inverse], np.floor(xyz / 0.05))
    assert np.array_equal(normals[representatives][inverse], normals[[0, 0, 2, 3]])


def test_epoch_loader_seed_ignores_previous_episode_rng(runner):
    from torch.utils.data import DataLoader

    def sample_epoch(epoch):
        loader = DataLoader(
            range(25), batch_size=5, shuffle=True,
            generator=torch.Generator().manual_seed(runner.paired_training_seed(42, epoch, 0)),
        )
        return torch.cat(list(loader))

    expected = sample_epoch(5)
    torch.rand(1923)  # Different adaptive click path in the opposite arm.
    assert torch.equal(expected, sample_epoch(5))
    assert not torch.equal(expected, sample_epoch(6))
    runner.set_seed(runner.paired_training_seed(42, 5, 7))
    expected_objects = torch.randperm(20)
    torch.rand(301)
    runner.set_seed(runner.paired_training_seed(42, 5, 7))
    assert torch.equal(expected_objects, torch.randperm(20))


def test_stale_resume_cannot_overwrite_an_advanced_attempt(runner, tmp_path):
    from delimit3d.tasks.agile3d.trainer import validate_resume_frontier
    config = pair()[1]
    payload = {"epoch": 1, "step": 480, "lr_scheduler": {"last_epoch": 2}}
    resume = tmp_path / "staged_resume.pth"
    resume.write_bytes(b"checkpoint_epoch2")
    validate_resume_frontier(tmp_path, resume, payload, steps_per_epoch=240)
    latest = tmp_path / "checkpoints/checkpoint.pth"
    latest.parent.mkdir()
    latest.write_bytes(b"checkpoint_epoch3")
    with pytest.raises(ValueError, match="stale resume"):
        validate_resume_frontier(tmp_path, resume, payload, steps_per_epoch=240)
    latest.write_bytes(resume.read_bytes())
    (tmp_path / "metrics.jsonl").write_text('{"epoch":3,"step":720}\n')
    with pytest.raises(ValueError, match="metrics extend"):
        validate_resume_frontier(tmp_path, resume, payload, steps_per_epoch=240)
    payload["lr_scheduler"]["last_epoch"] = 1
    with pytest.raises(ValueError, match="scheduler epoch"):
        validate_resume_frontier(tmp_path, resume, payload, steps_per_epoch=240)


def test_checkpoint_save_rejects_silent_frozen_buffer_mutation(runner, monkeypatch, tmp_path):
    from delimit3d.tasks.agile3d.trainer import save_checkpoint
    config = pair()[1]
    model = fake_model(runner, monkeypatch, config)
    model.backbone.bn.running_mean.add_(1)
    with pytest.raises(RuntimeError, match="buffers changed"):
        save_checkpoint(tmp_path / "invalid.pth", model, None, None, 0, 240, config)
    assert not (tmp_path / "invalid.pth").exists()
