"""Budgets, matched-arm identity and immutable-input rejection."""
from argparse import Namespace
import json
from pathlib import Path

import pytest

from delimit3d.runtime.config import load_config
from delimit3d.tasks import limited_supervision as limited
from delimit3d.tasks.contracts import model_contract


@pytest.fixture(autouse=True)
def fixture_data_identity(monkeypatch):
    # Manifest tests use stand-in files; array-level identity is covered by
    # task data-identity and sparse-preparation tests.
    monkeypatch.setattr(limited, "data_identity", lambda config: {"fixture": "ordered data"})


def args(tmp_path, **kwargs):
    for name in ("encoder.pt", "decoder.pt"):
        (tmp_path / name).write_bytes(name.encode())
    for name in ("litept", "scans", "cache"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "litept/model.py").write_text("# synthetic dependency\n")
    return Namespace(protocol="low-data", percent=5, budget=None,
        output=tmp_path / "experiment", train_split=limited.ROOT / "configs/splits/scannet_train.txt",
        val_split=limited.ROOT / "configs/splits/scannet_val.txt", backbone_checkpoint=tmp_path / "encoder.pt",
        checkpoint_label="synthetic_fixture", scratch_checkpoint=None, decoder_template=tmp_path / "decoder.pt",
        litept_root=tmp_path / "litept", scans_root=tmp_path / "scans", gt_cache_root=tmp_path / "cache",
        prepared_root=None, **kwargs)


def test_checked_in_nested_subsets_match_actual_selection_and_exposure():
    pool = limited.scenes(limited.ROOT / "configs/splits/scannet_train.txt")
    previous = []
    for percent, count, updates in ((1,12,1800),(5,60,7200),(10,120,14400),(20,240,28800)):
        selected = limited.select_subset(pool, percent)
        assert len(selected) == count
        assert selected[:len(previous)] == previous
        assert selected == limited.scenes(limited.ROOT / f"configs/splits/low_data/scannet_{percent}pct_seed42.txt")
        assert ((count + 4) // 5) * 600 == updates
        previous = selected


def test_preparation_keeps_pair_matched_and_protects_inputs(tmp_path):
    a = args(tmp_path)
    m = limited.prepare(a)
    assert m["exposure"]["updates_total"] == 7200
    scratch = load_config(a.output / "configs/scratch.yaml")
    trained = load_config(a.output / "configs/pretrained.yaml")
    assert scratch["initialization"]["backbone_checkpoint"] is None
    assert trained["initialization"]["backbone_checkpoint"] == str(a.backbone_checkpoint)
    assert scratch["train"]["backbone_lr"] == trained["train"]["decoder_lr"] == .0001
    assert scratch["initialization"]["decoder_template"] == trained["initialization"]["decoder_template"]
    with pytest.raises(FileExistsError, match="fresh"):
        limited.prepare(a)
    a.backbone_checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="Input changed"):
        limited.verify(a.output)


def test_modified_recipe_is_rejected(tmp_path):
    a = args(tmp_path)
    limited.prepare(a)
    config = a.output / "configs/scratch.yaml"
    config.write_text(config.read_text().replace("backbone_lr: 0.0001", "backbone_lr: 0.00001"))
    with pytest.raises(ValueError, match="Sealed file changed"):
        limited.verify(a.output)


def test_changed_training_data_is_rejected(tmp_path, monkeypatch):
    a = args(tmp_path)
    limited.prepare(a)
    monkeypatch.setattr(limited, "data_identity", lambda config: {"fixture": "changed geometry"})
    with pytest.raises(ValueError, match="data changed"):
        limited.verify(a.output)


def test_wrong_scene_universe_is_rejected_before_outputs(tmp_path):
    a = args(tmp_path)
    a.train_split = tmp_path / "train.txt"
    ids = limited.scenes(limited.ROOT / "configs/splits/scannet_train.txt")
    a.train_split.write_text("\n".join([*ids[:-1], "scene9999_99"]) + "\n")
    with pytest.raises(ValueError, match="official"):
        limited.prepare(a)
    assert not a.output.exists()


def test_sparse_and_dense_mask_domains_cannot_be_interchanged():
    dense = load_config(limited.ROOT / "configs/tasks/mask3d_low_data.yaml", expand_env=False)
    sparse = load_config(limited.ROOT / "configs/tasks/mask3d_low_annotation.yaml", expand_env=False)
    assert model_contract(dense) != model_contract(sparse)
    assert model_contract(sparse)["mask_domain"] == "geometry_voxels"


def test_evaluation_rejects_other_arm_or_budget_checkpoint(tmp_path):
    from delimit3d.runtime.records import config_sha256
    a = args(tmp_path)
    manifest = limited.prepare(a)
    scratch = load_config(a.output / "configs/scratch.yaml")
    trained = load_config(a.output / "configs/pretrained.yaml")
    payload = {"baseline_config_sha256": config_sha256(scratch), "data_identity": manifest["data_identity"],
               "model_contract": model_contract(scratch)}
    limited.validate_checkpoint(payload, scratch, manifest)
    with pytest.raises(ValueError, match="different arm"):
        limited.validate_checkpoint(payload, trained, manifest)
    scratch["data"]["train_split"] = "different_budget.txt"
    with pytest.raises(ValueError, match="different arm"):
        limited.validate_checkpoint(payload, scratch, manifest)


def test_changed_external_backbone_source_is_rejected(tmp_path):
    a = args(tmp_path)
    limited.prepare(a)
    (tmp_path / "litept/model.py").write_text("# changed behavior\n")
    with pytest.raises(ValueError, match="dependency sources changed"):
        limited.verify(a.output)


def test_resume_after_validation_failure_accepts_newer_numbered_checkpoint(tmp_path):
    import torch
    from delimit3d.tasks.cli import validate_resume_frontier
    common = {"baseline_config_sha256": "same_recipe", "data_identity": {"same": "data"},
              "model_contract": {"same": "model"}, "initialization": {"same": "encoder"}}
    latest, numbered = tmp_path / "latest.pt", tmp_path / "epoch_0010.pt"
    torch.save({**common, "epoch": 9}, latest)
    pending = {**common, "epoch": 10, "validation_completed": False}
    torch.save(pending, numbered)
    validate_resume_frontier(latest, numbered, pending)
    validate_resume_frontier(latest, numbered, {**pending,
        "initialization": {**common["initialization"], "resume": {"completed_epoch": 9}}})
    with pytest.raises(ValueError, match="different experiment"):
        validate_resume_frontier(latest, numbered, {**pending, "data_identity": {"different": "data"}})
    torch.save({**common, "epoch": 11}, latest)
    with pytest.raises(ValueError, match="newer checkpoint"):
        validate_resume_frontier(latest, numbered, pending)
