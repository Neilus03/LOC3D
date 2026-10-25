"""Regression coverage for the one-way official sparse-annotation boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from delimit3d.tasks.mask3d import prepare_low_annotation as preparation


def _groups():
    return [
        {"objectId": 7, "label": "chair", "segments": [10]},
        {"objectId": 99, "label": "wall", "segments": [20]},
        {"objectId": 8, "label": "chair", "segments": [30]},
        {"objectId": 10, "label": "table", "segments": [40]},
    ]


def test_sparse_export_preserves_official_duplicates_order_and_hidden_invariance():
    indices = np.array([2, 0, 2, 1], dtype=np.int64)
    segments = [10, 20, 30, 40]
    labels = {"chair": 5, "wall": 1, "table": 7, "other": 13}
    first = preparation.select_annotations(indices, 4, segments, _groups(), labels)
    np.testing.assert_array_equal(first["vertex_indices"], indices)
    np.testing.assert_array_equal(first["instance_ids"], [1, 2, 1, 0])
    np.testing.assert_array_equal(first["semantic_ids"], [5, 5, 5, 1])
    changed = _groups()
    changed[-1] = {"objectId": 888, "label": "other", "segments": [40]}
    second = preparation.select_annotations(indices, 4, segments, changed, labels)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    unknown = preparation.select_annotations(np.array([3]), 4, segments, changed, labels)
    assert unknown["instance_ids"].tolist() == [-1]
    assert unknown["semantic_ids"].tolist() == [-1]


@pytest.mark.parametrize("indices", [np.array([-1]), np.array([4]), np.array([1.0]), np.array([[1]]),
                                      np.array([2**64 - 1], dtype=np.uint64)])
def test_invalid_official_indices_fail(indices):
    with pytest.raises(ValueError, match="Official indices|Out-of-range"):
        preparation.select_annotations(indices, 4, [10, 20, 30, 40], _groups(), {"chair": 5})


def test_conflicting_permitted_segment_rejected_but_hidden_conflict_unobserved():
    groups = _groups() + [{"objectId": 23, "label": "chair", "segments": [40]}]
    preparation.select_annotations(np.array([0]), 4, [10, 20, 30, 40], groups, {"chair": 5})
    with pytest.raises(ValueError, match="Contradictory annotation"):
        preparation.select_annotations(np.array([3]), 4, [10, 20, 30, 40], groups, {"chair": 5})


def test_archive_must_authenticate_before_deserialization(tmp_path):
    archive = tmp_path / "points.zip"
    archive.write_bytes(b"untrusted replacement archive")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        preparation._load_index_sets(archive, ["scene0000_00"], tmp_path)


def test_split_names_duplicates_and_membership(tmp_path, monkeypatch):
    split = tmp_path / "split.txt"
    for text in ("scene0000_00\nscene0000_00\n", "../scene0000_00\n"):
        split.write_text(text)
        with pytest.raises(ValueError):
            preparation.read_split(split)
    monkeypatch.setattr(preparation, "TRAIN_SCENES", 1)
    monkeypatch.setattr(preparation, "VALIDATION_SCENES", 1)
    with pytest.raises(ValueError, match="overlap"):
        preparation.validate_splits(["scene0000_00"], ["scene0000_00"])
    with pytest.raises(ValueError, match="membership"):
        preparation.validate_splits(["scene0000_00"], ["scene0001_00"])


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    """Only the official archive/cardinality boundary is replaced for this fixture."""
    train, val = "scene0000_00", "scene0001_00"
    monkeypatch.setattr(preparation, "TRAIN_SCENES", 1)
    monkeypatch.setattr(preparation, "VALIDATION_SCENES", 1)
    monkeypatch.setattr(preparation, "OFFICIAL_SPLIT_SHA256", {
        name: hashlib.sha256((scene + "\n").encode()).hexdigest()
        for name, scene in (("train", train), ("validation", val))
    })
    raw_root, pack_root = tmp_path / "raw", tmp_path / "packs"
    for scene in (train, val):
        raw, pack = raw_root / scene, pack_root / scene / "training_pack"
        raw.mkdir(parents=True)
        pack.mkdir(parents=True)
        mesh = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n2 0 0\n3 0 0\n"
        (raw / f"{scene}_vh_clean_2.ply").write_text(mesh)
        (raw / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(json.dumps({"segIndices": [10, 20, 30, 40]}))
        (raw / f"{scene}.aggregation.json").write_text(json.dumps({"segGroups": _groups()}))
        points = np.column_stack([np.arange(4), np.zeros(4), np.zeros(4)]).astype(np.float32)
        for name, array in (("points", points), ("colors", np.zeros((4, 3), dtype=np.uint8)),
                            ("normals", np.ones((4, 3), dtype=np.float32))):
            np.save(pack / f"{name}.npy", array)
        # Deliberately present prohibited pack extras: preparation must ignore them.
        np.save(pack / "instance_ids.npy", np.array([777, 888, 999, 123]))
        np.save(pack / "segment_ids.npy", np.array([10, 20, 30, 40]))
    train_split, val_split = tmp_path / "train.txt", tmp_path / "val.txt"
    train_split.write_text(train + "\n")
    val_split.write_text(val + "\n")
    label_map = tmp_path / "labels.tsv"
    label_map.write_text("raw_category\tnyu40id\nchair\t5\nwall\t1\ntable\t7\n")
    archive = tmp_path / "points.zip"
    archive.write_bytes(b"synthetic fixture, never passed to torch.load")

    def synthetic_archive(source, scenes, inputs):
        (inputs / "official_points.zip").write_bytes(source.read_bytes())
        result = {}
        for budget in preparation.BUDGETS:
            indices = np.resize(np.array([2, 0, 2, 1]), budget)
            result[budget] = {train: indices}
            (inputs / f"points{budget}").write_bytes(indices.tobytes())
        return result

    monkeypatch.setattr(preparation, "_load_index_sets", synthetic_archive)
    return dict(raw_root=raw_root, pack_root=pack_root, train_split=train_split, val_split=val_split,
                label_map=label_map, points_archive=archive, output=tmp_path / "prepared")


def test_end_to_end_exports_only_sparse_train_and_separate_dense_validation(synthetic_inputs):
    manifest = preparation.prepare(**synthetic_inputs)
    output = synthetic_inputs["output"]
    assert manifest["hidden_training_annotations_exported"] is False
    assert manifest["scene_proofs"][0]["budgets"]["20"]["duplicate_entries"] == 17
    with np.load(output / "geometry/scene0000_00.npz") as payload:
        assert set(payload.files) == {"points", "colors", "normals"}
    with np.load(output / "train_annotations/20/scene0000_00.npz") as payload:
        assert set(payload.files) == {"vertex_indices", "instance_ids", "semantic_ids"}
        assert payload["instance_ids"].shape == (20,)
        assert payload["vertex_indices"].tolist()[:4] == [2, 0, 2, 1]
    assert not (output / "validation/scene0000_00.npz").exists()
    assert not (output / "train_annotations/20/scene0001_00.npz").exists()
    with np.load(output / "validation/scene0001_00.npz") as payload:
        assert payload["instance_ids"].tolist() == [8, -1, 9, 11]
        assert payload["instance_class_pairs"].tolist() == [[8, 4], [9, 4], [11, 6]]
    assert preparation.verify_prepared(output)["schema"] == preparation.SCHEMA
    with pytest.raises(FileExistsError, match="immutable"):
        preparation.prepare(**synthetic_inputs)


def test_original_mesh_order_required_and_failure_is_atomic(synthetic_inputs):
    path = synthetic_inputs["pack_root"] / "scene0000_00/training_pack/points.npy"
    np.save(path, np.load(path)[::-1].copy())
    with pytest.raises(ValueError, match="vertex ordering"):
        preparation.prepare(**synthetic_inputs)
    assert not synthetic_inputs["output"].exists()
    assert not list(synthetic_inputs["output"].parent.glob(".prepared.prepare-*"))


@pytest.mark.parametrize("corruption", ["artifact", "manifest", "extra_dense_labels"])
def test_seal_rejects_tampering_and_accidental_dense_exports(synthetic_inputs, corruption):
    preparation.prepare(**synthetic_inputs)
    output = synthetic_inputs["output"]
    if corruption == "artifact":
        with (output / "geometry/scene0000_00.npz").open("ab") as handle:
            handle.write(b"corrupted")
    elif corruption == "manifest":
        with (output / "data_manifest.json").open("a") as handle:
            handle.write(" ")
    else:
        np.save(output / "train_annotations/dense_instance_ids.npy", np.arange(4))
    with pytest.raises(ValueError, match="changed|Unexpected"):
        preparation.verify_prepared(output)


def test_prepared_hashes_remain_verifiable_after_relocation(synthetic_inputs, tmp_path):
    import shutil

    preparation.prepare(**synthetic_inputs)
    moved = tmp_path / "moved"
    synthetic_inputs["output"].rename(moved)
    shutil.rmtree(synthetic_inputs["raw_root"])
    shutil.rmtree(synthetic_inputs["pack_root"])
    assert preparation.verify_prepared(moved)["train_scenes"] == 1
