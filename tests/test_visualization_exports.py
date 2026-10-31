"""Append-only export and exact correspondence checks using synthetic fixtures."""
import json
import hashlib

import numpy as np
import pytest

from delimit3d.visualization.episodes import load_episode
from delimit3d.visualization.export_contracts import (geometry_hash, pairing_status,
                                                     save_episode_export, save_proposal_export)
from delimit3d.visualization.proposals import load_proposals
from test_visualization_episodes import episode_fixture
from test_visualization_proposals import proposal_fixture


def test_episode_roundtrip_is_safe_npz_and_preserves_trace(tmp_path):
    scene, record, arrays = episode_fixture()
    output = save_episode_export(tmp_path / "episode", scene, record=record, arrays=arrays)
    episode = load_episode(output, scene)
    np.testing.assert_array_equal(episode.arrays["predictions"], arrays["predictions"])
    assert episode.record["clicks"] == record["clicks"]
    with pytest.raises(FileExistsError):
        save_episode_export(output.parent, scene, record=record, arrays=arrays)
    assert len(episode.record["source_record_sha256"]) == 64


def test_proposal_roundtrip_preserves_raw_overlap_and_query_ids(tmp_path):
    scene, record, arrays = proposal_fixture()
    path = save_proposal_export(tmp_path / "proposals", scene, record=record, arrays=arrays)
    proposals = load_proposals(path, scene)
    np.testing.assert_array_equal(proposals.masks, arrays["masks"])
    np.testing.assert_array_equal(proposals.scores, arrays["scores"])
    with (path.parent / "arrays.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_proposals(path, scene)


def test_exact_geometry_hash_does_not_collapse_float64_points():
    first = np.array([[1., 2., 3.]], dtype=np.float64)
    second = first.copy()
    second[0, 0] += 1e-10
    np.testing.assert_array_equal(first.astype(np.float32), second.astype(np.float32))
    assert geometry_hash(first) != geometry_hash(second)


def test_export_rejects_wrong_scene_or_arrays_and_does_not_leave_outputs(tmp_path):
    scene, record, arrays = episode_fixture()
    arrays["points"] = scene.points[::-1]
    with pytest.raises(ValueError, match="correspondence"):
        save_episode_export(tmp_path / "bad", scene, record=record, arrays=arrays)
    assert not (tmp_path / "bad").exists()


def test_pairing_preprocessing_expected_contract_and_incomplete_report():
    preprocessing = {"grid_size": .02, "channels": "rgbn6"}
    digest = hashlib.sha256(json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model = {"encoder": {"checkpoint": "encoder.pt", "sha256": "a"},
             "decoder": {"checkpoint": "decoder.pt", "sha256": "b"},
             "decoder_trained_with_encoder_sha256": "a", "preprocessing": preprocessing,
             "preprocessing_sha256": digest}
    assert pairing_status(model)["status"] == "record_complete"
    assert "not been checked" in pairing_status(model)["verification"]
    assert pairing_status({})["status"] == "incomplete"
    with pytest.raises(ValueError, match="Expected pairing mismatch"):
        pairing_status(model, {"encoder": {"sha256": "wrong"}})
    model["preprocessing"]["grid_size"] = .05
    with pytest.raises(ValueError, match="preprocessing_sha256"):
        pairing_status(model)
