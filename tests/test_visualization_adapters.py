"""Alignment, format and inference-contract tests; no pretrained GPU model needed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from delimit3d.visualization.adapters import (
    _strict_backbone_state,
    geometry_sha256,
    load_observation,
    load_scene,
    observation_from_mask3d_output,
)
from delimit3d.visualization.contracts import Scene


@pytest.fixture
def scene() -> Scene:
    return Scene("tiny", np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0]], dtype=np.float32))


def test_scene_npz_normalizes_rgb_and_records_hashes(tmp_path: Path) -> None:
    points = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    colors = np.array([[0, 128, 255], [255, 0, 12]], dtype=np.uint8)
    np.savez(tmp_path / "scene.npz", points=points, colors=colors, labels=[1, 2])
    result = load_scene({"path": "scene.npz", "background_ids": [0]}, tmp_path)
    np.testing.assert_allclose(result.colors, colors / 255)
    assert result.metadata["geometry_sha256"] == geometry_sha256(points)
    assert len(result.metadata["source_files"]) == 1
    assert result.metadata["background_ids"] == [0]


def test_training_pack_expands_config_relative_environment(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    np.save(root / "coords.npy", np.ones((3, 3), dtype=np.float32))
    np.save(root / "instances.npy", np.array([1, 1, 2]))
    monkeypatch.setenv("VIZ_TEST_PACK", "pack")
    result = load_scene({"path": "$VIZ_TEST_PACK", "fields": {"labels": "instances"}}, tmp_path)
    assert result.labels.tolist() == [1, 1, 2]
    assert len(result.metadata["source_files"]) == 2


def test_packed_npy_explicit_columns_and_scale(tmp_path: Path) -> None:
    data = np.array([[100, 0, 0, 255, 0, 0, 4], [0, 100, 0, 0, 128, 255, 7]], dtype=np.float64)
    np.save(tmp_path / "room.npy", data)
    result = load_scene({"path": "room.npy", "color_range": "0_255", "coordinate_scale": 0.01,
        "fields": {"points": {"key": "data", "columns": [0, 1, 2]},
                   "colors": {"key": "data", "columns": [3, 4, 5]},
                   "labels": {"key": "data", "columns": 6}}}, tmp_path)
    assert result.labels.tolist() == [4, 7]
    assert result.points[0, 0] == 1
    assert result.colors[0, 0] == 1


def test_noncanonical_rgb_requires_explicit_range(tmp_path: Path) -> None:
    np.savez(tmp_path / "scene.npz", points=np.zeros((1, 3)), colors=np.array([[25., 50., 200.]]))
    with pytest.raises(ValueError, match="color_range"):
        load_scene({"path": "scene.npz"}, tmp_path)


def test_cached_tokens_expand_all_outputs_in_original_order(tmp_path: Path, scene: Scene) -> None:
    inverse = np.array([0, 0, 1])
    np.savez(tmp_path / "cache.npz", points=scene.points, features=[[1., 0], [0, 1]],
             inverse_map=inverse, representative_indices=[0, 2], scene_xyz=scene.points[[0, 2]],
             prediction=[4, 8], confidence=[.9, .7], steps=[[0, 8], [4, 8]])
    result = load_observation({"path": "cache.npz"}, scene, tmp_path)
    assert result.prediction.tolist() == [4, 4, 8]
    assert result.steps.tolist() == [[0, 0, 8], [4, 4, 8]]
    np.testing.assert_array_equal(result.features["dec0"], [[1, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(result.metadata["token_ids"]["dec0"], inverse)


def test_point_features_need_order_witness_even_when_counts_match(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "cache.npz", features=np.ones((3, 4)))
    with pytest.raises(ValueError, match="Cache requires matching ordered points"):
        load_observation({"path": "cache.npz"}, scene, tmp_path)


def test_cache_rejects_reordered_scene(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "cache.npz", points=scene.points[::-1], features=np.ones((3, 4)))
    with pytest.raises(ValueError, match="geometry/order"):
        load_observation({"path": "cache.npz"}, scene, tmp_path)


def test_cache_canonical_geometry_hash(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "cache.npz", source_geometry_sha256=geometry_sha256(scene.points),
             features=np.ones((3, 4)))
    result = load_observation({"path": "cache.npz"}, scene, tmp_path)
    assert result.metadata["alignment_check"] == "source_geometry_sha256"


def test_shifted_legacy_cache(tmp_path: Path, scene: Scene) -> None:
    shift = np.array([0.5, 0, 0], dtype=np.float32)
    np.savez(tmp_path / "cache.npz", shifted_points=scene.points - shift, shift=shift,
             points=3, features=np.ones((3, 2)))
    result = load_observation({"path": "cache.npz"}, scene, tmp_path)
    assert result.metadata["alignment_check"] == "shifted_points"


@pytest.mark.parametrize("inverse,reps,match", [
    ([0, 0, 2], [0, 2], "outside"),
    ([0., 0., 1.], [0, 2], "integer array"),
    ([0, 0, 1], [2, 0], "does not invert"),
])
def test_cache_rejects_bad_inverse_maps(tmp_path: Path, scene: Scene, inverse, reps, match) -> None:
    np.savez(tmp_path / "cache.npz", points=scene.points, features=np.ones((2, 4)),
             inverse_map=inverse, representative_indices=reps)
    with pytest.raises(ValueError, match=match):
        load_observation({"path": "cache.npz"}, scene, tmp_path)


def test_cache_checks_checkpoint_and_scene_identity(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "cache.npz", points=scene.points, features=np.ones((3, 4)),
             checkpoint_sha256="actual", scene="tiny")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_observation({"path": "cache.npz", "expected_checkpoint_sha256": "other"}, scene, tmp_path)


def test_declared_metadata_cannot_override_recorded_geometry(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "cache.npz", points=scene.points, features=np.ones((3, 4)))
    result = load_observation({"path": "cache.npz", "metadata": {
        "synthetic": True, "cache_sha256": "spoof", "confidence_semantics": "demo score",
    }}, scene, tmp_path)
    assert result.metadata["synthetic"] is True
    assert result.metadata["cache_sha256"] != "spoof"
    assert result.metadata["confidence_semantics"] == "demo score"


def test_callable_receives_full_scene_and_resolved_paths(tmp_path: Path, scene: Scene, monkeypatch) -> None:
    module = ModuleType("_viz_test_plugin")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"fixture")
    def infer(*, scene, config):
        assert len(scene.points) == 3
        assert Path(config["checkpoint"]).is_absolute()
        return {"prediction": np.array([1, 1, 2])}
    module.infer = infer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = load_observation({"kind": "callable", "callable": "_viz_test_plugin:infer",
                               "checkpoint": "weights.bin", "name": "Mask3D fixture"}, scene, tmp_path)
    assert result.name == "Mask3D fixture"
    assert len(result.metadata["checkpoint_sha256"]) == 64


def test_strict_checkpoint_mapping_rejects_collisions_and_missing_keys() -> None:
    expected = {"model.weight": np.ones((2, 3)), "model.bias": np.zeros(2)}
    source = {"module.backbone.weight": np.ones((2, 3)), "module.backbone.bias": np.zeros(2),
              "head.weight": np.zeros((4, 3))}
    assert set(_strict_backbone_state(source, expected)) == set(expected)
    with pytest.raises(ValueError, match="Duplicate"):
        _strict_backbone_state({**source, "model.weight": np.ones((2, 3))}, expected)
    with pytest.raises(ValueError, match="misses"):
        _strict_backbone_state({"weight": np.ones((2, 3))}, expected)


def test_litept_uses_explicit_training_preprocessing_on_full_scene(tmp_path: Path, scene: Scene, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    seen = {}
    class FakeBackbone(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.model = torch.nn.Linear(6, 2, bias=False)
        def forward(self, points, features):
            seen["points"] = points.detach().numpy()
            seen["features"] = features.detach().numpy()
            return SimpleNamespace(point_feat=self.model(features), inverse_map=torch.arange(len(points)))
    module = ModuleType("delimit3d.models.litept_wrapper")
    module.LitePTBackbone = FakeBackbone
    monkeypatch.setitem(sys.modules, module.__name__, module)
    torch.save({"state_dict": {"module.backbone.weight": torch.ones((2, 6))}}, tmp_path / "encoder.pt")
    scene.colors = np.ones((3, 3), dtype=np.float32)
    scene.normals = np.tile([2., 0., 0.], (3, 1)).astype(np.float32)
    result = load_observation({"kind": "litept", "checkpoint": "encoder.pt", "litept_root": ".",
        "grid_size": .05, "input_features": "rgbn6", "rgb_mean": [.5, .5, .5],
        "rgb_std": [.25, .25, .25], "center_shift": "center_xy_ground_z", "device": "cpu"}, scene, tmp_path)
    assert seen["grid_size"] == .05
    assert len(seen["points"]) == len(scene.points)
    np.testing.assert_array_equal(seen["features"], np.tile([2, 2, 2, 1, 0, 0], (3, 1)))
    assert result.features["dec0"].shape == (3, 2)
    assert result.metadata["preprocessing"]["grid_size"] == .05


def test_agile_decoder_replays_clicks_from_trusted_token_cache(tmp_path: Path, scene: Scene) -> None:
    torch = pytest.importorskip("torch")
    from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder
    kwargs = dict(feature_dim=4, hidden_dim=8, num_heads=2, dim_feedforward=16,
                  num_decoders=1, num_bg_queries=1, max_click_events=10, aux=False)
    decoder = Agile3DClickDecoder(**kwargs)
    torch.save({"decoder_kwargs": kwargs, "decoder_state_dict": decoder.state_dict()}, tmp_path / "decoder.pt")
    np.savez(tmp_path / "tokens.npz", points=scene.points, features=np.ones((2, 4)),
             scene_xyz=scene.points[[0, 2]], inverse_map=[0, 0, 1], representative_indices=[0, 2])
    result = load_observation({"kind": "agile3d", "checkpoint": "decoder.pt",
        "cache": {"path": "tokens.npz"},
        "clicks": [{"point_index": 0, "object_id": 9}, {"point_index": 2, "object_id": 2}]}, scene, tmp_path)
    assert result.steps.shape == (2, 3)
    assert result.prediction.tolist() == [9, 9, 2]
    assert result.metadata["adapter"] == "agile3d_compatible"
    assert result.metadata["confidence_calibrated"] is False
    assert result.metadata["prediction_background_ids"] == [0]


def test_mask3d_queries_drop_no_object_and_preserve_instance_ids(scene: Scene) -> None:
    from delimit3d.visualization.diagnostics import build_panels
    scene.labels = np.array([10, 0, 20])
    scene.metadata["background_ids"] = [0]
    output = {"pred_masks": [np.array([[10., -10, -10], [-10, 10, -10], [-10, -10, 10]])],
              "pred_logits": np.array([[[8., 0, -3], [0, -3, 8], [-3, 8, 0]]])}
    result = observation_from_mask3d_output(output, scene)
    assert result.prediction.tolist() == [1, 0, 3]
    assert result.confidence[1] == 0
    assert result.metadata["queries"][2]["semantic_index"] == 1
    assert result.metadata["prediction_background_ids"] == [0]
    panels, report = build_panels(scene, [result], query_indices=[], requested=["prediction"])
    assert panels
    assert report["predictions"]["Mask3D"]["prediction_kind"] == "instance"


def test_mask3d_cached_qn_layout_expands_inverse(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "mask3d.npz", points=scene.points, inverse_map=[0, 0, 1],
             mask_logits=[[8., -8], [-8, 8]], object_logits=[[8., -8], [8., -8]])
    result = load_observation({"kind": "mask3d", "path": "mask3d.npz", "mask_layout": "QN",
                               "mask_key": "mask_logits", "class_key": "object_logits"}, scene, tmp_path)
    assert result.prediction.tolist() == [1, 1, 2]
    assert result.metadata["alignment_check"] == "ordered_points"


def test_mask3d_needs_inverse_for_token_predictions(scene: Scene) -> None:
    with pytest.raises(ValueError, match="exact inverse_map"):
        observation_from_mask3d_output({"pred_masks": np.ones((2, 4)),
                                        "pred_logits": np.ones((4, 3))}, scene)
