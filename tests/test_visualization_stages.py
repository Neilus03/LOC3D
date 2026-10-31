import numpy as np
import pytest

from delimit3d.visualization.adapters import load_observation
from delimit3d.visualization.contracts import Scene
from delimit3d.visualization.stages import split_feature_stages
from delimit3d.visualization.feature_research import fit_frozen_pca


def test_stage_normalization_is_explicit_and_preserves_original_input():
    features = np.array([[3., 4., 0., 2.], [0., 2., 3., 4.]], dtype=np.float32)
    saved = features.copy()
    raw = split_feature_stages(features, {"fine": 2, "coarse": 2})
    normalized = split_feature_stages(features, {"fine": 2, "coarse": 2}, normalization="per_stage_l2")
    np.testing.assert_array_equal(np.concatenate(list(raw.values()), axis=1), saved)
    for layer in normalized.values():
        np.testing.assert_allclose(np.linalg.norm(layer, axis=1), 1)
    np.testing.assert_array_equal(features, saved)
    assert not np.array_equal(raw["fine"], normalized["fine"])


@pytest.mark.parametrize("widths", [{}, {"a": 1}, {"a": 5}, {"a": True, "b": 3}, {"a": 0, "b": 4}])
def test_stage_widths_cannot_silently_drop_or_reuse_channels(widths):
    with pytest.raises(ValueError):
        split_feature_stages(np.ones((3, 4)), widths)


def test_stage_norm_rejects_undefined_directions():
    with pytest.raises(ValueError, match="zero"):
        split_feature_stages(np.array([[0., 0., 1., 1.]]), {"a": 2, "b": 2}, normalization="per_stage_l2")


def test_pca_state_cannot_be_reused_across_stage_normalization_variants():
    source = {"source_id": "same_checkpoint", "feature_level": "fine", "feature_stages": {
        "normalization": "as_saved", "widths": {"fine": 2, "coarse": 2}}}
    features = np.array([[3., 4.], [0., 2.], [2., 3.]])
    state = fit_frozen_pca(features, original_point_ids=np.arange(3), source_record=source)
    other = {**source, "feature_stages": {**source["feature_stages"], "normalization": "per_stage_l2"}}
    with pytest.raises(ValueError, match="feature_stages"):
        state.rgb(features, source_record=other)


def test_stage_cache_expands_exact_cell_mapping_and_records_normalization(tmp_path):
    points = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    inverse = np.array([1, 0, 1])
    features = np.array([[3., 4., 0., 2.], [0., 2., 3., 4.]])
    np.savez(tmp_path / "cache.npz", points=points, features=features, inverse_map=inverse)
    scene = Scene("fixture", points)
    spec = {"kind": "cache", "path": "cache.npz", "feature_stages": {
        "widths": {"fine": 2, "coarse": 2}, "normalization": "per_stage_l2",
        "source_normalization": "joint_l2"}}
    observation = load_observation(spec, scene, tmp_path)
    np.testing.assert_allclose(observation.features["fine"], [[0., 1.], [.6, .8], [0., 1.]])
    assert observation.metadata["feature_stages"]["source_normalization"] == "joint_l2"
    with pytest.raises(ValueError, match="Choose features or feature_stages"):
        load_observation({**spec, "features": {"dec0": "features"}}, scene, tmp_path)
