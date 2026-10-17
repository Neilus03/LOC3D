"""Contract checks for click mapping and real inference preprocessing."""
import numpy as np
import pytest

from delimit3d.inference.scene import Scene
from delimit3d.inference.backends import prepare_features, prepare_clicks, quantize_interactive, _checkpoint_state


def test_task_features_preserve_normal_frame_and_normalize_dark_uint8_rgb():
    scene = Scene(np.array([[1., 2., 3.], [2., 3., 4.]]),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0., 0., 1.], [0., -1., 0.]]))
    result = prepare_features(scene, channels=6)
    np.testing.assert_allclose(result[:, :3], scene.colors / 255.)
    np.testing.assert_array_equal(result[:, 3:], scene.normals)
    with pytest.raises(ValueError, match="surface normals"):
        prepare_features(Scene(scene.points, scene.colors), channels=6)


def test_agile_quantization_maps_full_points_to_lexicographic_first_representatives():
    xyz = np.array([[5.11, -1., 2.], [5., -1., 2.], [5.01, -1., 2.]], dtype=np.float32)
    shifted, grid, first, inverse = quantize_interactive(xyz, .05)
    np.testing.assert_array_equal(first, [1, 0])
    np.testing.assert_array_equal(inverse, [1, 0, 0])
    np.testing.assert_array_equal(grid, [[0, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(shifted, (xyz - xyz.min(0))[first])


def test_clicks_map_original_point_indices_and_restore_user_object_identity():
    inverse = np.array([2, 0, 1, 1])
    clicks = [{"point_index": 0, "object_id": 7}, {"point_index": 3, "object_id": 2},
              {"point_index": 1, "object_id": 7, "positive": False}]
    indices, times, user_ids = prepare_clicks(clicks, inverse, 200)
    np.testing.assert_array_equal(user_ids, [2, 7])
    assert indices == {1: [1], 2: [2], 0: [0]}
    assert times == {1: [1], 2: [0], 0: [2]}
    for invalid in ([], [{"point_index": 4, "object_id": 1}], [{"point_index": 1, "object_id": 0}]):
        with pytest.raises(ValueError):
            prepare_clicks(invalid, inverse, 200)
    with pytest.raises(ValueError, match="at most"):
        prepare_clicks(clicks, inverse, 2)


def test_full_task_checkpoint_is_required_and_ddp_prefix_is_supported():
    expected = {"backbone.weight": object(), "decoder.weight": object()}
    assert _checkpoint_state({"model_state_dict": {"module." + k: v for k, v in expected.items()}}) == expected
    for payload in ({"backbone_state_dict": {}}, {"model": {"backbone.weight": 1}}, {"model": {"decoder.weight": 1}}):
        with pytest.raises(ValueError):
            _checkpoint_state(payload)


def test_automatic_export_retains_overlaps_while_display_labels_are_exclusive():
    from delimit3d.inference.backends import ModelRunner
    runner = object.__new__(ModelRunner)
    runner.torch = None
    runner.provenance = {}
    masks = np.array([[True, True, False, False], [False, True, True, False], [False, False, False, True]])
    runner._automatic = (masks, np.array([.6, .9, .1], dtype=np.float32), "provided_segments")
    scene = Scene(np.zeros((4, 3)), np.zeros((4, 3), dtype=np.uint8))
    result = runner._predict_automatic(scene, .3)
    np.testing.assert_array_equal(result["labels"], [2, 1, 1, 0])
    restored = np.unpackbits(result["packed_masks"], axis=0, count=int(result["mask_point_count"]))
    np.testing.assert_array_equal(restored, masks[[1, 0]].T)
    np.testing.assert_allclose(result["scores"], [.9, .6])
    empty = runner._predict_automatic(scene, 1.)
    assert not empty["labels"].any()
    assert empty["packed_masks"].shape == (1, 0)


def test_click_constraints_apply_before_inverse_expansion_and_latest_conflict_wins():
    from delimit3d.inference.backends import constrain_token_labels
    inverse = np.array([0, 0, 1, 2])
    clicks = [{"point_index": 0, "object_id": 7}, {"point_index": 1, "object_id": 2},
              {"point_index": 2, "object_id": 7}, {"point_index": 2, "object_id": 0}]
    labels = constrain_token_labels(np.array([0, 1, 2]), clicks, inverse, np.array([2, 7]))
    np.testing.assert_array_equal(labels[inverse], [1, 1, 0, 2])
