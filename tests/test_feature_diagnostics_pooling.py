"""Scientific invariants for the full ScanNet200 clustering extension."""
import importlib.util
from pathlib import Path

import numpy as np


from delimit3d.evaluation import feature_diagnostics as suite


def test_pooling_uses_original_point_multiplicity():
    # Three original points in one cell: two inherit token 0, one token 1.
    result, bijective = suite.pool_features(
        np.array([[3., 0.], [0., 6.]], dtype=np.float32),
        np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([3]))
    np.testing.assert_allclose(result, [[2., 2.]])
    assert not bijective


def test_bijective_support_preserves_descriptors_in_common_order():
    result, bijective = suite.pool_features(
        np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        np.array([1, 0, 1]), np.array([0, 1, 0]), np.array([2, 1]))
    np.testing.assert_array_equal(result, [[3., 4.], [1., 2.]])
    assert bijective


def test_restart_selection_ignores_ground_truth_scores():
    assert suite.select_restart([
        {'mean_cosine_to_center': .5, 'gt_iou': 1.},
        {'mean_cosine_to_center': .8, 'gt_iou': 0.},
        {'mean_cosine_to_center': .7, 'gt_iou': .9}]) == 1


def test_large_object_subset_does_not_remove_small_objects_from_primary():
    objects = [{'vertices': 5, 'iou': 0., 'precision': 0., 'recall': 0.,
                'class_index': 1, 'class_id': 2, 'class_name': 'chair'},
               {'vertices': 100, 'iou': 1., 'precision': 1., 'recall': 1.,
                'class_index': 3, 'class_id': 4, 'class_name': 'table'}]
    result = suite.aggregate([{'per_object': objects, 'macro_iou': .5,
                              'macro_precision': .5, 'macro_recall': .5, 'micro_iou': .9}])
    assert result['objects'] == 2 and result['pooled_object_mean_iou'] == .5
    assert result['objects_at_least_100_vertices']['objects'] == 1
    assert result['objects_at_least_100_vertices']['mean_iou'] == 1.
