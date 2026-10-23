"""Contract checks for frozen-mask classification and semantic AP conversion."""
import numpy as np

from delimit3d.evaluation import linear_instance_classifier as c

def test_pool_uses_original_vertex_weights_at_mixed_voxels():
    features = np.asarray([[1, 3], [1, 3], [10, -1], [90, 90]], dtype=np.float32)
    masks = [np.asarray([1, 1, 1, 0], dtype=bool), np.asarray([0, 0, 0, 1], dtype=bool)]
    value = c.pool_reference(features, masks)
    np.testing.assert_allclose(value[0], [4, 5/3, 10, 3])
    np.testing.assert_allclose(value[1], [90, 90, 90, 90])


def test_linear_predictions_are_independent_of_mask_order():
    x = np.asarray([[1, 2], [4, -1], [0, 0]])
    w = np.asarray([[2, 1], [-2, -1], [0, 0]])
    p = c.probabilities(x, w, np.zeros(3), np.zeros(2), np.ones(2))
    q = c.probabilities(x[::-1], w, np.zeros(3), np.zeros(2), np.ones(2))
    np.testing.assert_allclose(p, q[::-1]); np.testing.assert_allclose(p.sum(1), 1)


def test_matching_is_geometry_only_and_excludes_small_gt():
    pairs = c.geometry_matches(np.asarray([[0, 110, 0], [120, 0, 0]]), [110, 120], [120, 110, 5],
                               np.asarray([1, 1, 0], dtype=bool), np.asarray([1, 1], dtype=bool))
    assert pairs == [(0, 1, 1.0), (1, 0, 1.0)]


def test_ap_records_keep_false_positives_and_background_extent():
    value = c.official_match_record('s', [3001], [3], [120], [3, 5], [200, 130], [.8, .9],
                                    np.asarray([[120], [0]]), [80, 130])
    a = value['pred']['cabinet'][0]
    assert a['vert_count'] == 200 and a['void_intersection'] == 80
    assert a['matched_gt'][0]['intersection'] == 120
    assert value['pred']['chair'][0]['matched_gt'] == []
    assert len(value['gt']['cabinet'][0]['matched_pred']) == 1


def test_small_gt_retained_for_reference_ignore_rules():
    value = c.official_match_record('s', [3001], [3], [50], [3], [110], [.8], np.asarray([[50]]), [60])
    assert value['gt']['cabinet'][0]['vert_count'] == 50
    assert value['pred']['cabinet'][0]['matched_gt'][0]['vert_count'] == 50


def test_empty_scene_has_all_18_classes_and_no_proposals():
    value = c.official_match_record('s', [], [], [], [], [], [], np.empty((0, 0)), [])
    assert len(value['gt']) == len(value['pred']) == 18
    assert not any(value['gt'].values()) and not any(value['pred'].values())
