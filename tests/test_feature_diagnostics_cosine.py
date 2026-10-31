from pathlib import Path
import importlib.util

import numpy as np
import pytest
import torch


from delimit3d.evaluation import feature_diagnostics as probes


def test_cosine_anchors_use_shared_cells_and_keep_background_assignments():
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    features /= features.norm(dim=1, keepdim=True)
    inverse = np.array([0, 0, 1, 2, 2, 2])
    queries = np.array([1, 2])
    scores, winner, cells = probes.cosine_probe(features, inverse, queries)
    assert np.array_equal(cells, [0, 1])
    assert np.array_equal(winner, [0, 1, 0])
    # The last three original points may be background; they stay assigned,
    # so the downstream full-scene metric counts their false positives.
    assert np.array_equal(winner[inverse], [0, 0, 1, 0, 0, 0])
    assert scores.shape == (3, 2)


def test_distinct_gt_queries_in_one_cell_are_retained_with_stable_ties():
    features = torch.tensor([[1., 0.], [0., 1.]])
    scores, winner, cells = probes.cosine_probe(features, np.array([0, 0, 1]), np.array([0, 1]))
    assert np.array_equal(cells, [0, 0])
    assert np.array_equal(scores[:, 0], scores[:, 1])
    assert np.array_equal(winner, [0, 0])


def test_geometry_comparison_includes_original_order_and_grid_mapping():
    data = {key: np.arange(6).reshape(2, 3) for key in
            ('xyz', 'rgb255', 'normal', 'shift', 'common_grid', 'point_to_cell')}
    changed = {key: value.copy() for key, value in data.items()}
    probes.shared_geometry(data, changed)
    changed['point_to_cell'] = changed['point_to_cell'][::-1]
    with pytest.raises(ValueError, match='point_to_cell'):
        probes.shared_geometry(data, changed)
