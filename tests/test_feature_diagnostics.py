"""Guard metric support, target/cache identity and completion semantics."""
import json

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from delimit3d.evaluation.feature_diagnostics import evaluate_features, match_clusters, metrics, pool_features
from delimit3d.evaluation.feature_diagnostics_cli import main
from delimit3d.runtime.records import file_sha256


@pytest.fixture(autouse=True, scope='module')
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_scoring_counts_background_leakage_and_hungarian_uses_full_support():
    truth = np.array([0, 0, 1, 1, -1, -1])
    prediction = np.array([1, 1, 0, 0, 1, 1])
    objects = [{'index': 0, 'vertices': 2}, {'index': 1, 'vertices': 2}]
    aligned, mapping, _ = match_clusters(prediction, truth, objects)
    assert mapping.tolist() == [1, 0]
    measured = metrics(aligned, truth, objects)
    assert measured['macro_iou'] == .75
    assert measured['background_assigned_points'] == 2
    assert measured['per_object'][0]['false_positive_points'] == 2


def test_complete_evaluation_retains_background_and_selects_feature_objective():
    features = np.array([[1., 0.], [0., 1.], [1., 0.]], dtype=np.float32)
    inverse = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([0, 0, 1, 1, -1, -1])
    objects = [{'index': 0, 'vertices': 2}, {'index': 1, 'vertices': 2}]
    arrays, record = evaluate_features(features, inverse, inverse, labels, np.array([0, 2]), objects)
    assert record['common_cells'] == 3
    assert len(record['kmeans_restarts']) == 5
    assert all('common occupied cells' in r['fit_scope'] for r in record['kmeans_restarts'])
    expected = max(range(5), key=lambda i: record['kmeans_restarts'][i]['mean_cosine_to_center'])
    assert record['selected_restart'] == expected
    assert record['metrics']['cosine']['macro_iou'] == .75
    assert record['metrics']['kmeans']['background_assigned_points'] == 2
    np.testing.assert_array_equal(arrays['prediction'][arrays['inverse_map']], arrays['point_prediction'])
    with pytest.raises(ValueError, match='Queries'):
        evaluate_features(features, inverse, inverse, labels, np.array([2, 0]), objects)


@pytest.mark.parametrize('native,common,counts', [
    ([0, 2], [0, 0], [2]),
    ([0, 1], [0, 0], [1]),
    ([0, 1], [0, 2], [1, 0, 1]),
    ([0., 1.], [0, 0], [2]),
])
def test_pooling_rejects_invalid_support(native, common, counts):
    with pytest.raises(ValueError):
        pool_features(np.eye(2), np.array(native), np.array(common), np.array(counts))


def test_pooling_rejects_complex_features_instead_of_discarding_imaginary_parts():
    with pytest.raises(ValueError, match='finite nonempty'):
        pool_features(np.array([[1 + 2j]]), np.array([0]), np.array([0]), np.array([1]))


def test_cli_seals_targets_checks_cache_order_and_emits_verifiable_output(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    points = np.array([[0., 0., 0.], [.01, 0., 0.], [.1, 0., 0.], [.11, 0., 0.], [.2, 0., 0.]], dtype=np.float32)
    for name, values in {'coord': points, 'color': np.full((5, 3), 127, np.uint8),
                         'normal': np.tile([0., 0., 1.], (5, 1)).astype(np.float32),
                         'segment20': np.array([4, 4, 4, 4, -1]),
                         'instance': np.array([0, 0, 1, 1, -1])}.items():
        np.save(source / f'{name}.npy', values)
    vertices = np.empty(5, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i, name in enumerate(('x', 'y', 'z')):
        vertices[name] = points[:, i]
    mesh = tmp_path / 'mesh.ply'
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(mesh)
    target_dir = tmp_path / 'targets'
    main(['prepare', '--taxonomy', 'scannet20', '--scene', 'synthetic', '--source', str(source),
          '--mesh', str(mesh), '--output', str(target_dir)])
    features = tmp_path / 'features.npz'
    native = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.]], dtype=np.float32)
    np.savez(features, points=points[::-1], features=native, original_to_token=np.arange(5))
    provenance = tmp_path / 'features.json'
    provenance.write_text(json.dumps({'model': 'synthetic', 'checkpoint': 'synthetic fixture',
                                     'checkpoint_sha256': 'f' * 64, 'epoch': 5}))
    output = tmp_path / 'result'
    args = ['evaluate', '--targets', str(target_dir / 'target.json'), '--features', str(features),
            '--feature-provenance', str(provenance), '--normal-source', 'synthetic_unit_z', '--output', str(output)]
    with pytest.raises(ValueError, match='vertex order'):
        main(args)
    assert not output.exists()
    np.savez(features, points=points, features=native, original_to_token=np.arange(5))
    main(args)
    marker = json.loads((output / 'COMPLETE.json').read_text())
    assert marker['coverage'] == {'evaluated_scenes': 1, 'full_validation_completion_claimed': False}
    for name, expected in marker['artifacts'].items():
        assert file_sha256(output / name) == expected
    record = json.loads((output / 'metrics.json').read_text())
    assert record['feature_provenance']['frozen_state_independently_verified_by_this_command'] is False
    assert record['normal_source'] == 'synthetic_unit_z'
    with np.load(output / 'diagnostics.npz') as data:
        assert data['features'].shape[0] == data['prediction'].shape[0]
        assert data['points'].shape[0] == data['point_prediction'].shape[0] == 5
        assert data['checkpoint_sha256'].item() == 'f' * 64
        exported = json.loads(data['experiment_record_json'].item())
        assert exported['normal_source'] == 'synthetic_unit_z'
        assert exported['epoch'] == 5
    with pytest.raises(FileExistsError):
        main(args)
