"""Training-cap identity and the independent, fixed 500k evaluation protocol."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from delimit3d.data import sam2_source
from delimit3d.pretraining.arguments import parse_args
from delimit3d.pretraining.checkpoint import validate_negative_policy_resume
from delimit3d.pretraining.sources import scene_source_for_visit
from test_single_granularity_pretraining import _scene, REQUIRED

ROOT = Path(__file__).resolve().parents[1]


def test_presets_keep_training_caps_explicit():
    for preset, cap in [('sam2_dec0', 500000), ('sam2_selected_proposal_negatives', 100000)]:
        args = parse_args(['--config', str(ROOT / f'configs/pretrain/{preset}.yaml'), *REQUIRED,
                           '--fixed-eval-pack', '/unused/pack', '--source-audit', '/unused/audit',
                           '--code-manifest', '/unused/code'])
        assert args.sam2_maximum_raw_points == cap


def test_real_100k_visits_cover_scene_without_replacement():
    scene = _scene('sam2', 200004)
    scene = replace(scene, source_variant_id=sam2_source.fragment_policy(100000),
                    colors=np.ones((len(scene.points), 3)), normals=np.ones((len(scene.points), 3)))
    visits = [scene_source_for_visit(scene, epoch=e, global_seed=42) for e in range(4)]
    assert all(len(v.points) <= 100000 for v in visits)
    assert len({v.crop_id for v in visits}) == 4
    assert len(np.unique(np.concatenate([v.points[:, 0] for v in visits]))) == len(scene.points)
    assert visits[0].crop_id == scene_source_for_visit(scene, epoch=4, global_seed=42).crop_id
    from delimit3d.training.contracts import stable_seed
    partitions = sam2_source.geometry_fragments(scene.points, 100000)
    position = stable_seed(42, scene.scene_id, sam2_source.FRAGMENT_POLICY) % len(partitions)
    np.testing.assert_array_equal(visits[0].points, scene.points[partitions[position]])
    legacy = replace(scene, source_variant_id=sam2_source.FRAGMENT_POLICY)
    assert scene_source_for_visit(legacy, epoch=0, global_seed=42) is legacy


def test_resume_rejects_cap_changes_and_inconsistent_checkpoint():
    legacy = {'allow_shared_proposal_negatives': True}
    current = {**legacy, 'sam2_maximum_raw_points': 100000}
    for before, after in [(legacy, current), (current, legacy)]:
        with pytest.raises(ValueError, match='raw-point limit differs'):
            validate_negative_policy_resume(before, after)
    validate_negative_policy_resume(legacy, {**legacy, 'sam2_maximum_raw_points': 500000})
    validate_negative_policy_resume(current, current, current)
    with pytest.raises(ValueError, match='raw-point limit differs'):
        validate_negative_policy_resume(current, current, {'sam2_maximum_raw_points': 500000})
    historical = {**legacy, 'sam2_scene_visit_contract': {
        'maximum_raw_points': 500000, 'fragment_policy': sam2_source.fragment_policy(500000),
        'training_maximum_raw_points': 100000, 'training_fragment_policy': sam2_source.fragment_policy(100000)}}
    validate_negative_policy_resume(historical, current)
    with pytest.raises(ValueError, match='disagrees'):
        sam2_source.recorded_point_limit({**current, 'sam2_scene_visit_contract': {'maximum_raw_points': 500000}})


@pytest.mark.parametrize('bad', [True, 100000.0, 0, 100001, '100000'])
def test_invalid_cap_is_rejected(bad):
    with pytest.raises(ValueError):
        sam2_source.fragment_policy(bad)


def test_fixed_pack_rejects_different_cap(tmp_path):
    from delimit3d.pretraining.fixed_eval import prepare_fixed_holdout_eval_pack
    from delimit3d.pretraining.settings import SamplingConfig
    kwargs = dict(pack_dir=tmp_path/'fixed', scenes=[_scene('sam2')],
        sampling=SamplingConfig(proposals_per_forward=4, positive_pairs_per_proposal=3,
            uniform_negatives=2, spatial_hard_negatives=0, feature_hard_negatives=0, feature_candidate_pool=2),
        sampling_profile='test', seed=42, cells=('2d/sam2',), grid_size=.02)
    assert prepare_fixed_holdout_eval_pack(**kwargs)['passed']
    manifest = json.loads((kwargs['pack_dir']/'manifest.json').read_text())
    assert manifest['sam2_maximum_raw_points'] == 500000
    with pytest.raises(ValueError, match='raw-point limit differs'):
        prepare_fixed_holdout_eval_pack(**kwargs, sam2_maximum_raw_points=100000)


def test_fixed_eval_cli_keeps_500k_for_100k_training_recipe(tmp_path, monkeypatch):
    from delimit3d.data.cli import fixed_eval_main
    import delimit3d.pretraining.sources as sources
    import delimit3d.pretraining.fixed_eval as fixed
    import delimit3d.training.sam2_pretraining as admission
    split = tmp_path/'heldout.txt'; split.write_text('scene_00000\n')
    calls = {}
    def load(path, **kw):
        calls['loaded_cap'] = kw['sam2_maximum_raw_points']
        return _scene('sam2')
    def visit(scene, **kw):
        calls['visit'] = kw
        return scene
    def admit(scenes, **kw):
        assert calls['visit'] == {'epoch': 0, 'global_seed': 42}
        return scenes, {}
    def pack(**kw):
        calls['pack'] = kw
        return {'passed': True}
    monkeypatch.setattr(sources, 'load_scene_source', load)
    monkeypatch.setattr(sources, 'scene_source_for_visit', visit)
    monkeypatch.setattr(admission, 'admit_sam2_holdout', admit)
    monkeypatch.setattr(fixed, 'prepare_fixed_holdout_eval_pack', pack)
    argv = ['--source-root', str(tmp_path), '--holdout-split', str(split), '--output', str(tmp_path/'pack'),
        '--teacher', 'sam2', '--grid-size', '.02', '--pretraining-contract',
        str(ROOT/'configs/pretrain/sam2_selected_proposal_negatives.yaml')]
    fixed_eval_main(argv)
    assert calls['loaded_cap'] == calls['pack']['sam2_maximum_raw_points'] == 500000
    assert calls['pack']['allow_shared_proposal_negatives'] is True
    with pytest.raises(SystemExit):
        fixed_eval_main([*argv, '--sam2-maximum-raw-points', '100000'])
