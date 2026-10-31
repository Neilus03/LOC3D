import numpy as np
import pytest
import torch

from delimit3d.data.contrastive_sampler_v2 import FrameProposalCatalog, sample_multigranular_frame_group_plan
from delimit3d.data.sam2_source import FRAGMENT_POLICY, sam2_scene_for_visit
from delimit3d.losses.litept_hierarchy_supervision import build_token_pure_dec0_contrastive_batch
from delimit3d.training.adaptation import RawFrame, SceneSource
from delimit3d.training.sampling_ablation import negative_budget


@pytest.mark.parametrize('variant,before,after', [
    ('no_hard', (384, 0, 0), (512, 0, 0)),
    ('no_feature_hard', (256, 128, 0), (384, 128, 0)),
    ('overlap_veto', (256, 128, 0), (256, 128, 128)),
    ('no_small_positive', (256, 128, 0), (256, 128, 128)),
])
def test_stage_budgets(variant, before, after):
    for update, expected in [(6105, before), (6106, after)]:
        actual = negative_budget(variant, update=update)
        assert (actual.uniform, actual.spatial, actual.feature) == expected
        assert actual.total == sum(expected)


def test_filtered_positives_keep_complete_negative_relation():
    # Proposal 1 is excluded only as a positive proposal. It is the sole
    # teacher coverage for points 4..7, which must remain valid negatives.
    points = torch.zeros(8, 3)
    points[:, 0] = torch.arange(8) * .05
    catalog = FrameProposalCatalog(
        '2d/sam2', 'frame', torch.arange(8), torch.tensor([0, 4, 8]),
        torch.arange(8), positive_eligible=torch.tensor([True, False]))
    plan = sample_multigranular_frame_group_plan(
        scene_id='toy', epoch=0, seed=42, points=points,
        frames_by_cell={'2d/sam2': (catalog,)}, cells=('2d/sam2',),
        total_proposal_quota=1, positive_pairs_per_proposal=3,
        num_uniform_negatives=2, num_spatial_hard_negatives=0,
        num_feature_hard_negatives=0, require_hierarchy_routes=False,
        require_negative_proposal_membership=True,
        allow_shared_proposal_negatives=True, grid_size=.02)
    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert group.selected_proposal_indices == (0,)
    batch, _ = build_token_pure_dec0_contrastive_batch(group.batch, torch.arange(8), points)
    assert batch is not None
    assert torch.isin(batch.negative_indices, torch.tensor([4, 5, 6, 7])).all()


def test_fragmentation_preserves_2d_area_identity(monkeypatch):
    points = np.zeros((8, 3), dtype=np.float32)
    points[:, 0] = np.arange(8)
    frame = RawFrame('sam2', 'frame', 'train', np.arange(8), np.array([0, 4, 8]),
                     np.arange(8), np.array([31, 400]))
    scene = SceneSource(scene_id='toy', manifest_path=None, manifest_sha256='x',
                        points=points, features=np.zeros((8, 6), np.float32),
                        labels_by_key={}, raw_frames=(frame,),
                        colors=np.zeros((8, 3), np.float32), normals=np.zeros((8, 3), np.float32),
                        crop_mode='full_scene', source_variant_id=FRAGMENT_POLICY)
    monkeypatch.setattr("delimit3d.data.sam2_source.maximum_points_for_policy", lambda policy: 4)
    fragment = sam2_scene_for_visit(scene, epoch=0, global_seed=42)
    assert fragment.raw_frames[0].proposal_areas.tolist() == [31, 400]
    assert fragment.raw_frames[0].proposal_offsets.shape == (3,)


@pytest.mark.parametrize('update,activation', [(True, 2), (1, True), (-1, 2), (1, 0), (1.5, 2)])
def test_budget_rejects_invalid_boundaries(update, activation):
    with pytest.raises(ValueError):
        negative_budget('no_hard', update=update, activation_update=activation)


def test_resume_rejects_silent_arm_and_activation_changes():
    from delimit3d.pretraining.checkpoint import validate_negative_policy_resume
    from delimit3d.training.sampling_ablation import validate_ablation_resume
    arm = {'ablation_variant': 'no_hard', 'ablation_proposal_area_cutoff': None,
           'ablation_area_audit_sha256': None, 'ablation_activation_update': 6106,
           'ablation_training_allow_shared_negatives': True,
           'ablation_fixed_evaluation_allow_shared_negatives': True,
           'ablation_gate_policy': 'fixed_to_reference_actual_activation'}
    validate_ablation_resume({}, {})
    validate_ablation_resume(arm, dict(arm))
    for old, new in [(arm, {}), ({}, arm), (arm, {**arm, 'ablation_activation_update': 2})]:
        with pytest.raises(ValueError, match='Sampling ablation'):
            validate_negative_policy_resume(old, new)


def _area_fixture(tmp_path):
    import json
    from delimit3d.runtime.records import file_sha256
    from delimit3d.training.sampling_ablation_cli import audit_areas
    scene = tmp_path / 'sources' / 'scene_00000'; scene.mkdir(parents=True)
    cache = scene / 'cache.npz'
    # Heldout and non-trainable tiny masks must not enter the quartile.
    np.savez(cache, scene_id=np.asarray('scene_00000'), physical_frame_ids=['a', 'b'],
             split_codes=np.asarray([0, 1], np.int8), proposal_frame_offsets=np.asarray([0, 5, 9]),
             proposal_areas=np.asarray([10, 20, 20, 40, 1, 1, 1, 1, 1]),
             proposal_trainable=np.asarray([True, True, True, True, False, True, True, True, True]))
    ann = scene / 'annotation.json'
    ann.write_text(json.dumps({'frames': [{'frame_id': name, 'split': split, 'image_shape': [10, 10]}
                                         for name, split in [('a', 'train'), ('b', 'heldout')]]}))
    def record(path):
        return {'path': path.name, 'sha256': file_sha256(path), 'size_bytes': path.stat().st_size}
    manifest = scene / 'source_manifest.json'
    manifest.write_text(json.dumps({'scene_id': 'scene_00000', 'complete': True,
        'artifacts': {'sam2_projection_cache': record(cache), 'annotation_manifest': record(ann)}}))
    import shutil
    second = scene.with_name('scene_00002'); shutil.copytree(scene, second)
    with np.load(cache, allow_pickle=False) as source:
        arrays = dict(source)
    arrays['scene_id'] = np.asarray('scene_00002')
    np.savez(second / 'cache.npz', **arrays)
    second_manifest = second / 'source_manifest.json'
    second_manifest.write_text(json.dumps({'scene_id': 'scene_00002', 'complete': True,
        'artifacts': {'sam2_projection_cache': record(second / 'cache.npz'),
                      'annotation_manifest': record(second / 'annotation.json')}}))
    split = tmp_path / 'train.txt'; split.write_text('scene_00000\nscene_00002\n')
    audit = tmp_path / 'source_audit.json'
    audit.write_text(json.dumps({'passed': True, 'artifact_hashes_verified': True,
        'splits': {'optimization': {'sha256': file_sha256(split)}},
        'scenes': [{'scene_id': 'scene_00000', 'manifest_sha256': file_sha256(manifest)},
                   {'scene_id': 'scene_00002', 'manifest_sha256': file_sha256(second_manifest)}]}))
    result = audit_areas(scene.parent, split, audit)
    area = tmp_path / 'areas.json'; area.write_text(json.dumps(result))
    return scene.parent, split, audit, area


def test_area_audit_is_training_only_and_binds_source_bytes(tmp_path):
    import json
    from delimit3d.training.sampling_ablation_cli import audit_areas
    source, split, audit, area = _area_fixture(tmp_path)
    result = json.loads(area.read_text())
    assert result['proposals'] == 8 and result['threshold_pixels_inclusive'] == 10
    assert result['removed_fraction'] == .25
    assert result['train_frames'] == result['train_scenes'] == 2
    with (source / 'scene_00000/cache.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Source artifact changed'):
        audit_areas(source, split, audit)


def _prepared(tmp_path):
    import json
    from pathlib import Path
    from delimit3d.runtime.config import load_config
    from delimit3d.training.sampling_ablation_cli import prepare_study
    source, split, audit, area = _area_fixture(tmp_path)
    values = load_config(Path(__file__).resolve().parents[1] / 'configs/pretrain/sam2_selected_proposal_negatives.yaml', expand_env=False)
    val = tmp_path / 'val.txt'; val.write_text('scene_00001\n')
    code = tmp_path / 'code.json'; code.write_text('{}')
    fixed = tmp_path / 'fixed'; fixed.mkdir(); (fixed / 'manifest.json').write_text('{}')
    initial = tmp_path / 'initial.pt'; initial.write_bytes(b'common init')
    values.update(source_root=str(source), source_audit=str(audit), train_split=str(split), holdout_split=str(val),
                  code_manifest=str(code), fixed_eval_pack=str(fixed), litept_root=str(tmp_path / 'LitePT'))
    recipe = tmp_path / 'recipe.json'; recipe.write_text(json.dumps(values))
    output = tmp_path / 'study'
    study = prepare_study(recipe, output, initial, area, activation_update=8)
    return output, study


def test_preparation_preserves_horizon_policy_and_initialization(tmp_path):
    import json
    from delimit3d.pretraining.arguments import parse_args
    output, study = _prepared(tmp_path)
    assert study['logical_batch_schedule']['epochs'] == 150
    assert study['stop_epoch'] == 50 and study['stop_update'] == 10
    initializations, fixed = set(), set()
    for variant, record in study['recipes'].items():
        args = parse_args(['--config', record['path']])
        initializations.add(args.initial_checkpoint); fixed.add(args.fixed_eval_pack)
        assert args.epochs == 150 and args.pause_after_update == 10
        assert args.allow_shared_proposal_negatives is True
        assert args.ablation_variant == variant
    assert len(initializations) == len(fixed) == 1
    args = parse_args(['--config', study['recipes']['full']['path']])
    assert args.proposal_area_cutoff is None
    area = tmp_path / 'areas.json'
    area.write_text(json.dumps({**json.loads(area.read_text()), 'threshold_pixels_inclusive': 9}))
    with pytest.raises(SystemExit):
        parse_args(['--config', study['recipes']['no_small_positive']['path']])


def _checkpoints(output, study, *, epoch=50, update=10):
    from delimit3d.training.sampling_ablation_cli import MATCHED_KEYS
    from delimit3d.pretraining.arguments import parse_args
    from delimit3d.training.sampling_ablation import ablation_contract
    paths = {}
    for variant, record in study['recipes'].items():
        args = parse_args(['--config', record['path']])
        cfg = {key: getattr(args, key, 'same') for key in MATCHED_KEYS}
        cfg.update(logical_batch_schedule=study['logical_batch_schedule'],
                   world_size=2, total_updates=30, warmup_updates=3, scene_visits_per_epoch_per_rank=1,
                   dataset=args.dataset, litept_variant=args.litept_variant, sampling_mode=args.sampling_mode,
                   coordinate_normalization_policy=args.coordinate_normalization_policy,
                   multiscale_supervision=False, warmup_epochs=17, precision='fp32', token_pure_dec0=True,
                   supervised_stages=['dec0'], rgb_color_mean=args.color_mean, rgb_color_std=args.color_std,
                   eval_epochs=list(args.evaluation_epochs), checkpoint_epochs=list(args.checkpoint_epochs),
                   normalization={'policy': 'sync_batchnorm', 'bn_recalibration_scenes': 48},
                   hierarchy_supervision={'mode': 'off'},
                   lr_schedule={'kind': 'linear_warmup_cosine', 'start_epoch': 0, 'start_update': 0,
                                'warmup_epochs': 17, 'warmup_updates': 3, 'start_factor': 0.,
                                'peak_factor': 1., 'final_factor': .01},
                   initialization_seed=42, fixed_eval_seed=42, sam2_maximum_raw_points=100000,
                   allow_shared_proposal_negatives=True,
                   initialization={'sha256': study['inputs']['initial_checkpoint']['sha256']},
                   fixed_eval_pack={'manifest_sha256': study['inputs']['fixed_eval_manifest']['sha256']},
                   **ablation_contract(args))
        for key, binding in [('train_split', 'train_split'), ('holdout_split', 'holdout_split'),
                             ('source_dataset_audit', 'source_audit'), ('code_manifest', 'code_manifest')]:
            cfg[key] = {'sha256': study['inputs'][binding]['sha256']}
        path = output / (variant + '.pt'); paths[variant] = path
        torch.save({'epoch': epoch, 'update': update, 'resumable': True, 'resolved_config': cfg,
                    'model_state_dict': {'weight': torch.ones(2)},
                    'feature_hard_state': {'enabled': update >= 8, 'activation_update': 8 if update >= 8 else None}}, path)
    return paths


def test_checkpoint_admission_rejects_unmatched_arm_nonfinite_and_wrong_boundary(tmp_path):
    from delimit3d.training.sampling_ablation_cli import verify_checkpoints
    output, study = _prepared(tmp_path)
    paths = _checkpoints(output, study)
    kwargs = dict(epoch=50, update=10, checkpoints=paths)
    result = verify_checkpoints(output / 'study.json', **kwargs)
    assert result['status'] == 'CHECKPOINTS_VERIFIED' and result['feature_hard_has_activated']
    with pytest.raises(ValueError, match='explicitly marked interim'):
        verify_checkpoints(output / 'study.json', epoch=5, update=1, checkpoints=paths)
    path = paths['no_feature_hard']
    payload = torch.load(path, weights_only=False)
    payload['model_state_dict']['weight'][0] = float('nan'); torch.save(payload, path)
    with pytest.raises(ValueError, match='Nonfinite'):
        verify_checkpoints(output / 'study.json', **kwargs)
    payload['model_state_dict']['weight'][0] = 1
    payload['resolved_config']['ablation_variant'] = 'full'; torch.save(payload, path)
    with pytest.raises(ValueError, match='Wrong checkpoint sampling arm'):
        verify_checkpoints(output / 'study.json', **kwargs)


def test_interim_before_feature_hard_activation_is_explicit(tmp_path):
    from delimit3d.training.sampling_ablation_cli import verify_checkpoints
    output, study = _prepared(tmp_path)
    paths = _checkpoints(output, study, epoch=25, update=5)
    result = verify_checkpoints(output / 'study.json', epoch=25, update=5, checkpoints=paths, interim=True)
    assert result['interim'] and not result['feature_hard_has_activated']


@pytest.mark.parametrize("field", ["train_split", "holdout_split", "source_dataset_audit", "code_manifest", "backbone_lr"])
def test_consistently_wrong_checkpoints_do_not_form_a_matched_study(tmp_path, field):
    from delimit3d.training.sampling_ablation_cli import verify_checkpoints
    output, study = _prepared(tmp_path)
    paths = _checkpoints(output, study)
    for path in paths.values():
        payload = torch.load(path, weights_only=False)
        payload["resolved_config"][field] = .1 if field == "backbone_lr" else {"sha256": "wrong"}
        torch.save(payload, path)
    with pytest.raises(ValueError, match="prepared"):
        verify_checkpoints(output / "study.json", epoch=50, update=10, checkpoints=paths)


def test_interim_rejects_unflushed_epoch_and_wrong_update(tmp_path):
    from delimit3d.training.sampling_ablation_cli import verify_checkpoints
    output, study = _prepared(tmp_path)
    for epoch, update in [(5, 1), (25, 6)]:
        with pytest.raises(ValueError, match="boundary|schedule"):
            verify_checkpoints(output / "study.json", epoch=epoch, update=update, interim=True)


@pytest.mark.parametrize("gate", [{"enabled": False, "activation_update": 8},
                                  {"enabled": True, "activation_update": 7}])
def test_checkpoint_admission_requires_observed_activation(tmp_path, gate):
    from delimit3d.training.sampling_ablation_cli import verify_checkpoints
    output, study = _prepared(tmp_path)
    paths = _checkpoints(output, study)
    payload = torch.load(paths["full"], weights_only=False)
    payload["feature_hard_state"] = gate
    torch.save(payload, paths["full"])
    with pytest.raises(ValueError, match="activation not matched"):
        verify_checkpoints(output / "study.json", epoch=50, update=10, checkpoints=paths)
