"""Portable target preparation and full-scene diagnostic evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from delimit3d.runtime.records import file_sha256, write_json


def _target_module(taxonomy):
    if taxonomy == 'scannet20':
        from . import scannet20_feature_targets as target
    elif taxonomy == 'scannet200':
        from . import scannet200_feature_targets as target
    else:
        raise ValueError(f'Unsupported target taxonomy: {taxonomy}')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Seal original-vertex ScanNet diagnostic targets.')
    prepare.add_argument('--taxonomy', choices=('scannet20', 'scannet200'), required=True)
    prepare.add_argument('--scene', required=True)
    prepare.add_argument('--source', type=Path, required=True, help='Processed scene directory.')
    prepare.add_argument('--mesh', type=Path, required=True, help='Original ordered ScanNet mesh PLY.')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--grid-size', type=float, default=.05)
    prepare.add_argument('--min-instance-points', type=int, default=1)
    prepare.add_argument('--geometry-parent', type=Path, help='Reuse exact ScanNet200 geometry for ScanNet20.')
    evaluate = commands.add_parser('evaluate', help='Run GT-K and GT-query diagnostics on all scene points.')
    evaluate.add_argument('--targets', type=Path, required=True, help='Sealed target.json.')
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument('--features', type=Path, help='Native cache NPZ: features, original_to_token, points.')
    source.add_argument('--config', type=Path, help='Existing probe backbone recipe; extracts frozen native features.')
    evaluate.add_argument('--feature-provenance', type=Path, help='Required with cache; JSON describing checkpoint and input protocol.')
    evaluate.add_argument('--normal-source', required=True, help='Normal source/orientation; distinct conventions define different protocols.')
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--seeds', default='20261010,20261011,20261012,20261013,20261014')
    evaluate.add_argument('--max-iterations', type=int, default=100)
    evaluate.add_argument('--tolerance', type=float, default=1e-6)
    evaluate.add_argument('--output', type=Path, required=True, help='New result directory; existing directories are rejected.')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        kwargs = {}
        if args.geometry_parent:
            if args.taxonomy != 'scannet20':
                parser.error('--geometry-parent is supported for scannet20 only')
            kwargs['parent_scene'] = json.loads(args.geometry_parent.read_text())
        record = _target_module(args.taxonomy).prepare_scene(
            args.scene, args.source, args.output, mesh_path=args.mesh,
            grid_size=args.grid_size, min_instance_points=args.min_instance_points, **kwargs)
        print(json.dumps({'target': str(args.output / 'target.json'), 'objects': len(record['objects']),
                          'original_points': record['original_points'], 'common_cells': record['common_cells']}))
        return
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace evaluation output: {args.output}')
    if args.features and args.feature_provenance is None:
        parser.error('--features requires --feature-provenance')
    if args.config and args.feature_provenance is not None:
        parser.error('--feature-provenance is only used with --features')
    import numpy as np
    import torch
    from .feature_diagnostics import evaluate_features
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    target_digest = file_sha256(args.targets)
    target_record = json.loads(args.targets.read_text())
    taxonomy = {'delimit3d.scannet20_feature_targets/v1': 'scannet20',
                'delimit3d.scannet200_feature_targets/v1': 'scannet200'}.get(target_record.get('schema'))
    data, objects = _target_module(taxonomy).load_scene(target_record)
    input_hashes = {str(args.targets): target_digest, target_record['input']: target_record['sha256']}
    if args.features:
        input_hashes[str(args.features)] = file_sha256(args.features)
        input_hashes[str(args.feature_provenance)] = file_sha256(args.feature_provenance)
        with np.load(args.features, allow_pickle=False) as cache:
            if not np.array_equal(cache['points'], data['xyz']):
                raise ValueError('Feature cache points differ from sealed original vertex order')
            features = cache['features']
            inverse = cache['original_to_token']
        provenance = json.loads(args.feature_provenance.read_text())
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError('Cache provenance must be a nonempty JSON object')
        checkpoint_hash = provenance.get('checkpoint_sha256')
        checkpoint_epoch = provenance.get('epoch')
        provenance = {'mode': 'supplied_native_cache', 'supplied_provenance': provenance,
                      'frozen_state_independently_verified_by_this_command': False}
    else:
        from delimit3d.runtime.config import load_config
        from delimit3d.probes.features import extract_native_features
        input_hashes[str(args.config)] = file_sha256(args.config)
        # Match scripts/probe.py configuration loading exactly.
        config = load_config(args.config)
        features, inverse, provenance = extract_native_features(
            config, data['xyz'], data['rgb255'], data['normal'], device=args.device)
        features, inverse = features.detach().cpu().numpy(), inverse.detach().cpu().numpy()
        checkpoint_hash = provenance['checkpoint'].get('file_sha256')
        checkpoint_epoch = provenance['checkpoint'].get('payload_epoch')
        provenance = {'mode': 'release_ordered_input_bridge', 'extraction': provenance,
                      'config': config, 'frozen_state_independently_verified_by_this_command': False}
    if checkpoint_hash is not None and (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
            or any(c not in '0123456789abcdef' for c in checkpoint_hash.lower())):
        raise ValueError('checkpoint_sha256 must be a 64-character hexadecimal digest')
    arrays, record = evaluate_features(
        features, inverse, data['point_to_cell'], data['labels'], data['queries'], objects,
        seeds=tuple(int(s) for s in args.seeds.split(',')), device=args.device,
        max_iter=args.max_iterations, tol=args.tolerance, grid_size=target_record['common_grid_m'])
    for path, expected in input_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f'Input changed during evaluation: {path}')
    record.update(scene=target_record['scene'], taxonomy=taxonomy, normal_source=args.normal_source,
                  coverage={'evaluated_scenes': 1, 'full_validation_completion_claimed': False},
                  feature_provenance=provenance, input_sha256=input_hashes,
                  query_selection='centroid-nearest existing original GT vertex',
                  target_eligibility=target_record['eligibility'])
    export_metadata = {'feature_provenance': provenance, 'normal_source': args.normal_source,
                       'preprocessing': {'common_grid_m': target_record['common_grid_m'],
                                         'normal_source': args.normal_source,
                                         'pooling': 'original-point-weighted then L2 normalized'},
                       'claim_boundary': record['claim_boundary']}
    if checkpoint_hash is not None:
        arrays['checkpoint_sha256'] = np.asarray(checkpoint_hash)
    if checkpoint_epoch is not None:
        export_metadata['epoch'] = checkpoint_epoch
    arrays['experiment_record_json'] = np.asarray(json.dumps(export_metadata, default=str, allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / 'diagnostics.npz'
    with output.open('xb') as handle:
        np.savez_compressed(handle, points=data['xyz'], colors=data['rgb255'], normals=data['normal'],
                            labels=data['labels'], queries=data['queries'], common_grid=data['common_grid'],
                            shift=data['shift'], **arrays)
    write_json(args.output / 'metrics.json', record)
    write_json(args.output / 'COMPLETE.json', {
        'schema': 'loc3d.frozen_feature_diagnostics_complete/v1', 'status': 'COMPLETE',
        'scene': record['scene'], 'taxonomy': taxonomy, 'target_sha256': target_digest,
        'artifacts': {name: file_sha256(args.output / name) for name in ('diagnostics.npz', 'metrics.json')},
        'coverage': record['coverage'], 'claim_boundary': record['claim_boundary']})
    print(json.dumps({'scene': record['scene'], 'taxonomy': taxonomy,
                      'kmeans_macro_iou': record['metrics']['kmeans']['macro_iou'],
                      'cosine_macro_iou': record['metrics']['cosine']['macro_iou'],
                      'output': str(args.output), 'claim_boundary': record['claim_boundary']}))
