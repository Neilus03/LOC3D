"""Staging is a byte-copy operation, including a file-addressed bundle's arrays."""
import importlib.util
import json
from pathlib import Path
import yaml
import pytest

spec=importlib.util.spec_from_file_location('stage_visual_inputs',Path(__file__).parents[1]/'scripts/visualization/stage_saved_inputs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_file_addressed_bundle_keeps_companion_bytes_and_original_identity(tmp_path):
    # Synthetic bytes exercise staging only; not an experimental model export.
    source=tmp_path/'source';source.mkdir()
    (source/'scene.npz').write_bytes(b'synthetic geometry fixture')
    (source/'record.json').write_text(json.dumps({'schema':'delimit3d.mask3d_proposals.v1','arrays_file':'native.npz'}))
    (source/'native.npz').write_bytes(b'synthetic native arrays fixture')
    manifest=source/'config.yaml';manifest.write_text(yaml.safe_dump({'scenes':[{'scene_id':'fixture','path':'scene.npz',
        'models':[{'name':'fixture','path':'record.json'}]}]}))
    output=tmp_path/'staged';record=module.stage([manifest],output)
    result=yaml.safe_load((output/'config.yaml').read_text())['scenes'][0]['models'][0]
    copied=Path(result['path'])
    assert copied.read_bytes()==(source/'record.json').read_bytes()
    assert (copied.parent/'native.npz').read_bytes()==(source/'native.npz').read_bytes()
    assert result['metadata']['input_staging']['original_path']==str(source/'record.json')
    assert len(record['inputs'][1]['files'])==2
    with pytest.raises(FileExistsError):module.stage([manifest],output)
