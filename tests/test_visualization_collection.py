"""Portable collection assembly preserves scientific payloads and input outputs."""
import importlib.util
import json
from pathlib import Path
import re

import pytest

spec=importlib.util.spec_from_file_location('assemble_visual_collection',Path(__file__).parents[1]/'scripts/visualization/assemble_collection.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_portable_copy_refresh_keeps_payload_and_original_bytes(tmp_path):
    root=tmp_path/'source';scene=root/'000-fixture';scene.mkdir(parents=True)
    payload='{"scene_id": "SYNTHETIC fixture", "points": [[1, 2, 3]]}'
    original=f'<html><script id="scene-data" type="application/json">{payload}</script>old controls</html>'
    (scene/'index.html').write_text(original)
    run={'status':'complete','scenes':[{'scene_id':'fixture','directory':'000-fixture','dataset':'synthetic',
         'points':1,'panels':1,'files':{'html':'index.html'},'status':'complete'}]}
    (root/'run.json').write_text(json.dumps(run))
    output=tmp_path/'collection'
    record=module.assemble([root],output,refresh_viewers=True)
    updated=(output/'00-source/000-fixture/index.html').read_text()
    assert re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',updated,re.S).group(1)==payload
    assert (scene/'index.html').read_text()==original
    assert record['viewer_refresh']['pages'][0]['original_html_sha256']!=record['viewer_refresh']['pages'][0]['refreshed_html_sha256']
    assert (output/'viewer_refresh_record.json').is_file()
    locations=json.loads((output/'artifact_locations.json').read_text())['files']
    assert locations[str(scene/'index.html')]['path']=='00-source/000-fixture/index.html'
    with pytest.raises(FileExistsError):module.assemble([root],output)


def test_incomplete_source_blocks_assembly_before_output_creation(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'run.json').write_text('{"status":"failed"}')
    output=tmp_path/'collection'
    with pytest.raises(ValueError,match='incomplete'):module.assemble([source],output)
    assert not output.exists()


def test_mesh_payload_refresh_uses_mesh_template_and_preserves_packed_arrays(tmp_path):
    from delimit3d.visualization.mesh_render import _HTML
    payload=json.dumps({'scene_id':'synthetic mesh','vertices':{'data':'fixture'},'faces':{'data':'fixture'},
        'point_indices':{'data':'fixture'},'vertex_count':3,'face_count':1})
    (tmp_path/'index.html').write_text(f'<script id="scene-data" type="application/json">{payload}</script>')
    record=module.refresh_point_viewers(tmp_path)
    updated=(tmp_path/'index.html').read_text()
    assert updated==_HTML.replace('__TITLE__','synthetic mesh').replace('__PAYLOAD__',payload)
    assert record['pages'][0]['viewer']=='mesh'
