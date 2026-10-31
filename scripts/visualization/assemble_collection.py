#!/usr/bin/env python3
"""Copy completed offline galleries into one portable, versioned collection."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from delimit3d.visualization.suite import _index, write_json
from delimit3d.visualization.cases import validate_collection


def refresh_point_viewers(root):
    """Apply current point/mesh controls to a fresh copy without recomputing data.

    The embedded payload is byte-preserved. Original pages remain at their source
    gallery paths, while a separate record hashes both versions and the template.
    """
    from delimit3d.visualization.render import _HTML
    from delimit3d.visualization.mesh_render import _HTML as MESH_HTML
    import html
    records=[]
    for path in sorted(root.rglob('index.html')):
        original=path.read_text()
        match=re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',original,re.S)
        if not match:continue
        payload=match.group(1);data=json.loads(payload)
        is_mesh=all(key in data for key in ('vertices','faces','point_indices','vertex_count','face_count'))
        if not is_mesh and 'points' not in data:raise ValueError(f'Unknown viewer payload: {path}')
        template=MESH_HTML if is_mesh else _HTML
        document=template.replace('__TITLE__',html.escape(str(data['scene_id']))).replace('__PAYLOAD__',payload)
        path.write_text(document)
        records.append({'path':str(path.relative_to(root)),
            'viewer':'mesh' if is_mesh else 'points',
            'original_html_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'refreshed_html_sha256':hashlib.sha256(document.encode()).hexdigest(),
            'unchanged_payload_sha256':hashlib.sha256(payload.encode()).hexdigest()})
    record={'schema':'delimit3d_viewer_refresh/v1','method':'Current point/mesh controls; embedded scientific payload byte-preserved',
            'template_sha256':hashlib.sha256(_HTML.encode()).hexdigest(),
            'mesh_template_sha256':hashlib.sha256(MESH_HTML.encode()).hexdigest(),'pages':records}
    write_json(root/'viewer_refresh_record.json',record)
    return record


def assemble(galleries,output,case_records=(),missing_record=None,refresh_viewers=False):
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    inspected=[]
    for root in galleries:
        record=json.loads((root/'run.json').read_text())
        if record['status']!='complete':raise ValueError(f'Gallery is incomplete: {root}')
        inspected.append((root,record))
    cases=[]
    for p in case_records:
        record=json.loads(p.read_text())
        if record.get('schema')=='delimit3d_case_collection/v1':validate_collection(record)
        cases.append((p,record))
    output.mkdir(parents=True)
    combined={'schema':'delimit3d_visualization_collection/v1','version':1,
              'title':'Delimit3D · Reproducible qualitative evaluation','status':'complete',
              'captured_utc':datetime.now(timezone.utc).isoformat(),'scenes':[],'errors':[],
              'sources':[],'case_records':[],'portability':'All gallery pages and rendered assets copied locally; source artifact paths in experiment records remain original.'}
    for index,(root,record) in enumerate(inspected):
        name=f'{index:02d}-{root.name}'
        shutil.copytree(root,output/name)
        combined['sources'].append({'directory':name,'original_gallery':str(root),
            'run_sha256':hashlib.sha256((root/'run.json').read_bytes()).hexdigest(),'code':record.get('code')})
        for scene in record['scenes']:
            combined['scenes'].append({**scene,'directory':name+'/'+scene['directory']})
    for index,(p,record) in enumerate(cases):
        dest=f'cases-{index:02d}.json';write_json(output/dest,record)
        combined['case_records'].append({'path':dest,'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if missing_record:
        shutil.copyfile(missing_record,output/'missing_inputs.json')
        combined['missing_inputs']='missing_inputs.json'
    if refresh_viewers:
        combined['viewer_refresh']=refresh_point_viewers(output)
    # Original render records can contain absolute local paths (e.g. frozen PCA
    # states). Keep those records intact and provide a portable location map.
    locations={}
    for index,(root,record) in enumerate(inspected):
        name=f'{index:02d}-{root.name}'
        for source in sorted(root.rglob('*')):
            if not source.is_file():continue
            relative=Path(name)/source.relative_to(root);copied=output/relative
            locations[str(source)]={'path':str(relative),'sha256':hashlib.sha256(copied.read_bytes()).hexdigest()}
    write_json(output/'artifact_locations.json',{'schema':'delimit3d_portable_artifact_locations/v1','files':locations})
    combined['artifact_locations']='artifact_locations.json'
    combined['completed_scenes']=len(combined['scenes'])
    write_json(output/'run.json',combined)
    _index(output,combined)
    (output/'README.txt').write_text('Open index.html directly in a browser. No server or network is needed.\nCopy this entire directory to retain all links. Source artifacts/checkpoints are recorded, not bundled.\nOriginal generated-artifact paths (including local PCA states) map to portable copies in artifact_locations.json.\nSynthetic fixtures are separately labeled. See run.json, cases-*.json and missing_inputs.json.\n')
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--galleries',type=Path,nargs='+',required=True)
    p.add_argument('--case-records',type=Path,nargs='*',default=[]);p.add_argument('--missing-record',type=Path)
    p.add_argument('--refresh-viewers',action='store_true',help='Update controls in fresh copies; preserve embedded scientific payloads and record both hashes')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    record=assemble([r.resolve() for r in a.galleries],a.output.resolve(),a.case_records,a.missing_record,a.refresh_viewers)
    print(f"Assembled {record['completed_scenes']} cases: {a.output/'index.html'}")
if __name__=='__main__':main()
