#!/usr/bin/env python3
"""Stage byte-verified saved inputs for CPU rendering on local or RAM-backed storage.

Only explicit scene/model input paths and case collections are copied. Checkpoint
and experiment source references remain original. This never performs inference.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import yaml


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stage(manifests, output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    copied={};records=[];configs=[]
    def copy_input(value,base):
        original=Path(value).expanduser()
        if not original.is_absolute():original=(base/original).resolve()
        if str(original) in copied:return copied[str(original)]
        destination=output/'inputs'/hashlib.sha256(str(original).encode()).hexdigest()[:16]/original.name
        destination.parent.mkdir(parents=True)
        record_path=original/'record.json' if original.is_dir() else original
        record=json.loads(record_path.read_text()) if record_path.suffix=='.json' else {}
        bundle=record.get('schema') in {'delimit3d.agile3d_episode.v1','delimit3d.mask3d_proposals.v1'}
        sources=[record_path]
        if bundle:
            array_file=record.get('arrays_file','arrays.npz')
            if Path(array_file).name!=array_file:raise ValueError('Unsafe bundle array filename')
            sources.append(record_path.parent/array_file)
        entry={'original_path':str(original),'staged_path':str(destination),'files':[],'method':'byte-identical saved-input copy; no inference'}
        if original.is_dir():destination.mkdir()
        for source in sources:
            if not source.is_file():raise FileNotFoundError(source)
            target=destination/source.name if original.is_dir() else destination if source==record_path else destination.parent/source.name
            shutil.copyfile(source,target)
            source_hash=digest(source)
            if digest(target)!=source_hash:raise ValueError(f'Staging SHA mismatch: {source}')
            entry['files'].append({'original_path':str(source),'staged_path':str(target),'sha256':source_hash,'bytes':source.stat().st_size})
        copied[str(original)]=entry;records.append(entry)
        print(f'Verified {original.name}',flush=True)
        return entry
    for manifest in manifests:
        config=copy.deepcopy(yaml.safe_load(manifest.read_text()))
        witness={'original_manifest':str(manifest),'original_manifest_sha256':digest(manifest)}
        if config.get('case_collection'):
            witness['case_collection']=copy_input(config['case_collection'],manifest.parent)
            config['case_collection']=witness['case_collection']['staged_path']
        for scene in config['scenes']:
            entry=copy_input(scene['path'],manifest.parent);scene['path']=entry['staged_path']
            scene.setdefault('metadata',{})['input_staging']={**witness,**entry}
            for model in scene.get('models',[]):
                entry=copy_input(model['path'],manifest.parent);model['path']=entry['staged_path']
                model.setdefault('metadata',{})['input_staging']=entry
        for model in config.get('models',[]):
            entry=copy_input(model['path'],manifest.parent);model['path']=entry['staged_path']
            model.setdefault('metadata',{})['input_staging']=entry
        config['cpu_only']=True
        path=output/manifest.name
        if path.exists():raise ValueError('Manifest basenames must be unique')
        path.write_text(yaml.safe_dump(config,sort_keys=False));configs.append(str(path))
    record={'schema':'delimit3d_saved_input_staging/v1','inference_performed':False,'inputs':records,'manifests':configs}
    (output/'staging_record.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();stage([m.resolve() for m in args.manifests],args.output.resolve())
