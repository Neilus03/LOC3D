#!/usr/bin/env python3
"""Read-only artifact inventory; inspect NPZ headers without loading feature arrays."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def npz_headers(path):
    import numpy as np
    arrays={}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith('.npy'):continue
            with archive.open(member) as handle:
                version=np.lib.format.read_magic(handle)
                shape,fortran,dtype=np.lib.format._read_array_header(handle,version)
                arrays[member[:-4]]={'shape':list(shape),'dtype':str(dtype),'fortran_order':fortran}
    return arrays


def classify(path, arrays):
    keys=set(arrays)
    if keys & {'pred_masks','mask_logits','proposal_masks'}:return 'proposal_export_candidate'
    if any(k.startswith('click_') for k in keys) or 'predictions' in keys:return 'interactive_prediction_candidate'
    if any('feature' in k for k in keys):return 'feature_export'
    if path.suffix in {'.pt','.pth','.ckpt'}:
        return 'feature_cache' if any(x in str(path) for x in ('feature_cache','features/')) else 'checkpoint_or_tensor_archive'
    if path.suffix=='.json' and '/traces/' in str(path):return 'evaluation_trace'
    if path.suffix in {'.ply','.obj'}:return 'geometry_or_colored_mesh'
    if path.suffix in {'.png','.pdf','.html','.mp4'}:return 'rendered_output'
    if path.suffix in {'.json','.jsonl','.csv'}:return 'experiment_or_metric_record'
    return 'array_or_other'


def inventory(roots):
    entries=[];missing=[]
    excluded={'__pycache__','.git','node_modules','source','code','runtime','site-packages'}
    for root in roots:
        if not root.exists():missing.append(str(root));continue
        import os
        for current,dirs,files in os.walk(root):
            dirs[:]=sorted(d for d in dirs if d not in excluded and not d.startswith('frozen_execution'))
            for name in sorted(files):
                path=Path(current)/name
                if path.suffix.lower() not in {'.npz','.npy','.pt','.pth','.ckpt','.json','.jsonl','.csv','.ply','.png','.pdf','.html','.mp4','.tar'}:continue
                stat=path.stat();arrays={};error=None
                if path.suffix=='.npz':
                    try:arrays=npz_headers(path)
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                entries.append({'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                'kind':classify(path,arrays),'arrays':arrays,'inspection_error':error})
    return {'schema':'delimit3d_artifact_inventory/v1','captured_utc':datetime.now(timezone.utc).isoformat(),
            'roots':[str(r) for r in roots],'missing_roots':missing,'excluded_directory_names':sorted(excluded),
            'scope':'Local paths only; candidate classifications are not evidence of scientific compatibility. NPZ headers inspected; tensors not unpickled.',
            'counts':dict(Counter(e['kind'] for e in entries)),'entries':entries}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a fresh record')
    record=inventory(args.roots);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record['counts'],indent=2))

if __name__=='__main__':main()
