"""Integration bridge for saved research artifacts and the existing visual suite."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

from .adapters import load_observation, resolve_path
from .contracts import Observation

SAVED_KINDS = {'cache','npz','mask3d','agile3d_episode','agile3d_recorded','agile3d_scannet40','mask3d_proposals'}


def load_research_observation(spec, scene, base_dir, *, cpu_only=False):
    kind=spec.get('kind','cache')
    if cpu_only and kind not in SAVED_KINDS:
        raise ValueError(f"CPU-only saved-artifact mode prohibits inference adapter {kind!r}")
    if kind in {'agile3d_episode','agile3d_recorded','agile3d_scannet40'}:
        from .episodes import load_episode, import_scannet40_episode
        name=str(spec['name'])
        if kind in {'agile3d_episode','agile3d_recorded'}:
            episode=load_episode(resolve_path(spec['path'],base_dir),scene,expected_pairing=spec.get('expected_pairing'))
        else:
            episode=import_scannet40_episode(resolve_path(spec['trace_path'],base_dir),
                resolve_path(spec['path'],base_dir),resolve_path(spec['cache_path'],base_dir),scene,
                model=spec.get('model_record',{}),
                source_snapshot=resolve_path(spec['source_snapshot'],base_dir) if spec.get('source_snapshot') else None,
                geometry_atol=float(spec.get('geometry_atol',0)))
        if episode.predictions is not None:
            obs=episode.to_observation(name)
        else:
            obs=Observation(name,metadata={'adapter':kind,'episode':episode.record,'pairing':episode.pairing})
        return obs,('episode',episode)
    if kind=='mask3d' and (spec.get('output_contract')=='mask_only' or spec.get('preserve_proposals')):
        from .adapters import _read_payload, _ordered_geometry, _scalar, sha256_file
        from .proposals import proposals_from_mask3d_output
        path=resolve_path(spec['path'],base_dir)
        payload,_=_read_payload(path,spec.get('format'))
        alignment=_ordered_geometry(payload,spec,scene,base_dir)
        checkpoint=_scalar(payload.get('checkpoint_sha256'))
        if spec.get('expected_checkpoint_sha256',checkpoint)!=checkpoint:
            raise ValueError('Mask3D output checkpoint hash mismatch')
        proposals=proposals_from_mask3d_output(scene,payload,spec)
        proposals.record['source_artifacts']={'path':str(path),'sha256':sha256_file(path),'alignment_check':alignment,'checkpoint_sha256':checkpoint}
        return proposals.to_observation(spec['name']),('proposals',proposals)
    if kind=='mask3d_proposals':
        from .proposals import load_proposals
        proposals=load_proposals(resolve_path(spec['path'],base_dir),scene,expected_pairing=spec.get('expected_pairing'))
        return proposals.to_observation(spec['name']),('proposals',proposals)
    return load_observation(spec,scene,base_dir),None


def compose_research_panels(scene, observations, structured, *, query_indices, seed,
                            max_analysis_points, reference_name, requested, destination,
                            research_config, base_dir):
    from .diagnostics import build_panels
    from .feature_research import feature_research_panels
    base=[replace(obs,prediction=None,steps=None,confidence=None) if obj else obs
          for obs,obj in zip(observations,structured)]
    wanted=requested or ['rgb','labels','pca','similarity','delta','boundaries','norm','prediction','confidence','clicks']
    wanted_base=[w for w in wanted if w not in {'pca','norm','leakage'}]
    panels,diagnostics=build_panels(scene,base,query_indices=query_indices,seed=seed,
        max_analysis_points=max_analysis_points,reference_name=reference_name,requested=wanted_base)
    if any(obs.features for obs in observations):
        settings=dict(research_config.get('features',{}))
        settings['pca_states']={k:str(resolve_path(v,base_dir)) for k,v in settings.get('pca_states',{}).items()}
        feature_panels,report=feature_research_panels(scene,observations,query_indices=query_indices,
            output_directory=destination/'feature_states',seed=seed,max_fit_points=max_analysis_points,
            reference_name=reference_name,settings=settings)
        panels.extend(p for p in feature_panels if p.metadata.get('kind') in wanted or
                      (p.metadata.get('kind')=='leakage_context' and 'leakage' in wanted))
        diagnostics['feature_research']=report
    episodes=[]
    for obs,entry in zip(observations,structured):
        if not entry:continue
        family,obj=entry
        if family=='episode':
            from .episodes import episode_panels
            more,report=episode_panels(scene,obj,obs.name,
                boundary_radius=research_config.get('episodes',{}).get('boundary_radius'))
            episodes.append(obj)
        else:
            from .proposals import proposal_panels
            more,report=proposal_panels(scene,obj,obs.name,**research_config.get("proposals",{}))
        for p in more:
            p.metadata.setdefault('model_name',obs.name)
            p.metadata.setdefault('categorical',p.limits is None)
        panels.extend(more);diagnostics.setdefault('downstream',{})[obs.name]=report
    if episodes:
        from .episodes import compare_click_sequences
        diagnostics['click_comparison']=compare_click_sequences(episodes)
        if len(episodes)>1 and any(ep.record['mode']=='controlled_replay' for ep in episodes):
            if not all(ep.record['mode']=='controlled_replay' for ep in episodes) or not diagnostics['click_comparison']['identical_click_identities']:
                raise ValueError('Controlled replay requires identical recorded clicks and matching modes across models')
    names=sorted([obs.name for obs in observations],key=len,reverse=True)
    for panel in panels:
        for name in names:
            if panel.key.startswith(name+'/'):
                panel.metadata.setdefault('model_name',name);break
    return panels,diagnostics
