from pathlib import Path
import json
import numpy as np
import pytest
from delimit3d.visualization.contracts import Scene, Panel
from delimit3d.visualization.recipes import camera_record, crop_record, project, select_panels, render_recipes
from delimit3d.visualization.suite import run_suite


def scene():
    return Scene('fixture',np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,1.],[0.,1.,1.]]),
                 colors=np.full((4,3),.5), labels=np.array([1,1,2,2]),metadata={'synthetic':True})


def test_camera_crop_and_selection_are_explicit():
    s=scene();c=camera_record(s,{'yaw':0,'pitch':0})
    np.testing.assert_array_equal(project(s.points,c)[:,0],s.points[:,0]-.5)
    crop=crop_record(s,{'target_id':1,'padding':.1},c)
    np.testing.assert_allclose(crop['bounds'],[[-.1,-.1,-.1],[1.1,.1,.1]])
    with pytest.raises(ValueError,match='absent'):crop_record(s,{'target_id':3},c)
    panels=[Panel('m/pca','PCA',np.full((4,3),.5)),Panel('m/query','query',np.full((4,3),.7))]
    assert [p.key for p in select_panels(panels,['m/*','m/pca'])]==['m/pca','m/query']
    with pytest.raises(FileNotFoundError):select_panels(panels,['missing'])


def test_recipe_dimensions_and_required_caption_semantics(tmp_path):
    s=scene();p=Panel('m/episode_step_0','one click',np.full((4,3),.5),
        metadata={'metric_domain':'raw_points','click_budget':{'definition':'one click per object'},
                  'pairing':{'missing':['decoder_sha256']}})
    result=render_recipes(s,[p],tmp_path,recipes=[{'name':'errors','panels':['m/*']}],
        sample_indices=np.arange(4),camera=camera_record(s))
    rec=json.loads((tmp_path/'errors.json').read_text())
    assert 'raw_points' in rec['visible_caption'] and 'incomplete' in rec['visible_caption']
    assert rec['sizes']['paper']['width_inches']==7.2
    from PIL import Image
    im=Image.open(tmp_path/'errors.presentation.png')
    assert im.size==(1920,1080)
    assert (tmp_path/'errors.paper.pdf').read_bytes().startswith(b'%PDF')


def test_optional_artifact_blocks_only_dependent_recipe_and_cpu_mode(tmp_path):
    np.savez(tmp_path/'scene.npz',points=scene().points,colors=scene().colors)
    config={'formats':['html'],'queries':[],'cpu_only':True,'visualizations':['rgb'],
            'scenes':[{'scene_id':'fixture','path':'scene.npz'}],
            'models':[{'name':'absent','kind':'cache','path':'absent.npz','optional':True}],
            'recipes':[{'name':'missing','panels':['absent/prediction']}]}
    report=run_suite(config,base_dir=tmp_path,output=tmp_path/'out')
    assert report['status']=='complete'
    assert report['scenes'][0]['unavailable'][0]['status']=='blocked'
    assert report['scenes'][0]['files']['recipes']['records'][0]['status']=='blocked'
    config['models']=[{'name':'inference','kind':'callable','callable':'impossible:infer'}]
    report=run_suite(config,base_dir=tmp_path,output=tmp_path/'denied')
    assert report['status']=='failed' and 'prohibits inference' in report['errors'][0]['error']


def test_two_row_scalar_recipe_measures_legend_caption_separation(tmp_path):
    s=scene();p=Panel('m/cosine','cosine',np.array([[0.,0.,1.],[.3,.3,1.],[1.,.3,.3],[1.,0.,0.]]),
                      values=np.array([-1.,-.5,.5,1.]),limits=(-1.,1.),cmap='coolwarm')
    render_recipes(s,[p],tmp_path,recipes=[{'name':'crop','panels':['m/cosine'],'crop':{'target_id':1}}],
                   sample_indices=np.arange(4),camera=camera_record(s))
    record=json.loads((tmp_path/'crop.json').read_text())
    for size in ('paper','presentation'):
        assert record['layout_checks'][size]['legend_caption_gap_inches']>0
        assert not record['layout_checks'][size]['legend_caption_overlap']
        assert record['layout_checks'][size]['panel_title_subtitle_gap_inches']>0
        assert not record['layout_checks'][size]['panel_title_subtitle_overlap']
