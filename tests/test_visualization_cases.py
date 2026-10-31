import copy
import pytest
from delimit3d.visualization.cases import select_cases, validate_collection


def rows():
    return [{"case_id":str(i), "scene_id":str(i), "dataset":"ScanNet", "experiment":family,
             "comparison":{"reference_value":.5,"model_value":value,"metric":"IoU@5",
                           "metric_domain":"raw_points","budget_definition":"5 per object","unit":"fraction"}}
            for i,(family,value) in enumerate([("old",.8),("old",.5),("old",.2),("new",.6)])]


def test_order_invariance_family_strata_and_effect_labels():
    first=select_cases(rows(),representative_count=1)
    assert first==select_cases(list(reversed(rows())),representative_count=1)
    tags={r['case_id']:r['selection_tags'] for r in first['cases']}
    assert 'win' in tags['0'] and 'tie' in tags['1'] and 'regression' in tags['2']
    assert 'difficult' in tags['2'] and 'representative' in tags['3']
    validate_collection(first)


def test_population_tampering_and_incomplete_comparisons_fail():
    record=select_cases(rows());record['candidate_population'][0]['scene_id']='changed'
    with pytest.raises(ValueError,match='hash'): validate_collection(record)
    invalid=rows();del invalid[0]['comparison']['metric_domain']
    with pytest.raises(ValueError,match='metric_domain'):select_cases(invalid)
    with pytest.raises(ValueError,match='unique'):select_cases(rows()+rows())


def test_representatives_do_not_depend_on_effects():
    a=rows();b=copy.deepcopy(a)
    for row in b:row['comparison']['model_value']=.1
    f=lambda data:{r['case_id'] for r in select_cases(data, effect_count=0)['cases']}
    assert f(a)==f(b)


def test_selected_case_cannot_silently_change_query_or_checkpoint():
    record=select_cases(rows());record['cases'][0]['scene_id']='wrong-scene'
    with pytest.raises(ValueError,match='differs'):validate_collection(record)


def test_forged_representative_tag_rejected():
    record=select_cases(rows(),representative_count=1,effect_count=0)
    first=record['cases'][0]
    source=next(r for r in rows() if r['experiment']==first['experiment'] and r['case_id']!=first['case_id'])
    record['cases'][0]={**source,'selection_tags':first['selection_tags'],'selection_reasons':first['selection_reasons']}
    with pytest.raises(ValueError,match='deterministic'):validate_collection(record)


def test_collection_binding_preserves_original_query_and_hashes_metric_source(tmp_path):
    import hashlib,json
    from delimit3d.visualization.cases import resolve_scene_cases
    metric=tmp_path/'metrics.json';metric.write_text('{"iou":0.8}')
    row={**rows()[0],'query_original_point_id':17,'quantitative_rows':[{'path':str(metric),'json_pointer':'/iou'}]}
    collection=select_cases([row]);path=tmp_path/'cases.json';path.write_text(json.dumps(collection))
    spec={'scene_id':row['scene_id'],'dataset':row['dataset'],'queries':[17],
          'metadata':{'case_id':row['case_id']}}
    config={'case_collection':str(path),'scenes':[spec]}
    scenes,bound=resolve_scene_cases(config,tmp_path)
    assert scenes[0]['experiment']==row['experiment']
    assert scenes[0]['selection']['population_sha256']==collection['population_sha256']
    assert scenes[0]['quantitative_rows'][0]['sha256']==hashlib.sha256(metric.read_bytes()).hexdigest()
    assert 'experiment' not in spec
    config['scenes'][0]['queries']=[18]
    with pytest.raises(ValueError,match='original candidate identity'):resolve_scene_cases(config,tmp_path)
