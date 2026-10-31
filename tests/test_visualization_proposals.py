"""Synthetic mask-only/class-aware contract tests; no model inference."""
import numpy as np
import pytest

from delimit3d.visualization.contracts import Scene
from delimit3d.visualization.export_contracts import PROPOSAL_SCHEMA, geometry_hash
from delimit3d.visualization.proposals import (proposal_diagnostics, proposal_panels,
                                              proposals_from_arrays, proposals_from_mask3d_output)


def proposal_fixture():
    scene = Scene("fixture", np.arange(36, dtype=np.float32).reshape(12, 3),
                  labels=np.array([1] * 4 + [2] * 4 + [0] * 4), metadata={"background_ids": [0]})
    masks = np.zeros((6, 12), dtype=bool)
    masks[0, :2] = True
    masks[1, 2:4] = True  # split object 1
    masks[2, 4:8] = True
    masks[3, 4:8] = True  # duplicate object 2
    masks[4, :8] = True   # merged objects 1 and 2
    masks[5, 8:] = True   # false positive on background
    arrays = {"masks": masks, "query_ids": np.array([17, 18, 31, 42, 51, 73]),
              "scores": np.array([.4, .5, .8, .9, .6, .7])}
    record = {"schema": PROPOSAL_SCHEMA, "scene_id": scene.scene_id,
              "source_geometry_sha256": geometry_hash(scene.points), "output_contract": "mask_only",
              "mask_values": "binary", "mask_layout": "QN", "prediction_domain": "raw_points",
              "score_semantics": "synthetic exported mask confidence", "evaluated_postprocessing": {"nms": False},
              "recorded_matching": [{"query_id": 31, "gt_id": 2, "source": "synthetic fixture"}]}
    return scene, record, arrays


def test_mask_only_preserves_queries_overlap_scores_and_processing():
    scene, record, arrays = proposal_fixture()
    proposals = proposals_from_arrays(record, arrays, scene)
    before = proposals.masks.copy()
    labels, display = proposals.display_segmentation()
    assert labels[4] == 4  # row four is query 42
    assert display["display_id_to_query_id"]["4"] == "42"
    assert not display["official_benchmark_processing"]
    assert record["evaluated_postprocessing"] == {"nms": False}
    np.testing.assert_array_equal(proposals.masks, before)
    assert proposals.record["overlapping_point_count"] == 8
    assert "class_logits" not in proposals.arrays


def test_split_merge_duplicate_and_false_positive_diagnostics():
    scene, record, arrays = proposal_fixture()
    proposals = proposals_from_arrays(record, arrays, scene)
    report = proposal_diagnostics(scene, proposals)
    assert not report["official_metric"]
    assert "73" in report["unmatched_query_ids"]
    assert any(item["gt_id"] == 1 for item in report["split_candidates"])
    assert any(item["query_id"] == "51" for item in report["merge_candidates"])
    assert report["duplicates"] == [{"query_ids": ["31", "42"], "pair_iou": 1., "gt_ids": [2]}]
    assert report["recorded_matching"] == record["recorded_matching"]
    panels, details = proposal_panels(scene, proposals, "fixture", query_ids=["73"])
    assert panels[-1].metadata["proposal"]["query_id"] == "73"
    assert "not official AP" in panels[2].caption
    assert details["inspection_selection"]["method"] == "explicit persistent query_ids"


def test_missing_scores_only_blocks_score_dependent_display():
    scene, record, arrays = proposal_fixture()
    arrays.pop("scores")
    proposals = proposals_from_arrays(record, arrays, scene)
    with pytest.raises(ValueError, match="exported scores"):
        proposals.display_segmentation(policy="highest_score")
    assert proposals.to_observation("fixture").confidence is None
    assert proposal_panels(scene, proposals, "fixture")[0]
    assert proposals.record["missing_artifacts"]


def test_saved_proposal_pairing_expectation_and_visible_incomplete_record(tmp_path):
    from delimit3d.visualization.export_contracts import save_proposal_export
    from delimit3d.visualization.proposals import load_proposals
    scene,record,arrays=proposal_fixture()
    record['model']={'encoder':{'checkpoint':'synthetic.pt','sha256':'synthetic-encoder'}}
    path=tmp_path/'bundle'
    save_proposal_export(path,scene,record=record,arrays=arrays)
    with pytest.raises(ValueError,match='Expected pairing mismatch'):
        load_proposals(path,scene,expected_pairing={'encoder':{'sha256':'other'}})
    proposals=load_proposals(path,scene,expected_pairing={'encoder':{'sha256':'synthetic-encoder'}})
    panels,_=proposal_panels(scene,proposals,'fixture')
    assert all(p.metadata['pairing']['status']=='incomplete' for p in panels)
    assert all('record incomplete' in p.caption for p in panels)


def test_native_mask_only_positive_logit_threshold_and_opt_in_score_formula():
    scene = Scene("tiny", np.arange(9, dtype=np.float32).reshape(3, 3))
    output = {"mask_logits": np.array([[0., 2, -2], [3., -1, -1]])}
    proposals = proposals_from_mask3d_output(scene, output, {"output_contract": "mask_only",
                                                           "derive_scores": "mean_positive_mask_probability"})
    assert proposals.masks.tolist() == [[False, True, False], [True, False, False]]
    np.testing.assert_allclose(proposals.scores, 1 / (1 + np.exp(-np.array([2., 3.]))))
    assert "not asserted" in proposals.record["score_source"]
    assert "class_logits" not in proposals.arrays


def test_class_aware_export_retains_logits_without_dropping_no_object_queries():
    scene = Scene("tiny", np.arange(9, dtype=np.float32).reshape(3, 3))
    output = {"pred_masks": [np.ones((3, 2))], "pred_logits": np.array([[[1, 8], [8, 1]]]),
              "query_ids": np.array([4, 9]), "scores": np.array([.1, .8])}
    proposals = proposals_from_mask3d_output(scene, output, {"output_contract": "class_aware",
                                                           "score_semantics": "recorded class-times-mask score"})
    assert len(proposals.masks) == 2
    np.testing.assert_array_equal(proposals.arrays["class_logits"], output["pred_logits"][0])
    assert proposals.query_ids.tolist() == [4, 9]


@pytest.mark.parametrize("mutation,match", [
    (lambda r, a: a.update(query_ids=np.array([1] * 6)), "unique"),
    (lambda r, a: a.update(scores=np.array([np.nan] * 6)), "finite"),
    (lambda r, a: a.update(class_logits=np.ones((6, 2))), "classification-head"),
    (lambda r, a: r.update(mask_values="unspecified"), "mask_values"),
    (lambda r, a: r.update(prediction_domain="tokens"), "inverse_map"),
])
def test_invalid_exports_rejected(mutation, match):
    scene, record, arrays = proposal_fixture()
    mutation(record, arrays)
    with pytest.raises(ValueError, match=match):
        proposals_from_arrays(record, arrays, scene)


def test_empty_proposals_remain_valid_missed_scene():
    scene, record, arrays = proposal_fixture()
    arrays = {key: value[:0] for key, value in arrays.items()}
    record["recorded_matching"] = []
    proposals = proposals_from_arrays(record, arrays, scene)
    assert proposal_diagnostics(scene, proposals)["missed_gt_ids"] == [1, 2]
    assert proposals.to_observation("empty").prediction.tolist() == [0] * 12


def test_display_colors_follow_persistent_query_identity_after_row_permutation():
    scene, record, arrays = proposal_fixture()
    first = proposals_from_arrays(record, arrays, scene)
    order = np.array([5, 4, 3, 2, 1, 0])
    second = proposals_from_arrays(record, {key: value[order] for key, value in arrays.items()}, scene)
    first_panel = next(panel for panel in proposal_panels(scene, first, "first")[0] if panel.key.endswith("proposal_display"))
    second_panel = next(panel for panel in proposal_panels(scene, second, "second")[0] if panel.key.endswith("proposal_display"))
    np.testing.assert_array_equal(first_panel.colors, second_panel.colors)


def test_ignored_gt_stays_gray_in_error_panels_and_matching_identity_is_checked():
    scene, record, arrays = proposal_fixture()
    scene.labels[-1] = 99
    scene.metadata["ignore_label_ids"] = [99]
    proposals = proposals_from_arrays(record, arrays, scene)
    panels, _ = proposal_panels(scene, proposals, "fixture")
    error = next(panel for panel in panels if panel.key.endswith("proposal_errors"))
    np.testing.assert_allclose(error.colors[-1], [.53, .55, .58])
    assert proposals.record["matching_record_status"]["status"] == "incomplete"
    record["recorded_matching"][0]["query_id"] = "missing"
    with pytest.raises(ValueError, match="unknown query_id"):
        proposals_from_arrays(record, arrays, scene)
