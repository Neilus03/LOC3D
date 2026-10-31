"""Synthetic contract fixtures, not experimental AGILE3D evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from delimit3d.visualization.contracts import Scene
from delimit3d.visualization.episodes import (compare_click_sequences, episode_from_arrays,
                                             episode_panels, import_scannet40_episode,
                                             SCANNET40_SOURCE_IMPLEMENTATION)
from delimit3d.visualization.export_contracts import EPISODE_SCHEMA, geometry_hash


def episode_fixture():
    scene = Scene("fixture", np.arange(18, dtype=np.float32).reshape(6, 3),
                  colors=np.ones((6, 3)) * .5, labels=np.array([10, 10, 20, 20, 0, 0]),
                  metadata={"synthetic": True, "background_ids": [0]}).validate()
    record = {"schema": EPISODE_SCHEMA, "scene_id": "fixture", "source_geometry_sha256": geometry_hash(scene.points),
              "episode_id": "synthetic/fixture", "mode": "recorded_benchmark", "background_id": 0,
              "targets": [{"target_id": "object10", "label_id": 10}, {"target_id": "object20", "label_id": 20}],
              "prediction_target_map": {"1": "object10", "2": "object20"}, "metric_domain": "raw_points",
              "prediction_domain": "raw_points", "model": {},
              "click_budget": {"definition": "total events divided by two objects", "maximum": 20,
                               "counts_background": True, "denominator": 2}, "click_policy": "synthetic test",
              "clicks": [{"click_id": "q0", "order": 0, "point_index": 0, "position": scene.points[0].tolist(),
                          "polarity": "positive", "target_id": "object10", "assigned_label_id": 10},
                         {"click_id": "q1", "order": 1, "point_index": 2, "position": scene.points[2].tolist(),
                          "polarity": "positive", "target_id": "object20", "assigned_label_id": 20}],
              "frames": [{"frame_id": "first", "click_count": 1}, {"frame_id": "second", "click_count": 2}]}
    arrays = {"predictions": np.array([[1, 1, 2, 2, 0, 0], [2, 2, 1, 1, 0, 0]])}
    return scene, record, arrays


def test_temporal_identity_switch_and_regression_cannot_be_hidden():
    scene, record, arrays = episode_fixture()
    episode = episode_from_arrays(record, arrays, scene)
    assert episode.predictions[1].tolist() == [20, 20, 10, 10, 0, 0]
    panels, report = episode_panels(scene, episode, "fixture")
    assert report["frames"][1]["targets"][0]["regression_from_previous"]
    assert report["frames"][1]["targets"][0]["illustrative_raw_point_iou"] == 0
    assert report["frames"][1]["targets"][0]["interference_points"]["object20"] == 2
    assert "incomplete" in panels[1].caption
    assert episode.to_observation("fixture").metadata["prediction_kind"] == "persistent_instance"


@pytest.mark.parametrize("mutate,match", [
    (lambda r, a: r["clicks"][1].update(order=0), "recorded event order"),
    (lambda r, a: r["clicks"][0].update(position=[9, 9, 9]), "position"),
    (lambda r, a: r["clicks"][0].update(polarity="negative"), "background assignment"),
    (lambda r, a: r["clicks"][1].update(click_id="q0"), "unique"),
    (lambda r, a: r["frames"][1].update(click_count=0), "monotone"),
    (lambda r, a: r.update(mode="official_looking_replay"), "mode"),
    (lambda r, a: r.pop("prediction_target_map"), "fixed prediction_target_map"),
    (lambda r, a: r["click_budget"].pop("denominator"), "semantics"),
    (lambda r, a: r["model"].update(encoder={"sha256": "a"}, decoder_trained_with_encoder_sha256="b"), "pairing mismatch"),
])
def test_rejects_ambiguous_or_contradictory_episode(mutate, match):
    scene, record, arrays = episode_fixture()
    mutate(record, arrays)
    with pytest.raises(ValueError, match=match):
        episode_from_arrays(record, arrays, scene)


def test_token_domain_click_identity_and_inverse_expansion():
    scene, record, arrays = episode_fixture()
    arrays.update(predictions=np.array([[1, 2, 0], [2, 1, 0]]), inverse_map=np.array([0, 0, 1, 1, 2, 2]),
                  representative_indices=np.array([0, 2, 4]))
    record["prediction_domain"] = "tokens"
    record["metric_domain"] = "tokens"
    record["clicks"][0]["token_index"] = 0
    episode = episode_from_arrays(record, arrays, scene)
    assert episode.predictions.shape == (2, 6)
    record["clicks"][0]["token_index"] = 1
    with pytest.raises(ValueError, match="does not correspond"):
        episode_from_arrays(record, arrays, scene)


def test_real_click_identity_comparison_and_trace_only_partial_output():
    scene, record, arrays = episode_fixture()
    first = episode_from_arrays(record, arrays, scene)
    second = episode_from_arrays(record, arrays, scene)
    assert compare_click_sequences([first, second])["identical_click_identities"]
    second.clicks[0]["click_id"] = "different"
    assert not compare_click_sequences([first, second])["identical_click_identities"]
    trace = episode_from_arrays(record, {}, scene)
    panels, report = episode_panels(scene, trace, "trace only")
    assert len(panels) == 1 and report["missing_artifacts"]
    with pytest.raises(ValueError, match="no saved prediction"):
        trace.to_observation("trace only")


def test_metric_units_domain_and_multiobject_caption():
    scene, record, arrays = episode_fixture()
    record["frames"][0]["metrics"] = [{"name": "iou", "value": 1., "unit": "fraction",
                                          "domain": "raw_points", "target_id": "object10"}]
    panels, report = episode_panels(scene, episode_from_arrays(record, arrays, scene), "fixture")
    assert "iou[object10]=1 fraction (raw_points)" in panels[1].caption
    assert report["frames"][0]["targets"][0]["recorded_iou_matches_display_domain"]
    record["frames"][0]["metrics"][0]["value"] = 81
    with pytest.raises(ValueError, match="declared unit"):
        episode_from_arrays(record, arrays, scene)


def test_historical_import_budget_domain_and_trace_consistency(tmp_path, monkeypatch):
    scene, _, _ = episode_fixture()
    cache = {"points": scene.points, "inverse_map": np.array([0, 0, 1, 1, 2, 2]),
             "representative_indices": np.array([0, 2, 4]), "checkpoint_sha256": "encoder"}
    monkeypatch.setattr("delimit3d.visualization.adapters._torch_load", lambda p: cache)
    cache_path = tmp_path / "cache.pt"
    cache_path.write_bytes(b"SYNTHETIC NOT PYTORCH")
    state = {"total_clicks": 2, "clicks": {"0": [], "1": [0], "2": [1]},
             "click_times": {"0": [], "1": [0], "2": [1]},
             "mean_iou": 1., "object_ious": {"1": 1., "2": 1.}}
    trace = {"scene": "fixture", "object_count": 2, "object_ids": [10, 20],
             "checkpoint_sha256": "decoder", "states": [state]}
    trace_path, prediction_path = tmp_path / "trace.json", tmp_path / "saved.npz"
    trace_path.write_text(json.dumps(trace))
    np.savez(prediction_path, click_1=np.array([1, 2, 0]))
    episode = import_scannet40_episode(trace_path, prediction_path, cache_path, scene, model={})
    assert episode.record["frames"][0]["click_count"] == 2
    assert episode.record["frames"][0]["clicks_per_object_budget"] == 1
    assert episode.record["metric_domain"] == "raw_points"
    assert episode.record["experiment_family"] == "public_vs_256_update_adaptation"
    assert episode.record["target_label_policy"] == "all_non_targets_are_background"
    assert episode.clicks[1]["point_index"] == 2
    np.savez(prediction_path, click_1=np.array([2, 1, 0]))
    with pytest.raises(ValueError, match="differs from its trace"):
        import_scannet40_episode(trace_path, prediction_path, cache_path, scene, model={})


def test_controlled_replay_reports_actual_mismatched_click_identity():
    scene, record, arrays = episode_fixture()
    record["mode"] = "controlled_replay"
    first = episode_from_arrays(record, arrays, scene)
    second_record = deepcopy(record)
    second_record["clicks"][0].update(point_index=1, position=scene.points[1].tolist())
    second = episode_from_arrays(second_record, arrays, scene)
    comparison = compare_click_sequences([first, second])
    assert not comparison["identical_click_identities"]
    assert comparison["identical_frame_click_counts"]
    assert "not established" in comparison["label"]


def test_error_legends_use_actual_panel_colors():
    scene, record, arrays = episode_fixture()
    panels, _ = episode_panels(scene, episode_from_arrays(record, arrays, scene), "fixture")
    for panel in panels:
        if panel.key.endswith("_error"):
            legend = panel.metadata["category_legend"]
            for value in (1, 2, 3):
                if np.any(panel.values == value):
                    np.testing.assert_allclose(panel.colors[panel.values == value],
                                               np.tile(legend[str(value)]["rgb"], (int((panel.values == value).sum()), 1)))


def test_contradictory_raw_iou_cannot_appear_as_valid_caption():
    scene,record,arrays=episode_fixture()
    record['frames'][0]['metrics']=[{'name':'iou','value':.2,'unit':'fraction',
                                   'domain':'raw_points','target_id':'object10'}]
    episode=episode_from_arrays(record,arrays,scene)
    with pytest.raises(ValueError,match='disagrees with saved mask'):
        episode_panels(scene,episode,'fixture')


def test_legacy_scannet40_unlabelled_points_remain_in_recorded_iou_denominator():
    """Synthetic reproduction of real early exports lacking target_labels."""
    scene, record, arrays = episode_fixture()
    scene.labels[-1] = -1
    arrays["predictions"][0, -1] = 1  # A false positive on an originally unlabelled point.
    record["source_implementation"] = SCANNET40_SOURCE_IMPLEMENTATION
    record["frames"][0]["metrics"] = [
        {"name": "raw_point_iou", "value": 2 / 3, "unit": "fraction",
         "domain": "raw_points", "target_id": "object10"},
        {"name": "mean_iou", "value": 5 / 6, "unit": "fraction", "domain": "raw_points"},
    ]
    episode = episode_from_arrays(record, arrays, scene)
    assert "target_labels" not in arrays
    assert episode.target_labels[-1] == 0
    assert episode.record["target_label_policy"] == "all_non_targets_are_background"
    assert episode.record["target_label_policy_compatibility"][
        "inferred_from_exact_source_implementation"] == SCANNET40_SOURCE_IMPLEMENTATION
    panels, report = episode_panels(scene, episode, "fixture")
    target = report["frames"][0]["targets"][0]
    assert target["fp"] == 1
    assert target["recorded_iou_matches_display_domain"]
    assert report["target_label_policy"] == "all_non_targets_are_background"
    assert panels[1].metadata["target_label_policy"] == "all_non_targets_are_background"

    # An unrelated evaluator must not silently inherit the historical policy.
    record["source_implementation"] += "-different-source"
    generic = episode_from_arrays(record, arrays, scene)
    assert generic.target_labels[-1] == -1
    assert "target_label_policy_compatibility" not in generic.record
    with pytest.raises(ValueError, match="disagrees with saved mask"):
        episode_panels(scene, generic, "fixture")


def test_explicit_target_label_policy_validated_and_metrics_still_checked():
    scene, record, arrays = episode_fixture()
    scene.labels[-1] = -1
    arrays["predictions"][0, -1] = 1
    record["target_label_policy"] = "all_non_targets_are_background"
    record["frames"][0]["metrics"] = [{"name": "iou", "value": 1., "unit": "fraction",
                                      "domain": "raw_points", "target_id": "object10"}]
    episode = episode_from_arrays(record, arrays, scene)
    with pytest.raises(ValueError, match="disagrees with saved mask"):
        episode_panels(scene, episode, "fixture")
    record["target_label_policy"] = "guess_from_best_metric"
    with pytest.raises(ValueError, match="target_label_policy"):
        episode_from_arrays(record, arrays, scene)
