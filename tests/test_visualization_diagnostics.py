from __future__ import annotations

import json

import numpy as np
import pytest

from delimit3d.visualization.contracts import Observation, Scene
from delimit3d.visualization.diagnostics import _average_precision, _precision_at, build_panels


def scene(n=8, *, labels=True):
    points = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(np.float32)
    return Scene("fixture", points, colors=np.full((n, 3), 0.4),
                 labels=np.arange(n, dtype=np.int64) // 4 if labels else None)


def panels_by_key(panels):
    return {panel.key: panel for panel in panels}


def test_pca_shares_basis_scaling_and_sampling_only_with_compatible_layers():
    data = scene(30)
    features = np.random.default_rng(7).normal(size=(30, 5)).astype(np.float32)
    observations = [Observation("a", {"enc": features}),
                    Observation("b", {"enc": features.copy()}),
                    Observation("wide", {"enc": np.pad(features, ((0, 0), (0, 2)))}),
                    Observation("different_layer", {"other": features.copy()})]
    panels, report = build_panels(data, observations, query_indices=[0],
                                  requested=["pca"], max_analysis_points=12, seed=2)
    keyed = panels_by_key(panels)
    np.testing.assert_array_equal(keyed["a/enc/pca"].colors, keyed["b/enc/pca"].colors)
    assert len(report["pca_groups"]) == 3
    assert report["pca_groups"]["enc:d5"]["members"] == ["a/enc", "b/enc"]
    assert report["pca_groups"]["enc:d5"]["total_fit_points"] == 12
    _, repeated = build_panels(data, observations, query_indices=[0], requested=["pca"],
                               max_analysis_points=12, seed=2)
    assert report["analysis_indices"] == repeated["analysis_indices"]


def test_shared_pca_does_not_individually_rescale_shifted_models():
    data = scene(12)
    features = np.arange(12, dtype=np.float32)[:, None]
    panels, _ = build_panels(data, [Observation("a", {"enc": features}),
                                     Observation("b", {"enc": features + 100})],
                             query_indices=[], requested=["pca"])
    keyed = panels_by_key(panels)
    assert keyed["a/enc/pca"].colors[:, 0].max() < keyed["b/enc/pca"].colors[:, 0].min()


def test_constant_and_zero_features_are_finite_without_fake_retrieval():
    data = scene()
    panels, report = build_panels(data, [Observation("zero", {"enc": np.zeros((8, 4))})],
                                  query_indices=[0], requested=["pca", "similarity", "boundaries"])
    keyed = panels_by_key(panels)
    np.testing.assert_allclose(keyed["zero/enc/pca"].colors, 0.5)
    assert keyed["zero/enc/similarity/q0"].metadata["undefined_count"] == 8
    np.testing.assert_array_equal(
        keyed["zero/enc/similarity/q0"].metadata["undefined_mask"], np.ones(8, dtype=bool),
    )
    assert report["queries"]["zero/enc/q0"]["average_precision"] is None
    assert keyed["zero/enc/boundaries"].metadata["undefined_count"] == 8
    for panel in panels:
        panel.validate(8)
    json.dumps(report, allow_nan=False)


def test_query_retrieval_excludes_self_token_ignored_and_background_points():
    data = scene(6)
    data.labels = np.array([4, 4, 4, 8, -1, 0])
    data.metadata = {"background_ids": [0]}
    features = np.array([[1, 0], [1, 0], [0.8, 0.2], [-1, 0], [1, 0], [1, 0]], dtype=np.float32)
    observation = Observation("a", {"enc": features}, metadata={"token_ids": {"enc": np.array([0, 0, 1, 2, 3, 4])}})
    _, report = build_panels(data, [observation], query_indices=[0, 4], requested=["similarity"])
    query = report["queries"]["a/enc/q0"]
    assert query["eligible_points"] == 2
    assert query["positive_points"] == 1
    assert query["average_precision"] == 1
    assert "token" in query["self_exclusion"]
    assert "ignored" in report["queries"]["a/enc/q4"]["skip_reason"]


def test_missing_token_id_does_not_exclude_other_unmapped_points():
    data = scene(6)
    observation = Observation("a", {"enc": np.ones((6, 2))}, metadata={"token_ids": np.full(6, -1)})
    _, report = build_panels(data, [observation], query_indices=[0], requested=["similarity"])
    query = report["queries"]["a/enc/q0"]
    assert query["eligible_points"] == 5
    assert query["average_precision"] == pytest.approx(3 / 5)


def test_tied_retrieval_scores_are_order_independent():
    scores = np.array([1.0, 0.5, 0.5, 0.5, 0.0])
    positive = np.array([False, True, False, True, True])
    assert _average_precision(scores, positive) == pytest.approx((2 / 3) * 0.5 + (1 / 3) * 0.6)
    assert _precision_at(scores, positive, 2) == pytest.approx(1 / 3)
    order = np.array([4, 3, 1, 2, 0])
    assert _average_precision(scores[order], positive[order]) == _average_precision(scores, positive)
    assert _average_precision(np.ones(4), np.array([True, False, False, False])) == 0.25


def test_missing_ground_truth_still_produces_features_but_no_scores():
    data = scene(labels=False)
    panels, report = build_panels(data, [Observation("a", {"enc": np.ones((8, 3))})],
                                  query_indices=[0], requested=["labels", "similarity"])
    assert len(panels) == 1
    assert report["queries"]["a/enc/q0"]["average_precision"] is None
    assert any(item["kind"] == "labels" for item in report["skipped"])


def test_prediction_matching_is_invariant_to_id_permutation_and_ignores_sparse_truth():
    data = scene(9)
    data.labels = np.array([100, 100, 100, 100, 4000, 4000, 4000, 4000, -1])
    one = np.array([8, 8, 8, 8, 2, 2, 2, 2, 99])
    two = np.array([900, 900, 900, 900, 700, 700, 700, 700, 13])
    observations = [Observation("a", prediction=one), Observation("b", prediction=two)]
    panels, report = build_panels(data, observations, query_indices=[], requested=["prediction", "labels"])
    keyed = panels_by_key(panels)
    for name in ("a", "b"):
        np.testing.assert_array_equal(keyed[f"{name}/prediction"].colors, keyed["labels"].colors)
        assert report["predictions"][name]["point_agreement"] == 1
        assert report["predictions"][name]["evaluated_points"] == 8
    np.testing.assert_array_equal(keyed["a/prediction_error"].colors, keyed["b/prediction_error"].colors)


def test_one_to_one_instance_matching_penalizes_split_instances_and_missing_predictions():
    data = scene(8)
    predictions = np.array([20, 20, 21, 21, 30, 30, -1, -1])
    panels, report = build_panels(data, [Observation("split", prediction=predictions)],
                                  query_indices=[], requested=["prediction"])
    assert report["predictions"]["split"]["point_agreement"] == 0.5
    assert report["predictions"]["split"]["missed_points"] == 2
    assert len(report["predictions"]["split"]["prediction_to_gt_id"]) == 2
    for panel in panels:
        panel.validate(8)


def test_prediction_background_is_never_matched_to_an_unclicked_instance():
    data = scene(4)
    data.labels = np.array([10, 10, 20, 20])
    observation = Observation("interactive", prediction=np.array([1, 1, 0, 0]),
                              metadata={"prediction_background_ids": [0]})
    panels, report = build_panels(data, [observation], query_indices=[], requested=["prediction"])
    result = report["predictions"]["interactive"]
    assert result["point_agreement"] == 0.5
    assert result["prediction_to_gt_id"] == {"1": 10}
    assert result["missed_points"] == 2
    np.testing.assert_allclose(panels[1].colors[2:], [[0.97, 0.66, 0.16]] * 2)


def test_prediction_background_is_separate_from_truth_background_and_default_zero_is_valid():
    data = scene(4)
    data.labels = np.array([0, 0, 8, 8])
    observation = Observation("mask", prediction=np.array([0, 0, 9, 9]))
    _, report = build_panels(data, [observation], query_indices=[], requested=["prediction"])
    assert report["predictions"]["mask"]["point_agreement"] == 1
    data.metadata = {"background_ids": [0]}
    observation.prediction = np.array([-9, -9, 0, 0])
    _, report = build_panels(data, [observation], query_indices=[], requested=["prediction"])
    assert report["predictions"]["mask"]["point_agreement"] == 1
    assert report["predictions"]["mask"]["prediction_to_gt_id"] == {"0": 8}
    observation.metadata["prediction_background_ids"] = [0]
    _, report = build_panels(data, [observation], query_indices=[], requested=["prediction"])
    assert report["predictions"]["mask"]["point_agreement"] == 0


def test_fixed_cosine_and_delta_scales_accommodate_different_feature_dimensions():
    data = scene(4)
    one = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    two = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, -1, 0]], dtype=np.float32)
    panels, _ = build_panels(data, [Observation("ref", {"enc": one}), Observation("new", {"enc": two})],
                             query_indices=[0], reference_name="ref", requested=["similarity", "delta"])
    keyed = panels_by_key(panels)
    assert keyed["new/enc/similarity/q0"].limits == (-1, 1)
    assert "undefined_mask" not in keyed["new/enc/similarity/q0"].metadata
    assert keyed["new/enc/delta/q0"].limits == (-2, 2)
    np.testing.assert_allclose(keyed["new/enc/delta/q0"].values, [0, 1, 2, 0])


def test_feature_boundaries_exclude_self_and_support_duplicate_coordinates():
    data = scene(2)
    data.points[:] = 0
    features = np.array([[1, 0], [-1, 0]], dtype=np.float32)
    panels, _ = build_panels(data, [Observation("a", {"enc": features})], query_indices=[], requested=["boundaries"])
    np.testing.assert_allclose(panels[0].values, 2)


def test_single_point_pca_and_skipped_boundaries():
    panels, report = build_panels(scene(1), [Observation("a", {"enc": np.array([[5.0]])})],
                                  query_indices=[0], requested=["pca", "boundaries", "similarity"])
    assert len(panels) == 2
    assert any(item["kind"] == "boundaries" for item in report["skipped"])
    assert report["queries"]["a/enc/q0"]["average_precision"] is None


def test_confidence_name_requires_explicit_calibration_and_semantics():
    data = scene()
    raw = Observation("raw", confidence=np.arange(8) / 8)
    declared = Observation("declared", confidence=np.arange(8) / 8,
                           metadata={"confidence_calibrated": True, "confidence_semantics": "probability of foreground"})
    panels, _ = build_panels(data, [raw, declared], query_indices=[], requested=["confidence"])
    assert panels[0].title.endswith("score")
    assert panels[1].title.endswith("confidence")
    assert panels[1].limits == (0, 1)


def test_step_history_retains_clicks_and_explicit_binary_target():
    data = scene()
    steps = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
    observation = Observation("interactive", steps=steps, clicks=[{"point_index": 0, "step": 0}, {"point_index": 2, "step": 1}],
                              metadata={"prediction_kind": "binary", "target_instance_id": 0})
    panels, report = build_panels(data, [observation], query_indices=[], requested=["clicks"])
    assert len(panels[0].metadata["clicks"]) == 1
    assert len(panels[1].metadata["clicks"]) == 2
    assert report["predictions"]["interactive/step_1"]["point_agreement"] == 1


def test_invalid_requested_names_reference_queries_and_nonfinite_features_fail():
    data = scene()
    observation = Observation("a", {"enc": np.ones((8, 2))})
    with pytest.raises(ValueError, match="Unknown visualization"):
        build_panels(data, [observation], query_indices=[], requested=["magic"])
    with pytest.raises(ValueError, match="Unknown reference"):
        build_panels(data, [observation], query_indices=[], reference_name="missing")
    with pytest.raises(ValueError, match="Query indices"):
        build_panels(data, [observation], query_indices=[8])
    observation.features["enc"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_panels(data, [observation], query_indices=[])
