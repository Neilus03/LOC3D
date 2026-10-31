"""Synthetic fixtures only: these test contracts, not experimental performance."""
from __future__ import annotations

import json

import numpy as np
import pytest

from delimit3d.visualization.contracts import Observation, Scene
from delimit3d.visualization.feature_research import (
    FrozenPCA, boundary_regions, feature_research_panels, fit_frozen_pca,
    matched_cosine_delta, query_cosine, query_diagnostics, spherical_kmeans,
    validate_requested_identity,
)


def record(source="checkpoint-a"):
    return {"source_id": source, "feature_level": "dec0", "checkpoint_sha256": source,
            "scene_id": "synthetic", "epoch": 36,
            "preprocessing": {"grid_size": 0.05, "input_features": "rgbn6"}}


def fit(x, **kwargs):
    return fit_frozen_pca(x, original_point_ids=np.arange(len(x)), source_record=record(), **kwargs)


def geometry(n):
    return np.column_stack((np.arange(n) * 0.1, np.zeros((n, 2)))).astype(np.float32)


def test_pca_roundtrip_preserves_exact_transform_fit_identities_and_scales(tmp_path):
    x = np.random.default_rng(3).normal(size=(31, 5)).astype(np.float32)
    state = fit(x, max_fit_points=13, seed=7)
    path = tmp_path / "pca.npz"
    state.save(path)
    loaded = FrozenPCA.load(path)
    np.testing.assert_array_equal(loaded.rgb(x, source_record=record()), state.rgb(x, source_record=record()))
    np.testing.assert_array_equal(loaded.fit_original_point_ids, state.fit_original_point_ids)
    np.testing.assert_array_equal(loaded.mean, x[state.fit_original_point_ids].astype(np.float64).mean(axis=0))
    assert loaded.record() == state.record()
    assert loaded.record()["fit_point_count"] == 13
    with pytest.raises(FileExistsError):
        state.save(path)
    with pytest.raises(ValueError, match="read-only"):
        loaded.basis[0, 0] = 0


def test_adding_model_and_changing_display_crop_cannot_recolor_existing_fit():
    x = np.random.default_rng(4).normal(size=(22, 5))
    state = fit(x, max_fit_points=9, seed=8)
    before = state.rgb(x, source_record=record())
    fit_frozen_pca(x * 100 + 200, original_point_ids=np.arange(len(x)),
                   source_record=record("checkpoint-b"), max_fit_points=9, seed=8)
    np.testing.assert_array_equal(before, state.rgb(x, source_record=record()))
    np.testing.assert_array_equal(before[[0, 7, 8]], state.rgb(x[[0, 7, 8]], source_record=record()))
    assert fit(x, max_fit_points=9, seed=8).record() == state.record()


def test_pca_fit_is_invariant_to_input_permutation_with_preserved_original_ids():
    x = np.random.default_rng(9).normal(size=(29, 4))
    permutation = np.random.default_rng(6).permutation(len(x))
    a = fit(x, max_fit_points=11, seed=5)
    b = fit_frozen_pca(x[permutation], original_point_ids=permutation,
                       source_record=record(), max_fit_points=11, seed=5)
    assert a.record() == b.record()
    np.testing.assert_array_equal(a.rgb(x, source_record=record()), b.rgb(x, source_record=record()))


def test_equal_dimensions_do_not_authorize_cross_checkpoint_pca_projection():
    x = np.random.default_rng(1).normal(size=(12, 3))
    state = fit(x)
    with pytest.raises(ValueError, match="equal dimensions are insufficient"):
        state.rgb(x, source_record=record("independent-checkpoint"))
    with pytest.raises(ValueError, match="preprocessing"):
        state.rgb(x, source_record={**record(), "preprocessing": {"grid_size": 0.02}})
    with pytest.raises(ValueError, match="epoch"):
        state.rgb(x, source_record={**record(), "epoch": 150})


def test_explicit_coordinate_compatibility_is_recorded_without_semantic_claim():
    x = np.arange(16).reshape(8, 2).astype(float)
    a = {**record(), "feature_coordinate_system": "declared-continuation-coordinate-system",
         "coordinate_compatibility_reason": "Same feature channel convention; projection is descriptive only"}
    b = {**a, "source_id": "checkpoint-b", "checkpoint_sha256": "checkpoint-b"}
    state = fit_frozen_pca(x, original_point_ids=np.arange(len(x)), source_record=a)
    transformed = state.rgb(x + 100, source_record=b)
    assert np.isfinite(transformed).all()
    assert state.compatibility(b)["mode"] == "explicit_common_coordinate_system"
    assert "not semantic correspondence" in state.compatibility(b)["interpretation"]
    assert state.rgb(x, source_record=a)[:, 0].mean() < transformed[:, 0].mean()


def test_constant_and_single_point_pca_are_gray_without_invented_variance():
    for x in (np.ones((8, 5)), np.ones((1, 1))):
        state = fit(x)
        np.testing.assert_array_equal(state.rgb(x, source_record=record()), np.full((len(x), 3), 0.5))
        np.testing.assert_array_equal(state.explained_variance_ratio, 0)
        json.dumps(state.record(), allow_nan=False)


def test_saved_pca_detects_tampered_numerical_state(tmp_path):
    x = np.arange(18).reshape(9, 2).astype(float)
    original = tmp_path / "original.npz"
    fit(x).save(original)
    with np.load(original, allow_pickle=False) as data:
        fields = dict(data)
    fields["mean"] = fields["mean"] + 1
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **fields)
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenPCA.load(tampered)


@pytest.mark.parametrize("field,actual", [("epoch", 150), ("scene_id", "scene-b"),
                                          ("query_original_point_id", 24),
                                          ("checkpoint_sha256", "wrong")])
def test_requested_exact_epoch_scene_checkpoint_and_query_never_substitute(field, actual):
    requested = {"epoch": 36, "scene_id": "scene-a", "query_original_point_id": 21,
                 "checkpoint_sha256": "right"}
    validate_requested_identity(requested, requested)
    with pytest.raises(ValueError, match="no substitution"):
        validate_requested_identity(requested, {**requested, field: actual})
    with pytest.raises(ValueError, match="missing"):
        validate_requested_identity(requested, {key: value for key, value in requested.items() if key != field})


def test_query_cosine_zero_vectors_are_undefined_and_rotations_do_not_change_scalar_maps():
    x = np.asarray([[1., 0], [0, 1], [-1, 0], [0, 0]])
    rotation = np.asarray([[0., -1], [1, 0]])
    np.testing.assert_allclose(query_cosine(x, 0), [1, 0, -1, np.nan], equal_nan=True)
    np.testing.assert_allclose(query_cosine(x, 0), query_cosine(x @ rotation, 0), equal_nan=True)
    assert np.isnan(query_cosine(x, 3)).all()
    with pytest.raises(ValueError, match="integer"):
        query_cosine(x, True)


def test_spherical_kmeans_is_query_free_and_recovers_cosine_groups():
    first = np.asarray([[1., 0.], [0.98, .2], [.95, -.1]])
    second = np.asarray([[0., 1.], [.2, .98], [-.1, .95]])
    labels, centers, record_ = spherical_kmeans(np.vstack([first, second]), 2, seed=4)
    assert labels.shape == (6,)
    assert centers.shape == (2, 2)
    assert record_["fit_uses_queries"] is False
    assert record_["n_clusters"] == 2
    assert set(labels[:3]) != set(labels[3:])
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert np.isfinite(centers).all()


def test_spherical_kmeans_marks_zero_rows_undefined_and_rejects_excess_k():
    features = np.asarray([[1., 0.], [0., 1.], [0., 0.]])
    labels, _centers, record_ = spherical_kmeans(features, 2, seed=2)
    assert labels[-1] == -1
    assert record_["zero_feature_rows"] == 1
    with pytest.raises(ValueError, match="nonzero feature rows"):
        spherical_kmeans(features, 3)


def test_cosine_delta_uses_matched_query_and_geometry_not_feature_coordinate_subtraction():
    a = np.asarray([[1., 0], [0, 1], [-1, 0]])
    b = np.asarray([[1., 0, 0], [1, 0, 0], [1, 0, 0]])
    kwargs = dict(points_a=geometry(3), points_b=geometry(3), original_point_ids_a=np.array([2, 4, 8]),
                  original_point_ids_b=np.array([2, 4, 8]), query_original_point_id=2,
                  scene_id_a="same", scene_id_b="same")
    np.testing.assert_allclose(matched_cosine_delta(a, b, **kwargs), [0, 1, 2])
    with pytest.raises(ValueError, match="identities/order"):
        matched_cosine_delta(a, b, **{**kwargs, "original_point_ids_b": np.array([4, 2, 8])})
    with pytest.raises(ValueError, match="geometry"):
        matched_cosine_delta(a, b, **{**kwargs, "points_b": geometry(3)[::-1]})
    with pytest.raises(ValueError, match="absent"):
        matched_cosine_delta(a, b, **{**kwargs, "query_original_point_id": 7})
    with pytest.raises(ValueError, match="same scene"):
        matched_cosine_delta(a, b, **{**kwargs, "scene_id_b": "different"})


def diagnostic(scores, **overrides):
    n = len(scores)
    return query_diagnostics(np.asarray(scores, dtype=float), **{
        "points": geometry(n), "labels": np.array([1, 1, 1, 2, 2, 3, 3, -1, 0])[:n],
        "original_point_ids": np.arange(n), "query_original_point_id": 0,
        "semantic_labels": np.array([4, 4, 4, 4, 4, 7, 7, -1, 0])[:n],
        "background_ids": (0,), **overrides})


def test_leakage_excludes_query_token_ignored_background_and_records_denominators():
    record_, regions = diagnostic([1, 1, .9, .95, .1, .2, .1, 1, 1],
                                  token_ids=np.array([5, 5, 6, 7, 8, 9, 10, 11, 12]))
    assert record_["eligible_points"] == 5
    assert record_["retrieval_depth"] == 1
    assert record_["regions"]["same_category_other_instances"]["fraction_of_top_target_size"] == 1
    assert record_["regions"]["target"]["fraction_of_top_target_size"] == 0
    assert not regions["eligible"][[0, 1, 7, 8]].any()
    assert "not downstream AP" in record_["scope"]
    assert record_["boundary_definition"]["official_metric"] is False
    json.dumps(record_, allow_nan=False)


def test_leakage_ties_use_expected_counts_and_are_permutation_invariant():
    scores = np.ones(7)
    a, _ = diagnostic(scores)
    assert a["retrieval_depth"] == 2
    assert a["regions"]["other_instances"]["fraction_of_top_target_size"] == pytest.approx(4 / 6)
    permutation = np.array([0, 6, 3, 2, 4, 1, 5])
    b, _ = diagnostic(scores[permutation], points=geometry(7)[permutation],
                      original_point_ids=np.arange(7)[permutation],
                      labels=np.array([1, 1, 1, 2, 2, 3, 3])[permutation],
                      semantic_labels=np.array([4, 4, 4, 4, 4, 7, 7])[permutation])
    assert a["regions"] == b["regions"]


def test_part_diagnostics_separate_sibling_parts_from_other_objects():
    result, masks = diagnostic([1, .8, .7, .9, .6, .4, .3], label_kind="part",
                               parent_labels=np.array([9, 9, 9, 9, 9, 20, 20]))
    np.testing.assert_array_equal(np.flatnonzero(masks["sibling_parts"]), [3, 4])
    assert result["regions"]["sibling_parts"]["eligible_points"] == 2
    assert result["label_kind"] == "part"


def test_boundary_transition_respects_metric_radius_and_excludes_self_duplicates():
    labels = np.array([1, 1, 2, 2])
    points = np.array([[0., 0, 0], [.1, 0, 0], [.2, 0, 0], [10, 0, 0]])
    np.testing.assert_array_equal(boundary_regions(points, labels, k=3, radius=.11), [False, True, True, False])
    np.testing.assert_array_equal(boundary_regions(np.zeros((2, 3)), np.array([1, 2]), k=1), [True, True])


def test_common_eligibility_and_missing_targets_do_not_manufacture_scores():
    result, masks = diagnostic([1, .5, .5, .5, .5, .5, .5],
                               common_eligible_mask=np.array([False, False, False, True, True, True, True]))
    assert result["retrieval_depth"] == 0
    assert result["regions"]["other_instances"]["fraction_of_top_target_size"] is None
    assert result["status"] == "no eligible target points"
    assert masks["eligible"].sum() == 4
    undefined, _ = diagnostic([np.nan] * 7)
    assert undefined["eligible_points"] == 0
    json.dumps(undefined, allow_nan=False)


def test_suite_bridge_pins_pca_files_and_uses_same_leakage_population_across_models(tmp_path):
    points = geometry(7)
    scene = Scene("synthetic", points, labels=np.array([1, 1, 1, 2, 2, 3, 3]),
                  semantic_labels=np.array([4, 4, 4, 4, 4, 7, 7]))
    x = np.random.default_rng(3).normal(size=(7, 4))
    a = Observation("a", {"dec0": x}, metadata={"cache_sha256": "cache-a",
                    "epoch": 36, "artifact_identity": {"epoch": 36}, "token_ids": np.array([1, 1, 2, 3, 4, 5, 6])})
    b = Observation("b", {"dec0": x * 100}, metadata={"cache_sha256": "cache-b",
                    "epoch": 36, "artifact_identity": {"epoch": 36}, "token_ids": np.array([1, 2, 3, 1, 4, 5, 6])})
    panels, report_ = feature_research_panels(scene, [a, b], query_indices=[0],
                                              output_directory=tmp_path / "first", reference_name="a")
    for panel in panels:
        panel.validate(7)
    assert report_["queries"]["a/dec0/q0"]["eligible_points"] == 4
    assert report_["queries"]["a/dec0/q0"]["eligible_point_ids_sha256"] == report_["queries"]["b/dec0/q0"]["eligible_point_ids_sha256"]
    pca_files = {key: value["path"] for key, value in report_["pca"].items()}
    repeated, _ = feature_research_panels(scene, [a, b], query_indices=[0],
                                          output_directory=tmp_path / "second", reference_name="a",
                                          settings={"pca_states": pca_files})
    for before, after in zip(panels, repeated):
        np.testing.assert_array_equal(before.colors, after.colors)
    norms = [panel for panel in panels if panel.key.endswith("/norm")]
    assert norms[1].limits[1] == pytest.approx(norms[0].limits[1] * 100)
    assert "independent" in norms[0].caption
    assert "checkpoint identity incomplete" in report_["pca"]["a/dec0"]["source_record"]["identity_basis"]
    with pytest.raises(ValueError, match="epoch.*no substitution"):
        feature_research_panels(scene, [a], query_indices=[0], output_directory=tmp_path / "wrong",
                                settings={"requested_identities": {"a": {"epoch": 150}}})
    json.dumps(report_, allow_nan=False)


def test_invalid_pca_sampling_and_identity_fail_closed():
    x = np.ones((5, 2))
    with pytest.raises(ValueError, match="unique"):
        fit_frozen_pca(x, original_point_ids=np.zeros(5, dtype=int), source_record=record())
    with pytest.raises(ValueError, match="source_id"):
        fit_frozen_pca(x, original_point_ids=np.arange(5), source_record={"feature_level": "dec0"})
    with pytest.raises(ValueError, match="positive integer"):
        fit(x, max_fit_points=0)


def test_loaded_custom_scaling_quantiles_are_reflected_in_captions(tmp_path):
    scene = Scene("synthetic", geometry(6))
    features = np.random.default_rng(8).normal(size=(6, 3))
    observation = Observation("a", {"dec0": features}, metadata={"cache_sha256": "cache-a"})
    _, original = feature_research_panels(scene, [observation], query_indices=[],
                                           output_directory=tmp_path / "original")
    source = original["pca"]["a/dec0"]["source_record"]
    state = fit_frozen_pca(features, original_point_ids=np.arange(6), source_record=source,
                           quantiles=(.05, .95))
    path = tmp_path / "custom.npz"
    state.save(path)
    panels, _ = feature_research_panels(scene, [observation], query_indices=[],
                                        output_directory=tmp_path / "custom-run",
                                        settings={"pca_states": {"a/dec0": str(path)}})
    assert "5–95%" in panels[0].caption
    assert "95%" in panels[1].caption


@pytest.mark.parametrize("artifact_identity", [None, {"checkpoint_sha256": "actual-checkpoint"}])
def test_manifest_only_epoch_cannot_satisfy_exact_cache_identity(tmp_path, artifact_identity):
    metadata = {"adapter": "cache", "cache_sha256": "cache-a", "epoch": 36}
    if artifact_identity is not None:
        metadata["artifact_identity"] = artifact_identity
    observation = Observation("a", {"dec0": np.ones((3, 2))}, metadata=metadata)
    with pytest.raises(ValueError, match="epoch.*missing from artifact record"):
        feature_research_panels(Scene("fixture", geometry(3)), [observation], query_indices=[0],
            output_directory=tmp_path / "rejected", settings={"requested_identities": {"a": {"epoch": 36}}})
