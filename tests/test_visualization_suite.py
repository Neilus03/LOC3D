import json

import numpy as np
import pytest

from delimit3d.visualization.contracts import Mesh, Observation, Scene
from delimit3d.visualization.suite import (
    choose_queries, display_sample, export_ply, json_safe, run_suite,
)


def test_common_grid_diagnostic_exports_render_with_exact_point_correspondence(tmp_path):
    """The diagnostic CLI's mixed point/cell NPZ must survive the gallery adapter."""
    from delimit3d.evaluation.feature_diagnostics import evaluate_features
    from delimit3d.visualization.adapters import load_observation

    points = np.array([[0., 0., 0.], [.01, 0., 0.], [.1, 0., 0.], [.11, 0., 0.], [.2, 0., 0.]])
    inverse = np.array([0, 0, 1, 1, 2])
    labels = np.array([0, 0, 1, 1, -1])
    arrays, _ = evaluate_features(np.array([[1., 0.], [0., 1.], [1., 1.]]),
        inverse, inverse, labels, np.array([0, 2]),
        [{"index": 0, "vertices": 2}, {"index": 1, "vertices": 2}], seeds=(42,))
    np.savez(tmp_path / "diagnostics.npz", points=points, colors=np.full((5, 3), 128), labels=labels, **arrays)
    spec = {"kind": "cache", "name": "Synthetic oracle-K diagnostic", "path": "diagnostics.npz"}
    observation = load_observation(spec, Scene("fixture", points), tmp_path)
    np.testing.assert_array_equal(observation.prediction, arrays["point_prediction"])
    np.testing.assert_array_equal(observation.features["dec0"], arrays["features"][inverse])
    config = {"cpu_only": True, "formats": ["html"], "queries": [0],
        "visualizations": ["rgb", "labels", "pca", "similarity", "prediction"],
        "research": {"features": {"enabled": True}}, "models": [spec],
        "scenes": [{"scene_id": "fixture", "path": "diagnostics.npz", "color_range": "0_255",
                    "metadata": {"ignore_label_ids": [-1], "synthetic": True}}]}
    result = run_suite(config, base_dir=tmp_path, output=tmp_path / "gallery")
    assert result["status"] == "complete", result["errors"]
    assert result["completed_scenes"] == 1


def test_shared_sample_keeps_queries_without_changing_requested_limit():
    first = display_sample(10000, 99, [2, 400, 9999], seed=8)
    second = display_sample(10000, 99, [9999, 2, 400], seed=8)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 99
    assert {2, 400, 9999} <= set(first)
    with pytest.raises(ValueError):
        display_sample(100, 1, [2, 3], seed=8)


def test_queries_do_not_depend_on_model_or_labels():
    points = np.random.default_rng(4).normal(size=(500, 3))
    scene = Scene("same", points, labels=np.zeros(500, dtype=int))
    first = choose_queries(scene, {"count": 3}, seed=3)
    scene.labels[:] = np.arange(500)
    assert first == choose_queries(scene, {"count": 3}, seed=3)
    assert len(set(first)) == 3
    with pytest.raises(ValueError):
        choose_queries(scene, [500], seed=3)


def test_active_selection_and_part_labels_reach_scene_record(tmp_path):
    np.savez(tmp_path/'parts.npz',points=np.array([[0.,0.,0.],[1.,1.,1.]]),labels=np.array([1,2]))
    config={'formats':['html'],'queries':[],'visualizations':['labels'],'scenes':[{
        'scene_id':'parts','path':'parts.npz','selection':{'method':'effect_selected'},
        'metadata':{'label_kind':'part','selection':{'method':'original_anchor'}}}]}
    result=run_suite(config,base_dir=tmp_path,output=tmp_path/'out')
    assert result['status']=='complete',result['errors']
    record=json.loads((tmp_path/'out/000-parts/report.json').read_text())
    assert record['scene']['selection']==record['selection']=={'method':'effect_selected'}
    assert record['scene']['input_selection']=={'method':'original_anchor'}
    assert record['panels'][0]['title']=='GT parts'


def test_array_contract_prevents_silent_misalignment():
    scene = Scene("scene", np.zeros((4, 3))).validate()
    with pytest.raises(ValueError, match="shape"):
        Observation("bad", features={"dec0": np.zeros((3, 6))}).validate(scene)
    with pytest.raises(ValueError, match="integer"):
        Observation("bad", prediction=np.zeros(4)).validate(scene)


def test_record_hashes_large_arrays_instead_of_embedding_them():
    value = json_safe({"tokens": np.arange(1000), "unavailable": float("nan")})
    assert value["tokens"]["shape"] == [1000]
    assert len(value["tokens"]["sha256"]) == 64
    assert value["unavailable"] is None
    json.dumps(value, allow_nan=False)


def test_ply_export_preserves_order_and_rgb(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    export_ply(path, points, np.array([[1, 0, 0], [0, 1, 0]]))
    header, data = path.read_bytes().split(b"end_header\n")
    assert b"element vertex 2" in header
    vertices = np.frombuffer(data, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    np.testing.assert_array_equal(vertices["xyz"], points)
    np.testing.assert_array_equal(vertices["rgb"], [[255, 0, 0], [0, 255, 0]])


def test_mesh_ply_preserves_faces_and_original_point_ids(tmp_path):
    from plyfile import PlyData

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    path = tmp_path / "surface.ply"
    export_ply(path, points, np.eye(3), faces=faces, point_indices=np.array([5, 2, 9]))
    mesh = PlyData.read(str(path))
    np.testing.assert_array_equal(mesh["vertex"]["original_point_id"], [5, 2, 9])
    np.testing.assert_array_equal(np.stack(mesh["face"]["vertex_indices"]), faces)


def test_mesh_contract_rejects_bad_topology_and_mapping():
    scene = Scene("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float))
    with pytest.raises(ValueError, match="degenerate"):
        Mesh(scene.points, np.array([[0, 0, 2]]), np.arange(3)).validate(scene)
    with pytest.raises(ValueError, match="point index"):
        Mesh(scene.points, np.array([[0, 1, 2]]), np.array([0, 1, 3])).validate(scene)


def test_suite_mesh_html_and_ply_preserve_full_topology(tmp_path):
    from plyfile import PlyData

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    np.savez(tmp_path / "surface.npz", points=points, faces=faces, colors=np.full((4, 3), 0.4))
    config = {"formats": ["html", "ply"], "max_display_points": 2, "queries": [],
              "visualizations": ["rgb"], "mesh": {"kind": "scene"},
              "scenes": [{"scene_id": "surface", "path": "surface.npz"}]}
    output = tmp_path / "result"
    result = run_suite(config, base_dir=tmp_path, output=output)
    assert result["status"] == "complete", result["errors"]
    item = result["scenes"][0]
    scene_dir = output / item["directory"]
    record = json.loads((scene_dir / "report.json").read_text())
    assert record["display_points"] == 2
    assert record["mesh"]["vertices"] == 4
    assert record["mesh"]["faces"] == 2
    assert (scene_dir / "mesh" / item["files"]["mesh"]["html"]).is_file()
    exported = PlyData.read(str(scene_dir / "mesh" / item["files"]["mesh"]["ply"][0]))
    np.testing.assert_array_equal(np.stack(exported["face"]["vertex_indices"]), faces)
    assert "Explore mesh" in (output / "index.html").read_text()


def test_suite_real_npz_to_offline_gallery_and_failure_record(tmp_path):
    points = np.random.default_rng(42).normal(size=(50, 3)).astype(np.float32)
    np.savez(tmp_path / "scene.npz", points=points, colors=np.full((50, 3), 0.4))
    np.savez(tmp_path / "model.npz", points=points, features=points)
    config = {"formats": ["html"], "max_display_points": 17, "queries": [24],
              "visualizations": ["rgb", "pca", "similarity"],
              "scenes": [{"scene_id": "room", "path": "scene.npz"}],
              "models": [{"kind": "cache", "name": "Encoder", "path": "model.npz"}]}
    output = tmp_path / "result"
    result = run_suite(config, base_dir=tmp_path, output=output)
    assert result["status"] == "complete", result["errors"]
    item = result["scenes"][0]
    record = json.loads((output / item["directory"] / "report.json").read_text())
    assert record["models"][0]["layers"]["dec0"][0] == 50
    assert record["display_points"] == 17
    assert (output / "index.html").exists()
    assert (output / item["directory"] / "index.html").exists()
    with pytest.raises(FileExistsError):
        run_suite(config, base_dir=tmp_path, output=output)
    config["models"][0]["path"] = "missing.npz"
    failed = run_suite(config, base_dir=tmp_path, output=tmp_path / "failed")
    assert failed["status"] == "failed"
    assert failed["completed_scenes"] == 0
    assert len(failed["errors"]) == 1
    assert (tmp_path / "failed" / "run.json").exists()
