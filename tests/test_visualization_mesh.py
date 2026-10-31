"""Surface topology and point-color alignment must survive every import mode."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from delimit3d.visualization.adapters import geometry_sha256, load_scene
from delimit3d.visualization.contracts import Mesh, Scene
from delimit3d.visualization.mesh import attach_mesh


def test_float64_scene_and_original_mesh_keep_exact_coordinate_witness(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [1.1, 0.2, 0.3], [0.1, 1.2, 0.3]], dtype=np.float64)
    np.savez(tmp_path / "scene.npz", points=points)
    np.savez(tmp_path / "mesh.npz", points=points, faces=np.array([[0, 1, 2]]))
    scene = load_scene({"scene_id": "double_precision", "path": "scene.npz"}, tmp_path)
    mesh = attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)
    np.testing.assert_array_equal(scene.points, points)
    np.testing.assert_array_equal(mesh.vertices, points)
    assert mesh.metadata["mapping_max_distance"] == 0


@pytest.fixture
def scene() -> Scene:
    return Scene("square", np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32))


@pytest.fixture
def faces() -> np.ndarray:
    return np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)


def test_original_mesh_preserves_topology_and_hashes(tmp_path: Path, scene: Scene, faces) -> None:
    np.savez(tmp_path / "mesh.npz", vertices=scene.points, faces=faces)
    mesh = attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(mesh.point_indices, np.arange(4))
    assert mesh.metadata["source_kind"] == "original"
    assert mesh.metadata["geometry_sha256"] == geometry_sha256(scene.points)
    assert mesh.metadata["source_scene_geometry_sha256"] == geometry_sha256(scene.points)
    assert len(mesh.metadata["source_files"][str(tmp_path / "mesh.npz")]) == 64
    assert len(mesh.metadata["faces_sha256"]) == 64
    assert len(mesh.metadata["point_indices_sha256"]) == 64
    assert mesh.metadata["uncovered_scene_points"] == 0
    assert mesh.metadata["scene_points_without_faces"] == 0


def test_reordered_vertices_require_explicit_matching_map(tmp_path: Path, scene: Scene, faces) -> None:
    order = np.array([3, 1, 2, 0])
    np.savez(tmp_path / "mesh.npz", vertices=scene.points[order], faces=faces)
    with pytest.raises(ValueError, match="geometry/order differs"):
        attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)
    np.save(tmp_path / "map.npy", order)
    mesh = attach_mesh(scene, {"path": "mesh.npz", "vertex_to_point": "map.npy"}, tmp_path)
    np.testing.assert_array_equal(mesh.vertices, scene.points[mesh.point_indices])
    np.testing.assert_array_equal(mesh.faces, faces)
    assert mesh.metadata["alignment_check"] == "vertex_to_point"
    assert str(tmp_path / "map.npy") in mesh.metadata["source_files"]


@pytest.mark.parametrize("mapping,match", [
    ([0, 1, 2, 9], "out-of-range"),
    ([0, -1, 2, 3], "out-of-range"),
    ([0., 1., 2., 3.], "integer array"),
    ([0, 1, 2], "shape"),
    ([3, 2, 1, 0], "geometry/order differs"),
])
def test_invalid_explicit_mapping_is_rejected(tmp_path: Path, scene: Scene, faces, mapping, match) -> None:
    np.savez(tmp_path / "mesh.npz", vertices=scene.points, faces=faces)
    np.save(tmp_path / "map.npy", mapping)
    with pytest.raises(ValueError, match=match):
        attach_mesh(scene, {"path": "mesh.npz", "vertex_to_point": "map.npy"}, tmp_path)


def test_geometry_roundoff_requires_explicit_tolerance(tmp_path: Path, scene: Scene, faces) -> None:
    points = scene.points.copy()
    points[0, 2] = 1e-5
    np.savez(tmp_path / "mesh.npz", vertices=points, faces=faces)
    with pytest.raises(ValueError, match="geometry/order differs"):
        attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)
    mesh = attach_mesh(scene, {"path": "mesh.npz", "geometry_atol": 2e-5}, tmp_path)
    assert 0 < mesh.metadata["mapping_max_distance"] < 2e-5


def test_nearest_mapping_requires_explicit_distance_and_checks_every_vertex(tmp_path: Path, scene: Scene) -> None:
    order = np.array([2, 0, 1])
    np.savez(tmp_path / "mesh.npz", vertices=scene.points[order] + .01, faces=[[0, 1, 2]])
    with pytest.raises(ValueError, match="explicit max_distance"):
        attach_mesh(scene, {"path": "mesh.npz", "mapping": "nearest"}, tmp_path)
    with pytest.raises(ValueError, match="3/3 vertices beyond max_distance"):
        attach_mesh(scene, {"path": "mesh.npz", "mapping": "nearest", "max_distance": .005}, tmp_path)
    mesh = attach_mesh(scene, {"path": "mesh.npz", "mapping": "nearest", "max_distance": .02}, tmp_path)
    np.testing.assert_array_equal(mesh.point_indices, order)
    assert mesh.metadata["uncovered_mesh_vertices"] == 0
    assert mesh.metadata["uncovered_scene_points"] == 1
    assert mesh.metadata["scene_points_without_faces"] == 1
    assert mesh.metadata["max_distance"] == .02


def test_mesh_subset_without_mapping_rejected(tmp_path: Path, scene: Scene) -> None:
    np.savez(tmp_path / "mesh.npz", vertices=scene.points[:3], faces=[[0, 1, 2]])
    with pytest.raises(ValueError, match="different vertex count"):
        attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)


@pytest.mark.parametrize("triangles,match", [
    ([[0, 1, 4]], "out-of-range"),
    ([[-1, 1, 2]], "out-of-range"),
    ([[0, 0, 2]], "degenerate"),
    ([[0., 1., 2.]], "integer"),
    ([[0, 1, 2, 3]], "triangles"),
    (np.empty((0, 3), dtype=int), "nonempty"),
])
def test_invalid_surface_topology_is_rejected(tmp_path: Path, scene: Scene, triangles, match) -> None:
    np.savez(tmp_path / "mesh.npz", vertices=scene.points, faces=triangles)
    with pytest.raises(ValueError, match=match):
        attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)


def test_distinct_collinear_vertices_are_degenerate() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    scene = Scene("line", points)
    with pytest.raises(ValueError, match="degenerate"):
        Mesh(points, np.array([[0, 1, 2]]), np.arange(3)).validate(scene)


def test_embedded_scene_mesh_uses_named_fields_and_same_scale(tmp_path: Path, scene: Scene, faces) -> None:
    np.savez(tmp_path / "scene.npz", xyz=scene.points * 100, triangles=faces)
    loaded = load_scene({"path": "scene.npz", "fields": {"points": "xyz", "faces": "triangles"},
                         "coordinate_scale": .01}, tmp_path)
    mesh = attach_mesh(loaded, {"kind": "scene"}, tmp_path)
    np.testing.assert_array_equal(mesh.vertices, loaded.points)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert mesh.metadata["coordinate_scale"] == .01


def test_external_mesh_does_not_inherit_scene_source_scale(tmp_path: Path, scene: Scene, faces) -> None:
    scene.metadata["coordinate_scale"] = .01
    np.savez(tmp_path / "mesh.npz", vertices=scene.points, faces=faces)
    mesh = attach_mesh(scene, {"path": "mesh.npz"}, tmp_path)
    assert mesh.metadata["coordinate_scale"] == 1.0


def test_directory_mesh_loads_original_triangles(tmp_path: Path, scene: Scene, faces) -> None:
    np.save(tmp_path / "coords.npy", scene.points)
    np.save(tmp_path / "faces.npy", faces)
    scene.metadata["source_path"] = str(tmp_path)
    mesh = attach_mesh(scene, {"kind": "scene"}, tmp_path)
    np.testing.assert_array_equal(mesh.faces, faces)
    assert len(mesh.metadata["source_files"]) == 2


def test_original_ply_mesh_preserves_source_face_order(tmp_path: Path, scene: Scene, faces) -> None:
    plyfile = pytest.importorskip("plyfile")
    vertices = np.empty(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = scene.points[:, index]
    triangles = np.empty(2, dtype=[("vertex_indices", "i4", (3,))])
    triangles["vertex_indices"] = faces[::-1]
    plyfile.PlyData([
        plyfile.PlyElement.describe(vertices, "vertex"),
        plyfile.PlyElement.describe(triangles, "face"),
    ]).write(str(tmp_path / "original.ply"))
    mesh = attach_mesh(scene, {"path": "original.ply"}, tmp_path)
    np.testing.assert_array_equal(mesh.faces, faces[::-1])
    np.testing.assert_array_equal(mesh.vertices, scene.points)


def test_ply_polygons_are_not_silently_triangulated(tmp_path: Path, scene: Scene) -> None:
    path = tmp_path / "quad.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\n"
                    "property float y\nproperty float z\nelement face 1\n"
                    "property list uchar int vertex_indices\nend_header\n"
                    "0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
    with pytest.raises(ValueError, match="not silently triangulated"):
        attach_mesh(scene, {"path": str(path)}, tmp_path)


def test_ball_pivoting_requires_normals_and_explicit_radii(tmp_path: Path, scene: Scene) -> None:
    with pytest.raises(ValueError, match="requires scene normals"):
        attach_mesh(scene, {"kind": "ball_pivoting", "radii": [.1]}, tmp_path)
    scene.normals = np.tile([0., 0., 1.], (4, 1))
    with pytest.raises(ValueError, match="explicit positive radii"):
        attach_mesh(scene, {"kind": "ball_pivoting"}, tmp_path)
    with pytest.raises(ValueError, match="finite positive"):
        attach_mesh(scene, {"kind": "ball_pivoting", "radii": [0]}, tmp_path)
    scene.normals[0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        attach_mesh(scene, {"kind": "ball_pivoting", "radii": [.1]}, tmp_path)


def test_missing_open3d_has_actionable_error(tmp_path: Path, scene: Scene, monkeypatch) -> None:
    scene.normals = np.tile([0., 0., 1.], (4, 1))
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(ImportError, match="sources"):
        attach_mesh(scene, {"kind": "ball_pivoting", "radii": [.1]}, tmp_path)


def test_ball_pivoting_checks_generated_vertex_alignment(tmp_path: Path, scene: Scene, faces, monkeypatch) -> None:
    scene.normals = np.tile([0., 0., 2.], (4, 1))
    recorded = {}
    def reconstruct(cloud, radii):
        np.testing.assert_array_equal(cloud.normals, np.tile([0., 0., 1.], (4, 1)))
        recorded["radii"] = radii
        return SimpleNamespace(vertices=cloud.points, triangles=faces)
    module = SimpleNamespace(
        __version__="test",
        geometry=SimpleNamespace(PointCloud=SimpleNamespace,
            TriangleMesh=SimpleNamespace(create_from_point_cloud_ball_pivoting=reconstruct)),
        utility=SimpleNamespace(Vector3dVector=np.asarray, DoubleVector=list),
    )
    monkeypatch.setitem(sys.modules, "open3d", module)
    mesh = attach_mesh(scene, {"kind": "ball_pivoting", "radii": [.5, 1.]}, tmp_path)
    assert recorded["radii"] == [.5, 1.]
    assert mesh.metadata["source_kind"] == "reconstructed"
    np.testing.assert_array_equal(mesh.point_indices, np.arange(4))
    module.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting = lambda cloud, radii: SimpleNamespace(
        vertices=cloud.points[::-1], triangles=faces,
    )
    with pytest.raises(ValueError, match="unexpectedly changed vertex order"):
        attach_mesh(scene, {"kind": "ball_pivoting", "radii": [.5]}, tmp_path)
