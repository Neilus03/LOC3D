"""Verified surface topology for point-aligned visualization panels.

Dataset triangles are preferable to reconstructed surfaces. Every mesh vertex
must have an explicit, checked correspondence to an original scene point, so
feature colors, predictions and clicks keep their original meaning. Point-only
inputs are reconstructed only when ball pivoting is explicitly requested.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import _field, _scaled_geometry, geometry_sha256, resolve_path, sha256_file
from .contracts import Mesh, Scene


def _integer_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8", order="C").tobytes()).hexdigest()


def _finite_nonnegative(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _vertices(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("Mesh vertices must have nonempty shape [V, 3]")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Mesh vertices must contain finite numeric coordinates")
    return array


def _triangles(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError("Mesh faces must be nonempty triangles of shape [F, 3]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Mesh faces must contain integer vertex indices")
    return array


def _read_surface(
    path: Path, fields: dict[str, Any], *, coordinate_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Read original triangles without changing their order or connectivity."""
    if path.is_dir():
        payload = {}
        # Only read mesh arrays; feature and training arrays can be very large.
        selectors = [fields.get("vertices", fields.get("points")), fields.get("faces")]
        names = {"points", "vertices", "coords", "faces"}
        for selector in selectors:
            if isinstance(selector, str):
                names.add(selector)
            elif isinstance(selector, dict):
                names.add(str(selector["key"]))
            elif isinstance(selector, (list, tuple)):
                names.update(map(str, selector))
        files = []
        for name in sorted(names):
            candidate = path / f"{name}.npy"
            if candidate.is_file():
                payload[name] = np.load(candidate, allow_pickle=False)
                files.append(candidate)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        files = [path]
    elif path.suffix.lower() == ".ply":
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("PLY meshes require plyfile; install the project dependencies") from exc
        ply = PlyData.read(str(path))
        try:
            vertex = ply["vertex"].data
            polygon = ply["face"].data
        except KeyError as exc:
            raise ValueError(f"{path}: PLY must contain vertex and face elements") from exc
        payload = {name: np.asarray(vertex[name]) for name in vertex.dtype.names}
        if all(name in payload for name in ("x", "y", "z")):
            payload["vertices"] = np.column_stack([payload[name] for name in ("x", "y", "z")])
        face_field = fields.get("faces", "vertex_indices")
        if not isinstance(face_field, str) or face_field not in polygon.dtype.names:
            raise ValueError(f"{path}: PLY face field {face_field!r} is missing")
        values = polygon[face_field]
        if not len(values) or any(np.asarray(face).shape != (3,) for face in values):
            raise ValueError("PLY faces must be nonempty triangles; polygons are not silently triangulated")
        payload["faces"] = np.stack(values)
        fields = {**fields, "faces": "faces"}
        files = [path]
    else:
        raise ValueError("Original mesh path must be a PLY, NPZ, or directory of NPY arrays")
    vertex_field = fields.get("vertices", fields.get("points"))
    if vertex_field is None:
        vertex_field = next((name for name in ("points", "vertices", "coords") if name in payload), None)
    if vertex_field is None:
        raise ValueError(f"{path}: no mesh vertices; configure fields.vertices")
    try:
        vertices = _vertices(_field(payload, vertex_field))
        faces = _triangles(_field(payload, fields.get("faces", "faces")))
    except KeyError as exc:
        raise ValueError(f"{path}: mesh field {exc.args[0]!r} is missing") from exc
    # Match the scene loader while preserving stored float64 coordinates.
    vertices = _scaled_geometry(vertices, coordinate_scale)
    return vertices, faces, files


def _correspondence(
    vertices: np.ndarray, scene: Scene, spec: dict[str, Any], base_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], list[Path]]:
    tolerance = _finite_nonnegative(spec.get("geometry_atol", 0.0), "geometry_atol")
    method = spec.get("mapping", "ordered")
    files: list[Path] = []
    if "vertex_to_point" in spec:
        if method != "ordered":
            raise ValueError("Choose vertex_to_point or mapping: nearest, not both")
        path = resolve_path(spec["vertex_to_point"], base_dir)
        if path.suffix.lower() != ".npy":
            raise ValueError("vertex_to_point must name an integer NPY array")
        indices = np.load(path, allow_pickle=False)
        if indices.shape != (len(vertices),) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("vertex_to_point must be an integer array of shape [V]")
        if np.any(indices < 0) or np.any(indices >= len(scene.points)):
            raise ValueError("vertex_to_point contains out-of-range scene point indices")
        indices = indices.astype(np.int64, copy=False)
        method = "vertex_to_point"
        files = [path]
    elif method == "nearest":
        if "max_distance" not in spec:
            raise ValueError("mapping: nearest requires an explicit max_distance in scene coordinate units")
        threshold = _finite_nonnegative(spec["max_distance"], "max_distance")
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(scene.points).query(vertices, k=1, workers=1)
        uncovered = ~np.isfinite(distances) | (distances > threshold)
        if uncovered.any():
            raise ValueError(
                f"Nearest mesh mapping leaves {int(uncovered.sum())}/{len(vertices)} vertices "
                f"beyond max_distance={threshold:g}; largest distance={float(distances.max()):g}"
            )
        indices = indices.astype(np.int64, copy=False)
    elif method == "ordered":
        if len(vertices) != len(scene.points):
            raise ValueError(
                "Ordered mesh geometry has a different vertex count from the scene; "
                "provide vertex_to_point or explicit mapping: nearest with max_distance"
            )
        indices = np.arange(len(vertices), dtype=np.int64)
    else:
        raise ValueError(f"Unknown mesh mapping {method!r}; use ordered or nearest")
    expected = np.asarray(scene.points)[indices]
    if method != "nearest" and not np.allclose(vertices, expected, atol=tolerance, rtol=0):
        raise ValueError(
            "Mesh geometry/order differs from mapped scene points; provide correct vertex_to_point "
            "or an explicit geometry_atol for known coordinate roundoff"
        )
    distances = np.linalg.norm(vertices.astype(np.float64) - expected, axis=1)
    unique_count = len(np.unique(indices))
    metadata = {
        "alignment_check": method,
        "geometry_atol": tolerance,
        "mapping_max_distance": float(distances.max()),
        "mapped_vertex_count": len(vertices),
        "unique_mapped_scene_points": unique_count,
        "uncovered_scene_points": len(scene.points) - unique_count,
        "uncovered_mesh_vertices": 0,
    }
    if method == "nearest":
        metadata["max_distance"] = threshold
    return indices, metadata, files


def _ball_pivoting(scene: Scene, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if scene.normals is None:
        raise ValueError("Ball pivoting requires scene normals; supply oriented normals in the scene")
    normals = np.asarray(scene.normals, dtype=np.float64)
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm == 0):
        raise ValueError("Ball pivoting requires nonzero oriented normals for every scene point")
    if "radii" not in spec:
        raise ValueError("Ball pivoting requires explicit positive radii in scene coordinate units")
    radii = np.asarray(spec["radii"], dtype=float)
    if radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Ball pivoting radii must be a nonempty list of finite positive values")
    if np.any(np.diff(radii) <= 0):
        raise ValueError("Ball pivoting radii must be strictly increasing")
    if "vertex_to_point" in spec or spec.get("mapping", "ordered") != "ordered":
        raise ValueError("Ball pivoting retains original point order; external vertex mappings do not apply")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Ball pivoting requires optional Open3D; install with pip install -e '.[sources]' "
            "or provide an original dataset mesh"
        ) from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(scene.points, dtype=np.float64))
    cloud.normals = o3d.utility.Vector3dVector(normals / norm[:, None])
    surface = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii.tolist()),
    )
    vertices = _vertices(np.asarray(surface.vertices)).copy()
    triangles = np.asarray(surface.triangles)
    if not len(triangles):
        raise ValueError("Ball pivoting produced no triangles; check point spacing, oriented normals and radii")
    if not np.array_equal(vertices, scene.points):
        raise ValueError("Ball pivoting unexpectedly changed vertex order; refusing unverified color transfer")
    return vertices, _triangles(triangles).copy(), {
        "source_kind": "reconstructed", "reconstruction": "ball_pivoting",
        "radii": radii.tolist(), "open3d_version": o3d.__version__,
        "normals_sha256": geometry_sha256(scene.normals),
        "normals_normalized_for_reconstruction": True,
    }


def attach_mesh(scene: Scene, spec: dict[str, Any], base_dir: Path) -> Mesh:
    """Build checked surface geometry; the caller assigns it to ``scene.mesh``.

    ``kind: scene`` uses the source file's original triangle topology, honoring
    scene field names and coordinate scale. ``kind: original`` reads ``path``
    (PLY, NPZ, or a directory containing vertices/points and faces NPY arrays).
    ``vertex_to_point`` may name an explicit NPY map; ``mapping: nearest`` is
    opt-in and requires ``max_distance``. ``kind: ball_pivoting`` requires
    oriented scene normals and explicit ``radii``; no holes are filled.
    """
    scene.validate()
    kind = spec.get("kind", "original")
    if kind == "ball_pivoting":
        vertices, faces, metadata = _ball_pivoting(scene, spec)
        files = []
        metadata["source_files"] = dict(scene.metadata.get("source_files", {}))
        metadata["source_path"] = scene.metadata.get("source_path")
    elif kind in {"original", "scene"}:
        source = scene.metadata.get("source_path") if kind == "scene" else spec.get("path")
        if source is None:
            raise ValueError(f"Mesh kind {kind!r} needs {'scene source_path metadata' if kind == 'scene' else 'path'}")
        path = resolve_path(source, base_dir)
        fields = dict(scene.metadata.get("fields", {})) if kind == "scene" else {}
        fields.update(spec.get("fields", {}))
        default_scale = scene.metadata.get("coordinate_scale", 1.0) if kind == "scene" else 1.0
        scale = float(spec.get("coordinate_scale", default_scale))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Mesh coordinate_scale must be finite and positive")
        vertices, faces, files = _read_surface(path, fields, coordinate_scale=scale)
        metadata = {
            "source_kind": "original", "source_path": str(path), "source_files": {},
            "coordinate_scale": scale, "fields": fields,
        }
    else:
        raise ValueError(f"Unknown mesh kind {kind!r}; use original, scene, or ball_pivoting")
    indices, correspondence, mapping_files = _correspondence(vertices, scene, spec, base_dir)
    metadata.update(correspondence)
    metadata["source_files"].update({str(path): sha256_file(path) for path in files + mapping_files})
    metadata.update({
        "kind": kind,
        "geometry_sha256": geometry_sha256(vertices),
        "faces_sha256": _integer_hash(faces),
        "point_indices_sha256": _integer_hash(indices),
        "source_scene_geometry_sha256": geometry_sha256(scene.points),
    })
    mesh = Mesh(vertices=vertices, faces=faces, point_indices=indices, metadata=metadata).validate(scene)
    used_vertices = np.unique(mesh.faces)
    covered_scene_points = len(np.unique(mesh.point_indices[used_vertices]))
    mesh.metadata.update({
        "vertices_used_in_faces": len(used_vertices),
        "scene_points_used_in_faces": covered_scene_points,
        "scene_points_without_faces": len(scene.points) - covered_scene_points,
    })
    return mesh
