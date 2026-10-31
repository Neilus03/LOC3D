"""Array contracts shared by dataset, model, analysis, and rendering adapters.

Every observation is aligned to the original scene point order. Display sampling
is deliberately a later operation; it must never change encoder inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array(value: Any, name: str, shape: tuple[int | None, ...], *, integer=False):
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite numeric values")
    if integer and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name}: expected integer IDs")
    return array


@dataclass
class Mesh:
    """Surface geometry with an explicit vertex-to-original-point correspondence."""

    vertices: np.ndarray
    faces: np.ndarray
    point_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, scene: "Scene") -> "Mesh":
        self.vertices = _array(self.vertices, "mesh.vertices", (None, 3))
        self.faces = _array(self.faces, "mesh.faces", (None, 3), integer=True)
        self.point_indices = _array(self.point_indices, "mesh.point_indices",
                                    (len(self.vertices),), integer=True)
        if len(self.vertices) < 3 or not len(self.faces):
            raise ValueError("A mesh needs at least three vertices and one triangle")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("Mesh faces contain an out-of-range vertex index")
        if self.point_indices.min() < 0 or self.point_indices.max() >= len(scene.points):
            raise ValueError("Mesh point_indices contain an out-of-range original point index")
        for start in range(0, len(self.faces), 65536):
            triangles = self.vertices[self.faces[start:start + 65536]].astype(np.float64)
            area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                           triangles[:, 2] - triangles[:, 0]), axis=1)
            if np.any(area == 0):
                raise ValueError("Mesh contains degenerate triangles with zero area")
        return self


@dataclass
class Scene:
    scene_id: str
    points: np.ndarray
    colors: np.ndarray | None = None
    normals: np.ndarray | None = None
    labels: np.ndarray | None = None
    semantic_labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mesh: Mesh | None = None

    def validate(self) -> "Scene":
        self.points = _array(self.points, "points", (None, 3))
        if not self.scene_id or len(self.points) == 0:
            raise ValueError("A scene needs a nonempty ID and at least one point")
        n = len(self.points)
        for name in ("colors", "normals"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _array(value, name, (n, 3)))
        if self.colors is not None and (self.colors.min() < 0 or self.colors.max() > 1):
            raise ValueError("colors must be normalized to [0, 1]")
        for name in ("labels", "semantic_labels"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _array(value, name, (n,), integer=True))
        if self.mesh is not None:
            self.mesh.validate(self)
        return self


@dataclass
class Observation:
    name: str
    features: dict[str, np.ndarray] = field(default_factory=dict)
    prediction: np.ndarray | None = None
    confidence: np.ndarray | None = None
    steps: np.ndarray | None = None
    clicks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, scene: Scene | int) -> "Observation":
        n = len(scene.points) if isinstance(scene, Scene) else int(scene)
        if not self.name:
            raise ValueError("An observation needs a nonempty model name")
        if not isinstance(self.features, dict):
            raise ValueError("features must map layer names to [N, C] arrays")
        for name, value in self.features.items():
            array = _array(value, f"features[{name}]", (n, None))
            if array.shape[1] == 0:
                raise ValueError(f"features[{name}]: empty feature dimension")
            self.features[name] = array
        if self.prediction is not None:
            self.prediction = _array(self.prediction, "prediction", (n,), integer=True)
        if self.confidence is not None:
            self.confidence = _array(self.confidence, "confidence", (n,))
        if self.steps is not None:
            self.steps = _array(self.steps, "steps", (None, n), integer=True)
            if len(self.steps) == 0:
                raise ValueError("steps must contain at least one prediction")
        for click in self.clicks:
            index = click.get("point_index", click.get("index"))
            if not isinstance(index, (int, np.integer)) or not 0 <= index < n:
                raise ValueError("Every click needs an in-range integer point_index")
        if not self.features and all(
            value is None for value in (self.prediction, self.confidence, self.steps)
        ):
            raise ValueError(f"{self.name}: no features or predictions supplied")
        return self


@dataclass
class Panel:
    key: str
    title: str
    colors: np.ndarray
    caption: str = ""
    values: np.ndarray | None = None
    cmap: str | None = None
    limits: tuple[float, float] | None = None
    query_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, n: int) -> "Panel":
        self.colors = _array(self.colors, f"{self.key}.colors", (n, 3))
        if self.colors.min() < 0 or self.colors.max() > 1:
            raise ValueError(f"{self.key}: colors must lie in [0, 1]")
        if self.values is not None:
            self.values = _array(self.values, f"{self.key}.values", (n,))
        if self.query_index is not None and not 0 <= self.query_index < n:
            raise ValueError(f"{self.key}: query index out of range")
        return self
