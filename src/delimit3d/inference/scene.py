"""Unlabelled scene IO. All returned arrays preserve the uploaded point order."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import zipfile

import numpy as np


@dataclass
class Scene:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray | None = None
    segments: np.ndarray | None = None
    name: str = "scene"
    sha256: str = ""

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        n = len(self.points)
        if n == 0 or self.points.shape != (n, 3) or not np.isfinite(self.points).all():
            raise ValueError("points must be a nonempty finite N x 3 array in metres")
        colors = np.asarray(self.colors)
        if colors.shape != (n, 3) or not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 255:
            raise ValueError("colors must be N x 3 RGB in [0,255] or float RGB in [0,1]")
        if np.issubdtype(colors.dtype, np.floating) and colors.max() <= 1:
            colors = colors * 255
        self.colors = np.rint(colors).astype(np.uint8)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32)
            if self.normals.shape != (n, 3) or not np.isfinite(self.normals).all():
                raise ValueError("normals must be a finite N x 3 array aligned to points")
            lengths = np.linalg.norm(self.normals, axis=1)
            if np.any(lengths < 1e-6) or np.any(np.abs(lengths - 1) > .15):
                raise ValueError("normals must be nonzero unit vectors; preserve the model's normal orientation")
        if self.segments is not None:
            segments = np.asarray(self.segments)
            if segments.shape != (n,) or not np.issubdtype(segments.dtype, np.integer) or np.any(segments < 0):
                raise ValueError("segments must be a nonnegative integer array of length N")
            self.segments = segments.astype(np.int64)
        if not self.sha256:
            digest = hashlib.sha256()
            for array in (self.points, self.colors, self.normals, self.segments):
                if array is not None:
                    digest.update(str(array.shape).encode())
                    digest.update(array.tobytes())
            self.sha256 = digest.hexdigest()

    def preview(self, limit=50000):
        ids = np.linspace(0, len(self.points) - 1, min(limit, len(self.points)), dtype=np.int64)
        return {"name": self.name, "point_count": len(self.points),
                "points": self.points[ids].tolist(), "colors": self.colors[ids].tolist(),
                "point_ids": ids.tolist(), "has_normals": self.normals is not None,
                "has_segments": self.segments is not None}, ids


def load_scene(source: str | Path | bytes, *, filename=None, max_points=1000000, max_bytes=128*1024*1024):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise ValueError("Scene file exceeds the upload size limit")
        filename = path.name
        content = path.read_bytes()
    else:
        content = source
    if len(content) > max_bytes:
        raise ValueError("Scene file exceeds the upload size limit")
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".npz":
        # Bound decompression before NumPy allocates arrays, including ZIP bombs.
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > max_bytes * 4:
                raise ValueError("Expanded NPZ exceeds the scene size limit")
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            if not {"points", "colors"}.issubset(data.files):
                raise ValueError("NPZ requires points and colors; RGBN6 models also require normals")
            values = {key: data[key] for key in ("points", "colors", "normals", "segments") if key in data}
    elif suffix == ".ply":
        header = content[:65536].split(b"end_header", 1)
        if len(header) != 2:
            raise ValueError("PLY header is missing or exceeds 64 KiB")
        for line in header[0].decode("ascii").splitlines():
            if line.startswith("element "):
                fields = line.split()
                if len(fields) != 3 or int(fields[2]) < 0 or int(fields[2]) > max_points * 10:
                    raise ValueError("PLY element count exceeds the scene size limit")
                if fields[1] == "vertex" and int(fields[2]) > max_points:
                    raise ValueError(f"Scene exceeds {max_points:,} points")
        from plyfile import PlyData
        vertex = PlyData.read(io.BytesIO(content))["vertex"].data
        fields = set(vertex.dtype.names)
        if not {"x", "y", "z", "red", "green", "blue"}.issubset(fields):
            raise ValueError("PLY vertices require x/y/z and red/green/blue properties")
        values = {"points": np.column_stack([vertex[k] for k in ("x", "y", "z")]),
                  "colors": np.column_stack([vertex[k] for k in ("red", "green", "blue")])}
        if {"nx", "ny", "nz"}.issubset(fields):
            values["normals"] = np.column_stack([vertex[k] for k in ("nx", "ny", "nz")])
        if "segment_id" in fields:
            values["segments"] = vertex["segment_id"]
    else:
        raise ValueError("Choose an NPZ or PLY scene; checkpoint uploads are not accepted")
    if len(values["points"]) > max_points:
        raise ValueError(f"Scene exceeds {max_points:,} points")
    return Scene(**values, name=Path(filename).name, sha256=hashlib.sha256(content).hexdigest())
