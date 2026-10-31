"""Small deterministic synthetic fixtures; these are never research results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_demo(directory: Path, *, with_mesh: bool = False) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Demo inputs already exist; choose a fresh output name: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    scenes = []
    for scene_id in ("synthetic_room", "synthetic_cabinet"):
        points, colors, normals, labels, faces = [], [], [], [], []

        def box(center, extent, label, color, count=180):
            if with_mesh:
                resolution = max(3, int(np.ceil(np.sqrt(count / 6))))
                u, v = np.meshgrid(np.linspace(-0.5, 0.5, resolution),
                                   np.linspace(-0.5, 0.5, resolution), indexing="ij")
                surfaces, surface_normals, triangles = [], [], []
                offset = sum(len(value) for value in points)
                for side in range(6):
                    axis, sign = side // 2, (-1 if side % 2 == 0 else 1)
                    other = [i for i in range(3) if i != axis]
                    grid = np.zeros((resolution ** 2, 3))
                    grid[:, axis] = sign * 0.5
                    grid[:, other[0]], grid[:, other[1]] = u.ravel(), v.ravel()
                    normal = np.zeros_like(grid)
                    normal[:, axis] = sign
                    local = []
                    for i in range(resolution - 1):
                        for j in range(resolution - 1):
                            a = i * resolution + j
                            local.extend([[a, a + resolution, a + 1],
                                          [a + 1, a + resolution, a + resolution + 1]])
                    local = np.asarray(local, dtype=np.int64)
                    triangle = grid[local[0]]
                    outward = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                    if outward[axis] * sign < 0:
                        local = local[:, [0, 2, 1]]
                    triangles.append(local + offset + side * resolution ** 2)
                    surfaces.append(grid)
                    surface_normals.append(normal)
                xyz, normal = np.concatenate(surfaces), np.concatenate(surface_normals)
                count = len(xyz)
                faces.extend(triangles)
            else:
                xyz = rng.uniform(-0.5, 0.5, (count, 3))
                sides = rng.integers(0, 6, count)
                normal = np.zeros_like(xyz)
                for side in range(6):
                    selected = sides == side
                    xyz[selected, side // 2] = -0.5 if side % 2 == 0 else 0.5
                    normal[selected, side // 2] = -1 if side % 2 == 0 else 1
            points.append(xyz * np.asarray(extent) + np.asarray(center))
            normals.append(normal)
            colors.append(np.clip(np.asarray(color) + rng.normal(0, 0.025, (count, 3)), 0, 1))
            labels.append(np.full(count, label, dtype=np.int64))

        if scene_id == "synthetic_room":
            box((0, 0, -0.05), (5.4, 4.2, 0.08), 0, (0.6, 0.63, 0.64), 450)
            box((0, 1.6, 1.0), (5.4, 0.05, 2.1), 0, (0.82, 0.84, 0.8), 400)
            box((0, 0, 0.76), (1.6, 0.85, 0.1), 1, (0.68, 0.44, 0.27), 300)
            for x in (-0.68, 0.68):
                for y in (-0.32, 0.32):
                    box((x, y, 0.36), (0.08, 0.08, 0.74), 1, (0.38, 0.3, 0.24), 50)
            for x, label in ((-1.35, 2), (1.35, 3)):
                box((x, 0, 0.44), (0.6, 0.66, 0.1), label, (0.15, 0.55, 0.58), 180)
                box((x, 0.29, 0.82), (0.6, 0.08, 0.75), label, (0.16, 0.5, 0.53), 220)
                for y in (-0.22, 0.22):
                    box((x, y, 0.2), (0.48, 0.04, 0.4), label, (0.22, 0.25, 0.28), 80)
            box((1.85, 1.0, 0.75), (0.8, 0.5, 1.5), 4, (0.78, 0.64, 0.39), 350)
        else:
            box((0, 0.1, 0.9), (2.3, 0.7, 1.8), 1, (0.42, 0.48, 0.55), 500)
            box((-0.58, -0.31, 0.95), (1.05, 0.09, 1.6), 2, (0.23, 0.65, 0.58), 380)
            box((0.57, -0.31, 1.39), (1.04, 0.09, 0.65), 3, (0.84, 0.58, 0.3), 260)
            box((0.57, -0.52, 0.69), (1.04, 0.5, 0.65), 4, (0.79, 0.43, 0.3), 350)
            for x, z, label in ((-0.2, 1, 5), (0.57, 1.39, 6), (0.57, 0.69, 7)):
                box((x, -0.59 if label == 7 else -0.39, z), (0.24, 0.04, 0.05), label, (0.15, 0.18, 0.22), 90)
        xyz = np.concatenate(points).astype(np.float32)
        rgb = np.concatenate(colors).astype(np.float32)
        normal = np.concatenate(normals).astype(np.float32)
        truth = np.concatenate(labels)
        semantic = truth.copy()
        if scene_id == "synthetic_room":
            semantic[truth == 3] = 2
        path = directory / f"{scene_id}.npz"
        surface = {"faces": np.concatenate(faces)} if with_mesh else {}
        np.savez_compressed(path, points=xyz, colors=rgb, normals=normal,
                            labels=truth, semantic_labels=semantic, **surface)
        basis = rng.normal(size=(int(truth.max()) + 1, 16))
        geometry = xyz @ rng.normal(size=(3, 16))
        coarse = xyz @ rng.normal(size=(3, 8))
        models = []
        for variant, strength in (("A", 0.35), ("B", 1.8)):
            features = (geometry + strength * basis[truth] + rng.normal(0, 0.15, geometry.shape)).astype(np.float32)
            prediction = truth * 7 + 10  # Arbitrary IDs ensure errors must use instance matching.
            error = rng.random(len(truth)) < (0.19 if variant == "A" else 0.06)
            prediction[error] = np.roll(prediction, 17)[error]
            steps = np.stack([np.where(rng.random(len(truth)) < rate, 99, prediction)
                              for rate in (0.45, 0.22, 0.0)])
            score = np.where(prediction == truth * 7 + 10, 0.88, 0.32).astype(np.float32)
            model_path = directory / f"{scene_id}_{variant}.npz"
            np.savez_compressed(model_path, points=xyz, features=features, coarse=coarse,
                                prediction=prediction, confidence=score, steps=steps)
            models.append({"kind": "cache", "name": f"Synthetic encoder {variant}",
                           "path": model_path.name, "features": {"fine": "features", "coarse": "coarse"},
                           "clicks": [{"point_index": int(np.flatnonzero(truth == 2)[offset]),
                                       "object_id": 24, "step": step, "polarity": "positive"}
                                      for step, offset in enumerate((50, 100, 150))],
                           "metadata": {"synthetic": True, "confidence_semantics": "synthetic score",
                                        "prediction_kind": "instance"}})
        queries = [int(np.flatnonzero(truth == 2)[50])]
        scenes.append({"kind": "npz", "path": path.name, "scene_id": scene_id,
                       "dataset": "Synthetic demonstration", "queries": queries,
                       "metadata": {"background_ids": [0], "synthetic": True}, "models": models})
        if with_mesh:
            scenes[-1]["mesh"] = {"kind": "scene"}
    config = {"title": "Delimit3D visual lab · synthetic demonstration", "seed": 42,
              "formats": ["html", "png", "pdf"], "max_display_points": 4000,
              "max_analysis_points": 2000, "reference": "Synthetic encoder A", "scenes": scenes}
    if with_mesh:
        config["title"] = "Delimit3D mesh lab · synthetic demonstration"
        config["visualizations"] = ["rgb", "labels", "pca", "similarity", "delta", "prediction"]
    (directory / "demo.json").write_text(json.dumps(config, indent=2) + "\n")
    return config
