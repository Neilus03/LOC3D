"""Run repeatable, multi-scene visualization jobs from one YAML/JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

from .contracts import Scene


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Keep records small: describe large arrays instead of dumping feature caches."""
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return json_safe(value.tolist())
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_hash(value)}
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot record {type(value).__name__}; adapter metadata must be JSON-compatible")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-")[:72]
    return cleaned or "scene"


def expand_templates(value: Any, scene: Scene) -> Any:
    if isinstance(value, str):
        return value.replace("{scene_id}", scene.scene_id).replace(
            "{dataset}", str(scene.metadata.get("dataset", "custom"))
        )
    if isinstance(value, list):
        return [expand_templates(item, scene) for item in value]
    if isinstance(value, dict):
        return {key: expand_templates(item, scene) for key, item in value.items()}
    return value


def choose_queries(scene: Scene, config: Any, *, seed: int) -> list[int]:
    """Choose queries from geometry only, before observing any model results."""
    if isinstance(config, list):
        if any(not isinstance(index, int) or isinstance(index, bool) for index in config):
            raise ValueError("queries must contain integer original point indices")
        queries = list(dict.fromkeys(config))
        if any(index < 0 or index >= len(scene.points) for index in queries):
            raise ValueError(f"{scene.scene_id}: query index out of range")
        return queries
    config = config or {}
    count = int(config.get("count", 2))
    if count < 0:
        raise ValueError("query count must be nonnegative")
    if count == 0:
        return []
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(scene.points))
    if len(candidates) > 30000:
        candidates = np.sort(rng.choice(candidates, size=30000, replace=False))
    points = scene.points[candidates]
    first = int(np.argmin(np.sum((points - np.median(points, axis=0)) ** 2, axis=1)))
    selected = [first]
    distances = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, min(count, len(candidates))):
        distances[selected] = -1
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.sum((points - points[index]) ** 2, axis=1))
    return candidates[selected].astype(int).tolist()


def display_sample(n: int, limit: int, required: list[int], *, seed: int) -> np.ndarray:
    required_array = np.unique(np.asarray(required, dtype=np.int64))
    if limit < 1 or len(required_array) > limit:
        raise ValueError("max_display_points must be positive and cover all query/click markers")
    if len(required_array) and (required_array.min() < 0 or required_array.max() >= n):
        raise ValueError("Required display index out of range")
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), required_array, assume_unique=True)
    chosen = np.random.default_rng(seed).choice(remaining, limit - len(required_array), replace=False)
    return np.sort(np.concatenate((required_array, chosen)))


def export_ply(path: Path, points: np.ndarray, colors: np.ndarray, *,
               faces: np.ndarray | None = None, point_indices: np.ndarray | None = None) -> None:
    """Portable binary point cloud or triangular mesh, without changing topology."""
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if point_indices is not None:
        point_indices = np.asarray(point_indices)
        if (point_indices.shape != (len(points),) or
                not np.issubdtype(point_indices.dtype, np.integer) or
                np.any(point_indices < 0) or np.any(point_indices > np.iinfo(np.int32).max)):
            raise ValueError("PLY original point IDs must be nonnegative int32-compatible indices")
        dtype.append(("original_point_id", "<i4"))
    vertices = np.empty(len(points), dtype=dtype)
    for column, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, column]
    rgb = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for column, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, column]
    if point_indices is not None:
        vertices["original_point_id"] = point_indices
    face_data = None
    if faces is not None:
        faces = np.asarray(faces)
        if (faces.ndim != 2 or faces.shape[1] != 3 or
                not np.issubdtype(faces.dtype, np.integer) or
                np.any(faces < 0) or np.any(faces >= len(points)) or
                np.any(faces > np.iinfo(np.int32).max)):
            raise ValueError("PLY faces must contain valid triangular vertex indices")
        face_data = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
        face_data["count"] = 3
        face_data["indices"] = faces
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n")
    if point_indices is not None:
        header += "property int original_point_id\n"
    if face_data is not None:
        header += f"element face {len(face_data)}\nproperty list uchar int vertex_indices\n"
    header += "end_header\n"
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())
        if face_data is not None:
            handle.write(face_data.tobytes())


def _code_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    record: dict[str, Any] = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        record["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None
    record["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return record


def _index(output: Path, report: dict[str, Any]) -> None:
    cards = []
    for item in report["scenes"]:
        directory = item["directory"]
        label = html.escape(item["scene_id"])
        if item["status"] != "complete":
            cards.append(f'<article><h2>{label}</h2><p class="error">{html.escape(item["error"])}</p></article>')
            continue
        files = item["files"]
        preview = files.get("contact_sheet_png")
        if preview is None:
            preview = next(("figures/"+f for recipe in files.get("recipes",{}).get("records",[])
                            for f in recipe.get("files",[]) if f.endswith("presentation.png")),None)
        thumbnail = f'<img alt="{label} overview" src="{directory}/{html.escape(preview)}">' if preview else ""
        links = []
        for key, title in (("html", "Explore 3D"), ("contact_sheet_pdf", "Figure PDF"),
                           ("contact_sheet_png", "Presentation PNG")):
            if key in files:
                links.append(f'<a href="{directory}/{html.escape(files[key])}">{title}</a>')
        mesh_files = files.get("mesh", {})
        for key, title in (("html", "Explore mesh"), ("contact_sheet_pdf", "Mesh figure PDF")):
            if key in mesh_files:
                links.append(f'<a href="{directory}/mesh/{html.escape(mesh_files[key])}">{title}</a>')
        links.append(f'<a href="{directory}/report.json">Scene record</a>')
        recipe_records = files.get("recipes", {}).get("records", [])
        for recipe in recipe_records:
            for filename in recipe.get("files", []):
                if filename.endswith("paper.pdf") or filename.endswith("presentation.png"):
                    links.append(f'<a href="{directory}/figures/{html.escape(filename)}">{html.escape(filename)}</a>')
        dataset=html.escape(item["dataset"],quote=True)
        experiment=html.escape(item.get("experiment","unspecified"),quote=True)
        tags=html.escape(" ".join(item.get("selection_tags",[])),quote=True)
        failures=html.escape(" ".join(item.get("failure_types",[])),quote=True)
        cards.append(f'<article data-dataset="{dataset}" data-experiment="{experiment}" data-tags="{tags}" data-failures="{failures}">{thumbnail}<h2>{label}</h2><p>{item["points"]:,} points · '
                     f'{item["panels"]} views · {dataset}</p><p>{experiment} · {tags}</p>'
                     f'<nav>{"".join(links)}</nav></article>')
    title = html.escape(report["title"])
    output.joinpath("index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>' + title + '</title><style>'
        'body{font:16px system-ui;background:#101923;color:#edf3f8;margin:0;padding:5vw}'
        'h1{font-size:clamp(28px,5vw,58px);letter-spacing:-.04em;margin:.3em 0}'
        'header{max-width:900px;margin-bottom:32px}p{color:#b6c7d7;line-height:1.6}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px}'
        'article{background:#1a2938;padding:22px;border:1px solid #304557;border-radius:14px}'
        'img{width:100%;border-radius:8px}a{color:#6ee7cf}nav{display:flex;gap:18px;flex-wrap:wrap}'
        '.tag{color:#6ee7cf;letter-spacing:.14em;font-size:12px}.error{color:#ffb8b8}</style>'
        f'<header><div class="tag">DELIMIT3D / VISUAL LAB</div><h1>{title}</h1>'
        '<p>Explore representations, point queries, and segmentation. Each comparison shares '
        'geometry, display samples, and camera. These selected-scene diagnostics explain behavior; '
        'they are not benchmark results.</p><a href="run.json">Experiment record</a></header>'
        '<label>Dataset <select id="dataset-filter"><option value="">All</option></select></label> '
        '<label>Experiment <select id="experiment-filter"><option value="">All</option></select></label> '
        '<label>Case type <select id="tag-filter"><option value="">All</option></select></label> '
        '<label>Failure <select id="failure-filter"><option value="">All</option></select></label>'
        '<main>' + ''.join(cards) + '</main><script>'
        'const cards=[...document.querySelectorAll("article[data-dataset]")];'
        'for(const [id,key] of [["dataset-filter","dataset"],["experiment-filter","experiment"],["tag-filter","tags"],["failure-filter","failures"]]){'
        'const s=document.getElementById(id);let values=cards.flatMap(c=>(key==="tags"||key==="failures")?c.dataset[key].split(" "):[c.dataset[key]]);'
        'for(const v of [...new Set(values)].filter(Boolean).sort()){const o=document.createElement("option");o.value=v;o.textContent=v;s.append(o)}'
        's.onchange=()=>{for(const c of cards){c.hidden=[["dataset-filter","dataset"],["experiment-filter","experiment"],["tag-filter","tags"],["failure-filter","failures"]].some(([i,k])=>{const v=document.getElementById(i).value;return v&&((k==="tags"||k==="failures")?!c.dataset[k].split(" ").includes(v):c.dataset[k]!==v)})}}}'
        '</script></html>'
    )


def run_suite(config: dict[str, Any], *, base_dir: Path, output: Path) -> dict[str, Any]:
    from .adapters import load_observation, load_scene
    from .diagnostics import build_panels
    from .render import render_scene
    from .research import load_research_observation, compose_research_panels
    from .recipes import camera_record, render_recipes

    scenes = config.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Config must contain a nonempty scenes list")
    if any(not isinstance(spec, dict) or not isinstance(spec.get("path"), str) for spec in scenes):
        raise ValueError("Each scene must be a mapping with a string path")
    from .cases import resolve_scene_cases
    scenes, case_collection = resolve_scene_cases(config, base_dir)
    formats = tuple(config.get("formats", ["html", "png", "pdf"]))
    if not formats or set(formats) - {"html", "png", "pdf", "ply"}:
        raise ValueError("formats must be a nonempty list drawn from html, png, pdf, ply")
    seed = int(config.get("seed", 42))
    limit = int(config.get("max_display_points", 12000))
    if limit < 1 or int(config.get("max_analysis_points", 12000)) < 2:
        raise ValueError("Display limit must be positive; analysis limit must be at least 2")
    # Fresh output directories prevent stale panels being mistaken for this run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {**config, "scenes": scenes}
    write_json(output / "resolved_config.json", config)
    if case_collection is not None:
        write_json(output / "case_collection.json", case_collection)
    report: dict[str, Any] = {"schema": "delimit3d_visualization/v1", "title": config.get(
        "title", "Delimit3D visual atlas"), "status": "running", "seed": seed,
        "config": config, "code": _code_record(), "scenes": [], "errors": [],
        "case_collection": "case_collection.json" if case_collection is not None else None}
    write_json(output / "run.json", report)
    for index, spec in enumerate(scenes):
        directory = f"{index:03d}-{slug(spec.get('scene_id', Path(spec.get('path', 'scene')).stem))}"
        destination = output / directory
        destination.mkdir()
        item: dict[str, Any] = {"scene_id": spec.get("scene_id", directory), "directory": directory}
        try:
            scene = load_scene(spec, base_dir=base_dir).validate()
            scene.metadata.setdefault("dataset", spec.get("dataset", "custom"))
            scene.metadata["illustrative_only"] = True
            scene.metadata["experiment"] = spec.get("experiment", config.get("experiment", "unspecified"))
            scene.metadata["case_id"] = spec.get("case_id", scene.scene_id)
            scene.metadata["selection_tags"] = spec.get("selection_tags", [])
            if scene.metadata.get("selection") and scene.metadata["selection"] != spec.get("selection",{}):
                scene.metadata["input_selection"] = scene.metadata["selection"]
            scene.metadata["selection"] = spec.get("selection",{})
            scene.metadata["camera"] = camera_record(scene, spec.get("camera", config.get("camera")))
            mesh_spec = spec.get("mesh", config.get("mesh"))
            if mesh_spec is not None and mesh_spec is not False:
                if not isinstance(mesh_spec, dict):
                    raise ValueError("mesh must be a mapping or false")
                from .mesh import attach_mesh
                scene.mesh = attach_mesh(scene, expand_templates(mesh_spec, scene), base_dir)
                scene.mesh.validate(scene)
            query_indices = choose_queries(scene, spec.get("queries", config.get("queries")), seed=seed)
            models = spec.get("models", config.get("models", []))
            if not isinstance(models, list):
                raise ValueError("models must be a list")
            observations, structured, unavailable = [], [], []
            for model in models:
                if not isinstance(model, dict):
                    raise ValueError("Each model must be a mapping")
                model = dict(model)
                model.setdefault("seed", seed)
                print(f"[{index+1}/{len(scenes)}] {scene.scene_id}: {model.get('name', model.get('kind'))}", flush=True)
                try:
                    observation, special = load_research_observation(expand_templates(model, scene), scene,
                        base_dir, cpu_only=bool(config.get("cpu_only", False)))
                except FileNotFoundError as exc:
                    if not model.get("optional", False):
                        raise
                    unavailable.append({"model": model.get("name"), "reason": str(exc), "status": "blocked"})
                    continue
                if special is None or observation.features or observation.prediction is not None:
                    observation.validate(scene)
                observations.append(observation)
                structured.append(special)
            names = [observation.name for observation in observations]
            if len(set(names)) != len(names):
                raise ValueError("Model names must be unique within each scene")
            reference_name = spec.get("reference", config.get("reference"))
            if reference_name is not None and reference_name not in names:
                unavailable.append({"kind": "delta", "reason": "Reference model unavailable", "status": "blocked"})
                reference_name = None
            panels, diagnostics = compose_research_panels(
                scene, observations, structured, query_indices=query_indices, seed=seed,
                max_analysis_points=int(config.get("max_analysis_points", 12000)),
                reference_name=reference_name, requested=spec.get("visualizations", config.get("visualizations")),
                destination=destination, research_config={**config.get("research", {}), **spec.get("research", {})},
                base_dir=base_dir)
            diagnostics["unavailable"] = unavailable
            if not panels:
                raise ValueError("No requested visualizations are available for this scene")
            required = list(query_indices)
            for observation in observations:
                required.extend(int(click.get("point_index", click.get("index"))) for click in observation.clicks)
            for panel in panels:
                panel.validate(len(scene.points))
                if panel.query_index is not None:
                    required.append(panel.query_index)
            sample = display_sample(len(scene.points), limit, required, seed=seed)
            np.save(destination / "sample_indices.npy", sample)
            panel_formats = tuple(fmt for fmt in formats if fmt != "ply" and
                                  (config.get("export_individual_panels", True) or fmt == "html"))
            files = render_scene(scene, panels, destination, sample_indices=sample,
                                 formats=panel_formats, seed=seed)
            recipes = spec.get("recipes", config.get("recipes", []))
            if recipes:
                files["recipes"] = render_recipes(scene, panels, destination / "figures",
                    recipes=recipes, sample_indices=sample, camera=scene.metadata["camera"],
                    formats=tuple(fmt for fmt in formats if fmt in {"png", "pdf"}),
                    source_record={"models": [{"name":obs.name,"metadata":obs.metadata} for obs in observations],
                        "scene":scene.metadata, "queries":query_indices, "code":report["code"],
                        "selection":spec.get("selection",{}), "quantitative_rows":spec.get("quantitative_rows",[])})
            if scene.mesh is not None:
                from .mesh_render import render_mesh_scene
                mesh_directory = destination / "mesh"
                mesh_directory.mkdir()
                if "ply" in formats:
                    scene.mesh.metadata["ply_exports"] = [
                        f"{number:03d}-{slug(panel.key)}.ply" for number, panel in enumerate(panels)
                    ]
                files["mesh"] = render_mesh_scene(
                    scene, panels, mesh_directory,
                    formats=panel_formats, seed=seed,
                )
                np.save(mesh_directory / "vertex_to_point.npy", scene.mesh.point_indices)
            if "ply" in formats:
                files["ply"] = []
                for number, panel in enumerate(panels):
                    filename = f"{number:03d}-{slug(panel.key)}.ply"
                    export_ply(destination / filename, scene.points[sample], panel.colors[sample])
                    files["ply"].append(filename)
                if scene.mesh is not None:
                    files["mesh"]["ply"] = []
                    for number, panel in enumerate(panels):
                        filename = f"{number:03d}-{slug(panel.key)}.ply"
                        export_ply(destination / "mesh" / filename, scene.mesh.vertices,
                                   panel.colors[scene.mesh.point_indices], faces=scene.mesh.faces,
                                   point_indices=scene.mesh.point_indices)
                        files["mesh"]["ply"].append(filename)
            record = {"scene_id": scene.scene_id, "dataset": scene.metadata.get("dataset"),
                      "scene": scene.metadata, "points": len(scene.points),
                      "geometry_sha256": array_hash(scene.points), "query_indices": query_indices,
                      "case_id": spec.get("case_id", scene.scene_id), "selection": spec.get("selection", {}),
                      "quantitative_rows":spec.get("quantitative_rows",[]), "camera":scene.metadata["camera"],
                      "query_selection": "explicit" if isinstance(spec.get("queries", config.get("queries")), list) else "geometry only",
                      "display_points": len(sample), "sample_sha256": array_hash(sample),
                      "mesh": None if scene.mesh is None else {
                          "vertices": len(scene.mesh.vertices), "faces": len(scene.mesh.faces),
                          "vertices_sha256": array_hash(scene.mesh.vertices),
                          "faces_sha256": array_hash(scene.mesh.faces),
                          "point_indices_sha256": array_hash(scene.mesh.point_indices),
                          "metadata": scene.mesh.metadata,
                      },
                      "models": [{"name": obs.name, "metadata": obs.metadata,
                                  "layers": {key: list(value.shape) for key, value in obs.features.items()}}
                                 for obs in observations], "diagnostics": diagnostics,
                      "panels": [{"key": panel.key, "title": panel.title, "caption": panel.caption,
                                  "limits": panel.limits, "metadata": panel.metadata} for panel in panels],
                      "files": files}
            write_json(destination / "report.json", record)
            item.update(status="complete", scene_id=scene.scene_id, points=len(scene.points),
                        dataset=str(scene.metadata.get("dataset", "custom")), panels=len(panels), files=files,
                        experiment=scene.metadata["experiment"], selection_tags=scene.metadata["selection_tags"],
                        unavailable=unavailable, failure_types=sorted({p.metadata["failure_type"] for p in panels if p.metadata.get("failure_type")}))
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["errors"].append({"scene_id": item["scene_id"], "error": item["error"]})
            write_json(destination / "error.json", item)
            print(f"FAILED {item['scene_id']}: {item['error']}", file=sys.stderr, flush=True)
        report["scenes"].append(item)
        write_json(output / "run.json", report)
        _index(output, report)
        if item["status"] == "failed" and not config.get("continue_on_error", False):
            break
    report["status"] = "failed" if report["errors"] else "complete"
    report["completed_scenes"] = sum(item["status"] == "complete" for item in report["scenes"])
    report["requested_scenes"] = len(scenes)
    write_json(output / "run.json", report)
    _index(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="YAML/JSON scene and model manifest")
    source.add_argument("--demo", action="store_true", help="Synthetic fixture, no dataset/checkpoint needed")
    source.add_argument("--demo-mesh", action="store_true", help="Synthetic surfaces with exact triangle topology")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--formats", nargs="+", choices=("html", "png", "pdf", "ply"))
    parser.add_argument("--max-display-points", type=int)
    parser.add_argument("--cpu-only", action="store_true", help="Allow saved-artifact adapters only; prohibit model inference")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be a fresh directory: {output}")
    if args.demo or args.demo_mesh:
        from .demo import create_demo
        fixture = output.parent / (output.name + "_inputs")
        try:
            config = create_demo(fixture, with_mesh=args.demo_mesh)
        except FileExistsError as exc:
            parser.error(str(exc))
        base_dir = fixture
    else:
        config_path = args.config.expanduser().resolve(strict=True)
        if config_path.suffix.lower() == ".json":
            config = json.loads(config_path.read_text())
        else:
            import yaml
            config = yaml.safe_load(config_path.read_text())
        base_dir = config_path.parent
    if not isinstance(config, dict):
        parser.error("Manifest root must be a mapping")
    if args.cpu_only:
        config["cpu_only"] = True
    for argument, key in ((args.formats, "formats"), (args.max_display_points, "max_display_points"),
                          (args.seed, "seed")):
        if argument is not None:
            config[key] = argument
    try:
        report = run_suite(config, base_dir=base_dir, output=output)
    except (ValueError, FileExistsError, ImportError) as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {report['completed_scenes']}/{report['requested_scenes']} scenes; {output / 'index.html'}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
