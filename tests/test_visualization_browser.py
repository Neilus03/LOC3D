"""Optional CPU Chromium checks; all generated geometry is a synthetic fixture."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")
from delimit3d.visualization.contracts import Panel, Scene
from delimit3d.visualization.render import render_scene
from delimit3d.visualization.suite import _index

spec = importlib.util.spec_from_file_location("visual_browser_check", Path(__file__).parents[1] / "scripts/visualization/check_browser.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture(scope="module")
def cpu_browser():
    with playwright.sync_playwright() as runtime:
        try:
            browser = checker.launch_chromium(runtime)
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc) or "error while loading shared libraries" in str(exc):
                pytest.skip(f"Optional CPU Chromium runtime unavailable: {exc}")
            raise
        yield browser
        browser.close()


def _fixture_scene(output, name, dataset, experiment, tags):
    rng = np.random.default_rng(100)
    points = rng.uniform(-1, 1, size=(72, 3))
    labels = np.repeat([1, 2, 3], 24)
    scene = Scene(name, points, labels=labels, metadata={
        "dataset": dataset, "experiment": experiment, "synthetic": True, "illustrative_only": True,
        "camera": {"yaw": -.4, "pitch": .55, "point_size": 3., "clip_z": 0.,
                   "center": [0., 0., 0.], "projection": "orthographic", "up_axis": "z"}})
    panels = [Panel("rgb", "SYNTHETIC FIXTURE · input", rng.uniform(.1, .9, size=(72, 3)))]
    for model in ["Fixture encoder A", "Fixture encoder B"]:
        for step in [0, 1]:
            panels.append(Panel(f"{model}/step_{step}", f"{model} · step {step}",
                np.tile([.05 + .1 * step, .4, .7], (72, 1)), values=labels,
                caption=f"Synthetic controlled replay fixture; saved step {step}; no experimental metric.",
                metadata={"model_name": model, "step": step, "clicks": [
                    {"point_index": 0, "is_positive": True, "object_id": 1},
                    *([{"point_index": 50, "is_positive": False, "object_id": 0}] if step else [])]}))
    # A proposal model has no temporal step. Its selection must still work when
    # recorded interaction models share this scene.
    panels.append(Panel("Fixture proposals/display", "SYNTHETIC proposals", rng.uniform(.1,.9,size=(72,3)),
                        metadata={"model_name":"Fixture proposals"}))
    output.mkdir()
    files = render_scene(scene, panels, output, sample_indices=np.arange(72), formats=("html",))
    return {"scene_id": name, "directory": output.name, "dataset": dataset,
            "experiment": experiment, "selection_tags": tags, "failure_types":["synthetic_failure"], "status": "complete",
            "points": 72, "panels": len(panels), "files": files}


def test_offline_gallery_models_steps_inspector_camera_and_filters(tmp_path, cpu_browser):
    scenes = [_fixture_scene(tmp_path / "first", "fixture_one", "Dataset A", "Historical fixture", ["representative"]),
              _fixture_scene(tmp_path / "second", "fixture_two", "Dataset B", "New fixture", ["regression", "difficult"])]
    _index(tmp_path, {"title": "SYNTHETIC browser test collection", "scenes": scenes})
    result = checker.check_gallery(tmp_path, browser=cpu_browser, require_steps=True, require_models=2)
    failures = [c for p in result["pages"] for c in p["checks"] if c["status"] == "failed"]
    assert result["status"] == "passed", failures
    assert not result["external_requests"]
    assert len(result["pages"]) == 3
    assert any(c["name"] == "collection_filters" and c["status"] == "passed" for p in result["pages"] for c in p["checks"])
    for page in result["pages"][1:]:
        checks = {c["name"]: c["status"] for c in page["checks"]}
        assert checks["paired_interaction_steps"] == "passed"
        assert checks["linked_camera_orbit_and_download"] == "passed"
        assert checks["declared_camera_initialization"] == "passed"


def test_browser_checker_rejects_external_network_dependencies(tmp_path, cpu_browser):
    _fixture_scene(tmp_path / "network_fixture", "network_fixture", "fixture", "fixture", [])
    html = tmp_path / "network_fixture/index.html"
    html.write_text(html.read_text().replace("</head>", '<script src="https://example.invalid/forbidden.js"></script></head>'))
    result = checker.check_gallery(html, browser=cpu_browser)
    assert result["status"] == "failed"
    # The request is blocked before reaching the network; failure is still an
    # offline-portability defect rather than a silently passing screenshot.
    assert result["external_requests"] or any(p["failed_requests"] for p in result["pages"])
