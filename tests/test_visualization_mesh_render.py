"""Mesh exports preserve complete topology, correspondence, and diagnostic scales."""

from __future__ import annotations

import base64
import hashlib
import json
import re

import numpy as np
import pytest

from delimit3d.visualization.contracts import Mesh, Panel, Scene
from delimit3d.visualization import mesh_render


def _example():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                       [.3, .3, .2]])
    # Duplicate correspondence is intentional: reconstructed mesh vertices can
    # map to the same scene point. Query point 4 has no mesh vertex.
    mesh = Mesh(points[[2, 0, 1, 3]], np.array([[0, 1, 2], [1, 2, 3],
                                             [2, 0, 3], [0, 1, 3]]),
                np.array([2, 0, 1, 1]), metadata={"source_kind": "original"})
    scene = Scene("mesh/example", points, mesh=mesh, labels=np.array([3, 4, 5, 6, 7]))
    colors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                       [1., 1., 0.], [.5, .5, .5]])
    panels = [Panel("rgb", "RGB", colors),
              Panel("model/query", "Similarity", colors, values=np.linspace(-.2, .2, 5),
                    limits=(-1., 1.), query_index=4,
                    metadata={"color_stops": [[0., 0., 1.], [1., 1., 1.], [1., 0., 0.]],
                              "value_label": "Cosine similarity", "clicks": [
                                  {"point_index": 3, "is_positive": True},
                                  {"point_index": 2, "is_positive": False}]}),
              Panel("labels", "GT instances", colors)]
    return scene, panels


def _read(path):
    document = path.read_text()
    match = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',
                      document, re.DOTALL)
    assert match is not None
    return document, json.loads(match.group(1))


def _unpack(array):
    return np.frombuffer(base64.b64decode(array["base64"]), dtype=array["dtype"]).reshape(
        array["shape"])


def test_offline_mesh_retains_full_topology_mapping_and_palettes(tmp_path, monkeypatch):
    scene, panels = _example()
    monkeypatch.setattr(mesh_render, "_matplotlib", lambda: pytest.fail("HTML imported matplotlib"))
    result = mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",), seed=17)
    assert result == {"html": "index.html"}
    document, data = _read(tmp_path / "index.html")
    np.testing.assert_array_equal(_unpack(data["vertices"]), scene.mesh.vertices)
    np.testing.assert_array_equal(_unpack(data["faces"]), scene.mesh.faces)
    np.testing.assert_array_equal(_unpack(data["point_indices"]), scene.mesh.point_indices)
    np.testing.assert_array_equal(_unpack(data["labels"]), [5, 3, 4, 4])
    np.testing.assert_array_equal(_unpack(data["panels"][0]["colors"]),
                                  [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 0]])
    assert data["panels"][1]["legend"]["limits"] == [-1., 1.]
    assert data["panels"][1]["legend"]["colors"] == [[0, 0, 255], [255, 255, 255], [255, 0, 0]]
    assert data["face_count"] == 4 and data["vertex_count"] == 4 and data["seed"] == 17
    assert data["camera"] == {"yaw": mesh_render._YAW, "pitch": mesh_render._PITCH}
    assert "no display simplification" in data["rendering"]["topology"]
    assert "fetch(" not in document
    assert not re.search(r'<script[^>]+src\s*=', document)
    assert "https://" not in document
    assert "does not recompute model predictions" in document


def test_mesh_markers_retain_original_scene_positions(tmp_path):
    scene, panels = _example()
    mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    _, data = _read(tmp_path / "index.html")
    panel = data["panels"][1]
    assert panel["query"] == {"index": 4, "point": [.3, .3, .2]}
    assert panel["clicks"][0]["point"] == [0., 0., 1.]
    assert panel["clicks"][0]["index"] == 3
    assert panel["clicks"][1]["polarity"] == "negative"


def test_mesh_category_majority_never_invents_palette_colors():
    vertex_colors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])
    faces = np.array([[0, 1, 3], [0, 1, 2], [3, 1, 0]])
    np.testing.assert_array_equal(
        mesh_render._face_colors(vertex_colors, faces, categorical=True),
        [[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_allclose(
        mesh_render._face_colors(vertex_colors, faces[:1], categorical=False),
        [[1 / 3, 2 / 3, 0.]])
    scene, panels = _example()
    payload = mesh_render._mesh_payload(scene, panels, 0)
    assert [p["categorical"] for p in payload["panels"]] == [False, False, True]
    assert not mesh_render._categorical(Panel("model/fpn/pca", "PCA", vertex_colors))


def test_undefined_scalar_is_nan_binary_and_reconstruction_is_disclosed(tmp_path):
    scene, panels = _example()
    panels[1].metadata["undefined_mask"] = np.array([False, True, False, False, False])
    scene.mesh.metadata.update(source_kind="reconstructed", reconstruction="ball_pivoting")
    mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    _, data = _read(tmp_path / "index.html")
    values = _unpack(data["panels"][1]["values"])
    assert np.isnan(values[2:]).all()
    assert values[:2].tolist() == pytest.approx([0., -.2])
    assert "Reconstructed / approximate surface" in data["surface_note"]
    assert "ball_pivoting" in data["surface_note"]


def test_binary_indices_exceed_webgl_uint16_limit_without_lost_faces(tmp_path):
    count = 65538
    points = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), (count // 3, 1))
    faces = np.arange(count, dtype=np.int64).reshape(-1, 3)
    scene = Scene("large_mesh", points, mesh=Mesh(points.copy(), faces, np.arange(count)))
    panel = Panel("rgb", "RGB", np.ones((count, 3)))
    mesh_render.render_mesh_scene(scene, [panel], tmp_path, formats=("html",))
    document, data = _read(tmp_path / "index.html")
    np.testing.assert_array_equal(_unpack(data["faces"]), faces)
    assert _unpack(data["faces"]).max() == 65537
    assert "OES_element_index_uint" in document
    assert "complete non-indexed compatibility mode" in document


def test_mesh_html_escapes_scene_and_metadata(tmp_path):
    scene, panels = _example()
    scene.scene_id = '</script><script>alert("oops")</script>'
    panels[0].caption = '<img src=x onerror="alert(1)">'
    scene.metadata["big"] = np.arange(10000)
    scene.mesh.metadata["ply_exports"] = ["rgb.ply", "query.ply", "labels.ply"]
    mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    document, data = _read(tmp_path / "index.html")
    assert scene.scene_id not in document and panels[0].caption not in document
    assert data["scene_id"] == scene.scene_id
    assert data["metadata"]["big"]["shape"] == [10000]
    assert data["mesh_metadata"]["ply_exports"] == ["rgb.ply", "query.ply", "labels.ply"]


def test_mesh_static_exports_real_png_pdf_and_full_face_collection(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.collections import PolyCollection

    scene, panels = _example()
    counts = []
    original = PolyCollection.__init__

    def capture(self, verts, *args, **kwargs):
        counts.append(len(verts))
        return original(self, verts, *args, **kwargs)

    monkeypatch.setattr(PolyCollection, "__init__", capture)
    result = mesh_render.render_mesh_scene(scene, panels[:1], tmp_path,
                                           formats=("html", "png", "pdf"))
    assert result["png"] == ["rgb.png"] and result["pdf"] == ["rgb.pdf"]
    assert (tmp_path / result["png"][0]).read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / result["pdf"][0]).read_bytes().startswith(b"%PDF")
    assert result["contact_sheets_png"] == ["contact_sheet.png"]
    assert counts and all(count == len(scene.mesh.faces) for count in counts)
    _, data = _read(tmp_path / "index.html")
    assert data["files"]["contact_sheet_pdf"] == "contact_sheet.pdf"


def test_mesh_validation_rejects_missing_mesh_and_invalid_panel(tmp_path):
    scene, panels = _example()
    scene.mesh = None
    with pytest.raises(ValueError, match="scene.mesh"):
        mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    scene, panels = _example()
    panels[1].query_index = 6
    with pytest.raises(ValueError, match="query index"):
        mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    panels[1].query_index = 1
    panels[0].key = "model/query"
    with pytest.raises(ValueError, match="unique"):
        mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))


def test_mesh_browser_interaction_and_category_switch(tmp_path):
    """Optional real browser check; install Playwright/Chromium to run locally."""
    playwright = pytest.importorskip("playwright.sync_api")
    scene, panels = _example()
    scene.mesh.metadata["ply_exports"] = ["rgb.ply", "query.ply", "labels.ply"]
    scene.metadata["source_files"] = {"synthetic-source.npz": "a" * 64}
    mesh_render.render_mesh_scene(scene, panels, tmp_path, formats=("html",))
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu",
                "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except playwright.Error as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 950})
            page.set_default_timeout(60000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto((tmp_path / "index.html").as_uri(), wait_until="domcontentloaded")
            page.wait_for_function("views.length === 2 && views.every(v => v.drawCount > 0)")
            assert page.locator("#status").inner_text().startswith("WebGL depth-tested")
            page.get_by_label("Layer in panel A").select_option("2")
            page.wait_for_function("views[0].colorIndex === 2")
            assert page.evaluate("views[0].expanded")
            for mode in ("edges", "wireframe", "surface"):
                page.locator("#mode").select_option(mode)
                page.wait_for_function("state.mode === '" + mode + "'")
            page.get_by_label("Layer in panel A").select_option("1")
            page.wait_for_function("views[0].colorIndex === 1")
            page.locator("#height").fill("60")
            assert page.locator("#height-value").inner_text() == "60%"
            page.locator("#top").click()
            assert page.evaluate("state.pitch") == pytest.approx(np.pi / 2)
            page.locator("#reset").click()
            page.locator("#height").fill("100")
            page.evaluate("state.pin = 1; $('point-info').textContent = pointText(1); requestDraw()")
            assert "Original point ID: 0" in page.locator("#point-info").inner_text()
            assert page.get_by_text("Colored mesh PLY").count() == 2
            # A capture records the current interactive view rather than only
            # the initial manifest camera; both artifact downloads remain paired.
            page.locator("#height").fill("60")
            page.locator("canvas.overlay").first.focus()
            page.keyboard.press("ArrowLeft")
            expected_state = page.evaluate("({...state})")
            with page.expect_download(predicate=lambda item: item.suggested_filename.endswith(".json")) as record_download:
                with page.expect_download(predicate=lambda item: item.suggested_filename.endswith(".png")) as image_download:
                    page.locator("#export-view").click()
            image_path, record_path = tmp_path / "capture.png", tmp_path / "capture.json"
            image_download.value.save_as(image_path)
            record_download.value.save_as(record_path)
            capture = json.loads(record_path.read_text())
            assert capture["schema"] == "delimit3d.mesh_view_capture.v1"
            assert capture["scope"] == "exploratory browser capture; not benchmark evidence"
            assert capture["interactive_state"] == expected_state
            assert capture["camera"]["yaw"] == expected_state["yaw"]
            assert capture["camera"]["height_percent"] == 60
            assert capture["camera"]["clip_z"] == pytest.approx(.6)
            assert capture["rendering"]["inspected_original_point_id"] == 0
            assert capture["source"]["scene_metadata"]["source_files"] == scene.metadata["source_files"]
            assert capture["source"]["face_count"] == len(scene.mesh.faces)
            assert capture["panels"][0]["key"] == "model/query"
            assert capture["panels"][0]["legend"]["limits"] == [-1., 1.]
            assert capture["panels"][0]["query"]["index"] == 4
            assert len(capture["panels"][0]["clicks"]) == 2
            document, _ = _read(tmp_path / "index.html")
            payload_text = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',
                                     document, re.DOTALL).group(1)
            assert capture["source"]["payload_sha256"] == hashlib.sha256(payload_text.encode()).hexdigest()
            assert capture["files"]["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
            assert len(capture["source"]["renderer_script_sha256"]) == 64
            assert "companion JSON" in page.locator("#status").inner_text()
            assert page.evaluate("views.every(v => v.gl.getError() === v.gl.NO_ERROR)")
            assert not errors
        finally:
            browser.close()
