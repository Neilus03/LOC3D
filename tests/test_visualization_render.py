"""Renderer checks cover offline portability and matched sample correspondence."""

from __future__ import annotations

import builtins
import json
import re

import numpy as np
import pytest

from delimit3d.visualization.contracts import Panel, Scene
from delimit3d.visualization import render


def _example():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    scene = Scene("tiny/scene", points, labels=np.array([0, 1, 1, 2]),
                  metadata={"dataset": "tiny", "illustrative_only": True})
    panels = [
        Panel("model/layer/pca", "PCA", np.eye(4, 3), caption="Shared basis"),
        Panel("model/query", "Query response", np.full((4, 3), 0.5),
              values=np.array([-1., 0., 0.5, 1.]), cmap="coolwarm", limits=(-1., 1.),
              query_index=3, metadata={"value_label": "Cosine similarity"}),
    ]
    return scene, panels


def _read_payload(path):
    document = path.read_text()
    data = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',
                     document, re.DOTALL)
    assert data is not None
    return document, json.loads(data.group(1))


def test_offline_gallery_preserves_original_point_correspondence(tmp_path, monkeypatch):
    scene, panels = _example()
    indices = np.array([3, 1], dtype=np.int64)

    def forbidden_import():
        raise AssertionError("HTML-only export must not import matplotlib")

    monkeypatch.setattr(render, "_matplotlib", forbidden_import)
    result = render.render_scene(scene, panels, tmp_path, sample_indices=indices,
                                 formats=("html",), seed=17)
    assert result == {"html": "index.html"}
    document, data = _read_payload(tmp_path / "index.html")
    assert data["indices"] == [3, 1]
    assert data["points"] == scene.points[indices].tolist()
    assert data["labels"] == scene.labels[indices].tolist()
    assert data["panels"][1]["values"] == [1.0, 0.0]
    assert data["panels"][1]["query"] == {"index": 3, "point": [0., 0., 1.]}
    assert data["panels"][1]["legend"]["limits"] == [-1., 1.]
    assert data["panels"][0]["colors"] == [[0, 0, 0], [0, 255, 0]]
    assert data["seed"] == 17
    assert data["center"] == [0.5, 0.5, 0.5]
    assert data["radius"] == pytest.approx(np.sqrt(0.75))
    assert "fetch(" not in document
    assert not re.search(r'<script[^>]+src\s*=', document)
    assert "https://" not in document
    assert "does not recompute model predictions" in document


def test_gallery_escapes_untrusted_labels_and_bounds_metadata(tmp_path):
    scene, panels = _example()
    scene.scene_id = '</script><script>alert("oops")</script>'
    scene.metadata["token_ids"] = np.arange(100_000)
    scene.metadata["invalid_metric"] = float("nan")
    panels[0].caption = '<img src=x onerror="alert(1)">'
    render.render_scene(scene, panels, tmp_path, sample_indices=np.array([0, 2]),
                        formats=("html",))
    document, data = _read_payload(tmp_path / "index.html")
    assert scene.scene_id not in document
    assert panels[0].caption not in document
    assert data["scene_id"] == scene.scene_id
    assert data["panels"][0]["caption"] == panels[0].caption
    assert data["metadata"]["token_ids"]["shape"] == [100_000]
    assert data["metadata"]["invalid_metric"] is None
    assert len(document) < 80_000


@pytest.mark.parametrize("indices", [np.array([]), np.array([0., 1.]), np.array([0, 0]),
                                      np.array([-1]), np.array([4]), np.array([[0]])])
def test_bad_sampling_indices_rejected(tmp_path, indices):
    scene, panels = _example()
    with pytest.raises(ValueError, match="sample_indices"):
        render.render_scene(scene, panels, tmp_path, sample_indices=indices, formats=("html",))


def test_filename_collisions_rejected(tmp_path):
    scene, panels = _example()
    panels[0].key, panels[1].key = "a/b", "a_b"
    with pytest.raises(ValueError, match="unique"):
        render.render_scene(scene, panels, tmp_path, sample_indices=np.arange(4))


def test_query_marker_keeps_unsampled_original_index(tmp_path):
    scene, panels = _example()
    render.render_scene(scene, panels, tmp_path, sample_indices=np.array([0, 1]),
                        formats=("html",))
    _, data = _read_payload(tmp_path / "index.html")
    assert 3 not in data["indices"]
    assert data["panels"][1]["query"]["index"] == 3
    np.testing.assert_allclose(data["panels"][1]["query"]["point"], scene.points[3])


def test_fixed_palette_survives_missing_extremes_and_clicks_keep_original_ids(tmp_path):
    scene, panels = _example()
    stops = [[0., 0., 0.8], [1., 1., 1.], [0.8, 0., 0.]]
    panels[1].metadata.update(color_stops=stops, clicks=[
        {"point_index": 3, "is_positive": True}, {"point_index": 2, "object_id": 0}])
    panels[1].values = np.array([-.2, -.1, .1, .2])
    render.render_scene(scene, panels, tmp_path, sample_indices=np.array([0, 1]),
                        formats=("html",))
    _, data = _read_payload(tmp_path / "index.html")
    panel = data["panels"][1]
    assert panel["legend"]["colors"] == [[0, 0, 204], [255, 255, 255], [204, 0, 0]]
    assert panel["legend"]["limits"] == [-1., 1.]
    assert panel["legend"]["palette_source"] == "fixed color stops"
    assert panel["clicks"][0]["index"] == 3
    assert panel["clicks"][0]["point"] == [0., 0., 1.]
    assert panel["clicks"][0]["polarity"] == "positive"
    assert panel["clicks"][1]["polarity"] == "background"


def test_undefined_finite_placeholders_are_null_in_point_inspector(tmp_path):
    scene, panels = _example()
    panels[1].values = np.array([0., 0., .5, 1.])
    panels[1].metadata["undefined_mask"] = np.array([True, False, False, False])
    render.render_scene(scene, panels, tmp_path, sample_indices=np.array([1, 0, 3]),
                        formats=("html",))
    _, data = _read_payload(tmp_path / "index.html")
    assert data["panels"][1]["values"] == [0., None, 1.]


def test_static_exports_are_real_artifacts_and_have_safe_filenames(tmp_path):
    pytest.importorskip("matplotlib")
    scene, panels = _example()
    result = render.render_scene(scene, panels, tmp_path, sample_indices=np.arange(4),
                                 formats=("html", "png", "pdf"))
    assert result["png"] == ["model_layer_pca.png", "model_query.png"]
    assert (tmp_path / result["png"][0]).read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / result["pdf"][1]).read_bytes().startswith(b"%PDF")
    assert result["contact_sheet_pdf"] == "contact_sheet.pdf"
    assert result["contact_sheets_png"] == ["contact_sheet.png"]
    _, data = _read_payload(tmp_path / "index.html")
    assert data["files"]["contact_sheet_png"] == "contact_sheet.png"
    assert data["stems"] == ["model_layer_pca", "model_query"]


def test_matplotlib_missing_gives_actionable_html_fallback(monkeypatch):
    original_import = builtins.__import__

    def intercept(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept)
    with pytest.raises(RuntimeError, match=r"formats=\('html',\)"):
        render._matplotlib()
