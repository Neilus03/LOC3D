"""Input/order, model switching, export, and browser-service boundary checks."""
import io
import json
from pathlib import Path
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from delimit3d.inference.registry import load_registry
from delimit3d.inference.scene import Scene, load_scene
from delimit3d.inference.server import make_handler
from delimit3d.inference.service import InferenceService


def scene_bytes(n=20):
    file = io.BytesIO()
    np.savez_compressed(file, points=np.arange(n*3).reshape(n, 3).astype(np.float32),
        colors=np.full((n, 3), 90, dtype=np.uint8), normals=np.tile([0, 0, 1.], (n, 1)))
    return file.getvalue()


@pytest.fixture
def registry(tmp_path):
    (tmp_path / "task.yaml").write_text("task: mask3d\n")
    (tmp_path / "weights.pt").write_bytes(b"test fixture; never loaded")
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema": "delimit3d_model_registry/v1", "models": [
        {"id": name, "task": "mask3d", "config": "task.yaml", "checkpoint": "weights.pt"}
        for name in ("a", "b")]}))
    return path


class FixtureRunner:
    """Known point-order witness, not a production backend."""
    loaded = []

    def __init__(self, entry, device):
        self.loaded.append(entry["id"])

    def predict(self, scene, clicks, score_threshold):
        labels = np.arange(len(scene.points), dtype=np.int32) % 3
        return {"labels": labels, "scores": np.array([.8, .9]), "instance_ids": np.array([1, 2]),
                "provenance": {"test_fixture": True}}


def test_scene_format_mapping_and_limits():
    scene = load_scene(scene_bytes(), filename="test.npz")
    preview, ids = scene.preview(7)
    np.testing.assert_array_equal(np.asarray(preview["points"]), scene.points[ids])
    assert preview["point_count"] == 20 and preview["point_ids"] == ids.tolist()
    with pytest.raises(ValueError, match="exceeds"):
        load_scene(scene_bytes(), filename="test.npz", max_points=10)
    with pytest.raises(ValueError, match="Choose an NPZ"):
        load_scene(b"test", filename="weights.pt")
    with pytest.raises(ValueError, match="unit vectors"):
        Scene(scene.points, scene.colors, np.zeros((20, 3)))
    with pytest.raises(ValueError, match="colors"):
        Scene(scene.points, scene.colors.astype(float)-100)


def test_npz_object_arrays_are_not_unpickled():
    file = io.BytesIO()
    np.savez(file, points=np.array([object()], dtype=object), colors=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="Object arrays"):
        load_scene(file.getvalue(), filename="unsafe.npz")


def test_registry_missing_weights_are_visible_and_relative_paths_resolve(registry, monkeypatch):
    entries = load_registry(registry)
    assert entries[0]["available"] and Path(entries[0]["checkpoint"]).is_absolute()
    (registry.parent / "weights.pt").unlink()
    entries = load_registry(registry)
    assert not entries[0]["available"] and "checkpoint" in entries[0]["reason"]
    data = json.loads(registry.read_text())
    data["models"][1]["id"] = "a"
    registry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(registry)


def test_service_model_switching_export_and_invalid_requests(registry, tmp_path):
    FixtureRunner.loaded = []
    service = InferenceService(registry, tmp_path / "out", device="cpu", runner_factory=FixtureRunner)
    scene = service.add_scene(load_scene(scene_bytes(), filename="test.npz"))
    request = {"model_id": "a", "scene_id": scene["scene_id"]}
    result = service.predict(request)
    export = service.output / Path(result["download_url"]).name
    with np.load(export, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["labels"], np.arange(20) % 3)
        record = json.loads(str(data["provenance_json"]))
        assert record["point_order"] == "original_input_rows" and record["scene_sha256"]
    service.predict(request)
    service.predict({**request, "model_id": "b"})
    assert FixtureRunner.loaded == ["a", "b"]
    for change in ({"score_threshold": float("nan")}, {"clicks": [{"point_index": True, "object_id": 1}]},
                   {"clicks": [{"point_index": 20, "object_id": 1}]}, {"model_id": "missing"}):
        with pytest.raises(ValueError):
            service.predict({**request, **change})
    assert not service.predict_lock.locked()


def test_http_rejects_cross_site_and_serves_real_point_order(registry, tmp_path):
    service = InferenceService(registry, tmp_path / "out", device="cpu", runner_factory=FixtureRunner)
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = HTTPConnection("127.0.0.1", server.server_port)
        try:
            client.request("GET", "/api/models", headers={"Origin": "https://unrelated.example"})
            response = client.getresponse()
            assert response.status == 403
            response.read()
            client.request("GET", "/api/models")
            response = client.getresponse()
            assert response.status == 200
            assert len(json.loads(response.read())["models"]) == 2
            client.request("POST", "/api/scenes", scene_bytes(), {"X-Filename": "sample.npz"})
            response = client.getresponse()
            scene = json.loads(response.read())
            assert response.status == 200 and scene["point_count"] == 20
            client.request("POST", "/api/predict", json.dumps({"model_id": "a", "scene_id": scene["scene_id"]}))
            response = client.getresponse()
            result = json.loads(response.read())
            assert response.status == 200 and result["labels"] == [i % 3 for i in range(20)]
            client.request("GET", result["download_url"])
            response = client.getresponse()
            assert response.status == 200 and response.read().startswith(b"PK")
            client.request("GET", "/static/../../pyproject.toml")
            response = client.getresponse()
            assert response.status == 404
            response.read()
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=5)


def test_availability_refresh_and_replaced_checkpoint_invalidate_model(registry, tmp_path):
    FixtureRunner.loaded = []
    service = InferenceService(registry, tmp_path / "out", device="cpu", runner_factory=FixtureRunner)
    scene = service.add_scene(load_scene(scene_bytes(), filename="test.npz"))
    request = {"model_id": "a", "scene_id": scene["scene_id"]}
    service.predict(request)
    (registry.parent / "weights.pt").write_bytes(b"replacement checkpoint")
    service.predict(request)
    assert FixtureRunner.loaded == ["a", "a"]
    (registry.parent / "weights.pt").unlink()
    assert not service.models()["models"][0]["available"]
    (registry.parent / "weights.pt").write_bytes(b"restored checkpoint")
    assert service.models()["models"][0]["available"]
