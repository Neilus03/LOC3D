"""Real HTTP transport and CPU inference contracts for the outbound GPU bridge."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from delimit3d.inference.remote_worker import BridgeClient, BridgeError, RemoteWorker, allocation_deadline
from delimit3d.inference.service import InferenceService


def scene_bytes():
    file = io.BytesIO()
    np.savez_compressed(file, points=np.arange(36, dtype=np.float32).reshape(12, 3),
                        colors=np.full((12, 3), 90, dtype=np.uint8))
    return file.getvalue()


class Runner:
    calls = 0

    def __init__(self, entry, device):
        self.model_id = entry["id"]

    def predict(self, scene, clicks, score_threshold):
        self.__class__.calls += 1
        return {"labels": np.arange(len(scene.points), dtype=np.int32) % 3,
                "instance_ids": np.array([1, 2]), "scores": np.array([.8, .9]),
                "provenance": {"fixture": self.model_id}}


@pytest.fixture
def service(tmp_path):
    Runner.calls = 0
    (tmp_path / "config.yaml").write_text("task: mask3d\n")
    (tmp_path / "weights.pt").write_bytes(b"fixture; never loaded")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema": "delimit3d_model_registry/v1", "models": [
        {"id": "pretrained", "task": "mask3d", "config": "config.yaml", "checkpoint": "weights.pt"},
        {"id": "scratch", "task": "mask3d", "config": "config.yaml", "checkpoint": "weights.pt"}]}))
    example = tmp_path / "example.npz"
    example.write_bytes(scene_bytes())
    return InferenceService(registry, tmp_path / "out", device="cpu", examples=[example], runner_factory=Runner)


class BridgeFixture:
    def __init__(self):
        self.jobs = deque()
        self.completed = {}
        self.results = {}
        self.heartbeats = []
        self.requests = []
        self.failures = {}
        self.record_then_fail = set()
        self.active_heartbeat = threading.Event()

    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, status=200, headers=None):
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.dispatch()

            def do_POST(self):
                self.dispatch()

            def dispatch(self):
                path = urlsplit(self.path).path
                owner.requests.append((self.command, path, dict(self.headers)))
                if (self.headers.get("Authorization") != "Bearer test-secret"
                    or self.headers.get("OAI-Sites-Authorization") != "Bearer site-secret"):
                    return self.reply({"error": "denied"}, 401)
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(body) if body and self.headers.get("Content-Type") == "application/json" else None
                if owner.failures.get(path, 0):
                    owner.failures[path] -= 1
                    return self.reply({"error": "temporary"}, 503)
                if path == "/bridge/job":
                    assert parse_qs(urlsplit(self.path).query)["worker_id"] == ["test-worker"]
                    return self.reply({"job": owner.jobs.popleft() if owner.jobs else None})
                if path.startswith("/bridge/blob/"):
                    assert self.headers.get("X-Worker-Id") == "test-worker"
                    return self.reply(scene_bytes())
                if path == "/bridge/heartbeat":
                    owner.heartbeats.append(payload)
                    if payload["active_job_id"]:
                        owner.active_heartbeat.set()
                    return self.reply({"ok": True})
                if path.startswith("/bridge/result/"):
                    assert self.headers.get("X-Worker-Id") == "test-worker"
                    owner.results[Path(path).name] = body
                    return self.reply({"ok": True})
                if path.startswith("/bridge/complete/"):
                    owner.completed[Path(path).name] = payload
                    if path in owner.record_then_fail:
                        owner.record_then_fail.remove(path)
                        return self.reply({"error": "acknowledgment lost"}, 503)
                    return self.reply({"ok": True})
                if path == "/bridge/redirect":
                    return self.reply({}, 302, {"Location": "/bridge/forbidden-redirect-target"})
                if path == "/bridge/large":
                    return self.reply(b"x" * 200)
                return self.reply({"error": "not found"}, 404)

        return Handler


@pytest.fixture
def bridge():
    fixture = BridgeFixture()
    with ThreadingHTTPServer(("127.0.0.1", 0), fixture.handler()) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        fixture.client = BridgeClient(f"http://127.0.0.1:{server.server_port}", "test-secret",
            headers={"OAI-Sites-Authorization": "Bearer site-secret"}, timeout=2)
        try:
            yield fixture
        finally:
            server.shutdown()
            thread.join(timeout=5)


def make_worker(service, bridge):
    return RemoteWorker(service, bridge.client, worker_id="test-worker", gpu_name="fixture CPU",
                        heartbeat_interval=.03, poll_interval=.01, retry_interval=.03)


def test_http_upload_prediction_switch_and_download(service, bridge):
    worker = make_worker(service, bridge)
    worker.heartbeat()
    heartbeat = bridge.heartbeats[-1]
    assert heartbeat["gpu_name"] == "fixture CPU"
    assert heartbeat["examples"] == [{"id": "0", "name": "example"}]
    assert [entry["id"] for entry in heartbeat["models"]] == ["pretrained", "scratch"]
    assert "checkpoint" not in json.dumps(heartbeat)
    bridge.jobs.append({"id": "scene1", "kind": "scene", "blob_id": "blob1", "filename": "test.npz"})
    assert worker.step()
    scene = bridge.completed["scene1"]["result"]
    assert scene["point_count"] == 12
    for model in ("pretrained", "scratch"):
        bridge.jobs.append({"id": model, "kind": "predict", "request": {"model_id": model, "scene_id": scene["scene_id"]}})
        assert worker.step()
        receipt = bridge.completed[model]
        assert receipt["ok"] and receipt["worker_id"] == "test-worker"
        result = receipt["result"]
        assert result["labels"] == [i % 3 for i in range(12)]
        with np.load(io.BytesIO(bridge.results[Path(result["download_url"]).name]), allow_pickle=False) as data:
            np.testing.assert_array_equal(data["labels"], np.arange(12) % 3)
            assert json.loads(str(data["provenance_json"]))["model_id"] == model
    assert Runner.calls == 2
    assert not worker.step()


def test_failed_ack_and_duplicate_delivery_do_not_rerun_inference(service, bridge):
    worker = make_worker(service, bridge)
    bridge.jobs.append({"id": "example", "kind": "example", "example_id": "0"})
    worker.step()
    scene_id = bridge.completed["example"]["result"]["scene_id"]
    job = {"id": "prediction", "kind": "predict", "request": {"model_id": "scratch", "scene_id": scene_id}}
    bridge.jobs.extend([job, job])
    bridge.record_then_fail.add("/bridge/complete/prediction")
    with pytest.raises(BridgeError, match="HTTP 503"):
        worker.step()
    assert Runner.calls == 1 and worker.active_job_id == "prediction"
    assert worker.pending is not None
    worker.step()  # Repeat the acknowledgment only after it was already accepted.
    worker.step()  # Redelivered job also reuses the successful receipt.
    assert Runner.calls == 1 and worker.active_job_id is None
    uploads = [path for _, path, _ in bridge.requests if path.startswith("/bridge/result/")]
    assert len(uploads) == 1


def test_failed_blob_download_retains_job_for_retry(service, bridge):
    worker = make_worker(service, bridge)
    bridge.jobs.append({"id": "scene1", "kind": "scene", "filename": "test.npz"})
    bridge.failures["/bridge/blob/scene1"] = 1
    with pytest.raises(BridgeError, match="HTTP 503"):
        worker.step()
    assert worker.active_job_id == "scene1"
    worker.step()
    assert bridge.completed["scene1"]["ok"]
    assert len([path for _, path, _ in bridge.requests if path == "/bridge/job"]) == 1


def test_failed_export_upload_is_retried_without_repeating_prediction(service, bridge, monkeypatch):
    worker = make_worker(service, bridge)
    bridge.jobs.append({"id": "example", "kind": "example", "example_id": "0"})
    worker.step()
    scene_id = bridge.completed["example"]["result"]["scene_id"]
    bridge.jobs.append({"id": "prediction", "kind": "predict", "request": {
        "model_id": "scratch", "scene_id": scene_id}})
    original = bridge.client.request
    uploads = []

    def interrupt_upload(method, path, **kwargs):
        if path.startswith("/bridge/result/"):
            uploads.append(path)
            if len(uploads) == 1:
                raise BridgeError("simulated lost upload connection")
        return original(method, path, **kwargs)

    monkeypatch.setattr(bridge.client, "request", interrupt_upload)
    with pytest.raises(BridgeError, match="lost upload"):
        worker.step()
    assert "prediction" not in bridge.completed
    assert worker.pending is not None and not worker.pending[1].uploaded
    worker.step()
    assert Runner.calls == 1 and len(uploads) == 2 and uploads[0] == uploads[1]
    assert bridge.completed["prediction"]["ok"] and len(bridge.results) == 1


def test_invalid_jobs_and_expired_scenes_complete_with_errors(service, bridge):
    worker = make_worker(service, bridge)
    bridge.jobs.extend([
        {"id": "expired", "kind": "predict", "request": {"model_id": "pretrained", "scene_id": "old-worker-scene"}},
        {"id": "badpath", "kind": "scene", "blob_id": "../../outside", "filename": "test.npz"},
        {"id": "unknown", "kind": "example", "example_id": "missing"},
    ])
    for _ in range(3):
        worker.step()
    assert "Scene expired" in bridge.completed["expired"]["error"]
    assert all(not value["ok"] for value in bridge.completed.values())
    assert Runner.calls == 0
    assert not any("/bridge/blob/" in path for _, path, _ in bridge.requests)


def test_heartbeats_renew_active_job_while_gpu_call_is_blocked(service, bridge):
    entered, release = threading.Event(), threading.Event()
    original = service.predict

    def blocked(request):
        entered.set()
        assert release.wait(timeout=5)
        return original(request)

    service.predict = blocked
    worker = make_worker(service, bridge)
    bridge.jobs.append({"id": "example", "kind": "example", "example_id": "0"})
    worker.step()
    bridge.jobs.append({"id": "slow", "kind": "predict", "request": {
        "model_id": "pretrained", "scene_id": bridge.completed["example"]["result"]["scene_id"]}})
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        assert entered.wait(timeout=3)
        assert bridge.active_heartbeat.wait(timeout=3)
        assert bridge.heartbeats[-1]["active_job_id"] == "slow"
    finally:
        release.set()
        worker.stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert bridge.completed["slow"]["ok"]


def test_allocation_deadline_is_normalized_and_published(service, bridge):
    assert allocation_deadline(environ={}) is None
    assert allocation_deadline(environ={"SLURM_JOB_END_TIME": "3600"}) == "1970-01-01T01:00:00+00:00"
    assert allocation_deadline("2026-10-18T01:19:00+02:00") == "2026-10-17T23:19:00+00:00"
    with pytest.raises(ValueError, match="timezone"):
        allocation_deadline("2026-10-18T01:19:00")
    worker = make_worker(service, bridge)
    worker.available_until = allocation_deadline("2026-10-18T01:19:00+02:00")
    worker.heartbeat()
    assert bridge.heartbeats[-1]["available_until"] == "2026-10-17T23:19:00+00:00"


def test_expired_job_stops_delivery_retries_and_accepts_next_job(service, bridge, monkeypatch):
    worker = make_worker(service, bridge)
    bridge.jobs.append({"id": "expired", "kind": "example", "example_id": "0"})
    original = bridge.client.request

    def expired_ack(method, path, **kwargs):
        if path == "/bridge/complete/expired":
            raise BridgeError("Bridge request returned HTTP 410", status=410)
        return original(method, path, **kwargs)

    monkeypatch.setattr(bridge.client, "request", expired_ack)
    assert worker.step()
    assert worker.pending is None and worker.active_job_id is None
    bridge.jobs.append({"id": "next", "kind": "example", "example_id": "0"})
    assert worker.step()
    assert bridge.completed["next"]["ok"]


def test_transport_refuses_redirects_oversize_and_insecure_credentials(bridge):
    with pytest.raises(BridgeError, match="HTTP 302"):
        bridge.client.json("GET", "/bridge/redirect")
    assert not any(path.endswith("forbidden-redirect-target") for _, path, _ in bridge.requests)
    with pytest.raises(BridgeError, match="size limit"):
        bridge.client.request("GET", "/bridge/large", limit=10)
    for url in ("http://remote.example", "https://user:password@example.com", "https://example.com?token=secret"):
        with pytest.raises(ValueError, match="HTTPS"):
            BridgeClient(url, "secret")
    with pytest.raises(ValueError, match="HEADERS"):
        BridgeClient("https://example.com", "secret", headers={"Host": "other.example"})
    with pytest.raises(ValueError, match="Only bridge"):
        bridge.client.request("GET", "https://other.example")
    assert "test-secret" not in bridge.client.safe_error(ValueError("test-secret and Bearer site-secret"))
    assert "site-secret" not in bridge.client.safe_error(ValueError("test-secret and Bearer site-secret"))
