"""Outbound-only GPU worker for a private, authenticated inference workbench.

Run with ``python -m delimit3d.inference.remote_worker --help``. Credentials are
read from the environment and are never accepted as command-line arguments.
The bridge must lease jobs to one worker and renew that lease on heartbeats.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
import json
import logging
import os
from pathlib import Path
import re
import signal
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from .scene import load_scene


LOG = logging.getLogger(__name__)
MAX_BLOB = 128 * 1024 * 1024
MAX_JSON = 16 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class BridgeError(RuntimeError):
    """A safely printable transport failure, without credentials or payloads."""

    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib otherwise forwards credentials to the redirected host.
        return None


class BridgeClient:
    def __init__(self, url, token, *, headers=None, timeout=60):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )) or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Bridge URL must use HTTPS (HTTP is allowed only for loopback tests)")
        if not token or "\n" in token or "\r" in token:
            raise ValueError("Set DELIMIT3D_BRIDGE_TOKEN to the private bridge credential")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {token}"}
        self.secrets = [token]
        for name, value in (headers or {}).items():
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]+", name)
                or name.lower() in {"authorization", "host", "content-length", "content-type"}
                or not isinstance(value, str) or "\r" in value or "\n" in value):
                raise ValueError("Invalid DELIMIT3D_BRIDGE_HEADERS entry")
            self.headers[name] = value
            if value:
                self.secrets.append(value)
        self.opener = build_opener(_NoRedirect())

    def request(self, method, path, *, body=None, content_type=None, limit=MAX_JSON, worker_id=None):
        if not path.startswith("/bridge/") or path.startswith("//"):
            raise ValueError("Only bridge endpoints may receive worker credentials")
        if body is not None and len(body) > MAX_BLOB:
            raise ValueError("Bridge upload exceeds 128 MiB")
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        if worker_id:
            headers["X-Worker-Id"] = checked_id(worker_id)
        request = Request(self.url + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and (int(length) < 0 or int(length) > limit):
                    raise BridgeError("Bridge response exceeds the size limit")
                result = response.read(limit + 1)
                if len(result) > limit:
                    raise BridgeError("Bridge response exceeds the size limit")
                if length is not None and len(result) != int(length):
                    raise BridgeError("Bridge response was incomplete")
                return result
        except HTTPError as exc:
            exc.close()
            raise BridgeError(f"Bridge request returned HTTP {exc.code}", status=exc.code) from None
        except (URLError, TimeoutError, OSError, HTTPException) as exc:
            raise BridgeError(f"Bridge connection failed ({type(exc).__name__})") from None

    def json(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if body is not None and len(body) > MAX_JSON:
            raise ValueError("Bridge JSON request exceeds the size limit")
        response = self.request(method, path, body=body, content_type="application/json")
        try:
            value = json.loads(response) if response else {}
        except (ValueError, UnicodeError):
            raise BridgeError("Bridge returned invalid JSON") from None
        if not isinstance(value, dict):
            raise BridgeError("Bridge JSON response must be an object")
        return value

    def safe_error(self, error):
        value = str(error) or type(error).__name__
        for secret in self.secrets:
            value = value.replace(secret, "[redacted]")
        return value[:2000]


def checked_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid bridge job or result identifier")
    return value


def allocation_deadline(value=None, *, environ=None):
    """Return a UTC deadline without exposing scheduler details to the browser."""
    if value:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            raise ValueError("Allocation deadline must include a timezone")
    else:
        environment = os.environ if environ is None else environ
        epoch = environment.get("SLURM_JOB_END_TIME")
        if not epoch:
            return None
        deadline = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
    return deadline.astimezone(timezone.utc).isoformat()


@dataclass
class Receipt:
    payload: dict
    result_path: Path | None = None
    result_id: str | None = None
    uploaded: bool = False


class RemoteWorker:
    """Process one GPU call at a time; retry delivery without repeating inference."""

    def __init__(self, service, client, *, worker_id=None, gpu_name=None, available_until=None, poll_interval=1,
                 heartbeat_interval=15, retry_interval=5):
        self.service, self.client = service, client
        self.worker_id = checked_id(worker_id or uuid.uuid4().hex)
        self.gpu_name = gpu_name
        self.available_until = available_until
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.retry_interval = retry_interval
        self.stop = threading.Event()
        self.active_job_id = None
        self.job = None
        self.pending = None
        self.completed = OrderedDict()
        self.heartbeat_ready = threading.Event()

    def heartbeat(self):
        payload = {"worker_id": self.worker_id, "active_job_id": self.active_job_id,
                   "models": self.service.models()["models"],
                   "examples": [{"id": key, "name": value.stem}
                                for key, value in self.service.examples.items()]}
        if self.gpu_name:
            payload["gpu_name"] = self.gpu_name
        if self.available_until:
            payload["available_until"] = self.available_until
        self.client.json("POST", "/bridge/heartbeat", payload)
        self.heartbeat_ready.set()

    def _heartbeats(self):
        while not self.stop.is_set():
            try:
                self.heartbeat()
                delay = self.heartbeat_interval
            except Exception as exc:
                # Do not print backend exceptions, URLs, or authentication headers.
                LOG.warning("Worker heartbeat failed (%s); retrying",
                            str(exc) if isinstance(exc, BridgeError) else type(exc).__name__)
                delay = self.retry_interval
            self.stop.wait(delay)

    def _execute(self, job):
        kind = job.get("kind")
        if kind == "scene":
            blob_id = checked_id(job.get("blob_id", job["id"]))
            filename = job.get("filename", "")
            if not isinstance(filename, str) or len(filename) > 512:
                raise ValueError("Invalid scene filename")
            data = self.client.request("GET", f"/bridge/blob/{blob_id}", limit=MAX_BLOB,
                                       worker_id=self.worker_id)
            result = self.service.add_scene(load_scene(data, filename=filename))
        elif kind == "example":
            example_id = job.get("example_id")
            if example_id not in self.service.examples:
                raise ValueError("Unknown example scene")
            result = self.service.add_scene(load_scene(self.service.examples[example_id]))
        elif kind == "predict":
            request = job.get("request")
            if not isinstance(request, dict):
                raise ValueError("Prediction request must be a JSON object")
            result = self.service.predict(request)
            match = re.fullmatch(r"/api/results/([A-Za-z0-9_-]+)\.npz", result["download_url"])
            if match is None or match[1] not in self.service.downloads:
                raise ValueError("Prediction did not produce a registered download")
            result_id = checked_id(match[1])
            path = self.service.downloads[result_id]
            if path.stat().st_size > MAX_BLOB:
                raise ValueError("Prediction export exceeds the 128 MiB download limit")
            return Receipt({"worker_id": self.worker_id, "ok": True, "result": result}, path, result_id)
        else:
            raise ValueError("Unknown bridge job kind")
        return Receipt({"worker_id": self.worker_id, "ok": True, "result": result})

    def _deliver(self, job_id, receipt):
        if receipt.result_path is not None and not receipt.uploaded:
            # Upload before acknowledging: a successful browser response always has a download.
            self.client.request("POST", f"/bridge/result/{receipt.result_id}.npz",
                body=receipt.result_path.read_bytes(), content_type="application/octet-stream",
                worker_id=self.worker_id)
            receipt.uploaded = True
        self.client.json("POST", f"/bridge/complete/{job_id}", receipt.payload)
        self.completed[job_id] = receipt
        self.completed.move_to_end(job_id)
        while len(self.completed) > 32:
            self.completed.popitem(last=False)
        self.pending = None
        self.job = None
        self.active_job_id = None

    def step(self):
        """Perform one poll/delivery cycle; delivery failures retain the receipt."""
        try:
            return self._step()
        except BridgeError as exc:
            if exc.status != 410 or self.active_job_id is None:
                raise
            # An expired/deleted queue item cannot accept another result. Do not
            # let its retry block later work, and never repeat its GPU call.
            LOG.info("Discarded an expired bridge job")
            self.pending = None
            self.job = None
            self.active_job_id = None
            return True

    def _step(self):
        if self.pending is not None:
            self._deliver(*self.pending)
            return True
        if self.job is None:
            response = self.client.json("GET", "/bridge/job?" + urlencode({"worker_id": self.worker_id}))
            self.job = response.get("job")
        job = self.job
        if job is None:
            return False
        if not isinstance(job, dict):
            raise BridgeError("Bridge job must be an object")
        job_id = checked_id(job.get("id"))
        self.active_job_id = job_id
        if job_id in self.completed:
            receipt = self.completed[job_id]
        else:
            try:
                receipt = self._execute(job)
            except BridgeError:
                # Transport errors while fetching a scene are retryable; retain the lease.
                raise
            except Exception as exc:
                receipt = Receipt({"worker_id": self.worker_id, "ok": False,
                                   "error": self.client.safe_error(exc)})
        self.pending = (job_id, receipt)
        self._deliver(*self.pending)
        return True

    def run(self):
        heartbeat = threading.Thread(target=self._heartbeats, name="bridge-heartbeat", daemon=True)
        heartbeat.start()
        LOG.info("GPU bridge worker started")
        try:
            while not self.stop.is_set():
                if not self.heartbeat_ready.wait(timeout=.5):
                    continue
                try:
                    worked = self.step()
                    delay = 0 if worked else self.poll_interval
                except Exception as exc:
                    LOG.warning("Worker request failed (%s); retrying",
                                str(exc) if isinstance(exc, BridgeError) else type(exc).__name__)
                    delay = self.retry_interval
                self.stop.wait(delay)
        finally:
            self.stop.set()
            heartbeat.join(timeout=self.client.timeout + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Private HTTPS workbench URL")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--gpu-name", default=None)
    parser.add_argument("--available-until", help="GPU allocation end, as ISO 8601 with timezone; defaults to SLURM_JOB_END_TIME")
    args = parser.parse_args(argv)
    try:
        extra_headers = json.loads(os.environ.get("DELIMIT3D_BRIDGE_HEADERS", "{}"))
        if not isinstance(extra_headers, dict):
            raise ValueError("Headers must be an object")
        client = BridgeClient(args.url, os.environ.get("DELIMIT3D_BRIDGE_TOKEN", ""), headers=extra_headers)
    except (ValueError, TypeError):
        parser.error("Set a valid HTTPS URL, DELIMIT3D_BRIDGE_TOKEN, and optional DELIMIT3D_BRIDGE_HEADERS JSON object")
    try:
        available_until = allocation_deadline(args.available_until)
    except (ValueError, OverflowError, OSError):
        parser.error("Use an ISO 8601 --available-until with timezone, or an epoch SLURM_JOB_END_TIME")
    from .service import InferenceService
    service = InferenceService(args.registry, args.output, device=args.device, examples=args.example)
    worker = RemoteWorker(service, client, worker_id=args.worker_id, gpu_name=args.gpu_name,
                          available_until=available_until)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: worker.stop.set())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker.run()


if __name__ == "__main__":
    main()
