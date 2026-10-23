"""Loopback-only workbench, intended for local use or an SSH tunnel to a GPU."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import traceback
import zipfile
from urllib.parse import unquote, urlsplit

from .scene import load_scene


MAX_UPLOAD = 128 * 1024 * 1024
STATIC = Path(__file__).with_name("static")


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LOC3D"

        def _headers(self, status, kind, length):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()

        def _json(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode()
            self._headers(status, "application/json", len(body))
            self.wfile.write(body)

        def _same_origin(self):
            host = self.headers.get("Host", "")
            if urlsplit("http://" + host).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise PermissionError("Access this workbench through localhost or an SSH tunnel")
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise PermissionError("Cross-site requests are disabled")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                raise PermissionError("Cross-origin requests are disabled")

        def _dispatch(self, action):
            try:
                self._same_origin()
                action()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except PermissionError as exc:
                self._json({"error": str(exc)}, 403)
            except (ValueError, KeyError, FileNotFoundError, zipfile.BadZipFile) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc) or type(exc).__name__}, 503)

        def do_GET(self):
            self._dispatch(self._get)

        def _get(self):
            path = urlsplit(self.path).path
            if path == "/api/models":
                return self._json(service.models())
            if path == "/api/examples":
                return self._json({"examples": [{"id": key, "name": value.stem}
                    for key, value in service.examples.items()]})
            if path.startswith("/api/examples/"):
                key = path.removeprefix("/api/examples/")
                if key not in service.examples:
                    raise ValueError("Unknown example scene")
                return self._json(service.add_scene(load_scene(service.examples[key])))
            if path.startswith("/api/results/") and path.endswith(".npz"):
                key = path.removeprefix("/api/results/").removesuffix(".npz")
                if key not in service.downloads:
                    raise ValueError("Result download expired or does not exist")
                file = service.downloads[key]
                kind = "application/octet-stream"
            else:
                relative = "index.html" if path == "/" else path.removeprefix("/static/")
                if path != "/" and not path.startswith("/static/"):
                    return self._json({"error": "Not found"}, 404)
                file = (STATIC / relative).resolve()
                if not file.is_relative_to(STATIC.resolve()) or not file.is_file():
                    return self._json({"error": "Not found"}, 404)
                kind = "text/javascript" if file.suffix == ".mjs" else mimetypes.guess_type(file)[0] or "application/octet-stream"
            self._headers(200, kind, file.stat().st_size)
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    self.wfile.write(block)

        def do_POST(self):
            self._dispatch(self._post)

        def _post(self):
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Use a Content-Length request")
            length = int(self.headers.get("Content-Length", "0"))
            path = urlsplit(self.path).path
            limit = MAX_UPLOAD if path == "/api/scenes" else 128 * 1024
            if not 0 < length <= limit:
                raise ValueError("Request is empty or exceeds the upload limit")
            self.connection.settimeout(120)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete upload")
            if path == "/api/scenes":
                scene = load_scene(body, filename=unquote(self.headers.get("X-Filename", "")))
                return self._json(service.add_scene(scene))
            if path == "/api/predict":
                request = json.loads(body)
                if not isinstance(request, dict):
                    raise ValueError("Prediction request must be a JSON object")
                return self._json(service.predict(request))
            self._json({"error": "Not found"}, 404)

    return Handler


def serve(service, *, host="127.0.0.1", port=7860):
    if host not in {"localhost", "127.0.0.1"}:
        raise ValueError("Use a loopback address and an SSH tunnel for remote GPU access")
    with ThreadingHTTPServer((host, port), make_handler(service)) as server:
        print(f"LOC3D workbench: http://localhost:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
