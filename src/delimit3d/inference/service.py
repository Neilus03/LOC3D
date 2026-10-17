"""A bounded, single-GPU inference service shared by CLI and browser UI."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import numpy as np

from .registry import load_registry, public_models
from .scene import load_scene
from delimit3d.runtime.config import load_config


def export_prediction(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["provenance_json"] = np.asarray(json.dumps(result["provenance"], sort_keys=True))
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


class InferenceService:
    def __init__(self, registry, output, *, device="cuda", examples=(), runner_factory=None):
        self.registry = Path(registry)
        self.entries = {entry["id"]: entry for entry in load_registry(self.registry)}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.runner_factory = runner_factory
        self.runner = self.runner_id = None
        self.runner_signature = None
        self.scenes = OrderedDict()
        self.scene_lock = threading.Lock()
        self.predict_lock = threading.Lock()
        self.examples = {str(i): Path(path) for i, path in enumerate(examples)}
        self.downloads = OrderedDict()

    def models(self):
        self.entries = {entry["id"]: entry for entry in load_registry(self.registry)}
        return {"models": public_models(self.entries.values())}

    def add_scene(self, scene):
        identity = uuid.uuid4().hex
        preview, ids = scene.preview()
        with self.scene_lock:
            self.scenes[identity] = (scene, ids)
            while len(self.scenes) > 2:
                self.scenes.popitem(last=False)
        return {"scene_id": identity, **preview}

    def predict(self, request):
        if not self.predict_lock.acquire(blocking=False):
            raise RuntimeError("The GPU is busy with another prediction. Try again when it finishes.")
        try:
            return self._predict(request)
        finally:
            self.predict_lock.release()

    def _predict(self, request):
        self.models()
        identity = request.get("model_id")
        if identity not in self.entries:
            raise ValueError("Choose a registered model")
        entry = self.entries[identity]
        if not entry["available"]:
            raise ValueError(entry["reason"])
        with self.scene_lock:
            if request.get("scene_id") not in self.scenes:
                raise ValueError("Scene expired or is missing. Load it again.")
            scene, display_ids = self.scenes[request["scene_id"]]
        threshold = float(request.get("score_threshold", .3))
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Score threshold must be between 0 and 1")
        clicks = request.get("clicks", [])
        if not isinstance(clicks, list) or len(clicks) > 200:
            raise ValueError("Supply at most 200 ordered clicks")
        for click in clicks:
            if not isinstance(click, dict):
                raise ValueError("Each click must contain point_index and object_id")
            for key in ("point_index", "object_id"):
                if type(click.get(key)) is not int or click[key] < 0:
                    raise ValueError(f"Click {key} must be a nonnegative integer")
            if click["point_index"] >= len(scene.points) or click["object_id"] > 100:
                raise ValueError("Click point is outside the scene or object id exceeds 100")
        checkpoint_stat = Path(entry["checkpoint"]).stat()
        signature = json.dumps({"entry": entry,
            "config": load_config(entry["config"], expand_env=False),
            "checkpoint_size": checkpoint_stat.st_size, "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns}, sort_keys=True)
        if self.runner_id != identity or self.runner_signature != signature:
            # Release the previous model before allocating the next one.
            self.runner = None
            self.runner_id = None
            if self.runner_factory is None:
                from .backends import ModelRunner
                factory = ModelRunner
            else:
                factory = self.runner_factory
            self.runner = factory(entry, device=self.device)
            self.runner_id = identity
            self.runner_signature = signature
        result = self.runner.predict(scene, clicks=clicks, score_threshold=threshold)
        labels = np.asarray(result["labels"])
        if labels.shape != (len(scene.points),) or not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError("Model did not return labels in original point order")
        result["labels"] = labels
        result.setdefault("provenance", {}).update({
            "schema": "delimit3d_inference/v1", "model_id": identity,
            "scene_sha256": scene.sha256, "point_count": len(scene.points),
            "point_order": "original_input_rows", "clicks": clicks,
            "score_threshold": threshold, "created_at": datetime.now(timezone.utc).isoformat(),
            "display_note": "Exclusive labels are a visualization; packed Mask3D proposals may overlap.",
        })
        result_id = uuid.uuid4().hex
        path = self.output / f"{result_id}.npz"
        export_prediction(path, result)
        self.downloads[result_id] = path
        while len(self.downloads) > 32:
            self.downloads.popitem(last=False)
        return {"labels": labels[display_ids].tolist(),
                "instance_count": len(result.get("instance_ids", [])),
                "scores": np.asarray(result.get("scores", [])).tolist(),
                "instance_ids": np.asarray(result.get("instance_ids", [])).tolist(),
                "provenance": result["provenance"], "download_url": f"/api/results/{result_id}.npz"}
