"""Small content records and atomic output helpers shared by commands."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config):
    # Output destinations do not change the experiment. Evaluation and a
    # recovered run may safely write somewhere else with the same recipe.
    values = json.loads(json.dumps({key: value for key, value in config.items()
                                   if not key.startswith("_")}, default=str))
    for section in ("paths", "experiment"):
        if isinstance(values.get(section), dict):
            values[section].pop("output_root", None)
    if isinstance(values.get("experiment"), dict):
        values["experiment"].pop("name", None)
    values.pop("output_dir", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
