"""Explicit local model registrations; weights are never downloaded implicitly."""
from __future__ import annotations

import os
from pathlib import Path
import re

from delimit3d.runtime.config import load_config, expand_environment


def load_registry(path):
    path = Path(path).expanduser().resolve()
    content = load_config(path, expand_env=False)
    if content.get("schema") != "delimit3d_model_registry/v1":
        raise ValueError("Expected schema delimit3d_model_registry/v1")
    entries, seen = [], set()
    if not isinstance(content.get("models"), list):
        raise ValueError("Registry models must be a list")
    for raw in content["models"]:
        entry = dict(raw)
        identity = entry.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", identity) or identity in seen:
            raise ValueError(f"Invalid or duplicate model id: {identity!r}")
        seen.add(identity)
        if entry.get("task") not in {"mask3d", "agile3d"}:
            raise ValueError(f"{identity}: task must be mask3d or agile3d")
        missing = []
        for key in ("config", "checkpoint"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{identity}: specify {key}")
            try:
                value = expand_environment(value)
                resolved = Path(value).expanduser()
                if not resolved.is_absolute():
                    resolved = path.parent / resolved
                entry[key] = str(resolved.resolve())
                if not resolved.is_file():
                    missing.append(f"{key} file is missing")
            except ValueError as exc:
                missing.append(str(exc))
        entry["name"] = str(entry.get("name", identity))
        entry["description"] = str(entry.get("description", ""))
        entry["available"] = not missing
        entry["reason"] = "; ".join(missing)
        entries.append(entry)
    return entries


def public_models(entries):
    keys = ("id", "name", "task", "description", "available", "reason")
    return [{key: entry[key] for key in keys} for entry in entries]
