"""Small YAML/JSON configurations with explicit inheritance and environment paths."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
METADATA_KEYS = frozenset({"schema", "name", "description", "reference", "recipe"})


def merge_config(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings; lists and scalars are replaced in full."""
    result = dict(parent)
    for key, value in child.items():
        previous = result.get(key)
        result[key] = (merge_config(previous, value)
                       if isinstance(previous, Mapping) and isinstance(value, Mapping)
                       else value)
    return result


def expand_environment(value: Any, *, strict: bool = True) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item, strict=strict) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item, strict=strict) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match[str]) -> str:
        name = match[1]
        if name in os.environ:
            return os.environ[name]
        if strict:
            raise ValueError(f"Set {name} or replace ${{{name}}} in your configuration")
        return match[0]

    return _VARIABLE.sub(substitute, value)


def load_config(path: str | Path, *, expand_env: bool = True,
                _ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load an explicit configuration; relative extends paths use the config directory."""
    path = Path(path).expanduser().resolve(strict=True)
    if path in _ancestors:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        values = json.loads(text)
    else:
        import yaml
        values = yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a configuration mapping")
    parents = values.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        raise ValueError(f"{path}: extends must be a path or list of paths")
    result: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        result = merge_config(result, load_config(parent_path, expand_env=False,
                              _ancestors=(*_ancestors, path)))
    result = merge_config(result, values)
    return expand_environment(result) if expand_env else result
