"""Fetch locked optional sources and weights into a new user-owned directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

from .records import file_sha256, write_json


def source_main(argv):
    parser = argparse.ArgumentParser(description="Fetch a pinned optional backbone/teacher source.")
    parser.add_argument("--source", choices=("litept", "concerto", "sonata", "sam2", "unsamv2"), required=True)
    parser.add_argument("--destination", type=Path, required=True, help="Fresh source checkout directory.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent / "assets"
    record = json.loads((root / "sources.lock.json").read_text())["sources"][args.source]
    if args.destination.exists():
        raise FileExistsError("Use a fresh destination; source setup preserves existing checkouts")
    subprocess.run(["git", "clone", "--no-checkout", record["url"], str(args.destination)], check=True)
    subprocess.run(["git", "-C", str(args.destination), "checkout", "--detach", record["commit"]], check=True)
    if record.get("patch"):
        patch = root / record["patch"]
        if file_sha256(patch) != record["patch_sha256"]:
            raise ValueError("Source patch changed")
        subprocess.run(["git", "-C", str(args.destination), "apply", str(patch)], check=True)
    for path, expected in record.get("patched_files_sha256", {}).items():
        if file_sha256(args.destination / path) != expected:
            raise ValueError(f"Pinned source mismatch: {path}")
    write_json(args.destination / "delimit3d_source_record.json", record)
    print(f"Prepared {args.source} at {record['commit']}")


def weight_main(argv):
    parser = argparse.ArgumentParser(description="Download a checkpoint and require its locked content hash.")
    root = Path(__file__).resolve().parent / "assets"
    weights = json.loads((root / "weights.lock.json").read_text())["weights"]
    parser.add_argument("--weight", choices=tuple(weights), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = weights[args.weight]
    if args.output.exists():
        if file_sha256(args.output) != record["sha256"]:
            raise ValueError("Existing checkpoint differs; use a fresh output path")
        print("Existing checkpoint matches the locked hash")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(record["url"], timeout=60) as response:
                shutil.copyfileobj(response, handle)
            handle.flush()
            if file_sha256(temporary) != record["sha256"]:
                raise ValueError("Downloaded checkpoint differs from the executed reference")
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(args.output.with_suffix(".json"), record)
