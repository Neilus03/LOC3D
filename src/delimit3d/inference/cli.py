"""Public trained-model inference and local workbench commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use trained Delimit3D models on your own point clouds.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Inspect the configured model registry without loading CUDA.")
    listing.add_argument("--registry", type=Path, required=True)
    for command in ("predict", "serve"):
        sub = commands.add_parser(command, help="Run inference." if command == "predict" else "Open the local 3D workbench.")
        sub.add_argument("--registry", type=Path, required=True)
        sub.add_argument("--device", default="cuda")
        if command == "predict":
            sub.add_argument("--model", required=True)
            sub.add_argument("--scene", type=Path, required=True)
            sub.add_argument("--clicks", type=Path, help="JSON array of {point_index, object_id}; 0 is background.")
            sub.add_argument("--score-threshold", type=float, default=.3)
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
            sub.add_argument("--port", type=int, default=7860)
            sub.add_argument("--output", type=Path, default=Path("runs/inference"))
            sub.add_argument("--example", type=Path, action="append", default=[], help="Add a locally prepared NPZ/PLY to the example chooser.")
    args = parser.parse_args(argv)
    from .registry import load_registry, public_models
    if args.command == "list":
        print(json.dumps({"models": public_models(load_registry(args.registry))}, indent=2))
        return
    from .service import InferenceService
    if args.command == "serve":
        from .server import serve
        service = InferenceService(args.registry, args.output, device=args.device, examples=args.example)
        return serve(service, host=args.host, port=args.port)
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    from .scene import load_scene
    service = InferenceService(args.registry, args.output.parent, device=args.device)
    scene = service.add_scene(load_scene(args.scene))
    result = service.predict({"model_id": args.model, "scene_id": scene["scene_id"],
        "clicks": json.loads(args.clicks.read_text()) if args.clicks else [],
        "score_threshold": args.score_threshold})
    generated = service.output / Path(result["download_url"]).name
    with generated.open("rb") as source, args.output.open("xb") as target:
        shutil.copyfileobj(source, target)
    generated.unlink()
    print(json.dumps({"output": str(args.output.resolve()), "provenance": result["provenance"]}, indent=2))


def demo_main():
    main(["serve", *sys.argv[1:]])


if __name__ == "__main__":
    main()
