"""Simple user-supplied queries/K; benchmark oracles stay in reproduction tools."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from delimit3d.runtime.config import load_config
from delimit3d.runtime.records import file_sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segment a scene with cosine similarity or spherical k-means.")
    parser.add_argument("--method", choices=("cosine", "kmeans"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", type=Path, help="NPZ with native features and optional original_to_token map.")
    source.add_argument("--scene", type=Path, help="NPZ with ordered points, colors and normals.")
    parser.add_argument("--config", type=Path, help="Backbone recipe for --scene.")
    parser.add_argument("--queries", help="Comma-separated original point row indices for cosine retrieval.")
    parser.add_argument("--k", type=int, help="User-selected cluster count.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import torch
    from .segmentation import cosine_segmentation, fit_kmeans
    if args.features:
        with np.load(args.features, allow_pickle=False) as data:
            features = torch.as_tensor(data["features"], device=args.device)
            inverse = torch.as_tensor(data["original_to_token"] if "original_to_token" in data
                else np.arange(len(features)), device=args.device, dtype=torch.long)
        metadata = {"input_sha256": file_sha256(args.features), "native_feature_dim": features.shape[1]}
    else:
        if args.config is None:
            parser.error("--scene requires --config")
        from .features import extract_native_features
        with np.load(args.scene, allow_pickle=False) as data:
            features, inverse, metadata = extract_native_features(load_config(args.config),
                data["points"], data["colors"], data["normals"], device=args.device)
        metadata["input_sha256"] = file_sha256(args.scene)
    if inverse.ndim != 1 or inverse.numel() == 0 or inverse.min() < 0 or inverse.max() >= len(features):
        raise ValueError("Invalid original_to_token mapping")
    arrays = {}
    if args.method == "cosine":
        if args.queries is None:
            parser.error("cosine requires --queries")
        queries = torch.as_tensor([int(value) for value in args.queries.split(",")], device=args.device)
        if queries.min() < 0 or queries.max() >= len(inverse):
            raise ValueError("Query point is outside the scene")
        labels, scores = cosine_segmentation(features, inverse[queries])
        arrays["similarity"] = scores[inverse].cpu().numpy()
        labels = labels.cpu().numpy()
        metadata.update(query_points=queries.tolist(), query_source="user", rejection_threshold=None)
    else:
        if args.k is None:
            parser.error("kmeans requires --k")
        labels, centers, record = fit_kmeans(features, args.k, seed=args.seed, restarts=args.restarts)
        arrays["centers"] = centers
        metadata.update(record)
    inverse = inverse.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, labels=labels[inverse], token_labels=labels, original_to_token=inverse, **arrays)
    write_json(args.output.with_suffix(".json"), {"schema": "delimit3d_feature_segmentation/v1", "method": args.method, **metadata})
    print(json.dumps(metadata, indent=2))
