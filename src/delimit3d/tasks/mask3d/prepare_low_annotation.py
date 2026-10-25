"""Prepare sealed, mesh-aligned official ScanNet limited-annotation inputs.

Only this preparation boundary reads raw training annotations. The exported
training inputs contain geometry and labels at the official selected vertices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

BUDGETS = (20, 50, 100, 200)
TRAIN_SCENES = 1201
VALIDATION_SCENES = 312
SCHEMA = "loc3d.scannet_low_annotation_data.v1"
OFFICIAL_POINTS_URL = "https://kaldir.vc.in.tum.de/3dsis/data_efficient/points.zip"
OFFICIAL_POINTS_SHA256 = "c265f249c446155d32e751576288f599caed46ee11a32ecaccd51e037d14cbd6"
THING_CLASSES = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
VALID_CLASSES = frozenset(THING_CLASSES) | {1, 2}
SCANNET20_CLASSES = (1, 2, *THING_CLASSES)
# SHA256 of sorted scene IDs, each followed by one LF; independent of split
# ordering, comments, and line endings. Verified against ScanNet/ScanNet
# Tasks/Benchmark/scannetv2_{train,val}.txt and the research source splits.
OFFICIAL_SPLIT_SHA256 = {
    "train": "366f82219e7021997830566a1302d696137754190cad06fd3a0f0de260b3671b",
    "validation": "25a996252ef67aa84ee091c6fa593ad860bab0addcc72e4b80ef9f64daba8db4",
}
RESEARCH_SOURCE = {
    "prepare_data.py": "8f70e954f87699e060a8dbd6c1db180e3869210d0ed21c5cf31a70948a15b18a",
    "sparse_supervision.py": "f8b1aa61824e48b3f418afc98c7ca4b57865cd70282fd43bcf80eca9b31d7589",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict:
    return {"path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_split(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError(f"Empty or duplicate scene split: {path}")
    if any(re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene) is None for scene in scenes):
        raise ValueError(f"Invalid ScanNet scene name in {path}")
    return scenes


def validate_splits(train: list[str], validation: list[str]) -> None:
    if len(train) != TRAIN_SCENES or len(validation) != VALIDATION_SCENES:
        raise ValueError(f"Official protocol requires {TRAIN_SCENES} train/{VALIDATION_SCENES} validation scenes")
    if set(train) & set(validation):
        raise ValueError("Training/validation scene overlap")
    for name, scenes in (("train", train), ("validation", validation)):
        digest = hashlib.sha256(("\n".join(sorted(scenes)) + "\n").encode("ascii")).hexdigest()
        if digest != OFFICIAL_SPLIT_SHA256[name]:
            raise ValueError(f"Official {name} scene membership differs")


def _load_index_sets(archive: Path, train: list[str], inputs: Path) -> dict[int, dict]:
    # The released file is a legacy torch pickle. Authenticate its exact bytes
    # before deserialization; arbitrary replacement archives are never loaded.
    if sha256(archive) != OFFICIAL_POINTS_SHA256:
        raise ValueError("Official points.zip SHA256 mismatch; obtain the pinned official archive")
    import torch

    result = {}
    with zipfile.ZipFile(archive) as source:
        for budget in BUDGETS:
            payload = source.read(f"points/points{budget}")
            indices = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            if not isinstance(indices, dict) or set(indices) != set(train):
                raise ValueError(f"Official points{budget} scene set differs from training split")
            for scene, values in indices.items():
                array = np.asarray(values)
                if array.ndim != 1 or array.dtype.kind not in "iu" or len(array) != budget:
                    raise ValueError(f"Invalid official points{budget} indices: {scene}")
            (inputs / f"points{budget}").write_bytes(payload)
            result[budget] = indices
    shutil.copyfile(archive, inputs / "official_points.zip")
    return result


def select_annotations(indices, vertex_count: int, segment_ids, groups, label_map: dict) -> dict:
    """Export only selected observations, retaining official order/duplicates.

    Unobserved or invalid labels are -1. Explicit wall/floor observations are 0;
    foreground identities are local and assigned in selected-vertex order.
    """
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("Official indices must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= vertex_count):
        raise ValueError("Out-of-range official vertex index")
    indices = indices.astype(np.int64)
    segment_ids = np.asarray(segment_ids)
    if segment_ids.shape != (vertex_count,) or segment_ids.dtype.kind not in "iu":
        raise ValueError("Segment/mesh vertex alignment changed")
    selected = segment_ids[indices]
    permitted = set(selected.tolist())
    lookup = {}
    for group in groups:
        intersection = permitted.intersection(group["segments"])
        if not intersection:
            continue
        semantic = int(label_map.get(group["label"], 0))
        identity = int(group["objectId"])
        if identity < 0:
            continue
        for segment in intersection:
            value = (identity, semantic)
            if segment in lookup and lookup[segment] != value:
                raise ValueError("Contradictory annotation for a permitted vertex")
            lookup[segment] = value
    instances = np.full(len(indices), -1, dtype=np.int64)
    semantics = np.full(len(indices), -1, dtype=np.int64)
    local_ids = {}
    for index, segment in enumerate(selected):
        pair = lookup.get(int(segment))
        if pair is None or pair[1] not in VALID_CLASSES:
            continue
        identity, semantic = pair
        semantics[index] = semantic
        if semantic in THING_CLASSES:
            if identity not in local_ids:
                local_ids[identity] = len(local_ids) + 1
            instances[index] = local_ids[identity]
        else:
            instances[index] = 0
    return {"vertex_indices": indices, "instance_ids": instances, "semantic_ids": semantics}


def _dense_validation(segment_ids: np.ndarray, groups: list, label_map: dict) -> dict:
    """ScanNet20 object classes; ignored vertices are -1, instances positive."""
    mapping, class_map = {}, {}
    for group in groups:
        semantic = label_map.get(group["label"], 0)
        if semantic not in THING_CLASSES or int(group["objectId"]) < 0:
            continue
        identity = int(group["objectId"]) + 1
        class_id = SCANNET20_CLASSES.index(semantic)
        if identity in class_map and class_map[identity] != class_id:
            raise ValueError("Contradictory validation object class")
        class_map[identity] = class_id
        for segment in group["segments"]:
            if segment in mapping and mapping[segment] != identity:
                raise ValueError("Contradictory validation segment ownership")
            mapping[segment] = identity
    instances = np.fromiter((mapping.get(int(s), -1) for s in segment_ids), dtype=np.int64,
                            count=len(segment_ids))
    pairs = np.asarray(sorted(class_map.items()), dtype=np.int64).reshape(-1, 2)
    return {"instance_ids": instances, "instance_class_pairs": pairs}


def _geometry(raw: Path, pack: Path, scene: str) -> tuple[dict, Path, list[Path]]:
    from plyfile import PlyData

    mesh = raw / f"{scene}_vh_clean_2.ply"
    vertex = PlyData.read(str(mesh), known_list_len={"face": {"vertex_indices": 3}})["vertex"]
    xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    candidates = [pack / "training_pack", pack / "litept_pack", pack]
    source = next((candidate for candidate in candidates if (candidate / "points.npy").is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Geometry pack missing: {pack}")
    paths = [source / f"{name}.npy" for name in ("points", "colors", "normals")]
    geometry = {name: np.load(path, allow_pickle=False)
                for name, path in zip(("points", "colors", "normals"), paths, strict=True)}
    points = geometry["points"]
    if points.shape != xyz.shape or not np.array_equal(points, xyz):
        raise ValueError(f"Original mesh vertex ordering differs: {scene}")
    if len(points) == 0:
        raise ValueError(f"Empty scene geometry: {scene}")
    for name, array in geometry.items():
        if array.shape != xyz.shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Invalid or misaligned geometry {name}: {scene}")
    return geometry, mesh, paths


def prepare(*, raw_root: Path, pack_root: Path, train_split: Path, val_split: Path,
            label_map: Path, points_archive: Path, output: Path) -> dict:
    """Prepare all budgets atomically; refuse any existing output directory."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Prepared data is immutable; use a new output directory: {output}")
    train, validation = read_split(train_split), read_split(val_split)
    validate_splits(train, validation)
    with label_map.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    labels = {}
    for row in rows:
        name, label = row["raw_category"], int(row["nyu40id"])
        if name in labels and labels[name] != label:
            raise ValueError(f"Contradictory label-map category: {name}")
        labels[name] = label
    if not labels:
        raise ValueError("Empty ScanNet label map")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        for folder in ("inputs", "geometry", "validation", *(f"train_annotations/{b}" for b in BUDGETS)):
            (stage / folder).mkdir(parents=True)
        for name, source in (("train.txt", train_split), ("validation.txt", val_split)):
            shutil.copyfile(source, stage / "inputs" / name)
        indices = _load_index_sets(points_archive, train, stage / "inputs")
        files, source_inputs, proofs = {}, {}, []
        for source in (train_split, val_split, label_map, points_archive):
            source_inputs[str(source.resolve())] = _record(source)
        for scene in train + validation:
            raw = raw_root / scene
            geometry, mesh, geometry_sources = _geometry(raw, pack_root / scene, scene)
            segment_path = raw / f"{scene}_vh_clean_2.0.010000.segs.json"
            aggregation_path = raw / f"{scene}.aggregation.json"
            segments = np.asarray(_read_json(segment_path)["segIndices"])
            groups = _read_json(aggregation_path)["segGroups"]
            n = len(geometry["points"])
            if segments.shape != (n,) or segments.dtype.kind not in "iu":
                raise ValueError(f"Segment/mesh alignment changed: {scene}")
            for source in (mesh, *geometry_sources, segment_path, aggregation_path):
                source_inputs[str(source.resolve())] = _record(source)
            np.savez(stage / "geometry" / f"{scene}.npz", **geometry)
            proof = {"scene": scene, "vertices": n, "exact_mesh_order": True, "budgets": {}}
            if scene in indices[BUDGETS[0]]:
                for budget in BUDGETS:
                    annotations = select_annotations(indices[budget][scene], n, segments, groups, labels)
                    if len(annotations["vertex_indices"]) != budget:
                        raise ValueError(f"Official annotation budget changed: {scene}/{budget}")
                    np.savez(stage / "train_annotations" / str(budget) / f"{scene}.npz", **annotations)
                    identities = annotations["instance_ids"]
                    unique = len(np.unique(annotations["vertex_indices"]))
                    proof["budgets"][str(budget)] = {
                        "official": budget, "unique_vertices": unique, "duplicate_entries": budget - unique,
                        "valid": int((identities >= 0).sum()), "foreground": int((identities > 0).sum()),
                        "explicit_wall_floor": int((identities == 0).sum()),
                        "observed_instances": len(np.unique(identities[identities > 0])),
                    }
            else:
                np.savez(stage / "validation" / f"{scene}.npz", **_dense_validation(segments, groups, labels))
            proofs.append(proof)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                files[str(path.relative_to(stage))] = _record(path, relative_to=stage)
        manifest = {
            "schema": SCHEMA, "budgets": list(BUDGETS), "files": files,
            "train_scenes": len(train), "validation_scenes": len(validation),
            "scene_ids": {"train": train, "validation": validation},
            "splits": {"train": files["inputs/train.txt"], "validation": files["inputs/validation.txt"]},
            "scene_proofs": proofs, "source_inputs": source_inputs,
            "split_membership_sha256": OFFICIAL_SPLIT_SHA256,
            "annotation_origin": OFFICIAL_POINTS_URL, "official_archive": files["inputs/official_points.zip"],
            "hidden_training_annotations_exported": False,
            "validation_benchmark": "scannet20", "validation_class_ids": list(SCANNET20_CLASSES),
            "implementation": _record(Path(__file__)),
            "derived_from": {"research_directory": "reports/scannet_la_20261021_v2", "sha256": RESEARCH_SOURCE},
        }
        _write_json(stage / "data_manifest.json", manifest)
        (stage / "data_manifest.sha256").write_text(sha256(stage / "data_manifest.json") + "\n", encoding="ascii")
        verify_prepared(stage)
        # Directory rename publishes only a complete preparation. Never merge
        # with an existing dataset or replace an already sealed dataset.
        if output.exists():
            raise FileExistsError(f"Output appeared while preparing: {output}")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_prepared(root: Path) -> dict:
    """Check the seal and every prepared artifact, independent of source paths."""
    root = root.resolve(strict=True)
    manifest_path = root / "data_manifest.json"
    expected = (root / "data_manifest.sha256").read_text(encoding="ascii").strip()
    if sha256(manifest_path) != expected:
        raise ValueError("Prepared data manifest changed")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("hidden_training_annotations_exported") is not False:
        raise ValueError("Invalid limited-annotation manifest")
    train = read_split(root / "inputs/train.txt")
    validation = read_split(root / "inputs/validation.txt")
    validate_splits(train, validation)
    if manifest.get("scene_ids") != {"train": train, "validation": validation}:
        raise ValueError("Prepared split identity differs")
    expected_files = {"inputs/train.txt", "inputs/validation.txt", "inputs/official_points.zip"}
    expected_files.update(f"inputs/points{budget}" for budget in BUDGETS)
    expected_files.update(f"geometry/{scene}.npz" for scene in train + validation)
    expected_files.update(f"train_annotations/{budget}/{scene}.npz" for scene in train for budget in BUDGETS)
    expected_files.update(f"validation/{scene}.npz" for scene in validation)
    if set(manifest["files"]) != expected_files:
        raise ValueError("Prepared artifact set differs from the sparse-only protocol")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed prepared data: {path}")
        if path.is_file():
            actual.add(str(path.relative_to(root)))
    if actual != expected_files | {"data_manifest.json", "data_manifest.sha256"}:
        raise ValueError("Unexpected or missing prepared files")
    for relative, record in manifest["files"].items():
        path = root / relative
        if record.get("path") != relative or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Prepared artifact changed: {relative}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="Verify an existing preparation and exit")
    for name in ("raw-root", "pack-root", "train-split", "val-split", "label-map", "points-archive", "output"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        manifest = verify_prepared(args.verify)
    else:
        required = ("raw_root", "pack_root", "train_split", "val_split", "label_map", "points_archive", "output")
        missing = ["--" + key.replace("_", "-") for key in required if getattr(args, key) is None]
        if missing:
            parser.error("preparation requires " + ", ".join(missing))
        manifest = prepare(**{key: getattr(args, key) for key in required})
    print(json.dumps({"train_scenes": manifest["train_scenes"], "validation_scenes": manifest["validation_scenes"],
                      "budgets": manifest["budgets"], "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
