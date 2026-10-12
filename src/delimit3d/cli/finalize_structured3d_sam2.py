#!/usr/bin/env python3
"""Validate SAM 2 Structured3D scene packs and seal a dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from delimit3d.cli.export_structured3d_sam2 import (
    ALBEDO_SOURCE,
    ANNOTATION_SCHEMA,
    EMPTY_SOURCE,
    IMAGE_INPUT_POLICY,
    PROJECTION_CACHE_SCHEMA,
    RAWLIGHT_SOURCE,
    RAW_INPUT_SCHEMA,
    SCENE_SCHEMA,
    SOURCE_KEY,
)


DATASET_SCHEMA = "delimit3d_structured3d_sam2_dataset/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _scene_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid scene list: {path}")
    return values


def _artifact_path(pack_scene: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else pack_scene / path


def _validate_cache(
    path: Path,
    *,
    scene_id: str,
    point_count: int,
    physical_frames: list[str],
) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["schema_version"].item()) != PROJECTION_CACHE_SCHEMA:
            raise ValueError(f"{path}: projection-cache schema drift")
        if str(cache["scene_id"].item()) != scene_id:
            raise ValueError(f"{path}: scene identity drift")
        if str(cache["teacher_source_key"].item()) != SOURCE_KEY:
            raise ValueError(f"{path}: teacher source drift")
        cached_frames = [str(value) for value in cache["physical_frame_ids"]]
        if cached_frames != physical_frames:
            raise ValueError(f"{path}: physical frame ordering drift")

        frame_count = len(physical_frames)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        visible_offsets = np.asarray(
            cache["visible_frame_offsets"], dtype=np.int64
        )
        visible = np.asarray(cache["visible_point_indices"], dtype=np.int64)
        known_offsets = np.asarray(cache["frame_offsets"], dtype=np.int64)
        known = np.asarray(cache["eligible_point_indices"], dtype=np.int64)
        proposal_frame_offsets = np.asarray(
            cache["proposal_frame_offsets"], dtype=np.int64
        )
        proposal_point_offsets = np.asarray(
            cache["proposal_point_offsets"], dtype=np.int64
        )
        members = np.asarray(cache["proposal_point_indices"], dtype=np.int64)
        proposal_count = int(proposal_frame_offsets[-1])

        expected_shapes = {
            "split_codes": (split_codes.shape, (frame_count,)),
            "visible_frame_offsets": (
                visible_offsets.shape,
                (frame_count + 1,),
            ),
            "frame_offsets": (known_offsets.shape, (frame_count + 1,)),
            "proposal_frame_offsets": (
                proposal_frame_offsets.shape,
                (frame_count + 1,),
            ),
            "proposal_point_offsets": (
                proposal_point_offsets.shape,
                (proposal_count + 1,),
            ),
            "proposal_order_indices": (
                cache["proposal_order_indices"].shape,
                (proposal_count,),
            ),
            "proposal_source_indices": (
                cache["proposal_source_indices"].shape,
                (proposal_count,),
            ),
            "proposal_areas": (cache["proposal_areas"].shape, (proposal_count,)),
            "proposal_predicted_ious": (
                cache["proposal_predicted_ious"].shape,
                (proposal_count,),
            ),
            "proposal_stability_scores": (
                cache["proposal_stability_scores"].shape,
                (proposal_count,),
            ),
            "proposal_trainable": (
                cache["proposal_trainable"].shape,
                (proposal_count,),
            ),
        }
        for name, (actual, expected) in expected_shapes.items():
            if actual != expected:
                raise ValueError(f"{path}: {name} shape {actual} != {expected}")
        if not bool(np.isin(split_codes, (0, 1)).all()):
            raise ValueError(f"{path}: invalid frame split codes")
        for name, offsets, final_size in (
            ("visible", visible_offsets, visible.size),
            ("known", known_offsets, known.size),
            ("proposal frames", proposal_frame_offsets, proposal_count),
            ("proposal points", proposal_point_offsets, members.size),
        ):
            if (
                offsets.size == 0
                or int(offsets[0]) != 0
                or bool(np.any(offsets[1:] < offsets[:-1]))
                or int(offsets[-1]) != int(final_size)
            ):
                raise ValueError(f"{path}: invalid {name} offsets")
        for name, values in (("visible", visible), ("known", known), ("member", members)):
            if values.size and (
                int(values.min()) < 0 or int(values.max()) >= point_count
            ):
                raise ValueError(f"{path}: {name} point index outside scene")
        supports = np.diff(proposal_point_offsets)
        visible_counts_by_proposal = np.repeat(
            np.diff(visible_offsets), np.diff(proposal_frame_offsets)
        )
        known_counts_by_proposal = np.repeat(
            np.diff(known_offsets), np.diff(proposal_frame_offsets)
        )
        trainable = np.asarray(cache["proposal_trainable"], dtype=np.bool_)
        if supports.size and (
            bool(np.any(supports < 2))
            or bool(np.any(supports >= visible_counts_by_proposal))
            or not np.array_equal(trainable, supports < known_counts_by_proposal)
        ):
            raise ValueError(f"{path}: invalid projected proposal support")
        for frame_index in range(frame_count):
            visible_row = visible[
                visible_offsets[frame_index] : visible_offsets[frame_index + 1]
            ]
            known_row = known[
                known_offsets[frame_index] : known_offsets[frame_index + 1]
            ]
            if known_row.size and not bool(np.isin(known_row, visible_row).all()):
                raise ValueError(f"{path}: known points are not visible in frame")

        proposal_split_codes = np.repeat(
            split_codes, np.diff(proposal_frame_offsets)
        )
        trainable_proposals = int(
            np.count_nonzero(trainable & (proposal_split_codes == 0))
        )
        heldout_trainable_proposals = int(
            np.count_nonzero(trainable & (proposal_split_codes == 1))
        )

    return {
        "physical_frames": frame_count,
        "visible_memberships": int(visible.size),
        "known_memberships": int(known.size),
        "retained_proposals": proposal_count,
        "projected_memberships": int(members.size),
        "trainable_proposals": trainable_proposals,
        "heldout_trainable_proposals": heldout_trainable_proposals,
    }


def _validate_scene(
    *, scene_id: str, pack_root: Path, verify_cache_hash: bool
) -> dict[str, Any]:
    pack_scene = pack_root / scene_id
    manifest_path = pack_scene / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCENE_SCHEMA or not manifest.get("complete"):
        raise ValueError(f"{manifest_path}: incomplete or wrong schema")
    if manifest.get("scene_id") != scene_id:
        raise ValueError(f"{manifest_path}: scene identity drift")
    if manifest.get("source_cells") != [SOURCE_KEY]:
        raise ValueError(f"{manifest_path}: source-cell drift")
    if manifest.get("teacher_source_key") != SOURCE_KEY:
        raise ValueError(f"{manifest_path}: teacher-source drift")
    if manifest.get("crop", {}).get("mode") != "full_scene":
        raise ValueError(f"{manifest_path}: not a full-scene pack")
    if manifest.get("teacher", {}).get("image_input_policy") != IMAGE_INPUT_POLICY:
        raise ValueError(f"{manifest_path}: image-input policy drift")
    if not all(bool(value) for value in manifest.get("checks", {}).values()):
        raise ValueError(f"{manifest_path}: at least one scene check failed")

    reference_manifest_path = Path(manifest["reference"]["source_manifest"])
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    if _sha256(reference_manifest_path) != manifest["reference"][
        "source_manifest_sha256"
    ]:
        raise ValueError(f"{manifest_path}: reference manifest hash drift")
    point_count = int(manifest["crop"]["num_points"])

    artifacts = manifest["artifacts"]
    for key in ("points", "colors", "normals", "crop_indices"):
        record = artifacts[key]
        path = _artifact_path(pack_scene, record).resolve(strict=True)
        reference_record = reference_manifest["artifacts"][key]
        reference_path = _artifact_path(
            reference_manifest_path.parent, reference_record
        ).resolve(strict=True)
        if (record.get("storage") != "verified_copy_of_immutable_reference_geometry"
                and not os.path.samefile(path, reference_path)):
            raise ValueError(f"{manifest_path}: {key} is not the accepted hardlink")
        if (
            int(record["size_bytes"]) != path.stat().st_size
            or str(record["sha256"]) != str(reference_record["sha256"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"{manifest_path}: {key} provenance drift")

    annotation_record = artifacts["annotation_manifest"]
    annotation_path = _artifact_path(pack_scene, annotation_record).resolve(strict=True)
    if (
        annotation_path.stat().st_size != int(annotation_record["size_bytes"])
        or _sha256(annotation_path) != annotation_record["sha256"]
    ):
        raise ValueError(f"{manifest_path}: annotation manifest drift")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if (
        annotation.get("schema_version") != ANNOTATION_SCHEMA
        or not annotation.get("complete")
        or annotation.get("scene_id") != scene_id
        or annotation.get("teacher") != manifest.get("teacher")
    ):
        raise ValueError(f"{annotation_path}: invalid annotation manifest")
    physical_frames = [str(value) for value in manifest["frames"]["physical"]]
    if [str(row["frame_id"]) for row in annotation["frames"]] != physical_frames:
        raise ValueError(f"{annotation_path}: frame ordering drift")
    raw_dir = annotation_path.parent / "raw_sam2"
    expected_raw = sorted(f"{frame_id}.npz" for frame_id in physical_frames)
    if sorted(path.name for path in raw_dir.glob("*.npz")) != expected_raw:
        raise ValueError(f"{annotation_path}: raw mask frame-set drift")
    expected_inputs = sorted(
        f"{frame_id}.input.json" for frame_id in physical_frames
    )
    if sorted(path.name for path in raw_dir.glob("*.input.json")) != expected_inputs:
        raise ValueError(f"{annotation_path}: raw input frame-set drift")
    for row in annotation["frames"]:
        frame_id = str(row["frame_id"])
        input_source = str(row.get("input_source", ""))
        if input_source not in (RAWLIGHT_SOURCE, ALBEDO_SOURCE, EMPTY_SOURCE):
            raise ValueError(f"{annotation_path}: invalid input source for {frame_id}")
        record = row.get("raw_artifact", {})
        raw_path = _artifact_path(annotation_path.parent, record).resolve(strict=True)
        if raw_path.parent != raw_dir.resolve(strict=True):
            raise ValueError(f"{annotation_path}: raw mask path drift")
        if (
            raw_path.stat().st_size != int(record["size_bytes"])
            or _sha256(raw_path) != record["sha256"]
        ):
            raise ValueError(f"{annotation_path}: raw mask artifact drift")
        input_record = row.get("input_metadata_artifact", {})
        input_path = _artifact_path(
            annotation_path.parent, input_record
        ).resolve(strict=True)
        if input_path.parent != raw_dir.resolve(strict=True):
            raise ValueError(f"{annotation_path}: raw input path drift")
        if input_path.name != f"{frame_id}.input.json":
            raise ValueError(f"{annotation_path}: raw input identity drift")
        if (
            input_path.stat().st_size != int(input_record["size_bytes"])
            or _sha256(input_path) != input_record["sha256"]
        ):
            raise ValueError(f"{annotation_path}: raw input artifact drift")
        input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        selection = input_payload.get("selection", {})
        raw_identity = input_payload.get("raw_artifact", {})
        if (
            input_payload.get("schema_version") != RAW_INPUT_SCHEMA
            or not input_payload.get("complete")
            or input_payload.get("scene_id") != scene_id
            or input_payload.get("frame_id") != frame_id
            or selection.get("policy") != IMAGE_INPUT_POLICY
            or selection.get("selected_source") != input_source
            or selection.get("selected_member") != row.get("teacher_input_member")
            or selection.get("selected_sha256") != row.get("teacher_input_sha256")
            or raw_identity.get("filename") != raw_path.name
            or int(raw_identity.get("size_bytes", -1)) != raw_path.stat().st_size
            or raw_identity.get("sha256") != record["sha256"]
        ):
            raise ValueError(f"{annotation_path}: raw input provenance drift")
        if input_source == EMPTY_SOURCE and (
            row.get("teacher_input_member") is not None
            or row.get("teacher_input_sha256") is not None
            or int(row.get("num_raw_proposals", -1)) != 0
        ):
            raise ValueError(f"{annotation_path}: invalid empty-frame input")
        if int(row.get("num_raw_proposals", -1)) == 0 and (
            int(row.get("known_points", -1)) != 0
            or int(row.get("retained_projected_proposals", -1)) != 0
            or int(row.get("trainable_projected_proposals", -1)) != 0
        ):
            raise ValueError(f"{annotation_path}: empty frame created supervision")

    source_counts = {
        source: sum(
            str(row.get("input_source")) == source for row in annotation["frames"]
        )
        for source in (RAWLIGHT_SOURCE, ALBEDO_SOURCE, EMPTY_SOURCE)
    }
    empty_frames = sum(
        int(row.get("num_raw_proposals", -1)) == 0
        for row in annotation["frames"]
    )
    if (
        annotation.get("summary", {}).get("input_source_counts") != source_counts
        or int(annotation.get("summary", {}).get("empty_teacher_frames", -1))
        != empty_frames
    ):
        raise ValueError(f"{annotation_path}: input summary drift")

    cache_record = artifacts["sam2_projection_cache"]
    cache_path = _artifact_path(pack_scene, cache_record).resolve(strict=True)
    if cache_path.stat().st_size != int(cache_record["size_bytes"]):
        raise ValueError(f"{manifest_path}: cache size drift")
    if verify_cache_hash and _sha256(cache_path) != cache_record["sha256"]:
        raise ValueError(f"{manifest_path}: cache hash drift")
    cache_counts = _validate_cache(
        cache_path,
        scene_id=scene_id,
        point_count=point_count,
        physical_frames=physical_frames,
    )
    if cache_counts != manifest["counts"]["projection_cache"]:
        raise ValueError(f"{manifest_path}: cache count drift")
    if (
        cache_counts["trainable_proposals"]
        != int(manifest["counts"]["trainable_proposals"][SOURCE_KEY])
        or cache_counts["heldout_trainable_proposals"]
        != int(manifest["counts"]["heldout_trainable_proposals"][SOURCE_KEY])
    ):
        raise ValueError(f"{manifest_path}: split proposal count drift")
    return {
        "scene_id": scene_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "point_count": point_count,
        "physical_frames": len(physical_frames),
        "train_frames": len(manifest["frames"]["train"]),
        "heldout_frames": len(manifest["frames"]["heldout"]),
        "trainable_proposals": int(
            manifest["counts"]["trainable_proposals"][SOURCE_KEY]
        ),
        "heldout_trainable_proposals": int(
            manifest["counts"]["heldout_trainable_proposals"][SOURCE_KEY]
        ),
        "cache_size_bytes": cache_path.stat().st_size,
        "raw_mask_size_bytes": sum(path.stat().st_size for path in raw_dir.glob("*.npz")),
        "rawlight_frames": source_counts[RAWLIGHT_SOURCE],
        "albedo_fallback_frames": source_counts[ALBEDO_SOURCE],
        "doubly_saturated_frames": source_counts[EMPTY_SOURCE],
        "empty_teacher_frames": empty_frames,
        "teacher": manifest["teacher"],
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--verify-cache-hashes", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    scene_ids_file = args.scene_ids_file.resolve(strict=True)
    scene_ids = _scene_ids(scene_ids_file)
    pack_root = args.pack_root.resolve(strict=True)
    rows = [
        _validate_scene(
            scene_id=scene_id,
            pack_root=pack_root,
            verify_cache_hash=bool(args.verify_cache_hashes),
        )
        for scene_id in scene_ids
    ]
    teacher_hashes = {
        str(row["teacher"]["checkpoint"]["sha256"]) for row in rows
    }
    teacher_settings = {
        json.dumps(
            row["teacher"]["automatic_mask_generator"], sort_keys=True
        )
        for row in rows
    }
    teacher_input_policies = {
        json.dumps(row["teacher"]["image_input_policy"], sort_keys=True)
        for row in rows
    }
    if (
        len(teacher_hashes) != 1
        or len(teacher_settings) != 1
        or len(teacher_input_policies) != 1
    ):
        raise ValueError("Teacher provenance differs across scene packs")

    payload = {
        "schema_version": DATASET_SCHEMA,
        "complete": True,
        "label": str(args.label),
        "teacher_source_key": SOURCE_KEY,
        "source_cells": [SOURCE_KEY],
        "single_granularity": True,
        "full_scene": True,
        "scene_ids_file": {
            "path": str(scene_ids_file),
            "size_bytes": scene_ids_file.stat().st_size,
            "sha256": _sha256(scene_ids_file),
        },
        "pack_root": str(pack_root),
        "scene_count": len(rows),
        "physical_frames": sum(int(row["physical_frames"]) for row in rows),
        "train_frames": sum(int(row["train_frames"]) for row in rows),
        "heldout_frames": sum(int(row["heldout_frames"]) for row in rows),
        "point_count": sum(int(row["point_count"]) for row in rows),
        "trainable_proposals": sum(
            int(row["trainable_proposals"]) for row in rows
        ),
        "heldout_trainable_proposals": sum(
            int(row["heldout_trainable_proposals"]) for row in rows
        ),
        "raw_mask_size_bytes": sum(
            int(row["raw_mask_size_bytes"]) for row in rows
        ),
        "projection_cache_size_bytes": sum(
            int(row["cache_size_bytes"]) for row in rows
        ),
        "teacher_checkpoint_sha256": next(iter(teacher_hashes)),
        "automatic_mask_generator": json.loads(next(iter(teacher_settings))),
        "image_input_policy": json.loads(next(iter(teacher_input_policies))),
        "rawlight_frames": sum(int(row["rawlight_frames"]) for row in rows),
        "albedo_fallback_frames": sum(
            int(row["albedo_fallback_frames"]) for row in rows
        ),
        "doubly_saturated_frames": sum(
            int(row["doubly_saturated_frames"]) for row in rows
        ),
        "empty_teacher_frames": sum(
            int(row["empty_teacher_frames"]) for row in rows
        ),
        "cache_hashes_verified": bool(args.verify_cache_hashes),
        "scenes": rows,
    }
    _atomic_json(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
