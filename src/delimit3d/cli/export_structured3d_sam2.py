#!/usr/bin/env python3
"""Export single-granularity SAM 2 supervision for full Structured3D scenes.

The exporter keeps overlapping SAM 2 masks frame-local, projects them onto the
accepted full-scene geometry, and commits one compact training cache per scene.
It is restartable at frame granularity and never invents cross-view identities
or UnSAM-style granularity labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import cv2
import numpy as np

from delimit3d.data.unsam_proposals import (
    UnSAMProposalSet,
    load_proposal_set,
    proposal_overlap_statistics,
    proposals_from_teacher_output,
    save_proposal_set,
)
from delimit3d.source.common.types import FrameRecord, VisibilityConfig
from delimit3d.source.core.lifting.project import project_points_to_image
from delimit3d.source.core.lifting.visibility import compute_visible_points


log = logging.getLogger("export_structured3d_sam2")

SCENE_SCHEMA = "delimit3d_structured3d_sam2_scene_source/v2"
ANNOTATION_SCHEMA = "delimit3d_structured3d_sam2_annotation/v2"
PROJECTED_FRAME_SCHEMA = "delimit3d_sam2_projected_frame/v2"
PROJECTION_CACHE_SCHEMA = "delimit3d_structured3d_sam2_projection_cache/v1"
WORKER_REPORT_SCHEMA = "delimit3d_structured3d_sam2_worker/v2"
RAW_INPUT_SCHEMA = "delimit3d_structured3d_sam2_frame_input/v1"
SOURCE_KEY = "2d/sam2"
MIN_PROJECTED_POINTS = 2
RAWLIGHT_SOURCE = "rgb_rawlight"
ALBEDO_SOURCE = "albedo"
EMPTY_SOURCE = "empty"
SATURATION_CHANNEL_THRESHOLD = 250
SATURATION_PIXEL_FRACTION = 0.99
IMAGE_INPUT_POLICY = {
    "schema_version": "delimit3d_structured3d_sam2_image_input_policy/v1",
    "primary_source": "rgb_rawlight.png",
    "fallback_source": "albedo.png",
    "saturation_definition": "fraction_pixels_all_rgb_channels_gte_threshold",
    "channel_threshold": SATURATION_CHANNEL_THRESHOLD,
    "minimum_pixel_fraction": SATURATION_PIXEL_FRACTION,
    "fallback_trigger": "primary_saturation_only_independent_of_sam2_output",
    "doubly_saturated_policy": "valid_empty_frame_no_supervision",
}
CAMERA_AXIS_CONVERSION = np.asarray(
    [[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float32
)
ARCHIVE_SCENE_STARTS = (
    (0, "00"),
    (200, "01"),
    (400, "02"),
    (600, "03"),
    (800, "04"),
    (1000, "05"),
    (1200, "06"),
    (1400, "07"),
    (1800, "08"),
    (2000, "10"),
    (2200, "11"),
    (2400, "12"),
    (2600, "13"),
    (2800, "14"),
    (3000, "15"),
    (3200, "16"),
    (3400, "17"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _saturation_fraction(image: np.ndarray) -> float:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {image.dtype} {image.shape}")
    return float(
        np.all(image >= SATURATION_CHANNEL_THRESHOLD, axis=2).mean()
    )


def _choose_input_source(
    rawlight_saturation_fraction: float,
    albedo_saturation_fraction: float | None,
) -> str:
    if rawlight_saturation_fraction < SATURATION_PIXEL_FRACTION:
        return RAWLIGHT_SOURCE
    if (
        albedo_saturation_fraction is not None
        and albedo_saturation_fraction < SATURATION_PIXEL_FRACTION
    ):
        return ALBEDO_SOURCE
    return EMPTY_SOURCE


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    stored_path = (
        str(resolved.relative_to(relative_to.resolve(strict=True)))
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": stored_path,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


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


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_scene_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Scene list is empty or contains duplicates: {path}")
    if any(not value.startswith("scene_") for value in values):
        raise ValueError(f"Invalid Structured3D scene ID in {path}")
    return values


def _archive_code_for_scene(scene_id: str) -> str:
    try:
        number = int(scene_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Structured3D scene ID: {scene_id}") from exc
    code: str | None = None
    for start, candidate in ARCHIVE_SCENE_STARTS:
        if number >= start:
            code = candidate
        else:
            break
    if code is None:
        raise ValueError(f"No archive mapping for {scene_id}")
    return code


class Structured3DReader:
    """Minimal ZIP reader that opens one perspective archive at a time."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve(strict=True)
        self.reader = zipfile.ZipFile(self.path, "r")
        self.names = set(self.reader.namelist())

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "Structured3DReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def listdir(self, directory: str) -> list[str]:
        prefix = directory.strip("/") + "/"
        values = {
            name[len(prefix) :].split("/", 1)[0]
            for name in self.names
            if name.startswith(prefix) and name != prefix
        }
        values.discard("")
        return sorted(values, key=_natural_key)

    def read(self, name: str) -> bytes:
        return self.reader.read(name)


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return sys.maxsize, value


@dataclass(frozen=True)
class _ArchiveFrame:
    room: str
    frame: str
    record: FrameRecord


@dataclass(frozen=True)
class TeacherImageSelection:
    source: str
    reason: str
    image: np.ndarray | None
    image_shape: tuple[int, int]
    selected_member: str | None
    selected_sha256: str | None
    rawlight_member: str
    rawlight_sha256: str
    rawlight_saturation_fraction: float
    albedo_member: str | None
    albedo_sha256: str | None
    albedo_saturation_fraction: float | None

    def provenance(self) -> dict[str, Any]:
        return {
            "policy": dict(IMAGE_INPUT_POLICY),
            "selected_source": self.source,
            "selection_reason": self.reason,
            "image_shape": list(self.image_shape),
            "selected_member": self.selected_member,
            "selected_sha256": self.selected_sha256,
            "rawlight": {
                "member": self.rawlight_member,
                "sha256": self.rawlight_sha256,
                "saturation_fraction": self.rawlight_saturation_fraction,
            },
            "albedo": (
                {
                    "member": self.albedo_member,
                    "sha256": self.albedo_sha256,
                    "saturation_fraction": self.albedo_saturation_fraction,
                }
                if self.albedo_member is not None
                else None
            ),
        }


class Structured3DArchiveAdapter:
    """Read one Structured3D scene without extracting its perspective ZIP."""

    def __init__(self, *, scene_id: str, reader: Structured3DReader) -> None:
        self.scene_id = scene_id
        self.reader = reader
        self._frames: list[_ArchiveFrame] | None = None
        self._by_id: dict[str, _ArchiveFrame] = {}
        self._camera_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _base(self, value: _ArchiveFrame) -> str:
        return (
            f"Structured3D/{self.scene_id}/2D_rendering/{value.room}/"
            f"perspective/full/{value.frame}"
        )

    def list_frames(self) -> list[FrameRecord]:
        if self._frames is None:
            root = f"Structured3D/{self.scene_id}/2D_rendering"
            rows: list[_ArchiveFrame] = []
            for room in self.reader.listdir(root):
                frame_root = f"{root}/{room}/perspective/full"
                for frame in self.reader.listdir(frame_root):
                    prefix = f"{frame_root}/{frame}"
                    required = (
                        f"{prefix}/rgb_rawlight.png",
                        f"{prefix}/depth.png",
                        f"{prefix}/camera_pose.txt",
                    )
                    if not all(name in self.reader.names for name in required):
                        continue
                    frame_id = f"{room}:{frame}"
                    record = FrameRecord(
                        frame_id=frame_id,
                        rgb_path=Path(f"archive:/{prefix}/rgb_rawlight.png"),
                        depth_path=Path(f"archive:/{prefix}/depth.png"),
                        pose_path=Path(f"archive:/{prefix}/camera_pose.txt"),
                        intrinsics_path=Path(f"archive:/{prefix}/camera_pose.txt"),
                    )
                    value = _ArchiveFrame(room=room, frame=frame, record=record)
                    rows.append(value)
                    self._by_id[frame_id] = value
            if not rows:
                raise RuntimeError(f"{self.scene_id}: no complete RGB-D frames")
            self._frames = rows
        return [value.record for value in self._frames]

    def _value(self, frame: FrameRecord) -> _ArchiveFrame:
        if not self._by_id:
            self.list_frames()
        return self._by_id[frame.frame_id]

    def _load_color(
        self, frame: FrameRecord, filename: str
    ) -> tuple[np.ndarray, str, str]:
        value = self._value(frame)
        member = f"{self._base(value)}/{filename}"
        payload = self.reader.read(member)
        image = cv2.imdecode(
            np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image is None or image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(
                f"{self.scene_id}/{frame.frame_id}: invalid {filename}"
            )
        rgb = np.ascontiguousarray(image[..., :3][..., ::-1])
        return rgb, member, _sha256_bytes(payload)

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        return self._load_color(frame, "rgb_rawlight.png")[0]

    def select_teacher_image(self, frame: FrameRecord) -> TeacherImageSelection:
        rawlight, rawlight_member, rawlight_sha256 = self._load_color(
            frame, "rgb_rawlight.png"
        )
        rawlight_saturation = _saturation_fraction(rawlight)
        image_shape = (int(rawlight.shape[0]), int(rawlight.shape[1]))
        if _choose_input_source(rawlight_saturation, None) == RAWLIGHT_SOURCE:
            return TeacherImageSelection(
                source=RAWLIGHT_SOURCE,
                reason="rawlight_below_saturation_threshold",
                image=rawlight,
                image_shape=image_shape,
                selected_member=rawlight_member,
                selected_sha256=rawlight_sha256,
                rawlight_member=rawlight_member,
                rawlight_sha256=rawlight_sha256,
                rawlight_saturation_fraction=rawlight_saturation,
                albedo_member=None,
                albedo_sha256=None,
                albedo_saturation_fraction=None,
            )

        value = self._value(frame)
        albedo_member = f"{self._base(value)}/albedo.png"
        if albedo_member not in self.reader.names:
            return TeacherImageSelection(
                source=EMPTY_SOURCE,
                reason="rawlight_saturated_albedo_missing",
                image=None,
                image_shape=image_shape,
                selected_member=None,
                selected_sha256=None,
                rawlight_member=rawlight_member,
                rawlight_sha256=rawlight_sha256,
                rawlight_saturation_fraction=rawlight_saturation,
                albedo_member=None,
                albedo_sha256=None,
                albedo_saturation_fraction=None,
            )
        albedo, loaded_albedo_member, albedo_sha256 = self._load_color(
            frame, "albedo.png"
        )
        if albedo.shape != rawlight.shape:
            raise ValueError(
                f"{self.scene_id}/{frame.frame_id}: rawlight/albedo shape drift "
                f"{rawlight.shape} != {albedo.shape}"
            )
        albedo_saturation = _saturation_fraction(albedo)
        if (
            _choose_input_source(rawlight_saturation, albedo_saturation)
            == EMPTY_SOURCE
        ):
            return TeacherImageSelection(
                source=EMPTY_SOURCE,
                reason="rawlight_and_albedo_saturated",
                image=None,
                image_shape=image_shape,
                selected_member=None,
                selected_sha256=None,
                rawlight_member=rawlight_member,
                rawlight_sha256=rawlight_sha256,
                rawlight_saturation_fraction=rawlight_saturation,
                albedo_member=loaded_albedo_member,
                albedo_sha256=albedo_sha256,
                albedo_saturation_fraction=albedo_saturation,
            )
        return TeacherImageSelection(
            source=ALBEDO_SOURCE,
            reason="rawlight_saturated_albedo_below_threshold",
            image=albedo,
            image_shape=image_shape,
            selected_member=loaded_albedo_member,
            selected_sha256=albedo_sha256,
            rawlight_member=rawlight_member,
            rawlight_sha256=rawlight_sha256,
            rawlight_saturation_fraction=rawlight_saturation,
            albedo_member=loaded_albedo_member,
            albedo_sha256=albedo_sha256,
            albedo_saturation_fraction=albedo_saturation,
        )

    def _camera(self, frame: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._camera_cache.get(frame.frame_id)
        if cached is not None:
            return cached
        value = self._value(frame)
        camera = np.fromstring(
            self.reader.read(f"{self._base(value)}/camera_pose.txt"),
            dtype=np.float32,
            sep=" ",
        )
        if camera.ndim != 1 or camera.size < 11:
            raise ValueError(
                f"{self.scene_id}/{frame.frame_id}: invalid camera {camera.shape}"
            )
        z_to_y = np.asarray(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        )
        translation = z_to_y @ (camera[:3] / 1000.0)
        front = camera[3:6]
        up = camera[6:9]
        normal = np.cross(front, up)
        rotation = z_to_y @ np.stack((front, up, normal), axis=1)

        depth_data = np.frombuffer(
            self.reader.read(f"{self._base(value)}/depth.png"), np.uint8
        )
        depth = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise RuntimeError(f"{self.scene_id}/{frame.frame_id}: invalid depth")
        height, width = depth.shape
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 2] = width / 2.0
        intrinsic[1, 2] = height / 2.0
        intrinsic[0, 0] = intrinsic[0, 2] / np.tan(camera[9])
        intrinsic[1, 1] = intrinsic[1, 2] / np.tan(camera[10])

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation @ CAMERA_AXIS_CONVERSION.T
        pose[:3, 3] = translation
        result = (pose, intrinsic, depth.astype(np.float32) / 1000.0)
        self._camera_cache[frame.frame_id] = result
        return result

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        return self._camera(frame)[0]

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return self._camera(frame)[1]

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        return self._camera(frame)[2]


class SAM2Provider:
    def __init__(self, args: argparse.Namespace) -> None:
        sam2_root = args.sam2_root.resolve(strict=True)
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 2 export requires a CUDA GPU")
        properties = torch.cuda.get_device_properties(0)
        if properties.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.device_name = str(torch.cuda.get_device_name(0))
        self.checkpoint = args.checkpoint.resolve(strict=True)
        self.sam2_root = sam2_root
        self.model_config = str(args.model_config)
        self.settings = {
            "points_per_side": int(args.points_per_side),
            "points_per_batch": int(args.points_per_batch),
            "pred_iou_thresh": float(args.pred_iou_thresh),
            "stability_score_thresh": float(args.stability_score_thresh),
            "stability_score_offset": float(args.stability_score_offset),
            "mask_threshold": float(args.mask_threshold),
            "box_nms_thresh": float(args.box_nms_thresh),
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.7,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
            "output_mode": "binary_mask",
            "use_m2m": False,
            "multimask_output": True,
        }
        started = time.time()
        model = build_sam2(
            self.model_config,
            str(self.checkpoint),
            device=str(args.device),
            apply_postprocessing=True,
        )
        self.generator = SAM2AutomaticMaskGenerator(model=model, **self.settings)
        self.model_load_seconds = time.time() - started
        self.peak_memory_bytes = 0
        self._provenance = {
            "name": "official_sam2.1_hiera_large",
            "official_repository": str(self.sam2_root),
            "official_repository_commit": _git_head(self.sam2_root),
            "checkpoint": _file_record(self.checkpoint),
            "model_config": self.model_config,
            "automatic_mask_generator": dict(self.settings),
            "image_input_policy": dict(IMAGE_INPUT_POLICY),
        }

    def generate(self, image: np.ndarray) -> tuple[UnSAMProposalSet, float, int]:
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            masks_data = self.generator.generate(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak)
        proposals = proposals_from_teacher_output(
            masks_data,
            image_shape=(int(image.shape[0]), int(image.shape[1])),
        )
        del masks_data
        return proposals, elapsed, peak

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)


def _raw_artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "filename": resolved.name,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _raw_input_payload(
    *,
    scene_id: str,
    frame_id: str,
    selection: TeacherImageSelection,
    raw_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RAW_INPUT_SCHEMA,
        "complete": True,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "selection": selection.provenance(),
        "raw_artifact": _raw_artifact_identity(raw_path),
    }


def _raw_input_matches(
    *,
    path: Path,
    scene_id: str,
    frame_id: str,
    selection: TeacherImageSelection,
    raw_path: Path,
) -> bool:
    if not path.is_file() or not raw_path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
        expected = _raw_input_payload(
            scene_id=scene_id,
            frame_id=frame_id,
            selection=selection,
            raw_path=raw_path,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return actual == expected


def _empty_proposals(image_shape: tuple[int, int]) -> UnSAMProposalSet:
    return proposals_from_teacher_output((), image_shape=image_shape)


def _project_geometry(
    *, adapter: Structured3DArchiveAdapter, frame: FrameRecord, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = adapter.load_depth_m(frame)
    u, v, z, valid = project_points_to_image(
        points,
        adapter.load_pose_c2w(frame),
        adapter.load_intrinsics(frame),
    )
    visible, visible_u, visible_v = compute_visible_points(
        u,
        v,
        z,
        valid,
        depth,
        VisibilityConfig(
            min_depth_m=0.01,
            z_tolerance_m=1.0,
            depth_scale_to_m=1.0,
            depth_aligned_to_rgb=True,
        ),
    )
    return (
        visible.astype(np.int64, copy=False),
        visible_u.astype(np.int32, copy=False),
        visible_v.astype(np.int32, copy=False),
    )


def _save_projected_frame(
    *,
    path: Path,
    frame_id: str,
    proposals: UnSAMProposalSet,
    visible: np.ndarray,
    visible_u: np.ndarray,
    visible_v: np.ndarray,
) -> dict[str, Any]:
    membership = proposals.masks[:, visible_v, visible_u]
    all_supports = membership.sum(axis=1, dtype=np.int64)
    retained = np.where(
        (all_supports >= MIN_PROJECTED_POINTS)
        & (all_supports < int(visible.size))
    )[0].astype(np.int32, copy=False)
    supports = all_supports[retained]
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(supports, dtype=np.int64))
    )
    members = (
        np.concatenate([visible[membership[index]] for index in retained]).astype(
            np.int64, copy=False
        )
        if retained.size
        else np.zeros(0, dtype=np.int64)
    )
    membership_count = (
        membership[retained].sum(axis=0, dtype=np.uint16)
        if retained.size
        else np.zeros(visible.size, dtype=np.uint16)
    )
    known = visible[membership_count > 0].astype(np.int64, copy=False)
    trainable = (supports < int(known.size)).astype(np.bool_, copy=False)
    _atomic_savez(
        path,
        schema_version=np.asarray(PROJECTED_FRAME_SCHEMA),
        teacher_source_key=np.asarray(SOURCE_KEY),
        frame_id=np.asarray(frame_id),
        visible_point_indices=visible,
        known_point_indices=known,
        proposal_offsets=offsets,
        proposal_point_indices=members,
        proposal_order_indices=retained,
        proposal_source_indices=proposals.source_indices[retained].astype(
            np.int32, copy=False
        ),
        proposal_areas=proposals.areas[retained].astype(np.int64, copy=False),
        proposal_predicted_ious=proposals.predicted_ious[retained].astype(
            np.float32, copy=False
        ),
        proposal_stability_scores=proposals.stability_scores[retained].astype(
            np.float32, copy=False
        ),
        proposal_trainable=trainable,
        projected_multiple_membership_points=np.asarray(
            np.count_nonzero(membership_count > 1), dtype=np.int64
        ),
        projected_max_memberships_per_point=np.asarray(
            membership_count.max(initial=0), dtype=np.int64
        ),
    )
    return {
        "visible_points": int(visible.size),
        "known_points": int(known.size),
        "retained_projected_proposals": int(retained.size),
        "trainable_projected_proposals": int(np.count_nonzero(trainable)),
        "projected_memberships": int(members.size),
        "projected_multiple_membership_points": int(
            np.count_nonzero(membership_count > 1)
        ),
        "projected_max_memberships_per_point": int(
            membership_count.max(initial=0)
        ),
        "projected_support_min": int(supports.min()) if supports.size else 0,
        "projected_support_median": (
            float(np.median(supports)) if supports.size else 0.0
        ),
        "projected_support_max": int(supports.max()) if supports.size else 0,
    }


def _load_projected(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema_version"].item())
        if schema != PROJECTED_FRAME_SCHEMA:
            raise ValueError(f"Unsupported projected schema {schema!r}: {path}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _compact_projection_cache(
    *,
    output: Path,
    scene_id: str,
    frame_ids: Sequence[str],
    train_frames: set[str],
    projected_dir: Path,
    crop_indices_sha256: str,
) -> dict[str, Any]:
    visible_offsets = [0]
    known_offsets = [0]
    proposal_frame_offsets = [0]
    proposal_point_offsets = [0]
    visible_parts: list[np.ndarray] = []
    known_parts: list[np.ndarray] = []
    member_parts: list[np.ndarray] = []
    order_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    iou_parts: list[np.ndarray] = []
    stability_parts: list[np.ndarray] = []
    trainable_parts: list[np.ndarray] = []
    split_codes: list[int] = []

    for frame_id in frame_ids:
        payload = _load_projected(projected_dir / f"{frame_id}.npz")
        if str(payload["frame_id"].item()) != frame_id:
            raise ValueError(f"Projected frame identity drift for {frame_id}")
        visible = payload["visible_point_indices"].astype(np.int64, copy=False)
        known = payload["known_point_indices"].astype(np.int64, copy=False)
        offsets = payload["proposal_offsets"].astype(np.int64, copy=False)
        members = payload["proposal_point_indices"].astype(np.int64, copy=False)
        proposal_count = int(offsets.size) - 1

        visible_parts.append(visible)
        known_parts.append(known)
        member_parts.append(members)
        order_parts.append(payload["proposal_order_indices"].astype(np.int32))
        source_parts.append(payload["proposal_source_indices"].astype(np.int32))
        area_parts.append(payload["proposal_areas"].astype(np.int64))
        iou_parts.append(payload["proposal_predicted_ious"].astype(np.float32))
        stability_parts.append(
            payload["proposal_stability_scores"].astype(np.float32)
        )
        trainable_parts.append(payload["proposal_trainable"].astype(np.bool_))
        visible_offsets.append(visible_offsets[-1] + int(visible.size))
        known_offsets.append(known_offsets[-1] + int(known.size))
        proposal_frame_offsets.append(
            proposal_frame_offsets[-1] + proposal_count
        )
        proposal_point_offsets.extend(
            proposal_point_offsets[-1] + int(value) for value in np.diff(offsets)
        )
        split_codes.append(0 if frame_id in train_frames else 1)

    def concatenate(parts: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
        return (
            np.concatenate(parts).astype(dtype, copy=False)
            if parts
            else np.zeros(0, dtype=dtype)
        )

    _atomic_savez(
        output,
        schema_version=np.asarray(PROJECTION_CACHE_SCHEMA),
        scene_id=np.asarray(scene_id),
        teacher_source_key=np.asarray(SOURCE_KEY),
        crop_indices_sha256=np.asarray(crop_indices_sha256),
        physical_frame_ids=np.asarray(frame_ids),
        split_codes=np.asarray(split_codes, dtype=np.int8),
        visible_frame_offsets=np.asarray(visible_offsets, dtype=np.int64),
        visible_point_indices=concatenate(visible_parts, np.dtype(np.int64)),
        frame_offsets=np.asarray(known_offsets, dtype=np.int64),
        eligible_point_indices=concatenate(known_parts, np.dtype(np.int64)),
        proposal_frame_offsets=np.asarray(proposal_frame_offsets, dtype=np.int64),
        proposal_point_offsets=np.asarray(proposal_point_offsets, dtype=np.int64),
        proposal_point_indices=concatenate(member_parts, np.dtype(np.int64)),
        proposal_order_indices=concatenate(order_parts, np.dtype(np.int32)),
        proposal_source_indices=concatenate(source_parts, np.dtype(np.int32)),
        proposal_areas=concatenate(area_parts, np.dtype(np.int64)),
        proposal_predicted_ious=concatenate(iou_parts, np.dtype(np.float32)),
        proposal_stability_scores=concatenate(
            stability_parts, np.dtype(np.float32)
        ),
        proposal_trainable=concatenate(trainable_parts, np.dtype(np.bool_)),
    )
    trainable_by_frame = [
        int(np.count_nonzero(values)) for values in trainable_parts
    ]
    return {
        "physical_frames": len(frame_ids),
        "visible_memberships": int(visible_offsets[-1]),
        "known_memberships": int(known_offsets[-1]),
        "retained_proposals": int(proposal_frame_offsets[-1]),
        "projected_memberships": int(proposal_point_offsets[-1]),
        "trainable_proposals": int(
            sum(
                count
                for frame_id, count in zip(frame_ids, trainable_by_frame)
                if frame_id in train_frames
            )
        ),
        "heldout_trainable_proposals": int(
            sum(
                count
                for frame_id, count in zip(frame_ids, trainable_by_frame)
                if frame_id not in train_frames
            )
        ),
    }


def _reference_artifact_path(
    reference_scene: Path, record: dict[str, Any]
) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else reference_scene / path


def _hardlink_geometry(
    *, reference_scene: Path, pack_scene: Path, reference_manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key in ("points", "colors", "normals", "crop_indices"):
        source_record = dict(reference_manifest["artifacts"][key])
        source = _reference_artifact_path(reference_scene, source_record).resolve(
            strict=True
        )
        destination = pack_scene / source.name
        storage = "hardlink_to_immutable_reference_geometry"
        if destination.exists():
            if destination.stat().st_size != int(source_record["size_bytes"]):
                raise ValueError(f"Existing geometry size drift: {destination}")
        else:
            pack_scene.mkdir(parents=True, exist_ok=True)
            temporary = pack_scene / f".{source.name}.{os.getpid()}.link"
            temporary.unlink(missing_ok=True)
            import errno
            import shutil
            try:
                os.link(source, temporary)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, temporary)
            temporary.replace(destination)
        if not os.path.samefile(source, destination):
            storage = "verified_copy_of_immutable_reference_geometry"
        if _sha256(destination) != source_record["sha256"]:
            raise ValueError(f"Geometry content drift: {destination}")
        output[key] = {
            "path": destination.name,
            "size_bytes": int(source_record["size_bytes"]),
            "sha256": str(source_record["sha256"]),
            "storage": storage,
        }
    return output


def _existing_complete(
    *, pack_manifest: Path, reference_manifest_sha256: str, teacher: dict[str, Any]
) -> bool:
    if not pack_manifest.is_file():
        return False
    try:
        payload = json.loads(pack_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("complete")
        and payload.get("schema_version") == SCENE_SCHEMA
        and payload.get("teacher_source_key") == SOURCE_KEY
        and payload.get("reference", {}).get("source_manifest_sha256")
        == reference_manifest_sha256
        and payload.get("teacher", {}).get("checkpoint", {}).get("sha256")
        == teacher["checkpoint"]["sha256"]
        and payload.get("teacher", {}).get("automatic_mask_generator")
        == teacher["automatic_mask_generator"]
        and payload.get("teacher", {}).get("image_input_policy")
        == teacher["image_input_policy"]
    )


def _scene_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Record non-fatal scene properties without turning them into gates."""
    return {
        "raw_overlap_observed": any(
            int(row["overlap_pixels"]) > 0 for row in rows
        ),
        "empty_teacher_frames": sum(
            int(row["num_raw_proposals"]) == 0 for row in rows
        ),
        "albedo_fallback_frames": sum(
            str(row["input_source"]) == ALBEDO_SOURCE for row in rows
        ),
        "doubly_saturated_frames": sum(
            str(row["input_source"]) == EMPTY_SOURCE for row in rows
        ),
    }


def _export_scene(
    *,
    scene_id: str,
    reader: Structured3DReader,
    reference_root: Path,
    annotation_root: Path,
    pack_root: Path,
    provider: SAM2Provider,
) -> tuple[Path, bool, dict[str, Any]]:
    scene_started = time.time()
    reference_scene = reference_root / scene_id
    reference_manifest_path = reference_scene / "source_manifest.json"
    reference_manifest_sha256 = _sha256(reference_manifest_path)
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    if reference_manifest.get("scene_id") != scene_id:
        raise ValueError(f"Reference scene identity drift: {reference_manifest_path}")
    if reference_manifest.get("crop", {}).get("mode") != "full_scene":
        raise ValueError(f"Reference scene is not full-scene: {reference_manifest_path}")

    teacher = provider.provenance()
    pack_scene = pack_root / scene_id
    pack_manifest_path = pack_scene / "source_manifest.json"
    if _existing_complete(
        pack_manifest=pack_manifest_path,
        reference_manifest_sha256=reference_manifest_sha256,
        teacher=teacher,
    ):
        return pack_manifest_path, True, {
            "scene_id": scene_id,
            "reused": True,
            "seconds": time.time() - scene_started,
        }

    points_path = _reference_artifact_path(
        reference_scene, reference_manifest["artifacts"]["points"]
    )
    points = np.asarray(np.load(points_path, allow_pickle=False), dtype=np.float32)
    point_count = int(points.shape[0])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid reference point shape: {points.shape}")

    adapter = Structured3DArchiveAdapter(scene_id=scene_id, reader=reader)
    frames_by_id = {frame.frame_id: frame for frame in adapter.list_frames()}
    frames_manifest = reference_manifest["frames"]
    frame_ids = [str(value) for value in frames_manifest["physical"]]
    if set(frame_ids) != set(frames_by_id):
        raise RuntimeError(
            f"{scene_id}: archive/reference frame mismatch "
            f"archive={len(frames_by_id)} reference={len(frame_ids)}"
        )
    train_frames = set(str(value) for value in frames_manifest["train"])
    heldout_frames = set(str(value) for value in frames_manifest["heldout"])
    if train_frames | heldout_frames != set(frame_ids) or train_frames & heldout_frames:
        raise ValueError(f"{scene_id}: invalid reference frame split")

    annotation_scene = annotation_root / scene_id
    raw_dir = annotation_scene / "raw_sam2"
    projected_dir = annotation_scene / ".projected_work"
    raw_dir.mkdir(parents=True, exist_ok=True)
    projected_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for frame_index, frame_id in enumerate(frame_ids):
        frame_started = time.perf_counter()
        frame = frames_by_id[frame_id]
        raw_path = raw_dir / f"{frame_id}.npz"
        raw_input_path = raw_dir / f"{frame_id}.input.json"
        projected_path = projected_dir / f"{frame_id}.npz"
        selection = adapter.select_teacher_image(frame)
        metadata_reused = _raw_input_matches(
            path=raw_input_path,
            scene_id=scene_id,
            frame_id=frame_id,
            selection=selection,
            raw_path=raw_path,
        )
        # The v1 run contains rawlight-only proposal files without per-frame
        # input metadata.  They remain valid only when the v2 policy also
        # selects rawlight.  Saturated legacy frames are deliberately
        # regenerated from albedo (or replaced by an explicit empty result).
        legacy_rawlight_reuse = bool(
            raw_path.is_file()
            and not raw_input_path.exists()
            and selection.source == RAWLIGHT_SOURCE
        )
        raw_reused = bool(metadata_reused or legacy_rawlight_reuse)
        raw_changed = False
        inference_seconds = 0.0
        frame_peak_memory = 0
        if raw_reused:
            proposals = load_proposal_set(raw_path)
        else:
            if selection.source == EMPTY_SOURCE:
                proposals = _empty_proposals(selection.image_shape)
            else:
                if selection.image is None:
                    raise RuntimeError(
                        f"{scene_id}/{frame_id}: selected image is missing"
                    )
                proposals, inference_seconds, frame_peak_memory = provider.generate(
                    selection.image
                )
            save_proposal_set(raw_path, proposals)
            proposals = load_proposal_set(raw_path)
            raw_changed = True

        raw_input_payload = _raw_input_payload(
            scene_id=scene_id,
            frame_id=frame_id,
            selection=selection,
            raw_path=raw_path,
        )
        if not metadata_reused:
            _atomic_json(raw_input_path, raw_input_payload)

        projection_reused = bool(projected_path.is_file() and not raw_changed)
        if projection_reused:
            projected = _load_projected(projected_path)
            if str(projected["frame_id"].item()) != frame_id:
                raise ValueError(f"Projected frame identity drift: {projected_path}")
            supports = np.diff(projected["proposal_offsets"].astype(np.int64))
            visible_size = int(projected["visible_point_indices"].size)
            known_size = int(projected["known_point_indices"].size)
            projected_stats = {
                "visible_points": visible_size,
                "known_points": known_size,
                "retained_projected_proposals": int(supports.size),
                "trainable_projected_proposals": int(
                    np.count_nonzero(projected["proposal_trainable"])
                ),
                "projected_memberships": int(
                    projected["proposal_point_indices"].size
                ),
                "projected_multiple_membership_points": int(
                    projected["projected_multiple_membership_points"].item()
                ),
                "projected_max_memberships_per_point": int(
                    projected["projected_max_memberships_per_point"].item()
                ),
                "projected_support_min": (
                    int(supports.min()) if supports.size else 0
                ),
                "projected_support_median": (
                    float(np.median(supports)) if supports.size else 0.0
                ),
                "projected_support_max": (
                    int(supports.max()) if supports.size else 0
                ),
            }
        else:
            visible, visible_u, visible_v = _project_geometry(
                adapter=adapter, frame=frame, points=points
            )
            projected_stats = _save_projected_frame(
                path=projected_path,
                frame_id=frame_id,
                proposals=proposals,
                visible=visible,
                visible_u=visible_u,
                visible_v=visible_v,
            )

        overlap = proposal_overlap_statistics(proposals)
        row = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "split": "train" if frame_id in train_frames else "heldout",
            "image_shape": list(proposals.image_shape),
            "input_source": selection.source,
            "input_selection_reason": selection.reason,
            "teacher_input_member": selection.selected_member,
            "teacher_input_sha256": selection.selected_sha256,
            "rawlight_saturation_fraction": (
                selection.rawlight_saturation_fraction
            ),
            "albedo_saturation_fraction": (
                selection.albedo_saturation_fraction
            ),
            "raw_reused": raw_reused,
            "input_metadata_reused": metadata_reused,
            "projection_reused": projection_reused,
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_bytes": frame_peak_memory,
            "frame_total_seconds": time.perf_counter() - frame_started,
            "raw_file": str(raw_path.relative_to(annotation_scene)),
            "raw_artifact": _file_record(
                raw_path, relative_to=annotation_scene
            ),
            "input_metadata_file": str(
                raw_input_path.relative_to(annotation_scene)
            ),
            "input_metadata_artifact": _file_record(
                raw_input_path, relative_to=annotation_scene
            ),
            **overlap,
            **projected_stats,
        }
        rows.append(row)
        log.info(
            "%s [%d/%d] %s input=%s raw=%d projected=%d visible=%d "
            "known=%d %.2fs",
            scene_id,
            frame_index + 1,
            len(frame_ids),
            frame_id,
            selection.source,
            proposals.num_proposals,
            int(projected_stats["retained_projected_proposals"]),
            int(projected_stats["visible_points"]),
            int(projected_stats["known_points"]),
            float(row["frame_total_seconds"]),
        )

    geometry = _hardlink_geometry(
        reference_scene=reference_scene,
        pack_scene=pack_scene,
        reference_manifest=reference_manifest,
    )
    cache_path = pack_scene / "sam2_projection_cache.npz"
    cache_counts = _compact_projection_cache(
        output=cache_path,
        scene_id=scene_id,
        frame_ids=frame_ids,
        train_frames=train_frames,
        projected_dir=projected_dir,
        crop_indices_sha256=str(
            reference_manifest["artifacts"]["crop_indices"]["sha256"]
        ),
    )
    annotation_manifest_path = annotation_scene / "annotation_manifest.json"
    annotation_manifest = {
        "schema_version": ANNOTATION_SCHEMA,
        "complete": True,
        "scene_id": scene_id,
        "teacher_source_key": SOURCE_KEY,
        "teacher": teacher,
        "reference": {
            "source_manifest": str(reference_manifest_path.resolve()),
            "source_manifest_sha256": reference_manifest_sha256,
            "point_count": point_count,
            "selection_mode": "full_scene",
        },
        "projection": {
            "camera_contract": "structured3d_prepare_v2_axis_pose",
            "minimum_positive_points": MIN_PROJECTED_POINTS,
            "overlap_preserved": True,
            "cross_view_mask_identity_inferred": False,
            "unmasked_points_are_unknown": True,
            "zero_proposal_frames_are_valid": True,
            "visibility": {
                "min_depth_m": 0.01,
                "z_tolerance_m": 1.0,
                "depth_aligned_to_rgb": True,
            },
        },
        "frames": rows,
        "summary": {
            "physical_frames": len(rows),
            "raw_accepted_masks": int(
                sum(int(row["num_raw_proposals"]) for row in rows)
            ),
            "retained_projected_masks": int(
                sum(int(row["retained_projected_proposals"]) for row in rows)
            ),
            "visible_memberships": int(
                sum(int(row["visible_points"]) for row in rows)
            ),
            "known_memberships": int(
                sum(int(row["known_points"]) for row in rows)
            ),
            "input_source_counts": {
                source: sum(str(row["input_source"]) == source for row in rows)
                for source in (RAWLIGHT_SOURCE, ALBEDO_SOURCE, EMPTY_SOURCE)
            },
            "empty_teacher_frames": int(
                sum(int(row["num_raw_proposals"]) == 0 for row in rows)
            ),
        },
    }
    _atomic_json(annotation_manifest_path, annotation_manifest)

    artifacts = dict(geometry)
    artifacts["sam2_projection_cache"] = _file_record(
        cache_path, relative_to=pack_scene
    )
    artifacts["annotation_manifest"] = _file_record(annotation_manifest_path)
    train_proposals = int(
        sum(
            int(row["trainable_projected_proposals"])
            for row in rows
            if row["split"] == "train"
        )
    )
    heldout_proposals = int(
        sum(
            int(row["trainable_projected_proposals"])
            for row in rows
            if row["split"] == "heldout"
        )
    )
    if (
        cache_counts["trainable_proposals"] != train_proposals
        or cache_counts["heldout_trainable_proposals"] != heldout_proposals
    ):
        raise RuntimeError(f"{scene_id}: compact-cache proposal-count drift")
    pack_manifest = {
        "schema_version": SCENE_SCHEMA,
        "complete": True,
        "scene_id": scene_id,
        "dataset": "structured3d",
        "teacher_source_key": SOURCE_KEY,
        "source_cells": [SOURCE_KEY],
        "single_granularity": True,
        "teacher": teacher,
        "crop": {
            "mode": "full_scene",
            "num_points": point_count,
            "indices": "crop_indices.npy",
            "selection_is_teacher_independent": True,
        },
        "frames": {
            "selection_mode": "all_complete_valid_pose",
            "physical": frame_ids,
            "train": [value for value in frame_ids if value in train_frames],
            "heldout": [value for value in frame_ids if value in heldout_frames],
        },
        "artifacts": artifacts,
        "counts": {
            "trainable_proposals": {SOURCE_KEY: train_proposals},
            "heldout_trainable_proposals": {SOURCE_KEY: heldout_proposals},
            "projection_cache": cache_counts,
        },
        "semantics": {
            "overlap_preserved": True,
            "frame_local": True,
            "cross_view_mask_identity_inferred": False,
            "unseen_points_are_unknown": True,
            "visible_but_unmasked_points_are_unknown": True,
            "zero_proposal_frames_contribute_no_supervision": True,
            "image_input_policy": dict(IMAGE_INPUT_POLICY),
            "negative_policy": (
                "derive only from known proposal members with no shared "
                "frame-local proposal membership"
            ),
        },
        "reference": {
            "source_manifest": str(reference_manifest_path.resolve()),
            "source_manifest_sha256": reference_manifest_sha256,
        },
        "checks": {
            "full_scene_geometry": True,
            "single_sam2_source": True,
            "all_frames_have_raw_results": (
                len(rows) == len(frame_ids)
                and all(bool(row["raw_artifact"]) for row in rows)
            ),
            "all_frames_have_input_provenance": (
                len(rows) == len(frame_ids)
                and all(bool(row["input_metadata_artifact"]) for row in rows)
            ),
            "all_frames_have_projection_results": (
                cache_counts["physical_frames"] == len(frame_ids)
            ),
            "empty_frames_are_unknown": all(
                int(row["num_raw_proposals"]) > 0
                or (
                    int(row["known_points"]) == 0
                    and int(row["retained_projected_proposals"]) == 0
                    and int(row["trainable_projected_proposals"]) == 0
                )
                for row in rows
            ),
            "geometry_hashes_inherited_from_reference": True,
        },
        # Some simple scenes legitimately have no overlapping accepted masks.
        # The contract is to preserve overlap when present, not to require it.
        "observations": _scene_observations(rows),
        "runtime": {
            "device_name": provider.device_name,
            "model_load_seconds": provider.model_load_seconds,
            "peak_cuda_memory_bytes": provider.peak_memory_bytes,
            "scene_seconds": time.time() - scene_started,
        },
    }
    if not all(bool(value) for value in pack_manifest["checks"].values()):
        raise RuntimeError(f"{scene_id}: pack checks failed: {pack_manifest['checks']}")
    _atomic_json(pack_manifest_path, pack_manifest)

    # Per-frame projections are restart state only.  The committed cache is the
    # canonical projection artifact; raw 2D masks remain for future audits.
    shutil.rmtree(projected_dir)
    return pack_manifest_path, False, {
        "scene_id": scene_id,
        "reused": False,
        "seconds": time.time() - scene_started,
        "frames": len(rows),
        "raw_masks": int(annotation_manifest["summary"]["raw_accepted_masks"]),
        "projected_masks": int(
            annotation_manifest["summary"]["retained_projected_masks"]
        ),
        "empty_teacher_frames": int(
            annotation_manifest["summary"]["empty_teacher_frames"]
        ),
        "albedo_fallback_frames": int(
            annotation_manifest["summary"]["input_source_counts"][ALBEDO_SOURCE]
        ),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-ids-file", type=Path, required=True)
    parser.add_argument("--raw-zips-dir", type=Path, required=True)
    parser.add_argument("--reference-source-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker-label", required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.points_per_side != 32:
        raise ValueError("Production SAM 2 export is frozen at points_per_side=32")
    scene_ids = _load_scene_ids(args.scene_ids_file.resolve(strict=True))
    reference_root = args.reference_source_root.resolve(strict=True)
    raw_zips_dir = args.raw_zips_dir.resolve(strict=True)
    annotation_root = args.annotation_root.resolve()
    pack_root = args.pack_root.resolve()
    annotation_root.mkdir(parents=True, exist_ok=True)
    pack_root.mkdir(parents=True, exist_ok=True)

    archives = {
        path.stem.rsplit("_", 1)[-1]: path
        for path in raw_zips_dir.glob("Structured3D_perspective_full_*.zip")
    }
    required_codes = {_archive_code_for_scene(scene_id) for scene_id in scene_ids}
    missing_codes = sorted(required_codes - set(archives))
    if missing_codes:
        raise FileNotFoundError(f"Missing Structured3D archive codes: {missing_codes}")

    groups: OrderedDict[str, list[str]] = OrderedDict()
    for scene_id in scene_ids:
        groups.setdefault(_archive_code_for_scene(scene_id), []).append(scene_id)

    started = time.time()
    provider = SAM2Provider(args)
    records: list[dict[str, Any]] = []
    for code, grouped_scene_ids in groups.items():
        archive = archives[code]
        log.info("Opening %s for %d scene(s)", archive.name, len(grouped_scene_ids))
        with Structured3DReader(archive) as reader:
            for scene_id in grouped_scene_ids:
                manifest, reused, stats = _export_scene(
                    scene_id=scene_id,
                    reader=reader,
                    reference_root=reference_root,
                    annotation_root=annotation_root,
                    pack_root=pack_root,
                    provider=provider,
                )
                records.append(
                    {
                        **stats,
                        "manifest": str(manifest.resolve()),
                        "manifest_sha256": _sha256(manifest),
                        "reused": reused,
                    }
                )

    report = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "complete": len(records) == len(scene_ids),
        "worker_label": str(args.worker_label),
        "scene_ids_file": _file_record(args.scene_ids_file),
        "scene_count": len(scene_ids),
        "wall_seconds": time.time() - started,
        "device_name": provider.device_name,
        "model_load_seconds": provider.model_load_seconds,
        "peak_cuda_memory_bytes": provider.peak_memory_bytes,
        "scenes": records,
    }
    _atomic_json(args.report.resolve(), report)
    print(args.report.resolve())


if __name__ == "__main__":
    main()
