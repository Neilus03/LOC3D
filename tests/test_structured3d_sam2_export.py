from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from delimit3d.cli.export_structured3d_sam2 import (
    ALBEDO_SOURCE,
    EMPTY_SOURCE,
    PROJECTION_CACHE_SCHEMA,
    RAWLIGHT_SOURCE,
    SOURCE_KEY,
    TeacherImageSelection,
    _archive_code_for_scene,
    _choose_input_source,
    _compact_projection_cache,
    _empty_proposals,
    _load_projected,
    _raw_input_matches,
    _raw_input_payload,
    _save_projected_frame,
    _scene_observations,
)
from delimit3d.cli.finalize_structured3d_sam2 import _validate_cache
from delimit3d.data.unsam_proposals import (
    UnSAMProposalSet,
    load_proposal_set,
    save_proposal_set,
)


def _proposals() -> UnSAMProposalSet:
    masks = np.zeros((3, 2, 3), dtype=bool)
    masks[0, 0, :2] = True
    masks[1, 0, 1:] = True
    masks[2, 1, 0] = True  # filtered after projection: one point
    areas = masks.reshape(3, -1).sum(axis=1, dtype=np.int64)
    return UnSAMProposalSet(
        masks=masks,
        areas=areas,
        reported_areas=areas.astype(np.float32),
        predicted_ious=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        stability_scores=np.asarray([0.99, 0.98, 0.97], dtype=np.float32),
        bboxes=np.zeros((3, 4), dtype=np.float32),
        source_indices=np.asarray([4, 8, 12], dtype=np.int32),
    )


def test_projected_frame_preserves_overlap_and_unknowns(tmp_path: Path) -> None:
    output = tmp_path / "0:0.npz"
    visible = np.asarray([10, 11, 12, 13], dtype=np.int64)
    stats = _save_projected_frame(
        path=output,
        frame_id="0:0",
        proposals=_proposals(),
        visible=visible,
        visible_u=np.asarray([0, 1, 2, 2], dtype=np.int32),
        visible_v=np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    payload = _load_projected(output)

    assert stats["retained_projected_proposals"] == 2
    assert stats["known_points"] == 3
    assert stats["trainable_projected_proposals"] == 2
    assert payload["known_point_indices"].tolist() == [10, 11, 12]
    assert payload["visible_point_indices"].tolist() == [10, 11, 12, 13]
    assert payload["proposal_offsets"].tolist() == [0, 2, 4]
    assert payload["proposal_point_indices"].tolist() == [10, 11, 11, 12]
    assert payload["proposal_source_indices"].tolist() == [4, 8]
    assert payload["proposal_trainable"].tolist() == [True, True]


def test_compact_cache_keeps_visible_and_known_universes_separate(
    tmp_path: Path,
) -> None:
    projected = tmp_path / "projected"
    for frame_id in ("0:0", "0:1"):
        _save_projected_frame(
            path=projected / f"{frame_id}.npz",
            frame_id=frame_id,
            proposals=_proposals(),
            visible=np.asarray([0, 1, 2, 3], dtype=np.int64),
            visible_u=np.asarray([0, 1, 2, 2], dtype=np.int32),
            visible_v=np.asarray([0, 0, 0, 1], dtype=np.int32),
        )
    output = tmp_path / "cache.npz"
    counts = _compact_projection_cache(
        output=output,
        scene_id="scene_00000",
        frame_ids=("0:0", "0:1"),
        train_frames={"0:0"},
        projected_dir=projected,
        crop_indices_sha256="a" * 64,
    )
    with np.load(output, allow_pickle=False) as cache:
        assert str(cache["schema_version"].item()) == PROJECTION_CACHE_SCHEMA
        assert str(cache["teacher_source_key"].item()) == SOURCE_KEY
        assert cache["visible_frame_offsets"].tolist() == [0, 4, 8]
        assert cache["frame_offsets"].tolist() == [0, 3, 6]
        assert cache["proposal_frame_offsets"].tolist() == [0, 2, 4]
        assert cache["proposal_point_offsets"].tolist() == [0, 2, 4, 6, 8]
        assert cache["split_codes"].tolist() == [0, 1]
        assert cache["proposal_trainable"].tolist() == [True] * 4
    assert counts == {
        "physical_frames": 2,
        "visible_memberships": 8,
        "known_memberships": 6,
        "retained_proposals": 4,
        "projected_memberships": 8,
        "trainable_proposals": 2,
        "heldout_trainable_proposals": 2,
    }


def test_empty_teacher_result_roundtrips_and_projects_as_unknown(
    tmp_path: Path,
) -> None:
    proposals = _empty_proposals((2, 3))
    raw = tmp_path / "empty.npz"
    save_proposal_set(raw, proposals)
    loaded = load_proposal_set(raw)

    assert loaded.masks.shape == (0, 2, 3)
    assert loaded.areas.shape == (0,)
    output = tmp_path / "projected" / "0:0.npz"
    stats = _save_projected_frame(
        path=output,
        frame_id="0:0",
        proposals=loaded,
        visible=np.asarray([10, 11, 12], dtype=np.int64),
        visible_u=np.asarray([0, 1, 2], dtype=np.int32),
        visible_v=np.asarray([0, 0, 1], dtype=np.int32),
    )
    payload = _load_projected(output)

    assert stats["visible_points"] == 3
    assert stats["known_points"] == 0
    assert stats["retained_projected_proposals"] == 0
    assert stats["trainable_projected_proposals"] == 0
    assert payload["visible_point_indices"].tolist() == [10, 11, 12]
    assert payload["known_point_indices"].size == 0
    assert payload["proposal_offsets"].tolist() == [0]
    assert payload["proposal_point_indices"].size == 0
    assert payload["proposal_trainable"].size == 0


def test_compact_cache_supports_mixed_empty_and_nonempty_frames(
    tmp_path: Path,
) -> None:
    projected = tmp_path / "projected"
    visible = np.asarray([0, 1, 2, 3], dtype=np.int64)
    visible_u = np.asarray([0, 1, 2, 2], dtype=np.int32)
    visible_v = np.asarray([0, 0, 0, 1], dtype=np.int32)
    _save_projected_frame(
        path=projected / "0:0.npz",
        frame_id="0:0",
        proposals=_empty_proposals((2, 3)),
        visible=visible,
        visible_u=visible_u,
        visible_v=visible_v,
    )
    _save_projected_frame(
        path=projected / "0:1.npz",
        frame_id="0:1",
        proposals=_proposals(),
        visible=visible,
        visible_u=visible_u,
        visible_v=visible_v,
    )
    output = tmp_path / "cache.npz"
    counts = _compact_projection_cache(
        output=output,
        scene_id="scene_00000",
        frame_ids=("0:0", "0:1"),
        train_frames={"0:1"},
        projected_dir=projected,
        crop_indices_sha256="b" * 64,
    )
    with np.load(output, allow_pickle=False) as cache:
        assert cache["visible_frame_offsets"].tolist() == [0, 4, 8]
        assert cache["frame_offsets"].tolist() == [0, 0, 3]
        assert cache["proposal_frame_offsets"].tolist() == [0, 0, 2]
        assert cache["proposal_point_offsets"].tolist() == [0, 2, 4]
    assert counts["physical_frames"] == 2
    assert counts["retained_proposals"] == 2
    assert counts["trainable_proposals"] == 2
    assert counts["heldout_trainable_proposals"] == 0
    assert _validate_cache(
        output,
        scene_id="scene_00000",
        point_count=4,
        physical_frames=["0:0", "0:1"],
    ) == counts


def test_saturation_policy_is_fixed_and_independent_of_mask_output() -> None:
    assert _choose_input_source(0.989, None) == RAWLIGHT_SOURCE
    assert _choose_input_source(0.99, None) == EMPTY_SOURCE
    assert _choose_input_source(0.99, 0.989) == ALBEDO_SOURCE
    assert _choose_input_source(0.99, 0.99) == EMPTY_SOURCE


def test_raw_input_metadata_binds_selection_and_raw_artifact(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "0:0.npz"
    save_proposal_set(raw, _empty_proposals((2, 3)))
    selection = TeacherImageSelection(
        source=EMPTY_SOURCE,
        reason="rawlight_and_albedo_saturated",
        image=None,
        image_shape=(2, 3),
        selected_member=None,
        selected_sha256=None,
        rawlight_member="raw.png",
        rawlight_sha256="a" * 64,
        rawlight_saturation_fraction=1.0,
        albedo_member="albedo.png",
        albedo_sha256="b" * 64,
        albedo_saturation_fraction=1.0,
    )
    metadata = tmp_path / "0:0.input.json"
    metadata.write_text(
        json.dumps(
            _raw_input_payload(
                scene_id="scene_00000",
                frame_id="0:0",
                selection=selection,
                raw_path=raw,
            )
        )
    )
    assert _raw_input_matches(
        path=metadata,
        scene_id="scene_00000",
        frame_id="0:0",
        selection=selection,
        raw_path=raw,
    )
    metadata.write_text(
        metadata.read_text().replace('"complete": true', '"complete": false')
    )
    assert not _raw_input_matches(
        path=metadata,
        scene_id="scene_00000",
        frame_id="0:0",
        selection=selection,
        raw_path=raw,
    )


def test_archive_code_mapping_skips_unreleased_code_nine() -> None:
    assert _archive_code_for_scene("scene_00000") == "00"
    assert _archive_code_for_scene("scene_01000") == "05"
    assert _archive_code_for_scene("scene_01800") == "08"
    assert _archive_code_for_scene("scene_02000") == "10"
    assert _archive_code_for_scene("scene_03499") == "17"


def test_scene_overlap_is_an_observation_not_a_required_condition() -> None:
    empty = {
        "overlap_pixels": 0,
        "num_raw_proposals": 0,
        "input_source": EMPTY_SOURCE,
    }
    assert _scene_observations([empty]) == {
        "raw_overlap_observed": False,
        "empty_teacher_frames": 1,
        "albedo_fallback_frames": 0,
        "doubly_saturated_frames": 1,
    }
    assert _scene_observations(
        [
            empty,
            {
                "overlap_pixels": 4,
                "num_raw_proposals": 2,
                "input_source": ALBEDO_SOURCE,
            },
        ]
    ) == {
        "raw_overlap_observed": True,
        "empty_teacher_frames": 1,
        "albedo_fallback_frames": 1,
        "doubly_saturated_frames": 1,
    }
