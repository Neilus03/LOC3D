import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    derive_litept_hierarchy_ancestry,
    sample_multigranular_frame_group_plan,
    source_cell_quotas,
)
from delimit3d.models.litept_wrapper import LitePTBackboneOutput
from delimit3d.training import adaptation
from delimit3d.training.contracts import EXPERIMENT_ID, TWO_D_CELLS, source_cells_for_family
from delimit3d.training.runner import (
    _native_token_pure_fixed_pack_cosine_gap,
    _strict_initial_load,
    parse_args,
)


PRESET = Path(__file__).resolve().parents[1] / "configs/reproduce/unsamv2_g05_dec0_336ep_original.json"
REQUIRED = [
    "--train-split", "/unused/train.txt", "--holdout-split", "/unused/heldout.txt",
    "--source-root", "/unused/source", "--initial-checkpoint", "/unused/init.pt",
    "--litept-root", "/unused/LitePT", "--output-dir", "/unused/new-run",
]


def _scene(granularity="g05", count=128):
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 0] = np.arange(count, dtype=np.float32) * 0.06
    frame = adaptation.RawFrame(
        granularity=granularity, physical_frame_id="frame", split="heldout",
        visible_indices=np.arange(count), proposal_offsets=np.arange(0, count + 1, 4),
        proposal_point_indices=np.arange(count),
    )
    return adaptation.SceneSource(
        scene_id="scene_00000", manifest_path=Path("/synthetic/source_manifest.json"),
        manifest_sha256="synthetic", points=points,
        features=np.random.default_rng(0).normal(size=(count, 6)).astype(np.float32),
        labels_by_key={}, raw_frames=(frame,),
    )


def _plan(scene, *, epoch=0, states=None, hierarchy=False, quota=16, grid_size=0.05):
    frame = scene.raw_frames[0]
    cell = f"2d/{frame.granularity}"
    catalog = FrameProposalCatalog(
        cell=cell, frame_id=frame.physical_frame_id,
        visible_indices=torch.as_tensor(frame.visible_indices),
        proposal_offsets=torch.as_tensor(frame.proposal_offsets),
        proposal_point_indices=torch.as_tensor(frame.proposal_point_indices),
    )
    return sample_multigranular_frame_group_plan(
        scene_id=scene.scene_id, epoch=epoch, seed=42, points=torch.from_numpy(scene.points),
        frames_by_cell={cell: (catalog,)}, cells=(cell,), total_proposal_quota=quota,
        positive_pairs_per_proposal=3, num_uniform_negatives=2,
        num_spatial_hard_negatives=0, num_feature_hard_negatives=0,
        require_negative_proposal_membership=True, require_hierarchy_routes=hierarchy,
        grid_size=grid_size, coverage_states=states,
    )


def test_preset_and_legacy_cli_contracts():
    args = parse_args(["--config", str(PRESET), *REQUIRED])
    assert args.source_granularity == "g05"
    assert args.hierarchy_supervision == "off" and not args.multiscale_supervision
    assert args.grid_size == 0.05 and args.epochs == 336
    assert source_cells_for_family(args.source_family, granularity=args.source_granularity) == ("2d/g05",)
    assert parse_args(["--config", str(PRESET), *REQUIRED, "--source-granularity", "g08"]).source_granularity == "g08"
    old = parse_args([*REQUIRED, "--epochs", "336", "--backbone-lr", "0.0001"])
    assert old.source_granularity is None and old.grid_size == 0.02
    assert source_cells_for_family("2d") == TWO_D_CELLS
    with pytest.raises(SystemExit):
        parse_args(["--config", str(PRESET), *REQUIRED, "--multiscale-supervision"])
    with pytest.raises(SystemExit):
        parse_args(["--config", str(PRESET), *REQUIRED, "--hierarchy-supervision", "query_mask"])


@pytest.mark.parametrize("override", [
    {"source_granularty": "g05"}, {"source_granularity": "sam2"},
    {"multiscale_supervision": "false"}, {"grid_size": 0},
    {"color_std": [1, 1]},
])
def test_invalid_preset_fails_before_distributed_setup(tmp_path, override):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({**json.loads(PRESET.read_text()), **override}))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), *REQUIRED])


@pytest.mark.parametrize("granularity", ["g02", "g05", "g08"])
def test_one_cell_gets_full_quota_and_resumes_coverage(granularity):
    scene = _scene(granularity)
    cell = f"2d/{granularity}"
    plan = _plan(scene)
    assert plan.quota_by_cell == {cell: 16}
    assert sum(len(group.selected_proposal_indices) for group in plan.groups) == 16
    state = {cell: DeterministicCoverageState.from_dict(plan.state_after_by_cell()[cell]).for_epoch(1)}
    resumed = _plan(scene, epoch=1, states=state)
    replay = _plan(scene, epoch=1, states=state)
    first = {index for group in plan.groups for index in group.selected_proposal_indices}
    second = {index for group in resumed.groups for index in group.selected_proposal_indices}
    assert len(first | second) == 32 and not first & second
    assert resumed.state_after_by_cell() == replay.state_after_by_cell()
    assert source_cell_quotas(0) == {"2d/g02": 6, "2d/g05": 5, "2d/g08": 5}


def test_dec0_does_not_require_representable_coarse_masks():
    scene = _scene("g08", count=8)
    assert len(_plan(scene, quota=2).groups) == 1
    with pytest.raises(NoRetainedFrameGroupsError):
        _plan(scene, quota=2, hierarchy=True)
    # The configured grid also governs sampler admission, not just the backbone.
    with pytest.raises(NoRetainedFrameGroupsError):
        _plan(scene, quota=2, grid_size=1.0)


class _CPUBackbone(torch.nn.Module):
    """Real voxel mapping with a small differentiable stand-in for CUDA LitePT."""

    def __init__(self, **kwargs):
        super().__init__()
        self.grid_size = kwargs["grid_size"]
        self.capture_hierarchy = kwargs["capture_hierarchy"]
        self.projection = torch.nn.Linear(6, 72)

    def forward(self, points, features):
        ancestry = derive_litept_hierarchy_ancestry(points, grid_size=self.grid_size)
        inverse = ancestry.point_to_stage["dec0"]
        representatives = ancestry.representative_indices
        tokens = self.projection(features[representatives])
        return LitePTBackboneOutput(
            point_feat=tokens[inverse], point_xyz=points, scene_tokens=tokens,
            scene_xyz=points[representatives], inverse_map=inverse,
            representative_indices=representatives,
        )


def test_single_source_forward_backward_and_fixed_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr("delimit3d.pretraining.model.LitePTBackbone", _CPUBackbone)
    scene = _scene()
    plan = _plan(scene)
    model = adaptation.LitePTContrastiveModel(
        litept_root=tmp_path, token_pure_dec0=True, token_pure_cells=("2d/g05",),
        grid_size=0.05,
    )
    result = model(torch.from_numpy(scene.points), torch.from_numpy(scene.features), frame_groups=plan.groups)
    assert result["token_pure_dec0"]["optimization_valid"]
    assert result["token_pure_dec0"]["expected_cells"] == ["2d/g05"]
    assert result["loss_multiscale"] == 0
    assert model.hierarchy_mask_supervisor is None and not model.multiscale_projectors
    assert not model.backbone.capture_hierarchy
    assert {name.split(".")[0] for name in model.state_dict()} == {"backbone", "projector", "criterion"}
    result["loss_total"].backward()
    assert model.backbone.projection.weight.grad.norm() > 0
    assert all(parameter.grad is not None for parameter in model.projector.parameters())

    initial = tmp_path / "wrong_grid.pt"
    torch.save({
        "schema_version": "litept_gt3_unsam3_initial/v1", "experiment_id": EXPERIMENT_ID,
        "architecture": {"voxel_reduce": "representative", "grid_size": 0.02},
    }, initial)
    with pytest.raises(ValueError, match="grid size differs"):
        _strict_initial_load(model=model, path=initial, input_features="rgbn6")

    sampling = adaptation.SamplingConfig(
        proposals_per_forward=4, positive_pairs_per_proposal=3, uniform_negatives=2,
        spatial_hard_negatives=0, feature_hard_negatives=0, feature_candidate_pool=2,
    )
    kwargs = dict(pack_dir=tmp_path / "fixed", scenes=[scene], sampling=sampling,
                  sampling_profile="test", seed=42, cells=("2d/g05",), grid_size=0.05)
    report = adaptation.prepare_fixed_holdout_eval_pack(**kwargs)
    assert report["passed"] and report["record_count"] == 1
    assert report["token_pure_dec0_fixed_pack"]["source_cells"] == ["2d/g05"]
    assert adaptation.validate_fixed_holdout_eval_pack(**kwargs)["passed"]
    with pytest.raises(ValueError, match="source-cell drift"):
        adaptation.validate_fixed_holdout_eval_pack(**{**kwargs, "cells": TWO_D_CELLS})
    with pytest.raises(ValueError, match="grid-size drift"):
        adaptation.validate_fixed_holdout_eval_pack(**{**kwargs, "grid_size": 0.02})
    evaluation = adaptation.evaluate_holdout(
        model=model, scenes=[scene], device=torch.device("cpu"), sampling=sampling, seed=42,
        fixed_eval_pack=kwargs["pack_dir"], fixed_eval_pack_report=report, cells=("2d/g05",),
    )
    assert np.isfinite(_native_token_pure_fixed_pack_cosine_gap(evaluation, expected_cells=("2d/g05",)))
    with pytest.raises(ValueError, match="lacks exact valid"):
        _native_token_pure_fixed_pack_cosine_gap(evaluation)

    # An unused coarse mask covers the whole frame. It must not enter the
    # relation graph and erase the selected source's safe negatives.
    poison = replace(scene.raw_frames[0], granularity="g08",
                     proposal_offsets=np.asarray([0, len(scene.points)]))
    other = replace(scene, raw_frames=scene.raw_frames + (poison,))
    assert adaptation.validate_fixed_holdout_eval_pack(**{**kwargs, "scenes": [other]})["passed"]
