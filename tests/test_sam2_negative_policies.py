"""Both SAM2 policies must survive sampling, fixed evaluation and recovery."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState, FrameProposalCatalog,
    derive_litept_hierarchy_ancestry, sample_multigranular_frame_group_plan,
)
from delimit3d.data.partfield_contrastive import (
    concatenate_triplet_batches, negative_policy_name,
    sample_safe_overlapping_proposal_triplets,
)
from delimit3d.losses.litept_hierarchy_supervision import build_token_pure_dec0_contrastive_batch
from delimit3d.models.litept_wrapper import LitePTBackboneOutput
from delimit3d.pretraining.arguments import parse_args
from delimit3d.pretraining.checkpoint import _checkpoint_payload, validate_negative_policy_resume
from delimit3d.pretraining.fixed_eval import _triplet_batch_from_record, _triplet_batch_record
from delimit3d.pretraining.validation import _summarize_v2_sampling
from delimit3d.training import adaptation
from test_single_granularity_pretraining import REQUIRED, _scene

ROOT = Path(__file__).resolve().parents[1]
CELL = "2d/sam2"


def recipe(flag):
    return ROOT / "configs/pretrain" / (
        "sam2_selected_proposal_negatives.yaml" if flag else "sam2_overlap_veto.yaml"
    )


def raw_batch(flag, *, deferred=False):
    # 0/1 are positive. 2/3 share a broader mask. 4 is unrelated but known.
    # 5 is visible and unmasked, so neither policy may use it as a negative.
    return sample_safe_overlapping_proposal_triplets(
        num_scene_points=6, visible_indices=torch.arange(6),
        proposal_offsets=torch.tensor([0, 2, 6, 7]),
        proposal_point_indices=torch.tensor([0, 1, 0, 1, 2, 3, 4]),
        selected_proposal_indices=torch.tensor([0]),
        positive_pairs_per_proposal=1, num_uniform_negatives=4,
        require_negative_proposal_membership=True,
        allow_shared_proposal_negatives=flag, defer_token_loss=deferred,
        generator=torch.Generator().manual_seed(42),
    )


def nested_scene():
    scene = _scene("sam2")
    frame = scene.raw_frames[0]
    return replace(scene, raw_frames=(replace(frame,
        proposal_offsets=np.append(frame.proposal_offsets, len(frame.proposal_point_indices) + 64),
        proposal_point_indices=np.concatenate((frame.proposal_point_indices, np.arange(64))),
    ),))


def plan(scene, flag, *, epoch=0, states=None):
    frame = scene.raw_frames[0]
    catalog = FrameProposalCatalog(cell=CELL, frame_id=frame.physical_frame_id,
        visible_indices=torch.as_tensor(frame.visible_indices),
        proposal_offsets=torch.as_tensor(frame.proposal_offsets),
        proposal_point_indices=torch.as_tensor(frame.proposal_point_indices))
    return sample_multigranular_frame_group_plan(
        scene_id=scene.scene_id, epoch=epoch, seed=42, points=torch.from_numpy(scene.points),
        frames_by_cell={CELL: (catalog,)}, cells=(CELL,), total_proposal_quota=4,
        positive_pairs_per_proposal=3, num_uniform_negatives=2,
        num_spatial_hard_negatives=0, num_feature_hard_negatives=0,
        require_negative_proposal_membership=True, allow_shared_proposal_negatives=flag,
        require_hierarchy_routes=False, grid_size=.02, coverage_states=states,
    )


class CPUBackbone(torch.nn.Module):
    """Differentiable CPU stand-in using the real native voxel mapping."""
    def __init__(self, **kwargs):
        super().__init__()
        self.grid_size = kwargs["grid_size"]
        self.capture_hierarchy = kwargs["capture_hierarchy"]
        self.model = torch.nn.Linear(6, 72)

    def forward(self, points, features):
        ancestry = derive_litept_hierarchy_ancestry(points, grid_size=self.grid_size)
        inverse, representatives = ancestry.point_to_stage["dec0"], ancestry.representative_indices
        tokens = self.model(features[representatives])
        return LitePTBackboneOutput(point_feat=tokens[inverse], point_xyz=points,
            scene_tokens=tokens, scene_xyz=points[representatives], inverse_map=inverse,
            representative_indices=representatives)


@pytest.mark.parametrize("flag", [False, True])
def test_recipes_and_cli_overrides(flag):
    args = parse_args(["--config", str(recipe(flag)), *REQUIRED,
        "--fixed-eval-pack", "/unused/fixed", "--source-audit", "/unused/audit",
        "--code-manifest", "/unused/code"])
    assert args.allow_shared_proposal_negatives is flag
    assert args.source_granularity == "sam2" and args.hierarchy_supervision == "off"
    override = "--no-allow-shared-proposal-negatives" if flag else "--allow-shared-proposal-negatives"
    assert parse_args(["--config", str(recipe(flag)), *REQUIRED,
        "--fixed-eval-pack", "/unused/fixed", "--source-audit", "/unused/audit",
        "--code-manifest", "/unused/code", override]).allow_shared_proposal_negatives is not flag
    with pytest.raises(SystemExit):
        parse_args([*REQUIRED, "--epochs", "336", "--backbone-lr", ".0001",
                    "--allow-shared-proposal-negatives"])


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_nested_mask_negatives_and_native_token_rebuild(flag, deferred):
    batch, metadata = raw_batch(flag, deferred=deferred)
    expected = {2, 3, 4} if flag else {4}
    if not deferred:
        assert set(batch.negative_indices.flatten().tolist()) == expected
    tokens, metrics = build_token_pure_dec0_contrastive_batch(
        batch, torch.arange(6), torch.zeros(6, 3), max_positive_pairs_per_proposal=1)
    assert tokens is not None and tokens.allow_shared_proposal_negatives is flag
    assert set(tokens.negative_indices.flatten().tolist()) == expected
    assert metadata["teacher_co_membership_filtered"] is not flag
    assert metrics["teacher_co_membership_filtered_after_tokenization"] is not flag
    # Mixing known negative point 2 with unknown point 5 excludes that token.
    mixed, _ = build_token_pure_dec0_contrastive_batch(
        batch, torch.tensor([0, 1, 2, 3, 4, 2]), torch.zeros(5, 3),
        max_positive_pairs_per_proposal=1)
    assert mixed is not None and 2 not in mixed.negative_indices
    assert set(mixed.negative_indices.flatten().tolist()) <= expected - {2}


@pytest.mark.parametrize("flag", [False, True])
def test_serialization_and_mixed_policy_batches(flag):
    scene = nested_scene()
    batch = plan(scene, flag).groups[0].batch
    record = _triplet_batch_record(batch, token_pure_fixed_pack=True,
                                  fixed_runtime_token_pair_rebuild=True)
    sampling = adaptation.SamplingConfig(proposals_per_forward=4, positive_pairs_per_proposal=3,
        uniform_negatives=2, spatial_hard_negatives=0, feature_hard_negatives=0)
    restored = _triplet_batch_from_record(record, device=torch.device("cpu"),
        include_feature_hard=True, sampling=sampling, num_scene_points=len(scene.points))
    assert restored.allow_shared_proposal_negatives is flag
    assert restored.deferred_token_loss.allow_shared_proposal_negatives is flag
    bad = copy.deepcopy(record)
    bad["allow_shared_proposal_negatives"] = not flag
    with pytest.raises(ValueError, match="policy drift"):
        _triplet_batch_from_record(bad, device=torch.device("cpu"),
            include_feature_hard=True, sampling=sampling, num_scene_points=len(scene.points))
    with pytest.raises(ValueError, match="shared-proposal negatives"):
        concatenate_triplet_batches([batch, replace(batch, allow_shared_proposal_negatives=not flag)])


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_all_negative_families_share_the_selected_policy(flag, deferred):
    batch, _ = sample_safe_overlapping_proposal_triplets(
        num_scene_points=6, visible_indices=torch.arange(6),
        proposal_offsets=torch.tensor([0, 2, 6, 7]),
        proposal_point_indices=torch.tensor([0, 1, 0, 1, 2, 3, 4]),
        selected_proposal_indices=torch.tensor([0]), points=torch.arange(18).reshape(6, 3).float(),
        positive_pairs_per_proposal=1, num_uniform_negatives=1,
        num_spatial_hard_negatives=1, spatial_candidate_pool=4,
        num_feature_hard_negatives=1, feature_candidate_pool=4,
        require_negative_proposal_membership=True, allow_shared_proposal_negatives=flag,
        defer_token_loss=deferred, generator=torch.Generator().manual_seed(42))
    tokens, _ = build_token_pure_dec0_contrastive_batch(
        batch, torch.arange(6), torch.arange(18).reshape(6, 3).float(), max_positive_pairs_per_proposal=1)
    assert tokens is not None
    base = tokens.negative_indices[0].tolist()
    candidates = [] if tokens.feature_candidate_indices is None else tokens.feature_candidate_indices[0].tolist()
    assert set(base + candidates) == ({2, 3, 4} if flag else {4})
    assert len(base + candidates) == len(set(base + candidates))
    assert tokens.num_uniform_negatives == 1
    assert tokens.num_spatial_hard_negatives == int(flag)
    assert tokens.num_feature_hard_negatives == int(flag)


@pytest.mark.parametrize("flag", [False, True])
def test_coverage_recovery_and_sampling_health(flag):
    scene = nested_scene()
    first = plan(scene, flag)
    states = {CELL: DeterministicCoverageState.from_dict(first.state_after_by_cell()[CELL]).for_epoch(1)}
    recovered, repeated = plan(scene, flag, epoch=1, states=states), plan(scene, flag, epoch=1, states=states)
    assert recovered.state_after_by_cell() == repeated.state_after_by_cell()
    assert all(group.batch.allow_shared_proposal_negatives is flag for group in recovered.groups)
    record = {**recovered.audit_metadata(), "optimization_valid": True,
              "groups": [{"cell": group.cell, "sampler": group.coverage_metadata["sampler"]}
                         for group in recovered.groups]}
    health = _summarize_v2_sampling([record], cells=(CELL,),
                                   allow_shared_proposal_negatives=flag)
    assert health["negative_policy"] == negative_policy_name(flag)
    assert all(health["hard_invariants"].values())
    with pytest.raises(ValueError, match="negative policy"):
        _summarize_v2_sampling([record], cells=(CELL,),
                              allow_shared_proposal_negatives=not flag)


@pytest.mark.parametrize("flag", [False, True])
def test_public_fixed_preparation_uses_recipe_and_rejects_wrong_policy(flag, tmp_path, monkeypatch):
    from delimit3d.data.cli import fixed_eval_main
    scene = nested_scene()
    monkeypatch.setattr("delimit3d.pretraining.sources.load_scene_source", lambda *a, **kw: scene)
    split = tmp_path / "val.txt"; split.write_text(scene.scene_id + "\n")
    argv = ["--source-root", str(tmp_path), "--holdout-split", str(split),
        "--output", str(tmp_path / "fixed"), "--teacher", "sam2", "--grid-size", ".02",
        "--pretraining-contract", str(recipe(flag))]
    fixed_eval_main(argv)
    import json
    manifest = json.loads((tmp_path / "fixed/manifest.json").read_text())
    assert manifest["allow_shared_proposal_negatives"] is flag
    assert manifest["negative_policy"] == negative_policy_name(flag)
    with pytest.raises(SystemExit):
        fixed_eval_main(argv + ["--no-allow-shared-proposal-negatives" if flag else "--allow-shared-proposal-negatives"])
    with pytest.raises(ValueError, match="negative policy"):
        fixed_eval_main([*argv[:-1], str(recipe(not flag))])


def compare(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: compare(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): compare(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("flag", [False, True])
def test_training_fixed_evaluation_and_full_optimizer_recovery(flag, tmp_path, monkeypatch):
    monkeypatch.setattr("delimit3d.pretraining.model.LitePTBackbone", CPUBackbone)
    scene = nested_scene()
    def model():
        return adaptation.LitePTContrastiveModel(litept_root=tmp_path, token_pure_dec0=True,
                                                 token_pure_cells=(CELL,), grid_size=.02)
    torch.manual_seed(42)
    continuous = model()
    optimizer = torch.optim.AdamW(continuous.parameters(), lr=.0001)
    def update(m, opt, epoch, states=None):
        m.train(); opt.zero_grad()
        visit = plan(scene, flag, epoch=epoch, states=states)
        groups = visit.groups
        assert all(group.batch.allow_shared_proposal_negatives is flag for group in groups)
        result = m(torch.from_numpy(scene.points), torch.from_numpy(scene.features), frame_groups=groups)
        assert result["token_pure_dec0"]["optimization_valid"] and torch.isfinite(result["loss_total"])
        result["loss_total"].backward(); opt.step()
        return visit.state_after_by_cell()
    def next_states(states, epoch):
        return {cell: DeterministicCoverageState.from_dict(state).for_epoch(epoch)
                for cell, state in states.items()}
    states = update(continuous, optimizer, 0)
    args = argparse.Namespace(seed=42, sampling_mode="coverage_multigranular_v2",
                              source_granularity="sam2", allow_shared_proposal_negatives=flag)
    resolved = {"allow_shared_proposal_negatives": flag, "negative_policy": negative_policy_name(flag),
                "initialization": {"public_checkpoint_used": False}}
    checkpoint = _checkpoint_payload(epoch=1, update=1, model=continuous, optimizer=optimizer,
        args=args, resolved=resolved, evaluation=None, evaluation_history={},
        rank_exposure_counts=[{}], rank_coverage_states=[states], cumulative_health={})
    path = tmp_path / "checkpoint.pt"; torch.save(checkpoint, path)
    for epoch in (1, 2): states = update(continuous, optimizer, epoch, next_states(states, epoch))
    saved = torch.load(path, weights_only=False)
    assert saved["allow_shared_proposal_negatives"] is flag
    validate_negative_policy_resume(saved["resolved_config"], resolved, saved)
    recovered = model(); recovered.load_state_dict(saved["model_state_dict"], strict=True)
    resumed_optimizer = torch.optim.AdamW(recovered.parameters(), lr=.0001)
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
    recovered_states = saved["rank_coverage_states"][0]
    for epoch in (1, 2):
        recovered_states = update(recovered, resumed_optimizer, epoch, next_states(recovered_states, epoch))
    assert recovered_states == states
    compare(continuous.state_dict(), recovered.state_dict())
    compare(optimizer.state_dict(), resumed_optimizer.state_dict())
    sampling = adaptation.SamplingConfig(proposals_per_forward=4, positive_pairs_per_proposal=3,
        uniform_negatives=2, spatial_hard_negatives=0, feature_hard_negatives=0)
    kwargs = dict(pack_dir=tmp_path / "fixed", scenes=[scene], sampling=sampling,
                  sampling_profile="test", seed=42, cells=(CELL,), grid_size=.02,
                  allow_shared_proposal_negatives=flag)
    report = adaptation.prepare_fixed_holdout_eval_pack(**kwargs)
    evaluation = adaptation.evaluate_holdout(model=recovered, scenes=[scene], device=torch.device("cpu"),
        sampling=sampling, seed=42, cells=(CELL,), fixed_eval_pack=kwargs["pack_dir"],
        fixed_eval_pack_report=report, allow_shared_proposal_negatives=flag)
    assert evaluation["native_fixed_comparable_dec0"]["valid"]
    with pytest.raises(ValueError, match="negative policy"):
        adaptation.validate_fixed_holdout_eval_pack(**{**kwargs, "allow_shared_proposal_negatives": not flag})
    with pytest.raises(ValueError, match="negative policy"):
        adaptation.evaluate_holdout(model=recovered, scenes=[scene], device=torch.device("cpu"),
            sampling=sampling, seed=42, fixed_eval_pack=kwargs["pack_dir"], fixed_eval_pack_report=report,
            cells=(CELL,), allow_shared_proposal_negatives=not flag)


@pytest.mark.parametrize("flag", [False, True])
def test_resume_rejects_a_policy_change_and_tampered_checkpoint(flag):
    previous = {"allow_shared_proposal_negatives": flag, "negative_policy": negative_policy_name(flag)}
    with pytest.raises(ValueError, match="differs on resume"):
        validate_negative_policy_resume(previous, {"allow_shared_proposal_negatives": not flag})
    with pytest.raises(ValueError, match="Checkpoint negative policy"):
        validate_negative_policy_resume(previous, previous, {"allow_shared_proposal_negatives": not flag})
    validate_negative_policy_resume({}, {"allow_shared_proposal_negatives": False})
    with pytest.raises(ValueError, match="differs on resume"):
        validate_negative_policy_resume({}, {"allow_shared_proposal_negatives": True})


@pytest.mark.parametrize("bad", ["false", 1, None])
def test_policy_must_be_an_actual_boolean(bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        negative_policy_name(bad)
