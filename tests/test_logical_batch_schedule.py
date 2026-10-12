import copy

import pytest
import torch

from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule
from delimit3d.training.runner import (
    _accumulate_rank_cumulative_health_update,
    _merge_rank_cumulative_health,
    _new_rank_cumulative_health,
    _restore_rank_cumulative_health,
    _update_for_completed_epoch,
    _global_auxiliary_budget_passed,
)


def test_production_has_exact_epochs_and_no_lost_or_padded_visits():
    schedule = LogicalBatchSchedule(1465, 6, 150, (25, 50, 75, 100, 125, 150))
    assert schedule.total_updates == 36630
    assert schedule.short_updates == {e // 25 * 6105: 1 for e in schedule.flush_epochs}
    for epoch in (0, *schedule.flush_epochs):
        update = schedule.update_at_epoch(epoch)
        assert update == epoch // 25 * 6105
        assert schedule.consumed_visits(update) == epoch * 1465
        assert _update_for_completed_epoch(
            epoch=epoch, scene_visits_per_epoch_per_rank=1465,
            scenes_per_rank_per_update=6, logical_batch_schedule=schedule,
        ) == update
    assert sum(schedule.batch_size(u) for u in range(1, schedule.total_updates + 1)) == 219750
    assert schedule.update_at_epoch(17, allow_partial=True) == 4151
    with pytest.raises(ValueError, match="Epoch boundary does not align"):
        _update_for_completed_epoch(epoch=25, scene_visits_per_epoch_per_rank=1465,
                                    scenes_per_rank_per_update=6)


@pytest.mark.parametrize("n,epochs,flush", [(7, 2, (1, 2)), (5, 4, (2, 4)), (8, 9, (3, 6, 9)), (3, 3, (1, 3))])
def test_streaming_groups_match_schedule_and_restart(n, epochs, flush):
    schedule = LogicalBatchSchedule(n, 6, epochs, flush)

    def groups(start):
        pending, result = [], []
        for epoch in range(start, epochs):
            for index in range(n):
                pending.append((epoch, index))
                if len(pending) == 6 or index + 1 == n and epoch + 1 in flush:
                    result.append(tuple(pending))
                    pending = []
        assert not pending
        return result

    full = groups(0)
    assert len(full) == schedule.total_updates
    assert [len(g) for g in full] == [schedule.batch_size(u) for u in range(1, len(full) + 1)]
    assert [v for group in full for v in group] == [(e, i) for e in range(epochs) for i in range(n)]
    for boundary in flush:
        update = schedule.update_at_epoch(boundary)
        assert full[:update] + groups(boundary) == full
        assert schedule.consumed_visits(update) == boundary * n


def _advance(state, update, attempted):
    _accumulate_rank_cumulative_health_update(
        state, update=update, local_was_clipped=False, clipped_rank_count=0,
        world_size=2, attempted_microbatch_count=attempted,
        optimization_valid_count=attempted, token_records=[], hierarchy_records=[],
    )


def test_short_batch_health_checkpoints_resume_with_exact_denominators():
    schedule = LogicalBatchSchedule(7, 6, 2, (1, 2))
    states = [_new_rank_cumulative_health(
        rank=r, cells=("2d/sam2",), hierarchy_enabled=False,
        scheduled_gradient_panel_updates=[], expected_microbatches_per_update=6,
        logical_batch_schedule=schedule,
    ) for r in range(2)]
    for update in (1, 2):
        for state in states:
            _advance(state, update, schedule.batch_size(update))
    merged = _merge_rank_cumulative_health(states, through_update=2, world_size=2,
                                         gradient_conflict_controller={})
    assert merged["token_delivery"]["expected_microbatch_count"] == 14
    assert merged["token_delivery"]["microbatch_count"] == 14
    restored = [_restore_rank_cumulative_health(
        merged, rank=r, checkpoint_update=2, world_size=2,
        scheduled_gradient_panel_updates=[], allow_schedule_extension=False,
    ) for r in range(2)]
    assert restored == states
    for update in (3, 4):
        for state in restored:
            _advance(state, update, schedule.batch_size(update))
    final = _merge_rank_cumulative_health(restored, through_update=4, world_size=2,
                                        gradient_conflict_controller={})
    assert final["token_delivery"]["microbatch_count"] == 28
    assert final["optimizer_update_count"] == 4
    with pytest.raises(ValueError, match="denominator drift"):
        _advance(copy.deepcopy(states[0]), 3, 1)
    broken = copy.deepcopy(merged)
    broken["token_delivery"]["microbatch_count"] += 1
    with pytest.raises(ValueError, match="aggregate is corrupt"):
        _restore_rank_cumulative_health(broken, rank=0, checkpoint_update=2,
            world_size=2, scheduled_gradient_panel_updates=[], allow_schedule_extension=False)


def test_partial_batch_adamw_restart_is_identical_and_uses_actual_average():
    schedule = LogicalBatchSchedule(7, 6, 2, (1, 2))
    inputs = torch.arange(14, dtype=torch.float32).reshape(-1, 1) / 14

    def train(end, saved=None):
        torch.manual_seed(42)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        start = 0
        if saved is not None:
            start, model_state, optimizer_state = saved
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        for update in range(start + 1, end + 1):
            lo, hi = schedule.consumed_visits(update - 1), schedule.consumed_visits(update)
            optimizer.zero_grad()
            for x in inputs[lo:hi]:
                ((model(x) - x).square().mean() / schedule.batch_size(update)).backward()
            expected_model = copy.deepcopy(model)
            expected_model.zero_grad()
            (expected_model(inputs[lo:hi]) - inputs[lo:hi]).square().mean().backward()
            assert all(torch.allclose(a.grad, b.grad, atol=1e-7) for a, b in
                       zip(model.parameters(), expected_model.parameters()))
            optimizer.step()
        return end, copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    complete = train(4)
    resumed = train(4, train(2))
    assert all(torch.equal(complete[1][k], resumed[1][k]) for k in complete[1])
    for key, value in complete[2]["state"].items():
        assert all(torch.equal(v, resumed[2]["state"][key][k]) for k, v in value.items())


@pytest.mark.parametrize("args", [(7, 0, 2, (1, 2)), (7, 6, 2, (1,)), (7, 6, 2, (2, 1)), (7, 6, 2, (True, 2))])
def test_invalid_schedule_fails_closed(args):
    with pytest.raises(ValueError):
        LogicalBatchSchedule(*args)


def test_inactive_auxiliary_budget_is_inapplicable_not_a_failed_measurement():
    inactive = {"rejected": False, "objectives": {}, "auxiliary_budget": {
        "passed": None, "violation_count": 0, "last_global_sum_aux_norm_ratio": None,
    }}
    assert _global_auxiliary_budget_passed(inactive, auxiliary_enabled=False)
    assert not _global_auxiliary_budget_passed(inactive, auxiliary_enabled=True)
    rejected = copy.deepcopy(inactive)
    rejected["rejected"] = True
    assert not _global_auxiliary_budget_passed(rejected, auxiliary_enabled=False)
    failed = copy.deepcopy(inactive)
    failed["auxiliary_budget"].update(passed=False, violation_count=1)
    assert not _global_auxiliary_budget_passed(failed, auxiliary_enabled=False)
    assert not _global_auxiliary_budget_passed(failed, auxiliary_enabled=True)
    active = copy.deepcopy(inactive)
    active["objectives"] = {"hierarchy_mask_scaled": {}}
    active["auxiliary_budget"].update(passed=True, last_global_sum_aux_norm_ratio=0.3)
    assert _global_auxiliary_budget_passed(active, auxiliary_enabled=True)
    assert not _global_auxiliary_budget_passed(active, auxiliary_enabled=False)
