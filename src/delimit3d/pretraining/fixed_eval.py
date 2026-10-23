"""Persistent fixed evaluation packs and their validation."""

from __future__ import annotations

import json

import math


from dataclasses import replace

from pathlib import Path

from tempfile import NamedTemporaryFile

from typing import Any, Mapping, Sequence


import torch


from delimit3d.training.contracts import GRANULARITY_KEYS, SOURCE_CELLS, TWO_D_CELLS, normalize_source_cells, sha256_file, stable_seed

from delimit3d.data.partfield_contrastive import SAMPLER_ALGORITHM_VERSION, SAMPLER_PRESELECTION_POLICY, ContrastiveTripletBatch, DeferredTokenLossPlan, validate_token_positive_pair_payload
from delimit3d.data.partfield_contrastive import negative_policy_name
from delimit3d.data.sam2_source import fragment_policy, validate_point_limit

from delimit3d.data.contrastive_sampler_v2 import (
    DeterministicCoverageState,
    FrameProposalCatalog,
    NoRetainedFrameGroupsError,
    source_cell_quotas,
    sample_multigranular_frame_group_plan,
)


from .sampling import (
    _holdout_batches,
)
from .settings import (
    FIXED_EVAL_PACK_SCHEMA,
    FIXED_EVAL_SCENE_SCHEMA,
    SamplingConfig,
    TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY,
    TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION,
    TOKEN_PURE_FIXED_EVAL_SCHEMA,
)
from .sources import (
    SceneSource,
)


def _token_pure_fixed_holdout_plan(
    *,
    scene: SceneSource,
    points: torch.Tensor,
    device: torch.device,
    sampling: SamplingConfig,
    seed: int,
    cells: Sequence[str] = TWO_D_CELLS,
    grid_size: float = 0.02,
    allow_shared_proposal_negatives: bool = False,
) -> Any:
    """Build one deterministic V2 heldout visit with exact frame locality.

    The V2 sampler, unlike the legacy holdout sampler above, carries the union
    of every selected-source mask for a physical frame as its relation CSR and
    restricts negative eligibility to visible points with known relation
    membership.  Rotating the 4-way evaluation quota by scene balances the
    unavoidable extra proposal over a 3-cell panel without changing a saved
    pack between checkpoints.
    """

    frames_by_cell: dict[str, tuple[FrameProposalCatalog, ...]] = {}
    for cell in cells:
        granularity = cell.split("/", 1)[1]
        frames_by_cell[cell] = tuple(
            FrameProposalCatalog(
                cell=cell,
                frame_id=frame.physical_frame_id,
                visible_indices=torch.as_tensor(
                    frame.visible_indices, dtype=torch.long, device=device
                ),
                proposal_offsets=torch.as_tensor(
                    frame.proposal_offsets, dtype=torch.long, device=device
                ),
                proposal_point_indices=torch.as_tensor(
                    frame.proposal_point_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
            for frame in scene.frames(granularity, "heldout")
        )
    quota_rotation = int(
        stable_seed(seed, scene.scene_id, "fixed-eval-v2-quota-rotation")
        % len(cells)
    )
    total_quota = max(len(cells), int(sampling.proposals_per_forward))
    quotas = source_cell_quotas(
        quota_rotation,
        total_proposals=total_quota,
        cells=cells,
    )
    populations = {
        cell: sum(
            int(catalog.valid_proposal_indices().numel())
            for catalog in frames_by_cell[cell]
        )
        for cell in cells
    }
    if any(populations[cell] <= 0 for cell in cells):
        raise ValueError(
            f"{scene.scene_id}: V2 fixed-eval catalog lacks a granularity: "
            f"{populations}"
        )
    # Retry within the exact same deterministic fixed-eval key by restoring
    # each cell's state_after cursor.  ``quota_rotation`` affects only which
    # cell receives the extra fixed-pack proposal; it is not a training epoch
    # and must not demand a non-existent training-resume cursor.  The bound
    # visits every catalog entry at least once, so failure means no complete
    # safe selected-cell slice exists rather than an unlucky first draw.
    maximum_attempts = max(
        math.ceil(populations[cell] / quotas[cell])
        for cell in cells
    )
    coverage_states: dict[str, DeterministicCoverageState | None] | None = None
    last_plan = None
    for attempt in range(maximum_attempts):
        try:
            plan = sample_multigranular_frame_group_plan(
                scene_id=scene.scene_id,
                epoch=0,
                seed=seed,
                points=points,
                frames_by_cell=frames_by_cell,
                total_proposal_quota=total_quota,
                cells=cells,
                positive_pairs_per_proposal=int(
                    sampling.positive_pairs_per_proposal
                ),
                num_uniform_negatives=int(sampling.uniform_negatives),
                num_spatial_hard_negatives=int(sampling.spatial_hard_negatives),
                spatial_candidate_pool=int(sampling.spatial_candidate_pool),
                num_feature_hard_negatives=int(sampling.feature_hard_negatives),
                feature_candidate_pool=int(sampling.feature_candidate_pool),
                require_negative_proposal_membership=True,
                allow_shared_proposal_negatives=allow_shared_proposal_negatives,
                require_hierarchy_routes=False,
                grid_size=grid_size,
                operation="multigranular-heldout-fixed-v2",
                coverage_states=coverage_states,
                quota_epoch=quota_rotation,
            )
        except NoRetainedFrameGroupsError as error:
            plan = error.plan
        last_plan = plan
        if {group.cell for group in plan.groups} == set(cells):
            annotated = {
                cell: {
                    **dict(metadata),
                    "fixed_eval_search_attempt": attempt,
                    "fixed_eval_search_maximum_attempts": maximum_attempts,
                }
                for cell, metadata in plan.coverage_by_cell.items()
            }
            return replace(plan, coverage_by_cell=annotated)
        coverage_states = {
            cell: (
                DeterministicCoverageState.from_dict(state)
                if isinstance(state, Mapping)
                else None
            )
            for cell, state in plan.state_after_by_cell().items()
        }
    raise ValueError(
        f"{scene.scene_id}: V2 fixed-eval queue exhausted without one safe "
        f"selected-cell slice after {maximum_attempts} attempts: "
        f"{last_plan.audit_metadata() if last_plan is not None else populations}"
    )


def _expected_holdout_record_count(
    scene: SceneSource,
    cells: Sequence[str] | None = None,
) -> int:
    active_cells = set(normalize_source_cells(cells))
    count = 0
    for key in GRANULARITY_KEYS:
        if f"3d/{key}" in active_cells:
            count += 1
        if f"2d/{key}" in active_cells:
            count += len(scene.frames(key, "heldout"))
    return count


def _triplet_batch_record(
    batch: ContrastiveTripletBatch,
    *,
    token_pure_fixed_pack: bool = False,
    fixed_runtime_token_pair_rebuild: bool = False,
) -> dict[str, Any]:
    token_pair_payload_fields = (
        batch.token_positive_pair_offsets,
        batch.token_positive_pair_indices,
        batch.token_positive_pair_digest,
    )
    token_pair_payload_presence = tuple(
        value is not None for value in token_pair_payload_fields
    )
    if any(token_pair_payload_presence) and not all(token_pair_payload_presence):
        raise ValueError(
            "native-dec0 positive-pair payload fields must be supplied together"
        )
    if fixed_runtime_token_pair_rebuild:
        if not token_pure_fixed_pack:
            raise ValueError(
                "runtime token-pair rebuild is only valid for a token-pure fixed pack"
            )
        if not all(token_pair_payload_presence):
            raise ValueError(
                "fixed token-pair rebuild requires a validated planner payload"
            )
        if (
            batch.proposal_member_offsets is None
            or batch.pair_proposal_indices is None
        ):
            raise ValueError(
                "fixed token-pair rebuild requires proposal-occurrence metadata"
            )
        assert batch.token_positive_pair_offsets is not None
        assert batch.token_positive_pair_indices is not None
        assert batch.token_positive_pair_digest is not None
        validate_token_positive_pair_payload(
            batch.token_positive_pair_offsets,
            batch.token_positive_pair_indices,
            batch.token_positive_pair_digest,
            proposal_count=int(batch.proposal_member_offsets.numel()) - 1,
            algorithm_version=batch.algorithm_version,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=batch.proposal_labels,
            pair_proposal_indices=batch.pair_proposal_indices,
        )
    return {
        "token_pure_fixed_pack": bool(token_pure_fixed_pack),
        "positive_pairs": batch.positive_pairs.detach().cpu().long(),
        "negative_indices": batch.negative_indices.detach().cpu().long(),
        "proposal_labels": batch.proposal_labels.detach().cpu().long(),
        "feature_candidate_indices": (
            batch.feature_candidate_indices.detach().cpu().long()
            if batch.feature_candidate_indices is not None
            else None
        ),
        "deferred_token_loss": (
            batch.deferred_token_loss.metadata()
            if batch.deferred_token_loss is not None
            else None
        ),
        "num_feature_hard_negatives": int(batch.num_feature_hard_negatives),
        "proposal_member_offsets": (
            batch.proposal_member_offsets.detach().cpu().long()
            if batch.proposal_member_offsets is not None
            else None
        ),
        "proposal_member_indices": (
            batch.proposal_member_indices.detach().cpu().long()
            if batch.proposal_member_indices is not None
            else None
        ),
        "pair_proposal_indices": (
            batch.pair_proposal_indices.detach().cpu().long()
            if batch.pair_proposal_indices is not None
            else None
        ),
        "eligible_indices": (
            batch.eligible_indices.detach().cpu().long()
            if batch.eligible_indices is not None
            else None
        ),
        "relation_proposal_offsets": (
            batch.relation_proposal_offsets.detach().cpu().long()
            if batch.relation_proposal_offsets is not None
            else None
        ),
        "relation_proposal_member_indices": (
            batch.relation_proposal_member_indices.detach().cpu().long()
            if batch.relation_proposal_member_indices is not None
            else None
        ),
        "token_positive_pair_offsets": (
            batch.token_positive_pair_offsets.detach().cpu().long()
            if (
                batch.token_positive_pair_offsets is not None
                and not fixed_runtime_token_pair_rebuild
            )
            else None
        ),
        "token_positive_pair_indices": (
            batch.token_positive_pair_indices.detach().cpu().long()
            if (
                batch.token_positive_pair_indices is not None
                and not fixed_runtime_token_pair_rebuild
            )
            else None
        ),
        "token_positive_pair_digest": (
            None
            if fixed_runtime_token_pair_rebuild
            else batch.token_positive_pair_digest
        ),
        "native_dec0_pair_selection": (
            TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
            if fixed_runtime_token_pair_rebuild
            else None
        ),
        "algorithm_version": batch.algorithm_version,
        "require_negative_proposal_membership": bool(
            batch.require_negative_proposal_membership
        ),
        **({"allow_shared_proposal_negatives": True}
           if batch.allow_shared_proposal_negatives else {}),
        "multiscale_seed": batch.multiscale_seed,
        "num_uniform_negatives": int(batch.num_uniform_negatives),
        "num_spatial_hard_negatives": int(batch.num_spatial_hard_negatives),
        "spatial_candidate_pool": int(batch.spatial_candidate_pool),
        "feature_candidate_pool": int(batch.feature_candidate_pool),
    }


def _triplet_batch_from_record(
    record: dict[str, Any],
    *,
    device: torch.device,
    include_feature_hard: bool,
    sampling: SamplingConfig,
    num_scene_points: int,
) -> ContrastiveTripletBatch:
    token_pure_fixed_pack = bool(record.get("token_pure_fixed_pack", False))
    positive_pairs = record["positive_pairs"].to(device=device, dtype=torch.long)
    negative_indices = record["negative_indices"].to(
        device=device, dtype=torch.long
    )
    proposal_labels = record["proposal_labels"].to(
        device=device, dtype=torch.long
    )
    feature_candidates_cpu = record.get("feature_candidate_indices")
    feature_candidates = (
        feature_candidates_cpu.to(device=device, dtype=torch.long)
        if include_feature_hard and feature_candidates_cpu is not None
        else None
    )
    stored_feature_hard_count = int(record["num_feature_hard_negatives"])
    feature_hard_count = stored_feature_hard_count if include_feature_hard else 0
    if token_pure_fixed_pack and "require_negative_proposal_membership" not in record:
        raise ValueError(
            "Token-pure fixed eval requires an explicit negative-membership policy"
        )
    require_negative_proposal_membership = bool(
        record.get("require_negative_proposal_membership", False)
    )
    allow_shared_proposal_negatives = record.get("allow_shared_proposal_negatives", False)
    negative_policy_name(allow_shared_proposal_negatives)
    if token_pure_fixed_pack and not require_negative_proposal_membership:
        raise ValueError(
            "Token-pure fixed eval must restrict negatives to known relation members"
        )
    stored_feature_candidate_pool = int(
        record.get("feature_candidate_pool", sampling.feature_candidate_pool)
    )
    deferred_metadata = record.get("deferred_token_loss")
    deferred_token_loss: DeferredTokenLossPlan | None = None
    if deferred_metadata is not None:
        if not isinstance(deferred_metadata, Mapping):
            raise ValueError("Fixed-eval deferred token-loss metadata is malformed")
        try:
            deferred_token_loss = DeferredTokenLossPlan(
                requested_positive_pairs_per_proposal=int(
                    deferred_metadata["requested_positive_pairs_per_proposal"]
                ),
                requested_uniform_negatives_per_pair=int(
                    deferred_metadata["requested_uniform_negatives_per_pair"]
                ),
                requested_spatial_hard_negatives_per_pair=int(
                    deferred_metadata["requested_spatial_hard_negatives_per_pair"]
                ),
                requested_feature_hard_negatives_per_pair=int(
                    deferred_metadata["requested_feature_hard_negatives_per_pair"]
                ),
                effective_uniform_negatives_per_pair=int(
                    deferred_metadata["effective_uniform_negatives_per_pair"]
                ),
                effective_spatial_hard_negatives_per_pair=int(
                    deferred_metadata["effective_spatial_hard_negatives_per_pair"]
                ),
                effective_feature_candidate_pool_per_pair=int(
                    deferred_metadata["effective_feature_candidate_pool_per_pair"]
                ),
                effective_feature_hard_negatives_per_pair=int(
                    deferred_metadata["effective_feature_hard_negatives_per_pair"]
                ),
                point_negative_ids_materialized=bool(
                    deferred_metadata["point_negative_ids_materialized"]
                ),
                consumed_loss_space=str(deferred_metadata["consumed_loss_space"]),
                schema_version=str(deferred_metadata["schema_version"]),
                algorithm_version=str(deferred_metadata["algorithm_version"]),
                require_negative_proposal_membership=bool(
                    deferred_metadata["require_negative_proposal_membership"]
                ),
                allow_shared_proposal_negatives=(
                    deferred_metadata.get("allow_shared_proposal_negatives", False)
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Fixed-eval deferred token-loss metadata is invalid"
            ) from error
        if deferred_token_loss.metadata() != dict(deferred_metadata):
            raise ValueError("Fixed-eval deferred token-loss metadata drift")
        if deferred_token_loss.allow_shared_proposal_negatives != allow_shared_proposal_negatives:
            raise ValueError("Fixed-eval shared-proposal policy drift")
        if (
            bool(deferred_token_loss.require_negative_proposal_membership)
            != require_negative_proposal_membership
        ):
            raise ValueError(
                "Fixed-eval deferred token-loss membership policy drift"
            )
        observed_deferred_widths = (
            int(record.get("num_uniform_negatives", sampling.uniform_negatives)),
            int(
                record.get(
                    "num_spatial_hard_negatives", sampling.spatial_hard_negatives
                )
            ),
            stored_feature_candidate_pool,
            stored_feature_hard_count,
        )
        expected_deferred_widths = (
            int(deferred_token_loss.effective_uniform_negatives_per_pair),
            int(deferred_token_loss.effective_spatial_hard_negatives_per_pair),
            int(deferred_token_loss.effective_feature_candidate_pool_per_pair),
            int(deferred_token_loss.effective_feature_hard_negatives_per_pair),
        )
        if observed_deferred_widths != expected_deferred_widths:
            raise ValueError("Fixed-eval deferred token-loss width drift")
    if (
        token_pure_fixed_pack
        and stored_feature_hard_count > 0
        and deferred_token_loss is None
        and feature_candidates_cpu is None
    ):
        raise ValueError(
            "Token-pure fixed eval needs deferred native-token feature candidates"
        )
    if not include_feature_hard:
        stored_feature_candidate_pool = 0
        if deferred_token_loss is not None:
            deferred_token_loss = replace(
                deferred_token_loss,
                requested_feature_hard_negatives_per_pair=0,
                effective_feature_candidate_pool_per_pair=0,
                effective_feature_hard_negatives_per_pair=0,
            )
    membership_keys = (
        "proposal_member_offsets",
        "proposal_member_indices",
        "pair_proposal_indices",
        "eligible_indices",
    )
    membership_presence = [record.get(key) is not None for key in membership_keys]
    if any(membership_presence) and not all(membership_presence):
        raise ValueError("Fixed-eval proposal-membership metadata is partial")
    proposal_member_offsets = (
        record["proposal_member_offsets"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    proposal_member_indices = (
        record["proposal_member_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    pair_proposal_indices = (
        record["pair_proposal_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    eligible_indices = (
        record["eligible_indices"].to(device=device, dtype=torch.long)
        if all(membership_presence)
        else None
    )
    relation_keys = (
        "relation_proposal_offsets",
        "relation_proposal_member_indices",
    )
    relation_presence = [record.get(key) is not None for key in relation_keys]
    if any(relation_presence) and not all(relation_presence):
        raise ValueError("Fixed-eval relation-proposal metadata is partial")
    relation_proposal_offsets = (
        record["relation_proposal_offsets"].to(
            device=device, dtype=torch.long
        )
        if all(relation_presence)
        else None
    )
    relation_proposal_member_indices = (
        record["relation_proposal_member_indices"].to(
            device=device, dtype=torch.long
        )
        if all(relation_presence)
        else None
    )
    token_pair_keys = (
        "token_positive_pair_offsets",
        "token_positive_pair_indices",
        "token_positive_pair_digest",
    )
    token_pair_presence = [record.get(key) is not None for key in token_pair_keys]
    if any(token_pair_presence) and not all(token_pair_presence):
        raise ValueError("Fixed-eval native-dec0 pair payload is partial")
    fixed_pair_selection = record.get("native_dec0_pair_selection")
    runtime_pair_rebuild = (
        fixed_pair_selection == TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
    )
    if token_pure_fixed_pack:
        if runtime_pair_rebuild:
            if any(token_pair_presence):
                raise ValueError(
                    "runtime fixed token-pair rebuild must not serialize backend token IDs"
                )
        elif not all(token_pair_presence):
            raise ValueError(
                "Token-pure fixed eval requires an exact payload or runtime "
                "native-token rebuild policy"
            )
    token_positive_pair_offsets = (
        record["token_positive_pair_offsets"].to(device=device, dtype=torch.long)
        if all(token_pair_presence)
        else None
    )
    token_positive_pair_indices = (
        record["token_positive_pair_indices"].to(device=device, dtype=torch.long)
        if all(token_pair_presence)
        else None
    )
    token_positive_pair_digest = (
        record["token_positive_pair_digest"] if all(token_pair_presence) else None
    )
    algorithm_version = record.get("algorithm_version")
    if token_pure_fixed_pack and (
        not all(membership_presence) or not all(relation_presence)
    ):
        raise ValueError(
            "Token-pure fixed eval requires selected and full relation CSR"
        )
    expected_pairs = (
        int(sampling.proposals_per_forward)
        * int(sampling.positive_pairs_per_proposal)
    )
    expected_base = int(sampling.uniform_negatives) + int(
        sampling.spatial_hard_negatives
    )
    if token_pure_fixed_pack:
        if positive_pairs.ndim != 2 or positive_pairs.shape[1] != 2:
            raise ValueError(
                "V2 fixed-eval positive pairs must have shape [P,2]"
            )
        if positive_pairs.shape[0] <= 0:
            raise ValueError("V2 fixed-eval batch contains no positive pair")
    elif positive_pairs.shape != (expected_pairs, 2):
        raise ValueError(
            f"Fixed-eval positive-pair shape drift: {tuple(positive_pairs.shape)}"
        )
    if token_pure_fixed_pack:
        if (
            negative_indices.ndim != 2
            or negative_indices.shape[0] != positive_pairs.shape[0]
        ):
            raise ValueError(
                "V2 fixed-eval negative rows do not match positive pairs"
            )
    elif negative_indices.shape != (expected_pairs, expected_base):
        raise ValueError(
            "Fixed-eval base-negative shape drift: "
            f"{tuple(negative_indices.shape)}"
        )
    expected_record_pairs = int(positive_pairs.shape[0])
    if proposal_labels.shape != (expected_record_pairs,):
        raise ValueError(
            f"Fixed-eval proposal-label shape drift: {tuple(proposal_labels.shape)}"
        )
    if include_feature_hard:
        if feature_candidates is None:
            if (
                not token_pure_fixed_pack
                or (feature_hard_count != 0 and deferred_token_loss is None)
            ):
                raise ValueError(
                    "Fixed-eval adaptive view lacks feature candidates"
                )
        else:
            if feature_candidates.ndim != 2:
                raise ValueError(
                    "Fixed-eval feature candidates must have shape [P,Q]"
                )
            expected_feature_shape = (
                (expected_record_pairs, int(feature_candidates.shape[1]))
                if token_pure_fixed_pack
                else (expected_pairs, int(sampling.feature_candidate_pool))
            )
            if feature_candidates.shape != expected_feature_shape:
                raise ValueError(
                    "Fixed-eval feature-candidate shape drift: "
                    f"{tuple(feature_candidates.shape)}"
                )
            if token_pure_fixed_pack:
                if (
                    feature_hard_count < 0
                    or feature_hard_count > int(feature_candidates.shape[1])
                    or feature_hard_count > int(sampling.feature_hard_negatives)
                ):
                    raise ValueError("V2 fixed-eval feature-hard count drift")
            elif feature_hard_count != int(sampling.feature_hard_negatives):
                raise ValueError("Fixed-eval feature-hard count drift")
    index_chunks = [positive_pairs.reshape(-1), negative_indices.reshape(-1)]
    if feature_candidates is not None:
        index_chunks.append(feature_candidates.reshape(-1))
    indices = torch.cat(index_chunks)
    if (
        indices.numel() == 0
        or int(indices.min()) < 0
        or int(indices.max()) >= int(num_scene_points)
    ):
        raise IndexError("Fixed-eval pack contains an out-of-scene point index")
    if proposal_member_offsets is not None:
        assert proposal_member_indices is not None
        assert pair_proposal_indices is not None
        assert eligible_indices is not None
        if (
            proposal_member_offsets.ndim != 1
            or proposal_member_offsets.numel() < 2
            or int(proposal_member_offsets[0]) != 0
            or bool((proposal_member_offsets[1:] < proposal_member_offsets[:-1]).any())
            or int(proposal_member_offsets[-1])
            != int(proposal_member_indices.numel())
            or pair_proposal_indices.shape != (expected_record_pairs,)
            or pair_proposal_indices.numel() == 0
            or proposal_member_indices.numel() == 0
            or int(proposal_member_indices.min()) < 0
            or int(proposal_member_indices.max()) >= int(num_scene_points)
            or eligible_indices.numel() == 0
            or int(eligible_indices.min()) < 0
            or int(eligible_indices.max()) >= int(num_scene_points)
            or int(pair_proposal_indices.min()) < 0
            or int(pair_proposal_indices.max())
            >= int(proposal_member_offsets.numel()) - 1
        ):
            raise ValueError("Fixed-eval proposal-membership contract drift")
    if all(token_pair_presence):
        assert token_positive_pair_offsets is not None
        assert token_positive_pair_indices is not None
        assert token_positive_pair_digest is not None
        if proposal_member_offsets is None or pair_proposal_indices is None:
            raise ValueError(
                "Native-dec0 pair payload requires proposal occurrence metadata"
            )
        validate_token_positive_pair_payload(
            token_positive_pair_offsets,
            token_positive_pair_indices,
            token_positive_pair_digest,
            proposal_count=int(proposal_member_offsets.numel()) - 1,
            algorithm_version=algorithm_version,
            preselection_policy=SAMPLER_PRESELECTION_POLICY,
            proposal_labels=proposal_labels,
            pair_proposal_indices=pair_proposal_indices,
        )
    if relation_proposal_offsets is not None:
        assert relation_proposal_member_indices is not None
        if (
            relation_proposal_offsets.ndim != 1
            or relation_proposal_offsets.numel() < 2
            or int(relation_proposal_offsets[0]) != 0
            or bool((relation_proposal_offsets[1:] < relation_proposal_offsets[:-1]).any())
            or int(relation_proposal_offsets[-1])
            != int(relation_proposal_member_indices.numel())
            or relation_proposal_member_indices.numel() == 0
            or int(relation_proposal_member_indices.min()) < 0
            or int(relation_proposal_member_indices.max())
            >= int(num_scene_points)
        ):
            raise ValueError("Fixed-eval full relation CSR contract drift")
    return ContrastiveTripletBatch(
        positive_pairs=positive_pairs,
        negative_indices=negative_indices,
        proposal_labels=proposal_labels,
        feature_candidate_indices=feature_candidates,
        num_feature_hard_negatives=feature_hard_count,
        proposal_member_offsets=proposal_member_offsets,
        proposal_member_indices=proposal_member_indices,
        pair_proposal_indices=pair_proposal_indices,
        eligible_indices=eligible_indices,
        relation_proposal_offsets=relation_proposal_offsets,
        relation_proposal_member_indices=(
            relation_proposal_member_indices
        ),
        multiscale_seed=(
            int(record["multiscale_seed"])
            if record.get("multiscale_seed") is not None
            else None
        ),
        num_uniform_negatives=int(
            record.get("num_uniform_negatives", sampling.uniform_negatives)
        ),
        num_spatial_hard_negatives=int(
            record.get(
                "num_spatial_hard_negatives",
                sampling.spatial_hard_negatives,
            )
        ),
        spatial_candidate_pool=int(
            record.get("spatial_candidate_pool", sampling.spatial_candidate_pool)
        ),
        feature_candidate_pool=int(
            stored_feature_candidate_pool
        ),
        algorithm_version=algorithm_version,
        deferred_token_loss=deferred_token_loss,
        require_negative_proposal_membership=require_negative_proposal_membership,
        allow_shared_proposal_negatives=allow_shared_proposal_negatives,
        token_positive_pair_offsets=token_positive_pair_offsets,
        token_positive_pair_indices=token_positive_pair_indices,
        token_positive_pair_digest=token_positive_pair_digest,
    )


def _validate_token_pure_fixed_record(
    *,
    scene: SceneSource,
    record: Mapping[str, Any],
    batch: ContrastiveTripletBatch,
    cells: Sequence[str] = TWO_D_CELLS,
) -> None:
    """Verify a saved V2 group against immutable same-frame source masks."""

    if not batch.require_negative_proposal_membership:
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record permits unknown negatives"
        )
    if (
        int(batch.num_feature_hard_negatives) > 0
        and batch.feature_candidate_indices is None
        and batch.deferred_token_loss is None
    ):
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record lacks deferred feature candidates"
        )

    cell = str(record.get("cell"))
    if cell not in cells:
        raise ValueError(
            f"{scene.scene_id}: token-pure fixed record has cell {cell!r}"
        )
    frame_id = str(record.get("frame_id", ""))
    if not frame_id or str(record.get("group")) != f"2d/{frame_id}":
        raise ValueError(
            f"{scene.scene_id}/{cell}: token-pure fixed frame identity drift"
        )
    granularity = cell.split("/", 1)[1]
    selected_frames = tuple(
        frame
        for frame in scene.frames(granularity, "heldout")
        if frame.physical_frame_id == frame_id
    )
    if len(selected_frames) != 1:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: expected one heldout frame"
        )
    selected_frame = selected_frames[0]

    relation_members: list[torch.Tensor] = []
    relation_offsets = [0]
    for relation_cell in cells:
        relation_granularity = relation_cell.split("/", 1)[1]
        matches = tuple(
            frame
            for frame in scene.frames(relation_granularity, "heldout")
            if frame.physical_frame_id == frame_id
        )
        if len(matches) > 1:
            raise ValueError(
                f"{scene.scene_id}/{frame_id}: ambiguous same-frame relation "
                f"for {relation_cell}"
            )
        if not matches:
            # Source packs need not contain the same physical camera in every
            # granularity cell.  The sampler's complete relation is all masks
            # that are actually available for this physical frame; uncovered
            # points remain outside eligible_indices and therefore unknown.
            continue
        frame = matches[0]
        for proposal_index in range(int(frame.proposal_offsets.size) - 1):
            start = int(frame.proposal_offsets[proposal_index])
            end = int(frame.proposal_offsets[proposal_index + 1])
            members = torch.unique(
                torch.as_tensor(
                    frame.proposal_point_indices[start:end], dtype=torch.long
                ),
                sorted=True,
            )
            relation_members.append(members)
            relation_offsets.append(
                relation_offsets[-1] + int(members.numel())
            )
    expected_relation_offsets = torch.tensor(relation_offsets, dtype=torch.long)
    expected_relation_members = torch.cat(relation_members, dim=0)
    if batch.relation_proposal_offsets is None or (
        batch.relation_proposal_member_indices is None
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: full relation CSR missing"
        )
    if not torch.equal(
        batch.relation_proposal_offsets.detach().cpu().long(),
        expected_relation_offsets,
    ) or not torch.equal(
        batch.relation_proposal_member_indices.detach().cpu().long(),
        expected_relation_members,
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: full relation CSR drift"
        )

    visible = torch.unique(
        torch.as_tensor(selected_frame.visible_indices, dtype=torch.long),
        sorted=True,
    )
    known = visible[
        torch.isin(visible, torch.unique(expected_relation_members, sorted=True))
    ]
    if batch.eligible_indices is None or not torch.equal(
        batch.eligible_indices.detach().cpu().long(), known
    ):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: known-support eligibility drift"
        )

    offsets = batch.proposal_member_offsets
    members = batch.proposal_member_indices
    pair_proposals = batch.pair_proposal_indices
    if offsets is None or members is None or pair_proposals is None:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: selected proposal CSR missing"
        )
    offsets = offsets.detach().cpu().long()
    members = members.detach().cpu().long()
    pair_proposals = pair_proposals.detach().cpu().long()
    labels = batch.proposal_labels.detach().cpu().long()
    payload_fields = (
        batch.token_positive_pair_offsets,
        batch.token_positive_pair_indices,
        batch.token_positive_pair_digest,
    )
    payload_presence = tuple(value is not None for value in payload_fields)
    pair_policy = record["batch"].get("native_dec0_pair_selection")
    if pair_policy != TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION:
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: fixed native-token pair policy drift"
        )
    if any(payload_presence):
        raise ValueError(
            f"{scene.scene_id}/{cell}/{frame_id}: backend token IDs were serialized"
        )
    for occurrence in range(int(offsets.numel()) - 1):
        rows = torch.where(pair_proposals == occurrence)[0]
        if rows.numel() == 0:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: selected occurrence has no row"
            )
        proposal_labels = torch.unique(labels[rows], sorted=True)
        if proposal_labels.numel() != 1:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: occurrence label drift"
            )
        proposal_index = int(proposal_labels[0])
        if not 0 <= proposal_index < int(selected_frame.proposal_offsets.size) - 1:
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: proposal index drift"
            )
        source_start = int(selected_frame.proposal_offsets[proposal_index])
        source_end = int(selected_frame.proposal_offsets[proposal_index + 1])
        expected_members = torch.unique(
            torch.as_tensor(
                selected_frame.proposal_point_indices[source_start:source_end],
                dtype=torch.long,
            ),
            sorted=True,
        )
        stored_members = members[int(offsets[occurrence]) : int(offsets[occurrence + 1])]
        if not torch.equal(stored_members, expected_members):
            raise ValueError(
                f"{scene.scene_id}/{cell}/{frame_id}: selected proposal CSR drift"
            )


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_fixed_eval_negative_policy(
    manifest: dict[str, Any], *, allow_shared_proposal_negatives: bool
) -> str:
    """Check policy before source admission; absent legacy flags mean veto."""
    policy = negative_policy_name(allow_shared_proposal_negatives)
    contract = manifest.get("token_pure_dec0_fixed_pack")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Fixed-eval token contract must be an object")
    token_policy = (contract or {}).get("allow_shared_proposal_negatives", False)
    saved_policy = manifest.get("allow_shared_proposal_negatives", token_policy)
    saved_name = negative_policy_name(saved_policy)
    if contract and "allow_shared_proposal_negatives" in contract:
        if negative_policy_name(token_policy) != saved_name:
            raise ValueError("Fixed-eval negative policy disagrees with its token contract")
    if saved_name != policy or manifest.get("negative_policy", policy) != policy:
        raise ValueError("Fixed-eval negative policy differs from the training recipe")
    return policy


def validate_fixed_holdout_eval_pack(
    *,
    pack_dir: Path,
    scenes: list[SceneSource],
    sampling: SamplingConfig,
    sampling_profile: str,
    seed: int,
    verify_scene_hashes: bool = True,
    cells: Sequence[str] | None = None,
    grid_size: float = 0.02,
    allow_shared_proposal_negatives: bool = False,
    sam2_maximum_raw_points: int = 500_000,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    policy = negative_policy_name(allow_shared_proposal_negatives)
    if allow_shared_proposal_negatives and active_cells != ("2d/sam2",):
        raise ValueError("Selected-proposal fixed evaluation requires SAM2")
    token_pure_fixed_pack = tuple(active_cells) == TWO_D_CELLS or (
        len(active_cells) == 1 and active_cells[0] in (*TWO_D_CELLS, "2d/sam2")
    )
    coordinate_policies = {
        str(scene.coordinate_normalization_policy) for scene in scenes
    }
    if len(coordinate_policies) != 1:
        raise ValueError(
            "Fixed-eval scenes must share one coordinate-normalization policy: "
            f"{sorted(coordinate_policies)}"
        )
    coordinate_policy = next(iter(coordinate_policies))
    manifest_path = (pack_dir / "manifest.json").resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if active_cells == ("2d/sam2",):
        validate_point_limit(manifest, sam2_maximum_raw_points)
    validate_fixed_eval_negative_policy(
        manifest, allow_shared_proposal_negatives=allow_shared_proposal_negatives
    )
    if manifest.get("schema_version") != FIXED_EVAL_PACK_SCHEMA:
        raise ValueError(f"Fixed-eval pack schema drift: {manifest_path}")
    if float(manifest.get("grid_size", 0.02)) != float(grid_size):
        raise ValueError(f"Fixed-eval grid-size drift: {manifest_path}")
    if manifest.get("sampling_profile") != sampling_profile:
        raise ValueError(f"Fixed-eval sampling-profile drift: {manifest_path}")
    if manifest.get("sampling") != vars(sampling):
        raise ValueError(f"Fixed-eval sampling contract drift: {manifest_path}")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"Fixed-eval seed drift: {manifest_path}")
    if manifest.get("coordinate_normalization_policy", "native") != coordinate_policy:
        raise ValueError(
            f"Fixed-eval coordinate-normalization drift: {manifest_path}"
        )
    manifest_cells = manifest.get("source_cells")
    if manifest_cells is None:
        manifest_cells = list(SOURCE_CELLS)
    if list(manifest_cells) != list(active_cells):
        raise ValueError(
            f"Fixed-eval source-cell drift: expected {list(active_cells)}, "
            f"got {list(manifest_cells)}"
        )
    token_contract = manifest.get("token_pure_dec0_fixed_pack")
    if token_pure_fixed_pack:
        if not isinstance(token_contract, dict) or token_contract != {
            "schema_version": TOKEN_PURE_FIXED_EVAL_SCHEMA,
            "enabled": True,
            "source_cells": list(active_cells),
            "sampler": "coverage_multigranular_v2",
            "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "sampler_preselection_policy": SAMPLER_PRESELECTION_POLICY,
            "frame_local": True,
            "relation_csr": (
                "same_physical_frame_all_"
                + "_".join(cell.split("/", 1)[1] for cell in active_cells)
                + "_masks"
            ),
            "eligible_indices": "visible_intersection_known_relation_members",
            "hierarchy_route_eligibility": (
                "not_required_for_dec0_feature_evaluation"
            ),
            "native_token_feature_candidates": (
                TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY
            ),
            "fixed_comparable_feature_hard_negatives": 0,
            "native_dec0_pair_selection": TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION,
            **({"allow_shared_proposal_negatives": True}
               if allow_shared_proposal_negatives else {}),
        }:
            raise ValueError(
                "V2 token-pure fixed-eval contract is absent or drifted; "
                "legacy dense packs are not token-safe"
            )
    expected_scenes = [
        {
            "scene_id": scene.scene_id,
            "source_manifest_sha256": scene.manifest_sha256,
        }
        for scene in scenes
    ]
    observed_scenes = [
        {
            "scene_id": row.get("scene_id"),
            "source_manifest_sha256": row.get("source_manifest_sha256"),
        }
        for row in manifest.get("scenes", [])
    ]
    if observed_scenes != expected_scenes:
        raise ValueError(f"Fixed-eval scene/source ordering drift: {manifest_path}")
    total_records = 0
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    for row in manifest["scenes"]:
        scene_id = str(row["scene_id"])
        scene = scene_by_id[scene_id]
        expected_file = f"{scene_id}.pt"
        if row.get("file") != expected_file:
            raise ValueError(
                f"Fixed-eval scene filename drift for {scene_id}: "
                f"{row.get('file')!r}"
            )
        scene_path = (pack_dir / row["file"]).resolve(strict=True)
        if verify_scene_hashes and sha256_file(scene_path) != row["sha256"]:
            raise ValueError(f"Fixed-eval scene artifact hash drift: {scene_path}")
        payload = torch.load(scene_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") != FIXED_EVAL_SCENE_SCHEMA
            or payload.get("scene_id") != scene_id
            or payload.get("source_manifest_sha256")
            != scene.manifest_sha256
            or int(payload.get("num_scene_points", -1))
            != int(scene.points.shape[0])
            or payload.get("coordinate_normalization_policy", "native")
            != scene.coordinate_normalization_policy
            or bool(payload.get("token_pure_dec0_fixed_pack", False))
            != token_pure_fixed_pack
        ):
            raise ValueError(f"Fixed-eval scene payload drift: {scene_path}")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Fixed-eval scene records missing: {scene_path}")
        expected_record_count = (
            int(row["record_count"])
            if token_pure_fixed_pack
            else _expected_holdout_record_count(scene, cells=active_cells)
        )
        if (
            len(records) != expected_record_count
            or int(row["record_count"]) != expected_record_count
        ):
            raise ValueError(f"Fixed-eval scene record-count drift: {scene_path}")
        observed_cells: set[str] = set()
        for record in records:
            cell = str(record.get("cell"))
            if cell not in active_cells:
                raise ValueError(
                    f"Fixed-eval scene contains an unknown cell: {scene_path}"
                )
            observed_cells.add(cell)
            batch = _triplet_batch_from_record(
                record["batch"],
                device=torch.device("cpu"),
                include_feature_hard=True,
                sampling=sampling,
                num_scene_points=int(scene.points.shape[0]),
            )
            if token_pure_fixed_pack:
                if batch.allow_shared_proposal_negatives != allow_shared_proposal_negatives:
                    raise ValueError("Fixed-eval record shared-proposal policy drift")
                if not bool(record["batch"].get("token_pure_fixed_pack", False)):
                    raise ValueError(
                        f"V2 fixed-eval record is not token-safe: {scene_path}"
                    )
                _validate_token_pure_fixed_record(
                    scene=scene,
                    record=record,
                    batch=batch,
                    cells=active_cells,
                )
        if observed_cells != set(active_cells):
            raise ValueError(f"Fixed-eval scene cell coverage drift: {scene_path}")
        total_records += expected_record_count
    checks = manifest.get("checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError(f"Fixed-eval pack checks failed: {manifest_path}")
    return {
        "schema_version": manifest["schema_version"],
        "path": str(pack_dir.resolve(strict=True)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sampling_profile": sampling_profile,
        "sampling": vars(sampling),
        "seed": int(seed),
        "coordinate_normalization_policy": coordinate_policy,
        "scene_count": len(expected_scenes),
        "record_count": total_records,
        "selection_view": "fixed_comparable",
        "fixed_negatives_per_pair": int(sampling.uniform_negatives)
        + int(sampling.spatial_hard_negatives),
        "adaptive_feature_hard_negatives_per_pair": int(
            sampling.feature_hard_negatives
        ),
        "token_pure_dec0_fixed_pack": token_contract,
        "allow_shared_proposal_negatives": allow_shared_proposal_negatives,
        "negative_policy": policy,
        "passed": True,
    }


def prepare_fixed_holdout_eval_pack(
    *,
    pack_dir: Path,
    scenes: list[SceneSource],
    sampling: SamplingConfig,
    sampling_profile: str,
    seed: int,
    cells: Sequence[str] | None = None,
    grid_size: float = 0.02,
    precomputed_token_plans: Mapping[str, Any] | None = None,
    allow_shared_proposal_negatives: bool = False,
    sam2_maximum_raw_points: int = 500_000,
) -> dict[str, Any]:
    active_cells = normalize_source_cells(cells)
    if active_cells == ("2d/sam2",):
        fragment_policy(sam2_maximum_raw_points)
        if any(len(scene.points) > sam2_maximum_raw_points for scene in scenes):
            raise ValueError("SAM2 fixed evaluation requires capped scene visits")
    policy = negative_policy_name(allow_shared_proposal_negatives)
    if allow_shared_proposal_negatives and active_cells != ("2d/sam2",):
        raise ValueError("Selected-proposal fixed evaluation requires SAM2")
    token_pure_fixed_pack = tuple(active_cells) == TWO_D_CELLS or (
        len(active_cells) == 1 and active_cells[0] in (*TWO_D_CELLS, "2d/sam2")
    )
    coordinate_policies = {
        str(scene.coordinate_normalization_policy) for scene in scenes
    }
    if len(coordinate_policies) != 1:
        raise ValueError(
            "Fixed-eval scenes must share one coordinate-normalization policy: "
            f"{sorted(coordinate_policies)}"
        )
    coordinate_policy = next(iter(coordinate_policies))
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.exists():
        return validate_fixed_holdout_eval_pack(
            pack_dir=pack_dir,
            scenes=scenes,
            sampling=sampling,
            sampling_profile=sampling_profile,
            seed=seed,
            cells=active_cells,
            grid_size=grid_size,
            allow_shared_proposal_negatives=allow_shared_proposal_negatives,
            sam2_maximum_raw_points=sam2_maximum_raw_points,
        )
    if pack_dir.exists() and any(pack_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite incomplete fixed-eval pack: {pack_dir}"
        )
    pack_dir.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, Any]] = []
    expected_pairs = (
        int(sampling.proposals_per_forward)
        * int(sampling.positive_pairs_per_proposal)
    )
    for scene in scenes:
        points = torch.from_numpy(scene.points)
        plan_audit: dict[str, Any] | None = None
        if token_pure_fixed_pack:
            plan = (
                precomputed_token_plans.get(scene.scene_id)
                if precomputed_token_plans is not None
                else None
            )
            if plan is None:
                plan = _token_pure_fixed_holdout_plan(
                    scene=scene,
                    points=points,
                    device=torch.device("cpu"),
                    sampling=sampling,
                    seed=seed,
                    cells=active_cells,
                    grid_size=grid_size,
                    allow_shared_proposal_negatives=allow_shared_proposal_negatives,
                )
            if {group.cell for group in plan.groups} != set(active_cells):
                raise ValueError(
                    f"{scene.scene_id}: supplied V2 fixed-eval plan does not "
                    "cover all selected source cells"
                )
            plan_audit = plan.audit_metadata()
            if any(group.batch.allow_shared_proposal_negatives != allow_shared_proposal_negatives
                   for group in plan.groups):
                raise ValueError("Supplied token plan shared-proposal policy drift")
            records = [
                {
                    "cell": group.cell,
                    "group": f"2d/{group.frame_id}",
                    "frame_id": group.frame_id,
                    "selected_proposal_indices": list(
                        group.selected_proposal_indices
                    ),
                    "coverage_metadata": dict(group.coverage_metadata),
                    "batch": _triplet_batch_record(
                        group.batch,
                        token_pure_fixed_pack=True,
                        fixed_runtime_token_pair_rebuild=True,
                    ),
                }
                for group in plan.groups
            ]
        else:
            batches = _holdout_batches(
                scene=scene,
                points=points,
                device=torch.device("cpu"),
                sampling=sampling,
                seed=seed,
                cells=active_cells,
            )
            records = [
                {
                    "cell": cell,
                    "group": group,
                    "batch": _triplet_batch_record(batch),
                }
                for cell, group, batch in batches
            ]
        cells_present = {str(record["cell"]) for record in records}
        if cells_present != set(active_cells):
            raise RuntimeError(
                f"{scene.scene_id}: fixed-eval cell coverage drift: {sorted(cells_present)}"
            )
        scene_path = pack_dir / f"{scene.scene_id}.pt"
        _atomic_torch_save(
            {
                "schema_version": FIXED_EVAL_SCENE_SCHEMA,
                "scene_id": scene.scene_id,
                "source_manifest_sha256": scene.manifest_sha256,
                "coordinate_normalization_policy": (
                    scene.coordinate_normalization_policy
                ),
                "num_scene_points": int(scene.points.shape[0]),
                "token_pure_dec0_fixed_pack": token_pure_fixed_pack,
                "token_pure_dec0_plan": plan_audit,
                "records": records,
            },
            scene_path,
        )
        scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "source_manifest_sha256": scene.manifest_sha256,
                "file": scene_path.name,
                "sha256": sha256_file(scene_path),
                "record_count": len(records),
            }
        )
    manifest = {
        "schema_version": FIXED_EVAL_PACK_SCHEMA,
        **({"sam2_maximum_raw_points": sam2_maximum_raw_points}
           if active_cells == ("2d/sam2",) else {}),
        "allow_shared_proposal_negatives": allow_shared_proposal_negatives,
        "negative_policy": policy,
        "sampling_profile": sampling_profile,
        "sampling": vars(sampling),
        "seed": int(seed),
        "coordinate_normalization_policy": coordinate_policy,
        "source_cells": list(active_cells),
        "token_pure_dec0_fixed_pack": (
            {
                "schema_version": TOKEN_PURE_FIXED_EVAL_SCHEMA,
                "enabled": True,
                "source_cells": list(active_cells),
                "sampler": "coverage_multigranular_v2",
                "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
                "sampler_preselection_policy": SAMPLER_PRESELECTION_POLICY,
                "frame_local": True,
                "relation_csr": (
                    "same_physical_frame_all_"
                    + "_".join(cell.split("/", 1)[1] for cell in active_cells)
                    + "_masks"
                ),
                "eligible_indices": (
                    "visible_intersection_known_relation_members"
                ),
                "hierarchy_route_eligibility": (
                    "not_required_for_dec0_feature_evaluation"
                ),
                "native_token_feature_candidates": (
                    TOKEN_PURE_FIXED_EVAL_FEATURE_CANDIDATE_POLICY
                ),
                "fixed_comparable_feature_hard_negatives": 0,
                "native_dec0_pair_selection": (
                    TOKEN_PURE_FIXED_EVAL_PAIR_SELECTION
                ),
                **({"allow_shared_proposal_negatives": True}
                   if allow_shared_proposal_negatives else {}),
            }
            if token_pure_fixed_pack
            else None
        ),
        "views": {
            "fixed_comparable": {
                "negative_identity": "byte-identical_across_representations",
                "uniform_negatives": int(sampling.uniform_negatives),
                "spatial_hard_negatives": int(sampling.spatial_hard_negatives),
                "feature_hard_negatives": 0,
            },
            "adaptive_feature_hard": {
                "base_negative_identity": "byte-identical_across_representations",
                "feature_candidate_identity": "byte-identical_across_representations",
                "selected_feature_hard_identity": "representation_specific",
                "feature_hard_negatives": int(sampling.feature_hard_negatives),
            },
        },
        "scenes": scene_rows,
        "checks": {
            "scene_order_exact": len(scene_rows) == len(scenes),
            "all_active_cells_per_scene": True,
            **(
                {"all_six_cells_per_scene": True}
                if tuple(active_cells) == SOURCE_CELLS
                else {}
            ),
            **(
                {
                    (
                        "token_pure_three_cells_exact" if len(active_cells) == 3
                        else "token_pure_selected_cells_exact"
                    ): True,
                    "frame_local_relation_csr_recorded": True,
                    "known_support_eligibility_recorded": True,
                }
                if token_pure_fixed_pack
                else {}
            ),
            "positive_pairs_per_record_exact": expected_pairs > 0,
            "base_negative_count_exact": (
                int(sampling.uniform_negatives)
                + int(sampling.spatial_hard_negatives)
                > 0
            ),
            "feature_candidate_pool_recorded": (
                int(sampling.feature_candidate_pool) > 0
            ),
        },
    }
    if float(grid_size) != 0.02:
        manifest["grid_size"] = float(grid_size)
    _atomic_json_write(manifest, manifest_path)
    return validate_fixed_holdout_eval_pack(
        pack_dir=pack_dir,
        scenes=scenes,
        sampling=sampling,
        sampling_profile=sampling_profile,
        seed=seed,
        cells=active_cells,
        grid_size=grid_size,
        allow_shared_proposal_negatives=allow_shared_proposal_negatives,
        sam2_maximum_raw_points=sam2_maximum_raw_points,
    )


def _fixed_scene_batch_views(
    *,
    pack_dir: Path,
    scene: SceneSource,
    device: torch.device,
    sampling: SamplingConfig,
) -> dict[str, list[tuple[str, str, ContrastiveTripletBatch]]]:
    scene_path = pack_dir / f"{scene.scene_id}.pt"
    payload = torch.load(scene_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != FIXED_EVAL_SCENE_SCHEMA
        or payload.get("scene_id") != scene.scene_id
        or payload.get("source_manifest_sha256") != scene.manifest_sha256
        or int(payload.get("num_scene_points", -1)) != int(scene.points.shape[0])
        or payload.get("coordinate_normalization_policy", "native")
        != scene.coordinate_normalization_policy
    ):
        raise ValueError(f"Fixed-eval scene payload drift: {scene_path}")
    output: dict[str, list[tuple[str, str, ContrastiveTripletBatch]]] = {
        "fixed_comparable": [],
        "adaptive_feature_hard": [],
    }
    for record in payload["records"]:
        for view, include_feature_hard in (
            ("fixed_comparable", False),
            ("adaptive_feature_hard", True),
        ):
            output[view].append(
                (
                    str(record["cell"]),
                    str(record["group"]),
                    _triplet_batch_from_record(
                        record["batch"],
                        device=device,
                        include_feature_hard=include_feature_hard,
                        sampling=sampling,
                        num_scene_points=int(scene.points.shape[0]),
                    ),
                )
            )
    return output
