"""Matched negative budgets and validation for SAM2 sampling ablations."""
from dataclasses import dataclass

VARIANTS = ('full', 'no_hard', 'no_feature_hard', 'overlap_veto', 'no_small_positive')
# The reference seed-42 full run enabled the feature-hard branch at this
# exact update, immediately after the epoch-25 validation panel.  This fixed
# control schedule matches its observed execution even if an ablation's
# validation gap would have opened its own adaptive gate at another time.
REFERENCE_FEATURE_HARD_ACTIVATION_UPDATE = 6106


@dataclass(frozen=True)
class NegativeBudget:
    uniform: int
    spatial: int
    feature: int

    @property
    def total(self):
        return self.uniform + self.spatial + self.feature


def negative_budget(variant: str, *, update: int,
                    activation_update: int = REFERENCE_FEATURE_HARD_ACTIVATION_UPDATE) -> NegativeBudget:
    if (variant not in VARIANTS or type(update) is not int or update < 0
            or type(activation_update) is not int or activation_update < 1):
        raise ValueError('invalid sampling-ablation variant or update')
    feature_active = update >= activation_update
    if variant == 'no_hard':
        budget = NegativeBudget(512 if feature_active else 384, 0, 0)
    elif variant == 'no_feature_hard':
        budget = NegativeBudget(384 if feature_active else 256, 128, 0)
    else:
        budget = NegativeBudget(256, 128, 128 if feature_active else 0)
    if budget.total != (512 if feature_active else 384):
        raise AssertionError('ablation changed the requested negative budget')
    return budget


def ablation_contract(args) -> dict:
    """Only changed variants add metadata; historical full runs remain compatible."""
    variant = getattr(args, "ablation_variant", "full")
    if variant == "full":
        return {}
    from delimit3d.runtime.records import file_sha256
    audit = getattr(args, "proposal_area_audit", None)
    return {
        "ablation_variant": variant,
        "ablation_proposal_area_cutoff": args.proposal_area_cutoff,
        "ablation_area_audit_sha256": file_sha256(audit) if audit is not None else None,
        "ablation_activation_update": args.ablation_activation_update,
        "ablation_training_allow_shared_negatives": variant != "overlap_veto",
        "ablation_fixed_evaluation_allow_shared_negatives": True,
        "ablation_gate_policy": "fixed_to_reference_actual_activation",
    }


def validate_ablation_resume(previous, current):
    """Missing metadata denotes an ordinary full run, never a controlled arm."""
    before, after = previous.get("ablation_variant", "full"), current.get("ablation_variant", "full")
    if before not in VARIANTS or after not in VARIANTS or before != after:
        raise ValueError("Sampling ablation differs on resume; start a separate run")
    if before != "full":
        keys = ("ablation_proposal_area_cutoff", "ablation_area_audit_sha256", "ablation_activation_update",
                "ablation_training_allow_shared_negatives", "ablation_fixed_evaluation_allow_shared_negatives",
                "ablation_gate_policy")
        if any(key not in previous or key not in current or previous[key] != current[key] for key in keys):
            raise ValueError("Sampling ablation contract differs on resume")


def validate_ablation_args(args) -> None:
    """Fail before initializing DDP; the original recipe remains unchanged."""
    variant = args.ablation_variant
    negative_budget(variant, update=0, activation_update=args.ablation_activation_update)
    if variant != "full" and (
        args.source_granularity != "sam2" or not args.allow_shared_proposal_negatives
        or args.sampling_profile != "pf_hnm_512_p16_v1"
        or args.sampling_mode != "coverage_multigranular_v2"
        or args.hierarchy_supervision != "off" or args.multiscale_supervision
    ):
        raise ValueError("Sampling ablations require the selected-proposal SAM2 dec0 recipe")
    if variant == "no_small_positive":
        import json
        from delimit3d.runtime.records import file_sha256
        if args.proposal_area_cutoff is None or args.proposal_area_cutoff < 1 or args.proposal_area_audit is None:
            raise ValueError("no_small_positive requires a positive cutoff and its training area audit")
        audit = json.loads(args.proposal_area_audit.read_text())
        if (audit.get("schema") != "loc3d_train_proposal_area_audit/v1"
                or audit.get("passed") is not True
                or audit.get("threshold_pixels_inclusive") != args.proposal_area_cutoff
                or audit.get("train_split_sha256") != file_sha256(args.train_split)
                or audit.get("source_audit_sha256") != file_sha256(args.source_audit)):
            raise ValueError("Proposal-area audit does not match the cutoff or training inputs")
    elif args.proposal_area_cutoff is not None or args.proposal_area_audit is not None:
        raise ValueError("Proposal-area filtering is reserved for no_small_positive")
