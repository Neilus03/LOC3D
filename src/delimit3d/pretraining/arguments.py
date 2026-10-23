"""Pretraining arguments extracted from the executed trainer."""

from __future__ import annotations

import argparse


from pathlib import Path


from typing import Sequence


from delimit3d.training.contracts import SOURCE_CELL_SCHEDULE_CHOICES, SOURCE_CELL_SCHEDULE_FIXED, SOURCE_FAMILY_CHOICES

from delimit3d.training.adaptation import INPUT_FEATURE_CHOICES, HIERARCHY_QUERY_PROJECTION_POLICIES, MULTISCALE_AUXILIARY_WEIGHT, MULTISCALE_AUXILIARY_WARMUP_EPOCHS, SAMPLING_PROFILES, TRAIN_AUGMENTATION_CHOICES


from delimit3d.training.sam2_rank_balance import STRICT as STRICT_RANK_BALANCE, ROTATING as ROTATING_RANK_BALANCE


from .engine_settings import (
    BACKBONE_LR_CHOICES,
    COORDINATE_NORMALIZATION_POLICIES,
    HIERARCHY_SUPERVISION_MODES,
    NORMALIZATION_POLICIES,
    TRAINING_SAMPLING_MODES,
)


def _comma_separated_nonnegative_ints(value: str) -> tuple[int, ...]:
    """Parse a sorted, unique epoch list while keeping CLI provenance explicit."""

    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integer epochs, got {value!r}"
        ) from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("Epoch lists must contain non-negative values")
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("Epoch lists must be sorted and unique")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML/JSON recipe; explicit CLI flags override it.")
    parser.add_argument(
        "--allow-shared-proposal-negatives", action=argparse.BooleanOptionalAction,
        default=False,
        help="SAM2 dec0: allow known, pure negatives outside the selected mask despite another shared mask. Disable to use overlap veto.",
    )
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--sam2-maximum-raw-points", type=int, choices=(100_000, 500_000), default=500_000,
                        help="SAM2 training fragment limit; fixed evaluation stays at 500k.")
    parser.add_argument("--flush-logical-batch-at-eval", action="store_true",
                        help="SAM2 only: flush an actual-size tail at exact evaluation/checkpoint epochs.")
    parser.add_argument(
        "--source-granularity", choices=("g02", "g05", "g08", "sam2"), default=None,
        help="Select one 2D mask source for dec0-only V2 pretraining.",
    )
    parser.add_argument(
        "--dataset",
        choices=("scannet", "structured3d"),
        default="scannet",
        help="Dataset contract used for split validation and source auditing.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=None,
        help=(
            "Required DDP world size. Defaults to the frozen ScanNet value "
            "when omitted."
        ),
    )
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--litept-variant", choices=("litept_s_star", "litept_b_star"), default="litept_s_star")
    parser.add_argument("--rank-balance-policy", choices=(STRICT_RANK_BALANCE, ROTATING_RANK_BALANCE),
                        default=STRICT_RANK_BALANCE)
    parser.add_argument(
        "--input-features",
        choices=INPUT_FEATURE_CHOICES,
        default="rgbn6",
        help="Input feature contract: RGB (3) or RGB+normal (6).",
    )
    parser.add_argument(
        "--train-augmentation-profile",
        choices=TRAIN_AUGMENTATION_CHOICES,
        default="none",
        help=(
            "Training-only input augmentation. Evaluation is always "
            "deterministic and unaugmented."
        ),
    )
    parser.add_argument(
        "--coordinate-normalization-policy",
        choices=COORDINATE_NORMALIZATION_POLICIES,
        default="native",
        help=(
            "Deterministic source-frame conversion applied to points and normals "
            "before both training augmentation and unaugmented evaluation."
        ),
    )
    parser.add_argument(
        "--required-crop-mode",
        choices=("any", "sphere_50k", "full_scene"),
        default="any",
        help="Fail unless every loaded source manifest uses this crop mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        choices=(3, 18, 34, 36, 84, 150, 168, 252, 336, 1008),
        required=True,
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        choices=BACKBONE_LR_CHOICES,
        required=True,
    )
    parser.add_argument("--head-lr", type=float, default=3e-3)
    parser.add_argument(
        "--representative-sampling",
        choices=("first", "random"),
        default="first",
        help="Representative point policy for deterministic transfer parity.",
    )
    parser.add_argument(
        "--color-mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="RGB mean applied after RGBN6 augmentation, in [0,1] space.",
    )
    parser.add_argument(
        "--color-std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="RGB standard deviation applied after RGBN6 augmentation.",
    )
    parser.add_argument(
        "--multiscale-supervision",
        action="store_true",
        help=(
            "Add disposable dec3/dec2/dec1 contrastive heads while retaining "
            "the final dec0 objective as the primary loss."
        ),
    )
    parser.add_argument(
        "--multiscale-loss-weight",
        type=float,
        default=MULTISCALE_AUXILIARY_WEIGHT,
        help="Maximum multiplier for the tapered auxiliary multi-scale sum.",
    )
    parser.add_argument(
        "--multiscale-warmup-epochs",
        type=int,
        default=MULTISCALE_AUXILIARY_WARMUP_EPOCHS,
        help="Linearly ramp the auxiliary loss over this many scene epochs.",
    )
    parser.add_argument(
        "--hierarchy-supervision",
        choices=HIERARCHY_SUPERVISION_MODES,
        default="off",
        help=(
            "Opt-in V2 supervision recipe. 'legacy' must accompany "
            "--multiscale-supervision; query modes expose enc4 through dec0 "
            "and use proposal-conditioned occupancy supervision."
        ),
    )
    parser.add_argument("--hierarchy-mask-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--hierarchy-mask-stage-weights",
        default="{}",
        help=(
            "Optional JSON mapping of relative proposal-query weights for "
            "enc4/dec3/dec2/dec1; omitted stages default to 1.0."
        ),
    )
    parser.add_argument(
        "--hierarchy-mask-stage-weight-schedule",
        default="null",
        help=(
            "Optional JSON object {transition_fraction,early,late} for a "
            "deterministic hierarchy-mask stage-weight transition."
        ),
    )
    parser.add_argument(
        "--hierarchy-query-projection-policy",
        choices=HIERARCHY_QUERY_PROJECTION_POLICIES,
        default="shared_dec0",
        help=(
            "Detached dec0 query-head policy: the legacy shared head or "
            "a disposable target-stage-specific head."
        ),
    )
    parser.add_argument("--hierarchy-vicreg-loss-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--hierarchy-variance-target", type=float, default=1.0)
    parser.add_argument("--hierarchy-covariance-weight", type=float, default=0.04)
    parser.add_argument("--hierarchy-vicreg-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--feature-hard-warmup-fraction",
        type=float,
        default=0.05,
        help="Minimum update fraction before V2 feature-hard mining may activate.",
    )
    parser.add_argument("--feature-hard-gap-threshold", type=float, default=0.05)
    parser.add_argument("--feature-hard-required-panels", type=int, default=2)
    parser.add_argument(
        "--normalization-policy",
        choices=NORMALIZATION_POLICIES,
        default="batchnorm",
        help="Backbone normalization policy; SyncBN conversion occurs after initial load.",
    )
    parser.add_argument(
        "--bn-recalibration-scenes",
        type=int,
        default=0,
        help=(
            "Number of fixed training scenes used by the retained-checkpoint BN "
            "recalibration sidecar contract. Zero preserves the legacy runner."
        ),
    )
    parser.add_argument(
        "--evaluation-epochs",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help="Optional sorted epoch labels for fixed health evaluation.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Optional sorted retained epoch labels. Unaligned labels are saved as "
            "non-resumable update-aligned audit snapshots."
        ),
    )
    parser.add_argument(
        "--audit-snapshot-updates",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Optional update-aligned health-panel snapshots. These never alter "
            "the scene stream and may therefore carry an approximate epoch label."
        ),
    )
    parser.add_argument(
        "--audit-snapshot-labels",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help="Epoch-style labels paired one-to-one with --audit-snapshot-updates.",
    )
    parser.add_argument(
        "--checkpoint-updates",
        type=_comma_separated_nonnegative_ints,
        default=None,
        help=(
            "Subset of audit snapshot updates whose raw state is retained as a "
            "non-resumable checkpoint."
        ),
    )
    parser.add_argument(
        "--maximum-updates",
        type=int,
        default=None,
        help=(
            "Optional exact update horizon for short funnel panels. It may stop "
            "before the nominal final scene boundary but never executes a partial "
            "logical batch."
        ),
    )
    parser.add_argument(
        "--pause-after-update",
        type=int,
        default=None,
        help=(
            "Operational V2 production pause at an exact resumable epoch "
            "boundary. Unlike --maximum-updates, this does not alter the "
            "declared optimization horizon or LR/supervision schedules."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=17)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Training/sampling seed. Fixed-evaluation and initialization seeds "
            "default to this value unless explicitly separated below."
        ),
    )
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=None,
        help=(
            "Seed declared by the immutable initialization checkpoint. This "
            "may differ from --seed for a controlled training-seed repeat."
        ),
    )
    parser.add_argument(
        "--fixed-eval-seed",
        type=int,
        default=None,
        help=(
            "Seed declared by the immutable fixed holdout pack. This may "
            "differ from --seed for a controlled training-seed repeat."
        ),
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--sampling-profile",
        choices=tuple(sorted(SAMPLING_PROFILES)),
        default="pf_hnm_512_v1",
        help="Frozen training positive/negative sampling contract.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=TRAINING_SAMPLING_MODES,
        default="random_with_replacement",
        help=(
            "Training tuple scheduler. V2 coverage visits all three 2D "
            "granularities with frame-local, checkpoint-reproducible groups."
        ),
    )
    parser.add_argument(
        "--evaluation-sampling-profile",
        choices=tuple(sorted(SAMPLING_PROFILES)),
        default=None,
        help=(
            "Optional holdout sampling contract. Defaults to the training "
            "profile; the matched p16 arm pins pf_hnm_512_v1 so evaluation "
            "remains comparable to earlier runs."
        ),
    )
    parser.add_argument(
        "--scenes-per-rank-per-update",
        type=int,
        default=1,
        help=(
            "Number of single-scene forwards accumulated on each DDP rank "
            "before one optimizer update."
        ),
    )
    parser.add_argument(
        "--source-loading-mode",
        choices=("preload", "logical_batch"),
        default="preload",
        help=(
            "Keep every rank-local training source resident, or load only the "
            "current logical scene group from a pre-audited source archive."
        ),
    )
    parser.add_argument(
        "--source-loader-workers",
        type=int,
        default=1,
        help="Parallel source-loader threads used by logical_batch mode.",
    )
    parser.add_argument(
        "--source-cell-schedule",
        choices=SOURCE_CELL_SCHEDULE_CHOICES,
        default=SOURCE_CELL_SCHEDULE_FIXED,
        help=(
            "Per-scene source-cell ordering. The balanced-random schedule "
            "uses one reproducible permutation per complete cell cycle."
        ),
    )
    parser.add_argument(
        "--fixed-eval-pack",
        type=Path,
        default=None,
        help=(
            "Directory containing or receiving the immutable heldout tuple pack. "
            "Required for pf_hnm_768_v1."
        ),
    )
    parser.add_argument(
        "--code-manifest",
        type=Path,
        default=None,
        help=(
            "Hash manifest for the immutable code snapshot used by this run. "
            "Required for pf_hnm_768_v1."
        ),
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=None,
        help=(
            "Optional frozen dataset source-audit JSON. The ScanNet contract "
            "uses source_root/../audits/source_dataset_audit.json by default; "
            "Structured3D currently validates each scene manifest directly."
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument(
        "--resume-sha256",
        type=str,
        default=None,
        help="Required content digest for --resume.",
    )
    parser.add_argument(
        "--resume-into-fresh-attempt",
        action="store_true",
        help=(
            "Continue a hash-pinned checkpoint in a new output directory, "
            "preserving the parent attempt unchanged."
        ),
    )
    parser.add_argument(
        "--execution-provenance",
        type=Path,
        default=None,
        help="Frozen pre-run trust-root manifest embedded in every checkpoint.",
    )
    parser.add_argument(
        "--execution-provenance-sha256",
        type=str,
        default=None,
        help="Required SHA-256 of --execution-provenance.",
    )
    parser.add_argument(
        "--extension-start-epoch",
        type=int,
        default=None,
        help=(
            "Start a new warm-restarted cosine phase from a completed resume "
            "checkpoint at this global epoch."
        ),
    )
    parser.add_argument(
        "--extension-parent-checkpoint",
        type=Path,
        default=None,
        help=(
            "Pinned completed checkpoint that begins the extension lineage. "
            "For the first extension launch this must equal --resume; retain "
            "the same value when exactly resuming an interrupted extension."
        ),
    )
    parser.add_argument(
        "--three-d-mode",
        choices=("gt3", "g08_repeated"),
        default="gt3",
        help=(
            "Use g02/g05/g08 in their three logical 3D slots, or repeat exact "
            "g08 GT supervision in all three slots for the matched control. "
            "Forbidden when --source-family 2d."
        ),
    )
    parser.add_argument(
        "--source-family",
        choices=SOURCE_FAMILY_CHOICES,
        default="2d3d",
        help=(
            "Active source cells. 2d3d is the six-cell default; 2d trains and "
            "evaluates only UnSAM 2d/g02, 2d/g05, and 2d/g08."
        ),
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=None,
        help=(
            "Explicit four-rank integration smoke: repeat one audited scene on "
            "every rank and evaluate that scene only."
        ),
    )
    parser.add_argument(
        "--smoke-manifest-list",
        type=Path,
        default=None,
        help=(
            "Text file containing at least two audited scene manifests. Every "
            "rank cycles through the listed scenes, exercising scene-changing "
            "voxelization inside one process."
        ),
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        choices=(2, 6),
        default=2,
        help=(
            "Number of multi-scene smoke epochs. Six exercises every source "
            "cell exactly once per scene and rank."
        ),
    )
    from delimit3d.training.pretraining_config import parse_with_config

    args = parse_with_config(parser, argv)
    if args.sam2_maximum_raw_points != 500_000 and args.source_granularity != "sam2":
        parser.error("SAM2 training point limits require source_granularity=sam2")
    return args


def _resolve_seed_contract(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve immutable-artifact seeds while preserving legacy defaults.

    The training/sampling seed is intentionally kept in ``args.seed``.  A
    controlled seed repeat can retain the exact initialization checkpoint and
    fixed holdout pack by supplying their declared seeds separately.
    """

    initialization_seed = (
        int(args.seed)
        if args.initialization_seed is None
        else int(args.initialization_seed)
    )
    fixed_eval_seed = (
        int(args.seed)
        if args.fixed_eval_seed is None
        else int(args.fixed_eval_seed)
    )
    if initialization_seed < 0 or fixed_eval_seed < 0 or int(args.seed) < 0:
        raise ValueError("Seeds must be non-negative")
    return initialization_seed, fixed_eval_seed
