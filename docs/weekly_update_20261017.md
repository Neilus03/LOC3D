# Development update: 11–17 October 2026

The reproducible GitHub release was merged on 12 October 2026 at
`ea5dfe3cf34596e57f6e4cb7164c0aaa5f99c0cb`. Its training implementation already
contains the two SAM2 negative policies, explicit inference normalization,
strict checkpoint loading and recovery, native Mask3D/AGILE3D adapters, and
complete single-object and multi-object evaluation. The main gaps since that
release were a usable inference entry point, model selection, and this week's
experiment evidence. The original research checkout and running experiments
remain separate from this release.

The [evidence snapshot](weekly_update_20261017.json) records the source-report
hashes and exact transfer values inspected on 17 October. It summarizes existing
verification records; these benchmarks were not rerun as part of repository
maintenance. Source paths in that snapshot refer to the preserved research
checkout, whose generated outputs and machine-specific launchers are not bundled.

## Models and evaluation contracts

- **Historical UnSAMv2 pretrained LitePT-S*.** The completed RGBN6 hierarchy
  pretraining used a 5 cm grid. The epoch-150 encoder initialized the historical
  Mask3D and AGILE3D task models reported below. These are distinct from SAM2.
- **SAM2 overlap veto and selected-proposal negatives.** These newer LitePT-S*
  recipes use a 2 cm pretraining grid, dec0 supervision and no hierarchy/VICReg
  loss. Downstream task recipes use 5 cm. Comparing them with the older 5 cm
  hierarchy recipe changes several factors; it does not isolate teacher quality,
  grid size or a single loss change.
- **Task checkpoints.** Automatic segmentation needs a full trained Mask3D
  checkpoint, and interactive segmentation needs a full trained AGILE3D task
  checkpoint or a historical decoder plus its exact original encoder. An exported
  encoder alone supports features/probing, not either trained task head.
- **Inference normalization.** The corrected LitePT-S* task policy is
  `train_no_update`: current-scene BatchNorm statistics, stochastic layers in
  evaluation mode, no optimizer, and unchanged saved buffers. Saved-statistics
  evaluation is a separate protocol. See [task contracts](tasks.md).

## Completed automatic segmentation

All rows use the completed epoch-600 task checkpoint, all 312 ScanNet validation
scenes, and the same recorded decoder initialization. Values are percentages of
the custom **class-agnostic raw-query CA-AP**, not semantic ScanNet leaderboard AP.
The pretrained rows use the historical UnSAMv2 epoch-150 encoder initialization.

| Arm | CA-AP | CA-AP50 | CA-AP25 |
| --- | ---: | ---: | ---: |
| Scratch | 57.83 | 78.33 | 90.35 |
| Pretrained, frozen encoder | 25.79 | 49.56 | 78.20 |
| Pretrained, encoder LR multiplier 0.1 | 54.06 | 75.94 | 89.49 |
| Pretrained, equal encoder/decoder LR | 59.21 | 80.02 | 91.16 |

These are explicit recipe comparisons. Differing learning-rate arms must not be
combined into an initialization-only causal estimate. The task remains
class-agnostic; semantic labeling is a separate stage.

## Completed matched interactive segmentation

The corrected equal-LR single-object comparison is complete at epoch 1100:
312 validation scenes, 10,357 objects, and 217,497 trace rows per arm covering
zero through twenty clicks. Training checks match seed, architecture, optimizer,
decoder initialization, runtime and data; the varying factor is historical
UnSAMv2 epoch-150 encoder initialization versus scratch. Both evaluations use
the same `train_no_update` policy and RTX 4090 hardware.

| Mean object IoU | Scratch | Pretrained | Difference |
| --- | ---: | ---: | ---: |
| One click | 64.36% | 65.71% | +1.35 points |
| Two clicks | 72.48% | 73.08% | +0.60 points |
| Five clicks | 81.17% | 81.22% | +0.05 points |
| Twenty clicks | 86.20% | 86.25% | +0.05 points |

The paired base-scene bootstrap interval for the one-click difference is
0.95–1.73 points; the five-click interval includes zero. This supports an early
click improvement for this single training seed, with similar late-click IoU.
It does not measure training-seed variability or establish a newer SAM2 result.

## The one-percent experiment

The current four arms are scratch LitePT-S*, public Sonata-small, public
Concerto-small, and SAM2 overlap-veto LitePT-S* initialized from epoch 100.
They use the same deterministic 12-scene subset, fresh shared decoder core,
5 cm downstream grid, 600 epochs, and equal peak encoder/decoder LR of `1e-4`.
Each epoch visits the 12 scenes once, in batches of 5, 5 and 2: **1,800 total
optimizer updates**, with OneCycleLR stepping once per update. It does not
repeat those scenes to preserve the full-data update count.

All four GPU smoke and continuation checks have completed. At this snapshot the
four production jobs were pending and the production run directories contained
no training metrics; **there are no final low-data results**. Queue status is
dated evidence, not a live status feed. This release does not submit or modify
those experiments. Different architectures remain a confound in comparisons
with Sonata/Concerto.

One portable failure found by this experiment is fixed here: native
scratch/pretraining checkpoints can contain `criterion.log_temperature` beside
`backbone.*` and `projector.*`. Mask3D initialization now validates that finite
scalar and excludes it with the projector. All encoder tensors still match
strictly; task decoder state, duplicate normalized keys, non-tensor state and
invalid temperature values are rejected.

## Semantic labels after instance segmentation

This week's completed work shows that semantic labels can be attached separately
to fixed predicted instance masks. A supervised 18-class affine classifier uses
frozen multi-view RGB crop features, optionally combined with frozen 3D object
descriptors. It uses calibrated RGB-D frames, poses and intrinsics; the RGB
feature procedure is not available from an arbitrary point cloud alone.

The train-selected RGB+3D classifier reaches 92.22% accuracy on all 4,329 eligible
GT objects, and 95.33% on the 2,658 predicted masks geometrically matched at
IoU >=0.75. That matched subset covers 61.40% of GT objects. A classifier trained
on predicted training masks also reaches 95.33% on the unchanged matched
validation subset. Both completed verifiers cover 1,201 training and 312
validation scenes. This is supervised, closed-set recognition; conditional
accuracy does not count missed objects or establish open-vocabulary performance.

For the GT-mask-trained RGB+3D head, changing labels to the GT-overlap reference
while preserving every predicted mask and score changes semantic AP from 43.31
to 46.17 under the classifier-weighted score rule. This is a fixed-output
labeling diagnostic, not an optimal AP ceiling. Its prediction filtering and
semantic averaging differ from raw-query CA-AP; do not subtract the two metrics.

The release inference UI should first expose class-agnostic masks and interactive
object selection. Semantic labeling remains a documented separate stage until
the RGB-D inputs, classifier weights, frozen image dependencies and supported
label vocabulary are packaged as a portable workflow. The experimental scripts,
background fusion pilots and generated figures are retained in the research
checkout rather than advertised as a ready inference feature.

## Release scope

Reusable code, model/checkpoint contracts, targeted tests and compact evidence
belong in this repository. Scheduler recovery, dashboards, personal host paths,
large checkpoints, raw datasets and generated report trees remain external.
Completion markers and exact checkpoint identities take precedence over stale
handoff files that still describe earlier queued work.
