# 31 October 2026 code update

This update brings reusable visualization, frozen-feature diagnostics and sampling
ablation code into the public LOC3D package. It builds on GitHub main at
`fa26a9a17174852cdd418697a58abbd8f1dc3bdc`, including the latest README edits and
the previously released Mask3D limited-supervision workflows.

## Investigation and scope

The audit compared 778 source, script, configuration, test and documentation
files in the research checkout with the public release. It also inventoried
149 Python scripts in experiment directories dated 25–31 October. A file absent
from the release is not necessarily missing functionality: public training and
task engines were reorganized during the earlier release, while cluster
orchestration and generated experiment outputs were deliberately excluded.

The largest gaps were the saved-input visualization suite and the sampling
ablation implementation, which lived in a sealed experiment snapshot rather
than the active research package. The release already contained logical batch
scheduling, granularity helpers, both SAM2 negative policies, and the low-data
and official low-annotation Mask3D workflows. Those implementations are retained.

| Area | Public update | Boundary |
| --- | --- | --- |
| Visualization | Manifest-driven saved-feature and recorded-task galleries, point/mesh views, PCA, query affinity and export checks | User-supplied data and checkpoints; synthetic fixtures only demonstrate software |
| Feature evaluation | Common-grid feature pooling, GT-query cosine diagnostics, GT-count spherical K-means and ScanNet20/200 target preparation | Custom frozen-feature diagnostics; not instance AP |
| Sampling ablations | Matched negative budgets, positive-only small-mask exclusion, recipe preparation and checkpoint checks | Code and CPU verification; no new ablation scores |
| Packaging | Public command/documentation entry points, plotting dependencies in CPU CI and packaged examples | Native GPU execution remains a separate validation step |

See the [visualization guide](visualization.md),
[feature-diagnostic guide](feature_diagnostics.md), and
[sampling-ablation guide](sampling_ablations.md) for input contracts and commands.

## Scientific identities preserved

The sampling study retains the original 150-epoch learning-rate horizon while
stopping at epoch 50. Its requested negative total is 384 before feature-hard
activation and 512 afterward. Removing a negative family replaces its budget
with uniform negatives; candidate scarcity can still reduce delivered counts.
The positive-area exclusion retains small-mask membership for negative
eligibility. Overlap veto changes negative eligibility without deleting
overlapping proposals. The study fixes feature-hard activation at the reference's
observed update 6,106; the historical reference used an adaptive gate.
An epoch-5 checkpoint precedes that activation and cannot measure its effect.

Frozen-feature comparisons record input grid, evaluation grid, original-point
mapping, normal provenance, feature normalization and checkpoint identity.
GT-count clustering and GT-selected queries use annotation information and
must be labeled accordingly. Hungarian matching scores predictions after
clustering; ground truth must not choose the clustering restart. PCA colors
show a feature projection; query similarity provides more direct evidence
about which points share a representation.

The 2 cm foreground-only clustering pilot is a different experiment from the
full-scene common-grid diagnostic. Foreground filtering, geometry-graph
postprocessing, normal orientation, feature stage and per-stage normalization
each change the protocol. The configurable tools do not make these experiments
interchangeable, or reproduce their historical scores automatically.
The newer full-validation experiment also has a separate ScanNet20 object-only
target set that excludes wall/floor. The released ScanNet20 diagnostic retains
its existing taxonomy and does not claim to reproduce that object-only panel.

## Research work kept outside supported public workflows

- The AGILE3D official sparse-annotation prototype uses a new observed-only
  click policy and separate training integration. The audited prototype's
  readiness notes do not establish a portable production workflow. The public
  Mask3D low-annotation implementation remains documented separately.
- Utonia native model integration, graph-clustering pilots, nested-mask analyses
  and Articulate3D part-similarity experiments still depend on experiment-specific
  inputs or runners. Reusable saved-feature analysis does not imply those model
  or experiment integrations are supported.
- Scheduler submissions, machine-specific staging, queue recovery, dashboards,
  private paper layouts and generated figures remain local. Their existence is
  recorded in the audit but they are not runtime dependencies of this release.
- Datasets, annotations, trained checkpoints and unfinished result tables are
  not bundled. The existing downloadable epoch-150 encoder keeps its identity.

## Verification and provenance

The [refresh manifest](../manifests/code_refresh_20261031.json) records the base
revision, changed-file hashes, audit scope and validation results. Historical
migration/refactor manifests remain dated records of their original snapshots;
they do not assert that later modified files retain those original hashes.

Validation covers CPU scientific contracts, saved-input rendering and public
interfaces. No training or evaluation jobs are submitted by this release
update, and it makes no new benchmark or ablation performance claim. The active
research checkout and its running experiments are preserved.
