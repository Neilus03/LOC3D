# LOC3D project and release status

**Updated 31 October 2026.** Version 0.4.0 adds the public visualization suite,
common-grid frozen-feature diagnostics and controlled sampling-ablation tools.
The [code update](docs/release_update_20261031.md) describes the investigation,
supported workflows and deferred research integrations. Its
[verification manifest](manifests/code_refresh_20261031.json) records the current
CPU checks and file identities. This is a software update, with no new GPU
benchmark or ablation performance claim.

The 25 October update added the [Mask3D limited-supervision workflows](docs/limited_supervision.md),
including matched low-data experiments and official sparse annotations. The
AGILE3D sparse-annotation prototype remains a separate research implementation.

LOC3D is the current public name for the project. The Python package and commands remain `delimit3d`; existing environment variables, default artifact locations, checkpoints and historical run identities retain their compatibility names.

The [research direction](docs/research.md) describes the current selected-proposal SAM2 method, open questions and the distinction between native-system comparisons and controlled initialization comparisons. The [23 October evidence update](docs/research_update_20261023.md) and its [hashed snapshot](docs/research_update_20261023.json) supersede the 17 October research-status snapshot where newer completion artifacts exist. Dated evidence is not a live training dashboard.

Version 0.3.3 makes the selected-proposal training point cap explicitly 100k while retaining 500k fixed evaluation. It publishes the exact epoch-150 encoder with tensor-preserving export and checked download instructions in [the weights guide](docs/weights.md). New CPU tests cover capped traversal, historical seed ordering, fixed-pack identity and cap changes on recovery. No GPU training or benchmark rerun is claimed.

Version 0.3.2 updates method naming, recipe guidance, experiment evidence and the workbench's visible project name. It also exposes the tested semantic-classifier helper described in [the probe guide](docs/probes.md). This helper does not package the experimental multi-view RGB semantic pipeline. Utonia and the newest low-annotation runners remain research implementations outside the unified release commands; see [code audit provenance](manifests/research_code_refresh_20261023.json).

The 17 October trained-model inference functionality remains available. Full Mask3D or AGILE3D task checkpoints are required; an encoder checkpoint alone is insufficient. This maintenance update does not rerun GPU inference or benchmarks, submit training, alter running experiments, or publish the paper draft. Current CPU and interface validation is recorded in the update's pull request; the GPU checks below remain dated evidence for their original versions.

## Earlier release verification

**Updated 17 October 2026.** This update builds on the 12 October GitHub release at `ea5dfe3`, which already included both SAM2 negative policies, strict checkpoint recovery, native Mask3D/AGILE3D adapters and the corrected BatchNorm inference policy. This week's new evidence is recorded in [the weekly update](docs/weekly_update_20261017.md), with a [compact evidence snapshot](docs/weekly_update_20261017.json).

The release keeps training, evaluation and feature probing separate from inference with a trained task checkpoint. An UnSAMv2 or SAM2 encoder alone is not a trained instance or interactive segmentation model: Mask3D/AGILE3D also require the corresponding trained decoder. The current trained historical UnSAMv2 models and the newer SAM2 pretraining experiments remain distinct entries.

The 17 October maintenance fixes native scratch/pretraining checkpoint initialization for Mask3D: the known contrastive loss-temperature scalar is checked and excluded alongside the projector, while all encoder tensors still load strictly and task decoder tensors are refused. The new regression tests cover exact encoder loading, unchanged decoder weights and malformed or ambiguous checkpoint rejection.

Version 0.3.1 provides [trained-model inference and a 3D workbench](docs/inference.md): model selection, NPZ/PLY input, automatic Mask3D masks, click-guided AGILE3D objects and full-resolution exports. The local UI and outbound GPU worker are included in the wheel; an optional private hosted UI is included as a source template. It supports full LitePT-S* task checkpoints and requires a CUDA runtime.

The [verification record](manifests/inference_verification.json) now records successful GPU inference with all four completed pretrained/scratch task checkpoints on a 232,453-point scene, four model switches, unchanged normalization/RNG state, cached inference, click corrections and complete exports. All four also passed real inference through the private hosted browser. The CPU suite passed 280 cases with 19 skipped; 21 JavaScript checks passed. The GPU used Torch SDPA and the pinned Torch CUDA PointROPE implementation on an RTX 2080 Ti. This is inference functionality verification, not a new benchmark or verification of the incompatible installed PointROPE GPU binary.

The verification records below describe the **12 October base release**, not tests of subsequent changes or completion of full research experiments. Current change validation belongs in the change's review record.

The base release was verified at commit `1f4876b`: 228 CPU cases and 19 CUDA cases across isolated allocations, plus public two-GPU training/recovery with all 32 runtime checks passing. Model and optimizer-state tensors agreed within rtol 1e-5 / atol 1e-6; recovery job 16816629 completed with exit code zero.

The SAM2 negative-policy extension passes 251 CPU cases, including 23 cases covering both policies, negative families, fixed evaluation and model/optimizer/coverage recovery. It adds explicit overlap-veto and selected-proposal-negative recipes, separate fixed packs, and policy checks in training, evaluation and resume. Two-GPU native training, fixed evaluation, resume and uninterrupted repetition passed for both policies: three updates per run, all 33 runtime checks, exact RNG/sampler/exposure/counter state, and rejection of resumes under the other policy. Floating model/optimizer differences are recorded alongside uninterrupted repetition; GPU tensor identity is not asserted. The original numerical-comparison failures from allocations 16841097 and 16842561 are preserved. The completed policy assessment and all numerical diagnostics are recorded separately in `manifests/sam2_negative_policy_verification.json`.

The 12 October wheel contained 178 source modules, required assets and upstream notices. Its installed previews selected both new policies correctly, and the fixed-evaluation command exposed the recipe/policy options. The base release's seven-command and fifteen-action entry-point checks remain recorded in `manifests/release_verification.json`, explicitly scoped to the base commit.

Base native checks identified an older PointROPE binary containing only the RTX 3090 kernel. An isolated build from the pinned source now provides both A100 and RTX 3090 kernels; the shared research runtime was preserved. The base hierarchy recovery fixture uses mask coefficient .1 within the unchanged .5 auxiliary-gradient budget and stages a parent history ending at the resume checkpoint. Production recipes retain their original coefficients and continuation guards.

Full-data teacher export, recipe-specific calibration and multi-day training/evaluation are separate experiment steps. Historical runs remain in the original checkout with their original records.
