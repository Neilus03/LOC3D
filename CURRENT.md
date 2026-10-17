# Release status

**Updated 17 October 2026.** The GitHub release was last merged on 12 October at `ea5dfe3`. It already included both SAM2 negative policies, strict checkpoint recovery, native Mask3D/AGILE3D adapters and the corrected BatchNorm inference policy. This week's new evidence and the supported inference direction are recorded in [the weekly update](docs/weekly_update_20261017.md), with a [compact evidence snapshot](docs/weekly_update_20261017.json).

The release keeps training, evaluation and feature probing separate from inference with a trained task checkpoint. An UnSAMv2 or SAM2 encoder alone is not a trained instance or interactive segmentation model: Mask3D/AGILE3D also require the corresponding trained decoder. The current trained historical UnSAMv2 models and the newer SAM2 pretraining experiments remain distinct entries.

The 17 October maintenance fixes native scratch/pretraining checkpoint initialization for Mask3D: the known contrastive loss-temperature scalar is checked and excluded alongside the projector, while all encoder tensors still load strictly and task decoder tensors are refused. The new regression tests cover exact encoder loading, unchanged decoder weights and malformed or ambiguous checkpoint rejection.

The verification records below describe the **12 October base release**, not tests of subsequent changes or completion of full research experiments. Current change validation belongs in the change's review record.

The base release was verified at commit `1f4876b`: 228 CPU cases and 19 CUDA cases across isolated allocations, plus public two-GPU training/recovery with all 32 runtime checks passing. Model and optimizer-state tensors agreed within rtol 1e-5 / atol 1e-6; recovery job 16816629 completed with exit code zero.

The SAM2 negative-policy extension passes 251 CPU cases, including 23 cases covering both policies, negative families, fixed evaluation and model/optimizer/coverage recovery. It adds explicit overlap-veto and selected-proposal-negative recipes, separate fixed packs, and policy checks in training, evaluation and resume. Two-GPU native training, fixed evaluation, resume and uninterrupted repetition passed for both policies: three updates per run, all 33 runtime checks, exact RNG/sampler/exposure/counter state, and rejection of resumes under the other policy. Floating model/optimizer differences are recorded alongside uninterrupted repetition; GPU tensor identity is not asserted. The original numerical-comparison failures from allocations 16841097 and 16842561 are preserved. The completed policy assessment and all numerical diagnostics are recorded separately in `manifests/sam2_negative_policy_verification.json`.

The updated wheel contains exactly the 178 current source modules, required assets and upstream notices. Installed previews select both new policies correctly, and the fixed-evaluation command exposes the recipe/policy options. The base release's seven-command and fifteen-action entry-point checks remain recorded in `manifests/release_verification.json`, explicitly scoped to the base commit.

Base native checks identified an older PointROPE binary containing only the RTX 3090 kernel. An isolated build from the pinned source now provides both A100 and RTX 3090 kernels; the shared research runtime was preserved. The base hierarchy recovery fixture uses mask coefficient .1 within the unchanged .5 auxiliary-gradient budget and stages a parent history ending at the resume checkpoint. Production recipes retain their original coefficients and continuation guards.

Full-data teacher export, recipe-specific calibration and multi-day training/evaluation are separate experiment steps. Historical runs remain in the original checkout with their original records.
