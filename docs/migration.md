# Release migration

The release was built in a separate checkout on `codex/reproducible-release`. The original research checkout, untracked work, checkpoints, data and running jobs were preserved.

## Why the original tree had drifted

The working checkout mixed executable research code, copied downstream dependencies, per-experiment launchers/config snapshots, dashboards, queue recovery, reports and generated files. Recent functional code was often untracked and some task engines were imported from another project directory. The public README and committed packaging still described the older short-adaptation workflow.

Deleting only untracked files would have discarded necessary training and evaluation changes. The release first captured the working files, imported the current implementations, then selected the dependency closure of the supported commands.

## What changed

- SAM2 contrastive pretraining is the main project description. Earlier UnSAMv2 multiscale and hierarchy recipes remain explicit.
- Seven public commands replace duplicated launchers. Small inherited task/model recipes replace repeated configuration payloads.
- Pretraining definitions are grouped by data, sampling, objective, checkpointing, evaluation and training responsibility.
- Mask3D and AGILE3D engines, losses, initialization and official click/evaluation code are part of this package.
- LitePT-S*, Concerto and Sonata keep distinct native feature adapters. Shared decoder initialization and private projection RNG are retained.
- Dependency commits, LitePT patch hashes, public checkpoint hashes and ordered scene splits are versioned.
- Fresh pretraining preparation uses 2D masks without 3D label supervision. Historical source formats and their scientific distinctions remain supported.
- Task data preparation produces geometry, target caches and aligned normals without requiring a CHORUS checkout.
- New run/checkpoint records bind recipe, data and model interface. Inference normalization is explicit and stateless.
- Dated schedulers, dashboards, publication scripts, repeated experiment reports and generated outputs are excluded from the release.

## Preservation and reversibility

The original files and the release tree before curation are preserved in:

```text
/cluster/work/igp_psr/nedela/delimit3d_release_preservation_20261011_v1/
  working_files.tar.gz
  preservation.json
  release_before_curation.tar.gz
  curation_plan.json
```

`manifests/release_migration.json` records imported files; `release_curation.json` records each removed file and hash. These archives are local preservation artifacts, not runtime dependencies. Optional historical commands remain under `tools/reproduce/`.

## Verification boundary

CPU checks cover sampling, masks/unknown regions, pooling ancestry, fixed evaluation, logical schedules, strict loads, decoder behavior, normal alignment, freeze/BN rules and preparation records. Opt-in CUDA checks cover real native backbones, task forward/loss/backward/optimizer paths and pretraining objectives using isolated synthetic inputs.

These checks do not reproduce a complete multi-day training run or full benchmark table. A fresh teacher export and a full-data calibration are recorded separately from release verification. Current results are summarized in `CURRENT.md` and `manifests/release_verification.json`.

## Publishing

The release is locally reviewable. GitHub main, live experiments and public metadata are not changed by this migration. The original project's license remains the owner's decision; copied components carry their upstream notices.
