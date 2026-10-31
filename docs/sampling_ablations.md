# Matched sampling ablations

The October sampling study is now available as portable preparation, training
controls and checkpoint admission. The four controls inherit the selected-proposal
SAM2 LitePT-S* recipe: RGBN6, 2 cm native grid, 100k training fragments, a shared
500k fixed-evaluation pack, seed 42, two ranks and global batch twelve. No numerical
ablation result is released here. The update has CPU contract tests; it does not
establish a new GPU training reproduction.

## What each control changes

The original full reference activates feature-hard negatives adaptively from its
native fixed-evaluation gap. Its observed activation was update **6106**. Controlled
arms use that fixed update to avoid changing their schedules in response to their
own validation scores. A new full reference must have its actual activation checked;
if it differs, prepare new controlled runs with the observed update.

| Arm | Before activation: uniform / spatial / feature | After activation | Other change |
| --- | --- | --- | --- |
| `full` | 256 / 128 / 0 | 256 / 128 / 128 | Original adaptive gate |
| `no_hard` | 384 / 0 / 0 | 512 / 0 / 0 | Reallocate both hard-negative families to uniform |
| `no_feature_hard` | 256 / 128 / 0 | 384 / 128 / 0 | Reallocate feature-hard negatives to uniform |
| `overlap_veto` | 256 / 128 / 0 | 256 / 128 / 128 | Veto shared-mask negatives during training |
| `no_small_positive` | 256 / 128 / 0 | 256 / 128 / 128 | Exclude the smallest-area proposals from positive selection |

Requested totals remain 384 before activation and 512 after it. Actual delivery
can be smaller when safe candidates are unavailable; the trainer records delivery
and requested counts separately. Every arm uses the same selected-proposal fixed
panel. In particular, `overlap_veto` changes **training only**, unlike the separate
legacy `sam2_overlap_veto.yaml` recipe, which also uses a veto evaluation pack and
500k training fragments.

The area cutoff is the global lower quartile of original 2D mask areas for projected
proposals marked trainable, using optimization scenes and their training frames
only. Areas equal to the cutoff are excluded, so ties can remove more than 25%.
The audit reports the actual fraction. All original proposal relations remain
available for known-point support and negative checks. Geometry fragmentation
preserves each proposal's original image area.

## Prepare the common inputs

Follow [data.md](data.md) to prepare source packs, their artifact-verified audit,
the selected-proposal fixed pack and the code record. All masks must have a common
image resolution for a pixel-area cutoff. Configure the same environment variables
used by [pretraining.md](pretraining.md), including the selected-proposal fixed-pack
location. Generate one common initialization:

```bash
python scripts/prepare_data.py --action backbone-init \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" --grid-size 0.02 --input-features rgbn6 \
  --output "$DELIMIT3D_WORK_ROOT/sampling_initial.pt"

python scripts/sampling_ablation.py area-audit \
  --source-root "$DELIMIT3D_SOURCE_ROOT" --train-split "$DELIMIT3D_TRAIN_SPLIT" \
  --source-audit "$DELIMIT3D_SOURCE_AUDIT" \
  --output "$DELIMIT3D_WORK_ROOT/proposal_areas.json"

python scripts/sampling_ablation.py prepare \
  --config configs/pretrain/sam2_selected_proposal_negatives.yaml \
  --initial-checkpoint "$DELIMIT3D_WORK_ROOT/sampling_initial.pt" \
  --area-audit "$DELIMIT3D_WORK_ROOT/proposal_areas.json" \
  --output "$DELIMIT3D_WORK_ROOT/sampling_study"
```

Preparation writes five validated JSON recipes, input/config hashes and
`study.json`. It does not start jobs or claim training readiness on a GPU. Output
locations must be new. If recipe validation fails, the partial directory has no
published study index; preserve it for inspection and use a fresh destination.

The default stop is epoch 50/update **12210** for 2930 training scenes. The nominal
**150-epoch, 36630-update LR schedule remains unchanged**. Shortening `epochs` to 50
would change warmup/cosine progression and define another experiment. The helper
computes boundaries from the training split and existing evaluation flushes.

Standalone examples are also under `configs/ablations/sampling/`. Their fixed
12210 stop assumes the 2930-scene reference split. For the small-positive recipe,
set `DELIMIT3D_AREA_AUDIT` and `DELIMIT3D_AREA_CUTOFF` from the verified audit.
Prefer generated recipes when preparing a matched study.

## Train and recover

Launch the full reference and each desired control with the same runtime. For
example:

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config "$DELIMIT3D_WORK_ROOT/sampling_study/configs/full.json"
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config "$DELIMIT3D_WORK_ROOT/sampling_study/configs/no_hard.json"
```

Use `no_feature_hard.json`, `overlap_veto.json` and `no_small_positive.json` for the
other controls. Review the full reference's recorded activation before committing
to a long comparison. New controlled studies can pass `prepare --activation-update`
with that observed value. The ordinary full recipe retains its adaptive behavior.

Use the same generated recipe and the [fresh-attempt resume procedure](pretraining.md#resume)
for interruptions. Variant, cutoff/audit identity, activation, training policy and
fixed-evaluation policy are immutable on resume. Historical checkpoints without an
ablation stamp mean `full`; a controlled checkpoint cannot resume as that default.
Keep the original initialization and training/evaluation caps.

## Admit checkpoints and run diagnostics

After all five arms reach the intended boundary:

```bash
python scripts/sampling_ablation.py verify-checkpoints \
  --study "$DELIMIT3D_WORK_ROOT/sampling_study/study.json" \
  --epoch 50 --update 12210 \
  --output "$DELIMIT3D_WORK_ROOT/sampling_study/checkpoints_verified.json"
```

Fresh-attempt outputs can be supplied with repeated `--checkpoint VARIANT=PATH`
options. The verifier checks stable checkpoint bytes, finite model tensors, exact
boundary, arm, initialization, source/split/code/fixed-pack identities, cap, recipe,
LR horizon and the observed activation. It rejects five checkpoints that agree
with each other but differ from the prepared study. `CHECKPOINTS_VERIFIED` is
checkpoint admission, **not evaluation completion**. Load only trusted training
checkpoints; these contain optimizer and RNG objects as well as tensors.

`--interim` admits an earlier exact boundary without changing the production run.
For this schedule, epoch 25/update 6105 is before the feature-hard stage, so it
cannot measure the effect of removing feature-hard negatives. The historical
October epoch-5/update-1220 evaluation also predates activation. Its intermediate
checkpoint format requires the original snapshot/receipt validation: this portable
helper supports exact logical-batch boundaries and does not admit that nonflushed
snapshot. Adding epoch 5 to the flush schedule would change training and is not a
way to reproduce the historical checkpoint.

Export each admitted checkpoint with `scripts/export_backbone.py`, retaining the
same raw or recalibrated BatchNorm policy across arms. Point a probe recipe to the
exported checkpoint with native grid 0.02. Then use [feature_diagnostics.md](feature_diagnostics.md)
with the same prepared targets and preprocessing for every arm:

```bash
python scripts/feature_diagnostics.py evaluate \
  --targets /data/diagnostic_targets/scannet20/scene0000_00/target.json \
  --config /data/probes/no_hard.yaml --device cuda \
  --normal-source pointcept_scannet_mesh_normals \
  --output /data/diagnostics/no_hard/scene0000_00
```

Repeat for the exact declared scene list and taxonomy. Record checkpoint epoch,
normal source, native/common grid, coordinate and BN policy, seeds and coverage.
GT-K spherical-k-means and GT-query cosine are frozen-feature diagnostics, not
instance AP. A single-scene completion marker does not establish the full 312-scene
validation split, and one pretraining seed does not estimate training-seed variance.

## Existing granularity controls

`data/granularity_ablation.py`, `training/granularity_ablation.py` and the logical
batch scheduler already matched the research checkout at this refresh. Their
fixed-nine/six-arm granularity design and older fixed-fifteen design remain distinct
from these SAM2 sampling controls. Single-source UnSAMv2, hierarchy routing and
VICReg comparisons keep their existing protocols and tests; the new flags cannot
be used with those objectives.
