# Pretraining recipes

All standard recipes train LitePT-S* from scratch using RGBN6 and projected 2D masks. Earlier multiscale/hierarchy losses remain distinct configurations.

| Recipe | Input | Main supervision | Schedule / normalization |
| --- | --- | --- | --- |
| `sam2_overlap_veto.yaml` | Full scene, 2 cm; geometry fragments <=500k | SAM2 dec0, shared-mask negatives vetoed | 150 epochs, FP32 SyncBN |
| `sam2_selected_proposal_negatives.yaml` | Full scene, 2 cm; training fragments <=100k, fixed evaluation <=500k | SAM2 dec0, selected-mask complement negatives | Same schedule and normalization |
| `unsamv2_multiscale.yaml` | Full scene, 2 cm | dec0 + dec3/dec2/dec1 contrastive | 84 epochs, local BN |
| `unsamv2_hierarchy.yaml` | Sphere <=50k, 5 cm | dec0 + proposal-query hierarchy masks + VICReg | 336 epochs, SyncBN |
| `unsamv2_hierarchy_no_vicreg.yaml` | Same sphere and grid | Same hierarchy masks; VICReg coefficient zero | Same schedule and normalization |

These recipes inherit `base.yaml`. CLI flags override values before environment expansion. `--show-config` shows the merged file and common path overrides without loading a backbone.

## Shared contrastive sampling

- Physical-frame splits are fixed; unknown and overlapping mask memberships are retained.
- Training uses 16 proposals, 64 positive pairs and 512 negatives per proposal: 256 uniform, 128 spatial and 128 feature hard, with candidate pools of 1024.
- Native token supervision keeps exact first representatives and pooling ancestry. Relation checks prevent unsafe positives/negatives.
- Logical optimizer updates preserve the configured global scene batch. Coverage sampler state, rank RNG state, heads, optimizer, scheduler and counters are checkpointed.
- Recipe, source pack, code record, grid, feature interface and normalization are recorded. Changing any of these creates a different experiment.

## SAM2 dec0

The current LOC3D method uses **selected-proposal negatives**. Overlap veto is a separate comparison; historical runs retain their original policy. Both policies are implemented and use separate recipes:

- **Overlap veto** (`allow_shared_proposal_negatives: false`): a negative must be outside the selected mask and share no same-frame mask with either positive endpoint. `sam2_dec0.yaml` retains this default for compatibility.
- **Selected-proposal negatives** (`allow_shared_proposal_negatives: true`): a negative must remain outside the selected mask, but another broader mask may contain it and the positive endpoints. Known mask coverage, native token purity, unique pairs and disjoint negative families are still required.

The selected-proposal preset uses **100,000 raw points per training scene visit** (`sam2_maximum_raw_points: 100000`). Fixed evaluation independently retains **500,000 points**, matching the original epoch-150 protocol. The legacy overlap-veto preset keeps 500,000 training points. Partitioning and fragment rotation preserve the original 100k run's seed convention. Changing the training cap on resume or reusing a fixed pack with a different evaluation cap is rejected. The [published epoch-150 encoder](weights.md) preserves all encoder tensors and raw BatchNorm buffers. This release has CPU contract checks; it does not claim a fresh end-to-end retraining reproduction.

The selected-proposal policy is restricted to SAM2 dec0-only V2 pretraining. It does not change the older UnSAMv2 recipes. Membership in another overlapping proposal is allowed outside the selected proposal; this does not assert that every such pair belongs to different semantic objects. The policy is stored in execution records, resolved run settings, fixed-evaluation manifests/records and checkpoints. Mixing policies in a batch, using the other policy's evaluation pack, or changing policy during resume is rejected.

Prepare each policy's fixed pack using its recipe as described in [data.md](data.md). The selected-proposal recipe uses `${DELIMIT3D_FIXED_EVAL}_selected_proposal_negatives_100k`; the veto uses `$DELIMIT3D_FIXED_EVAL`. Legacy fixed packs without a policy flag are treated as overlap veto. Native fixed-panel losses and gaps use different negatives between policies, so compare their downstream features under the same independent evaluation protocol.

Launch two ranks with six scenes per rank (global batch twelve):

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/sam2_overlap_veto.yaml \
  --output-dir "$DELIMIT3D_WORK_ROOT/pretrain_sam2_overlap_veto"
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/sam2_selected_proposal_negatives.yaml \
  --output-dir "$DELIMIT3D_WORK_ROOT/pretrain_sam2_selected_proposal_negatives"
```

The reference schedule is 36,630 updates for 2,930 training scenes over 150 epochs, with logical-batch flushes at epochs 25/50/75/100/125/150. Encoder LR is 1e-4, head LR 3e-3, weight decay .05, clip norm 5 and warmup 17 epochs. Only native dec0 is supervised: no auxiliary hierarchy or VICReg objective is silently enabled.

Fragments are a geometry-based traversal of large scenes, retaining the logical scene and proposal membership. Persistent coverage cursors are part of exact recovery. Three/four-rank and B* historical runs are different hardware/model contracts, not aliases for this release preset.

The command creates a scratch initialization beside a new run when `--initial-checkpoint` is omitted. Supply an existing initialization for a controlled comparison. It verifies the sealed data/code records and publishes an execution record binding those bytes, the initialization and both splits before training. Each rank rechecks the same record.

## Sampling controls

See [sampling_ablations.md](sampling_ablations.md) for matched hard-negative, overlap-veto and positive-area controls, their train-only area audit, common initialization/fixed panel, preserved LR horizon and checkpoint admission.

## Earlier UnSAMv2 contrastive multiscale

This is the original local-BN, full-scene objective: dec0 contrastive plus dec3/dec2/dec1 weighted .2/.3/.5, scaled by .4 with a 20-epoch auxiliary warmup. It uses random sampling with replacement and a balanced three-cell permutation. Encoder LR is 1e-3, head LR 3e-3, weight decay .05, clip norm 1 and warmup 17.

The two-rank/global-twelve reference has 20,510 updates over 84 epochs. Full-scene memory is substantial: the earlier reference used two 80 GB A100s. SAM2's geometry fragments are not applied to this recipe.

A calibration report is required for the exact prepared data, source/code hashes, initialization and optimizer. First create an initialization:

```bash
python scripts/prepare_data.py --action backbone-init \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" --grid-size 0.02 --input-features rgbn6 \
  --multiscale-supervision --multiscale-loss-weight 0.4 --multiscale-warmup-epochs 20 \
  --output "$DELIMIT3D_WORK_ROOT/multiscale_initial.pt"
torchrun --standalone --nproc_per_node=2 scripts/prepare_data.py --action calibration \
  --source-root "$DELIMIT3D_SOURCE_ROOT" --source-audit "$DELIMIT3D_SOURCE_AUDIT" \
  --initial-checkpoint "$DELIMIT3D_WORK_ROOT/multiscale_initial.pt" \
  --code-manifest "$DELIMIT3D_CODE_MANIFEST" \
  --pretraining-contract configs/pretrain/unsamv2_multiscale.yaml \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" --output "$DELIMIT3D_CALIBRATION"
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/unsamv2_multiscale.yaml \
  --initial-checkpoint "$DELIMIT3D_WORK_ROOT/multiscale_initial.pt"
```

Calibration preserves the 96-visit/48-scene panel, global reductions, eight paired optimizer updates, token-purity checks and zero-valid handling. It publishes a completed/passed training gate rather than inferring readiness from losses. Changing code/data/initialization requires a new calibration.

## Earlier hierarchy objective

The hierarchy recipe keeps the 5 cm sphere protocol and V2 coverage sampler. It routes g08 to enc4/dec3, g05 to dec3/dec2, and g02 to dec2 when support is valid, otherwise dec1. It uses per-stage projection heads, detached two-fold dec0 queries, mask coefficient 1 and scene-balanced FP32 VICReg coefficient 2; enc4 mask weight is 1.25.

Preselection draws representative positive pairs from a proposal-local CPU stream. The optional CUDA planning check uses the same pairs, so planning placement cannot change proposal admission or its recorded evidence.

VICReg uses at most 4096 tokens, variance target 1, covariance weight .04 and a 5% warmup. Feature-hard activation requires the recorded native-gap gates. The auxiliary gradient controller preserves its halving/PCGrad/rejection rules and shared-parameter norm budget. The no-VICReg preset changes its coefficient to zero while retaining mask routes.

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/unsamv2_hierarchy.yaml
```

The reference 336-epoch schedule has 82,040 updates. Fresh 2D-only sphere packs use a different crop preparation from the original sealed packs; report them separately.

## Resume

Use the same recipe, initialization, data/code records and world-size contract. Resume into a **fresh output directory**:

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/sam2_dec0.yaml --initial-checkpoint /path/to/original_initial.pt \
  --resume /path/to/checkpoints/epoch_0025.pt --resume-sha256 CHECKPOINT_SHA256 \
  --resume-into-fresh-attempt \
  --output-dir /path/to/fresh_resume_run
```

The resume check verifies saved rank coverage/RNG state, optimizer and scheduler frontier. It does not reset Adam counters or rename a changed recipe as a continuation.

The CPU suite checks model/optimizer recovery at rtol 1e-5 / atol 1e-6. Native policy verification requires exact RNG, coverage, exposure, sampling totals and counters, and reports model/optimizer differences alongside an uninterrupted repetition. The synthetic GPU runs showed floating-point variation in both repetition and resume; they do not establish bitwise tensor reproduction.

Use the original negative-policy recipe, training cap and fixed pack when resuming SAM2. Before this update, selected-proposal presets defaulted to 500k; preserve those runs with `--sam2-maximum-raw-points 500000` and their original `--fixed-eval-pack`. Missing cap metadata means the historical 500k default. Changing `allow_shared_proposal_negatives` requires a separate run; it cannot be continued from a checkpoint trained under the other policy. Checkpoints predating the flag default to the veto, subject to the existing code/data compatibility checks.

Fresh-attempt continuation requires the latest parent checkpoint and the complete expected checkpoint history through that boundary. A parent folder containing later checkpoints is rejected. The synthetic recovery check stages a separate parent snapshot through epoch one; it preserves the uninterrupted run for comparison.

## Export and BatchNorm recalibration

```bash
python scripts/export_backbone.py --checkpoint /path/to/epoch_0150.pt \
  --litept-root "$DELIMIT3D_LITEPT_ROOT" --grid-size 0.02 \
  --output "$DELIMIT3D_BACKBONE_CHECKPOINT"
```

The export strictly loads all native backbone tensors and excludes pretraining heads and task decoder weights.

For the separate 48-scene BN recalibration protocol, first prepare its normal audit:

```bash
python scripts/prepare_data.py --action normal-audit --source-root "$DELIMIT3D_SOURCE_ROOT" \
  --train-split "$DELIMIT3D_TRAIN_SPLIT" --scenes 48 --output "$DELIMIT3D_WORK_ROOT/normal_audit.json"
torchrun --standalone --nproc_per_node=2 scripts/calibrate_bn.py \
  --checkpoint "$DELIMIT3D_BACKBONE_CHECKPOINT" --grid-size 0.02 \
  --source-root "$DELIMIT3D_SOURCE_ROOT" --train-split "$DELIMIT3D_TRAIN_SPLIT" \
  --normal-audit "$DELIMIT3D_WORK_ROOT/normal_audit.json" --litept-root "$DELIMIT3D_LITEPT_ROOT" \
  --scenes 48 --normalization-policy sync_batchnorm --world-size 2 \
  --augmentation-profile mask3d_scannet_rgbn6_fullscene_v1 \
  --output-checkpoint /path/to/recalibrated.pt --output-manifest /path/to/recalibrated.json
```

Use grid .05 and the sphere augmentation profile for the hierarchy recipe. Only running mean/variance and BN counters may change. Source checkpoint bytes and all non-BN tensors remain unchanged.
