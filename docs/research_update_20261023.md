# Research update: 23 October 2026

Completed low-data experiments, SAM2 overlap-veto transfer, frozen-feature probes,
and sparse-annotation evidence now extend the [17 October update](weekly_update_20261017.md).
The [compact evidence record](research_update_20261023.json) preserves exact recipe
IDs, metrics, checkpoint identities and hashes of the source reports.

This is an artifact audit, not a new experiment run or live status feed. The
main frozen snapshot is 22 October 2026 at 07:58 UTC; completion receipts and
original sources were rechecked on 23 October, including the newly completed
600-epoch sparse-annotation pair from that morning. Incomplete runs remain
separate from completed results. Research-checkout-relative and
external-study-relative source identifiers in the JSON refer to retained evidence
outside this curated release.

## Recipe identities and claim boundaries

- **Historical UnSAMv2:** 5 cm pretraining with hierarchy/recovery and VICReg;
  the historical downstream arms use the epoch-150 encoder.
- **SAM2 overlap-veto:** 2 cm pretraining, dec0 supervision, no hierarchy/VICReg;
  shared membership in another proposal can veto a selected-proposal negative.
  The completed task and low-data results below use pretraining epoch 100.
- **SAM2 selected-proposal negatives:** known points outside the selected proposal
  remain negatives even when another proposal overlaps them. The reported
  initializer is epoch 150/update 36,630 with a 100,000-point pretraining cap.
  This is the current LOC3D direction, with final full-data transfer still absent
  from the checked artifacts. Both SAM2 policies retain overlapping proposals.

The part probe uses **epoch 150 for both SAM2 recipes**; overlap-veto there has a
500,000-point cap. Differences in cap, training duration and earlier recipe
components prevent a negative-policy-only or teacher-only causal claim.
External encoders also differ in architecture, pretrained data and native feature
readout. These are valid stated system comparisons; initialization effects need
matched controls. Downstream tables below use 5 cm geometry and fresh task heads.

## Completed automatic segmentation

Custom class-agnostic raw-query **CA-AP**, in percent, on all 312 ScanNet validation
scenes at task epoch 600. These values are separate from semantic or class-aware
ScanNet leaderboard AP. The decoder peak LR is `1e-4`; equal LR means the same
encoder peak LR, and `0.1x` means encoder peak LR `1e-5`.

| Initialization / adaptation | CA-AP | CA-AP50 | CA-AP25 |
| --- | --- | --- | --- |
| Scratch LitePT-S* | 57.83 | 78.33 | 90.35 |
| Historical UnSAMv2, frozen | 25.79 | 49.56 | 78.20 |
| Historical UnSAMv2, encoder LR 0.1x | 54.06 | 75.94 | 89.49 |
| Historical UnSAMv2, equal LR | 59.21 | 80.02 | 91.16 |
| Concerto Small, frozen | 26.34 | 49.20 | 73.70 |
| Concerto Small, encoder LR 0.1x | 44.55 | 68.17 | 84.78 |
| SAM2 overlap-veto e100, equal LR | 58.08 | 79.11 | 90.84 |
| SAM2 overlap-veto e100, encoder LR 0.1x | 51.25 | 73.27 | 87.90 |

The equal-LR final gains over scratch are 1.38 points for historical UnSAMv2 and
0.25 points for SAM2 overlap-veto. Reduced-LR and frozen arms remain distinct.
Selected-proposal, Sonata and Utonia full-data results are not promoted from
interim checkpoints to final results here.

## Completed interactive segmentation

New completed AGILE3D single-object evaluations at epoch 1100 cover 10,357 objects
and 217,497 rows per arm, including zero through twenty clicks. These are separate
from multi-object training validation. LitePT uses the recorded `train_no_update`
normalization protocol; external backbones retain their audited native policy.

| Initialization / adaptation | IoU at 1 click | IoU at 5 clicks |
| --- | --- | --- |
| Concerto Small, frozen | 59.08 | 77.93 |
| Concerto Small, encoder LR 0.1x | 64.93 | 80.05 |
| SAM2 overlap-veto e100, equal LR | 65.35 | 81.20 |
| SAM2 overlap-veto e100, encoder LR 0.1x | 63.39 | 80.84 |

The earlier matched equal-LR scratch and historical UnSAMv2 results remain
64.36/81.17 and 65.71/81.22 at one/five clicks, respectively; see the
[prior evidence and uncertainty](weekly_update_20261017.md#completed-matched-interactive-segmentation).
SAM2 overlap-veto's equal-LR result is 65.35/81.20. Early-click improvements do not
establish a substantial final high-click improvement. Selected-proposal final
single-object results are absent from the checked run roots.

## Custom limited-scene data

These are **custom seed-42 subsets**, with dense labels in the selected scenes,
not the official ScanNet Data-Efficient scene lists. All twelve results below
completed 600 epochs and validation on 312 held-out scenes. Equal encoder/decoder
peak LR is `1e-4`; values are CA-AP percent.

| Initialization | 1% / 12 scenes | 5% / 60 scenes | 10% / 120 scenes |
| --- | --- | --- | --- |
| Scratch LitePT-S* | 0.13 | 13.48 | 31.82 |
| Sonata Small | 0.11 | 5.09 | 14.58 |
| Concerto Small | 1.20 | 11.48 | 23.07 |
| SAM2 overlap-veto e100 | 2.32 | 27.24 | 39.27 |

The 1%, 5% and 10% schedules contain 1,800, 7,200 and 14,400 optimizer updates,
respectively. Changing fraction changes both scene count and update exposure.
The 20% run roots had no final completion receipts when checked on 23 October;
no final 20% value or Utonia low-data value is reported.

A separate **5% selected-proposal comparison** completed downstream seeds 43 and
44 on the same 60 scenes, with 7,200 updates and the same paired decoder seeds:

| Downstream seed | Scratch CA-AP | Selected-proposal e150 CA-AP | Paired gain |
| --- | ---: | ---: | ---: |
| 43 | 14.59 | 31.58 | +16.99 |
| 44 | 15.56 | 30.29 | +14.73 |

The mean paired gain is 15.86 points. This is two downstream seeds, one scene
subset and one pretrained initializer. The earlier interrupted overlap-veto
repeat is preserved separately and is not pooled with these results. This
supports the fixed 5% initialization comparison, without establishing robustness
across pretraining seeds or independently sampled subsets.

## Official sparse annotations, custom evaluation

A completed **20-epoch pilot** uses the official 200-point-per-scene annotation
files on all 1,201 ScanNet training scenes, with 4,820 optimizer updates and dense
validation on 312 scenes. The evaluation is still custom CA-AP; the use of official
annotation files does not make this an official leaderboard result.

| Initialization | CA-AP | CA-AP50 | CA-AP25 |
| --- | ---: | ---: | ---: |
| Scratch LitePT-S* | 4.01 | 9.29 | 22.81 |
| SAM2 overlap-veto e100 | 8.21 | 14.78 | 23.79 |

The verified pilot uses selected point annotations only: unknown points are not
background, unobserved objects do not enter matching or losses, and the same
sparse rule applies to auxiliary layers. Its completion gate checks protocol
correctness and identical supervision, independently of which method wins.
The budget includes background/invalid annotation entries; it is not 200 labeled
foreground-instance vertices per scene.

A separate **600-epoch study was launched on 22 October** for scratch,
SAM2 overlap-veto e100, Sonata Small, Concerto Small and Utonia Small, with
20/50/100/200-point budgets, seed 42, equal peak LR `1e-4`, five real scenes per
batch and 144,600 scheduled optimizer slots. Final selection is epoch 600.
Dense validation occurs every ten epochs through 100 and every fifty thereafter.
Empty-supervision slots receive no parameter update; effective updates are
reported separately. It starts from the original encoders and a fresh shared
decoder, not the completed pilot checkpoints. **The 200-point scratch and veto
arms completed on 23 October**, at 07:18 and 08:12 UTC respectively:

| Initialization, epoch 600 | CA-AP | CA-AP50 | CA-AP25 |
| --- | ---: | ---: | ---: |
| Scratch LitePT-S* | 17.42 | 25.58 | 31.02 |
| SAM2 overlap-veto e100 | 19.15 | 27.71 | 33.57 |

Both have 144,599 effective optimizer updates and one empty-supervision slot.
The audit verified all 20 full-validation milestones, 600 epoch retention records,
identical supervision hashes across arms, final validation hashes and the saved
checkpoint bytes against their completion receipts and backup record. The final
CA-AP gain is 1.73 points for this single-seed pair. Other budgets and external
methods had no final receipt in the checked roots; the five-method study remains
incomplete.

The earlier external-model pilot expansion recorded a Sonata memory failure,
a protective Concerto stop and no Utonia production start. The new study uses a
shared sparse-training allocation correction: unused dense auxiliary-output
copies are omitted while voxel matching/loss and dense evaluation are retained.
Those failures and the revision are preserved; pilot success does not imply a
completed five-method 600-epoch comparison. The sealed protocol hash is in the
compact evidence record.

## Custom frozen-feature probes

The six-method part suite completed all 42 official Articulate3D validation scenes
at 5 cm and 2 cm, with independent prediction rescoring. **Part retrieval AP and
matched AUROC are custom query-assisted feature-selectivity diagnostics**, using
known parent regions; they are not decoder-backed predicted-mask segmentation AP.
At 2 cm, the eligible cohort contains 1,445 parts:

| Initialization | Part retrieval AP | Matched AUROC |
| --- | --- | --- |
| LitePT-S* public | 65.10 | 63.41 |
| Concerto Small | 61.42 | 64.78 |
| Sonata Small | 58.18 | 60.05 |
| Utonia Small | 59.88 | 62.41 |
| SAM2 overlap-veto e150 (500k cap) | 73.26 | 72.44 |
| SAM2 selected-proposal negatives e150 (100k cap) | 73.58 | 72.97 |

Matched AUROC compares positive and sibling-part negative scores within matched
distance/RGB bins, with equal scene averaging. The 5 cm cohort has 819 eligible
parts, so cross-resolution differences also change the cohort. The small
selected-proposal/veto gap is descriptive, with no significance claim.

A separate completed object probe covers ten ScanNet scenes and 390 objects on
common 5 cm occupied support. It reports scene-mean macro IoU for cosine assignment
with shared GT query anchors and spherical k-means with GT foreground K. Seven
scene encoders have verified outputs, including Utonia Small. Exact values and
source hashes are in the JSON; this remains a native-system frozen-feature probe,
separate from learned task-head AP and interactive segmentation.

## Completed cross-dataset transfer

These evaluations reuse complete epoch-1100 AGILE3D task models without target
dataset training. The table shows five-click object-mean IoU in percent with
encoder LR `0.1x` for both arms. Coverage is 2,330 objects on S3DIS, 3,421 on
KITTI-360 and 2,661 on ScanNet++, with all 21 click-budget rows checked per object.

| Target dataset | SAM2 overlap-veto e100 | Concerto Small |
| --- | --- | --- |
| s3dis | 82.20 | 78.81 |
| kitti360 | 47.77 | 72.14 |
| scannetpp | 62.77 | 60.01 |

The indoor and outdoor rankings differ. At one click on S3DIS, Concerto leads
57.52 to 53.94, so the five-click table does not establish superiority at every
click budget. ScanNet++ uses the pinned within-suite protocol; it is not a verified
reproduction of an unavailable author object manifest. The equal-LR veto series
is separately recorded in the JSON.

## Supervised semantic labels after adaptation

These are historical UnSAMv2 epoch-150 encoders after full ScanNet Mask3D
fine-tuning to epoch 600. They are not probes of the untouched SAM2 initializer.
A frozen 144D mean-plus-max object descriptor from 72D dec0 features feeds one
18-way affine head. A separate RGB-assisted head adds 768D frozen multi-view
CLIP features, using frozen SAM crop refinement and calibrated RGB-D views.

| Frozen features | All-GT-object accuracy | Accuracy on matched predicted masks |
| --- | ---: | ---: |
| 3D only, 144D | 80.11 | 86.64 |
| RGB + 3D, 912D | 92.22 | 95.33 |

Both fit on 15,664 GT objects from 1,201 training scenes, with train-only
standardization and grouped training holdout selection. Validation has 4,329 GT
objects from 312 scenes. The conditional predicted-mask column uses the same
2,658 matches at IoU >= 0.75, or 61.40% coverage; geometric misses are excluded.
This is supervised closed-set recognition, separate from whole-scene semantic
segmentation and open-vocabulary recognition. The RGB-assisted result does not
attribute its semantic knowledge to the 3D encoder alone.

## Remaining evidence

Final selected-proposal full-data transfer, the final 20% custom scene-data
series, and the remaining sparse-annotation budgets and external-method cells
remain absent from the checked final artifacts. Matched pretraining grid/cap/policy controls,
more downstream seeds and independent scene subsets are still needed for broader
causal and robustness claims. Historical recipes, failed attempts and completed
results retain their original identities.
