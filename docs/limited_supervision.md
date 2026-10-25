# Low-data and low-annotation Mask3D experiments

The portable commands prepare and run a matched **scratch/pretrained LitePT-S***
comparison, with shared Mask3D decoder initialization and explicit checkpoint
identity. Run them from the repository root after installing the
[task environment](../environment/README.md). Licensed ScanNet files, checkpoints
and CUDA dependencies remain external.

## Two distinct protocols

| Protocol | Training supervision | Training scenes | Scheduled updates at 600 epochs |
| --- | --- | ---: | ---: |
| Low-data 1% | Dense labels | 12 | 1,800 |
| Low-data 5% | Dense labels | 60 | 7,200 |
| Low-data 10% | Dense labels | 120 | 14,400 |
| Low-data 20% | Dense labels | 240 | 28,800 |
| Low-annotation 20/50/100/200 | Official selected vertices per scene | 1,201 | 144,600 |

All protocols evaluate on the **312 ScanNet v2 validation scenes**, with dense
ScanNet20 object annotations used only for evaluation. Metrics are custom
**class-agnostic raw-query AP (CA-AP, CA-AP50 and CA-AP25)**. They are separate
from semantic mIoU, class-aware instance AP and the official leaderboard scores.

Low-data selects the first `floor(1201 * percent / 100)` scenes after
`random.Random(42).shuffle(sorted(train_scenes))`. The four subsets are nested,
fixed across arms and epochs, and contain no validation scenes. These are custom
controlled subsets, not the published ScanNet Data-Efficient limited-scene lists.
Each selected scene is visited once per natural epoch, without repetition to
inflate small subsets. Equal epochs therefore give different update counts
across fractions; updates are matched between arms within each fraction.
The exact lists are versioned in [configs/splits/low_data](../configs/splits/low_data/),
with counts and SHA-256 values in
[manifests/low_data_splits.json](../manifests/low_data_splits.json).

Low-annotation uses the official `points20`, `points50`, `points100` or `points200`
indices in the original `*_vh_clean_2.ply` vertex order. It retains all training
geometry. The allocation is fixed, is not sampled again each epoch and has no
per-instance quota. Official duplicate index entries are retained and counted
separately from unique vertices. The four independently supplied point lists are
not assumed to be nested.

## Prepare the inputs

Prepare ScanNet geometry and dense task inputs with the
[Mask3D data preparation command](data.md#scannet-inputs-for-mask3d). Generate
one fresh decoder template and use it for both arms and every budget:

```bash
python scripts/prepare_data.py --action decoder-init --task mask3d \
  --output /data/checkpoints/mask3d_decoder_init.pt
```

For low-data, the geometry packs and dense GT cache are sufficient. For
low-annotation, download the official
[point allocation archive](https://kaldir.vc.in.tum.de/3dsis/data_efficient/points.zip)
from the [ScanNet Data-Efficient benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation)
and prepare an independent sparse dataset:

```bash
python scripts/prepare_low_annotation.py \
  --raw-root /data/scannet/scans \
  --pack-root /data/mask3d/scans \
  --train-split configs/splits/scannet_train.txt \
  --val-split configs/splits/scannet_val.txt \
  --label-map /data/scannet/scannetv2-labels.combined.tsv \
  --points-archive /data/downloads/points.zip \
  --output /data/scannet_low_annotation

python scripts/prepare_low_annotation.py --verify /data/scannet_low_annotation
```

The pack root contains `SCENE/training_pack/{points,colors,normals}.npy`
(`litept_pack` is also accepted). Preparation requires exact ordered XYZ equality
with each original raw mesh, validates the official scene set and index bounds,
and pins the archive SHA-256 to
`c265f249c446155d32e751576288f599caed46ee11a32ecaccd51e037d14cbd6`.

The output contains `geometry/`, `train_annotations/{20,50,100,200}/`, separate
`validation/` labels, copied split/index inputs, `data_manifest.json` and its
SHA-256 seal. Training annotations contain only selected vertex indices,
observed instance IDs and semantic IDs. Dense validation targets are constructed
from validation annotations only. Verification checks the prepared output without
requiring access to the raw source tree. An existing output directory is refused.

## Seal a matched experiment

Specify the encoder checkpoint and a descriptive label explicitly. The examples
below name the historical SAM2 overlap-veto epoch-100 encoder; provide that exact
file to compare its initialization. The public
[selected-proposal epoch-150 weights](weights.md) are a different encoder and
must be given a different label, for example `sam2_selected_proposal_e150`.
The command never chooses or downloads pretrained weights implicitly.

```bash
python scripts/limited_supervision.py prepare \
  --protocol low-data --percent 5 \
  --litept-root /data/dependencies/LitePT \
  --scans-root /data/mask3d/scans \
  --gt-cache-root /data/mask3d/gt_cache \
  --backbone-checkpoint /data/checkpoints/sam2_overlap_veto_e100_encoder.pt \
  --checkpoint-label sam2_overlap_veto_e100 \
  --decoder-template /data/checkpoints/mask3d_decoder_init.pt \
  --output /data/experiments/lowdata_5pct

python scripts/limited_supervision.py prepare \
  --protocol low-annotation --budget 200 \
  --litept-root /data/dependencies/LitePT \
  --prepared-root /data/scannet_low_annotation \
  --backbone-checkpoint /data/checkpoints/sam2_overlap_veto_e100_encoder.pt \
  --checkpoint-label sam2_overlap_veto_e100 \
  --decoder-template /data/checkpoints/mask3d_decoder_init.pt \
  --output /data/experiments/low_annotation_200
```

Use `--percent 1`, `5`, `10` or `20`, or `--budget 20`, `50`, `100` or `200`, with
a fresh experiment directory for each budget. `--litept-root` points to the
prepared LitePT dependency checkout; alternatively set `DELIMIT3D_LITEPT_ROOT`.
Preparation writes `configs/scratch.yaml`, `configs/pretrained.yaml`, `train.txt`,
`validation.txt`, `manifest.json` and `manifest.sha256` inside the experiment.
The manifest records checkpoint hashes, all consumed training/validation data
hashes, protocol identity, the release implementation hash and the external
LitePT Python source hashes. Verification checks these identities again. Keep
the referenced files, dependency checkout and sealed release unchanged;
code/configuration changes require a fresh experiment. Use the versioned
environment profile for CUDA and compiled dependencies.

The default scratch arm initializes a fresh encoder with seed 42. To reuse an
exact historical scratch state, pass `--scratch-checkpoint /path/to/encoder.pt`
when preparing the experiment. Matching a seed alone does not establish that
an exported historical state is identical. The historical overlap-veto encoder
export had SHA-256
`bb55e883cd8bd7897190e10cbf879b394d4c944777ae6b94fd3349f0d9e2c854`.
Reproducing a historical result additionally requires its original data,
initialization artifacts, code and full training/evaluation contract.

Both arms fully fine-tune the encoder and decoder for **600 natural epochs**,
with five scenes per update, seed 42, AdamW, equal encoder/decoder OneCycleLR peaks
of **1e-4**, weight decay 0.01 and no gradient clipping. The final incomplete
batch is retained. Each arm receives the same decoder template, ordered scenes,
supervision and augmentation policy. BatchNorm evaluation uses `train_no_update`;
DBSCAN is disabled. These settings differ from the generic Mask3D defaults in
[tasks.md](tasks.md#mask3d).

Low-data evaluates and saves numbered checkpoints every 50 epochs. Low-annotation
does so every 10 epochs through epoch 100, then every 50 epochs through 600.
Both retain a resumable latest checkpoint after each epoch. Scheduled update
counts include any empty sparse-supervision slots; the low-annotation retention
audit records effective updates and empty batches separately.

## Verify, train, resume and evaluate

Verify the sealed experiment before allocating a GPU:

```bash
python scripts/limited_supervision.py verify \
  --experiment /data/experiments/lowdata_5pct
```

Run each arm with a single CUDA worker and enough memory for the complete
five-scene batch:

```bash
python scripts/limited_supervision.py train \
  --experiment /data/experiments/lowdata_5pct --arm scratch
python scripts/limited_supervision.py train \
  --experiment /data/experiments/lowdata_5pct --arm pretrained
```

The same commands work with the low-annotation experiment directory. A full
training checkpoint, including optimizer and schedule state, is required to
resume; an encoder export cannot resume task training. Each arm writes its
metrics, initialization report and checkpoints beneath `runs/ARM/`:

```bash
python scripts/limited_supervision.py train \
  --experiment /data/experiments/lowdata_5pct --arm pretrained \
  --resume /data/experiments/lowdata_5pct/runs/pretrained/checkpoints/latest.pt

python scripts/limited_supervision.py evaluate \
  --experiment /data/experiments/lowdata_5pct --arm pretrained \
  --checkpoint /data/experiments/lowdata_5pct/runs/pretrained/checkpoints/epoch_0600.pt \
  --output /data/evaluations/lowdata_5pct_pretrained_epoch600
```

If validation fails after a numbered checkpoint was saved, resume that newer
checkpoint for the same arm: for example `epoch_0050.pt` for low-data, or
`epoch_0010.pt` for low-annotation. Pending validation is replayed before the next
training epoch. The command verifies the experiment identity and rejects older
checkpoints that would overwrite later progress. Resume and evaluation also
reject a checkpoint from a different arm, data budget or training recipe, even
when its tensor shapes are compatible.

Use a fresh evaluation output directory. Report the final scheduled epoch and
verify all 312 validation scenes before recording a completed result. Keep
intermediate, resumed, failed and incomplete runs distinguishable from finished
ones. Do not adjust the optimizer or replace failed arms under the same sealed
experiment identity.

## Sparse supervision boundary

Low-annotation uses geometry-derived 5 cm voxels for both training masks and
evaluation reconstruction. This differs from the dense low-data mesh-segment
mask domain and prevents dense segments from disclosing hidden object extent.
Voxel representatives do not discard annotated non-representative vertices.
Identical observed labels in a voxel are coalesced; conflicting observed labels
make that voxel unknown.

Observed foreground instances are positive targets. Explicitly annotated
wall/floor points and other observed instances can be mask negatives. Every
unannotated vertex remains unknown, including unannotated parts of an observed
instance. Matching and mask BCE/Dice use only the observed domain. Unmatched
queries receive no no-object, classification or mask penalty. Every auxiliary
decoder layer follows the same sparse objective. Empty supervision yields zero
instance loss, without inventing background targets or hidden object counts.

The release provides the scripts and contract checks for these experiments;
CPU verification does not establish a completed GPU rerun or reproduce historical
scores. Concerto and Sonata remain available through the general task commands;
the sealed experiment helper currently generates the matched LitePT-S* pair.
