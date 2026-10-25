# Preparing data

All preparation commands run from the release checkout. Edit `.env` first. Official Structured3D and ScanNet downloads remain external. Small, ordered scene splits are versioned in `configs/splits/` and hashed in `manifests/splits.json`: Structured3D has 2,930 training and 326 validation scenes; ScanNet has 1,201 training and 312 validation scenes.

## Structured3D geometry

Use the full perspective RGB/depth/pose archives. Preparation loads geometry, colors, normals and camera metadata. The public geometry action does not read 3D instance labels or bbox archives.

```bash
python scripts/prepare_data.py --action geometry \
  --scene-ids-file configs/splits/structured3d_all.txt \
  --raw-zips-dir "$DELIMIT3D_RAW_ZIPS" --output-root "$DELIMIT3D_GEOMETRY_ROOT"
```

This creates a geometry-only `source_manifest.json` beside the ordered point arrays and fixes a 90/10 physical-frame split. Geometry uses first representatives on a 2 cm grid. Newly computed normals use the recorded Open3D 0.19 policy. A fresh pack is a new data version: historical source artifacts keep their original hashes and normal policy.

## Main SAM2 supervision

In the separate SAM2 teacher environment, after fetching the pinned source and SAM2.1 Hiera-L weights:

```bash
python scripts/prepare_data.py --action sam2 \
  --scene-ids-file configs/splits/structured3d_all.txt \
  --raw-zips-dir "$DELIMIT3D_RAW_ZIPS" \
  --reference-source-root "$DELIMIT3D_GEOMETRY_ROOT" \
  --annotation-root "$DELIMIT3D_ANNOTATION_ROOT" --pack-root "$DELIMIT3D_SOURCE_ROOT" \
  --sam2-root "$DELIMIT3D_SAM2_ROOT" --checkpoint "$DELIMIT3D_SAM2_CHECKPOINT" \
  --worker-label export01 --report "$DELIMIT3D_WORK_ROOT/sam2_export.json"
python scripts/prepare_data.py --action sam2-seal \
  --scene-ids-file configs/splits/structured3d_all.txt --pack-root "$DELIMIT3D_SOURCE_ROOT" \
  --label sam2_fullscene --verify-cache-hashes --output "$DELIMIT3D_WORK_ROOT/sam2_seal.json"
```

The default export preserves points-per-side 32, batch 64, IoU .8, stability .95, box NMS .7 and no crop layers. Rawlight is used unless >=99% of pixels are saturated at RGB >=250; albedo is the fallback. Frames with both inputs saturated are valid empty observations. Overlapping mask memberships and unknown regions remain explicit.

The geometry files are hardlinked when possible or hash-verified copies across filesystems. Keep the geometry reference tree beside the export; SAM2 seals retain and verify the reference manifest.

## Earlier UnSAMv2 supervision

In the separate UnSAMv2 teacher environment, set `UNSAM_ROOT` and `UNSAMV2_CKPT`, then export all complete frames with valid poses:

```bash
python scripts/prepare_data.py --action unsamv2 \
  --scene-ids-file configs/splits/structured3d_all.txt \
  --scans-root "$DELIMIT3D_GEOMETRY_ROOT" --raw-zips-dir "$DELIMIT3D_RAW_ZIPS" \
  --output-root "$DELIMIT3D_UNSAM_PROPOSALS"
```

The recorded five excluded scenes are applied. Raw proposals retain the g02/g05/g08 granularities and teacher settings. Project them into a **new 2D-only pack** in the student environment:

```bash
python scripts/prepare_data.py --action sources \
  --scene-ids-file configs/splits/structured3d_all.txt \
  --raw-zips-dir "$DELIMIT3D_RAW_ZIPS" --geometry-root "$DELIMIT3D_GEOMETRY_ROOT" \
  --proposal-root "$DELIMIT3D_UNSAM_PROPOSALS" --output-root "$DELIMIT3D_SOURCE_ROOT" \
  --crop-mode full_scene
```

Use `full_scene` for the 84-epoch contrastive multiscale recipe; use `sphere_50k` for the hierarchy recipe. The fresh sphere chooses among eight seeds using projected 2D coverage. Its crop selection does not use 3D labels and is explicitly distinguished from the historical GT-assisted crop preparation. Exact earlier results require the original sealed packs.

## Seal and fix the evaluation input

Back in the student environment:

```bash
python scripts/prepare_data.py --action seal --source-root "$DELIMIT3D_SOURCE_ROOT" \
  --train-split "$DELIMIT3D_TRAIN_SPLIT" --holdout-split "$DELIMIT3D_VAL_SPLIT" \
  --output "$DELIMIT3D_SOURCE_AUDIT"
python scripts/prepare_data.py --action fixed-eval --teacher sam2 --grid-size 0.02 \
  --source-root "$DELIMIT3D_SOURCE_ROOT" --holdout-split "$DELIMIT3D_VAL_SPLIT" \
  --pretraining-contract configs/pretrain/sam2_selected_proposal_negatives.yaml \
  --output "${DELIMIT3D_FIXED_EVAL}_selected_proposal_negatives_100k"
python scripts/prepare_data.py --action code-record --output "$DELIMIT3D_CODE_MANIFEST"
```

For UnSAMv2 use `--teacher unsamv2`, and the training recipe's grid (.02 or .05). Sealing checks disjoint splits, scene identity, every artifact hash and the crop protocol. Fixed evaluation records native token plans. SAM2 admission tests the heldout scenes before publication; the historical pack admitted 322 of 326. A newly generated pack must report its own admission count.

For the overlap-veto comparison, prepare its separate pack from the same SAM2 source:

```bash
python scripts/prepare_data.py --action fixed-eval --teacher sam2 --grid-size 0.02 \
  --source-root "$DELIMIT3D_SOURCE_ROOT" --holdout-split "$DELIMIT3D_VAL_SPLIT" \
  --pretraining-contract configs/pretrain/sam2_overlap_veto.yaml \
  --output "$DELIMIT3D_FIXED_EVAL"
```

`--pretraining-contract` selects the negative policy from the recipe. An explicit conflicting policy flag is rejected. Each training recipe validates the saved policy, including deferred native-token rebuilding; the packs cannot be interchanged.

Generate the code record after editing recipes and before calibration/training. It includes source, configs, entry points, dependency patches, packaging and environment profiles.

The SAM2 fixed-evaluation command selects the epoch-zero geometry fragment before holdout admission, exactly as training evaluation does. With a pretraining contract, fixed evaluation uses the reference 500k limit even when training uses 100k. The pack records its evaluation cap and rejects incompatible reuse.

## ScanNet inputs for Mask3D

After downloading official ScanNet raw scans, combine the ordered scene lists and prepare geometry plus a separate GT cache:

```bash
cat configs/splits/scannet_train.txt configs/splits/scannet_val.txt > /tmp/delimit3d_scannet_all.txt
python scripts/prepare_data.py --action task-inputs --task mask3d \
  --raw-scans-root "$DELIMIT3D_SCANNET_ROOT" --scene-ids-file /tmp/delimit3d_scannet_all.txt \
  --output-root "$DELIMIT3D_MASK3D_SCANS" --gt-cache-root "$DELIMIT3D_MASK3D_GT_CACHE"
```

Mesh vertex order, RGB and area-weighted normals are kept together. Instance/segment targets come from official ScanNet aggregation, segmentation and label metadata. No teacher masks or CHORUS checkout are needed for task training.

For **low-data**, keep these dense inputs and use the fixed scene subsets made by
`scripts/limited_supervision.py prepare --protocol low-data`. The custom seed-42
subsets contain 12, 60, 120 or 240 scenes; they are separate from the official
ScanNet Data-Efficient limited-scene splits.

For **low-annotation**, run `scripts/prepare_low_annotation.py` on the raw ScanNet
meshes, these geometry packs and the official `points.zip` archive. It verifies
the archive hash and exact original mesh vertex order, then exports geometry,
only the selected training labels, and separate dense validation labels. The
training loader never receives a dense GT cache. Official duplicate index entries
are preserved; the budgets do not imply that every object is annotated.

See [the limited-supervision guide](limited_supervision.md) for the archive,
complete preparation command, verification and experiment generation. Existing
prepared output is immutable; choose a new directory for a different input set.

## ScanNet inputs for AGILE3D

Download the official processed benchmark described in the [pinned AGILE3D benchmark guide](https://github.com/ywyue/AGILE3D/tree/b73638da41edbabe52a1b578d52ddeb8fa552173/benchmark). Set the paths to its processed PLY files, training/validation sample JSON mappings, and SO `object_ids.npy` / `object_classes.txt` files. Keep the official object mappings and click indices.

```bash
python scripts/prepare_data.py --action task-inputs --task agile3d \
  --raw-scans-root "$DELIMIT3D_SCANNET_ROOT" --scene-ids-file /tmp/delimit3d_scannet_all.txt \
  --processed-scan-folder "$DELIMIT3D_AGILE3D_SCANS" --output-root "$DELIMIT3D_AGILE3D_NORMALS"
```

This verifies the processed PLY against the raw mesh and writes `normal.npy` plus the coordinate witness `coord.npy`. Training and evaluation verify per-row alignment and hash geometry, labels, normals, witness and ordered sample mappings. Replacing a sidecar or reordering points is rejected.

## Shared fresh decoder initialization

Create once and use the same template for all three backbone comparisons:

```bash
python scripts/prepare_data.py --action decoder-init --task mask3d --output "$DELIMIT3D_MASK3D_DECODER"
python scripts/prepare_data.py --action decoder-init --task agile3d --output "$DELIMIT3D_AGILE3D_DECODER"
```

The default seeds reproduce the earlier decoder templates: Mask3D seed 42, AGILE3D seed 20260912. The templates contain fresh decoder parameters. Native PTv3 input projections are replaced afterward using a separate CPU RNG stream.
