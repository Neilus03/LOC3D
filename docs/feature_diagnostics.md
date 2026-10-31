# Frozen-feature diagnostics on common scene support

`python scripts/feature_diagnostics.py` exposes the reusable parts of the
ScanNet20/ScanNet200 full-validation diagnostic work. It prepares verified
original-vertex targets and evaluates either supplied native features or a
backbone from an existing probe recipe. The default common grid is 5 cm.

These are **GT-K clustering and GT-query cosine diagnostics**, not instance AP.
They use annotated object count or query locations. Keep them separate from
class-aware AP, class-agnostic AP, interactive IoU/NoC and semantic mIoU.
`probe.py` remains the simpler command for user-supplied K or queries.

## Prepare targets

The input scene directory contains `coord.npy`, `color.npy`, `normal.npy`,
`instance.npy` and `segment20.npy` or `segment200.npy`. The label convention is
Pointcept's contiguous semantic indices and zero-based raw instance IDs, with
`-1` ignored. Supply the original ScanNet mesh to verify vertex order.

```bash
python scripts/feature_diagnostics.py prepare \
  --taxonomy scannet200 --scene scene0000_00 \
  --source /data/processed/val/scene0000_00 \
  --mesh /data/scans/scene0000_00/scene0000_00_vh_clean_2.ply \
  --output /data/diagnostic_targets/scannet200/scene0000_00

python scripts/feature_diagnostics.py prepare \
  --taxonomy scannet20 --scene scene0000_00 \
  --source /data/processed/val/scene0000_00 \
  --mesh /data/scans/scene0000_00/scene0000_00_vh_clean_2.ply \
  --geometry-parent /data/diagnostic_targets/scannet200/scene0000_00/target.json \
  --output /data/diagnostic_targets/scannet20/scene0000_00
```

ScanNet20 retains all 20 recognized classes, including **separate annotated
wall and floor raw instances**. ScanNet200 excludes wall and floor, leaving
198 eligible classes. Instance zero is valid. The default minimum target size
is one vertex; the 100-vertex flag is reported separately. Selecting
`--min-instance-points 100` changes K and the target protocol; it does not make
this an AP evaluation. `--grid-size` also changes the protocol.

Every source point remains in the network input, common grid and scoring
support, including ignored classes and filtered instances. The ScanNet20
`--geometry-parent` route verifies and reuses exactly the same geometry,
normals and cell mapping as ScanNet200. Targets record source hashes, original
coordinate identity, class/instance membership, query vertex and cell map.
A different source cannot silently replace an existing sealed target.

## Evaluate from a backbone recipe

Edit an existing `configs/probes/*.yaml` recipe with your local model paths.
Then run:

```bash
python scripts/feature_diagnostics.py evaluate \
  --targets /data/diagnostic_targets/scannet20/scene0000_00/target.json \
  --config configs/probes/litept.yaml --device cuda \
  --normal-source pointcept_scannet_mesh_normals \
  --output /data/diagnostics/litept/scene0000_00
```

This uses the release's ordered RGBN6 extraction bridge, with the recipe's
native grid, coordinate and BatchNorm policy. It supports the same LitePT-S*,
Concerto Small and Sonata Small models as `probe.py`. This extraction route is
a **new experiment** relative to historical runs that used independently
pinned, model-specific preprocessing. Matching common cells alone does not
make those extraction pipelines identical. CPU tests cover cached evaluation;
the refresh did not launch GPU extraction or validate a benchmark result.

## Evaluate a native feature cache

A cache is an NPZ containing:

| Key | Shape | Meaning |
| --- | --- | --- |
| `features` | `[T,C]` | Native descriptors before common-cell pooling |
| `original_to_token` | `[N]` | Integer native-token index for every original vertex |
| `points` | `[N,3]` | Original coordinates in exactly the target's vertex order |

Use a companion JSON to record model, checkpoint hash and epoch, native grid,
feature width/source, RGB/normal convention, coordinate policy, and any
frozen-state hash attestation. Those are supplied provenance; this command
cannot independently prove that an externally produced cache came from a
frozen model. It verifies cache/target hashes and exact coordinate ordering.

```bash
python scripts/feature_diagnostics.py evaluate \
  --targets /data/diagnostic_targets/scannet20/scene0000_00/target.json \
  --features /data/features/scene0000_00.npz \
  --feature-provenance /data/features/scene0000_00.json \
  --normal-source pointcept_scannet_mesh_normals --device cpu \
  --output /data/diagnostics/cached/scene0000_00
```

For CPU workloads, set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to a sensible
allocation. The implementation uses sparse pooling and never materializes an
`[N,C]` expanded native-feature array. Cosine scores and clustering use dense
`[common cells,K]` matrices, so memory still grows with scene size and K.

## Protocol and output

1. Map native tokens to occupied common cells by **original-point-weighted**
   averaging; a token inherited by two points contributes twice. Normalize
   only after this pooling. Different native tokenizations share common cells.
2. Fit spherical k-means to **all occupied cells, including background**.
   K is the eligible GT raw-instance count. The five default fixed seeds are
   `20261010` through `20261014`. Choose the restart by mean cosine to center;
   GT IoU never selects a restart. Fits use equal weight per common cell.
3. Expand the selected assignments back to **all original vertices**. Hungarian
   matching maximizes total instance IoU after fitting; background false
   positives remain in every union. Metrics use original-point weighting.
4. For cosine, use each instance's centroid-nearest existing GT vertex,
   mapped to its common cell, and assign every cell to its most similar query.
   There is no background rejection or Hungarian reassignment. Query collisions
   are retained; the first query wins exact ties.

`diagnostics.npz` saves the normalized common-cell features, original points,
RGB, normals, GT labels, queries, mappings, all restart centers/assignments,
selected restart, Hungarian mapping and cosine scores. `features` and
`prediction` are **cell-domain** arrays; `inverse_map` expands both to original
points. `point_prediction` stores the already expanded k-means labels.
This layout can be used by the [visualization utilities](visualization.md).

`metrics.json` records the fit scope, normal-source declaration, complete
restart metadata, per-object/scene metrics, and input/extraction provenance.
`COMPLETE.json` hashes both outputs and explicitly reports **one evaluated
scene**, without claiming a completed validation split. Output directories
must be new. To report full-validation results, independently check the exact
split, absence of duplicate scenes, every input/checkpoint identity and every
completion hash before aggregating. `feature_diagnostics.aggregate` reports
scene means, pooled object means, represented-class means and a separate
100-vertex subset; that subset does not refit clusters or change K.

CPU/CUDA arithmetic and extraction preprocessing can change a fit. Record the
actual device, checkpoint and native grid as well as the common grid. A
single result does not isolate pretraining effects across different backbones.

## Cross-dataset inputs and recent experiments

The Python `evaluate_features` API accepts arbitrary datasets with contiguous
eligible object IDs, one query per object, full-scene point maps and explicit
object metadata. The CLI target adapters currently cover ScanNet20/200.
S3DIS and ScanNet++ require their own verified target and input preparation;
this release does not claim a complete automated adapter for either dataset.

For the October S3DIS comparison, full-room inputs and **Pointcept reference
normals** were used for both models. Reference normals were checked against
source point/color ordering and rigidly transformed into the source frame.
The earlier outward-centroid estimates were superseded. Later interactive
crop inputs used `inward_native_center_v1`: inward estimated normals with
centering before float32 conversion. These are distinct normal protocols;
matching a grid or negating normals is insufficient to establish equivalence.
Released S3DIS crop IDs are zero-based and normally map to positive full-room
labels as `crop_id + 1`, but IDs can be noncontiguous and some room labels are
unlisted. Verify and record those exclusions instead of assuming contiguity.

The October 30 native-2-cm/pooled-5-cm pilot used a **GT foreground mask before
fitting and scoring**. Its graph variant also supplied spatial connectivity
and geometry-only bridges. Those protocols differ from the full-scene code
here; their local experiment wrappers and numerical results are not included.
Changing `--grid-size` here keeps background in the fit and does not reproduce
that foreground-only pilot. Native Utonia extraction and the experiment-specific
Slurm/host controllers remain outside this portable CLI.

The newer native-2-cm/native-5-cm full-validation experiment additionally defines
a ScanNet20 object-only target panel excluding wall/floor. That is distinct from
the existing ScanNet20 target builder released here; this CLI does not silently
substitute that target set or claim its scores.
