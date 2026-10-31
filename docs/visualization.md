# Saved-feature and prediction visualization

The visualization suite renders saved arrays as offline interactive point/mesh
galleries, PNG/PDF panels, presentation contact sheets, colored PLY files, and JSON
records. Shared original point IDs keep the scene, queries, caches, and rendered
subsamples aligned. Rendering saved features runs on CPU.

```bash
pip install -e '.[plots]'
python scripts/visualization/run_suite.py --demo --cpu-only \
  --output artifacts/visualization-demo --formats html png pdf
# Installed entry point: delimit3d-visualize
```

Open the generated `index.html` directly. It needs no server or CDN. The demo is
explicitly synthetic and contains no experimental results. Every run requires a
fresh output directory. `--cpu-only` rejects inference adapters, including custom
callables. PNG/PDF rendering needs the `plots` extra; HTML/PLY does not.

## Cached features

Start with [cached.example.yaml](../configs/visualization/cached.example.yaml).
Paths resolve relative to the manifest and support environment variables and `~`.
Model paths can contain `{scene_id}` and `{dataset}`. Scene-specific model lists
override the global models. Prepared NPZ, NPY, PLY, and array-directory readers
cover ScanNet, ScanNet++, Structured3D, S3DIS, and Articulate3D exports; they do not
download or reinterpret raw dataset annotations.

```python
import numpy as np

# All rows refer to the same original point order.
np.savez_compressed("scene.npz", points=xyz, colors=rgb, labels=instance_ids)
np.savez_compressed("encoder.npz", points=xyz, features=point_features,
                    checkpoint_sha256=np.asarray(checkpoint_digest))
```

```yaml
title: Frozen representation comparison
cpu_only: true
formats: [html, png, pdf]
seed: 42
queries: [1234]  # Original point IDs, shared across models.
visualizations: [rgb, labels, pca, similarity]
research:
  features: {enabled: true}
scenes:
  - scene_id: room_1
    dataset: S3DIS
    path: scene.npz
    metadata:
      coordinate_units: meters
      background_ids: [0]
      ignore_label_ids: [-1]
      normal_protocol: reference_normals_from_your_verified_export
models:
  - name: LOC3D
    kind: cache
    path: encoder.npz
    features: {dec0: features}
```

```bash
python scripts/visualization/run_suite.py --config /path/to/study.yaml \
  --cpu-only --output artifacts/study --formats html png pdf ply
```

Scene `points` must be finite N×3 coordinates. Optional arrays are `colors` N×3,
`normals` N×3, `labels` N integer instance/part IDs, and `semantic_labels` N integer
IDs. RGB accepts uint8 or floats in [0,1]; other conventions need an explicit
`color_range`. Label zero is valid unless declared background. Use
`metadata: {label_kind: part}` for part labels. Negative labels are ignored.

Feature matrices are N×C. Token/cell caches instead contain T×C features and an
N-length integer `inverse_map`; all prediction/confidence arrays in that cache
must use the same T-row domain. Different native stage resolutions need their
own mapping before combining them into one export. Matching point counts alone
are insufficient: an ordered-geometry array/hash, verified shifted coordinates,
source-file hash, or explicit geometry witness is required. NPZ reading disables
pickle. Only load trusted local PyTorch archives.

## Recent frozen-feature diagnostics

The [feature diagnostic workflow](feature_diagnostics.md) pools native features
onto a common grid, runs spherical k-means with ground-truth K, and selects its
restart using the feature objective. Its `diagnostics.npz` can be used directly:

```yaml
title: Oracle-K frozen-feature diagnostic
cpu_only: true
formats: [html, png, pdf]
queries: [1234]  # Copy an original-point query from the target record.
visualizations: [rgb, labels, pca, similarity, prediction]
research:
  features: {enabled: true}
scenes:
  - scene_id: scene0000_00
    dataset: ScanNet
    path: /path/to/evaluation/diagnostics.npz
    color_range: 0_255
    metadata: {ignore_label_ids: [-1], coordinate_units: meters}
models:
  - name: LOC3D oracle-K diagnostic
    kind: cache
    path: /path/to/evaluation/diagnostics.npz
    features: {dec0: features}
```

This is a selected-scene feature diagnostic. Its clustering, query retrieval, and
rendered masks are not instance AP or learned segmentation predictions. Use the
evaluator's metrics/provenance for the declared population; gallery retrieval
statistics use its own shared sampled raw points. Both input grid and normal
source/orientation belong to the protocol. Reference mesh normals and inward
estimated crop normals cannot be silently mixed. S3DIS released crop IDs need an
explicit mapping to full-room IDs, with unlisted instances recorded as excluded.
The suite supports prepared ScanNet++ inputs; this release does not claim a new
completed ScanNet++ experiment.

## PCA, stage inspection, and query similarity

The research feature mode saves the numerical PCA state (mean, basis, fit point
IDs, color limits, hashes). Fits are independent per source/layer by default.
Reuse a saved state through `research.features.pca_states`; requested exact
checkpoint/epoch/query identities are checked against artifact metadata. See
[features.research.example.yaml](../configs/visualization/features.research.example.yaml).
PCA colors alone do not establish object correspondence. Query cosine panels use
[-1,1], differences use [-2,2], and zero-feature directions are marked undefined.

The stage adapter carries the reusable part of the recent concatenated-descriptor
PCA inspection. Supply architecture-verified channel widths in stored order:

```yaml
models:
  - name: Saved stage magnitudes
    kind: cache
    path: /path/to/concatenated.npz
    feature_stages:
      key: features
      widths: {stage1: 36, stage2: 72, stage3: 144, stage4: 288, stage5: 432}
      source_normalization: joint_l2
      normalization: as_saved
  - name: Unit directions per stage
    kind: cache
    path: /path/to/concatenated.npz
    feature_stages:
      key: features
      widths: {stage1: 36, stage2: 72, stage3: 144, stage4: 288, stage5: 432}
      source_normalization: joint_l2
      normalization: per_stage_l2
```

The widths above illustrate a 972-D export; they are not inferred from a model
name. `as_saved` preserves each stored block; `per_stage_l2` discards its magnitude.
PCA states cannot cross those normalization variants. The gallery fits PCA on
original-point rows after inverse-map expansion, with a recorded fit sample; this
is a different weighting from fitting every common cell equally. It does not
reproduce a historical all-cell figure merely by matching its stage widths.
Standalone array use is available through `split_feature_stages` in
`delimit3d.visualization.stages`.

## Recorded predictions and surfaces

[agile3d.recorded.example.yaml](../configs/visualization/agile3d.recorded.example.yaml)
shows recorded episodes with stable target IDs and original clicks. Encoder and
decoder pairing is recorded explicitly. Native adaptive click sequences are
comparable as identical-click experiments only when the sequence check verifies
their actual identities.

[mask3d.mask_only.example.yaml](../configs/visualization/mask3d.mask_only.example.yaml)
preserves independent Mask3D proposals, original query IDs, scores, and evaluator
postprocessing. Visualization overlap resolution remains separate from benchmark
matching. `save_episode_export` and `save_proposal_export` in
`delimit3d.visualization.export_contracts` write portable, hashed bundles.

```bash
python scripts/visualization/create_research_fixture.py --output artifacts/fixture
python scripts/visualization/run_suite.py \
  --config artifacts/fixture/synthetic.validation.yaml --cpu-only \
  --output artifacts/fixture-gallery --formats html png pdf
```

Original meshes need triangular faces and explicit vertex-to-point correspondence.
Use `mesh: {kind: scene}` when faces are embedded in the scene, or
`mesh: {kind: original, path: mesh.ply}` for a verified original mesh. Exports
preserve topology and original point IDs. Static triangle rendering and WebGL
depth visibility can differ for intersecting triangles; this is recorded.

Named `recipes` select panels, cameras, crops, paper/presentation sizes, and
captions. Browser orbit/pan/height controls inspect stored values; they do not run
the encoder. Camera and current-view exports include provenance.

## Artifact utilities

- `audit_artifacts.py`: read-only inventory and NPZ header inspection.
- `stage_saved_inputs.py`: hash-verified copies of explicit saved input paths and
  episode/proposal bundles. Use concrete paths; expand model templates first.
- `assemble_collection.py`: combine completed galleries into a fresh portable
  collection, optionally refresh viewer controls while preserving payload bytes.
- `check_browser.py`: optional Playwright/Chromium checks for offline operation,
  controls, and missing/network assets. Install Playwright separately when used.

Each script supports `--help`. Generated galleries, datasets, checkpoints, and
historical figure outputs stay outside Git. `--demo` and the contract fixture are
the data-free CPU checks included in the public release; direct model inference
requires the corresponding model environment and hardware.
