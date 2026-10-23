# Direct feature segmentation

`probe.py` uses native backbone features without Mask3D or AGILE3D decoders. It accepts either an existing feature NPZ, or a scene NPZ and a backbone recipe.

## From features

The NPZ must contain `features` with shape [native tokens, channels]. Optional `original_to_token` maps every original point row to its token.

```bash
python scripts/probe.py --features /path/to/features.npz --method cosine \
  --queries 12,341 --output /path/to/cosine.npz
python scripts/probe.py --features /path/to/features.npz --method kmeans \
  --k 20 --seed 42 --restarts 5 --output /path/to/kmeans.npz
```

`--queries` are original point row indices. Cosine retrieval normalizes features and assigns each token to its most similar query. Spherical K-means normalizes features and centroids; restarts are selected by the clustering objective. K is supplied by the user. Neither method reads object/part labels to choose queries, K or the winning restart.

Use `--device cpu` for feature-only probes without CUDA.

## From a point cloud

The scene NPZ must contain ordered `points`, `colors` and `normals`, each [N,3]. RGB may be [0,1] or [0,255].

```bash
python scripts/probe.py --scene /path/to/scene.npz --config configs/probes/litept.yaml \
  --method kmeans --k 20 --output /path/to/segments.npz
```

Use `configs/probes/concerto.yaml` or `sonata.yaml` for the native baselines. The default bridge uses RGBN6, first representatives, the configured grid and `center_xy_min_z` coordinates. LitePT-S* returns 72D dec0; native PTv3 returns the 992D finest descriptor.

Output labels are expanded to original point rows. The companion JSON records the method, seed/restarts or queries, grid, coordinate policy, feature width, checkpoint and normalization policy.

## Comparing with older results

The simple probe uses the ordered downstream input bridge. Earlier five-model ScanNet feature results used separate official model-specific preprocessing, grids, checkpoints and maturity. Their historical evaluator remains in `tools/reproduce/run_feature_segmentation_scannet10.py`. Preserve those exact settings when reproducing their numbers; a result from this simpler bridge is a new experiment.

## Fixed-instance semantic readout helpers

`delimit3d.evaluation.linear_instance_classifier` exposes the CPU helpers used
to attach closed-vocabulary ScanNet18 labels to fixed object masks. This is a
Python API for prepared arrays and an already fitted linear head; it does not
train the head, extract backbone features, or run an RGB/CLIP pipeline.

```python
from delimit3d.evaluation.linear_instance_classifier import (
    CLASS_IDS, pool_reference, probabilities,
)

# Features and Boolean masks must use the same original point ordering.
descriptors = pool_reference(point_features, instance_masks)
scores = probabilities(descriptors, weights, intercept, train_mean, train_scale)
predicted_class_ids = CLASS_IDS[scores.argmax(axis=1)]
```

`pool_reference` concatenates the vertex-weighted mean and channelwise maximum
within each mask: 72D point features yield a 144D descriptor. Expand token
features to original point rows before pooling. Supply the fitted head and
normalization from training data; evaluation labels must not enter prediction.

`geometry_matches` performs class-blind maximum-total-IoU matching.
`official_match_record` preserves predicted mask extent, false positives and
small ground truths when building ScanNet18 evaluator records. `reference_ap`
accepts a separately supplied, pinned ScanNet18 reference evaluator. These
helpers alone do not establish official benchmark compliance. Accuracy on
geometrically matched objects excludes geometric misses and is distinct from
whole-scene semantic instance AP, the 3D semantic linear probe, and RGB-assisted
semantic labeling.

The release's unified task/probe commands support LitePT-S*, Concerto Small and
Sonata Small. Utonia and the recent low-data/low-annotation runner extensions
remain research implementations outside those commands. Their experiment
evidence can be documented without implying a portable command-line workflow.
The code audit and transferred-file hashes are recorded in
[`research_code_refresh_20261023.json`](../manifests/research_code_refresh_20261023.json).
