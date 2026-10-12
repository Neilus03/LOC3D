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
