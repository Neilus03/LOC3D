# SAM2 overlap-allowed epoch-150 encoder weights

[Download the encoder](https://github.com/Neilus03/LOC3D/releases/download/sam2-overlap-allowed-100k-e150/loc3d_sam2_overlap_allowed_100k_e150.pt) · [Release and checksums](https://github.com/Neilus03/LOC3D/releases/tag/sam2-overlap-allowed-100k-e150) · [Machine-readable identity](../manifests/loc3d_sam2_overlap_allowed_100k_e150.json)

This is the requested `sam2-overlap_allowed` model: **selected-proposal negatives**, with `allow_shared_proposal_negatives=true`. It was trained from scratch on Structured3D using projected SAM2 masks. These are native LitePT-S* encoder weights for feature extraction and downstream initialization.

| Property | Value |
| --- | --- |
| Checkpoint | Epoch 150, update 36,630, seed 42 |
| Training point limit | 100,000 raw points per scene visit |
| Fixed-evaluation point limit | 500,000 raw points per scene visit |
| Input | RGBN6: RGB channels followed by xyz surface normals |
| Pretraining voxel grid | 0.02 m |
| Output | Native dec0 features, 72 dimensions |
| Contents | 302 encoder tensors, including unchanged raw BatchNorm buffers |
| Download size | 64,073,536 bytes |

The export excludes the contrastive projector, optimizer and task decoders. It **cannot resume pretraining** or perform trained Mask3D/AGILE3D inference by itself. Downstream training creates a fresh task decoder. The original training checkpoint and this tensor-only export have different file hashes by design.

## Download and use

After following [environment setup](../environment/README.md), download with a mandatory SHA-256 check:

```bash
export DELIMIT3D_BACKBONE_CHECKPOINT="$DELIMIT3D_WORK_ROOT/weights/loc3d_sam2_overlap_allowed_100k_e150.pt"
python scripts/prepare_data.py --action weights \
  --weight loc3d_sam2_overlap_allowed_100k_e150 \
  --output "$DELIMIT3D_BACKBONE_CHECKPOINT"
```

For direct feature segmentation, supply a scene NPZ with point coordinates in meters and aligned `points`, `colors` and `normals` arrays, each [N,3]. RGB may be [0,1] or [0,255]; the public bridge normalizes it, centers XY and sets minimum Z to zero. The dedicated recipe uses the 2 cm pretraining grid, first voxel representatives and the saved raw BatchNorm statistics:

```bash
python scripts/probe.py --scene /path/to/scene.npz \
  --config configs/probes/loc3d_sam2_100k_e150.yaml \
  --method kmeans --k 20 --output /path/to/segments.npz
```

Native point-cloud inference requires the LitePT CUDA environment. This example is a feature probe with user-selected K, not an official benchmark result. See [probes](probes.md) for query-based alternatives and protocol boundaries. For [downstream training](tasks.md), keep the same checkpoint environment variable and use the LitePT-S* task recipes. Their 5 cm grids and task-specific normalization are deliberate downstream settings, distinct from pretraining.

## Identity and verification

Export SHA-256:

```text
5f84913185468868c14f793bf548cfb6a192238d3ac9feb0c98a59ad4ba31d5a
```

Original full training checkpoint SHA-256:

```text
bd11b9c19e21b4c220bd23bff0402fb73ad9a86197ef4ee5c3e4ee81229a447c
```

Every exported tensor was compared exactly with the original checkpoint's encoder and its full-model encoder entries, then reloaded with `torch.load(..., weights_only=True)`. All 302 tensors loaded strictly into the public LitePT-S* wrapper with matching names, shapes, dtypes and values. No recalibration or numerical conversion was applied. No GPU forward pass or fresh end-to-end training was performed for this publication.

The original experiment used a preserved research checkout. The current portable recipe now implements its 100k training / 500k evaluation limits and original fragment seed convention; publishing weights does not establish bit-for-bit retraining reproduction from newly prepared data. Historical results retain their original data, code, task and evaluation identities.

The repository currently specifies no blanket license grant. Publication does not add a separate license for these weights; consult [LICENSE.md](../LICENSE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
