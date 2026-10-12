# Reproducing earlier experiments

The supported public workflows live in seven scripts. A small historical section preserves experiments with different contracts instead of exposing every dated launcher as another workflow.

## Included historical entry points

| Entry | Purpose |
| --- | --- |
| `tools/reproduce/fixed9_pretrain.py` | Original fixed-nine, common-eligible-subset six-arm UnSAMv2 ablation |
| `tools/reproduce/run_feature_segmentation_scannet10.py` | Original direct feature cosine/K-means comparison with model-specific preprocessing |
| `tools/reproduce/run_articulate3d_parts.py` | Historical Articulate3D part evaluator |
| `configs/reproduce/sam2_pf69_2gpu_original.json` | Exact original main SAM2 recipe bytes |
| `configs/reproduce/unsamv2_g05_dec0_336ep_original.json` | Earlier single-granularity dec0 recipe |

The fixed-nine ablation preserves its quota, cell/stage routes, 4883-update schedule and geometry admission gates. Its one-GPU/local-BN contract differs from the main two-rank/SyncBN hierarchy recipe. Its complete contract, geometry report and initialization are required artifacts; `--help` describes the paths. Queue launchers are intentionally outside the release.

## Exact results require exact artifacts

Retain the source pack and frame/scene splits, normal preparation, source/code records, initial and selected checkpoints, optimizer schedule, grids, feature readout, task input bridge, evaluation normalization and protocol coverage. New 2D-only preparation does not reproduce the historical GT-assisted sphere selection. A new crop, checkpoint age or feature pipeline must receive its own result record.

## Historical checkpoints

Native LitePT-S* pretraining backbone tensors can be exported through the common strict loader. Earlier full Mask3D and AGILE3D task checkpoints can be evaluated when their architectures and input projections match. Frozen AGILE3D decoder-only files require the exact encoder that originally generated their features.

The release resume format includes data hashes and recipe checks that earlier task checkpoints do not carry. An old checkpoint is not assigned a new training history by changing its schema string. Continue historical runs in their preserved original checkout, or review a conversion that verifies model tensors, parameter-group order, every Adam counter, scheduler horizon, data identity and RNG/epoch frontier.

## Records

`manifests/pretraining_refactor.json`, `mask3d_extraction.json`, `agile3d_extraction.json` and `calibration_extraction.json` identify extracted definitions and source hashes. Changes needed for packaging, portable data preparation, common CLI entry points and strict records are listed in the release migration notes. `release_curation.json` identifies removed files and their preserved content hashes.

Public baseline sources and weight hashes are packaged under `src/delimit3d/runtime/assets/`. Decoders are freshly initialized; official Mask3D pretrained decoder weights are not used in these transfer recipes.
