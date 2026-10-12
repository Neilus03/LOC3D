# Delimit3D

Delimit3D is a **contrastive pretraining strategy for 3D representations using projected 2D masks**. The main recipe trains **LitePT-S*** from scratch with SAM2 supervision. RGB, geometry, normals and camera poses are inputs; 3D object or part labels are not training supervision.

The repository also supports the earlier **UnSAMv2 multigranular and multiscale recipes**, transfer through **Mask3D** and **AGILE3D**, and direct feature segmentation with cosine similarity or spherical K-means.

## What is supported

| Workflow | Recipes |
| --- | --- |
| Pretraining | SAM2 dec0 with overlap veto or selected-proposal negatives; UnSAMv2 contrastive multiscale; UnSAMv2 hierarchy with or without VICReg |
| Mask3D | LitePT-S*, Concerto-small, Sonata-small; trainable or frozen encoders |
| AGILE3D | The same backbones; official single-object and multi-object click evaluation |
| Feature segmentation | Native features; cosine retrieval from user-selected points; spherical K-means |

Concerto and Sonata are pretrained baselines. Their native PTv3 feature dimensions and task adapters are retained. A comparison between these models and LitePT-S* includes differences in architecture, pretraining and feature interfaces.

## Repository layout

```text
configs/       shared model settings and small workflow recipes
scripts/       seven public commands
src/delimit3d/ data, pretraining, backbone adapters, tasks and probes
environment/   versioned runtime profiles and installation instructions
tests/         scientific contract checks and opt-in CUDA checks
docs/          data, recipes, task protocols and migration notes
manifests/     source extraction, splits and release records
third_party/   upstream license texts
tools/reproduce/ limited historical experiment entry points
```

Data, checkpoints, dependency checkouts and run output live outside versioned code.

## Install

Use Python 3.12 and the CUDA 12.4 profile in [environment/README.md](environment/README.md). Install the package after the CUDA dependencies:

```bash
python -m pip install --no-deps -e .
```

Copy [env.example](env.example) to `.env`, edit the paths and source it from the repository root. It contains portable example paths. SAM2 and UnSAMv2 inference use separate teacher environments.

## Seven commands

| Script | Installed command | Purpose |
| --- | --- | --- |
| `scripts/prepare_data.py` | `delimit3d-prepare` | Fetch pinned dependencies, prepare geometry/masks/data records and decoder initialization |
| `scripts/pretrain.py` | `delimit3d-pretrain` | Train or resume a pretraining recipe |
| `scripts/train.py` | `delimit3d-train` | Train or resume Mask3D/AGILE3D |
| `scripts/evaluate.py` | `delimit3d-evaluate` | Evaluate a full task checkpoint |
| `scripts/probe.py` | `delimit3d-probe` | Segment native features directly |
| `scripts/export_backbone.py` | `delimit3d-export` | Export a strict LitePT-S* backbone without training heads |
| `scripts/calibrate_bn.py` | `delimit3d-calibrate-bn` | Produce a separate checkpoint with recalibrated BN buffers |

Every command has `--help`. Config inheritance merges mappings and replaces lists; explicit flags override defaults. Task settings can be overridden with `--set section.key=value`.

Preview recipes without data or GPU allocation:

```bash
python scripts/pretrain.py --config configs/pretrain/sam2_overlap_veto.yaml --show-config
python scripts/train.py --config configs/tasks/mask3d_concerto.yaml --show-config
```

After preparing and sealing data, launch the main pretraining recipe:

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/sam2_overlap_veto.yaml
```

Use [sam2_selected_proposal_negatives.yaml](configs/pretrain/sam2_selected_proposal_negatives.yaml) for the alternative negative policy, with its separate fixed evaluation pack. Both recipes and resume requirements are described in [the pretraining guide](docs/pretraining.md#sam2-dec0).

The full-scene recipes have substantial memory requirements. Follow the recipe-specific preparation and calibration steps before a production launch.

## Workflow guides

- [Prepare data](docs/data.md): 2D-only Structured3D packs, fixed evaluation and ScanNet inputs.
- [Pretrain](docs/pretraining.md): SAM2 and both earlier UnSAMv2 objectives, calibration, export and resume.
- [Train and evaluate tasks](docs/tasks.md): native adapters, frozen/trainable controls, losses and click protocols.
- [Feature segmentation](docs/probes.md): cosine retrieval and spherical K-means without task decoders.
- [Reproduce earlier experiments](docs/reproduction.md): historical recipes, feature pipelines and checkpoint boundaries.
- [Release migration](docs/migration.md): what was retained, removed and preserved.

Run status is written beside each run's resolved recipe, checkpoints and metrics. [CURRENT.md](CURRENT.md) describes the release's verification record; it is not a live research dashboard.

## Licenses

Original project code retains the license status described in [LICENSE.md](LICENSE.md). Copied and external components retain their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
