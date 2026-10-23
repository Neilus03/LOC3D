# LOC3D

LOC3D learns **object- and part-aware 3D representations from projected 2D masks**. The main recipe trains **LitePT-S*** from scratch on 2,930 Structured3D scenes, using frozen SAM2 proposals and a contrastive objective through a projection head on native dec0 features. Positives share a selected proposal; negatives lie outside it and may share another, broader proposal. The overlapping masks provide local relations rather than a global scene partition.

RGB, geometry and normals are encoder inputs; camera poses support projection during data preparation. No 3D object or part labels supervise pretraining. The projection head and 2D teacher are absent when reusing the encoder. Downstream supervised training and semantic classifiers use their own explicitly stated annotations.

**LOC3D is the current project name.** The Python package, installed commands, `DELIMIT3D_*` environment variables and legacy default artifact paths retain `delimit3d` for compatibility. Historical experiments keep their original names and recipes.

Start with the [research direction](docs/research.md) and [23 October experiment update](docs/research_update_20261023.md). The update distinguishes completed custom evaluations, frozen-feature diagnostics, pilot experiments and work without final results.

The repository also supports the **SAM2 overlap-veto comparison**, earlier **UnSAMv2 multigranular and multiscale recipes**, transfer through **Mask3D** and **AGILE3D**, and direct feature segmentation with cosine similarity or spherical K-means. Their results are not interchangeable with the main selected-proposal recipe.

## What is supported

| Workflow | Recipes |
| --- | --- |
| Pretraining | SAM2 dec0 with overlap veto or selected-proposal negatives; UnSAMv2 contrastive multiscale; UnSAMv2 hierarchy with or without VICReg |
| Mask3D | LitePT-S*, Concerto-small, Sonata-small; trainable or frozen encoders |
| AGILE3D | The same backbones; official single-object and multi-object click evaluation |
| Feature segmentation | Native features; cosine retrieval from user-selected points; spherical K-means |
| Trained-model inference | LitePT-S* + Mask3D automatic instances; LitePT-S* + AGILE3D foreground/background clicks |
| Browser workbench | Select registered checkpoints, load NPZ/PLY scenes, explore 3D predictions and download full-resolution results |

Concerto and Sonata are pretrained baselines. Their native PTv3 feature dimensions and task adapters are retained. A comparison between these models and LitePT-S* includes differences in architecture, pretraining and feature interfaces. Utonia Small appears in the research experiment records; its adapter and the newer low-annotation runners are not yet integrated into these portable commands.

## Repository layout

```text
configs/       shared model settings and small workflow recipes
scripts/       public preparation, training, evaluation and inference commands
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

## Commands

| Script | Installed command | Purpose |
| --- | --- | --- |
| `scripts/prepare_data.py` | `delimit3d-prepare` | Fetch pinned dependencies, prepare geometry/masks/data records and decoder initialization |
| `scripts/pretrain.py` | `delimit3d-pretrain` | Train or resume a pretraining recipe |
| `scripts/train.py` | `delimit3d-train` | Train or resume Mask3D/AGILE3D |
| `scripts/evaluate.py` | `delimit3d-evaluate` | Evaluate a full task checkpoint |
| `scripts/probe.py` | `delimit3d-probe` | Segment native features directly |
| `scripts/export_backbone.py` | `delimit3d-export` | Export a strict LitePT-S* backbone without training heads |
| `scripts/calibrate_bn.py` | `delimit3d-calibrate-bn` | Produce a separate checkpoint with recalibrated BN buffers |
| `scripts/infer.py` | `delimit3d-infer` | List trained models or predict masks on an unlabelled scene |
| `scripts/demo.py` | `delimit3d-demo` | Serve the local 3D segmentation workbench |

Every command has `--help`. Config inheritance merges mappings and replaces lists; explicit flags override defaults. Task settings can be overridden with `--set section.key=value`.

## Try trained models

The workbench runs beside a CUDA GPU, with model selection, 3D orbit/pan/zoom,
foreground/background clicks, multiple interactive objects, score filtering,
and prediction downloads. All browser assets are included; no CDN is required.
An optional [private hosted workbench](apps/inference-site/README.md) connects
to an outbound GPU worker, shows connection status and allocation expiry, and
lets the browser run inference without an SSH tunnel.

Copy [the model registry](configs/inference/models.example.yaml) and set each
entry to a **full trained task checkpoint** and its matching task configuration.
An encoder checkpoint alone does not contain a trained segmentation decoder.
Weights and licensed scenes are external files, not included in Git.

```bash
python scripts/infer.py list --registry /path/to/models.yaml
python scripts/demo.py --registry /path/to/models.yaml --example /path/to/scene.npz
# Open http://localhost:7860, or forward this port from your GPU host over SSH.

python scripts/infer.py predict --registry /path/to/models.yaml \
  --model unsamv2-mask3d --scene /path/to/scene.npz --output /path/to/prediction.npz
```

Scenes use ordered XYZ in metres, RGB and oriented surface normals. Mask3D also
accepts scan segments; without them it uses voxel segments and records this
change. Predictions are class-agnostic instance masks. Semantic labeling remains
a separate stage. See [inference and UI setup](docs/inference.md) for formats,
checkpoint compatibility, click input, GPU hosting and export details.

Preview recipes without data or GPU allocation:

```bash
python scripts/pretrain.py --config configs/pretrain/sam2_selected_proposal_negatives.yaml --show-config
python scripts/train.py --config configs/tasks/mask3d_concerto.yaml --show-config
```

After preparing and sealing data, launch the portable selected-proposal recipe:

```bash
torchrun --standalone --nproc_per_node=2 scripts/pretrain.py \
  --config configs/pretrain/sam2_selected_proposal_negatives.yaml
```

Use [sam2_overlap_veto.yaml](configs/pretrain/sam2_overlap_veto.yaml) for the overlap-veto comparison, with its separate fixed evaluation pack. Both recipes and resume requirements are described in [the pretraining guide](docs/pretraining.md#sam2-dec0). Existing runs must resume with their original policy.

The selected-proposal preset uses **100k raw points per training scene visit**, matching the epoch-150 checkpoint. Its fixed evaluation retains the reference 500k limit. The legacy overlap-veto preset keeps 500k training visits; comparing these two defaults does not isolate negative policy alone. See [recipe identities](docs/research.md#preserve-recipe-identity).

**Pretrained weights:** download the [SAM2 overlap-allowed, 100k, epoch-150 encoder](docs/weights.md) for feature extraction or downstream training. The download command verifies its SHA-256. These are encoder weights, not a trained Mask3D/AGILE3D decoder or a resumable optimizer checkpoint.

The full-scene recipes have substantial memory requirements. Follow the recipe-specific preparation and calibration steps before a production launch.

## Workflow guides

- [Research direction](docs/research.md): current method, hypotheses and comparison boundaries.
- [23 October experiment update](docs/research_update_20261023.md): verified evidence and exact experiment identities.
- [Prepare data](docs/data.md): 2D-only Structured3D packs, fixed evaluation and ScanNet inputs.
- [Download pretrained weights](docs/weights.md): exact epoch-150 identity, checksums and loading.
- [Pretrain](docs/pretraining.md): SAM2 and both earlier UnSAMv2 objectives, calibration, export and resume.
- [Train and evaluate tasks](docs/tasks.md): native adapters, frozen/trainable controls, losses and click protocols.
- [Feature segmentation](docs/probes.md): cosine retrieval and spherical K-means without task decoders.
- [Reproduce earlier experiments](docs/reproduction.md): historical recipes, feature pipelines and checkpoint boundaries.
- [Release migration](docs/migration.md): what was retained, removed and preserved.
- [11–17 October update](docs/weekly_update_20261017.md): current completed evidence, experiment contracts and release scope.

Run status is written beside each run's resolved recipe, checkpoints and metrics. [CURRENT.md](CURRENT.md) describes the release's verification record; it is not a live research dashboard.

## Licenses

Original project code retains the license status described in [LICENSE.md](LICENSE.md). Copied and external components retain their upstream terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
