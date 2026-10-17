# Trained-model inference and 3D workbench

Use a **full downstream checkpoint** containing both LitePT-S* and its trained
Mask3D or AGILE3D decoder. The supported automatic model predicts class-agnostic
instances; the interactive model predicts objects selected with clicks. The
pretraining teacher (UnSAMv2 or SAM2) is not loaded during inference.

## Prepare a model registry

Install the [CUDA student environment](../environment/README.md), then copy
[`models.example.yaml`](../configs/inference/models.example.yaml) to a local
file and replace its paths. For each entry, use the **resolved configuration
from that trained run** and its full final or intermediate task checkpoint.
Relative paths resolve against the registry directory; `${ENVIRONMENT_VARIABLES}`
are supported. Multiple models can reference the same task configuration.

```yaml
schema: delimit3d_model_registry/v1
models:
  - id: unsamv2-mask3d
    name: UnSAMv2 pretrained · Mask3D
    task: mask3d
    config: /models/mask3d/resolved_config.yaml
    checkpoint: /models/mask3d/epoch_0600.pt
    config_overrides:
      model:
        litept_root: /dependencies/LitePT
      eval:
        backbone_batchnorm_policy: train_no_update
  - id: unsamv2-agile3d
    name: UnSAMv2 pretrained · AGILE3D
    task: agile3d
    config: /models/agile3d/resolved_config.json
    checkpoint: /models/agile3d/checkpoint1100.pth
    config_overrides:
      paths:
        litept_root: /dependencies/LitePT
      evaluation:
        backbone_batchnorm_policy: train_no_update
```

The inference adapters support LitePT-S* RGBN6 and RGB3 full task checkpoints,
including the historical `model_state_dict` (Mask3D) and `model` (AGILE3D)
containers. All model tensors load strictly. Decoder-only frozen checkpoints,
encoder-only checkpoints, semantic heads, and Concerto/Sonata task checkpoints
are not supported by these inference adapters. Their training/evaluation
workflows remain available separately. Original initialization checkpoints,
optimizer states, training datasets and decoder templates are unnecessary for
full-task inference.

Inference selects FlashAttention on compatible Ampere-or-newer GPUs and the
existing Torch SDPA implementation on older CUDA GPUs. If the installed
PointROPE extension was compiled for a different GPU architecture, the pinned
LitePT dependency also supports `POINTROPE_FORCE_TORCH=1`, which computes the
same positional encoding with CUDA tensors. Set it before starting the process;
it does not require rebuilding or modifying the shared training environment.
The prediction provenance records the attention and PointROPE implementations.

Register scratch and pretrained checkpoints as separate entries for model
selection. A SAM2 entry is appropriate only when it points to an actual trained
SAM2-initialized downstream model. Displaying a model choice is not evidence
that its checkpoint finished training or outperforms another model.

`list` checks configured file availability without importing Torch or allocating
a GPU. The first prediction validates the actual architecture, checkpoint and
native CUDA dependencies. Registry checkpoints are trusted local files;
the browser accepts only scenes.

```bash
python scripts/infer.py list --registry /path/to/models.yaml
```

## Scene format

NPZ inputs contain these arrays in the **same original point order**:

| Array | Shape / meaning |
| --- | --- |
| `points` | float N×3 XYZ in metres |
| `colors` | uint8 N×3 RGB in [0,255], or floating RGB in [0,1] |
| `normals` | float N×3 unit surface normals, required by RGBN6 models |
| `segments` | optional nonnegative integer N-vector of scan segment IDs |

Do not pass training RGB already scaled to [-1,1]. The adapter applies the
selected task's RGB normalization. Normals must use the orientation and frame
expected by that checkpoint; they are not silently estimated or replaced with
zeros. This matters when moving from ScanNet to other scenes.

PLY vertices require `x/y/z`, `red/green/blue`, and, for RGBN6, `nx/ny/nz`.
An optional `segment_id` property supplies scan segments. Mesh faces are ignored.
The default upload limits are one million points and 128 MiB per file. With the
local server, uploads stay in that process. With a private hosted workbench,
uploads and downloadable predictions pass through its private object storage.

Mask3D uses supplied scan segments when present. Without them it pools over
5 cm voxels (or the configured grid) and records
`voxel_fallback_not_benchmark_segments`. Such output is useful for exploration,
but does not reproduce benchmark segment pooling. AGILE3D uses its official
per-axis-minimum coordinate shift and first voxel representatives, then expands
labels back to every original point.

No dataset or licensed example scene is distributed in Git. Prepare a scene
from data you can use, then expose it with `--example` or upload it in the UI.

## Browser UI

```bash
python scripts/demo.py --registry /path/to/models.yaml \
  --example /path/to/scene.npz --output /path/to/inference-results --port 7860
```

Open `http://localhost:7860`. Choose a model and load an example or upload a
scene. Drag to rotate, pan and zoom, switch RGB/instance colors, and run automatic
segmentation. For AGILE3D, select an object and place foreground clicks; use
background clicks to correct unwanted regions. Add another object for a
multi-object episode. Undo/reset edits the click history; Run/Update explicitly
requests a prediction. Model switching clears incompatible predictions.

The viewer samples at most 50,000 points for responsiveness. Picking uses the
displayed point's original index; inference and exported labels retain the
**entire input scene**. At most two scenes and one GPU model remain cached.
Interactive clicks reuse the scene's encoded features. Threshold changes reuse
Mask3D's cached raw proposals. Requests are serialized to avoid competing GPU
allocations. Download links cover the most recent 32 results; files remain in
the selected output directory.

The server binds to loopback. For a remote GPU, start it inside your own GPU
allocation and forward the port from your workstation:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-host
# If the GPU node is behind a login host:
ssh -N -J user@login-host -L 7860:127.0.0.1:7860 user@gpu-node
```

Keep the allocation and tunnel running while using the workbench. The static
frontend has no external CDN dependencies; inference requires the Python CUDA
service. Cloudflare/static hosting alone cannot execute these trained models.

### Private hosted workbench

The optional outbound GPU worker connects a hosted interface to an allocated
GPU without opening an inbound port on the cluster. The browser submits a job,
the worker runs the same inference service used locally, and the browser polls
for the full result. Checkpoints stay on the GPU host. Keep the hosted site
private and configure the same `DELIMIT3D_BRIDGE_TOKEN` as a server secret and a
worker environment variable; never include credentials in browser code.
The [hosted workbench template](../apps/inference-site/README.md) includes the
queue, authentication boundary, embedded UI, storage limits and deployment checks.

```bash
python -m delimit3d.inference.remote_worker \
  --url https://your-private-workbench.example \
  --registry /path/to/models.yaml --example /path/to/scene.npz \
  --output /path/to/inference-results
```

Sites service access additionally uses `DELIMIT3D_BRIDGE_HEADERS`, a JSON object
containing its `OAI-Sites-Authorization` header. Supply secrets through a
protected environment file or secret manager. The worker sends credentials only
to its configured HTTPS origin and refuses redirects. It sends heartbeats while
inference runs and retries result delivery without repeating completed inference.

The hosted UI reports whether the GPU is connected and, for Slurm jobs, when
the allocation expires. An offline worker cannot make predictions. After a
worker restart, load the scene again; its in-memory scene cache is not durable.
The underlying model and task selection, original point order, and full-scene
downloads are the same as in the local UI.

## Command-line predictions

```bash
python scripts/infer.py predict --registry /path/to/models.yaml \
  --model unsamv2-mask3d --scene /path/to/scene.npz \
  --score-threshold 0.3 --output /path/to/instances.npz

python scripts/infer.py predict --registry /path/to/models.yaml \
  --model unsamv2-agile3d --scene /path/to/scene.npz \
  --clicks /path/to/clicks.json --output /path/to/interactive.npz
```

Click JSON is an ordered array. Point indices are zero-based original input rows;
`object_id=0` means background, positive IDs select objects. At least one
foreground click is required. The latest click resolves conflicting constraints
in the same voxel. Example:

```json
[
  {"point_index": 120, "object_id": 1},
  {"point_index": 850, "object_id": 0},
  {"point_index": 2900, "object_id": 2}
]
```

## Outputs and interpretation

Every NPZ contains full-resolution `labels` (zero is background/unassigned),
`instance_ids`, `scores`, and UTF-8 JSON stored in `provenance_json`. Provenance
records model/checkpoint identity, normalization policy, input hash, grid,
point ordering and clicks. Commands refuse to overwrite an existing output.

Mask3D additionally exports `packed_masks` and `mask_point_count`. Masks are
packed along the point dimension and retain overlaps. Recover them with:

```python
import json
import numpy as np
with np.load("instances.npz", allow_pickle=False) as result:
    masks = np.unpackbits(result["packed_masks"], axis=0,
                          count=int(result["mask_point_count"])).astype(bool)
    # masks: original point count × retained proposals, aligned with scores.
    provenance = json.loads(str(result["provenance_json"]))
```

For display, the highest-confidence retained mask wins overlapping points.
The exclusive visualization is not the raw-query AP evaluator. Automatic scores
use objectness times mean foreground probability; interactive scores are mean
assigned-object probabilities. They are not calibrated semantic confidence.

This workbench is exploratory inference, not a benchmark evaluator. Use the
existing evaluation commands for CA-AP or official click metrics. Attaching
semantic names to masks requires a separately configured labeler; see the
[weekly update](weekly_update_20261017.md#semantic-labels-after-instance-segmentation).

## Verification

The [inference verification record](../manifests/inference_verification.json)
records successful GPU inference with all four pretrained/scratch Mask3D and
AGILE3D checkpoints on a 232,453-point scene, four model switches, state and
cache checks, click corrections and full-resolution mask exports. All four also
passed real predictions through the private hosted browser, including scene
upload and download validation. The CPU suite passed 280 tests, with 19 opt-in
or CUDA checks skipped; the separate GPU workflow completed successfully.

The verified GPU was an RTX 2080 Ti using Torch SDPA and the pinned Torch CUDA
PointROPE implementation. This validates that inference path; it does not
validate the incompatible installed PointROPE binary on that GPU or establish
benchmark accuracy. Rerun the portable check inside your GPU allocation:

```bash
python tests/native_inference_workflow.py --registry /path/to/models.yaml \
  --scene /path/to/scene.npz --output /path/to/fresh-inference-check
```

This uses the registered trained models and real input scene, checks point
mapping, prompt corrections, state preservation, mask export and model switching,
and writes a separate verification record. It is not an accuracy benchmark.
