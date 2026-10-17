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
The default upload limits are one million points and 128 MiB per file. Uploads
are local to the process and are not published.

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
