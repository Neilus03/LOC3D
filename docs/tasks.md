# Mask3D and AGILE3D

Use the same shared decoder template and ordered task data across backbone comparisons. Model recipes inherit the task defaults, then select LitePT-S*, Concerto-small or Sonata-small.

## Native adapters

| Interface | LitePT-S* | Concerto / Sonata PTv3 |
| --- | --- | --- |
| Native finest descriptor | dec0, 72D | Concatenated native enc0..enc4, 992D |
| Mask3D attention memory | 504/252/144/72 | Native 512/256/128/64 |
| Decoder inputs | Task projections from LitePT widths | Direct projections from native PTv3 widths |
| Encoder inputs | XYZ separately; RGBN6 | Nine-channel stem [XYZ, RGB, normals] |
| Quantization | First representatives and exact inverse map | The same downstream representative/grid contract |

All decoders use hidden dimension 128, eight attention heads, FFN 1024 and three decoder cycles. Common decoder tensors are loaded before native input projections are initialized with a private CPU RNG. A 72D bottleneck is not inserted into the default PTv3 interface.

The earlier 896D three-finest aggregation, all-five aggregation at every level, compatibility projections, and LitePT decoder-only FPN are separate readouts retained in the implementation. They must be selected explicitly and recorded; identical widths do not imply identical features.

## Mask3D

```bash
python scripts/train.py --config configs/tasks/mask3d_litept.yaml --output /path/to/runs/mask3d_litept
python scripts/train.py --config configs/tasks/mask3d_concerto.yaml --output /path/to/runs/mask3d_concerto
python scripts/train.py --config configs/tasks/mask3d_sonata.yaml --output /path/to/runs/mask3d_sonata
```

The reference uses 5 cm RGBN6, 100 queries, two classes including no-object, five scenes per update and a 600-epoch OneCycleLR horizon. AdamW uses encoder LR 1e-5, decoder LR 1e-4 and weight decay .01. Training retains ScanNet geometry/color augmentation with correctly transformed normals, segment BCE/Dice, Hungarian matching, classification and auxiliary decoder losses.

Targets use the ScanNet20 instance universe. Predictions and segment/voxel maps are expanded back to original point order. The reported **CA-AP is custom class-agnostic raw-query AP**, not canonical class-aware ScanNet leaderboard AP. DBSCAN is an explicit secondary diagnostic, disabled by default.

Frozen and scratch controls use overrides:

```bash
# Frozen encoder
python scripts/train.py --config configs/tasks/mask3d_litept.yaml --output /path/to/frozen \
  --set initialization.freeze_backbone=true
# Scratch encoder with the same decoder and task settings
python scripts/train.py --config configs/tasks/mask3d_litept.yaml --output /path/to/scratch \
  --set initialization.backbone_source=scratch --set initialization.backbone_checkpoint=null
# Equal encoder/decoder learning rates are a separately recorded comparison
python scripts/train.py --config configs/tasks/mask3d_litept.yaml --output /path/to/equal_lr \
  --set train.backbone_lr=0.0001
```

Evaluation loads the full task checkpoint and creates no training loader or optimizer:

```bash
python scripts/evaluate.py --config configs/tasks/mask3d_litept.yaml \
  --checkpoint /path/to/checkpoints/latest.pt --output /path/to/mask3d_evaluation
```

Use a single worker for evaluation. Training supports the retained distributed engine. Changing the batch/world-size layout changes the update contract and must be recorded.

### Low-data and low-annotation experiments

Use [the limited-supervision workflow](limited_supervision.md) to prepare, verify,
train, resume and evaluate a matched scratch/pretrained LitePT-S* pair. Low-data
uses dense annotations on fixed custom 1/5/10/20% scene subsets. Low-annotation
uses the official 20/50/100/200 vertex allocations across all 1,201 training
scenes, with a separate sparse loss and geometry-derived voxel masks. Both use
all 312 densely labelled validation scenes for evaluation.

These presets use **equal encoder and decoder learning rates of 1e-4**, 600
natural epochs and five scenes per update. Their equal-LR recipe differs from
the generic Mask3D recipe above. A point budget is not implemented by truncating
dense targets: unannotated vertices remain unknown, and unmatched queries do not
receive no-object supervision. The guide records the resulting protocol and
checkpoint boundaries explicitly.

## AGILE3D

```bash
python scripts/train.py --config configs/tasks/agile3d_litept.yaml --output /path/to/agile3d_litept
python scripts/train.py --config configs/tasks/agile3d_concerto.yaml --output /path/to/agile3d_concerto
python scripts/train.py --config configs/tasks/agile3d_sonata.yaml --output /path/to/agile3d_sonata
```

The reference trains ScanNet40 multi-object episodes for 1100 epochs, batch five, validation every 50 epochs and LR drop at epoch 1000. AdamW uses decoder LR 1e-4, encoder multiplier .1, weight decay 1e-4 and clip norm .1. The official per-axis-minimum coordinates, NumPy quantization order, object mappings, click overwrite/correction policy, iterative training and BCE/Dice weighting are preserved.

Set `model.freeze_encoder=true` for frozen comparisons. For a scratch LitePT-S* control set `encoder_initialization.kind=scratch` and `encoder_initialization.checkpoint=null`. Equal LR uses `optimizer.encoder_learning_rate_multiplier=1.0`. These settings share the task recipe and decoder template.

```bash
python scripts/evaluate.py --config configs/tasks/agile3d_litept.yaml \
  --checkpoint /path/to/checkpoints/checkpoint.pth --output /path/to/agile3d_evaluation \
  --protocol all
```

`--protocol all` evaluates MO and SO. Use `mo` or `so` only when reporting that coverage explicitly. SO retains all 21 rows from zero through twenty clicks for every object; MO retains complete scenes and the released simulated-correction schedule. SO episode checkpoints are tied to checkpoint, data, code and normalization identity.

The default AGILE3D command is a single CUDA process; the standard release does not silently introduce distributed episode training.

## BatchNorm inference policy

The selected policy is explicit in every recipe and evaluation record. Corrected LitePT-S* task evaluation uses `train_no_update`: current-scene statistics, stochastic layers off, zero optimizer steps, and unchanged parameters and buffers even on exceptions. Native Concerto/Sonata AGILE3D recipes use saved statistics (`eval`). Mask3D recipes use their explicitly selected policy.

Results from different normalization policies are separate evaluations. Buffer recalibration creates a separate checkpoint, as described in [pretraining.md](pretraining.md).

## Checkpoints and resume

```bash
python scripts/train.py --config configs/tasks/agile3d_litept.yaml \
  --resume /path/to/checkpoint.pth --output /path/to/fresh_recovery
python scripts/train.py --config configs/tasks/mask3d_litept.yaml \
  --resume /path/to/latest.pt --output /path/to/fresh_recovery
```

New checkpoints record the resolved recipe, exact data hashes and model/readout contract. Resume keeps optimizer and scheduler state and rejects changed data, a changed horizon or an advanced output frontier. AGILE3D also stores RNG and validation completion; interrupted validation is replayed before the next epoch. Mask3D retains deterministic epoch reseeding and its OneCycleLR frontier.

Evaluation may use new data with the same model interface. Resuming training requires the original data and recipe; input path relocation is not automatically treated as an identical recipe. Full task checkpoints are strictly loaded. Historical frozen AGILE3D decoder-only checkpoints additionally require the exact original encoder hash.

Historical training checkpoints without the new data/recipe records are supported for strict evaluation where their tensors match; direct resume is refused. Continue those experiments with their preserved original checkout, or perform an explicitly reviewed conversion. See [reproduction.md](reproduction.md).
