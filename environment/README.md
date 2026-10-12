# Environment setup

The student reference profile records the observed Python **3.12.8**, Torch **2.4.1 + CUDA 12.4** environment. Use a fresh environment. Python 3.11 is accepted by the package, but compiled extensions must be built for that interpreter; the older PF69 environment is a separate historical runtime.

## Student training and transfer

From the repository root, with Python 3.12 and a compatible NVIDIA driver:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/cuda124.txt
python -m pip install --no-deps -e .
```

The CUDA profile pins Torch, spconv, torch-scatter, torch-cluster and the shared Python dependencies. Install FlashAttention and the native LitePT kernels after Torch, with a compatible CUDA toolkit available:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/prepare_data.py --action dependencies --source litept --destination .deps/LitePT
python -m pip install --no-build-isolation .deps/LitePT/libs/pointrope
python -m pip install --no-build-isolation .deps/LitePT/libs/pointops
```

`CUDA_HOME` must identify the toolkit used to compile extensions. The locked PointROPE build targets Ampere architectures; rebuilding for other architectures is a separately recorded runtime change. Its Torch fallback is useful for diagnosis and is not CUDA-kernel verification.

An A100 needs the `sm_80` PointROPE kernel; an RTX 3090 needs `sm_86`. The pinned setup builds both. An older installed binary may contain only `sm_86`: it can import successfully and then fail on an A100 with “no kernel image”. Check the loaded extension's path and inspect its kernels with `cuobjdump --list-elf /path/to/pointrope.so`. Rebuild in the new environment when the required architecture is missing.

Fetch optional baseline sources and exact weights:

```bash
python scripts/prepare_data.py --action dependencies --source concerto --destination .deps/Concerto
python scripts/prepare_data.py --action dependencies --source sonata --destination .deps/Sonata
python scripts/prepare_data.py --action weights --weight concerto_small --output .deps/weights/concerto_small.pth
python scripts/prepare_data.py --action weights --weight sonata_small --output .deps/weights/sonata_small.pth
```

Each fetch uses a commit or content hash in `src/delimit3d/runtime/assets/`. LitePT receives the recorded runtime patch and all twelve expected source hashes are checked. Existing source directories are preserved; changed checkpoint files are rejected.

For raw geometry preparation:

```bash
python -m pip install -r environment/data.txt
```

## Separate teacher environments

The locked SAM2 and UnSAMv2 sources require Torch >=2.5.1 and both expose a package named `sam2`. Install each teacher in its own environment with `environment/teacher.txt`. Neither environment replaces the student runtime.

Fetch the source and weights using the student preparation command, then in the relevant teacher environment install the source and Delimit3D:

```bash
# SAM2 teacher environment
python -m pip install -r environment/teacher.txt
python -m pip install --no-build-isolation -e /absolute/path/to/.deps/SAM2
python -m pip install --no-deps -e /absolute/path/to/delimit3d

# UnSAMv2: the installable SAM2 project is nested inside the source checkout.
python -m pip install --no-build-isolation -e /absolute/path/to/.deps/UnSAMv2/sam2
```

Use `--source sam2` / `--weight sam2_1_hiera_large`, or `--source unsamv2` / `--weight unsamv2_plus`. Teacher preparation only imports the selected teacher. Student pretraining consumes saved masks and does not load either teacher.

`teacher.txt` is a compatible inference profile for the pinned source requirements. A fresh full teacher export is separate from the student release checks.

## Development checks

```bash
python -m pip install -r environment/dev.txt
python -m pytest -q
DELIMIT3D_NATIVE_CHECKS=1 python -m pytest -q tests/test_native_runtime.py
```

Native checks require CUDA, all three pinned source checkouts and both baseline weights under `.deps/`. They exercise forward, backward, optimizer updates and freezing for both tasks, plus all three pretraining objectives. The public task checks train, evaluate and resume a frozen LitePT-S* encoder with a trainable decoder. Trainable encoders are covered by the native forward/backward checks.

For the separate two-GPU public pretraining/resume check:

```bash
DELIMIT3D_NATIVE_CHECKS=1 python tests/native_pretraining_workflow.py
```

It needs exactly two visible CUDA GPUs and writes an isolated three-update hierarchy run under `runs/native_pretraining_check`. Set `DELIMIT3D_NATIVE_OUTPUT_DIR` to a fresh path for another attempt. These checks use synthetic inputs and do not replace full-data training, teacher export, calibration or benchmark evaluation.

This short workflow uses a mask-loss coefficient of .1 to keep its synthetic auxiliary gradients within the active .5 budget. The production hierarchy recipe retains coefficient 1. Native objective checks separately exercise the production coefficient.

For both SAM2 negative policies through the public training and fixed evaluation commands:

```bash
DELIMIT3D_NATIVE_CHECKS=1 python tests/native_sam2_policies_workflow.py
```

This requires the pinned LitePT source, native kernels and exactly two visible CUDA GPUs. It prepares nested synthetic SAM2 masks, uses a common scratch initialization, trains each policy for three updates, resumes each from update one, and runs an uninterrupted repetition control on the same GPUs. Negative-policy stamps, RNG, coverage, exposure, sampling totals and counters must agree exactly. Model/optimizer differences are reported alongside the repetition control; GPU tensor identity is not asserted. The report also records the model comparison at rtol 1e-5 / atol 1e-6. The CPU suite checks model and optimizer recovery with these strict tolerances. This workflow verifies policy handling on synthetic data and rejects a checkpoint from the other policy. Output defaults to `runs/native_sam2_policies_check`; use a fresh `DELIMIT3D_NATIVE_OUTPUT_DIR` for another attempt.
