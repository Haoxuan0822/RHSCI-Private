# ACT

Native implementation: `src/act`. RHSCI adapter: `src/rsci/backbones/act`. Supported variants are STATE, NO-STATE and EGQ-AE, with no language model or Reader. NO-STATE removes state projections. EGQ-AE uses native trainable ResNet features and four-frame history `[-6, -4, -2, 0]` for training-only ARDY supervision; inference needs only the current image. The current frame's features are shared between action and auxiliary paths.

Use an isolated Python 3.10 environment with the root PSI0 dependency group: `uv sync --group psi --group serve --group viz --active --frozen`. The locked LeRobot 0.3.3 uses its native `lerobot.datasets` layout; no legacy `src/lerobot_patch/common` overlay is needed. Follow the [PSI0 installation guide](../psi0/README.md#installation), including its layout check. Keep the environment outside the checkout and configure the appropriate dataset schema.

## RHSCI training

`ACT_PYTHON` selects the interpreter. Supply SIMPLE G1 data (32 measured state values, 36 action values, 50 Hz), normalization statistics and, for EGQ-AE, matching ARDY sidecars. Place the official ImageNet ResNet18 initialization at `/models/torch/hub/checkpoints/resnet18-f37072fd.pth`. Training verifies the local file and does not download it.

```bash
ACT_PYTHON=/path/to/act-env/bin/python bash scripts/train/act_rsci.sh \
  --task my_task --model EGQ-AE --gpus 0,1,2,3 \
  --dataset /datasets/my_task --sidecar /labels/my_task \
  --torch-home /models/torch --output-dir /runs/act/my_task \
  --batch-size 32 --gradient-accumulation-steps 1 --max-steps 40000 --check-data
```

Use `--check` instead for random-initialized full-size CPU model and processor checks without downloading weights. Remove the check flag to train. STATE and NO-STATE use the same entry and need no sidecars. The common launcher also accepts `--backbone act --mode rsci`; `scripts/train/act.sh` remains the native launcher.

The four-GPU example matches the local recipe's global batch 128 (32 per GPU, accumulation 1), seed 2026 and 40000 updates. GPU count is explicit: changing it changes the global batch unless accumulation or per-GPU batch is adjusted. The generic entry defaults to one visible GPU when none is specified.

The recipe retains native masked L1 + 10 × KL and adds 0.1 × ARDY for EGQ-AE. Validation reports the training objective and separate deterministic latent-zero action metrics. Inference uses latent zero and returns 16 × 36 action chunks. The optimizer is a single AdamW group, with a 1000-step warmup and no gradient clipping.

The guardian resumes complete checkpoints from the same run directory and validates source, data, initialization and configuration identities. Full checkpoints contain model buffers, optimizer, scheduler, RNG and progress. Do not point it at an older local run with different source contracts: cross-version training-state migration is not provided. Resume tests cover cumulative progress across epochs with zero and persistent workers; they do not establish bitwise CUDA trajectory equivalence.

Video decoding uses a bounded per-worker cache (16 decoders, 64 MiB decoded frames) and preserves duplicate/reordered timestamps and RNG state. Guardian termination waits for and reaps the training process. The default runtime uses the environment's NCCL; optional `--nccl-library` or `RSCI_NCCL_LIBRARY` is preloaded and checked, with expected version `2.27.3` unless `RSCI_NCCL_VERSION` is set.

The local multi-GPU experiment explicitly used NCCL 2.27.3; to reproduce that runtime, select your local copy with `--nccl-library /absolute/path/to/libnccl.so.2.27.3`. The public launcher does not assume the server's system-library path. Checkpoints bind the selected runtime's path, hash and actual loaded version.

## Inference and native training

Serve a complete RHSCI checkpoint with `python -m rsci.backbones.act.serve --checkpoint /runs/act/my_task/checkpoint-40000`. See [Inference](../../docs/INFERENCE.md). STATE requires measured 32D state; there is no fallback to previously predicted state.

For upstream ACT training, use `bash scripts/train/act.sh --help` and [the native training example](../../docs/TRAINING.md#act-and-diffusion-policy), with `configs/train/act.json`. Native training has not been validated in this standalone release.
