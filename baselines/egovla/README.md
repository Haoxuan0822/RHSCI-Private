# EgoVLA

Backbone: `src/egovla`. Adapter: `src/rsci/backbones/egovla` (`rsci.backbones.egovla`).
Entry: `bash scripts/train/egovla.sh --help`.

The adapter uses the SIMPLE 36D four-layer action decoder. VGQ-AE adds training-only Reader/query supervision; STATE uses the native state path. VLM training defaults to full finetuning. VGQ-AE inference does not use training history or auxiliary queries.

Use the separate EgoVLA/VILA environment and a registered SIMPLE VLM asset manifest. `EGOVLA_PYTHON` overrides `$PSI_RUNTIME_ROOT/.venv-egovla-rsci/bin/python`. The included native `llava/data` pipeline is incomplete; the RHSCI adapter supplies its own task reader. Native training needs the missing upstream data dependency.

## Install the RHSCI environment

Run from the repository root on Linux x86_64 with Python 3.11 and an NVIDIA driver compatible with CUDA 12.8. TorchCodec also needs a system FFmpeg installation with shared libraries. The RHSCI environment uses Torch 2.7.0, Transformers 4.53.2 and Accelerate 1.7.0. The vendored VILA `pyproject.toml` retains the upstream historical pins; do not run `pip install -e src/egovla/VILA` or install this repository's Python 3.10 root environment into this Python 3.11 environment.

```bash
python3.11 -m venv /opt/rsci-data/.venv-egovla-rsci
source /opt/rsci-data/.venv-egovla-rsci/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r baselines/egovla/requirements-rsci.txt
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl#sha256=cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af'
export PYTHONPATH="$PWD/src/egovla:$PWD/src/egovla/VILA:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export EGOVLA_PYTHON="$VIRTUAL_ENV/bin/python"
python -m pip check
python -c 'import torch, transformers, accelerate, torchcodec; from llava.model import LlavaLlamaModel; print(torch.__version__, transformers.__version__, accelerate.__version__)'
python tools/test_cpu.py --backbone egovla
```

The FlashAttention wheel above matches CPython 3.11, Linux x86_64, Torch 2.7 and its C++11 ABI; use a separately verified build for another platform. The RHSCI recipe uses SDPA, but the retained native model modules still import FlashAttention. CPU tests and import checks do not validate a driver's CUDA kernels, multi-GPU communication or simulator evaluation. The MANO warning concerns an optional human-hand path; the SIMPLE 36D vector adapter does not need MANO assets.

Prepare the fixed SIMPLE base VLM separately:

```bash
python -m rsci.backbones.egovla.prepare_assets --runtime-root /opt/rsci-data
```

This downloads and verifies the base VLM and registers its `RSCI_ASSETS.json`; it does not start training. For already downloaded base assets, use the training entry's `--register-assets` with `--checkpoint` and your task/data arguments. Run `--check` with those same arguments before starting training.

`--dry-run` and `--check-data` are available before training. Pass `--task`, `--dataset`, `--sidecar`, `--checkpoint`, `--output-dir`, batch/GPU settings, and either `--max-steps` or `--num-train-epochs`; see [the training guide](../../docs/TRAINING.md). It uses the environment's NCCL by default; set `RSCI_NCCL_LIBRARY` only for a validated alternate runtime.

## SIMPLE 115-epoch preset

The generic default is 40000 updates, which differs from the 115-epoch experiment. Use this explicit preset for either VGQ-AE or STATE:

```bash
bash scripts/train/egovla.sh \
  --preset simple-115epoch --model VGQ-AE --vlm-training full \
  --task my_task --dataset /datasets/my_task --sidecar /labels/my_task \
  --checkpoint /models/egovla --output-dir /runs/egovla/my_task \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 16 --gradient-accumulation-steps 4 \
  --num-workers 8 --seed 292285 --dry-run
```

This selects 115 epochs and enforces global batch 512. The budget is derived from complete 30-step action windows in the dataset metadata and checked against the actual distributed loader. The local container-pick dataset has 45046 valid windows: 352 microbatches and 88 updates per epoch, hence 10120 updates. Another dataset may produce a different update count. Remove `--dry-run` to train; use an independent output directory for STATE and initialize it from the same base checkpoint, not the VGQ-AE output.

## Optional NCCL runtime

The 115-epoch preset also defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `NCCL_SHM_DISABLE=0` before importing Torch or starting workers. Explicit environment values take precedence. SDPA, gradient checkpointing and DDP gradient bucket views remain enabled by the existing recipe. Without the preset, no allocator setting is injected.

Setting `RSCI_NCCL_LIBRARY=/absolute/path/to/libnccl.so` preloads that library before torchrun and its workers start. The launcher checks the mapped library and its actual runtime version. EgoVLA expects `2.27.3` by default for this override; set `RSCI_NCCL_VERSION` explicitly for another verified version. With no override, it uses the environment's NCCL. This option does not change batch size, gradient reduction or model settings.

Serving uses `python -m rsci.backbones.egovla.serve` in the same environment, with a complete checkpoint and the original base assets. See [Inference](../../docs/INFERENCE.md#egovla) for the command and state/action protocol.
