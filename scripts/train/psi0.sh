#!/usr/bin/env bash
set -euo pipefail
# RHSCI 统一入口：方案名决定结构，任务和 VLM 训练方式独立选择。
rsci_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PSI_RUNTIME_ROOT="${PSI_RUNTIME_ROOT:-/opt/rsci-data}"
export PYTHONPATH="$rsci_code/src:$rsci_code/packages:$rsci_code/scripts/train${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PSI_RUNTIME_ROOT/cache/rsci/psi0/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PSI_RUNTIME_ROOT/cache/rsci/psi0/torch"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PSI_ATTN_IMPLEMENTATION=sdpa
export TORCHINDUCTOR_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/rsci/psi0/torchinductor"
export TRITON_CACHE_DIR="$PSI_RUNTIME_ROOT/cache/rsci/psi0/triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" WANDB_MODE="${WANDB_MODE:-offline}"
cd -- "$rsci_code"
exec "${PSI0_PYTHON:-$PSI_RUNTIME_ROOT/.venv-psi/bin/python}" -m rsci.backbones.psi0.train "$@"
