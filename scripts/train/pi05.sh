#!/usr/bin/env bash
set -euo pipefail
rhsci_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PSI_RUNTIME_ROOT="${PSI_RUNTIME_ROOT:-/opt/rsci-data}"
export PYTHONPATH="$rhsci_code/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/pi05/torch}"
export HF_HOME="${HF_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/pi05/huggingface}"
rhsci_python="${PI05_RHSCI_PYTHON:-${PI05_RSCI_PYTHON:-$PSI_RUNTIME_ROOT/.venv-pi05-rsci/bin/python}}"
if [[ ! -x "$rhsci_python" ]]; then
  for rhsci_argument in "$@"; do
    case "$rhsci_argument" in -h|--help|--dry-run|--check-data) rhsci_python="$PSI_RUNTIME_ROOT/.venv-psi/bin/python";; esac
  done
fi
if [[ ! -x "$rhsci_python" ]]; then
  echo 'Set PI05_RHSCI_PYTHON to the interpreter in your pi05 environment; see baselines/pi05/README.md' >&2
  exit 1
fi
rhsci_gpu_list=${CUDA_VISIBLE_DEVICES:-0}
rhsci_next=0
for rhsci_argument in "$@"; do
  if ((rhsci_next)); then rhsci_gpu_list="$rhsci_argument"; rhsci_next=0; fi
  case "$rhsci_argument" in --gpus) rhsci_next=1;; --gpus=*) rhsci_gpu_list="${rhsci_argument#--gpus=}";; esac
done
export CUDA_VISIBLE_DEVICES="$rhsci_gpu_list"
cd -- "$rhsci_code"
exec "$rhsci_python" -u -m openpi.rhsci.train "$@"
