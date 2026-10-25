#!/usr/bin/env bash
set -euo pipefail
rsci_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PSI_RUNTIME_ROOT="${PSI_RUNTIME_ROOT:-/opt/rsci-data}"
export PYTHONPATH="$rsci_code/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_HOME="${TORCH_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/pi05/torch}"
export HF_HOME="${HF_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/pi05/huggingface}"
rsci_python="${PI05_RSCI_PYTHON:-$PSI_RUNTIME_ROOT/.venv-pi05-rsci/bin/python}"
if [[ ! -x "$rsci_python" ]]; then
  for rsci_argument in "$@"; do
    case "$rsci_argument" in -h|--help|--dry-run|--check-data) rsci_python="$PSI_RUNTIME_ROOT/.venv-psi/bin/python";; esac
  done
fi
if [[ ! -x "$rsci_python" ]]; then
  echo 'Set PI05_RSCI_PYTHON to the interpreter in your pi05 environment; see baselines/pi05/README.md' >&2
  exit 1
fi
rsci_gpu_list=${CUDA_VISIBLE_DEVICES:-0}
rsci_next=0
for rsci_argument in "$@"; do
  if ((rsci_next)); then rsci_gpu_list="$rsci_argument"; rsci_next=0; fi
  case "$rsci_argument" in --gpus) rsci_next=1;; --gpus=*) rsci_gpu_list="${rsci_argument#--gpus=}";; esac
done
export CUDA_VISIBLE_DEVICES="$rsci_gpu_list"
cd -- "$rsci_code"
exec "$rsci_python" -u -m rsci.backbones.pi05.train "$@"
