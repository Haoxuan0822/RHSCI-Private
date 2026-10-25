#!/usr/bin/env bash
set -euo pipefail
# GR00T RHSCI：数据/缓存/断点隔离，正式入口不改PSI0环境、不自动下载。
rsci_code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PSI_RUNTIME_ROOT="${PSI_RUNTIME_ROOT:-/opt/rsci-data}"
export PYTHONPATH="$rsci_code/src/gr00t:$rsci_code/src:$rsci_code/packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/gr00t/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_HOME="${TORCH_HOME:-$PSI_RUNTIME_ROOT/cache/rsci/gr00t/torch}"
export WANDB_MODE=offline
rsci_python="${GROOT_RSCI_PYTHON:-$PSI_RUNTIME_ROOT/.venv-gr00t-rsci/bin/python}"
if [[ ! -x "$rsci_python" ]]; then
  # 帮助/预检允许显式沿用已安装的CPU工具，不把它当作可正式训练的GR00T环境。
  for rsci_argument in "$@"; do
    case "$rsci_argument" in --help|-h|--check|--check-data|--dry-run) rsci_python="$PSI_RUNTIME_ROOT/.venv-psi/bin/python";; esac
  done
fi
if [[ ! -x "$rsci_python" ]]; then
  echo 'Set GROOT_RSCI_PYTHON to the interpreter in your GR00T environment; see baselines/gr00t/README.md' >&2
  exit 1
fi
# --gpus同时由worker校验；只修改本次子进程可见卡，不占用/停止其他任务。
rsci_gpu_list=${CUDA_VISIBLE_DEVICES:-0}
rsci_expect_gpu=0
for rsci_argument in "$@"; do
  if (( rsci_expect_gpu )); then rsci_gpu_list="$rsci_argument"; rsci_expect_gpu=0; fi
  case "$rsci_argument" in --gpus) rsci_expect_gpu=1;; --gpus=*) rsci_gpu_list="${rsci_argument#--gpus=}";; esac
done
export CUDA_VISIBLE_DEVICES="$rsci_gpu_list"
cd -- "$rsci_code"
exec "$rsci_python" -u -m rsci.backbones.gr00t.train "$@"
