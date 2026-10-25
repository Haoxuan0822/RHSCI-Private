#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
code_root=$(cd -- "$script_dir/../.." && pwd)
runtime_root=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
egovla_python=${EGOVLA_PYTHON:-$runtime_root/.venv-egovla-rsci/bin/python}
if [[ ! -x "$egovla_python" ]]; then
  if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
    egovla_python=python3
  else
    echo "EgoVLA独立环境未安装：$egovla_python。此脚本不会修改Psi0/π0.5环境或自动下载。" >&2
    exit 1
  fi
fi
export PYTHONPATH="$code_root/src/egovla:$code_root/src/egovla/VILA:$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# 查询入口不生成输出或后台进程；训练才使用独立守护及日志。
for arg in "$@"; do
  case "$arg" in
    --help|-h|--check|--check-data|--dry-run|--register-assets)
      exec "$egovla_python" -m rsci.backbones.egovla.train "$@" ;;
    --worker|--guardian)
      echo "--worker/--guardian仅供内部使用" >&2; exit 2 ;;
  esac
done

exec "$egovla_python" -m rsci.backbones.egovla.launch "$@"
