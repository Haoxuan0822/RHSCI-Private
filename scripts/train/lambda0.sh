#!/usr/bin/env bash
set -euo pipefail
lambda_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lambda_code=$(cd -- "$lambda_dir/../.." && pwd)
lambda_runtime=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
lambda_python=${LAMBDA0_PYTHON:-$lambda_runtime/.venv-lambda0-rsci/bin/python}
export PYTHONPATH="$lambda_code/src:$lambda_code/src/openlambda/lambda0${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
for lambda_arg in "$@"; do
  case "$lambda_arg" in
    --help|-h|--check|--check-data|--check-model|--dry-run)
      exec "$lambda_python" -m rsci.backbones.lambda0.train "$@" ;;
    --worker|--guardian) echo "内部参数不能直接使用" >&2; exit 2 ;;
  esac
done
exec "$lambda_python" -m rsci.backbones.lambda0.launch "$@"
