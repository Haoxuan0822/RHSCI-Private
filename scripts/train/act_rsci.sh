#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_root=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
act_python=${ACT_PYTHON:-$runtime_root/.venv-psi/bin/python}
if [[ ! -x "$act_python" ]]; then
  if [[ " ${*} " == *" --help "* || " ${*} " == *" -h "* ]]; then
    act_python=python3
  else
    echo "ACT environment not found: $act_python; set ACT_PYTHON." >&2
    exit 1
  fi
fi
export PYTHONPATH="$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
for argument in "$@"; do
  case "$argument" in
    --help|-h|--check|--check-data|--dry-run)
      exec "$act_python" -m rsci.backbones.act.train "$@" ;;
    --worker|--guardian)
      echo "--worker/--guardian are internal options" >&2; exit 2 ;;
  esac
done
exec "$act_python" -m rsci.backbones.act.launch "$@"
