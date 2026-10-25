#!/usr/bin/env bash
set -euo pipefail
lambda_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lambda_code=$(cd -- "$lambda_dir/../.." && pwd)
lambda_runtime=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
lambda_python=${LAMBDA0_PYTHON:-$lambda_runtime/.venv-lambda0-rsci/bin/python}
export PYTHONPATH="$lambda_dir:$lambda_code/src:$lambda_code/src/openlambda/lambda0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$lambda_python" -m rsci.backbones.lambda0.serve "$@"
