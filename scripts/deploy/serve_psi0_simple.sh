#!/usr/bin/env bash
set -euo pipefail
rsci_code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
rsci_runtime=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
rsci_python=${PSI0_PYTHON:-$rsci_runtime/.venv-psi/bin/python}
export PYTHONPATH="$rsci_code/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$rsci_code"
exec "$rsci_python" -m psi.deploy.psi0_serve_simple "$@"
