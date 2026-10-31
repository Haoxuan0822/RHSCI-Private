#!/usr/bin/env bash
set -euo pipefail
rhsci_code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
rhsci_runtime=${PSI_RUNTIME_ROOT:-/opt/rsci-data}
rhsci_python=${PSI0_PYTHON:-$rhsci_runtime/.venv-psi/bin/python}
export PYTHONPATH="$rhsci_code/src${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$rhsci_code"
exec "$rhsci_python" -m psi.deploy.psi0_serve_simple "$@"
