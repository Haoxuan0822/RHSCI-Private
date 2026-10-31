#!/usr/bin/env bash
set -euo pipefail
rhsci_code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec python3 "$rhsci_code/scripts/train/baseline.py" --backbone h-rdt "$@"
