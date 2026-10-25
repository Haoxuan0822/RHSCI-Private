#!/usr/bin/env bash
set -euo pipefail
rsci_code=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec python3 "$rsci_code/scripts/train/baseline.py" --backbone h-rdt "$@"
