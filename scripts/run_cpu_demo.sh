#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m steer.cli.main demo --config configs/runs/toy.json --out "${1:-runs/toy}" "${@:2}"
