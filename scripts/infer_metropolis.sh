#!/usr/bin/env bash
# Run Metropolis inference. Requires configs/paths.yaml (see paths.example.yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-src/groundcontrol/configs/infer_metropolis.yaml}"
python -m groundcontrol.training.eval --config "$CONFIG" "$@"
