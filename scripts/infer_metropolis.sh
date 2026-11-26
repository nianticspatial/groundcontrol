#!/usr/bin/env bash
# Run Metropolis inference from the lean inference YAML.
# Optional env overrides: CKPT, OUTPUT_DIR. Extra args pass through to eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-src/osm_diffusion/configs/infer_metropolis.yaml}"

ARGS=(--config "$CONFIG")
if [[ -n "${CKPT:-}" ]]; then
  ARGS+=(--checkpoint-path "$CKPT")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-path "$OUTPUT_DIR")
fi

python -m osm_diffusion.training.eval "${ARGS[@]}" "$@"
