#!/usr/bin/env bash
# Run the released checkpoint on the Metropolis validation split and save images.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="src/osm_diffusion/configs/seva_training_raydir_sep_block_ca_metropolis_initcheck_hires_mlp_gated.yaml"
CKPT="${CKPT:-/mnt/nas/personal/akshay/seva_training/ft_updated_svc_sat_ca_hires_gated_new_mlp/checkpoints/eval1.ckpt}"
RES="${RES:-576,576}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/metropolis}"
mkdir -p "$OUTPUT_DIR"

python -m osm_diffusion.training.eval \
  --config "$CONFIG" \
  --datasets metropolis \
  --checkpoint-path "$CKPT" \
  --output-path "$OUTPUT_DIR" \
  --validation_debug_dir "metropolis_${RES%%,*}_dual_cfg3_img2_baseline" \
  --pose_scale_factors 2.0 \
  --num_input_frames 7 \
  --num_total_frames 21 \
  --resolution "$RES" \
  --use_dual_cfg \
  --sat_cfg_value 3.0 \
  --img_cfg_value 2.0 \
  --datasets-config-path eval \
  "$@"
