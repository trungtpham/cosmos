#!/usr/bin/env bash
# Source this script to replicate the exact environment the notebook sets up.
# Usage:  source cookbooks/cosmos3/generator/transfer/setup_env.sh
#
# After sourcing, run inference the same way the notebook does:
#
#   Nano (single GPU):
#     cd "$COSMOS3_REPO"
#     CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
#     .venv/bin/python -m cosmos_framework.scripts.inference \
#       --parallelism-preset=latency \
#       -i "$COSMOS3_TRANSFER_ROOT/specs/edge.json" \
#       -o "$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Nano/" \
#       --checkpoint-path Cosmos3-Nano --seed 2026
#
#   Super (multi-GPU):
#     cd "$COSMOS3_REPO"
#     CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
#     .venv/bin/torchrun \
#       --nproc-per-node="$COSMOS3_NUM_GPUS" \
#       --master-addr="$COSMOS3_MASTER_ADDR" \
#       --master-port="$COSMOS3_MASTER_PORT" \
#       -m cosmos_framework.scripts.inference \
#       --parallelism-preset=throughput \
#       -i "$COSMOS3_TRANSFER_ROOT/specs/edge.json" \
#       -o "$COSMOS3_TRANSFER_OUTPUT_ROOT/Cosmos3-Super/" \
#       --checkpoint-path Cosmos3-Super --seed 2026

set -euo pipefail

# Run the notebook's Python setup cell and capture exports as shell variables.
_SETUP_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/setup_env.py"

if [[ ! -f "$_SETUP_SCRIPT" ]]; then
  echo "ERROR: setup_env.py not found at $_SETUP_SCRIPT" >&2
  return 1
fi

eval "$(python3 "$_SETUP_SCRIPT")"
echo "Transfer env loaded. Framework: $COSMOS3_REPO  GPUs: $COSMOS3_NUM_GPUS  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
