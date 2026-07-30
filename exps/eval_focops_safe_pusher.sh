#!/usr/bin/env bash
set -euo pipefail
# Evaluate FOCOPS on SafePusher
# Usage (from repo root inside container):
#   bash exps/eval_focops_safe_pusher.sh [num_seeds] [log_dir] [checkpoint]
# Defaults:
#   num_seeds = 5          # env seeds 0..num_seeds-1
#   log_dir   = latest FOCOPS SafePusher run under examples/logs
#   checkpoint= latest
#
# Prints per-trajectory: success, final cost, final reward

NUM_SEEDS="${1:-5}"
LOG_DIR="${2:-}"
CKPT="${3:-latest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/examples"
export PYTHONPATH="$ROOT/examples:${PYTHONPATH:-}"

ARGS=(--env pusher --checkpoint "$CKPT" --num-seeds "$NUM_SEEDS" --algo FOCOPS)
if [[ -n "$LOG_DIR" ]]; then
  ARGS+=(--log-dir "$LOG_DIR")
fi

python eval_long_horizon.py "${ARGS[@]}"
