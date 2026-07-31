#!/usr/bin/env bash
set -euo pipefail
# Evaluate TD3Lag on SafeAntMazeW (5 env seeds by default)
# Usage (from repo root inside container):
#   bash exps/eval_td3lag_safe_ant_maze_w.sh [num_seeds] [log_dir] [checkpoint]
# Defaults:
#   num_seeds = 5
#   log_dir   = latest TD3Lag SafeAntMazeW-Rand run under examples/logs
#   checkpoint= latest
#
# Prints per-trajectory: success, final cost, final reward

NUM_SEEDS="${1:-5}"
LOG_DIR="${2:-}"
CKPT="${3:-latest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/examples"
export PYTHONPATH="$ROOT/examples:${PYTHONPATH:-}"

ARGS=(--env w --checkpoint "$CKPT" --num-seeds "$NUM_SEEDS" --algo TD3Lag --device cpu)
if [[ -n "$LOG_DIR" ]]; then
  ARGS+=(--log-dir "$LOG_DIR")
fi

python eval_long_horizon.py "${ARGS[@]}"
