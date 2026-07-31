#!/usr/bin/env bash
set -euo pipefail
# Train TD3Lag on SafeAntMazeC-Rand (flat OmniSafe baseline)
# Comet: workspace=gregory-gorbov project=ites
# Experiment name: "TD3Lag SafeAntMazeC"
# Usage (from repo root inside container):
#   bash exps/train_td3lag_safe_ant_maze_c.sh [seed] [device]
# Defaults: seed=224424, device=cuda:0, cost_limit=25, comet=on, wandb=off

SEED="${1:-224424}"
DEVICE="${2:-cuda:0}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/examples"
export PYTHONPATH="$ROOT/examples:${PYTHONPATH:-}"

python train_long_horizon.py \
  --algo TD3Lag \
  --env c \
  --seed "$SEED" \
  --device "$DEVICE" \
  --cost-limit 25 \
  "$@"
