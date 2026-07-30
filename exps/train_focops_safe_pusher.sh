#!/usr/bin/env bash
set -euo pipefail
# Train FOCOPS on SafePusher-Rand
# Comet: workspace=gregory-gorbov project=ites
# Experiment name: "FOCOPS {SafeAntMazeC|SafeAntMazeW|SafePusher}"
# Usage (from repo root inside container):
#   bash exps/train_focops_safe_pusher.sh [seed] [device]
# Defaults: seed=224424, device=cuda:0, comet=on, wandb=off

SEED="${1:-224424}"
DEVICE="${2:-cuda:0}"
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true

export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/examples"

export PYTHONPATH="$ROOT/examples:${PYTHONPATH:-}"

python train_long_horizon.py \
  --algo FOCOPS \
  --env pusher \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$@"
