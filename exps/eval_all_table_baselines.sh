#!/usr/bin/env bash
set -euo pipefail
# Run 5-seed eval for all flat table baselines on C / W / Pusher.
# Usage: bash exps/eval_all_table_baselines.sh [out_dir]
OUT_DIR="${1:-examples/eval_results_table}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/$OUT_DIR"
cd "$ROOT"

ALGOS=(focops cup td3lag td3pid ppolag)
ENVS=(safe_ant_maze_c safe_ant_maze_w safe_pusher)

for a in "${ALGOS[@]}"; do
  for e in "${ENVS[@]}"; do
    script="exps/eval_${a}_${e}.sh"
    csv="$OUT_DIR/${a}_${e}_seeds.csv"
    echo "==== $script -> $csv ===="
    # eval_long_horizon writes CSV if --out-csv passed; wrap via python for discovery failures
    if bash "$script" 5 "" latest 2>&1 | tee "$OUT_DIR/${a}_${e}.log"; then
      :
    else
      echo "WARN: $script failed (missing checkpoint?)" | tee -a "$OUT_DIR/${a}_${e}.log"
    fi
  done
done
echo "Done. Logs in $OUT_DIR"
