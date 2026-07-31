#!/usr/bin/env bash
# Run ALL SafeDreamer validations INSIDE an already-running container.
# cwd must be /usr/home/workspace (conda env safe_dynalang active).
#
#   bash exps/validation_SafeDreamer_all.sh
#
# Optional:
#   SEEDS="0 1 2 3 4" bash exps/validation_SafeDreamer_all.sh
set -euo pipefail

cd /usr/home/workspace

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
TASKS="${TASKS:-cshape wshape pusher}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_TXT="${RESULTS_TXT:-logdir_eval_safedreamer/results_all_${STAMP}.txt}"
export FIX_JAX="${FIX_JAX:-0}"
export SKIP_COMET="${SKIP_COMET:-1}"
export GPU="${GPU:-0}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset COMET_API_KEY || true

ckpt_for() {
  case "$1" in
    cshape) echo "logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt" ;;
    wshape) echo "logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_wshape_0/checkpoint.ckpt" ;;
    pusher) echo "logdir_osrp_1124_costlimit2/20260729-191539_osrp_lag_safeantmaze_pusher_0/checkpoint.ckpt" ;;
    *) echo "" ;;
  esac
}

mkdir -p "$(dirname "$RESULTS_TXT")"
{
  echo "SafeDreamer validation — all environments (in-container)"
  echo "started: $(date -Is)"
  echo "seeds: $SEEDS"
  echo "pwd: $(pwd)"
  echo
  for t in $TASKS; do
    echo "ckpt[$t]=$(ckpt_for "$t")"
  done
  echo
} | tee "$RESULTS_TXT"

# Ensure deps (train images sometimes miss these in a fresh shell / new container).
python - <<'PY'
import importlib, subprocess, sys
from pathlib import Path

need = []
for mod, pkg in [("mujoco_py", "free-mujoco-py"), ("gymnasium", "gymnasium")]:
    try:
        importlib.import_module(mod)
    except Exception:
        need.append(pkg)
if need:
    print("[preflight] installing:", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need, "patchelf", "gym==0.15.7"])

env_pkg = Path("/usr/home/workspace/SafeDreamer/embodied/envs/Safety_ant_maze_pusher_envs")
try:
    importlib.import_module("safety_ant_maze_pusher_envs")
except Exception:
    print("[preflight] pip install -e Safety_ant_maze_pusher_envs")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(env_pkg)])

import mujoco_py, importlib.metadata as m
import safety_ant_maze_pusher_envs  # noqa: F401
print("[preflight]", m.version("jax"), m.version("jaxlib"), "mujoco_py OK, env OK")
PY

for TASK in $TASKS; do
  CKPT="$(ckpt_for "$TASK")"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: missing checkpoint for $TASK: $CKPT" | tee -a "$RESULTS_TXT"
    exit 1
  fi
  echo
  echo "########## TASK=$TASK ##########"
  export TASK CKPT SEEDS RESULTS_TXT
  bash exps/validation_SafeDreamer_task.sh
done

{
  echo
  echo "================================================================================"
  echo "ALL TASKS FINISHED: $(date -Is)"
  echo "results file: $RESULTS_TXT"
  echo "================================================================================"
} | tee -a "$RESULTS_TXT"

echo
echo "Done. TXT: /usr/home/workspace/$RESULTS_TXT"
