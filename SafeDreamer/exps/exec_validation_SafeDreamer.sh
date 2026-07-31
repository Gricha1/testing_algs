#!/usr/bin/env bash
# From ml3 HOST: run validation inside an EXISTING safe_dynalang container (no new docker).
#
#   bash exps/exec_validation_SafeDreamer.sh cshape
#   bash exps/exec_validation_SafeDreamer.sh wshape
#   bash exps/exec_validation_SafeDreamer.sh pusher
#
# Optional:
#   CID=413c8002abfd bash exps/exec_validation_SafeDreamer.sh cshape
#   FIX_JAX=1 bash exps/exec_validation_SafeDreamer.sh cshape
set -euo pipefail

TASK="${1:?usage: $0 cshape|wshape|pusher}"
case "$TASK" in
  cshape|wshape|pusher) ;;
  *) echo "TASK must be cshape|wshape|pusher" >&2; exit 1 ;;
esac

# Default: map task -> train container (already has working jax).
# Override with CID=...
case "$TASK" in
  cshape) DEFAULT_CID=413c8002abfd ;;
  wshape) DEFAULT_CID=b70f75c3ceb9 ;;
  pusher) DEFAULT_CID=9901d6143a7a ;;
esac
CID="${CID:-$DEFAULT_CID}"
FIX_JAX="${FIX_JAX:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LOG="${LOG:-/tmp/safedreamer_eval_${TASK}.log}"

echo "Using existing container CID=$CID for TASK=$TASK"
echo "Log -> $LOG"
echo "NOTE: this shares the container GPU with training; watch VRAM."

# Kill any previous hung eval in this container (safe: only eval_only).
docker exec "$CID" bash -lc 'pkill -f "SafeDreamer/train.py --configs osrp_lag --method osrp_lag --run.script eval_only" || true' >/dev/null 2>&1 || true

docker exec -i "$CID" bash -lc "
  source /opt/conda/etc/profile.d/conda.sh
  conda activate safe_dynalang
  cd /usr/home/workspace
  python -c 'import mujoco_py; print(\"[preflight] mujoco_py OK\")' || \
    pip install -q free-mujoco-py patchelf gym==0.15.7
  export TASK='$TASK'
  export FIX_JAX='$FIX_JAX'
  export SEEDS='$SEEDS'
  export GPU=0
  export SKIP_COMET=1
  unset COMET_API_KEY || true
  bash exps/validation_SafeDreamer_${TASK}.sh
" | tee "$LOG"

echo "Done. Summary: ~/safe_rl_nlp/safe_dynalang/logdir_eval_safedreamer_${TASK}/summary_${TASK}.csv"
