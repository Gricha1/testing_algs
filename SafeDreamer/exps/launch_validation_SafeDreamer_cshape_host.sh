#!/usr/bin/env bash
# Host-side launcher for SafeDreamer SafeAntMazeCshape validation on ml3.
# Spins a temporary container on a free physical GPU and runs 10-seed eval.
set -euo pipefail

REPO="${REPO:-/home/ggorbov/safe_rl_nlp/safe_dynalang}"
IMAGE="${IMAGE:-safe_dynalang_img}"
# Prefer free host GPUs 3/4 (0/1/2 are used by current trainings).
PHYS_GPU="${PHYS_GPU:-3}"
NAME="${NAME:-safe_dynalang_eval_cshape}"
CKPT="${CKPT:-logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LOG="${LOG:-/tmp/safedreamer_eval_cshape.log}"

cd "$REPO"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Removing old container $NAME"
  docker rm -f "$NAME" >/dev/null
fi

echo "Starting $NAME on physical GPU $PHYS_GPU"
docker run -d --name "$NAME" --memory=200g \
  --gpus "device=$PHYS_GPU" \
  --entrypoint bash \
  -v "$REPO:/usr/home/workspace" \
  -v "$REPO/logdir:/root/logdir" \
  "$IMAGE" \
  -lc 'sleep infinity'

cleanup() {
  echo "Stopping container $NAME"
  docker rm -f "$NAME" >/dev/null || true
}
trap cleanup EXIT

echo "Warming conda + env install (Safety_ant_maze_pusher_envs)"
docker exec "$NAME" bash -lc '
  source /opt/conda/etc/profile.d/conda.sh
  conda activate safe_dynalang
  cd /usr/home/workspace/SafeDreamer/embodied/envs/Safety_ant_maze_pusher_envs
  pip install -e . >/tmp/pip_env_eval.log 2>&1 || true
  tail -5 /tmp/pip_env_eval.log
'

echo "Launching validation; log -> $LOG"
docker exec "$NAME" bash -lc "
  source /opt/conda/etc/profile.d/conda.sh
  conda activate safe_dynalang
  cd /usr/home/workspace
  export CKPT='$CKPT'
  export SEEDS='$SEEDS'
  export GPU=0
  bash exps/validation_SafeDreamer_safeant_maze_cshape.sh
" | tee "$LOG"

echo "Done. Summary inside repo: logdir_eval_safedreamer_cshape/summary_cshape.csv"
