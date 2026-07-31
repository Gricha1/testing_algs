#!/usr/bin/env bash
# Persistent eval container on a FREE host GPU (default 3).
# Creates once, reuses later — no jax re-download every run.
#
#   bash exps/launch_validation_SafeDreamer.sh cshape
#   bash exps/launch_validation_SafeDreamer.sh wshape
#   bash exps/launch_validation_SafeDreamer.sh pusher
#
# Or: TASK=cshape bash exps/launch_validation_SafeDreamer.sh
set -euo pipefail

if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  TASK="$1"
  shift
fi
TASK="${TASK:?Set TASK=cshape|wshape|pusher (or pass as first arg)}"
case "$TASK" in
  cshape|wshape|pusher) ;;
  *) echo "ERROR: TASK must be cshape|wshape|pusher" >&2; exit 1 ;;
esac

REPO="${REPO:-/home/ggorbov/safe_rl_nlp/safe_dynalang}"
IMAGE="${IMAGE:-safe_dynalang_img}"
PHYS_GPU="${PHYS_GPU:-3}"
NAME="${NAME:-safe_dynalang_eval}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LOG="${LOG:-/tmp/safedreamer_eval_${TASK}.log}"

case "$TASK" in
  cshape)
    DEFAULT_CKPT="logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt"
    ;;
  wshape)
    DEFAULT_CKPT="logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_wshape_0/checkpoint.ckpt"
    ;;
  pusher)
    DEFAULT_CKPT="logdir_osrp_1124_costlimit2/20260729-191539_osrp_lag_safeantmaze_pusher_0/checkpoint.ckpt"
    ;;
esac
CKPT="${CKPT:-$DEFAULT_CKPT}"

cd "$REPO"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Starting existing stopped container $NAME"
    docker start "$NAME" >/dev/null
  else
    echo "Creating persistent eval container $NAME on GPU $PHYS_GPU (one-time jax setup may take a few minutes)"
    docker run -d --name "$NAME" --memory=200g \
      --gpus "device=$PHYS_GPU" \
      --entrypoint bash \
      -v "$REPO:/usr/home/workspace" \
      -v "$REPO/logdir:/root/logdir" \
      "$IMAGE" \
      -lc 'sleep infinity'

    echo "One-time env bootstrap..."
    docker exec "$NAME" bash -lc '
      source /opt/conda/etc/profile.d/conda.sh
      conda activate safe_dynalang
      cd /usr/home/workspace/SafeDreamer/embodied/envs/Safety_ant_maze_pusher_envs
      pip install -q gym==0.15.7 gymnasium free-mujoco-py patchelf
      pip install -e . >/tmp/pip_env_eval.log 2>&1 || true
      python - <<'"'"'PYFIX'"'"'
import importlib.metadata as m, subprocess, sys
j, jl = m.version("jax"), m.version("jaxlib")
print(f"[bootstrap] jax={j} jaxlib={jl}")
if not (j.startswith("0.4.11") and jl.startswith("0.4.11")):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "jax==0.4.11", "jaxlib==0.4.11+cuda11.cudnn86",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
    print("[bootstrap] jax fixed to 0.4.11")
import mujoco_py, jax
print("[bootstrap] mujoco_py OK; devices", jax.devices())
PYFIX
    '
    echo "Bootstrap done. Container $NAME kept alive for reuse."
  fi
else
  echo "Reusing running eval container $NAME"
fi

echo "Launching TASK=$TASK ; log -> $LOG"
docker exec -i "$NAME" bash -lc "
  source /opt/conda/etc/profile.d/conda.sh
  conda activate safe_dynalang
  cd /usr/home/workspace
  # Prefer system cuDNN 8.9 over any pip nvidia/cudnn shadow libs.
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  python -c 'import mujoco_py, importlib.metadata as m; print(\"[preflight]\", m.version(\"jax\"), m.version(\"jaxlib\"), \"mujoco_py OK\")'
  export TASK='$TASK'
  export CKPT='$CKPT'
  export SEEDS='$SEEDS'
  export GPU=0
  export FIX_JAX=0
  export SKIP_COMET=1
  export RESULTS_TXT='${RESULTS_TXT:-logdir_eval_safedreamer/results_${TASK}.txt}'
  unset COMET_API_KEY || true
  bash exps/validation_SafeDreamer_task.sh
" | tee "$LOG"

echo
echo "Done. Summary: $REPO/logdir_eval_safedreamer_${TASK}/summary_${TASK}.csv"
column -s, -t "$REPO/logdir_eval_safedreamer_${TASK}/summary_${TASK}.csv" 2>/dev/null || \
  cat "$REPO/logdir_eval_safedreamer_${TASK}/summary_${TASK}.csv"
