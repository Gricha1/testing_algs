#!/usr/bin/env bash
# Validate SafeDreamer on Cshape + Wshape + Pusher (10 env seeds each).
# Uses current training checkpoints in logdir_osrp_1124_costlimit2/.
# Writes a human-readable TXT with every trajectory + means.
#
# From ml3 host:
#   bash exps/launch_validation_SafeDreamer_all.sh
#
# Optional:
#   SEEDS="0 1 2 3 4" PHYS_GPU=3 bash exps/launch_validation_SafeDreamer_all.sh
set -euo pipefail

REPO="${REPO:-/home/ggorbov/safe_rl_nlp/safe_dynalang}"
IMAGE="${IMAGE:-safe_dynalang_img}"
PHYS_GPU="${PHYS_GPU:-3}"
NAME="${NAME:-safe_dynalang_eval}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
TASKS="${TASKS:-cshape wshape pusher}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_TXT="${RESULTS_TXT:-logdir_eval_safedreamer/results_all_${STAMP}.txt}"
LOG="${LOG:-/tmp/safedreamer_eval_all_${STAMP}.log}"

CKPT_CSHAPE="logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt"
CKPT_WSHAPE="logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_wshape_0/checkpoint.ckpt"
CKPT_PUSHER="logdir_osrp_1124_costlimit2/20260729-191539_osrp_lag_safeantmaze_pusher_0/checkpoint.ckpt"

ckpt_for() {
  case "$1" in
    cshape) echo "$CKPT_CSHAPE" ;;
    wshape) echo "$CKPT_WSHAPE" ;;
    pusher) echo "$CKPT_PUSHER" ;;
    *) echo "" ;;
  esac
}

cd "$REPO"

# Ensure persistent eval container on a free GPU (created once).
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Starting existing stopped container $NAME"
    docker start "$NAME" >/dev/null
  else
    echo "Creating persistent eval container $NAME on GPU $PHYS_GPU"
    docker run -d --name "$NAME" --memory=200g \
      --gpus "device=$PHYS_GPU" \
      --entrypoint bash \
      -v "$REPO:/usr/home/workspace" \
      -v "$REPO/logdir:/root/logdir" \
      "$IMAGE" \
      -lc 'sleep infinity'
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
import mujoco_py, jax
print("[bootstrap] OK devices", jax.devices())
PYFIX
    '
  fi
else
  echo "Reusing eval container $NAME"
fi

mkdir -p "$(dirname "$RESULTS_TXT")"
{
  echo "SafeDreamer validation — all environments"
  echo "started: $(date -Is)"
  echo "seeds: $SEEDS"
  echo "container: $NAME  phys_gpu: $PHYS_GPU"
  echo
  for t in $TASKS; do
    echo "ckpt[$t]=$(ckpt_for "$t")"
  done
  echo
} > "$RESULTS_TXT"

docker exec "$NAME" bash -lc "mkdir -p /usr/home/workspace/$(dirname "$RESULTS_TXT")"

echo "Results TXT -> $REPO/$RESULTS_TXT"
echo "Full log    -> $LOG"

for TASK in $TASKS; do
  CKPT="$(ckpt_for "$TASK")"
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: missing checkpoint for $TASK: $CKPT" | tee -a "$RESULTS_TXT"
    exit 1
  fi
  echo
  echo "########## TASK=$TASK ##########"
  docker exec -i "$NAME" bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate safe_dynalang
    cd /usr/home/workspace
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export TASK='$TASK'
    export CKPT='$CKPT'
    export SEEDS='$SEEDS'
    export GPU=0
    export FIX_JAX=0
    export SKIP_COMET=1
    export RESULTS_TXT='$RESULTS_TXT'
    unset COMET_API_KEY || true
    bash exps/validation_SafeDreamer_task.sh
  "
done | tee "$LOG"

{
  echo
  echo "================================================================================"
  echo "ALL TASKS FINISHED: $(date -Is)"
  echo "results file: $RESULTS_TXT"
  echo "================================================================================"
} | tee -a "$RESULTS_TXT"

echo
echo "Done."
echo "TXT: $REPO/$RESULTS_TXT"
echo "Log: $LOG"
