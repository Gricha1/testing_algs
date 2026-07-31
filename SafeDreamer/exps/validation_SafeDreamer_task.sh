#!/usr/bin/env bash
# Generic SafeDreamer eval_only over 10 env seeds.
# Required: TASK in {cshape,wshape,pusher}
# Run INSIDE safe_dynalang docker (cwd=/usr/home/workspace), logical GPU usually 0.
#
# Examples:
#   TASK=cshape bash exps/validation_SafeDreamer_task.sh
#   TASK=wshape CKPT=.../checkpoint.ckpt bash exps/validation_SafeDreamer_task.sh
set -euo pipefail

cd /usr/home/workspace

# Jax fix is OFF by default (avoids long downloads). Enable with FIX_JAX=1 if needed.
FIX_JAX="${FIX_JAX:-0}"
python - <<PYFIX
import importlib.metadata as m, os, subprocess, sys
j, jl = m.version("jax"), m.version("jaxlib")
print(f"[preflight] jax={j} jaxlib={jl}")
ok = j.startswith("0.4.11") and jl.startswith("0.4.11")
if not ok:
    if os.environ.get("FIX_JAX", "0") != "1":
        print(
            "[preflight] WARN: jax/jaxlib mismatch. "
            "Re-run with FIX_JAX=1 to install 0.4.11, or use a train container where jax is already fixed.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "jax[cuda11_pip]==0.4.11", "jaxlib==0.4.11+cuda11.cudnn86",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
PYFIX

TASK="${TASK:?Set TASK=cshape|wshape|pusher}"
case "$TASK" in
  cshape|wshape|pusher) ;;
  *) echo "ERROR: TASK must be cshape|wshape|pusher, got: $TASK" >&2; exit 1 ;;
esac

TASK_FULL="safeantmaze_${TASK}"
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
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EP_LEN="${EP_LEN:-500}"
STEPS="${STEPS:-$EP_LEN}"
OUT_ROOT="${OUT_ROOT:-logdir_eval_safedreamer_${TASK}}"
RESULTS_TXT="${RESULTS_TXT:-logdir_eval_safedreamer/results_${TASK}.txt}"
export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"
# Avoid Comet hanging the eval loop on upload confirmation.
export SKIP_COMET="${SKIP_COMET:-1}"
# Do not force Comet login during eval.
if [[ "${SKIP_COMET}" == "1" ]]; then
  unset COMET_API_KEY || true
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$(dirname "$RESULTS_TXT")"
SUMMARY_CSV="$OUT_ROOT/summary_${TASK}.csv"
echo "seed,logdir,final_reward,final_cost,success_rate,success_binary,length" > "$SUMMARY_CSV"

# Header for this task in the shared txt (append-safe).
{
  echo
  echo "================================================================================"
  echo "TASK: ${TASK_FULL}"
  echo "CKPT: ${CKPT}"
  echo "SEEDS: ${SEEDS}"
  echo "started: $(date -Is)"
  echo "================================================================================"
} | tee -a "$RESULTS_TXT"

echo "=== SafeDreamer eval ${TASK_FULL} ==="
echo "CKPT=$CKPT"
echo "GPU=$GPU"
echo "SEEDS=$SEEDS"
echo "EP_LEN=$EP_LEN STEPS=$STEPS"
echo "OUT_ROOT=$OUT_ROOT"
echo "RESULTS_TXT=$RESULTS_TXT"

for SEED in $SEEDS; do
  echo
  echo "===== seed=$SEED ====="
  mapfile -t BEFORE_ARR < <(ls -d logdir_osrp_1124_costlimit2/*_osrp_lag_${TASK_FULL}_${SEED} 2>/dev/null || true)

  # parallel=process spawns workers that often miss conda site-packages (mujoco_py).
  # wrapper.length forces episode end so scores.jsonl is written.
  python SafeDreamer/train.py \
    --configs osrp_lag \
    --method osrp_lag \
    --run.script eval_only \
    --run.from_checkpoint "$CKPT" \
    --jax.logical_gpus "$GPU" \
    --task "$TASK_FULL" \
    --seed "$SEED" \
    --env.safeantmaze.seed "$SEED" \
    --envs.amount 1 \
    --envs.parallel none \
    --envs.restart False \
    --wrapper.length "$EP_LEN" \
    --run.max_episode_steps "$EP_LEN" \
    --run.steps "$((EP_LEN + 5))" \
    --run.log_every 50

  NEWDIR=""
  while IFS= read -r d; do
    found=0
    for b in "${BEFORE_ARR[@]:-}"; do
      [[ "$b" == "$d" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
      NEWDIR="$d"
      break
    fi
  done < <(ls -dt logdir_osrp_1124_costlimit2/*_osrp_lag_${TASK_FULL}_${SEED} 2>/dev/null || true)

  if [[ -z "$NEWDIR" ]]; then
    NEWDIR=$(ls -dt logdir_osrp_1124_costlimit2/*_osrp_lag_${TASK_FULL}_${SEED} 2>/dev/null | head -1 || true)
  fi
  if [[ -z "$NEWDIR" ]]; then
    echo "WARN: no logdir for seed=$SEED" >&2
    echo "$SEED,,,," >> "$SUMMARY_CSV"
    continue
  fi

  LINK_DIR="$OUT_ROOT/seed_${SEED}"
  rm -rf "$LINK_DIR"
  ln -sfn "$(cd "$NEWDIR" && pwd)" "$LINK_DIR"

  python - "$NEWDIR" "$SEED" "$SUMMARY_CSV" "$RESULTS_TXT" "$TASK_FULL" "$CKPT" <<'PY'
import json, sys
from pathlib import Path

logdir = Path(sys.argv[1])
seed = sys.argv[2]
summary = Path(sys.argv[3])
results_txt = Path(sys.argv[4])
task = sys.argv[5]
ckpt = sys.argv[6]

def last_episode_metrics(logdir: Path):
    for fname in ("scores.jsonl", "metrics.jsonl"):
        path = logdir / fname
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        for row in reversed(rows):
            if any(k.startswith("episode/") for k in row):
                return row
        if rows:
            return rows[-1]
    return {}

row = last_episode_metrics(logdir)
final_reward = row.get("episode/score", row.get("episode/reward", ""))
final_cost = row.get("episode/cost", "")
success = row.get("episode/success_rate", row.get("episode/SR", ""))
length = row.get("episode/length", "")
success_bin = ""
try:
    success_bin = 1.0 if float(success) > 0 else 0.0
except Exception:
    success_bin = ""

block = "\n".join([
    "-" * 60,
    f"trajectory: task={task}  seed={seed}",
    f"  final_reward   = {final_reward}",
    f"  final_cost     = {final_cost}",
    f"  success_rate   = {success_bin if success_bin != '' else success}",
    f"  length         = {length}",
    f"  checkpoint     = {ckpt}",
    f"  logdir         = {logdir}",
    "-" * 60,
])
print(block)
results_txt.parent.mkdir(parents=True, exist_ok=True)
with results_txt.open("a") as f:
    f.write(block + "\n")

with summary.open("a") as f:
    f.write(f"{seed},{logdir},{final_reward},{final_cost},{success},{success_bin},{length}\n")
PY

done

echo
echo "=== done ==="
echo "Summary CSV: $SUMMARY_CSV"
echo "Results TXT: $RESULTS_TXT"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"

python - "$SUMMARY_CSV" "$RESULTS_TXT" "$TASK_FULL" <<'PY'
import csv, sys
from pathlib import Path

p = Path(sys.argv[1])
results_txt = Path(sys.argv[2])
task = sys.argv[3]
rows = list(csv.DictReader(p.open()))

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

rew, cost, sr = [], [], []
for r in rows:
    try:
        if r.get("final_reward"):
            rew.append(float(r["final_reward"]))
        if r.get("final_cost"):
            cost.append(float(r["final_cost"]))
        key = "success_binary" if r.get("success_binary") not in (None, "") else "success_rate"
        if r.get(key) not in (None, ""):
            sr.append(float(r[key]))
    except ValueError:
        pass

summary = "\n".join([
    f"[MEAN] task={task}  n={len(rows)}",
    f"  final_reward mean = {mean(rew):.6f}  values={rew}",
    f"  final_cost   mean = {mean(cost):.6f}  values={cost}",
    f"  success_rate mean = {mean(sr):.6f}  values={sr}",
    "",
])
print(summary)
with results_txt.open("a") as f:
    f.write(summary)
PY
