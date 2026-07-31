#!/usr/bin/env bash
# Validate SafeDreamer on SafeAntMazeCshape with 10 environment seeds.
# Run INSIDE a safe_dynalang docker container (cwd = /usr/home/workspace).
# The container should be bound to a free physical GPU; inside it use logical GPU 0.
#
# From host (recommended):
#   bash exps/launch_validation_SafeDreamer_cshape_host.sh
#
# Inside container:
#   bash exps/validation_SafeDreamer_safeant_maze_cshape.sh
set -euo pipefail

cd /usr/home/workspace

# Current ongoing Cshape run (override with CKPT=... if needed).
CKPT="${CKPT:-logdir_osrp_1124_costlimit2/20260729-190627_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt}"
# Inside single-GPU containers this must be 0.
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EP_LEN="${EP_LEN:-500}"
# One episode per seed: steps == episode length.
STEPS="${STEPS:-$EP_LEN}"
OUT_ROOT="${OUT_ROOT:-logdir_eval_safedreamer_cshape}"
export COMET_API_KEY="${COMET_API_KEY:-3OfuYHwcRgIwG7DzgzJ190igY}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary_cshape.csv"
echo "seed,logdir,final_reward,final_cost,success_rate,success_binary,length" > "$SUMMARY_CSV"

echo "=== SafeDreamer eval SafeAntMazeCshape ==="
echo "CKPT=$CKPT"
echo "GPU=$GPU"
echo "SEEDS=$SEEDS"
echo "EP_LEN=$EP_LEN STEPS=$STEPS"
echo "OUT_ROOT=$OUT_ROOT"

for SEED in $SEEDS; do
  echo
  echo "===== seed=$SEED ====="
  mapfile -t BEFORE_ARR < <(ls -d logdir_osrp_1124_costlimit2/*_osrp_lag_safeantmaze_cshape_${SEED} 2>/dev/null || true)

  python SafeDreamer/train.py \
    --configs osrp_lag \
    --method osrp_lag \
    --run.script eval_only \
    --run.from_checkpoint "$CKPT" \
    --jax.logical_gpus "$GPU" \
    --task safeantmaze_cshape \
    --seed "$SEED" \
    --env.safeantmaze.seed "$SEED" \
    --envs.amount 1 \
    --run.max_episode_steps "$EP_LEN" \
    --run.steps "$STEPS"

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
  done < <(ls -dt logdir_osrp_1124_costlimit2/*_osrp_lag_safeantmaze_cshape_${SEED} 2>/dev/null || true)

  if [[ -z "$NEWDIR" ]]; then
    NEWDIR=$(ls -dt logdir_osrp_1124_costlimit2/*_osrp_lag_safeantmaze_cshape_${SEED} 2>/dev/null | head -1 || true)
  fi
  if [[ -z "$NEWDIR" ]]; then
    echo "WARN: no logdir for seed=$SEED" >&2
    echo "$SEED,,,," >> "$SUMMARY_CSV"
    continue
  fi

  LINK_DIR="$OUT_ROOT/seed_${SEED}"
  rm -rf "$LINK_DIR"
  ln -sfn "$(cd "$NEWDIR" && pwd)" "$LINK_DIR"

  python - "$NEWDIR" "$SEED" "$SUMMARY_CSV" <<'PY'
import json, sys
from pathlib import Path

logdir = Path(sys.argv[1])
seed = sys.argv[2]
summary = Path(sys.argv[3])
scores = logdir / "scores.jsonl"
final_reward = final_cost = success = length = ""
success_bin = ""
if scores.exists():
    rows = []
    for line in scores.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if rows:
        last = rows[-1]
        final_reward = last.get("episode/score", last.get("episode/reward", ""))
        final_cost = last.get("episode/cost", "")
        success = last.get("episode/success_rate", last.get("episode/SR", ""))
        length = last.get("episode/length", "")
        try:
            success_bin = 1.0 if float(success) > 0 else 0.0
        except Exception:
            success_bin = ""
        print(
            f"seed={seed} reward={final_reward} cost={final_cost} "
            f"success_raw={success} success_bin={success_bin} length={length} n={len(rows)}"
        )
else:
    print(f"seed={seed}: scores.jsonl missing in {logdir}", file=sys.stderr)

with summary.open("a") as f:
    f.write(f"{seed},{logdir},{final_reward},{final_cost},{success},{success_bin},{length}\n")
PY

done

echo
echo "=== done ==="
echo "Summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"

python - "$SUMMARY_CSV" <<'PY'
import csv, sys
from pathlib import Path

p = Path(sys.argv[1])
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
        if r.get(key):
            sr.append(float(r[key]))
    except ValueError:
        pass

print(f"n={len(rows)}")
print(f"final_reward mean={mean(rew):.4f}  values={rew}")
print(f"final_cost   mean={mean(cost):.4f}  values={cost}")
print(f"success_rate mean={mean(sr):.4f}  values={sr}")
PY
