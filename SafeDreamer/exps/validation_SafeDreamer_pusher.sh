#!/usr/bin/env bash
# In-container SafeDreamer validation (no new docker, no jax reinstall by default).
# Run AFTER: docker exec -it <container> bash
#            source /opt/conda/etc/profile.d/conda.sh && conda activate safe_dynalang
#            cd /usr/home/workspace
#
#   bash exps/validation_SafeDreamer_cshape.sh
#   bash exps/validation_SafeDreamer_wshape.sh
#   bash exps/validation_SafeDreamer_pusher.sh
#
# Or directly:
#   TASK=cshape FIX_JAX=0 bash exps/validation_SafeDreamer_task.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
case "$(basename "$0")" in
  *cshape*) export TASK=cshape ;;
  *wshape*) export TASK=wshape ;;
  *pusher*) export TASK=pusher ;;
  *) echo "Use validation_SafeDreamer_{cshape,wshape,pusher}.sh" >&2; exit 1 ;;
esac
export FIX_JAX="${FIX_JAX:-0}"
export GPU="${GPU:-0}"
bash "$ROOT/validation_SafeDreamer_task.sh"
