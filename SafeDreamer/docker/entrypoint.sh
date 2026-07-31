#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate safe_dynalang

need_fix=0
if ! python - <<'PY'
import importlib.metadata as m
j = m.version("jax")
jl = m.version("jaxlib")
print(f"[entrypoint] jax={j} jaxlib={jl}")
ok = j.startswith("0.4.11") and jl.startswith("0.4.11")
raise SystemExit(0 if ok else 2)
PY
then
  need_fix=1
fi

# SystemExit(2) makes python return non-zero -> need_fix=1
# SystemExit(0) -> need_fix stays 0 because `if !` is false

if [ "$need_fix" -eq 1 ]; then
  echo "[entrypoint] fixing jax/jaxlib -> 0.4.11 ..."
  pip install --upgrade "jax[cuda11_pip]==0.4.11" "jaxlib==0.4.11+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  python - <<'PY'
import jax, jaxlib
print("[entrypoint] fixed", jax.__version__, jaxlib.__version__)
print("[entrypoint] devices", jax.devices())
PY
fi

cd /usr/home/workspace/exps 2>/dev/null || cd /usr/home/workspace
exec bash --login
