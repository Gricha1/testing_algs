cd ..

#pip install setuptools==65.5.0 "wheel<0.40.0"
pip install gym==0.15.7
pip install onnxruntime free-mujoco-py
pip install patchelf
cd /usr/home/workspace/SafeDreamer/embodied/envs/Safety_ant_maze_pusher_envs
pip install -e .
cd /usr/home/workspace

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"
# Ensure jax/jaxlib match (image often has jax>jaxlib after other pip installs)
python - <<'PYFIX'
import importlib.metadata as m, subprocess, sys
j, jl = m.version("jax"), m.version("jaxlib")
print(f"[preflight] jax={j} jaxlib={jl}")
if not (j.startswith("0.4.11") and jl.startswith("0.4.11")):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "jax[cuda11_pip]==0.4.11", "jaxlib==0.4.11+cuda11.cudnn86",
        "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ])
PYFIX

python SafeDreamer/train.py --configs osrp_lag --method osrp_lag \
                            --task safeantmaze_pusher \
                            --jax.logical_gpus 0 --envs.amount 1