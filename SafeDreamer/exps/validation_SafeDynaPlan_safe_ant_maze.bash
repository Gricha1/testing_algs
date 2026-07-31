cd ..

#pip install setuptools==65.5.0 "wheel<0.40.0"
pip install gym==0.15.7
pip install onnxruntime free-mujoco-py
pip install patchelf
cd /usr/home/workspace/SafeDreamer/embodied/envs/Safety_ant_maze_pusher_envs
pip install -e .
cd /usr/home/workspace

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

python  SafeDreamer/train.py --configs osrp_lag --method osrp_lag --run.script eval_only \
                             -run.from_checkpoint logdir_osrp_1124_costlimit2/20251010-144108_osrp_lag_safeantmaze_cshape_0/checkpoint.ckpt \
                             --jax.logical_gpus 0 \
                             --task safeantmaze_cshape --envs.amount 1 --env.batchcraftext.num_envs 1 \
                             --run.max_episode_steps 500 \
                             --run.steps 10000
                             #--run.from_checkpoint logdir_osrp_1124_costlimit2/20251113-150049_osrp_lag_safeantmaze_wshape_0/checkpoint.ckpt \
                             # 20250722-131304_osrp_lag_batchcraftext_drinkeasy_0
                             # 20250722-131412_osrp_lag_batchcraftext_drinkeasy_0