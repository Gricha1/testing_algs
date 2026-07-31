cd ..

python  SafeDreamer/train.py --configs bsrp_lang_lag --method osrp_lag --run.script eval_only \
                             --run.from_checkpoint logdir_osrp_1124_costlimit2/20250722-131304_osrp_lag_batchcraftext_drinkeasy_0/checkpoint.ckpt \
                             --jax.logical_gpus 0 \
                             --task batchcraftext_drinkeasy --envs.amount 1 --env.batchcraftext.num_envs 1 \
                             --envs.parallel none --envs.restart False \
                             --action_type discrete  \
                             --run.max_episode_steps 1000 \
                             --run.steps 10000 --pid.init_penalty 0.1
                             # 20250722-131304_osrp_lag_batchcraftext_drinkeasy_0
                             # 20250722-131412_osrp_lag_batchcraftext_drinkeasy_0