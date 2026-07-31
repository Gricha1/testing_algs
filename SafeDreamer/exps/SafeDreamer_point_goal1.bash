cd ..

export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"
python SafeDreamer/train.py --configs osrp_lag --method osrp_lag \
                            --task safetygym_SafetyPointGoal1-v0 \
                            --jax.logical_gpus 0 --envs.amount 1