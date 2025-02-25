if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --seed $seed \
               --manager_algo none \
               --controller_algo td3_lag \
               --img_horizon 10 \
               --validation_without_image --eval_freq 1000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

