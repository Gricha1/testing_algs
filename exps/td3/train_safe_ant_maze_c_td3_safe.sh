if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --env_name SafeAntMazeC \
               --seed $seed \
               --controller_algo td3_lag \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \

