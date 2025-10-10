if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

cd ../..
python main.py --seed $seed \
               --env_name SafeAntMazeC \
               --manager_algo none \
               --controller_algo sac_lag \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

