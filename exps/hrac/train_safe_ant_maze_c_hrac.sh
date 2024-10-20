if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafeAntMazeC \
               --validation_without_image \
               --eval_freq 30000 \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               --random_start_pose