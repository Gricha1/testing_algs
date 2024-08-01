if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

python main.py --seed $seed \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --man_rew_scale 0.1 --goal_loss_coeff 20 \
               --max_timesteps 4000000 \
               --wandb_postfix ""