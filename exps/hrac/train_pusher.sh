if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name SafetyMaze \
               --env_name SafePusher \
               --seed $seed \
               --goal_loss_coeff 20 \
               --validation_without_image \
               --eval_freq 30000 \
               --pusher_random_obj_start_poses \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

