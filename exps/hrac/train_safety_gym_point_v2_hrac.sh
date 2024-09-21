if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal2 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --cost_memmory \
               --manager_propose_freq 20 \
               --img_horizon 15 \
               --train_manager_freq 10 \
               --goal_loss_coeff 20 \
               --validation_without_image --eval_freq 30000 \
               --man_rew_scale 0.1 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

