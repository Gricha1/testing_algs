if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --action_repeat 2 \
               --seed $seed \
               --cost_memmory \
               --r_margin_pos 0.5 \
               --r_margin_neg 0.7 \
               --man_rew_scale 10.0 \
               --manager_propose_freq 10 \
               --img_horizon 10 \
               --train_manager_freq 5 \
               --goal_loss_coeff 0.01 \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

