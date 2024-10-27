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
               --action_repeat 2 \
               --seed $seed \
               --a_net_new_discretization_safety_gym \
               --a_net_discretization_koef 4.0 \
               --clip_a_net_xy \
               --cost_memmory \
               --r_margin_pos 0.5 \
               --r_margin_neg 0.7 \
               --man_rew_scale 100.0 \
               --manager_propose_freq 10 \
               --img_horizon 10 \
               --train_manager_freq 5 \
               --goal_loss_coeff 20 \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

