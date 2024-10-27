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
               --a_net_discretization_koef 3.0 \
               --clip_a_net_xy \
               --cost_memmory \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20 \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

