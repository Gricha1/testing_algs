if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi


python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --cost_memmory \
               --modelfree_safety \
               --validation_without_image --eval_freq 30000 \
               --train_safe_model --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --max_timesteps 4000000 \
               --wandb_postfix "" 

