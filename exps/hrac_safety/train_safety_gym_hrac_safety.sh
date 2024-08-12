if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

if [ -z "$2" ]; then
    level=1
else
    level=$2
fi



cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal$level \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --subgoal_lower_x 2 \
               --subgoal_lower_x 2 \
               --world_model \
               --modelfree_safety \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --cost_oracle

