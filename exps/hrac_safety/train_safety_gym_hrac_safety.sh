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
               --modelfree_safety \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --train_safe_model --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" 

