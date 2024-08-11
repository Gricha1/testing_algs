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
               --seed $seed \
               --world_model \
               --validation_without_image --eval_freq 30000 \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --controller_safe_model \
               --cost_oracle

