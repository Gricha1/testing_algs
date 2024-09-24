if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name Safexp \
               --task_name DoggoGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --cost_memmory \
               --validation_without_image --eval_freq 30000 \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb


