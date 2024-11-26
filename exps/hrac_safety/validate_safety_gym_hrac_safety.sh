cd ../..

python main.py --visulazied_episode 25 \
               --validate \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --img_horizon 10 \
               --manager_propose_freq 10 \
               --wandb_postfix "validate" \
               --load --loaded_exp_num ml3_89 \
               --not_use_wandb
