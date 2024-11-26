cd ../..
python main.py --visulazied_episode 25 --validate \
               --load --loaded_exp_num 880 \
               --img_horizon 10 \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --cost_memmory \
               --wandb_postfix "validate" \
               --not_use_wandb
