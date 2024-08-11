python main.py --visulazied_episode 0 --validate \
               --load --loaded_exp_num ml5_121 \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --cost_memmory \
               --wandb_postfix "validate"
