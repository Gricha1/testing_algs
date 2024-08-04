python main.py --visulazied_episode 0 --validate \
               --load --loaded_exp_num 499 \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --cost_memmory \
               --modelfree_safety \
               --train_safe_model --controller_safe_model \
               --wandb_postfix "validate"
