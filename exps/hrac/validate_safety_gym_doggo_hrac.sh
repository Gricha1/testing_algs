cd ../..
python main.py --visulazied_episode 10 --validate \
               --load --loaded_exp_num ml3_6 \
               --domain_name Safexp \
               --task_name DoggoGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --cost_memmory \
               --wandb_postfix "validate" \
               --not_use_wandb
