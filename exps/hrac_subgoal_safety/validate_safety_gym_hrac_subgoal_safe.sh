cd ../..
python main.py --visulazied_episode 10 --validate \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --cm_frame_stack_num 1 \
               --modelfree_safety \
               --controller_safe_model \
               --wandb_postfix "validate" \
               --load --loaded_exp_num 626 \
               --cost_oracle
