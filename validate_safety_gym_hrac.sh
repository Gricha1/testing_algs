# --visulazied_episode 0 \

python main.py --validate \
               --load --loaded_exp_num 490 \
               --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --validation_without_image \
               --wandb_postfix "" 

