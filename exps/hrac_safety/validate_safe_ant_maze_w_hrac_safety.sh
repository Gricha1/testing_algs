cd ../..
python main.py --seed 2 \
               --load --loaded_exp_num 9 \
               --modelfree_safety \
               --visulazied_episode 1 \
               --validate \
               --env_name SafeAntMazeW \
               --eval_freq 1000 \
               --random_start_pose \
               --world_model \
               --cost_memmory \
               --cost_model \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb
