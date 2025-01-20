cd ../..

python main.py --visulazied_episode 26 \
               --validate \
               --domain_name SafetyMaze \
               --env_name SafePusher \
               --seed 344 \
               --wandb_postfix "validate" \
               --load --loaded_exp_num 130 \
               --not_use_wandb
