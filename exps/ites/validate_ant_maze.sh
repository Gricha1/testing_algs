cd ../..

python main.py --visulazied_episode 10 \
               --validate \
               --env_name SafeAntMazeC \
               --seed 2 \
               --world_model \
               --cost_model \
               --manager_algo td3_adj_safe_cls \
               --controller_algo td3_img_safe \
               --wandb_postfix "validate" \
               --cost_model_heatmap \
               --load --loaded_exp_num 0 \
               --not_use_wandb