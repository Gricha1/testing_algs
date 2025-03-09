if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafeAntMazeS \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20.0 \
               --world_model \
               --cost_model_trajectory_buffer \
               --cost_model \
               --cm_frame_stack_num 1 \
               --manager_algo td3_adj_safe_cls_high_lag \
               --coef_safety_modelfree 800 \
               --controller_algo "td3_img_safe" \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \

