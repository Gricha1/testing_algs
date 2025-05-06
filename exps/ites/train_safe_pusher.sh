if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafePusher \
               --domain_name SafetyMaze \
               --pusher_safe_env_safe_zone \
               --pusher_random_obj_start_poses \
               --pusher_three_goal_dim \
               --traj_buffer_size 5000 \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20.0 \
               --cost_budget 0.0 \
               --validation_without_image --eval_freq 30000 \
               --wm_pretrain_epoches 200 \
               --wm_n_initial_exploration_steps 60000 \
               --cost_model \
               --cost_model_trajectory_buffer \
               --world_model \
               --num_networks 6 \
               --pred_hidden_size 50 \
               --num_elites 6 \
               --wm_update_poches 1 \
               --manager_algo "td3_adj_safe_cls_high_lag" \
               --coef_safety_modelfree 80 \
               --controller_algo "td3_img_safe" \
               --controller_safety_coef 1.0 \
               --img_horizon 10 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               --cm_pretrain \
               --wm_pretrain

