if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafePusher \
               --domain_name SafetyMaze \
               --pusher_safe_env \
               --pusher_random_obj_start_poses \
               --pusher_three_goal_dim \
               --traj_buffer_size 5000 \
               --cost_budget 10.0 \
               --load \
               --loaded_exp_num ml4_233 \
               --load_without_cost_model \
               --load_without_world_model \
               --validation_without_image --eval_freq 30000 \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --cost_model \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20.0 \
               --cost_model_trajectory_buffer \
               --manager_algo "td3_adj_safe_cls_high_lag" \
               --coef_safety_modelfree 80 \
               --controller_algo "td3" \
               --controller_safety_coef 6 \
               --img_horizon 10 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               #--cm_pretrain \
               #--wm_pretrain

