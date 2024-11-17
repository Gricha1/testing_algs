if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi



cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --action_repeat 2 \
               --goal_conditioned \
               --vector_env \
               --sparce_reward \
               --seed $seed \
               --validation_without_image --eval_freq 30000 \
               --a_net_new_discretization_safety_gym \
               --a_net_discretization_koef 3.0 \
               --manager_propose_freq 10 \
               --train_manager_freq 5 \
               --man_rew_scale 100.0 \
               --goal_loss_coeff 20.0 \
               --r_margin_pos 0.5 \
               --r_margin_pos 0.7 \
               --modelfree_safety \
               --coef_safety_modelfree 10.0 \
               --world_model \
               --wm_pretrain \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --cost_model \
               --cm_frame_stack_num 8 \
               --cm_pretrain \
               --cost_model_batch_size 512 \
               --controller_imagination_safety_loss \
               --controller_safety_coef 0.001 \
               --controller_cumul_img_safety \
               --controller_use_lagrange \
               --max_timesteps 4000000 \
               --img_horizon 10 \
               --wandb_postfix "" \
               --not_use_wandb

