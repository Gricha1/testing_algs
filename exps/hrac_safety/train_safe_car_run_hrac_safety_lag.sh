if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafeBulletCarRun \
               --domain_name BulletSafeGym \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20.0 \
               --world_model \
               --wm_pretrain \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --cost_memmory \
               --cost_model \
               --cm_frame_stack_num 8 \
               --cm_pretrain \
               --cost_model_batch_size 512 \
               --modelfree_safety \
               --coef_safety_modelfree 80 \
               --controller_imagination_safety_loss \
               --controller_safety_coef 0.6 \
               --controller_cumul_img_safety \
               --controller_use_lagrange \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

