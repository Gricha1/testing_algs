if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi



cd ../..
python main.py --domain_name Safexp \
               --task_name DoggoGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --world_model \
               --modelfree_safety \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --cost_model \
               --cm_frame_stack_num 1 \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 0.05 \
               --controller_safety_coef 0.05 \
               --max_timesteps 4000000 \
               --img_horizon 15 \
               --manager_propose_freq 20 \
               --wm_pretrain \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --wandb_postfix "" \
               --not_use_wandb

