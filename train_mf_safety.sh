
python main.py --modelfree_safety \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --train_safe_model --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20 \
               --coef_safety_modelfree 800 \
               --max_timesteps 1800000 \
               --wandb_postfix ""
