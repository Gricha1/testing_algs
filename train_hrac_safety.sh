
python main.py --seed 2 \
               --modelfree_safety \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --train_safe_model --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" 

