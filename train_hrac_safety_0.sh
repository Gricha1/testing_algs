# HRAC + MB
# --eval_freq 1000
#python main.py --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 1 --safety_subgoals --testing_safety_subgoal --safety_loss_coef 20 --max_timesteps 900000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"
#python main.py --train_safe_model --eval_freq 100000 --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 20 --max_timesteps 1500000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"
#python main.py --validation_without_image --random_start_pose --eval_freq 30000 --safety_subgoals --testing_safety_subgoal --train_safe_model --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 20 --safety_loss_coef 400 --max_timesteps 1800000 --wandb_postfix ""

python main.py --load_adj_net --load --loaded_exp_num 388 \
               --modelfree_safety \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --train_safe_model --controller_safe_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --controller_safety_coef 150 \
               --controller_imagination_safety_loss \
               --max_timesteps 1800000 \
               --wandb_postfix "" 
