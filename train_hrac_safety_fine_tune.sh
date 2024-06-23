# HRAC + MB
# --eval_freq 1000
#python main.py --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 1 --safety_subgoals --testing_safety_subgoal --safety_loss_coef 20 --max_timesteps 900000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"
#python main.py --train_safe_model --eval_freq 100000 --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 20 --max_timesteps 1500000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"
#python main.py --safety_loss_coef 1 --safety_subgoals --testing_safety_subgoal --train_safe_model --eval_freq 100000 --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 20 --max_timesteps 1500000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"
#python main.py --validation_without_image --random_start_pose --eval_freq 100000 --safety_subgoals --testing_safety_subgoal --train_safe_model --controller_safe_model --man_rew_scale 0.1 --goal_loss_coeff 20 --safety_loss_coef 1 --max_timesteps 1500000 --wandb_postfix ""


python main.py --load_adj_net --load --loaded_exp_num 278 \
               --safe_model_grad_clip 200 \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --safety_subgoals --train_safe_model --controller_safe_model \
               --ctrl_crit_lr 0.0 --ctrl_act_lr 0.0 --safe_model_loss_coef 0.0 \
               --adj_loss_coef 0.0 \
               --man_rew_scale 1 --goal_loss_coeff 20.0 --safety_loss_coef 400 \
               --max_timesteps 1500000 \
               --wandb_postfix ""


#python main.py --world_model --safety_subgoals --testing_safety_subgoal --safety_loss_coef 200 --max_timesteps 1500000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=200"
#python main.py --world_model --safety_subgoals --testing_safety_subgoal --safety_loss_coef 200 --max_timesteps 1500000 --wandb_postfix ", test_safety_loss, coef_safety=200"
#python main.py --world_model --safety_subgoals --testing_safety_subgoal --safety_loss_coef 2000 --max_timesteps 1500000 --wandb_postfix ", test_safety_loss, coef_safety=2000"
#python main.py --world_model --safety_subgoals --safety_loss_coef 20 --max_timesteps 1500000 --wandb_postfix "coef_safety=200"
#python main.py --world_model --safety_subgoals --safety_loss_coef 2000 --max_timesteps 1500000 --wandb_postfix "coef_safety=2000"
#python main.py --world_model --safety_subgoals --pred_hidden_size 600 --max_timesteps 2500000 --wandb_postfix "wm_hidden_size=600, safety_subgoals"
#python main.py --world_model --safety_subgoals --pred_hidden_size 1000 --max_timesteps 2500000 --wandb_postfix "wm_hidden_size=1000, safety_subgoals"
#python main.py --world_model --safety_subgoals --pred_hidden_size 200 --max_timesteps 2500000 --wandb_postfix "wm_hidden_size=1000, safety_subgoals"

#python main.py --world_model --safety_subgoals --img_horizon 5 --max_timesteps 1500000 --wandb_postfix "wm_hidden_size=1000, safety_subgoals"
#python main.py --world_model --safety_subgoals --img_horizon 10 --max_timesteps 1500000 --wandb_postfix "wm_hidden_size=1000, safety_subgoals"
#python main.py --world_model --safety_subgoals --img_horizon 30 --max_timesteps 1500000 --wandb_postfix "wm_hidden_size=1000, safety_subgoals"
#python main.py --world_model --safety_subgoals --max_timesteps 1500000 --wandb_postfix "safety_subgoals"
#python main.py --world_model --safety_subgoals --safety_loss_coef 20000 --max_timesteps 1500000 --wandb_postfix "safety_subgoals"


#python main.py --max_timesteps 3000000 --wandb_postfix "base"

# PPO
#python main.py --env_name AntMaze --PPO --max_timesteps 2000000 --wandb_postfix "test"


#python main.py --env_name AntMaze --PPO --inner_dones --ent_coef 0.2 --gae_lambda 0.95 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "inner_dones, ent_coef=0.2, gae=0.95, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"

#python main.py --env_name AntMaze --inner_dones --ctrl_batch_size 128 --ctrl_buffer_size 200000 --ctrl_discount 0.95 --max_timesteps 2000000 



#python main.py --env_name AntMaze --ctrl_discount 0.95 --max_timesteps 1500000 
#python main.py --env_name AntMaze --inner_dones --ctrl_batch_size --ctrl_discount 0.95 --max_timesteps 1500000 

#python main.py --env_name AntMaze --PPO --ent_coef 0.2 --gae_lambda 0.95 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "ent_coef=0.2, gae=0.95, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --inner_dones --ent_coef 0.2 --gae_lambda 0.95 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "inner_dones, ent_coef=0.2, gae=0.95, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --inner_dones --ctrl_batch_size --ctrl_discount 0.95 --max_timesteps 2000000 

#python main.py --env_name AntMaze --PPO --ent_coef 0.2 --weight_decay_ppo 0.0001 --gae_lambda 0.95 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "ent_coef=0.2, weight_decay=0.0001, gae=0.95, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --gae_lambda 0.95 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "gae=0.95, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --ctrl_rew_scale 0.001 --weight_decay_ppo 0.0001 --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "weight_decay=0.0001, ctrl_rew_scale=0.001, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --ctrl_rew_scale 0.001 --weight_decay_ppo 0.001 --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "weight_decay=0.001, ctrl_rew_scale=0.001, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"


#python main.py --env_name AntMaze --PPO --weight_decay_ppo 0.0001 --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "weight_decay=0.0001, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --max_grad_norm 0.001 --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "max_grad_norm=0.001, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --absolute_goal --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "abs_goal, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --norm_adv False --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "norm_adv=Faslse, gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --gae_lambda 0.8 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "gae=0.8, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --gae_lambda 0.9 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "gae=0.9, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --update_epochs 50 --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "epoches=50, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 2000000 --wandb_postfix "gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"


#python main.py --env_name AntMaze --PPO --ctrl_discount 0.99 --ctrl_batch_size 2048 --ctrl_buffer_size 2048 --max_timesteps 1500000


#python main.py --env_name AntMaze --PPO --ppo_lr 1e-3 --ctrl_discount 0.95 --num_minibatches 8 --ctrl_batch_size 8192 --ctrl_buffer_size 8192 --max_timesteps 2000000 --wandb_postfix "pi_lr=1e-3, gamma=0.95, minibatch=1024, ppo_lr=1e-3, batch=8192"

#python main.py --env_name AntMaze --PPO --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 8 --ctrl_batch_size 8192 --ctrl_buffer_size 8192 --max_timesteps 2000000 --wandb_postfix "pi_lr=1e-3, gamma=0.95, minibatch=1024, ppo_lr=1e-4, batch=8192"


#python main.py --env_name AntMaze --PPO --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 1500000 --wandb_postfix "gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --inner_dones --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 5000000 --wandb_postfix "inner_done=true, gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"
#python main.py --env_name AntMaze --PPO --ppo_lr 1e-4 --ctrl_discount 0.95 --num_minibatches 32 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 5000000 --wandb_postfix "gamma=0.95, minibatch=128, ppo_lr=1e-4, batch=4096"

# 300 arch, batch=4096, mini_batch=1024
#python main.py --env_name AntMaze --PPO --max_grad_norm 0.1 --num_minibatches 4 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 5000000 --wandb_postfix "gradnorm=0.1, minibatch=1024, batch=4096"
#python main.py --env_name AntMaze --PPO --inner_dones --max_grad_norm 0.1 --num_minibatches 4 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 5000000 --wandb_postfix "inner_done=true, gradnorm=0.1, minibatch=1024, batch=4096"

# 300 arch, batch=4096, mini_batch=1024
#python main.py --env_name AntMaze --PPO --max_grad_norm 0.01 --num_minibatches 4 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 1500000 --wandb_postfix "gradnorm=0.01, minibatch=1024, batch=4096"
#python main.py --env_name AntMaze --PPO --inner_dones --max_grad_norm 0.01 --num_minibatches 4 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --max_timesteps 1500000 --wandb_postfix "inner_done=true, gradnorm=0.01, minibatch=1024, batch=4096"



# 300 arch, batch=4096
#python main.py --env_name AntMaze --PPO --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --ctrl_discount 0.99 --max_timesteps 1500000

# 300 arch, batch=4096, mini_batch=1024
#python main.py --env_name AntMaze --PPO --num_minibatches 4 --ctrl_batch_size 4096 --ctrl_buffer_size 4096 --ctrl_discount 0.99 --max_timesteps 1500000

# inner dones(done when subgoal is meet)
#python main.py --env_name AntMaze --PPO --inner_dones --ctrl_discount 0.99 --ctrl_batch_size 2048 --ctrl_buffer_size 2048 --max_timesteps 1500000

# hidden dim 128
#python main.py --env_name AntMaze --PPO --hidden_dim_ppo 128 --ctrl_discount 0.99 --ctrl_batch_size 2048 --ctrl_buffer_size 2048 --max_timesteps 1500000





#python main.py --env_name AntMaze --PPO

#python main.py --env_name AntMaze --PPO --max_grad_norm 0.01
