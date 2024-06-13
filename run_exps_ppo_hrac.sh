# HRAC + PPO
python main.py --PPO --train_manager_freq 20 --ppo_gamma 0.99 --man_buffer_size 40000 --max_timesteps 40000000 --wandb_postfix ", testing_safety_subgoal, safety_loss_coef=2000"