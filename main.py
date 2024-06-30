import argparse

from hrac.train import run_hrac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # validation 
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--validation_without_image", action="store_true", default=False)
    parser.add_argument("--visulazied_episode", default=0, type=int)
    parser.add_argument("--test_train_dataset", action="store_true", default=False)
    
    # environment
    parser.add_argument("--random_start_pose", action="store_true", default=False)
    parser.add_argument("--algo", default="hrac", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval_freq", default=100_000, type=float) # 300_000
    parser.add_argument("--max_timesteps", default=5e6, type=float)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--env_name", default="SafeAntMaze", type=str)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--loaded_exp_num", default=0, type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--no_correction", default=True, action="store_true") # default=False
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--load_adj_net", default=False, action="store_true")

    # Adjacency Network Parameters
    parser.add_argument("--adj_loss_coef", default=1., type=float)
    parser.add_argument("--gid", default=0, type=int)
    parser.add_argument("--traj_buffer_size", default=50_000, type=int) # 50_000
    parser.add_argument("--lr_r", default=2e-4, type=float)
    parser.add_argument("--r_margin_pos", default=1.0, type=float)
    parser.add_argument("--r_margin_neg", default=1.2, type=float)
    parser.add_argument("--r_training_epochs", default=25, type=int)
    parser.add_argument("--r_batch_size", default=64, type=int)
    parser.add_argument("--r_hidden_dim", default=128, type=int)
    parser.add_argument("--r_embedding_dim", default=32, type=int)

    # Manager Parameters
    parser.add_argument("--absolute_goal", default=False, action="store_true")
    parser.add_argument("--goal_loss_coeff", default=20., type=float)
    parser.add_argument("--manager_propose_freq", default=20, type=int) # 10
    parser.add_argument("--train_manager_freq", default=10, type=int) # 10
    parser.add_argument("--man_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--man_batch_size", default=128, type=int)
    parser.add_argument("--man_buffer_size", default=2e5, type=int)
    parser.add_argument("--man_rew_scale", default=0.1, type=float)
    parser.add_argument("--man_act_lr", default=1e-4, type=float)
    parser.add_argument("--man_crit_lr", default=1e-3, type=float)
    parser.add_argument("--candidate_goals", default=10, type=int)
    parser.add_argument("--man_discount", default=0.99, type=float)

    # TD3 Controller Parameters
    parser.add_argument("--ctrl_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--ctrl_batch_size", default=128, type=int)
    parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
    parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
    parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
    parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)
    parser.add_argument("--ctrl_discount", default=0.95, type=float)

    # PPO controller Parameters
    parser.add_argument("--PPO", action='store_true', default=False)
    parser.add_argument("--ppo_ctrl_batch_size", default=4096, type=int)
    parser.add_argument("--ppo_ctrl_buffer_size", default=4096, type=int)
    parser.add_argument("--ppo_gae_lambda", default=0.95, type=float)
    parser.add_argument("--ppo_gamma", default=0.95, type=float)
    parser.add_argument("--ppo_num_minibatches", default=32, type=int)
    parser.add_argument("--ppo_clip_coef", default=0.2, type=float)
    parser.add_argument("--ppo_clip_vloss", default=True, type=bool)
    parser.add_argument("--ppo_norm_adv", default=True, type=bool)
    parser.add_argument("--ppo_max_grad_norm", default=0.5, type=float)
    parser.add_argument("--ppo_vf_coef", default=0.5, type=float)
    parser.add_argument("--ppo_ent_coef", default=0.2, type=float)
    parser.add_argument("--ppo_target_kl", default=None, type=float)
    parser.add_argument("--ppo_update_epochs", default=10, type=int)
    parser.add_argument("--ppo_lr", default=1e-4, type=float)
    parser.add_argument("--ppo_hidden_dim", default=300, type=int)
    parser.add_argument("--ppo_weight_decay", default=None, type=float)

    # WorldModel Parameters
    parser.add_argument("--world_model", action='store_true', default=False)
    parser.add_argument("--wm_learning_rate", default=1e-3, type=float)
    parser.add_argument("--wm_buffer_size", default=1e6, type=int)
    parser.add_argument("--wm_train_freq", default=20, type=int) # 20 episodes
    parser.add_argument("--wm_n_initial_exploration_steps", default=10_000, type=int)
    parser.add_argument("--num_networks", default=8, type=int)
    parser.add_argument("--num_elites", default=6, type=int)
    parser.add_argument("--pred_hidden_size", default=200, type=int)
    parser.add_argument("--use_decay", default=True, type=bool)
    parser.add_argument("--testing_mean_wm", action='store_true', default=False)

    # Safety Subgoal Parameters
    parser.add_argument("--modelbased_safety", action='store_true', default=False)
    parser.add_argument("--modelfree_safety", action='store_true', default=False)
    parser.add_argument("--cumul_modelbased_safety", action='store_true', default=False)
    parser.add_argument("--subgoal_grad_clip", default=0, type=float)
    parser.add_argument("--img_horizon", default=20, type=int)
    parser.add_argument("--safety_loss_coef", default=200., type=float)
    parser.add_argument("--coef_safety_modelbased", default=1.0, type=float)    
    parser.add_argument("--coef_safety_modelfree", default=1.0, type=float)

    # Safety model Parameters
    parser.add_argument("--controller_safe_model", action='store_true', default=False)
    parser.add_argument("--safe_model_loss_coef", default=1., type=float)
    parser.add_argument("--train_safe_model", action='store_true', default=False)
    parser.add_argument("--cost_model_batch_size", default=128, type=int)

    # Noise Parameters
    parser.add_argument("--noise_type", default="normal", type=str)
    parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
    parser.add_argument("--man_noise_sigma", default=1., type=float)

    # logger
    parser.add_argument("--not_use_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_postfix", default="", type=str)

    parser.add_argument("--tensorboard_descript", default="", type=str)

    # Run the algorithm
    args = parser.parse_args()

    assert not args.modelbased_safety or (args.world_model and args.modelbased_safety), \
            " to train safety you need world model"
    assert not args.cumul_modelbased_safety or (args.modelbased_safety and args.cumul_modelbased_safety)
    assert args.img_horizon <= args.manager_propose_freq

    if args.modelbased_safety == args.modelfree_safety == True:
        assert (args.coef_safety_modelbased + args.coef_safety_modelfree) == 1.0
    if args.modelbased_safety != args.modelfree_safety:
        assert args.coef_safety_modelbased == 1.0
        assert args.coef_safety_modelfree == 1.0

    # PPO
    args.ppo_minibatch_size = int(args.ppo_ctrl_batch_size // args.ppo_num_minibatches)
    if args.PPO:
        args.ctrl_noise_sigma = None
        assert args.ppo_ctrl_batch_size == args.ppo_ctrl_buffer_size

    if args.env_name in ["AntGather", "AntMazeSparse"]:
        args.man_rew_scale = 1.0
        if args.env_name == "AntGather":
            args.inner_dones = True

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    run_hrac(args)
