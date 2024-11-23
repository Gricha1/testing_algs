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
    ## safety ant maze
    parser.add_argument("--random_start_pose", action="store_true", default=False)
    parser.add_argument("--algo", default="hrac", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval_freq", default=100_000, type=float) # 300_000
    parser.add_argument("--max_timesteps", default=5e6, type=float)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--env_name", default="SafeAntMazeC", type=str)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--loaded_exp_num", default=0, type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--no_correction", default=True, action="store_true") # default=False
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--sparce_reward", action="store_true")

    ## safety gym
    parser.add_argument("--image_size", type=int, default=2)
    parser.add_argument("--vector_env", default=False, action="store_true")
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--domain_name", type=str, default="SafetyMaze", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--goal_conditioned", action="store_true", default=False)
    parser.add_argument("--pseudo_lidar", action="store_true", default=False)

    # Adjacency Network Parameters    
    parser.add_argument("--a_net_new_discretization_safety_gym", default=False, action="store_true")
    parser.add_argument("--a_net_discretization_koef", default=3.0, type=float) # 50_000
    parser.add_argument("--clip_a_net_xy", default=False, action="store_true")
    parser.add_argument("--load_adj_net", default=False, action="store_true")
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
    parser.add_argument("--subgoal_lower_x", default=5.0, type=float)
    parser.add_argument("--subgoal_lower_y", default=5.0, type=float)
    parser.add_argument("--subgoal_grad_clip", default=0, type=float)
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

    # Controller Parameters
    parser.add_argument("--sac_alpha", default=0.2, type=float)
    parser.add_argument("--controller_algo", default="td3", type=str)
    parser.add_argument("--train_only_td3", action='store_true', default=False)
    parser.add_argument("--self_td3_reward", action='store_true', default=False)
    parser.add_argument("--controller_grad_clip", default=0, type=float)
    parser.add_argument("--ctrl_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--ctrl_batch_size", default=128, type=int)
    parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
    parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
    parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
    parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)
    parser.add_argument("--ctrl_discount", default=0.95, type=float)

    # Safety Subgoal Parameters
    parser.add_argument("--modelfree_safety", action='store_true', default=False)
    parser.add_argument("--img_horizon", default=20, type=int)    
    parser.add_argument("--coef_safety_modelbased", default=0.0, type=float)    
    parser.add_argument("--coef_safety_modelfree", default=0.0, type=float)
    ## Cost Model Parameters
    parser.add_argument("--cost_model", action='store_true', default=False)
    parser.add_argument("--cm_pretrain", action='store_true', default=False) # to avoid wm explosion in beggining
    parser.add_argument("--cost_oracle", action='store_true', default=False)
    parser.add_argument("--cost_model_batch_size", default=128, type=int)
    parser.add_argument("--cost_model_buffer_size", default=1e6, type=int)
    parser.add_argument("--cm_lr", default=1e-3, type=float)
    parser.add_argument("--cm_frame_stack_num", default=1, type=int)
    parser.add_argument("--safe_model_loss_coef", default=1., type=float)

    # Safety Controller Parameters
    parser.add_argument("--controller_curriculumn", action='store_true', default=False)
    parser.add_argument("--controller_curriculum_start_step", default=600_000, type=int)
    parser.add_argument("--controller_curriculum_safety_coef", default=4000., type=float)
    parser.add_argument("--controller_cumul_img_safety", action='store_true', default=False)
    parser.add_argument("--controller_safety_coef", default=4000., type=float)
    parser.add_argument("--controller_imagination_safety_loss", action='store_true', default=False)
    parser.add_argument("--use_safe_threshold", action='store_true', default=False)
    parser.add_argument("--cost_budget", default=25, type=float)
    parser.add_argument("--controller_use_lagrange", action='store_true', default=False)
    parser.add_argument("--ctrl_pid_kp", default=1e-6, type=float)
    parser.add_argument("--ctrl_pid_ki", default=1e-7, type=float)
    parser.add_argument("--ctrl_pid_kd", default=1e-7, type=float)
    parser.add_argument("--ctrl_pid_d_delay", default=10, type=int)
    parser.add_argument("--ctrl_pid_delta_p_ema_alpha", default=0.95, type=float)
    parser.add_argument("--ctrl_pid_delta_d_ema_alpha", default=0.95, type=float)
    parser.add_argument("--ctrl_lagrangian_multiplier_init", default=0., type=float)
    ## WorldModel Parameters
    parser.add_argument("--cm_train_on_dataset", action='store_true', default=False) # to avoid wm explosion in beggining
    parser.add_argument("--wm_pretrain", action='store_true', default=False) # to avoid wm explosion in beggining
    parser.add_argument("--wm_pretrain_epoches", default=20, type=int) # to avoid wm explosion in beggining
    parser.add_argument("--wm_n_initial_exploration_steps", default=10_000, type=int)
    parser.add_argument("--wm_batch_size", default=256, type=int)
    parser.add_argument("--wm_train_freq", default=20, type=int)
    parser.add_argument("--cost_memmory", action='store_true', default=False)
    parser.add_argument("--world_model", action='store_true', default=False)
    parser.add_argument("--wm_learning_rate", default=1e-3, type=float)
    parser.add_argument("--wm_buffer_size", default=1e6, type=int)
    parser.add_argument("--num_networks", default=8, type=int)
    parser.add_argument("--num_elites", default=6, type=int)
    parser.add_argument("--pred_hidden_size", default=200, type=int)
    parser.add_argument("--use_decay", default=True, type=bool)
    parser.add_argument("--testing_mean_wm", action='store_true', default=False)

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

    assert args.controller_algo in ["td3_lag", "td3", "sac_lag", "sac"]
    if args.controller_algo=="td3_lag":
        assert args.controller_use_lagrange
        assert not args.controller_imagination_safety_loss

    if args.modelfree_safety:
        assert args.cost_model
    if args.controller_imagination_safety_loss:
        assert args.world_model and args.cost_model

    if args.controller_imagination_safety_loss and args.controller_use_lagrange:
        assert args.controller_cumul_img_safety
    if args.use_safe_threshold:
        assert not args.controller_use_lagrange
        assert args.controller_cumul_img_safety
    assert not args.controller_imagination_safety_loss or (args.controller_imagination_safety_loss and args.img_horizon <= args.manager_propose_freq)
    assert not args.cost_model or \
        ( (args.cost_model and args.domain_name == "Safexp") or \
          (args.cost_model and args.domain_name != "Safexp" and args.world_model)
        )

    if args.env_name in ["AntGather", "AntMazeSparse"]:
        args.man_rew_scale = 1.0
        if args.env_name == "AntGather":
            args.inner_dones = True

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    run_hrac(args)
