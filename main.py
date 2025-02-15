import argparse

from hrac.train import run_hrac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # validation 
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--validation_without_image", action="store_true", default=False)
    parser.add_argument("--visulazied_episode", default=0, type=int)
    parser.add_argument("--test_train_dataset", action="store_true", default=False)
    parser.add_argument("--validate_img_states", action="store_true", default=False)
    
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--loaded_exp_num", default=0, type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--save_models", default=True, type=bool)
    parser.add_argument("--no_correction", default=True, action="store_true") # default=False
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--sparce_reward", action="store_true")

    # environment
    parser.add_argument("--max_timesteps", default=5e6, type=float)
    parser.add_argument("--eval_freq", default=100_000, type=float) # 300_000
    parser.add_argument("--algo", default="hrac", type=str) # ites_hrac, ites_higl, hrac, higl
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--domain_name", type=str, default="SafetyMaze", help="Name of the domain")
    ## safety ant maze
    parser.add_argument("--random_start_pose", action="store_true", default=False)
    parser.add_argument("--env_name", default="SafeAntMazeC", type=str)
    ## safety gym
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--pseudo_lidar", action="store_true", default=False)
    ## safety bullet
    parser.add_argument("--bullet_env_tan_cost", action="store_true", default=False)

    # Adjacency Network Parameters    
    parser.add_argument("--a_net_discretization_koef", default=1.0, type=float) # 50_000
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

    # HIGL
    parser.add_argument("--landmark_loss_coeff", default=20., type=float)
    parser.add_argument("--delta", type=float, default=2)
    parser.add_argument("--adj_factor", default=0.5, type=float)

    # HIGL: Planner, Coverage
    #parser.add_argument("--landmark_sampling", type=str, choices=["fps", "none"])
    parser.add_argument("--landmark_sampling", default="fps", type=str)
    parser.add_argument('--clip_v', type=float, default=-38., help="clip bound for the planner")
    parser.add_argument("--n_landmark_coverage", type=int, default=20)
    parser.add_argument("--initial_sample", type=int, default=1000)
    parser.add_argument("--goal_thr", type=float, default=-10.)
    parser.add_argument("--planner_start_step", type=int, default=60000)

    # HIGL: Novelty
    parser.add_argument("--novelty_algo", type=str, default="none", choices=["rnd", "none"])
    parser.add_argument("--use_novelty_landmark", action="store_true")
    parser.add_argument("--close_thr", type=float, default=0.2)
    parser.add_argument("--n_landmark_novelty", type=int, default=20)
    parser.add_argument("--rnd_output_dim", type=int, default=128)
    parser.add_argument("--rnd_lr", type=float, default=1e-3)
    parser.add_argument("--rnd_batch_size", default=128, type=int)
    parser.add_argument("--use_ag_as_input", action="store_true")

    # Ablation
    parser.add_argument("--no_pseudo_landmark", action="store_true")
    parser.add_argument("--discard_by_anet", action="store_true")
    parser.add_argument("--automatic_delta_pseudo", action="store_true")

    # Manager Parameters
    parser.add_argument("--manager_algo", default="td3_adj", type=str) # ["td3_adj", "td3_adj_safe_cls", "td3_adj_safe_cls_high_lag", "td3_adj_safe_cls_low_lag"]
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
    parser.add_argument("--a_net_size", default=1500, type=int) # 10
    parser.add_argument("--man_hidden_size", default=300, type=int)

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
    parser.add_argument("--ctrl_hidden_size", default=300, type=int)

    # Safety Subgoal Parameters
    parser.add_argument("--noise_man_training", default=False, action="store_true")
    parser.add_argument("--man_safe_noise_sigma", default=1., type=float)
    parser.add_argument("--img_horizon", default=20, type=int)    
    parser.add_argument("--coef_safety_modelbased", default=0.0, type=float)    
    parser.add_argument("--coef_safety_modelfree", default=0.0, type=float)
    ## Cost Model Parameters
    parser.add_argument("--cost_model", action='store_true', default=False)
    parser.add_argument("--regression_cost_model", action='store_true', default=False)
    parser.add_argument("--cm_pretrain", action='store_true', default=False) # to avoid wm explosion in beggining
    parser.add_argument("--cost_model_batch_size", default=128, type=int)
    parser.add_argument("--cost_model_buffer_size", default=1e6, type=int)
    parser.add_argument("--cm_lr", default=1e-3, type=float)
    parser.add_argument("--cm_frame_stack_num", default=1, type=int)
    parser.add_argument("--safe_model_loss_coef", default=1., type=float)
    parser.add_argument("--cm_hidden_size", default=300, type=int)
    parser.add_argument("--cost_model_trajectory_buffer", action='store_true', default=False) # to avoid wm explosion in beggining

    # Safety Controller Parameters
    parser.add_argument("--noise_ctr_training", default=False, action="store_true")
    parser.add_argument("--ctr_safe_noise_sigma", default=1., type=float)
    parser.add_argument("--controller_curriculumn", action='store_true', default=False)
    parser.add_argument("--controller_curriculum_start_step", default=600_000, type=int)
    parser.add_argument("--controller_curriculum_safety_coef", default=4000., type=float)
    parser.add_argument("--controller_cumul_img_safety", action='store_true', default=False)
    parser.add_argument("--controller_safety_coef", default=4000., type=float)
    parser.add_argument("--cost_budget", default=25, type=float)
    parser.add_argument("--ctrl_pid_kp", default=1e-6, type=float)
    parser.add_argument("--ctrl_pid_ki", default=1e-7, type=float)
    parser.add_argument("--ctrl_pid_kd", default=1e-7, type=float)
    parser.add_argument("--ctrl_pid_d_delay", default=10, type=int)
    parser.add_argument("--ctrl_pid_delta_p_ema_alpha", default=0.95, type=float)
    parser.add_argument("--ctrl_pid_delta_d_ema_alpha", default=0.95, type=float)
    parser.add_argument("--ctrl_lagrangian_multiplier_init", default=0., type=float)
    ## WorldModel Parameters
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
    parser.add_argument("--train_policy_noise", default=0.2, type=float)
    parser.add_argument("--train_noise_clip", default=0.5, type=float)

    # logger
    parser.add_argument("--not_use_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_postfix", default="", type=str)
    parser.add_argument("--tensorboard_descript", default="", type=str)

    # Run the algorithm
    args = parser.parse_args()

    if args.manager_algo == "td3_adj_safe_cls_high_lag":
        assert not "lag" in args.controller_algo

    assert args.manager_algo in ["td3_adj", 
                                 "td3_adj_safe_cls", 
                                 "td3_adj_safe_cls_high_lag", 
                                 "td3_adj_safe_cls_low_lag"]
    assert args.algo in ["ites_hrac", "ites_higl", "hrac", "higl"]
    assert args.controller_algo in ["td3_img_safe_c_cost", 
                                    "td3_img_safe_lag", 
                                    "td3_img_safe", 
                                    "td3_lag", 
                                    "td3", 
                                    "sac_lag", 
                                    "sac"]

    if "img_safe" in args.controller_algo:
        assert args.world_model and args.cost_model
    if "td3_img_safe_lag" == args.controller_algo:
        assert args.controller_cumul_img_safety
    if "img_safe" in args.controller_algo:
        args.img_horizon <= args.manager_propose_freq

    if args.env_name in ["AntGather", "AntMazeSparse"]:
        args.man_rew_scale = 1.0
        if args.env_name == "AntGather":
            args.inner_dones = True

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    run_hrac(args)
