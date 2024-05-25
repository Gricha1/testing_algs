import argparse

from hrac.train import run_hrac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="hrac", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval_freq", default=1_000, type=float) # 300_000
    parser.add_argument("--max_timesteps", default=5e6, type=float)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--env_name", default="AntMaze", type=str)
    parser.add_argument("--load", default=False, type=bool)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--no_correction", default=True, action="store_true") # default=False
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--absolute_goal", default=False, action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--load_adj_net", default=False, action="store_true")

    parser.add_argument("--gid", default=0, type=int)
    parser.add_argument("--traj_buffer_size", default=50_000, type=int) # 50_000
    parser.add_argument("--lr_r", default=2e-4, type=float)
    parser.add_argument("--r_margin_pos", default=1.0, type=float)
    parser.add_argument("--r_margin_neg", default=1.2, type=float)
    parser.add_argument("--r_training_epochs", default=25, type=int)
    parser.add_argument("--r_batch_size", default=64, type=int)
    parser.add_argument("--r_hidden_dim", default=128, type=int)
    parser.add_argument("--r_embedding_dim", default=32, type=int)
    parser.add_argument("--goal_loss_coeff", default=20., type=float)

    parser.add_argument("--manager_propose_freq", default=20, type=int) # 10
    parser.add_argument("--train_manager_freq", default=10, type=int)

    # Manager Parameters
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
    # PPO controller parameters
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

    # Noise Parameters
    parser.add_argument("--noise_type", default="normal", type=str)
    parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
    parser.add_argument("--man_noise_sigma", default=1., type=float)

    # logger
    parser.add_argument("--use_wandb", default=True, type=bool)
    parser.add_argument("--wandb_postfix", default="", type=str)

    # Run the algorithm
    args = parser.parse_args()

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
