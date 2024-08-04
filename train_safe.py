import argparse
import os
from datetime import datetime
import random
import git
from matplotlib.pyplot import get

import torch

from slac.algo import LatentPolicySafetyCriticSlac, SafetyCriticSlacAlgorithm
from envs.safegym import make_safety
from envs.slac_wrapper import make_safe_ant_maze
from slac.trainer import Trainer
import json
from configuration import get_default_config

def get_git_short_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    sha_first_7_chars = sha[:7]
    return sha_first_7_chars

def main(args):
    config = get_default_config()
    config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["seed"] = args.seed
    config["num_steps"] = args.num_steps
    config["use_goalobs"] = args.use_goalobs

    if args.domain_name == "SafeAntMaze":
        env = make_safe_ant_maze(config["action_repeat"], config["seed"], False)
        env_test = make_safe_ant_maze(config["action_repeat"], config["seed"], True)
    else:
        env = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', 
                            image_size=config["image_size"], 
                            use_pixels=not args.vector_env, 
                            action_repeat=config["action_repeat"],
                            goal_conditioned=config["use_goalobs"])
        env_test = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', 
                            image_size=config["image_size"], 
                            use_pixels=not args.vector_env, 
                            action_repeat=config["action_repeat"],
                            goal_conditioned=config["use_goalobs"],
                        eval=True)
    
    short_hash = get_git_short_hash()
    log_dir = os.path.join(
        "logs",
        f"{short_hash}",
        f"{config['domain_name']}-{config['task_name']}",
        f'slac-seed{config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
        args.exp_num,
    )
    config["log_dir"] = log_dir

    algo = LatentPolicySafetyCriticSlac(
        num_sequences=config["num_sequences"],
        gamma_c=config["gamma_c"],
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=config["action_repeat"],
        max_episode_steps=env._max_episode_steps,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=config["seed"],
        buffer_size=config["buffer_size"],
        feature_dim=config["feature_dim"],
        z2_dim=config["z2_dim"],
        hidden_units=config["hidden_units"],
        batch_size_latent=config["batch_size_latent"],
        batch_size_sac=config["batch_size_sac"],
        lr_sac=config["lr_sac"],
        lr_latent=config["lr_latent"],
        start_alpha=config["start_alpha"],
        start_lagrange=config["start_lagrange"],
        grad_clip_norm=config["grad_clip_norm"],
        tau=config["tau"],
        image_noise=config["image_noise"],
        pixel_input=not args.vector_env,
    )
    trainer = Trainer(
        num_sequences=config["num_sequences"],
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=config["seed"],
        num_steps=config["num_steps"],
        initial_learning_steps=config["initial_learning_steps"],
        initial_collection_steps=config["initial_collection_steps"],
        collect_with_policy=config["collect_with_policy"],
        eval_interval=config["eval_interval"],
        num_eval_episodes=config["num_eval_episodes"],
        action_repeat=config["action_repeat"],
        train_steps_per_iter=config["train_steps_per_iter"],
        env_steps_per_train_step=config["env_steps_per_train_step"],
        use_wandb=args.use_wandb,
        config=config,
        pixel_input=not args.vector_env,
    )
    trainer.writer.add_text("config", json.dumps(config), 0)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_num", type=str, default="0")
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6, help="Number of training steps")
    parser.add_argument("--vector_env", default=False, action="store_true")
    parser.add_argument("--domain_name", type=str, default="Safexp", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Train using GPU with CUDA")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_goalobs", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
