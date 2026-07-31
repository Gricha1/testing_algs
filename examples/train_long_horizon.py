#!/usr/bin/env python3
"""Train FOCOPS / CUP / TD3Lag / TD3PID / PPOLag on long-horizon ITES envs with Comet ML."""
from __future__ import annotations

import argparse
import os

import omnisafe

# Registers SafeAntMazeC/W/Pusher-* envs with OmniSafe.
import custom_train  # noqa: F401


ENV_IDS = {
    "c": "SafeAntMazeC-Rand",
    "w": "SafeAntMazeW-Rand",
    "pusher": "SafePusher-Rand",
    "SafeAntMazeC-Rand": "SafeAntMazeC-Rand",
    "SafeAntMazeW-Rand": "SafeAntMazeW-Rand",
    "SafePusher-Rand": "SafePusher-Rand",
}

# Comet experiment display name: "FOCOPS SafeAntMazeC"
ENV_COMET_NAMES = {
    "c": "SafeAntMazeC",
    "w": "SafeAntMazeW",
    "pusher": "SafePusher",
    "SafeAntMazeC-Rand": "SafeAntMazeC",
    "SafeAntMazeW-Rand": "SafeAntMazeW",
    "SafePusher-Rand": "SafePusher",
}

ALGOS = ("FOCOPS", "CUP", "TD3Lag", "TD3PID", "PPOLag")

COMET_WORKSPACE = "gregory-gorbov"
COMET_PROJECT = "ites"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=ALGOS)
    p.add_argument("--env", required=True, choices=sorted(ENV_IDS.keys()))
    p.add_argument("--seed", type=int, default=224424)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-steps", type=int, default=4_020_000)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--cost-limit", type=float, default=25.0)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--save-model-freq", type=int, default=1)
    p.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging (off by default).",
    )
    p.add_argument(
        "--use-comet",
        action="store_true",
        default=True,
        help="Enable Comet ML logging (on by default).",
    )
    p.add_argument(
        "--no-comet",
        action="store_true",
        help="Disable Comet ML logging.",
    )
    p.add_argument("--comet-workspace", default=COMET_WORKSPACE)
    p.add_argument("--comet-project", default=COMET_PROJECT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_id = ENV_IDS[args.env]
    env_short = ENV_COMET_NAMES[args.env]
    comet_name = f"{args.algo} {env_short}"
    use_comet = bool(args.use_comet) and (not args.no_comet)

    if use_comet and not os.environ.get("COMET_API_KEY"):
        raise SystemExit(
            "COMET_API_KEY is not set. Export it before training, e.g.\n"
            '  export COMET_API_KEY="..."'
        )

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        # Off-policy defaults are smaller; keep on-policy aligned with custom_train.
        steps_per_epoch = 2000 if args.algo in ("TD3Lag", "TD3PID") else 30000

    custom_cfgs = {
        "seed": args.seed,
        "train_cfgs": {
            "total_steps": args.total_steps,
            "vector_env_nums": 1,
            "parallel": 1,
            "device": args.device,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
        },
        "logger_cfgs": {
            "use_wandb": args.use_wandb,
            "use_tensorboard": True,
            "use_comet": use_comet,
            "comet_workspace": args.comet_workspace,
            "comet_project": args.comet_project,
            "comet_experiment_name": comet_name,
            "log_dir": args.log_dir,
            "save_model_freq": args.save_model_freq,
        },
        "lagrange_cfgs": {
            "cost_limit": args.cost_limit,
        },
    }

    print(
        f"[train_long_horizon] algo={args.algo} env={env_id} "
        f"seed={args.seed} device={args.device} "
        f"wandb={args.use_wandb} comet={use_comet} name={comet_name!r}"
    )
    agent = omnisafe.Agent(args.algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


if __name__ == "__main__":
    main()
