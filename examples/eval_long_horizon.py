#!/usr/bin/env python3
"""Evaluate a saved long-horizon policy on SafeAntMazeC/W / SafePusher.

Reports per-trajectory success, final cost, final reward for given env seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

import omnisafe

# Register SafeAntMazeC/W/Pusher envs with OmniSafe.
import custom_train  # noqa: F401


ENV_DIR_GLOBS = {
    "c": "FOCOPS-{SafeAntMazeC-Rand}",
    "w": "FOCOPS-{SafeAntMazeW-Rand}",
    "pusher": "FOCOPS-{SafePusher-Rand}",
}

ENV_LABELS = {
    "c": "SafeAntMazeC",
    "w": "SafeAntMazeW",
    "pusher": "SafePusher",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(ENV_DIR_GLOBS.keys()))
    p.add_argument(
        "--log-dir",
        default=None,
        help="Path to a run dir containing torch_save/ and config.json. "
        "Default: latest FOCOPS run for --env under ./logs",
    )
    p.add_argument(
        "--algo",
        default="FOCOPS",
        help="Used only when auto-discovering --log-dir (default FOCOPS).",
    )
    p.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint file name (e.g. epoch-133.pt) or 'latest'.",
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of env seeds to evaluate: 0..num_seeds-1 (default: 5).",
    )
    p.add_argument(
        "--env-seeds",
        default=None,
        help="Optional explicit comma-separated seeds. Overrides --num-seeds if set.",
    )
    p.add_argument(
        "--episodes-per-seed",
        type=int,
        default=1,
        help="Episodes to run for each env seed (default: 1).",
    )
    p.add_argument("--device", default="cpu", help="Device for actor (cpu recommended).")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to write per-trajectory CSV.",
    )
    return p.parse_args()


def _find_attr(root: Any, name: str) -> Any | None:
    cur = root
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, name):
            return cur
        cur = getattr(cur, "_env", None)
    return None


def _parse_seeds(s: str) -> list[int]:
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [int(x) for x in parts]


def _epoch_num(name: str) -> int:
    m = re.search(r"epoch-(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1


def resolve_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir:
        path = Path(args.log_dir).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"log-dir not found: {path}")
        return path

    logs_root = Path("logs")
    # Prefer exact algo+env folder, then any matching env folder.
    candidates = [
        logs_root / f"{args.algo}-{{SafeAntMazeC-Rand}}",
        logs_root / f"{args.algo}-{{SafeAntMazeW-Rand}}",
        logs_root / f"{args.algo}-{{SafePusher-Rand}}",
    ]
    env_map = {
        "c": f"{args.algo}-{{SafeAntMazeC-Rand}}",
        "w": f"{args.algo}-{{SafeAntMazeW-Rand}}",
        "pusher": f"{args.algo}-{{SafePusher-Rand}}",
    }
    base = logs_root / env_map[args.env]
    if not base.exists():
        # fallback: glob
        matches = sorted(logs_root.glob(f"{args.algo}-*"))
        raise SystemExit(
            f"No log dir at {base}. Existing under logs/: {[m.name for m in matches]}"
        )
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed-")])
    if not runs:
        raise SystemExit(f"No seed-* runs under {base}")
    return runs[-1]


def resolve_checkpoint(log_dir: Path, checkpoint: str) -> str:
    save_dir = log_dir / "torch_save"
    if not save_dir.exists():
        raise SystemExit(f"No torch_save/ in {log_dir}")
    pts = [p.name for p in save_dir.iterdir() if p.suffix == ".pt"]
    if not pts:
        raise SystemExit(f"No .pt checkpoints in {save_dir}")
    if checkpoint == "latest":
        return sorted(pts, key=_epoch_num)[-1]
    if checkpoint not in pts:
        raise SystemExit(f"Checkpoint {checkpoint} not in {pts[-5:]} ...")
    return checkpoint


def run_episode(
    evaluator: omnisafe.Evaluator,
    env_seed: int,
) -> dict[str, float]:
    assert evaluator._env is not None
    assert evaluator._actor is not None

    base = _find_attr(evaluator._env, "activate_eval")
    if base is None:
        raise RuntimeError("Could not find activate_eval on wrapped env")
    base.activate_eval(True)

    obs, _ = evaluator._env.reset(seed=env_seed)
    ep_ret, ep_cost, length = 0.0, 0.0, 0
    done = False
    while not done:
        with torch.no_grad():
            act = evaluator._actor.predict(
                obs.reshape(-1, obs.shape[-1]),
                deterministic=False,
            ).reshape(-1)
        obs, rew, cost, terminated, truncated, _ = evaluator._env.step(act)
        ep_ret += float(rew.item())
        ep_cost += float(cost.item())
        length += 1
        done = bool(terminated or truncated)

    successes = list(getattr(base, "successes", []))
    if successes:
        # successes may be torch tensors
        last = successes[-1]
        success = float(last.item() if hasattr(last, "item") else last)
    else:
        success = 0.0

    base.activate_eval(False)
    return {
        "env_seed": float(env_seed),
        "success": success,
        "cost": ep_cost,
        "reward": ep_ret,
        "length": float(length),
    }


def main() -> None:
    args = parse_args()
    if args.env_seeds:
        seeds = _parse_seeds(args.env_seeds)
    else:
        if args.num_seeds < 1:
            raise SystemExit("--num-seeds must be >= 1")
        seeds = list(range(args.num_seeds))
    if not seeds:
        raise SystemExit("Empty seed list")

    log_dir = resolve_log_dir(args)
    ckpt = resolve_checkpoint(log_dir, args.checkpoint)
    label = ENV_LABELS[args.env]

    print(f"[eval] env={label} log_dir={log_dir}")
    print(f"[eval] checkpoint={ckpt}")
    print(f"[eval] env_seeds={seeds} episodes_per_seed={args.episodes_per_seed}")

    evaluator = omnisafe.Evaluator()
    evaluator.load_saved(save_dir=str(log_dir), model_name=ckpt)
    if evaluator._actor is not None:
        evaluator._actor.to(args.device)

    rows: list[dict[str, float]] = []
    traj_id = 0
    for seed in seeds:
        for ep in range(args.episodes_per_seed):
            # Re-seed distinctly if multiple eps per seed
            ep_seed = seed + 1000 * ep
            result = run_episode(evaluator, env_seed=ep_seed if args.episodes_per_seed > 1 else seed)
            result["traj_id"] = float(traj_id)
            result["requested_seed"] = float(seed)
            rows.append(result)
            print(
                f"traj={traj_id:02d} seed={seed} "
                f"success={result['success']:.0f} "
                f"cost={result['cost']:.4f} "
                f"reward={result['reward']:.4f} "
                f"len={int(result['length'])}"
            )
            traj_id += 1

    succ = np.array([r["success"] for r in rows], dtype=np.float64)
    costs = np.array([r["cost"] for r in rows], dtype=np.float64)
    rews = np.array([r["reward"] for r in rows], dtype=np.float64)

    print("#" * 60)
    print(f"summary env={label} n={len(rows)} ckpt={ckpt}")
    print(f"success_rate mean={succ.mean():.4f} std={succ.std():.4f}")
    print(f"final_cost    mean={costs.mean():.4f} std={costs.std():.4f}")
    print(f"final_reward  mean={rews.mean():.4f} std={rews.std():.4f}")
    print("#" * 60)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = str(log_dir / f"eval_{label}_{ckpt.replace('.pt','')}_seeds.csv")
    out_path = Path(out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["traj_id", "requested_seed", "env_seed", "success", "cost", "reward", "length"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {out_path}")

    summary = {
        "env": label,
        "log_dir": str(log_dir),
        "checkpoint": ckpt,
        "env_seeds": seeds,
        "success_rate_mean": float(succ.mean()),
        "cost_mean": float(costs.mean()),
        "reward_mean": float(rews.mean()),
        "trajectories": rows,
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    if evaluator._env is not None:
        evaluator._env.close()


if __name__ == "__main__":
    main()
