# Flat OmniSafe baselines (ITES table)

Algorithms (no ITES / no SafeDreamer): **FOCOPS**, **CUP**, **TD3Lag**, **TD3PID**, **PPOLag**.

Environments: SafeAntMazeC (`c`, d=25), SafeAntMazeW (`w`, d=200), SafePusher (`pusher`, d=25).

## Train

From repo root inside the OmniSafe container:

```bash
bash exps/train_focops_safe_ant_maze_c.sh [seed] [device]
bash exps/train_cup_safe_ant_maze_w.sh
bash exps/train_td3pid_safe_pusher.sh
bash exps/train_ppolag_safe_ant_maze_c.sh
bash exps/train_td3lag_safe_ant_maze_w.sh
```

Python entrypoint: `examples/train_long_horizon.py`.

## Eval (table protocol: 5 env seeds)

```bash
bash exps/eval_focops_safe_ant_maze_c.sh          # 5 seeds
bash exps/eval_td3pid_safe_pusher.sh 5 /path/to/run latest
bash exps/eval_all_table_baselines.sh examples/eval_results_table
```

Python entrypoint: `examples/eval_long_horizon.py`.
