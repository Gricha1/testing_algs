# Cursor task: long-horizon baseline train scripts (omnisafe)

## Repo / machine
- Host: **ml4**
- Path: `/home/ggorbov/articles/omnisafe/testing_algs`
- Branch: `omnisafe`
- Remote: https://github.com/Gricha1/testing_algs/tree/omnisafe

## Goal
Create **separate bash scripts** to launch training of:
- algorithms: **FOCOPS**, **CUP**, **TD3Lag**
- envs: **SafeAntMazeC-Rand**, **SafeAntMazeW-Rand**, **SafePusher-Rand**

Do **not** ask the user to hand-edit `custom_train.py` for each run.

## Implementation requirements
1. Add `examples/train_long_horizon.py` CLI with:
   - `--algo {FOCOPS,CUP,TD3Lag,PPOLag}`
   - `--env {c,w,pusher}` (or full env id)
   - `--seed`, `--device`, `--total-steps`, `--cost-limit`, `--log-dir`
   - reuse env registration from `examples/custom_train.py`
   - defaults aligned with existing `custom_train.train()`:
     - `total_steps=4020000`, on-policy `steps_per_epoch=30000`, `cost_limit=25`
     - for `TD3Lag` use off-policy-friendly `steps_per_epoch=2000` unless overridden
2. Add 9 scripts under `exps/`:
   - `train_focops_safe_ant_maze_c.sh`
   - `train_focops_safe_ant_maze_w.sh`
   - `train_focops_safe_pusher.sh`
   - `train_cup_safe_ant_maze_c.sh`
   - `train_cup_safe_ant_maze_w.sh`
   - `train_cup_safe_pusher.sh`
   - `train_td3lag_safe_ant_maze_c.sh`
   - `train_td3lag_safe_ant_maze_w.sh`
   - `train_td3lag_safe_pusher.sh`
3. Each script accepts optional `[seed] [device]` args and calls the CLI.
4. Fix bug in `custom_train.py`: `SafeAntMazew` → `SafeAntMazeW` in `max_episode_steps`.
5. Fix docker build python conflict in `docker/dockerfile`:
   - current `FROM continuumio/miniconda3` + `conda install python=3.8.5` fails
   - pin older base (e.g. `continuumio/miniconda3:4.12.0`) and install `python=3.8.13`

## Run example (after docker rebuild)
```bash
cd /home/ggorbov/articles/omnisafe/testing_algs/docker
sh build.sh
sh start.sh 0
# inside container:
cd /usr/home/workspace && pip install -e .
bash exps/train_focops_safe_ant_maze_c.sh 224424 cuda:0
```

## Done when
- 9 bash scripts exist and are executable
- CLI trains by changing only algo/env args
- `docker build` no longer fails on python 3.8.5 conflict
