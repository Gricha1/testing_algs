# SafeDreamer (ITES table baselines)

Embedded from `~/safe_rl_nlp/safe_dynalang` branch `safedyna_safeant_maze`.

## Train (SafeAntMaze C / W / Pusher)

```bash
bash exps/SafeDreamer_safeant_maze_cshape.sh
bash exps/SafeDreamer_safeant_maze_wshape.sh
bash exps/SafeDreamer_safeant_maze_pusher.sh
```

## Eval / validation

```bash
# host launchers (prefer these)
bash exps/launch_validation_SafeDreamer_cshape.sh
bash exps/launch_validation_SafeDreamer_wshape.sh
bash exps/launch_validation_SafeDreamer_pusher.sh

# or all
bash exps/launch_validation_SafeDreamer_all.sh
```

In-container task helpers: `exps/validation_SafeDreamer_{cshape,wshape,pusher}.sh`.

Method: `osrp_lag`. Checkpoints typically under `logdir_osrp_*` (not committed).
