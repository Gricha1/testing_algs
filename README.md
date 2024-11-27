# Docker installation
## Build docker
```
cd docker
sh build.sh
```
## Start docker and enter in it
```
cd docker
sh start.sh
```

# Train ITES on SafeAntMaze environments

## Cshape
```
cd exps/hrac_safety
sh train_safe_ant_maze_c_hrac_safety.sh
```

## Wshape
```
cd exps/hrac_safety
sh train_safe_ant_maze_w_hrac_safety.sh
```

# Train ITES on SafetyGym environments

## PointGoal1
```
cd exps/hrac_safety
sh train_safety_gym_point_hrac_safety.sh
```

## CarGoal1
```
cd exps/hrac_safety
sh train_safety_gym_car_hrac_safety.sh
```

## PointGoal sparce
```
cd exps/hrac_safety
sh train_safety_gym_point_sparce_hrac_safety.sh
```

# Tensorboard logging
tensorboard --logdir logs --bind_all