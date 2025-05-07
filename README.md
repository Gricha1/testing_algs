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

# Train ITES 
if not in exps/ites do
```
cd exps/ites
```

## Long Horizon Benchmarks

### SafeAneMazeCshape
```
sh train_safe_ant_maze_c.sh
```

### SafeAneMazeWshape
```
sh train_safe_ant_maze_w.sh
```
## SafePusher
```
sh train_safe_pusher.sh
```

## Short Horizon Benchmark

## PointGoal1
```
sh train_safety_gym_point.sh
```

## CarGoal1
```
sh train_safety_gym_car.sh
```

## PointGoal sparce
```
sh train_safety_gym_point_sparce_hrac_safety.sh
```


# Tensorboard logging
tensorboard --logdir logs --bind_all