# Docker installation
```
cd docker
sh build.sh
```

# Start docker
```
cd docker
sh start.sh
```

# Run exps
```
sh run_exps.sh
```


# Train AntMaze
python main.py --env_name AntMaze


# Tensorboard logging
tensorboard --logdir logs --bind_all