# Installation

```
cd docker
sh build.sh
```

# Docker start
```
cd docker
sh build.sh
```


# Train
```
python3 mbppo_lagrangian.py --exp_name=test_point_1 --seed=0 --env=Safexp-PointGoal2-v0 --beta=0.02
```

# Tensorboard logging
tensorboard --logdir data --bind_all