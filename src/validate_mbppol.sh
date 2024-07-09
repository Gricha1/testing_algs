#!/bin/bash


#weights_folder="final_base"
weights_folder=test_point_12
echo "Weights folder: $weights_folder"


#python3 mbppo_lagrangian.py --vizualize_validation --validate --load --exp_name=$weights_folder --seed=0 --env=Safexp-PointGoal2-v0 --beta=0.02
#python3 mbppo_lagrangian.py --vizualize_validation --validate --load --exp_name=$weights_folder --seed=0 --env_name=SafeAntMaze --beta=0.02
python3 mbppo_lagrangian.py --vizualize_validation --validate --exp_name=$weights_folder --seed=0 --env_name=SafeAntMaze --beta=0.02
