#!/bin/bash

generate_next_index() {
    local dir="./data"
    local prefix="test_point_"
    local next_index=1

    if [ -d "$dir" ]; then
        local existing_indices=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -name "${prefix}*" | sort -V | xargs -n 1 basename | sed "s/${prefix}//")
        
        for i in $(seq 1 9999); do
            if ! echo "$existing_indices" | grep -q "^$i$"; then
                next_index=$i
                break
            fi
        done
    fi

    echo "${prefix}${next_index}"
}

# Вызов функции для получения следующего индекса
weights_folder=$(generate_next_index)
echo "Weights folder: $weights_folder"

python3 mbppo_lagrangian.py --target_kl 0 --img_rollout_H 20 --hid 300 --cost_limit 70 --load --load_wm_from_pth --not_load_ac --loaded_exp_num final_mb_safety --wm_learning_rate 0.0 --random_start_pose --exp_name=$weights_folder --seed=0 --env_name=SafeAntMaze --beta=0.02
#python3 mbppo_lagrangian.py --cost_limit 70 --exp_name=$weights_folder --seed=0 --env_name=Safexp-PointGoal2-v0 --beta=0.02

