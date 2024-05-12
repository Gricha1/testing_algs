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
next_folder=$(generate_next_index)
echo "Следующая папка для экспериментов: $next_folder"


python3 mbppo_lagrangian.py --pred_hidden_size 100 --exp_name=$next_folder --seed=0 --env=Safexp-PointGoal2-v0 --beta=0.02


# Вызов функции для получения следующего индекса
next_folder=$(generate_next_index)
echo "Следующая папка для экспериментов: $next_folder"

python3 mbppo_lagrangian.py --num_networks 7 --num_elites 5 --exp_name=$next_folder --seed=0 --env=Safexp-PointGoal2-v0 --beta=0.02