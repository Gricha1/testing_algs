if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --env_name SafeAntMazeC \
               --seed $seed \
               --train_only_td3 \
               --world_model \
               --cost_model \
               --cm_pretrain \
               --cost_memmory \
               --controller_imagination_safety_loss \
               --controller_safety_coef 6 \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               --self_td3_reward \
               --random_start_pose

