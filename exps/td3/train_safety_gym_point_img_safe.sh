if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --seed $seed \
               --manager_algo none \
               --world_model \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \
               --cost_model \
               --cost_model_batch_size 512 \
               --controller_algo td3_img_safe \
               --controller_safety_coef 0.001 \
               --img_horizon 10 \
               --validation_without_image --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb \
               --cm_pretrain \
               --wm_pretrain \

