if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --modelfree_safety \
               --env_name SafeAntMazeW \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 1000 \
               --random_start_pose \
               --world_model \
               --cost_memmory \
               --cost_model \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 800 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

