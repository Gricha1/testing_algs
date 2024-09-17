if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --modelfree_safety \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --cost_model \
               --cost_memmory \
               --man_rew_scale 0.1 --goal_loss_coeff 20 \
               --coef_safety_modelfree 800 \
               --max_timesteps 4000000 \
               --wandb_postfix ""
