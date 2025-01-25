if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --env_name SafePusher \
               --domain_name SafetyMaze \
               --modelfree_safety \
               --controller_imagination_safety_loss \
               --validation_without_image --eval_freq 30000 \
               --world_model \
               --cost_memmory \
               --cost_model \
               --man_rew_scale 0.1 \
               --goal_loss_coeff 20.0 \
               --coef_safety_modelfree 80 \
               --a_net_size 3000 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --not_use_wandb

