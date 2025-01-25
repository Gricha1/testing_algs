if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi

cd ../..
python main.py --domain_name SafetyMaze \
               --env_name SafePusher \
               --seed $seed \
               --cost_memmory \
               --img_horizon 10 \
               --goal_loss_coeff 20 \
               --validation_without_image \
               --eval_freq 30000 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --a_net_size 3000 \
               --a_net_discretization_koef 3.0 \
               --not_use_wandb \
               #--r_margin_pos 0.5 \
               #--r_margin_neg 0.7 \
               #--man_rew_scale 100.0 \
               #--manager_propose_freq 10 \
               #--train_manager_freq 5 \

