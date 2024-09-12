if [ -z "$1" ]; then
    seed=2
else
    seed=$1
fi

cd ../..
python main.py --seed $seed \
               --controller_imagination_safety_loss \
               --controller_grad_clip 0 \
               --validation_without_image --eval_freq 30000 \
               --random_start_pose \
               --world_model \
               --cost_model \
               --cost_memmory \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" 

