if [ -z "$1" ]; then
    seed=344
else
    seed=$1
fi


cd ../..
python main.py --domain_name Safexp \
               --task_name PointGoal1 \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed $seed \
               --world_model \
               --controller_imagination_safety_loss \
               --validation_without_image --eval_freq 30000 \
               --man_rew_scale 0.1 --goal_loss_coeff 20.0 \
               --controller_safety_coef 6 \
               --max_timesteps 4000000 \
               --wandb_postfix "" \
               --cost_model \
               --img_horizon 10 \
               --wm_pretrain \
               --wm_pretrain_epoches 100 \
               --wm_n_initial_exploration_steps 30000 \

