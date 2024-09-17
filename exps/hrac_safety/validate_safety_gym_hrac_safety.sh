if [ -z "$2" ]; then
    level=1
else
    level=$2
fi

cd ../..
python main.py --visulazied_episode 25 --validate \
               --domain_name Safexp \
               --task_name PointGoal$level \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --world_model \
               --cost_model \
               --img_horizon 15 \
               --manager_propose_freq 20 \
               --wandb_postfix "validate" \
               --load --loaded_exp_num 768 \
               --not_use_wandb
