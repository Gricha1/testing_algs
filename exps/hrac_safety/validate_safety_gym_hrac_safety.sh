if [ -z "$2" ]; then
    level=1
else
    level=$2
fi

cd ../..
python main.py --visulazied_episode 10 --validate \
               --domain_name Safexp \
               --task_name PointGoal$level \
               --env_name SafeGym \
               --goal_conditioned \
               --vector_env \
               --seed 344 \
               --world_model \
               --controller_safe_model \
               --wandb_postfix "validate" \
               --load --loaded_exp_num 728
