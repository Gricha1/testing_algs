# HRAC + Safety
#python main.py --validation --controller_safe_model --safety_subgoals --testing_safety_subgoal --wandb_postfix ""
#python main.py --validation_without_image --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""
#python main.py --visulazied_episode 0 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --controller_safe_model --wandb_postfix ""
#python main.py --visulazied_episode 13 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --controller_safe_model --wandb_postfix ""

#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --controller_safe_model --wandb_postfix ""
#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 17 --safety_subgoals --testing_safety_subgoal --controller_safe_model --wandb_postfix ""


python main.py --env_name SafeAntMaze \
               --validation_without_image --visulazied_episode 0 --validate \
               --load --loaded_exp_num 233 \
               --controller_safe_model --wandb_postfix ""
#python main.py --env_name SafeAntMaze --visulazied_episode 0 --validate --load --loaded_exp_num 302 --world_model --safety_subgoals --controller_safe_model --wandb_postfix ""
#python main.py --env_name Safexp-PointGoal2 --visulazied_episode 0 --validate --load --loaded_exp_num 302 --world_model --safety_subgoals --controller_safe_model --wandb_postfix ""

#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 279 --world_model --safety_subgoals --controller_safe_model --wandb_postfix ""

#python main.py --test_train_dataset --random_start_pose --visulazied_episode 13 --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""
#python main.py --visulazied_episode 37 --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""