# HRAC + Safety
#python main.py --validation --controller_safe_model --safety_subgoals --testing_safety_subgoal --wandb_postfix ""
#python main.py --validation_without_image --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""
python main.py --test_train_dataset --random_start_pose --visulazied_episode 21 --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""
python main.py --test_train_dataset --random_start_pose --visulazied_episode 13 --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""
#python main.py --visulazied_episode 37 --validate --load --loaded_exp_num 153 --controller_safe_model --wandb_postfix ""