# HRAC + Safety
#python main.py --validation --cost_model --safety_subgoals --testing_safety_subgoal --wandb_postfix ""
#python main.py --validation_without_image --validate --load --loaded_exp_num 153 --cost_model --wandb_postfix ""
#python main.py --visulazied_episode 0 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --cost_model --wandb_postfix ""
#python main.py --visulazied_episode 13 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --cost_model --wandb_postfix ""

#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 231 --safety_subgoals --testing_safety_subgoal --cost_model --wandb_postfix ""
#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 17 --safety_subgoals --testing_safety_subgoal --cost_model --wandb_postfix ""


python main.py --visulazied_episode 0 --validate \
               --load --loaded_exp_num final_mb_safety --world_model \
               --cost_model \
               --wandb_postfix "mb_mb_safety"
"""
python main.py --visulazied_episode 1 --validate \
               --load --loaded_exp_num final_mb_safety --world_model \
               --cost_model \
               --wandb_postfix "mb_mb_safety"
python main.py --visulazied_episode 2 --validate \
               --load --loaded_exp_num final_mb_safety --world_model \
               --cost_model \
               --wandb_postfix "mb_mb_safety"
python main.py --visulazied_episode 3 --validate \
               --load --loaded_exp_num final_mb_safety --world_model \
               --cost_model \
               --wandb_postfix "mb_mb_safety"

python main.py --visulazied_episode 0 --validate \
               --load --loaded_exp_num 388 --world_model \
               --cost_model \
               --wandb_postfix "final_mf_safety"
python main.py --visulazied_episode 1 --validate \
               --load --loaded_exp_num 388 --world_model \
               --cost_model \
               --wandb_postfix "final_mf_safety"
python main.py --visulazied_episode 2 --validate \
               --load --loaded_exp_num 388 --world_model \
               --cost_model \
               --wandb_postfix "final_mf_safety"
python main.py --visulazied_episode 3 --validate \
               --load --loaded_exp_num 388 --world_model \
               --cost_model \
               --wandb_postfix "final_mf_safety"

"""

#python main.py --visulazied_episode 1 --validate --load --loaded_exp_num 279 --world_model --safety_subgoals --cost_model --wandb_postfix ""

#python main.py --test_train_dataset --random_start_pose --visulazied_episode 13 --validate --load --loaded_exp_num 153 --cost_model --wandb_postfix ""
#python main.py --visulazied_episode 37 --validate --load --loaded_exp_num 153 --cost_model --wandb_postfix ""
