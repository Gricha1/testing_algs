import os
import time
import copy
from math import ceil

import torch
import numpy as np
import pandas as pd
import wandb
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import Video

from safety_gym_wrapper.env import make_safety
from safety_gym_wrapper.experience_collection import get_safetydataset_as_random_experience
from safety_gym_wrapper.render_utils.utils import get_renderer
from envs.create_env_utils import create_env

import hrac.utils as utils
import hrac.hrac as hrac
from hrac.models import ANet
from hrac.world_models import EnsembleDynamicsModel, PredictEnv, TensorWrapper

from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/train_hiro.py
"""


def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=40, 
                    renderer=None, writer=None, total_timesteps=0, a_net=None, args=None):
    print("Starting evaluation number {}...".format(eval_idx))
    if args.test_train_dataset:
        env.evaluate = False
    else:
        env.evaluate = True
    validation_date = {}
    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        eval_image_ep = args.visulazied_episode
        if "Safe" in env_name:
            avg_cost = 0.
            avg_episode_safety_subgoal_rate = 0
            avg_episode_imagine_subgoal_safety = 0
            avg_episode_real_subgoal_safety = 0
            if env_name == "SafeAntMaze":
                safety_boundary, safe_dataset = env.get_safety_bounds(get_safe_unsafe_dataset=True)
            elif env_name == "SafeGym":
                safe_dataset = copy.copy(env.safe_dataset[0]), copy.copy(env.safe_dataset[1])
            if controller_policy.use_safe_model:
                x = safe_dataset[0]
                true = safe_dataset[1]
                #if lidar_observation:
                #    x_np = np.array(x, dtype=np.float32)
                #    manager_absolute_goal = x_np[:, :2]
                #    agent_pose = x_np[:, :2]
                #    obstacle_data = x_np[:, -16:]
                #    part_of_state = np.concatenate((agent_pose, obstacle_data), axis=1)
                #    x_np = np.concatenate((manager_absolute_goal, part_of_state), axis=1)
                #else:
                #    x_np = np.array(x, dtype=np.float32)
                x_np = np.array(x, dtype=np.float32)
                if env_name == "SafeAntMaze":
                    x_with_zeros = np.concatenate((x_np, 
                                                np.zeros((len(x), env.state_dim-2), dtype=np.float32)), 
                                                axis=1)
                else:
                    x_with_zeros = x_np
                x_tensor = torch.tensor(x_with_zeros)
                x_tensor = x_tensor.to(device)
                pred = controller_policy.safe_model(x_tensor)
                pred = (pred > 0.5).int().squeeze().tolist()
                val_safe_model_f1 = f1_score(true, pred)
                validation_date["safe_model_true_mean"] = np.mean(true)
                validation_date["safe_model_pred_mean"] = np.mean(pred)
                validation_date["safe_model_f1"] = val_safe_model_f1

        for eval_ep in range(eval_episodes):
            if env_name == "AntMazeMultiMap":
                obs = env.reset(validate=True)
            elif env_name == "SafeAntMaze":
                obs = env.reset(eval_idx=eval_ep)
            elif env_name == "SafeGym":
                # test
                # todo: make eval hard tasks
                obs = env.reset()

            goal = obs["desired_goal"]
            state = obs["observation"]

            # render env
            if eval_ep == eval_image_ep:
                if not args.validation_without_image:
                    positions_screens = []
                imagined_state_freq = 100
                prev_imagined_state = None

            prev_action = None
            done = False
            step_count = 0
            episode_reward = 0
            episode_cost = 0
            episode_controller_rew = 0
            episode_safety_subgoal_rate = 0
            episode_imagine_subgoal_safety = 0
            episode_subgoals_count = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)
                    # Get Safety Subgoal Metric
                    if manager_policy.absolute_goal:
                        if env_name == "SafeAntMaze":
                            episode_safety_subgoal_rate += env.cost_func(np.array(subgoal[:2]))
                    else:
                        if env_name == "SafeAntMaze":
                            episode_safety_subgoal_rate += env.cost_func(np.array(state[:2]) + np.array(subgoal[:2]))
                        if args.world_model:
                            with torch.no_grad():
                                state_torch = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                                subgoal_torch = torch.tensor(subgoal, dtype=torch.float32).to(device).unsqueeze(0)
                                episode_imagine_subgoal_safety += manager_policy.state_safety_on_horizon(
                                                        state_torch, subgoal_torch, 
                                                        controller_policy, 
                                                        max_horizon=manager_policy.img_horizon,
                                                        safety_cost=controller_policy.safe_model,
                                                        all_steps_safety=True
                                                        )
                    episode_subgoals_count += 1

                step_count += 1
                global_steps += 1
                if controller_policy.PPO:
                    with torch.no_grad():
                        action, _, _, _ = controller_policy.select_action_logprob_value(state, subgoal)
                    action = action.cpu().numpy().squeeze()
                else:
                    action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, info = env.step(action)
                if "Safe" in env_name:
                    cost = info["safety_cost"]
                if env_name == "SafeGym":
                    if "goal_met" in info:
                        goals_achieved += 1
                        done = True
                elif env_name != "AntGather" and env.success_fn(reward):
                    goals_achieved += 1
                    done = True

                # render env
                if not args.validation_without_image and not (renderer is None) and eval_ep == eval_image_ep:
                    if step_count == 1:
                        renderer.setup_renderer()
                    debug_info = {}
                    if env_name == "SafeAntMaze":
                        debug_info["safety_boundary"] = safety_boundary
                        debug_info["safe_dataset"] = safe_dataset
                        if args.world_model:
                            debug_info["imagine_subgoal_safety"] = episode_imagine_subgoal_safety
                    debug_info["acc_reward"] = episode_reward
                    debug_info["acc_cost"] = episode_cost
                    debug_info["acc_controller_reward"] = episode_controller_rew
                    debug_info["t"] = step_count
                    debug_info["dist_a_net_s_sg"] = 0
                    if env_name != "AntGather" and env_name != "AntMazeSparse":
                        x = a_net((torch.from_numpy(state[:2]).type('torch.FloatTensor')).to("cuda"))
                        y = a_net((torch.from_numpy(goal[:2]).type('torch.FloatTensor')).to("cuda"))
                        debug_info["dist_a_net_s_g"] = torch.sqrt(torch.pow(x - y, 2).sum() + 1e-12)
                    else:
                        debug_info["dist_a_net_s_g"] = 0
                    debug_info["dist_a_net_s_g"] = 0
                    current_step_info = {}
                    current_step_info["robot_pos"] = np.array(state[:2])
                    if env_name != "AntGather" and env_name != "AntMazeSparse":
                        current_step_info["goal_pos"] = np.array(goal[:2])
                    else:
                        current_step_info["goal_pos"] = None
                    if manager_policy.absolute_goal:
                        current_step_info["subgoal_pos"] = np.array(subgoal[:2])
                    else:
                        current_step_info["subgoal_pos"] = np.array(subgoal[:2]) + \
                                                           current_step_info["robot_pos"]
                    if env_name == "SafeGym":
                        current_step_info["robot_radius"] = env.goal_size
                    else:
                        current_step_info["robot_radius"] = 1.5
                    # get imagination of current state
                    if not(manager_policy.predict_env is None):
                        imagined_state = manager_policy.imagine_state(prev_imagined_state, prev_action, state, step_count, imagined_state_freq)
                        prev_imagined_state = imagined_state
                        current_step_info["imagined_robot_pos"] = imagined_state[:2]
                    # add apples and bombs if GatherEnv
                    if env_name =="AntGather":
                        current_step_info["apples_and_bombs"] = env.get_apples_and_bombs()
                        current_step_info["apple_bomb_radius"] = 1.0
                    if env_name == "SafeGym":
                        current_step_info["hazards"] = [hazard[:2] for hazard in env.hazards_pos]
                        current_step_info["hazards_radius"] = env.hazards_size
                        current_step_info["agent_full_obs"] = np.array(state)
                    if not args.validation_without_image:
                        screen = renderer.custom_render(current_step_info, 
                                                        debug_info=debug_info, 
                                                        plot_goal=True,
                                                        env_name=env_name,
                                                        safe_model=controller_policy.safe_model if controller_policy.use_safe_model else None)
                        positions_screens.append(screen.transpose(2, 0, 1))

                goal = new_obs["desired_goal"]
                new_state = new_obs["observation"]

                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                if "Safe" in env_name:
                    avg_cost += cost
                    episode_cost += cost
                avg_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)
                episode_reward += reward
                episode_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)

                state = new_state
                prev_action = action

            if "Safe" in env_name:
                avg_episode_safety_subgoal_rate += episode_safety_subgoal_rate / episode_subgoals_count
                avg_episode_imagine_subgoal_safety += episode_imagine_subgoal_safety / episode_subgoals_count
                avg_episode_real_subgoal_safety += episode_cost / episode_subgoals_count

        if not args.validation_without_image and not (renderer is None) and not (writer is None):
            writer.add_video(
                "eval/pos_video",
                torch.ByteTensor([positions_screens]),
                total_timesteps,
            )
            del positions_screens
            renderer.delete_data()
            
        avg_reward /= eval_episodes
        if "Safe" in env_name:
            avg_episode_safety_subgoal_rate /= eval_episodes
            validation_date["safety_subgoal_rate"] = avg_episode_safety_subgoal_rate
            if args.world_model:
                avg_episode_imagine_subgoal_safety /= eval_episodes
                avg_episode_real_subgoal_safety /= eval_episodes
                validation_date["imagine_subgoal_safety"] = avg_episode_imagine_subgoal_safety
                validation_date["real_subgoal_safety"] = avg_episode_real_subgoal_safety
            avg_cost /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        if env_name == "AntGather":
            print("Avg reward: {:.1f}".format(avg_reward))
        else:
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Avg Steps to finish: {:.1f}".format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        if "Safe" in env_name:
            return avg_reward, avg_cost, avg_controller_rew, avg_step_count, avg_env_finish, validation_date
        else:
            return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish, validation_date


def get_reward_function(dims, absolute_goal=False, binary_reward=False):
    if absolute_goal and binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(subgoal - next_z, axis=-1) <= 1.414) * scale
            return reward
    elif absolute_goal:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(subgoal - next_z, axis=-1) * scale
            return reward
    elif binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(z + subgoal - next_z, axis=-1) <= 1.414) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
        optimizer_r, controller_goal_dim, device, args,
        exp_num):
    print("train anet")
    for traj in traj_buffer.get_trajectory():
        for i in range(len(traj)):
            for j in range(1, min(args.manager_propose_freq, len(traj) - i)):
                s1 = tuple(np.round(traj[i][:controller_goal_dim]).astype(np.int32))
                s2 = tuple(np.round(traj[i+j][:controller_goal_dim]).astype(np.int32))
                if s1 not in state_list:
                    state_list.append(s1)
                    state_dict[s1] = n_states
                    n_states += 1
                if s2 not in state_list:
                    state_list.append(s2)
                    state_dict[s2] = n_states
                    n_states += 1
                adj_mat[state_dict[s1], state_dict[s2]] = 1
                adj_mat[state_dict[s2], state_dict[s1]] = 1
    print("Explored states: {}".format(n_states))
    flags = np.ones((30, 30))
    for s in state_list:
        flags[int(s[0]), int(s[1])] = 0
    print(flags)
    print("Training adjacency network...")
    loss = utils.train_adj_net(a_net, state_list, adj_mat[:n_states, :n_states],
                        optimizer_r, args.r_margin_pos, args.r_margin_neg,
                        n_epochs=args.r_training_epochs, batch_size=args.r_batch_size,
                        device=device, verbose=False, args=args)

    if args.save_models:
        r_filename = os.path.join(f"./models/{exp_num}", "{}_{}_a_network.pth".format(args.env_name, args.algo))
        torch.save(a_net.state_dict(), r_filename)
        #print("----- Adjacency network {} saved. -----".format(episode_num))

    traj_buffer.reset()

    return n_states, loss


def run_hrac(args):
    print("args:", args)

    # Environment initialization
    # goal conditioned env: obs["observation"], obs["desired_goal"], obs["achieved_goal"]
    # env.action_space
    # env.cost_func: state -> float()
    # env.evaluate = bool
    # env.success_fn: reward -> bool
    # state_dim
    # goal_dim
    # action_dim
    # renderer
    # low
    if args.domain_name == "SafetyMaze":
        env, state_dim, goal_dim, action_dim, renderer = create_env(args)
        low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
    elif args.domain_name == "Safexp":
        assert not args.goal_conditioned or (args.goal_conditioned and args.vector_env), "goal conditioned implemented only for vec obs"
        env = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', 
                            image_size=args.image_size, 
                            use_pixels=not args.vector_env, 
                            action_repeat=args.action_repeat,
                            goal_conditioned=args.goal_conditioned)
        state_dim = env.observation_space["observation"].shape[0]
        goal_dim = env.observation_space["desired_goal"].shape[0]
        action_dim = env.action_space.shape[0]
        env.state_dim = state_dim
        # test, cost unique = [0, 1, 2]
        print("get safedataset safetygym!!!")
        start_time = time.time()
        env.safe_dataset = get_safetydataset_as_random_experience(env)
        end_time = time.time()
        print("time for safe dataset:", end_time-start_time)
        renderer_args = {"plot_subgoal": True, 
                         "world_model_comparsion": False,
                         "plot_safety_boundary": False,
                         "plot_world_model_state": False,
                         "plot_cost_model_heatmap": True,
                         }
        renderer = get_renderer(env, args, renderer_args)
        env.seed(args.seed)
        # test
        # subgoal scale, only low[:2] is matter
        if "1" in args.task_name:
            low = np.array((-5, -5, -0.5, -1, -1, -1, -1,
                        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
        elif "2" in args.task_name:
            low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
        else:
            assert 1 == 0, "not recognized task lvl"
    else:
        assert 1 == 0, "there is no {args.domain_name} domain of envs"

    max_action = float(env.action_space.high[0])
    policy_noise = 0.2
    noise_clip = 0.5
    high = -low
    man_scale = (high - low) / 2
    if args.env_name == "AntFall":
        controller_goal_dim = 3
    else:
        controller_goal_dim = 2
    if args.absolute_goal:
        man_scale[0] = 30
        man_scale[1] = 30
        no_xy = False
    else:
        no_xy = True

    print("*******")
    print("env name:", args.env_name)
    print("state_dim:", state_dim)
    print("goal_dim:", goal_dim)
    print("action_dim:", action_dim)
    print("*******")
    print()


    # Set logger(Wandb logger, SummaryWriter logger) and seeds
    if not args.not_use_wandb:
        wandb_run_name = f"HRAC_{args.env_name}"
        if args.PPO:
            wandb_run_name = f"HRAC_PPO_{args.env_name}"
        wandb_run_name = wandb_run_name + "_" + args.wandb_postfix
        if args.validate:
            wandb_run_name = "validate_" + wandb_run_name + "_" + args.wandb_postfix
        run = wandb.init(
            project="safe_subgoal_model_based",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            name=wandb_run_name,
            config=args
        )
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models:
        exp_num = 0
        while os.path.exists(f"./models/{exp_num}"):
            exp_num += 1
        os.makedirs(f"./models/{exp_num}")
        if not args.not_use_wandb:
            wandb.config["model_save_path"] = f"./models/{exp_num}"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    output_dir = os.path.join(args.log_dir, args.algo)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir += "/" + args.env_name
    if args.PPO:
        output_dir += "PPO"
    output_dir += "_1"    
    while os.path.exists(output_dir + "_" + args.tensorboard_descript + "_model_" + str(exp_num)):
        run_number = int(output_dir.split("_")[-1])
        output_dir = "_".join(output_dir.split("_")[:-1])
        output_dir = output_dir + "_" + str(run_number + 1)
    output_dir += "_" + args.tensorboard_descript
    output_dir += "_model_" + str(exp_num)

    print("Logging in {}".format(output_dir))
    writer = SummaryWriter(log_dir=output_dir)
    config_text = '\t'.join([f"{key}: {value}" for key, value in vars(args).items()])
    writer.add_text('Training Configuration', config_text)
    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Initialize models
    manager_policy = hrac.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal,
        wm_no_xy=no_xy,
        modelbased_safety=args.modelbased_safety,
        img_horizon=args.img_horizon,        
        modelfree_safety=args.modelfree_safety,
        coef_safety_modelbased=args.coef_safety_modelbased,
        coef_safety_modelfree=args.coef_safety_modelfree,
        testing_mean_wm=args.testing_mean_wm,
        subgoal_grad_clip=args.subgoal_grad_clip,
        cumul_modelbased_safety=args.cumul_modelbased_safety,
        lidar_observation=True if args.domain_name == "Safexp" else False
    )
    
    cost_memmory = args.cost_memmory
    controller_policy = hrac.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        ppo_lr=args.ppo_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        PPO=args.PPO,
        hidden_dim_ppo=args.ppo_hidden_dim,
        weight_decay_ppo=args.ppo_weight_decay,
        cost_function=env.cost_func if not cost_memmory else None,
        use_safe_model=args.controller_safe_model,
        safe_model_loss_coef=args.safe_model_loss_coef,
        controller_imagination_safety_loss=args.controller_imagination_safety_loss,
        controller_grad_clip=args.controller_grad_clip,
        manager=manager_policy,
        controller_safety_coef=args.controller_safety_coef,
        lidar_observation=True if args.domain_name == "Safexp" else False
    )

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)

    if args.PPO:
        if args.noise_type == "ou":
            man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        elif args.noise_type == "normal":
            man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = None
        args.ctrl_noise_sigma = None
    else:
        if args.noise_type == "ou":
            man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
            ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

        elif args.noise_type == "normal":
            man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
            ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    if args.PPO:
        args.ppo_ctrl_batch_size == args.ppo_ctrl_buffer_size
    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size, ppo_memory=args.PPO)

    ## Train HRAC or PPO controller
    def train_controller(PPO, controller_buffer, next_done, next_state, subgoal, episode_timesteps, 
                         ep_controller_reward, episode_cost, man_episode_cost, episode_safety_subgoal_rate, 
                         ep_manager_reward, total_timesteps):
        print("train controller")
        if PPO:
            args.ppo_ctrl_batch_size = len(controller_buffer)
            # controller_buffer: 
            # 0 - x, 1 - y, 2 - g, 3 - u, 4 - r, 5 - d, 6 - l, 7 - v, 8 - x_seq, 9 - a_seq
            with torch.no_grad():
                next_value = controller_policy.get_value(next_state, subgoal).cpu().numpy().squeeze(axis=0)
                advantages = np.zeros_like(np.array(controller_buffer.storage[4]))
                lastgaelam = 0
                for t in reversed(range(args.ppo_ctrl_batch_size)):
                    if t == args.ppo_ctrl_batch_size - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - controller_buffer.storage[5][t + 1]
                        nextvalues = controller_buffer.storage[7][t + 1]
                    delta = controller_buffer.storage[4][t] + args.ppo_gamma * nextvalues * nextnonterminal - controller_buffer.storage[7][t]
                    advantages[t] = lastgaelam = delta + args.ppo_gamma * args.ppo_gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + np.array(controller_buffer.storage[7]).squeeze(1)
            controller_buffer.advantages = advantages
            controller_buffer.returns = returns
        ctrl_act_loss, ctrl_crit_loss, debug_info_controller = controller_policy.train(controller_buffer, 
            episode_timesteps if not PPO else args.ppo_update_epochs,
            batch_size=args.ppo_ctrl_batch_size if PPO else args.ctrl_batch_size, discount=args.ppo_gamma if PPO else args.ctrl_discount, 
            tau=args.ctrl_soft_sync_rate, minibatch_size=args.ppo_minibatch_size, 
            clip_coef=args.ppo_clip_coef, clip_vloss=args.ppo_clip_vloss, norm_adv=args.ppo_norm_adv, 
            max_grad_norm=args.ppo_max_grad_norm, vf_coef=args.ppo_vf_coef, 
            ent_coef=args.ppo_ent_coef, target_kl=args.ppo_target_kl,
            num_minibatches=args.ppo_num_minibatches)
        if PPO:
            controller_buffer.clear()
        if episode_num % 10 == 0:
            print("Controller actor loss: {:.3f}".format(ctrl_act_loss))
            print("Controller critic loss: {:.3f}".format(ctrl_crit_loss))
        writer.add_scalar("data/controller_actor_loss", ctrl_act_loss, total_timesteps)
        writer.add_scalar("data/controller_critic_loss", ctrl_crit_loss, total_timesteps)
        for key_ in debug_info_controller:
            writer.add_scalar(f"data/{key_}", debug_info_controller[key_], total_timesteps)

        writer.add_scalar("data/controller_ep_cost", episode_cost, total_timesteps)
        writer.add_scalar("data/controller_ep_rew", ep_controller_reward, total_timesteps)
        writer.add_scalar("data/manager_ep_rew", ep_manager_reward, total_timesteps)
        writer.add_scalar("data/manager_ep_cost", man_episode_cost, total_timesteps)
        writer.add_scalar("data/manager_ep_safety_subgoal_rate", episode_safety_subgoal_rate, total_timesteps)

    ## Initialize adjacency matrix and adjacency network
    n_states = 0
    state_list = []
    state_dict = {}
    adj_mat = np.diag(np.ones(1500, dtype=np.uint8))
    traj_buffer = utils.TrajectoryBuffer(capacity=args.traj_buffer_size)
    a_net = ANet(controller_goal_dim, args.r_hidden_dim, args.r_embedding_dim)
    if args.load_adj_net:
        print("Loading adjacency network...")
        a_net.load_state_dict(torch.load(f"./models/{args.loaded_exp_num}/{args.env_name}_{args.algo}_a_network.pth"))
    a_net.to(device)
    optimizer_r = optim.Adam(a_net.parameters(), lr=args.lr_r)

    ## Initialize world model
    num_networks = args.num_networks
    num_elites = args.num_elites
    pred_hidden_size = args.pred_hidden_size
    learning_rate = args.wm_learning_rate
    use_decay = args.use_decay
    reward_size = 0
    cost_size = 0
    env_name = 'safepg2'
    model_type='pytorch'
    if args.world_model or args.controller_safe_model:
        if args.world_model:
            with TensorWrapper():
                env_model = EnsembleDynamicsModel(num_networks, num_elites, state_dim, action_dim, 
                                                reward_size, cost_size, pred_hidden_size,
                                                learning_rate=learning_rate, use_decay=use_decay)
                predict_env = PredictEnv(env_model, env_name, model_type)
            manager_policy.set_predict_env(predict_env)
        world_model_buffer = utils.ReplayBuffer(maxsize=args.wm_buffer_size, cost_memmory=cost_memmory)
        if args.domain_name == "Safexp":
            cost_model_buffer = utils.CostModelTrajectoryBuffer(maxsize=args.wm_buffer_size)

        def train_cost_model(replay_buffer,
                             controller=None,
                             cost_model_iterations=10):
            assert controller.use_safe_model
            print("train cost model")
            debug_info = manager_policy.train_cost_model(replay_buffer, controller=controller, 
                                                            cost_model_iterations=cost_model_iterations,
                                                            cost_model_batch_size=args.cost_model_batch_size)
            for key_ in debug_info:
                if type(debug_info[key_]) == list:
                    debug_info[key_] = np.mean(debug_info[key_])
                writer.add_scalar(f"data/{key_}", debug_info[key_], total_timesteps)

            
        def train_world_model(replay_buffer, acc_wm_imagination_episode_metric):
            with TensorWrapper():
                print("train world model")
                world_model_loss = manager_policy.train_world_model(replay_buffer)
                
                writer.add_scalar("data/world_model_loss", world_model_loss, total_timesteps)
                if episode_num > 1:
                    writer.add_scalar("data/world_model_euclid_dist", acc_wm_imagination_episode_metric, total_timesteps)

                if episode_num % 10 == 0:
                    print("world model loss: {:.3f}".format(world_model_loss))
            

    if args.load:
        try:
            manager_policy.load("./models", args.env_name, args.algo, exp_num=args.loaded_exp_num)
            controller_policy.load("./models", args.env_name, args.algo, exp_num=args.loaded_exp_num)
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
            assert 1 == 0
    else:
        just_loaded = False

    if args.validate:
        # Start validation ...
        avg_ep_rew, avg_ep_cost, avg_controller_rew, avg_steps, avg_env_finish, validation_date = evaluate_policy(
            env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
            args.ctrl_rew_scale, args.manager_propose_freq, 0, 
            renderer=renderer, writer=writer, total_timesteps=0,
            a_net=a_net, args=args)
        
        writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, 0)
        writer.add_scalar("eval/avg_ep_cost", avg_ep_cost, 0)
        writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, 0)
        for key_ in validation_date:
            if type(validation_date[key_]) == list:
                validation_date[key_] = np.mean(validation_date[key_])
            writer.add_scalar(f"eval/{key_}", validation_date[key_], 0)
        if args.env_name != "AntGather":
            writer.add_scalar("eval/avg_steps_to_finish", avg_steps, 0)
            writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, 0)

        writer.close()

    else:
        # Start training ...
        ## Pretrain adj network for PPO controller
        if controller_policy.PPO:
            done = True
            print("collecting random episodes for adj network...")
            if not just_loaded:
                while not traj_buffer.full():
                    if done:
                        obs = env.reset()
                        state = obs["observation"]
                        done = False
                        traj_buffer.create_new_trajectory()
                        traj_buffer.append(state)
                    action = env.action_space.sample()
                    next_tup, manager_reward, done, info = env.step(action)   
                    next_state = next_tup["observation"]
                    traj_buffer.append(next_state)
                    state = next_state

        ## Collect transitions with random policy for world model, cost model
        done = True
        print("collecting random episodes for world model, cost model...")
        if not just_loaded:
            exploration_total_timesteps = 0
            if args.world_model or args.controller_safe_model:
                while exploration_total_timesteps < args.wm_n_initial_exploration_steps:
                    if done:
                        obs = env.reset()
                        state = obs["observation"]
                        done = False
                        if args.domain_name == "Safexp":
                            if len(cost_model_buffer.trajectory) != 0:
                                cost_model_buffer.add_trajectory_to_buffer()
                            cost_model_buffer.create_new_trajectory()
                    action = env.action_space.sample()
                    next_tup, manager_reward, done, info = env.step(action)   
                    next_state = next_tup["observation"]
                    if world_model_buffer.cost_memmory:
                        world_model_buffer.add(
                        (state, next_state, None, action, None, info["safety_cost"], None, [], [])) 
                    else:
                        world_model_buffer.add(
                            (state, next_state, None, action, None, None, [], [])) 
                    if args.domain_name == "Safexp":
                        cost_model_buffer.append(next_state, info["safety_cost"])
                    state = next_state
                    exploration_total_timesteps += 1

        ## Logging Parameters
        total_timesteps = 0
        timesteps_since_eval = 0
        timesteps_since_manager = 0
        episode_timesteps = 0
        timesteps_since_subgoal = 0
        episode_num = 0
        done = True
        evaluations = []

        ## Main training ...
        print("start training...")
        while total_timesteps < args.max_timesteps:
            if done:
                if total_timesteps != 0 and not just_loaded:
                    print("episode num:", episode_num)
                    if episode_num % 10 == 0:
                        print("Episode {}".format(episode_num))
                        
                    ## Train World Model or Cost Model
                    if args.train_safe_model:
                        if args.domain_name == "Safexp":
                            buffer = cost_model_buffer
                        else:
                            buffer = world_model_buffer
                        train_cost_model(buffer,
                                         controller=controller_policy,
                                         cost_model_iterations=episode_timesteps)
                    if args.world_model and (episode_num == 1 or (episode_num % args.wm_train_freq == 0)):
                        train_world_model(world_model_buffer, acc_wm_imagination_episode_metric)
                    
                    ## Train TD3 or PPO controller
                    train_controller(controller_policy.PPO, controller_buffer, ctrl_done, next_state, subgoal, 
                                    episode_timesteps, 
                                    ep_controller_reward, controller_episode_cost, episode_cost, 
                                    episode_safety_subgoal_rate/episode_subgoals_count, 
                                    ep_manager_reward, total_timesteps)

                    ## Train manager
                    if timesteps_since_manager >= args.train_manager_freq:
                        timesteps_since_manager = 0
                        r_margin = (args.r_margin_pos + args.r_margin_neg) / 2

                        print("train subgoal policy")
                        man_act_loss, man_crit_loss, man_goal_loss, man_safety_loss, debug_maganer_info = manager_policy.train(controller_policy,
                            manager_buffer, ceil(episode_timesteps/args.train_manager_freq),
                            batch_size=args.man_batch_size, discount=args.man_discount, tau=args.man_soft_sync_rate,
                            a_net=a_net, r_margin=r_margin)
                        
                        writer.add_scalar("data/manager_actor_loss", man_act_loss, total_timesteps)
                        writer.add_scalar("data/manager_critic_loss", man_crit_loss, total_timesteps)
                        writer.add_scalar("data/manager_goal_loss", man_goal_loss, total_timesteps)
                        for key_ in debug_maganer_info:
                            if type(debug_maganer_info[key_]) == list:
                                debug_maganer_info[key_] = np.mean(debug_maganer_info[key_])
                            writer.add_scalar(f"data/{key_}", debug_maganer_info[key_], total_timesteps)
                        if not(man_safety_loss is None):
                            writer.add_scalar("data/manager_safety_loss", man_safety_loss, total_timesteps)

                        if episode_num % 10 == 0:
                            print("Manager actor loss: {:.3f}".format(man_act_loss))
                            print("Manager critic loss: {:.3f}".format(man_crit_loss))
                            print("Manager goal loss: {:.3f}".format(man_goal_loss))
                            if not(man_safety_loss is None):
                                print("Manager safety loss: {:.3f}".format(man_safety_loss))

                    print("*************")
                    print()

                    ## Evaluate
                    if timesteps_since_eval >= args.eval_freq:
                        timesteps_since_eval = 0
                        avg_ep_rew, avg_ep_cost, avg_controller_rew, avg_steps, avg_env_finish, validation_date =\
                            evaluate_policy(env, args.env_name, manager_policy, controller_policy,
                                calculate_controller_reward, args.ctrl_rew_scale, 
                                args.manager_propose_freq, len(evaluations), 
                                renderer=renderer, writer=writer, total_timesteps=total_timesteps,
                                a_net=a_net, args=args)

                        writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
                        writer.add_scalar("eval/avg_ep_cost", avg_ep_cost, total_timesteps)
                        writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)
                        for key_ in validation_date:
                            if type(validation_date[key_]) == list:
                                validation_date[key_] = np.mean(validation_date[key_])
                            writer.add_scalar(f"eval/{key_}", validation_date[key_], total_timesteps)

                        evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
                        output_data["frames"].append(total_timesteps)
                        if args.env_name == "AntGather":
                            output_data["reward"].append(avg_ep_rew)
                        else:
                            output_data["reward"].append(avg_env_finish)
                            writer.add_scalar("eval/avg_steps_to_finish", avg_steps, total_timesteps)
                            writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, total_timesteps)
                        output_data["dist"].append(-avg_controller_rew)

                        if args.save_models:
                            controller_policy.save("./models", args.env_name, args.algo, exp_num)
                            manager_policy.save("./models", args.env_name, args.algo, exp_num)

                    if traj_buffer.full():
                        n_states, a_loss = update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
                            optimizer_r, controller_goal_dim, device, args, exp_num)
                        
                        writer.add_scalar("data/a_net_loss", a_loss, total_timesteps)


                    if len(manager_transition[-2]) != 1:                    
                        manager_transition[1] = state
                        manager_transition[5] = float(True)
                        manager_buffer.add(manager_transition)

                obs = env.reset()
                goal = obs["desired_goal"]
                state = obs["observation"]
                traj_buffer.create_new_trajectory()
                traj_buffer.append(state)
                if args.domain_name == "Safexp":
                    if len(cost_model_buffer.trajectory) != 0:
                        cost_model_buffer.add_trajectory_to_buffer()
                    cost_model_buffer.create_new_trajectory()
                done = False
                ep_controller_reward = 0
                ep_manager_reward = 0
                episode_timesteps = 0
                just_loaded = False
                episode_num += 1
                if "Safe" in args.env_name:
                    episode_cost = 0
                    controller_episode_cost = 0
                    episode_safety_subgoal_rate = 0
                    episode_subgoals_count = 0
                prev_action = None
                if args.world_model or args.train_safe_model:
                    prev_imagined_state = None
                    imagined_state_freq = args.img_horizon
                    acc_wm_imagination_episode_metric = 0
                    wm_imagination_episode_metric = 0

                subgoal = manager_policy.sample_goal(state, goal)
                if not args.absolute_goal:
                    subgoal = man_noise.perturb_action(subgoal,
                        min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
                else:
                    subgoal = man_noise.perturb_action(subgoal,
                        min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

                timesteps_since_subgoal = 0
                manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

            if controller_policy.PPO:
                with torch.no_grad():
                    action, logprob, _, value = controller_policy.select_action_logprob_value(state, subgoal)
                action = action.cpu().numpy().squeeze()
                logprob = logprob.cpu().numpy()
                value = value.cpu().numpy().squeeze(axis=0)
            else:
                action = controller_policy.select_action(state, subgoal)
                action = ctrl_noise.perturb_action(action, -max_action, max_action)

            action_copy = action.copy()

            next_tup, manager_reward, done, info = env.step(action_copy)
            cost = info["safety_cost"]

            manager_transition[4] += manager_reward * args.man_rew_scale
            ep_manager_reward += manager_reward * args.man_rew_scale
            manager_transition[-1].append(action)

            next_goal = next_tup["desired_goal"]
            next_state = next_tup["observation"]

            manager_transition[-2].append(next_state)
            traj_buffer.append(next_state)

            controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
            subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

            controller_goal = subgoal
            ep_controller_reward += controller_reward
            if "Safe" in args.env_name:
                episode_cost += cost
                controller_episode_cost += 0

            if args.inner_dones:
                ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
            else:
                ctrl_done = done


            if args.world_model or args.controller_safe_model:
                assert not controller_policy.PPO, "didnt implement wm + ppo controller"
                if args.domain_name == "Safexp":
                    cost_model_buffer.append(next_state, info["safety_cost"])
                if world_model_buffer.cost_memmory:
                    world_model_buffer.add(
                        (state, next_state, controller_goal, action, controller_reward, info["safety_cost"], float(ctrl_done), [], []))
                else:
                    world_model_buffer.add(
                        (state, next_state, controller_goal, action, controller_reward, float(ctrl_done), [], []))
            if controller_policy.PPO:
                controller_buffer.add(
                    (state, next_state, controller_goal, action, controller_reward, float(ctrl_done), logprob, value, [], []))
            else:
                controller_buffer.add(
                    (state, next_state, controller_goal, action, controller_reward, float(ctrl_done), [], []))

            state = next_state
            goal = next_goal

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            timesteps_since_manager += 1
            timesteps_since_subgoal += 1

            ## logging world model performance
            if args.world_model and episode_num > 1:
                imagined_state = manager_policy.imagine_state(prev_imagined_state, prev_action, state, episode_timesteps, imagined_state_freq)
                prev_imagined_state = imagined_state
                cur_wm_imagination_episode_metric = np.sqrt(np.sum((imagined_state[:2] - state[:2]) ** 2))
                wm_imagination_episode_metric += cur_wm_imagination_episode_metric
                if episode_timesteps % imagined_state_freq == 0:
                    acc_wm_imagination_episode_metric += wm_imagination_episode_metric / imagined_state_freq
                    wm_imagination_episode_metric = 0

            prev_action = action_copy

            if timesteps_since_subgoal % args.manager_propose_freq == 0:
                manager_transition[1] = state
                manager_transition[5] = float(done)

                manager_buffer.add(manager_transition)
                subgoal = manager_policy.sample_goal(state, goal)

                if "Safe" in args.env_name:
                    if manager_policy.absolute_goal:
                        if env_name == "SafeAntMaze":
                            episode_safety_subgoal_rate += env.cost_func(np.array(subgoal[:2]))
                    else:
                        if env_name == "SafeAntMaze":
                            episode_safety_subgoal_rate += env.cost_func(np.array(state[:2]) + np.array(subgoal[:2]))
                    episode_subgoals_count += 1

                if not args.absolute_goal:
                    subgoal = man_noise.perturb_action(subgoal,
                        min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
                else:
                    subgoal = man_noise.perturb_action(subgoal,
                        min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

                timesteps_since_subgoal = 0
                manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        ## Final evaluation
        avg_ep_rew, avg_ep_cost, avg_controller_rew, avg_steps, avg_env_finish, validation_date = evaluate_policy(
            env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
            args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations), 
            renderer=renderer, writer=writer, total_timesteps=total_timesteps,
            a_net=a_net, args=args)
        evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
        output_data["frames"].append(total_timesteps)
        if args.env_name == 'AntGather':
            output_data["reward"].append(avg_ep_rew)
        else:
            output_data["reward"].append(avg_env_finish)
        output_data["dist"].append(-avg_controller_rew)

        if args.save_models:
            controller_policy.save("./models", args.env_name, args.algo, exp_num)
            manager_policy.save("./models", args.env_name, args.algo, exp_num)

        writer.close()

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)
        print("Training finished.")
