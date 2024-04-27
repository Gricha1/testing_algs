import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import Video

import os
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pylab as plt


import wandb

import hrac.utils as utils
import hrac.hrac as hrac
from hrac.models import ANet
from envs import EnvWithGoal, GatherEnv, MultyEnvWithGoal
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env


"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/train_hiro.py
"""

class CustomVideoRendered:
    def __init__(self, env):
        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        self.add_subgoal_values = False
        self.add_mesurements = True
        self.env = env
        if self.add_subgoal_values:
            assert 1 == 0, "didnt implement"

    def custom_render(self, current_step_info, positions_render=False, 
                      plot_goal=True, debug_info={}, shape=(600, 600), env_name=""):    
        assert "robot_pos" in current_step_info and \
               "subgoal_pos" in current_step_info and \
               "goal_pos" in current_step_info and \
               "robot_radius" in current_step_info

        shift_x, shift_y = -8, -8
        env_min_x, env_max_x = -20, 20
        env_min_y, env_max_y = -20, 20
        if self.render_info["fig"] is None:
            if self.add_subgoal_values:
                self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                self.render_info["ax_subgoal_values"] = self.render_info["fig"].add_subplot(122)
            else:
                self.render_info["fig"] = plt.figure(figsize=[6.4, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(111)
        self.render_info["ax_states"].set_ylim(bottom=env_min_y, top=env_max_y)
        self.render_info["ax_states"].set_xlim(left=env_min_x, right=env_max_x)
        # robot pose
        x = current_step_info["robot_pos"][0] + shift_x
        y = current_step_info["robot_pos"][1] + shift_y
        circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="g", alpha=0.5)
        self.render_info["ax_states"].add_patch(circle_robot) 
        self.render_info["ax_states"].scatter(x, y, color="red")
        self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s")

        # subgoal
        x = current_step_info["subgoal_pos"][0] + shift_x
        y = current_step_info["subgoal_pos"][1] + shift_y
        circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="orange", alpha=0.5)
        self.render_info["ax_states"].add_patch(circle_robot)
        self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s_g")
        if self.add_subgoal_values:
            self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_s_sg"])), debug_info["v_s_sg"])
            self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_sg_g"])), debug_info["v_sg_g"])

        # goal
        if env_name != "AntGather" and env_name != "AntMazeSparse" and plot_goal:
            x = current_step_info["goal_pos"][0] + shift_x
            y = current_step_info["goal_pos"][1] + shift_y
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="y", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "g")        
        
        # print maze
        if env_name != "AntGather":
            env_map = self.env.get_maze()
            for i in env_map:
                for indx, val in enumerate(i):
                    if val == 'r':
                        i[indx] = 2

            self.render_info["ax_states"].imshow(env_map, cmap='viridis', 
                                             interpolation='nearest', 
                                             extent=[env_min_x, env_max_x, env_min_y, env_max_y])
        else:
            # add apples & bombs
            apples_and_bombs = current_step_info["apples_and_bombs"]

            apples = [(x, y) for (x, y, type_) in apples_and_bombs if type_ == 0]
            apples = [plt.Circle([obs[0] + shift_x, obs[1] + shift_y], radius=current_step_info["apple_bomb_radius"],  # noqa
                        color="y", alpha=0.5) for obs in apples]
            
            bombs = [(x, y) for (x, y, type_) in apples_and_bombs if type_ == 1]
            bombs = [plt.Circle([obs[0] + shift_x, obs[1] + shift_y], radius=current_step_info["apple_bomb_radius"],  # noqa
                        color="r", alpha=0.5) for obs in bombs]
            
            for item_ in apples:
                self.render_info["ax_states"].add_patch(item_)
            for item_ in bombs:
                self.render_info["ax_states"].add_patch(item_)
                
        

        if self.add_mesurements:    
            # debug info
            if len(debug_info) != 0:
                acc_reward = debug_info["acc_reward"]
                t = debug_info["t"]
                dist_a_net_s_sg = debug_info["dist_a_net_s_sg"]
                dist_a_net_s_g = debug_info["dist_a_net_s_g"]
                #acc_cost = debug_info["acc_cost"]
                #self.render_info["ax_states"].text(env_max_x - 4.5, env_max_y - 0.3, f"a0:{int(a0*100)/100}")
                #self.render_info["ax_states"].text(env_max_x - 3.5, env_max_y - 0.3, f"a1:{int(a1*100)/100}")
                #self.render_info["ax_states"].text(env_max_x - 1.5, env_max_y - 0.3, f"C:{int(acc_cost*10)/10}")
                #self.render_info["ax_states"].text(env_max_x - 10.5, env_max_y - 0.3, f"a_net(s, sg):{dist_a_net_s_sg}")
                self.render_info["ax_states"].text(env_max_x - 22.5, env_max_y - 2, f"a_net(s, g):{dist_a_net_s_g}")
                self.render_info["ax_states"].text(env_max_x - 8.5, env_max_y - 2, f"R:{int(acc_reward*10)/10}")
                #self.render_info["ax_states"].text(env_max_x - 10.5, env_max_y - 2, f"t:{t}")

        # render img
        # self.render_info["fig"].savefig("example.png")
        self.render_info["fig"].canvas.draw()
        data = np.frombuffer(self.render_info["fig"].canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.render_info["fig"].canvas.get_width_height()[::-1] + (3,))
        self.render_info["ax_states"].clear()
        if self.add_subgoal_values:
            self.render_info["ax_subgoal_values"].clear()
        return data


def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5, 
                    renderer=None, writer=None, total_timesteps=0, a_net=None):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            if env_name == "AntMazeMultiMap":
                obs = env.reset(validate=True)
            else:
                obs = env.reset()

            goal = obs["desired_goal"]
            state = obs["observation"]

            # render
            if eval_ep == 0:
                positions_screens = []

            done = False
            step_count = 0
            env_goals_achieved = 0
            episode_reward = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                step_count += 1
                global_steps += 1
                if controller_policy.PPO:
                    with torch.no_grad():
                        action, _, _, _ = controller_policy.select_action_logprob_value(state, subgoal)
                    action = action.cpu().numpy().squeeze()
                else:
                    action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, _ = env.step(action)
                if env_name != "AntGather" and env.success_fn(reward):
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                episode_reward += reward
                # render env
                if not (renderer is None) and eval_ep == 0:
                    debug_info = {}
                    debug_info["acc_reward"] = episode_reward
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
                    current_step_info["robot_radius"] = 1.5
                    
                    if env_name =="AntGather":
                        current_step_info["apples_and_bombs"] = env.get_apples_and_bombs()
                        current_step_info["apple_bomb_radius"] = 1.0
                    screen = renderer.custom_render(current_step_info, debug_info=debug_info, 
                                                    plot_goal=True,
                                                    env_name=env_name)
                    positions_screens.append(screen.transpose(2, 0, 1))

                goal = new_obs["desired_goal"]
                new_state = new_obs["observation"]

                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)

                state = new_state

        if not (renderer is None) and not (writer is None):
            writer.add_video(
                "eval/pos_video",
                #Video(torch.ByteTensor([positions_screens]), fps=40),
                torch.ByteTensor([positions_screens]),
                total_timesteps,
                #exclude=("stdout", "log", "json", "csv"),
            )
            del positions_screens
            
        avg_reward /= eval_episodes
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
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


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
        optimizer_r, controller_goal_dim, device, args):
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
    if not args.load_adj_net:
        print("Training adjacency network...")
        utils.train_adj_net(a_net, state_list, adj_mat[:n_states, :n_states],
                            optimizer_r, args.r_margin_pos, args.r_margin_neg,
                            n_epochs=args.r_training_epochs, batch_size=args.r_batch_size,
                            device=device, verbose=False)

        if args.save_models:
            r_filename = os.path.join("./models", "{}_{}_a_network.pth".format(args.env_name, args.algo))
            torch.save(a_net.state_dict(), r_filename)
            print("----- Adjacency network {} saved. -----".format(episode_num))

    traj_buffer.reset()

    return n_states


def run_hrac(args):
    print("args:", args)

    if args.use_wandb:
        run = wandb.init(
            project="safe_subgoal_model_based",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            name=f"HRAC_{args.env_name}"
        )
    
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    output_dir = os.path.join(args.log_dir, args.algo)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir += "/" + args.env_name
    if args.PPO:
        output_dir += "PPO"
    output_dir += "_1"
    # output_dir = "logs/hrac/AntGather_1"
    while os.path.exists(output_dir):
        run_number = int(output_dir.split("_")[-1])
        output_dir = "_".join(output_dir.split("_")[:-1])
        output_dir = output_dir + "_" + str(run_number + 1)
    print("Logging in {}".format(output_dir))

    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)   
    elif args.env_name in ["AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        if args.env_name == "AntMaze":
            maze_id = "Maze"
        elif args.env_name == "AntMazeSparse":
            maze_id = "Maze2"
        elif args.env_name == "AntPush":
            maze_id = "Push"
        elif args.env_name == "AntFall":
            maze_id = "Fall"
        else:
            assert 1 == 0
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed, maze_id=maze_id), args.env_name, maze_id=maze_id)
        env.seed(args.seed)
    elif args.env_name == "AntMazeMultiMap":    
        maze_ids = ["Maze_map_1", "Maze_map_2", "Maze_map_3", "Maze_map_4"]
        envs = []
        for maze_id in maze_ids:
            env = EnvWithGoal(create_maze_env(args.env_name, args.seed, maze_id=maze_id), args.env_name, maze_id=maze_id)
            env.seed(args.seed)
            envs.append(env)
            env = MultyEnvWithGoal(envs)
        env.seed(args.seed)
    else:
        raise NotImplementedError
    
    # render
    renderer = CustomVideoRendered(env)
    
    low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
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
    action_dim = env.action_space.shape[0]

    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    writer = SummaryWriter(log_dir=output_dir)
    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    if args.env_name in ["AntMaze", "AntPush", "AntFall", "AntMazeMultiMap"]:
        goal_dim = goal.shape[0]
    else:
        goal_dim = 0

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
        PPO=args.PPO
    )

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
        absolute_goal=args.absolute_goal
    )

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)

    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    if args.PPO:
        args.ctrl_batch_size == args.ctrl_buffer_size
    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size, ppo_memory=args.PPO)

    # Initialize adjacency matrix and adjacency network
    n_states = 0
    state_list = []
    state_dict = {}
    adj_mat = np.diag(np.ones(1500, dtype=np.uint8))
    traj_buffer = utils.TrajectoryBuffer(capacity=args.traj_buffer_size)
    a_net = ANet(controller_goal_dim, args.r_hidden_dim, args.r_embedding_dim)
    if args.load_adj_net:
        print("Loading adjacency network...")
        a_net.load_state_dict(torch.load("./models/a_network.pth"))
    a_net.to(device)
    optimizer_r = optim.Adam(a_net.parameters(), lr=args.lr_r)

    if args.load:
        try:
            manager_policy.load("./models")
            controller_policy.load("./models")
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    # Train
    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}".format(episode_num))
                # Train controller
                if not controller_policy.PPO or (controller_policy.PPO and len(controller_buffer) == args.ctrl_batch_size):
                    if controller_policy.PPO:
                        with torch.no_grad():
                            #next_value = controller_policy.agent.get_value(next_obs).reshape(1, -1)
                            advantages = np.zeros_like(np.array(controller_buffer.storage[4]))
                            lastgaelam = 0
                            for t in reversed(range(args.ctrl_batch_size)):
                                if t == args.ctrl_batch_size - 1:
                                    nextnonterminal = 1.0 - done
                                    nextvalues = value
                                else:
                                    nextnonterminal = 1.0 - controller_buffer.storage[5][t + 1]
                                    nextvalues = controller_buffer.storage[7][t + 1]
                                delta = controller_buffer.storage[4][t] + args.ctrl_discount * nextvalues * nextnonterminal - controller_buffer.storage[7][t]
                                advantages[t] = lastgaelam = delta + args.ctrl_discount * args.gae_lambda * nextnonterminal * lastgaelam
                            returns = advantages + np.array(controller_buffer.storage[7]).squeeze(1)
                        controller_buffer.advantages = advantages
                        controller_buffer.returns = returns
                    ctrl_act_loss, ctrl_crit_loss = controller_policy.train(controller_buffer, 
                        episode_timesteps if not controller_policy.PPO else args.update_epochs,
                        batch_size=args.ctrl_batch_size, discount=args.ctrl_discount, tau=args.ctrl_soft_sync_rate,
                        minibatch_size=args.minibatch_size, clip_coef=args.clip_coef, 
                        clip_vloss=args.clip_vloss, norm_adv=args.norm_adv, 
                        max_grad_norm=args.max_grad_norm, vf_coef=args.vf_coef, 
                        ent_coef=args.ent_coef, target_kl=args.target_kl)
                    if controller_policy.PPO:
                        controller_buffer.clear()
                    if episode_num % 10 == 0:
                        print("Controller actor loss: {:.3f}".format(ctrl_act_loss))
                        print("Controller critic loss: {:.3f}".format(ctrl_crit_loss))
                    writer.add_scalar("data/controller_actor_loss", ctrl_act_loss, total_timesteps)
                    writer.add_scalar("data/controller_critic_loss", ctrl_crit_loss, total_timesteps)

                    writer.add_scalar("data/controller_ep_rew", episode_reward, total_timesteps)
                    writer.add_scalar("data/manager_ep_rew", manager_transition[4], total_timesteps)

                # Train manager
                if timesteps_since_manager >= args.train_manager_freq:
                    timesteps_since_manager = 0
                    r_margin = (args.r_margin_pos + args.r_margin_neg) / 2

                    man_act_loss, man_crit_loss, man_goal_loss = manager_policy.train(controller_policy,
                        manager_buffer, ceil(episode_timesteps/args.train_manager_freq),
                        batch_size=args.man_batch_size, discount=args.man_discount, tau=args.man_soft_sync_rate,
                        a_net=a_net, r_margin=r_margin)
                    
                    writer.add_scalar("data/manager_actor_loss", man_act_loss, total_timesteps)
                    writer.add_scalar("data/manager_critic_loss", man_crit_loss, total_timesteps)
                    writer.add_scalar("data/manager_goal_loss", man_goal_loss, total_timesteps)

                    if episode_num % 10 == 0:
                        print("Manager actor loss: {:.3f}".format(man_act_loss))
                        print("Manager critic loss: {:.3f}".format(man_crit_loss))
                        print("Manager goal loss: {:.3f}".format(man_goal_loss))

                # Evaluate
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish =\
                        evaluate_policy(env, args.env_name, manager_policy, controller_policy,
                            calculate_controller_reward, args.ctrl_rew_scale, 
                            args.manager_propose_freq, len(evaluations), 
                            renderer=renderer, writer=writer, total_timesteps=total_timesteps,
                            a_net=a_net)

                    writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
                    writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)

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
                        controller_policy.save("./models", args.env_name, args.algo)
                        manager_policy.save("./models", args.env_name, args.algo)

                if traj_buffer.full():
                     n_states = update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
                        optimizer_r, controller_goal_dim, device, args)

                if len(manager_transition[-2]) != 1:                    
                    manager_transition[1] = state
                    manager_transition[5] = float(True)
                    manager_buffer.add(manager_transition)

            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]
            traj_buffer.create_new_trajectory()
            traj_buffer.append(state)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

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

        next_tup, manager_reward, done, _ = env.step(action_copy)

        manager_transition[4] += manager_reward * args.man_rew_scale
        manager_transition[-1].append(action)

        next_goal = next_tup["desired_goal"]
        next_state = next_tup["observation"]

        manager_transition[-2].append(next_state)
        traj_buffer.append(next_state)

        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)

        controller_goal = subgoal
        episode_reward += controller_reward

        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done

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

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            manager_transition[1] = state
            manager_transition[5] = float(done)

            manager_buffer.add(manager_transition)
            subgoal = manager_policy.sample_goal(state, goal)

            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
            else:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

            timesteps_since_subgoal = 0
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish = evaluate_policy(
        env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations), 
        renderer=renderer, writer=writer, total_timesteps=total_timesteps,
        a_net=a_net)
    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
    output_data["frames"].append(total_timesteps)
    if args.env_name == 'AntGather':
        output_data["reward"].append(avg_ep_rew)
    else:
        output_data["reward"].append(avg_env_finish)
    output_data["dist"].append(-avg_controller_rew)

    if args.save_models:
        controller_policy.save("./models", args.env_name, args.algo)
        manager_policy.save("./models", args.env_name, args.algo)

    writer.close()

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)
    print("Training finished.")
