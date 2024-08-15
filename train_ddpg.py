import os
import time
from collections import deque

import numpy as np
import wandb
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.colormaps['tab10']

from envs.SafeAntMaze import EnvWithGoal, SafeMazeAnt
from envs.SafeAntMaze.create_maze_env import create_maze_env
from ddpg import controller, utils


def evaluate_policy(
    env, policy, calculate_controller_reward, ctrl_rew_scale,
    eval_idx=0, eval_episodes=40
):
    print("Starting evaluation number {}...".format(eval_idx))
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        avg_cost = 0
        global_steps = 0
        goals_achieved = 0
        successes = np.zeros(eval_episodes)
        trajs = []
        goals = []
        for eval_ep in range(eval_episodes):
            obs = env.reset(eval_idx=eval_ep)
            
            goal = obs["desired_goal"]
            state = obs["observation"]
            traj = [state[:2].copy()]
            goals.append(goal.copy())
            subgoal = (goal - state[:2]).copy()

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                step_count += 1
                global_steps += 1
                action = policy.select_action(
                    state, subgoal, evaluation=True
                )
                new_obs, reward, done, info = env.step(action)
                avg_cost += info['safety_cost']

                if env.success_fn(reward):
                    env_goals_achieved += 1
                    goals_achieved += 1
                    successes[eval_ep] = 1
                    done = True

                goal = new_obs["desired_goal"]
                new_state = new_obs["observation"]
                traj.append(new_state[:2].copy())

                subgoal = policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(
                    state, subgoal, new_state, ctrl_rew_scale
                )

                state = new_state
            trajs.append(np.stack(traj))
        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes
        avg_cost /= eval_episodes

        print("---------------------------------------")
        print(
            f"Evaluation over {eval_episodes} episodes:",
            f"\nAvg Ctrl Reward: {avg_controller_rew:.3f}"
        )
        print(f"Goals achieved: {100*avg_env_finish:.1f}%")
        print(f"Avg Steps to finish: {avg_step_count:.1f}")
        print("---------------------------------------")

        env.evaluate = False
        goals = np.stack(goals)
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish, \
            goals, trajs, successes, avg_cost


def get_reward_function(dims, absolute_goal=False, binary_reward=False):
    if absolute_goal and binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(
                subgoal - next_z, axis=-1
            ) <= 1.414) * scale
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
            reward = float(np.linalg.norm(
                z + subgoal - next_z, axis=-1
            ) <= 1.414) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def generate_goal(env, step, xy, absolute_goal, scheduling, safe_goal):
    start_point_found = False
    n_points = 1000
    scale = min(400_000, step) / 400_000
    goal = np.array([0, 0])
    scale = 1 + scale * 24 * np.sqrt(2)

    while not (
        start_point_found and 
        (scale > np.linalg.norm(goal) or not scheduling)
    ):
        points = np.random.uniform((-4, -4), (20, 20), (n_points, 2))
        if safe_goal:
            cost_idx = env.cost_func(points)
            safety_states = (1 - cost_idx) == True
            if safety_states.any():
                safe_ind = np.where(safety_states)[0][0]
                start_point_found = True
            goal = points[safe_ind].copy() - xy
        else:
            goal = points[0].copy() - xy
            start_point_found = True

    if absolute_goal:
        goal = (goal + xy).copy()
    return goal


def make_plot(funcs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for func in funcs:
        func(ax)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def run_ddpg_controller():
    exp_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    config = {
        'name': exp_name,
        'logdir': os.path.join('/logdir/controller', exp_name),
        'seed': 0,
        'gid': 0,
        'ctrl_act_lr': 1e-4,
        'ctrl_crit_lr': 1e-3,
        'absolute_goal': False,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'ctrl_noise_sigma': 1,
        'ctrl_buffer_size': 1_000_000,
        'load': False,
        'max_timesteps': 4_000_000,
        'ctrl_rew_scale': 1,
        'ctrl_batch_size': 256,
        'ctrl_discount': 0.99,
        'ctrl_soft_sync_rate': 0.005,
        'eval_freq': 30_000,
        'save_models': True,
        'binary_int_reward': False,
        'no_xy': False,
        'scheduling': False,
        'safe_goal': False,
        'her': False,
        'her_ratio': 0.25,
        'her_strategy': 'episode',
        'pid_kp': 1e-6,
        'pid_ki': 1e-7,
        'pid_kd': 1e-7,
        'pid_d_delay': 10,
        'pid_delta_p_ema_alpha': 0.95,
        'pid_delta_d_ema_alpha': 0.95,
        'lagrangian_multiplier_init': 0.,
        'cost_limit': 40.,
        'use_lagrange': True,
    }

    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])

    maze_id = "MazeSafe_map_1"
    env = SafeMazeAnt(EnvWithGoal(create_maze_env(
        "AntMaze", config['seed'], maze_id=maze_id
    ), "AntMaze", maze_id=maze_id))
    env.set_train_start_pose_to_random()
    env.seed(config['seed'])

    max_action = float(env.action_space.high[0])
    goal_dim = 2
    action_dim = env.action_space.shape[0]
    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]
    state_dim = state.shape[0]

    run = wandb.init(project='SafeAntMaze', name=config['name'], config=config)
    torch.cuda.set_device(config['gid'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_data = {"frames": [], "reward": [], "dist": []}    

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    policy = controller.Controller(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=config['ctrl_act_lr'],
        critic_lr=config['ctrl_crit_lr'],
        no_xy=config['no_xy'],
        absolute_goal=config['absolute_goal'],
        policy_noise=config['policy_noise'],
        noise_clip=config['noise_clip'],
        pid_kp=config['pid_kp'],
        pid_ki=config['pid_ki'],
        pid_kd=config['pid_kd'],
        pid_d_delay=config['pid_d_delay'],
        pid_delta_p_ema_alpha=config['pid_delta_p_ema_alpha'],
        pid_delta_d_ema_alpha=config['pid_delta_d_ema_alpha'],
        lagrangian_multiplier_init=config['lagrangian_multiplier_init'],
        cost_limit=config['cost_limit'],
        use_lagrange=config['use_lagrange']
    )

    calculate_controller_reward = get_reward_function(
        goal_dim, config['absolute_goal'], config['binary_int_reward']
    )

    ctrl_noise = utils.NormalNoise(sigma=config['ctrl_noise_sigma'])
    if config['her']:
        buffer = utils.HERBuffer(
            500, calculate_controller_reward, config['absolute_goal'],
            maxsize=config['ctrl_buffer_size'], strategy=config['her_strategy'],
            ratio=config['her_ratio']
        )
    else:
        buffer = utils.ReplayBuffer(maxsize=config['ctrl_buffer_size'])

    if config['load']:
        try:
            policy.load("./models")
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
    episode_timesteps = 0
    episode_num = 0
    success = 0
    cumulative_cost = 0
    episode_cost = 0
    done = True
    evaluations = []
    train_subgoals = deque(maxlen=10000)
    train_starts = deque(maxlen=10000)

    # Train
    while total_timesteps < config['max_timesteps']:
        if done:
            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}".format(episode_num))
                # Train controller
                ctrl_act_loss, ctrl_crit_loss, cost_loss = policy.train(
                    buffer, episode_timesteps,
                    batch_size=config['ctrl_batch_size'],
                    discount=config['ctrl_discount'],
                    tau=config['ctrl_soft_sync_rate']
                )
                if episode_num % 10 == 0:
                    print(f"Controller actor loss: {ctrl_act_loss :.3f}")
                    print(f"Controller critic loss: {ctrl_crit_loss:.3f}")

                mets = {
                    "data/controller_actor_loss": ctrl_act_loss,
                    "data/controller_critic_loss": ctrl_crit_loss,
                    "data/controller_cost_critic_loss": cost_loss,
                    "data/controller_ep_rew": episode_reward,
                    "data/controller_ep_success": success / episode_timesteps,
                    "data/cost_rate": cumulative_cost / total_timesteps,
                    "data/episode_cost": episode_cost,
                    "data/lagrangian": policy.lagrangian()
                }
                run.log(mets, step=total_timesteps)

                # Evaluate
                if timesteps_since_eval >= config['eval_freq']:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish,\
                    eval_goals, eval_trajs, eval_sucs, avg_cost = \
                        evaluate_policy(
                            env, policy, calculate_controller_reward,
                            config['ctrl_rew_scale'], len(evaluations)
                        )

                    plot_goals = np.stack(train_subgoals)
                    plot_goals = make_plot([
                        lambda ax: ax.hist2d(
                            plot_goals[:, 0], plot_goals[:, 1], 100
                        ),
                        lambda ax: ax.scatter(0, 0),
                    ])

                    plot_starts = np.stack(train_starts)
                    plot_starts = make_plot([
                        lambda ax: ax.hist2d(
                            plot_starts[:, 0], plot_starts[:, 1], 100
                        ),
                        lambda ax: ax.scatter(0, 0),
                    ])

                    mets = {}
                    mets["eval/avg_ep_rew"] = avg_ep_rew
                    mets["eval/avg_controller_rew"] = avg_controller_rew
                    mets["eval/avg_cost"] = avg_cost
                    mets['eval/goals'] = wandb.Image(plot_goals)
                    mets['eval/starts'] = wandb.Image(plot_starts)

                    funcs = [lambda ax: ax.scatter(
                        eval_goals[:, 0], eval_goals[:, 1], c='r'
                    )]

                    def plot_trajs(ax):
                        for i, traj in enumerate(eval_trajs):
                            ax.plot(
                                traj[:, 0], traj[:, 1],
                                c=cmap(i) if not eval_sucs[i] else 'k'
                            )
                    funcs.append(plot_trajs)
                    mets['eval/trajs'] = wandb.Image(make_plot(funcs))

                    evaluations.append(
                        [avg_ep_rew, avg_controller_rew, avg_steps]
                    )
                    output_data["frames"].append(total_timesteps)
                    output_data["reward"].append(avg_env_finish)
                    mets["eval/avg_steps_to_finish"] = avg_steps
                    mets["eval/perc_env_goal_achieved"] = avg_env_finish
                    run.log(mets, step=total_timesteps)
                    output_data["dist"].append(-avg_controller_rew)

                    if config['save_models']:
                        policy.save(config['logdir'], 'models', config['name'])

            obs = env.reset()
            state = obs["observation"]
            train_starts.append(state[:2].copy())
            done = False
            episode_reward = 0
            episode_timesteps = 0
            success = 0
            just_loaded = False
            episode_num += 1
            episode_cost = 0

        if episode_timesteps % 20 == 0:
            subgoal = generate_goal(
                env, total_timesteps, state[:2].copy(), config['absolute_goal'],
                config['scheduling'], config['safe_goal']
            )
            train_subgoals.append(subgoal.copy())
        
        action = policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, -max_action, max_action)
        action_copy = action.copy()

        next_tup, manager_reward, done, info = env.step(action_copy)
        cumulative_cost += info['safety_cost']
        episode_cost += info['safety_cost']
        policy.update_lag(episode_cost)

        next_state = next_tup["observation"]
        controller_reward = calculate_controller_reward(
            state, subgoal, next_state, config['ctrl_rew_scale'])
        subgoal = policy.subgoal_transition(state, subgoal, next_state)

        success += np.linalg.norm(subgoal) < 1
        controller_goal = subgoal
        episode_reward += controller_reward
        # Check done
        ctrl_done = done

        buffer.add((
            state, next_state, controller_goal, action, controller_reward,
            float(ctrl_done), info['safety_cost']
        ))

        state = next_state
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish, eval_goals,\
        eval_trajs, eval_sucs, avg_cost = evaluate_policy(
            env, policy, calculate_controller_reward, config['ctrl_rew_scale'],
            len(evaluations)
        )
    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
    output_data["frames"].append(total_timesteps)
    output_data["reward"].append(avg_env_finish)
    output_data["dist"].append(-avg_controller_rew)

    if config['save_models']:
        policy.save(config['logdir'], 'models', config['name'])

    run.close()

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(
        os.path.join("results.csv"), float_format="%.4f", index=False
    )
    print("Training finished.")


if __name__ == "__main__":
    run_ddpg_controller()