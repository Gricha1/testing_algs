from collections import namedtuple

import numpy as np
import gym
import matplotlib.pyplot as plt

from .SafeAntMaze.create_env_utils import create_env


class SlacWrapper:
    def __init__(self, env, repeat, binary_cost=False):
        if not type(repeat) is int or repeat < 1:
            raise ValueError("Repeat value must be an integer and greater than 0.")
        self.action_repeat = repeat
        # hardcoded episode length 500
        self._max_episode_steps = 500//repeat
        self.binary_cost = binary_cost
        self.env = env

    def step(self, action):
        
        start = np.copy(self.env.env.base_env.wrapped_env.get_xy())
        observation, reward, done, info = self.env.step(action)
        success = reward > -5
        goal = observation['desired_goal']
        
        
        track_info = info.copy()
        track_reward = reward
        for i in range(self.action_repeat-1):
            if done or self.action_repeat==1:
                return self.get_obs(observation), reward, done, {'cost': info['safety_cost']}
            observation, reward1, done, info1 = self.env.step(action)
            success = reward > -5 or success
            for k in track_info.keys():
                track_info[k] += info1[k]
            track_reward += reward1

        if self.binary_cost:
            track_info["safety_cost"] = 1 if track_info["safety_cost"] > 0 else 0
        end = np.copy(self.env.env.base_env.wrapped_env.get_xy())
        reward = np.sqrt(np.sum((goal - start)**2)) - np.sqrt(np.sum((goal - end)**2))
        return self.get_obs(observation), reward, done, {'cost': track_info['safety_cost'], 'success': success}
    
    def get_obs(self, obs):
        vector = obs['observation']
        vector[:2] = (vector[:2] + 4) / 12 - 1
        goal = (obs['desired_goal'] + 4) / 12 - 1
        return np.concatenate([vector[:-1], goal])
    
    def reset(self):
        return self.get_obs(self.env.reset())
    
    def seed(self, seed):
        return 
    
    def render_track(self):
        pos = np.copy(self.env.env.base_env.wrapped_env.get_xy())
        goal = np.copy(self.env.env.desired_goal)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pos[0], pos[1], c='k')
        ax.scatter(goal[0], goal[1], c='r')
        ax.set_xlim(-4, 20)
        ax.set_ylim(-4, 20)
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data


def make_safe_ant_maze(action_repeat, seed, eval):
    Args = namedtuple('Args', ['env_name', 'random_start_pose', 'seed'])
    args = Args("SafeAntMaze", not eval, seed)
    renderer_args = {"plot_subgoal": False, 
                    "world_model_comparsion": False,
                    "plot_safety_boundary": True,
                    "plot_world_model_state": False}
    env, state_dim, goal_dim, action_dim, renderer = create_env(args, renderer_args=renderer_args)
    ar_env = SlacWrapper(env, repeat=action_repeat)
    if eval:
        ar_env.env.env.evaluate = True
    
    ar_env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
    ar_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return ar_env
