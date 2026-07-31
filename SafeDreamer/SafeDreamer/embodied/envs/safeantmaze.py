import os
import csv 
import pickle
import pathlib
import random
import torch

import jax
import jax.numpy as jnp
from jax import image

import embodied
import numpy as np
from gym import spaces
import pandas as pd
import cv2

from PIL import Image, ImageFont, ImageDraw
from safety_ant_maze_pusher_envs.create_env_utils import create_env


class SafeAntMaze(embodied.Env):

  def __init__(
    self,
    task,
    platform='gpu',
    mode="train",
    seed=2,
  ):
      
    class Args:
        def __init__(self, task, seed=2):
            self.seed = int(seed)
            self.domain_name = "SafetyMaze"
            self.random_start_pose = True
            self.cost_model_heatmap = False
            self.manager_algo = "none"
            
            self.pusher_random_obj_start_poses = True
            self.pusher_safe_env_safe_zone = False
            self.safe_env_hazards = False
            self.pusher_safe_env_hazards = False
            self.pusher_safe_env_dangerous_circle = True
            self.pusher_four_goal_dim = False
            self.pusher_three_goal_dim = True
            self.pusher_two_goal_dim = False
            self.pusher_hard_goal_dist = False
            self.pusher_always_random_obj_start_poses = False
            self.pusher_hard_task = False
            self.pusher_sparse_reward = False

            if task == "cshape":
                self.env_name = "SafeAntMazeC"
            elif task == "wshape":
                self.env_name = "SafeAntMazeW"
            elif task == "pusher":
                self.env_name = "SafePusher"
            else:
                assert 1 == 0, f"task: {task} doesnt exist"
            
    args = Args(task, seed=seed)
    self.args = args
      
    if args.domain_name == "SafetyMaze":
        renderer_args = {"cost_model_heatmap": args.cost_model_heatmap}
        if args.env_name == "SafePusher":
            renderer_args = {"plot_subgoal": False if args.manager_algo == "none" else True, 
                            "world_model_comparsion": False,
                            "plot_safety_boundary": True,
                            "cost_model_heatmap": args.cost_model_heatmap,
                            }
            if args.pusher_four_goal_dim:
                low = np.array([-2.0, -2.0, -2.0, -2.0])
            elif args.pusher_three_goal_dim:
                low = np.array([-2.0, -2.0, -2.0])
            elif args.pusher_two_goal_dim:
                low = np.array([-2.0, -2.0])
            else:
                low = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
            def phi(state):
                # manipulator_pose = [-6:-3], obj_pose=[-3:]
                if args.pusher_four_goal_dim:
                    return torch.cat((state[:, -3:-1], state[:, -6:-4]), dim=1)
                elif args.pusher_three_goal_dim:
                    return state[:, -3:]
                elif args.pusher_two_goal_dim:
                    return state[:, -3:-1]
                else:
                    return torch.cat((state[:, -3:], state[:, -6:-3]), dim=1)
            def pose(state):
                # manipulator_pose = [-6:-3], obj_pose=[-3:]
                if args.pusher_four_goal_dim:
                    return state[:, -6:-3]
                elif args.pusher_three_goal_dim:
                    return state[:, -6:-3]
                elif args.pusher_two_goal_dim:
                    return state[:, -6:-3]
                else:
                    return state[:, -6:-3]
        else:
            low = np.array((-10, -10))
            def phi(state):
                # ant_xy = [:2]
                return state[:, :2]
            def pose(state):
                # ant_xy = [:2]
                return state[:, :2]
    env, state_dim, goal_dim, action_dim, renderer = create_env(args, renderer_args=renderer_args)
    controller_goal_dim = goal_dim
    
    env.evaluate = True

    self._env = env

    print("****************")
    print("domain_name:", args.domain_name)
    print("env_name:", args.env_name)
    print("****************")

    from PIL import Image
    self._Image = Image
    from . import from_gymnasium
    self.wrappers = [
      from_gymnasium.FromGymnasium,
    ]

    #observation_space_image = self._env.observation_space(self.env_params)
    #action_space = self._env.action_space(self.env_params)
    state_dim = env.observation_space["observation"]
    goal_dim = env.observation_space["desired_goal"]
    
    vector_obs = spaces.Box(
                low=state_dim.low[0], 
                high=state_dim.high[0], 
                shape=(state_dim.shape[0] + goal_dim.shape[0],), 
                dtype=state_dim.dtype
    )

    is_read_step = spaces.Box(False, True, shape=(), dtype=bool)
    log_SR = spaces.Box(0, 1, shape=(), dtype=np.float16)
    obs_space = spaces.Dict({
          'vector_obs': vector_obs,
          'cost': spaces.Box(0, 1, shape=(), dtype=np.float16),
          "is_read_step": is_read_step,
          "log_success_rate": log_SR,
          "log_success_rate_reward": log_SR,
    })  
    
    self.observation_space = obs_space    
    self.action_space = self._env.action_space 

  def reset(self):
    
    obs = self._env.reset()

    dict_obs = {}
    dict_obs["vector_obs"] = np.concatenate([obs["observation"], obs["desired_goal"]])
    dict_obs["cost"] = np.float32(0)
    dict_obs["log_success_rate"] = np.float32(0)
    dict_obs["log_success_rate_reward"] = float(0)
    dict_obs["is_read_step"] = False
    
    # additional check if task is finished
    self.reward_finish = False

    return dict_obs, {}

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    cost = info["safety_cost"]

    dict_obs = {}
    dict_obs["vector_obs"] = np.concatenate([obs["observation"], obs["desired_goal"]])
    dict_obs["cost"] = np.float32(cost)
    dict_obs["is_read_step"] = False
    
    goals_achieved = 0
    if "Pusher" in self.args.env_name:
        goals_achieved += 1.0 * info["is_success"]
    elif self.args.env_name != "AntGather" and self._env.success_fn(reward):
        goals_achieved += 1
    
    dict_obs["log_success_rate"] = goals_achieved
  
    # additional check if task is finished
    if not self.reward_finish:
      if reward > 1:
        self.reward_finish = True
    dict_obs["log_success_rate_reward"] = float(self.reward_finish)

    return dict_obs, reward, cost, done, done, info
  
  def render(self):
    return self._env.render(mode="rgb_array")
  
  