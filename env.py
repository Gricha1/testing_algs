import gym
from gym.wrappers.transform_observation import TransformObservation
from gym.wrappers.transform_reward import TransformReward
from gym.spaces.box import Box
import numpy as np
import torch
import safety_gym
"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np

from gym import Wrapper, spaces
from gym import ObservationWrapper
from gym.envs.registration import register


STATE_KEY = 'state'
class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, repeat, binary_cost=False):
        super().__init__(env)
        if not type(repeat) is int or repeat < 1:
            raise ValueError("Repeat value must be an integer and greater than 0.")
        self.action_repeat = repeat
        self._max_episode_steps = env.config["num_steps"]//repeat
        self.binary_cost = binary_cost

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        track_info = info.copy()
        track_reward = reward
        for i in range(self.action_repeat-1):
            if done or self.action_repeat==1:
                return observation, reward, done, info
            observation1, reward1, done1, info1 = self.env.step(action)
            track_info["cost"] += info1["cost"]
            track_reward += reward1

        if self.binary_cost:
            track_info["cost"] = 1 if track_info["cost"] > 0 else 0
        track_info["safety_cost"] = track_info["cost"]
        return observation1, track_reward, done1, track_info
    


class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

# Pixel observation wrapper based on OpenAI Gym implementation.
# The MIT License

# Copyright (c) 2016 OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


    def __init__(self,
                 env,
                 pixels_only=True,
                 render_kwargs=None,
                 pixel_keys=('pixels', )):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains any of the
                specified `pixel_keys`.
        """

        super(PixelObservationWrapper, self).__init__(env)

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop('mode', 'rgb_array')
            assert render_mode == 'rgb_array', render_mode
            render_kwargs[key]['mode'] = 'rgb_array'

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError("Duplicate or reserved pixel keys {!r}."
                                 .format(overlapping_keys))

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.

        pixels_spaces = {}
        for pixel_key in pixel_keys:
            render_kwargs[pixel_key]["mode"] ="offscreen"
            pixels = self.env.sim.render(**render_kwargs[pixel_key])[::-1, :, :]

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float('inf'), float('inf'))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys
        self.buttons = None
        self.COLOR_BUTTON = np.array([1, .5, 0, 1])
        self.COLOR_GOAL = np.array([0, 1, 0, 1])

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(observation)(observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = observation
        if self.task == "button":
            if self.buttons is None:
                self.buttons = [i for i, name in enumerate(self.env.unwrapped.sim.model.geom_names) if name.startswith("button")]
            for j, button in enumerate(self.buttons):
                if j == self.env.unwrapped.goal_button:
                    self.env.unwrapped.sim.model.geom_rgba[button] = self.COLOR_GOAL
                else:   
                    self.env.unwrapped.sim.model.geom_rgba[button] = self.COLOR_BUTTON
        pixel_observations = {
            pixel_key: self.env.sim.render(**self._render_kwargs[pixel_key])[::-1, :, :]
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)

        return observation
    
class GoalConditionedWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

# Pixel observation wrapper based on OpenAI Gym implementation.
# The MIT License

# Copyright (c) 2016 OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


    def __init__(self,
                 env):
        """Initializes a new goal conditioned Wrapper.

        Args:
            env: The environment to wrap.
        """

        super(GoalConditionedWrapper, self).__init__(env)

        wrapped_observation_space = env.observation_space

        self.observation_space = spaces.Dict()
        gc_spaces = {"observation": spaces.Box(
                                    shape=(30,), 
                                    low=-np.inf, high=np.inf,  
                                    dtype=wrapped_observation_space.dtype),
                     "desired_goal": spaces.Box(
                                    shape=(2,), 
                                    low=-np.inf, high=np.inf,  
                                    dtype=wrapped_observation_space.dtype),
                     "achieved_goal": spaces.Box(
                                    shape=(2,), 
                                    low=-np.inf, high=np.inf, 
                                    dtype=wrapped_observation_space.dtype)}
        
        self.observation_space.spaces.update(gc_spaces)
        self._env = env

    def observation(self, observation):
        # observation_keys = {('accelerometer', Box(3,)), 
        #                     ('velocimeter', Box(3,)), 
        #                     ('gyro', Box(3,)), 
        #                     ('magnetometer', Box(3,)), 
        #                     ('goal_lidar', Box(16,)), 
        #                     ('hazards_lidar', Box(16,))}   
        agent_xy = np.array(self._env.env.robot_pos)[:2]
        accelerometer = observation[:3]
        velocimeter = observation[3:6]
        gyro = observation[6:9]
        magnetometer = observation[9:12]
        # goal_lidar = observation[12:28]
        hazards_lidar = observation[28:44]
        new_vec_observation = np.concatenate([agent_xy,
                                              accelerometer,
                                              velocimeter,
                                              gyro,
                                              magnetometer,
                                              hazards_lidar]) # (30,)
        
        agent_goal_xy = np.array(self._env.env.goal_pos)[:2]
        # test boundary
        #assert -1 <= agent_xy[0] <= 1
        #assert -1 <= agent_xy[1] <= 1
        #assert -1 <= agent_goal_xy[0] <= 1
        #assert -1 <= agent_goal_xy[1] <= 1
        gc_observation = {"observation": new_vec_observation, 
                          "desired_goal": agent_goal_xy,
                          "achieved_goal": agent_xy}

        return gc_observation

    
class SafetyEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def seed(self, seed):
        return self.env.seed(seed)
    
    @property
    def goal_size(self):
        return self.env.goal_size

    @property
    def hazards_size(self):
        return self.env.hazards_size
    
    @property
    def hazards_pos(self):
        return self.env.hazards_pos

    def cost_func(self, state, hazard_poses=None):
        # test add hazard_poses process
        current_hazards = [hazard[:2] for hazard in self.hazards_pos]

        if len(state.shape) == 1:
            current_hazards_tensor = np.array(current_hazards)
            distances = np.sqrt(np.sum((state[:2] - current_hazards_tensor) ** 2, axis=1))
            #distances = torch.sqrt(torch.sum((state[:2] - current_hazards_tensor) ** 2, axis=1))
            cost = 1 if np.sum(distances < self.hazards_size) > 0 else 0
        else:
            batch_size = state.shape[0]
            device = state.device
            current_hazards_tensor = torch.tensor(current_hazards, dtype=torch.float).to(device)
            cost = torch.zeros(batch_size, dtype=torch.float)

            for i in range(batch_size):
                distances = torch.sqrt(torch.sum((state[i, :2] - current_hazards_tensor) ** 2, axis=1))
                cost[i] = 1 if torch.sum(distances < self.hazards_size) > 0 else 0

        return cost
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        new_obs, reward, done, info = self.env.step(action)
        state = new_obs["observation"]
        info["safety_cost"] = self.cost_func(state)

        return new_obs, reward, done, info

gym.logger.set_level(40)

def make_safety(domain_name, image_size, use_pixels=True, action_repeat=1, goal_conditioned=False):
    env = gym.make(
        domain_name, 
    )

    env.reset()
    env._max_episode_steps = env.config["num_steps"]
    ar_env = ActionRepeatWrapper(env, repeat=action_repeat)
    if not use_pixels:
        if goal_conditioned:
            gc_env = GoalConditionedWrapper(ar_env)
            s_env = SafetyEnvWrapper(gc_env)
            return s_env 
        return ar_env


    # fixednear, fixedfar, vision, track
    wrapped = PixelObservationWrapper(ar_env, render_kwargs={'pixels': {'camera_name': "vision", 'mode': 'rgb_array', 'width':image_size,'height':image_size}})
    wrapped.reset()
    
    wrapped.observation_space = wrapped.observation_space.spaces["pixels"]
    filtered = TransformObservation(wrapped, lambda x: np.moveaxis(x["pixels"], -1, 0))
    w_o = wrapped.observation_space
    filtered.observation_space = Box(w_o.low.min(), 
                                    w_o.high.max(), 
                                    (w_o.shape[2],w_o.shape[0],w_o.shape[1]))
    filtered._max_episode_steps = ar_env._max_episode_steps
    
    return filtered
