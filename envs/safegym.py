import collections
import copy

import gym
from gym.wrappers.transform_observation import TransformObservation
from gym.spaces.box import Box
import numpy as np
import safety_gym
from gym import Wrapper, spaces
from gym import ObservationWrapper


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


class RenderWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self._render_kwargs = {'camera_name': "track", 'mode': 'offscreen', 'width': 64,'height': 64}
        self.env.unwrapped.sim.render(**self._render_kwargs)[::-1, :, :]
    
    def step(self, action):
        self.env.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1
        return self.env.step(action)
    
    def reset(self):
        self.env.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1
        obs = self.env.reset()
        self.env.unwrapped.sim.render(**self._render_kwargs)[::-1, :, :]
        self.env.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1
        return obs
    
    def render_track(self):
        return self.env.unwrapped.sim.render(**self._render_kwargs)[::-1, :, :]

 
class GoalConditionedWrapper(ObservationWrapper):

    def __init__(self, env):
        super(GoalConditionedWrapper, self).__init__(env)        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float)
        self._env = env
        self._max_episode_steps = env._max_episode_steps

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
        gc_observation = np.concatenate([new_vec_observation, agent_goal_xy])
        return gc_observation


gym.logger.set_level(40)

def make_safety(domain_name, image_size, use_pixels=True, action_repeat=1, goal_conditioned=False, eval=False):
    env = gym.make(
        domain_name, 
    )

    env.reset()
    env._max_episode_steps = env.config["num_steps"]
    ar_env = ActionRepeatWrapper(env, repeat=action_repeat)
    if not use_pixels:
        if goal_conditioned:
            gc_env = GoalConditionedWrapper(ar_env)
            if eval:
                gc_env = RenderWrapper(gc_env)
            return gc_env
        if eval:
            ar_env = RenderWrapper(ar_env) 
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
    
    return RenderWrapper(filtered)