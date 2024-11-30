import numpy as np
import gym
from gym import spaces
import bullet_safety_gym

class GCBulletCarRun:
    def __init__(self):
        registered_envs = gym.envs.registry.all()

        self.env = gym.make('SafetyCarRun-v0')

        obs_shape = self.env.observation_space.shape[0]
        wrapped_observation_space = self.env.observation_space
        self.observation_space = spaces.Dict()
        gc_spaces = {"observation": spaces.Box(
                                    shape=(obs_shape,), 
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

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        x = self.env.reset()
        agent_velocity_goal = np.array([self.env.agent.velocity_constraint])
        agent_cur_velocity = x[2]

        gc_observation = {"observation": x, 
                          "desired_goal": agent_velocity_goal,
                          "achieved_goal": agent_cur_velocity}
        return gc_observation

    def step(self, action):
        x, reward, done, info = self.env.step(action)
        info["safety_cost"] = info["cost"]
        agent_velocity_goal = np.array([self.env.agent.velocity_constraint])
        agent_cur_velocity = x[2]

        gc_observation = {"observation": x, 
                          "desired_goal": agent_velocity_goal,
                          "achieved_goal": agent_cur_velocity}


        return gc_observation, reward, done, info