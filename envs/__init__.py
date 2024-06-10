import numpy as np
import argparse
from collections import deque
from gym import spaces

import envs.create_maze_env

SHIFT_X, SHIFT_Y = -8, -8

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    @property
    def render_x(self):
        return self._x + SHIFT_X
    @property
    def render_y(self):
        return self._y + SHIFT_Y
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @x.setter
    def x(self, x):
        self._x = x
    @y.setter
    def y(self, y):
        self._y = y

def get_goal_sample_fn(env_name, evaluate, maze_id=None):
    if env_name == 'AntMaze':
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == "AntMazeMultiMap":
        if evaluate:
            if maze_id == "Maze_map_1":
                return lambda: np.array([16., 16.])
            elif maze_id == "Maze_map_2":
                return lambda: np.array([0., 0.])
            elif maze_id == "Maze_map_3":
                return lambda: np.array([16., 0.])
            elif maze_id == "Maze_map_4":
                return lambda: np.array([0., 16.])
            else:
                assert 1 == 0
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntMazeSparse':
        return lambda: np.array([2., 9.])
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name in ['AntMaze', 'AntPush', "AntMazeMultiMap"]:
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntMazeSparse':
        return lambda obs, goal: float(np.sum(np.square(obs[:2] - goal)) ** 0.5 < 1)
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def get_success_fn(env_name):
    if env_name in ['AntMaze', 'AntPush', 'AntFall', "AntMazeMultiMap"]:
        return lambda reward: reward > -5.0
    elif env_name == 'AntMazeSparse':
        return lambda reward: reward > 1e-6
    else:
        assert False, 'Unknown env'


class GatherEnv(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        obs = self.base_env.reset()
        self.count = 0
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }

    def step(self, a):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': None,
        }
        return next_obs, reward, done or self.count >= 500, info

    def get_apples_and_bombs(self):
        return self.base_env.objects # [(x, y, type)], where type=APPLE=0, type=BOMB=1

    @property
    def action_space(self):
        return self.base_env.action_space


class EnvWithGoal(object):

    def __init__(self, base_env, env_name, maze_id=None):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = 5 if env_name in ['AntMaze', 'AntPush', 'AntFall', "AntMazeMultiMap"] else 1
        self.count = 0
        self.early_stop = False if env_name in ['AntMaze', 'AntPush', 'AntFall', "AntMazeMultiMap"] else True
        self.early_stop_flag = False
        self.maze_id = maze_id
        assert (not env_name == 'AntMaze') or (env_name == 'AntMaze' and (maze_id == "Maze" or maze_id == "MazeSafe_map_1"))
        assert (not env_name == 'MazeSparse') or (env_name == 'MazeSparse' and maze_id == "Maze2")
        assert (not env_name == 'Push') or (env_name == 'Push' and maze_id == "Push")
        assert (not env_name == 'Fall') or (env_name == 'Fall' and maze_id == "Fall")

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self, validate=False):
        # self.viewer_setup()
        self.early_stop_flag = False
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate, maze_id=self.maze_id)
        obs = self.base_env.reset(validate=validate)
        self.count = 0
        self.goal = self.goal_sample_fn()
        self.desired_goal = self.goal if self.env_name in ['AntMaze', 'AntPush', 'AntFall', "AntMazeMultiMap"] else None
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }
        return next_obs, reward, done or self.count >= 500, info
    
    def get_maze(self):
        structure = self.base_env.MAZE_STRUCTURE
        size_scaling = self.base_env.MAZE_SIZE_SCALING
        return structure

    def render(self):
        self.base_env.render()

    @property
    def action_space(self):
        return self.base_env.action_space
    

class MultyEnvWithGoal(EnvWithGoal):
    def __init__(self, envs):
        assert type(envs) == list
        self.envs = envs
        self.env = None
        self.set_validate = False

    def seed(self, seed):
        np.random.seed(seed)

    @property
    def evaluate(self):
        return self.env.evaluate
    
    @evaluate.setter
    def evaluate(self, validate):
        if validate == True:
            self.set_validate = True
        else:
            self.set_validate = False

    @property
    def action_space(self):
        return self.envs[0].action_space
        
    def success_fn(self, reward):
        return self.env.success_fn(reward)
        
    def get_maze(self):
        return self.env.get_maze()

    def reset(self, validate=False):
        self.env = np.random.choice(self.envs)
        if self.set_validate:
            self.env.evaluate = True
        else:
            self.env.evaluate = False
        return self.env.reset(validate=validate)

    def step(self, action):
        return self.env.step(action)
    

class SafeMazeAnt:
    def __init__(self, env):
        self.env = env
        self.safety_bounds = self.get_safety_bounds()

    def seed(self, seed):
        self.env.seed(seed)

    @property
    def evaluate(self):
        return self.env.evaluate
    
    @evaluate.setter
    def evaluate(self, val):
        self.env.evaluate = val

    @property
    def action_space(self):
        return self.env.action_space
    
    def get_maze(self):
        return self.env.get_maze()
    
    def success_fn(self, reward):
        return self.env.success_fn(reward)
        
    def get_maze(self):
        return self.env.get_maze()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_tup, rew, done, info = self.env.step(action)
        safety_bounds = self.get_safety_bounds()
        info["safety_cost"] = self.cost_func(np.array(next_tup['achieved_goal']))

        return next_tup, rew, done, info
    

    def cost_func(self, state):
        if len(state.shape) == 1:
            robot_x, robot_y = state[:2]
            cost = 0
            if robot_x <= self.safety_bounds[3-1].x or robot_x >= self.safety_bounds[2-1].x:
                cost = 1
            elif robot_y <= self.safety_bounds[3-1].y or robot_y >= self.safety_bounds[1-1].y:
                cost = 1
            elif robot_y <= self.safety_bounds[7-1].y and robot_y >= self.safety_bounds[4-1].y:
                if robot_x <= self.safety_bounds[5-1].x:
                    cost = 1
        else:
            robot_x = state[:, 0]
            robot_y = state[:, 1]
            cost = (robot_x <= self.safety_bounds[3-1].x) + (robot_x >= self.safety_bounds[2-1].x)
            cost = cost + (robot_y <= self.safety_bounds[3-1].y) + (robot_y >= self.safety_bounds[1-1].y)
            cost = cost + (robot_y <= self.safety_bounds[7-1].y) * (robot_y >= self.safety_bounds[4-1].y) * (robot_x <= self.safety_bounds[5-1].x)
            cost = (cost >= 1)
            #print("cost:", cost)
            #print("cost:", cost.shape)
            #cost = cost.astype(float)
            #cost = cost.to(float)
            #assert (cost <= 1.0).all()
            
        return cost
    
    
    def get_safety_bounds(self):
        """
        8-------------------1=9
        |                    |
        |                    |
        7-----------6        |
                    |        |
                    |        |
                    |        |
        4-----------5        |
        |                    |
        |                    |
        3--------------------2
        """
        safety_point_9 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 + 18)
        safety_point_8 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 18)
        safety_point_7 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 14)
        safety_point_6 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 14)
        safety_point_5 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 2)
        safety_point_4 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 2)
        safety_point_3 = Point(0.03229626534308827 - 2, -0.06590457330324587 - 2)
        safety_point_2 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 - 2)
        safety_point_1 = safety_point_9
        safety_boundary = [safety_point_1,
                           safety_point_2, safety_point_3, 
                           safety_point_4, safety_point_5, 
                           safety_point_6, safety_point_7,
                           safety_point_8, safety_point_9]
        return safety_boundary
