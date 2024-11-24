import random
import argparse
from collections import deque

import numpy as np
from gym import spaces

import envs.create_maze_env

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
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

def get_goal_sample_fn(env_name, evaluate, maze_id=None, goal_xy=None):
    # test
    if not(goal_xy is None):
        return lambda: np.array([goal_xy[0], goal_xy[1]])
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
        assert (not env_name == 'AntMaze') or (env_name == 'AntMaze' and (maze_id == "Maze" or \
                                                maze_id == "MazeSafe_map_1" or \
                                                maze_id == "MazeSafe_map_2" or \
                                                maze_id == "MazeSafe_map_3"))
        assert (not env_name == 'MazeSparse') or (env_name == 'MazeSparse' and maze_id == "Maze2")
        assert (not env_name == 'Push') or (env_name == 'Push' and maze_id == "Push")
        assert (not env_name == 'Fall') or (env_name == 'Fall' and maze_id == "Fall")

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self, validate=False, start_point=None, goal_xy=None):
        # self.viewer_setup()
        self.early_stop_flag = False
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate, maze_id=self.maze_id, goal_xy=goal_xy)
        obs = self.base_env.reset(validate=validate, start_point=start_point)
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
        self.render_info = {}
        if self.env.maze_id == "MazeSafe_map_1":
            SHIFT_X, SHIFT_Y = -8, -8
        elif self.env.maze_id == "MazeSafe_map_2":
            SHIFT_X, SHIFT_Y = -12, -12
        elif self.env.maze_id == "MazeSafe_map_3":
            SHIFT_X, SHIFT_Y = -12, -12
        self.render_info["shift_x"] = SHIFT_X
        self.render_info["shift_y"] = SHIFT_Y
        self.train_random_start_pose = False
        self.set_start_pose(random_start_pose=False)

    def set_train_start_pose_to_random(self):
        self.train_random_start_pose = True
        self.set_start_pose(random_start_pose=True)

    def seed(self, seed):
        self.env.seed(seed)

    @property
    def evaluate(self):
        return self.env.evaluate
    
    @evaluate.setter
    def evaluate(self, val):
        self.env.evaluate = val
        if self.train_random_start_pose:
            if val == True:
                self.set_start_pose(random_start_pose=False)
            else:
                self.set_start_pose(random_start_pose=True)

    @property
    def action_space(self):
        return self.env.action_space
    
    def set_state_dim(self, x):
        self.state_dim = x
    
    def set_goal_dim(self, x):
        self.goal_dim = x
    
    def get_maze(self):
        return self.env.get_maze()
    
    def success_fn(self, reward):
        return self.env.success_fn(reward)
        
    def get_maze(self):
        return self.env.get_maze()
    
    def set_start_pose(self, random_start_pose=False):
        if random_start_pose:
            self.random_start_pose = True
        else:
            self.random_start_pose = False

    def reset(self, xy=None, goal_xy=None, eval_idx=None):
        if self.random_start_pose:
            if self.env.maze_id == "MazeSafe_map_1" or self.env.maze_id == "MazeSafe_map_2" or \
                                    self.env.maze_id == "MazeSafe_map_3":
                if self.env.maze_id == "MazeSafe_map_1":
                    low_val = 0
                    high_val = 16
                elif self.env.maze_id == "MazeSafe_map_2":
                    low_val = 0
                    high_val = 32
                elif self.env.maze_id == "MazeSafe_map_3":
                    low_val = 0
                    high_val = 32
                safe_start_point_found = False
                n_points = 1000
                while not safe_start_point_found:
                    x = np.random.uniform(low_val, high_val, n_points)
                    y = np.random.uniform(low_val, high_val, n_points)
                    points = np.column_stack((x, y))
                    cost_idx = self.cost_func(points)
                    safety_states = (1 - cost_idx) == True
                    if safety_states.any():
                        safe_ind = np.where(safety_states)[0][0]
                        safe_start_point_found = True
                xy = tuple(points[safe_ind])
            return self.env.reset(start_point=xy, goal_xy=goal_xy)
        else:
            # test
            eval_dataset = self.get_eval_dataset()
            if not(eval_idx is None):
                if len(eval_dataset) == 0:
                    print("!!!!!! eval dataset size 0 !!!!!!!!!")
                else:
                    xy, goal_xy = eval_dataset[eval_idx%len(eval_dataset)]
            else:
                eval_idx = random.randint(0, len(eval_dataset)-1)
                xy, goal_xy = eval_dataset[eval_idx]
            return self.env.reset(start_point=xy, goal_xy=goal_xy)

    def step(self, action):
        next_tup, rew, done, info = self.env.step(action)
        safety_bounds = self.get_safety_bounds()
        info["safety_cost"] = self.cost_func(np.array(next_tup['achieved_goal']))

        return next_tup, rew, done, info
    

    def cost_func(self, state):
        if self.env.maze_id == "MazeSafe_map_1" or self.env.maze_id == "MazeSafe_map_2" \
                    or self.env.maze_id == "MazeSafe_map_3":
            if len(state.shape) == 1:
                robot_x, robot_y = state[:2]
                cost = 0
                if self.env.maze_id == "MazeSafe_map_3":
                    if robot_x <= self.safety_bounds[3-1].x or robot_x >= self.safety_bounds[2-1].x:
                        cost = 1
                    elif robot_y <= self.safety_bounds[3-1].y or robot_y >= self.safety_bounds[9-1].y:
                        cost = 1
                    elif robot_y <= self.safety_bounds[7-1].y and robot_y >= self.safety_bounds[4-1].y:
                        if robot_x <= self.safety_bounds[5-1].x:
                            cost = 1
                    elif robot_y <= self.safety_bounds[11-1].y and robot_y >= self.safety_bounds[12-1].y:
                        if robot_x >= self.safety_bounds[11-1].x:
                            cost = 1
                else:
                    if robot_x <= self.safety_bounds[3-1].x or robot_x >= self.safety_bounds[2-1].x:
                        cost = 1
                    elif robot_y <= self.safety_bounds[3-1].y or robot_y >= self.safety_bounds[1-1].y:
                        cost = 1
                    elif robot_y <= self.safety_bounds[7-1].y and robot_y >= self.safety_bounds[4-1].y:
                        if robot_x <= self.safety_bounds[5-1].x:
                            cost = 1
                    elif self.env.maze_id == "MazeSafe_map_2":
                        if robot_y <= self.safety_bounds[11-1].y and robot_y >= self.safety_bounds[8-1].y:
                            if robot_x <= self.safety_bounds[10-1].x:
                                cost = 1
            else:
                robot_x = state[:, 0]
                robot_y = state[:, 1]
                if self.env.maze_id == "MazeSafe_map_3":
                    cost = (robot_x <= self.safety_bounds[3-1].x) + (robot_x >= self.safety_bounds[2-1].x)
                    cost = cost + (robot_y <= self.safety_bounds[3-1].y) + (robot_y >= self.safety_bounds[9-1].y)
                    cost = cost + (robot_y <= self.safety_bounds[7-1].y) * (robot_y >= self.safety_bounds[4-1].y) * (robot_x <= self.safety_bounds[5-1].x)
                    cost = cost + (robot_y <= self.safety_bounds[11-1].y) * (robot_y >= self.safety_bounds[12-1].y) * (robot_x >= self.safety_bounds[11-1].x)
                else:
                    cost = (robot_x <= self.safety_bounds[3-1].x) + (robot_x >= self.safety_bounds[2-1].x)
                    cost = cost + (robot_y <= self.safety_bounds[3-1].y) + (robot_y >= self.safety_bounds[1-1].y)
                    cost = cost + (robot_y <= self.safety_bounds[7-1].y) * (robot_y >= self.safety_bounds[4-1].y) * (robot_x <= self.safety_bounds[5-1].x)
                    if self.env.maze_id == "MazeSafe_map_2":
                        cost = cost + (robot_y <= self.safety_bounds[11-1].y) * (robot_y >= self.safety_bounds[8-1].y) * (robot_x <= self.safety_bounds[10-1].x)
                cost = (cost >= 1)
        else:
            assert 1 == 0
            
        return cost


    def get_eval_dataset(self):
        # return: 
        # [
        #   [(start_x_1, start_y_1), (goal_x_1, goal_y_1)],
        #   [(start_x_2, start_y_2), (goal_x_2, goal_y_2)],
        #    ...         
        # ]
        def get_patern_dataset(left_down_x, left_down_y, right_up_x, right_up_y):
            data = []
            # 1
            xy = (left_down_x, left_down_y)
            goal_xy = (left_down_x, right_up_y)
            dataset.append([xy, goal_xy])
            # 2
            xy = (left_down_x, right_up_y)
            goal_xy = (left_down_x, left_down_y)
            dataset.append([xy, goal_xy])
            # 3
            xy = (left_down_x, left_down_y)
            goal_xy = (right_up_x, right_up_y)
            dataset.append([xy, goal_xy])
            # 4
            xy = (right_up_x, right_up_y)
            goal_xy = (left_down_x, left_down_y)
            dataset.append([xy, goal_xy])
            # 5
            xy = (left_down_x, right_up_y)
            goal_xy = (right_up_x, left_down_y)
            dataset.append([xy, goal_xy])
            # 6
            xy = (right_up_x, left_down_y)
            goal_xy = (left_down_x, right_up_y)
            dataset.append([xy, goal_xy])

            return data

        dataset = []
        if self.env.maze_id == "MazeSafe_map_1":
            left_down_x, left_down_y, right_up_x, right_up_y = 0, 0, 16, 16
            dataset.extend(get_patern_dataset(left_down_x, left_down_y, right_up_x, right_up_y))
        elif self.env.maze_id == "MazeSafe_map_2" or self.env.maze_id == "MazeSafe_map_3":
            # 1 hight - 2 hight
            left_down_x, left_down_y, right_up_x, right_up_y = 0, 0, 16, 16
            dataset.extend(get_patern_dataset(left_down_x, left_down_y, right_up_x, right_up_y))
            # 1 hight - 3 hight
            left_down_x, left_down_y, right_up_x, right_up_y = 0, 0, 32, 32
            dataset.extend(get_patern_dataset(left_down_x, left_down_y, right_up_x, right_up_y))
            # 2 hight - 3 hight
            left_down_x, left_down_y, right_up_x, right_up_y = 16, 16, 32, 32
            dataset.extend(get_patern_dataset(left_down_x, left_down_y, right_up_x, right_up_y))

        return dataset

    def get_reward_cost(self, robot_pos, goal_pos, dist_xy=None):
        # get cost
        safety_cost = self.cost_func(np.array(robot_pos))
        
        # get reward
        reward = self.env.reward_fn(robot_pos, goal_pos)

        dist_goal = dist_xy(robot_pos, goal_pos)

        goal_flag = self.env.success_fn(reward)

        return reward, safety_cost, goal_flag
    
    
    def get_safety_bounds(self, get_safe_unsafe_dataset=False):

        def extrapolate_points(l):
            extrapolated_points = []
            for i in range(len(l) - 1):
                x1, y1 = l[i]
                x2, y2 = l[i + 1]
                dx = (x2 - x1) / 5
                dy = (y2 - y1) / 5
                extrapolated_points.append((x1, y1))
                for j in range(1, 5):
                    extrapolated_points.append((x1 + j * dx, y1 + j * dy))
            return extrapolated_points

        # dataset = (
            #             [(x11, x12), (x21, x22), ... ], 
            #             [y1, y2, ... ]
            #           )
        if self.env.maze_id == "MazeSafe_map_1":
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
            
            xs = []
            ys = []
            # usafe states
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs.append((safety_point_5.x - 1, safety_point_5.y + 1))
            xs.append((safety_point_6.x - 1, safety_point_6.y - 1))
            xs.append((safety_point_7.x - 1, safety_point_7.y - 1))
            xs.append((safety_point_8.x - 1, safety_point_8.y + 1))
            xs.append((safety_point_9.x + 1, safety_point_9.y + 1))
            xs.append((safety_point_2.x + 1, safety_point_2.y - 1))
            xs.append((safety_point_3.x - 1, safety_point_3.y - 1))
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs = extrapolate_points(xs)
            for i in range(len(xs)):
                ys.append(1)
            num_unsafe_states = len(xs)

            # safe states
            xs_safe = []
            xs_safe.append((safety_point_3.x + 1, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe.append((safety_point_7.x + 1, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe = extrapolate_points(xs_safe)
            xs.extend(xs_safe)
            for i in range(len(xs_safe)):
                ys.append(0)
            dataset = [xs, ys]

            safety_boundary = [safety_point_1,
                            safety_point_2, safety_point_3, 
                            safety_point_4, safety_point_5, 
                            safety_point_6, safety_point_7,
                            safety_point_8, safety_point_9]

        elif self.env.maze_id == "MazeSafe_map_2":
            """
            12-----------------1=13
            |                    |
            |                    |
            11------------10     |
                           |     |
                           |     |
                           |     |
            8--------------9     |
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
            safety_point_13 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 + 34)
            safety_point_12 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 34)
            safety_point_11 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 30)
            safety_point_10 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 30)
            safety_point_9 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 18)
            safety_point_8 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 18)
            safety_point_7 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 14)
            safety_point_6 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 14)
            safety_point_5 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 2)
            safety_point_4 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 2)
            safety_point_3 = Point(0.03229626534308827 - 2, -0.06590457330324587 - 2)
            safety_point_2 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 - 2)
            safety_point_1 = safety_point_13

            xs = []
            ys = []
            # usafe states
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs.append((safety_point_5.x - 1, safety_point_5.y + 1))
            xs.append((safety_point_6.x - 1, safety_point_6.y - 1))
            xs.append((safety_point_7.x - 1, safety_point_7.y - 1))

            xs.append((safety_point_8.x - 1, safety_point_8.y + 1))
            xs.append((safety_point_9.x - 1, safety_point_9.y + 1))
            xs.append((safety_point_10.x - 1, safety_point_10.y - 1))
            xs.append((safety_point_11.x - 1, safety_point_11.y - 1))

            xs.append((safety_point_12.x - 1, safety_point_12.y + 1))
            xs.append((safety_point_13.x + 1, safety_point_13.y + 1))

            xs.append((safety_point_2.x + 1, safety_point_2.y - 1))
            xs.append((safety_point_3.x - 1, safety_point_3.y - 1))
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs = extrapolate_points(xs)
            for i in range(len(xs)):
                ys.append(1)
            num_unsafe_states = len(xs)

            # safe states
            # test
            xs_safe = []
            xs_safe.append((safety_point_3.x + 1, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe.append((safety_point_7.x + 1, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe = extrapolate_points(xs_safe)
            xs.extend(xs_safe)
            for i in range(len(xs_safe)):
                ys.append(0)
            dataset = [xs, ys]

            safety_boundary = [safety_point_1,
                            safety_point_2, safety_point_3, 
                            safety_point_4, safety_point_5, 
                            safety_point_6, safety_point_7,
                            safety_point_8, safety_point_9,
                            safety_point_10, safety_point_11,
                            safety_point_12, safety_point_13]

        elif self.env.maze_id == "MazeSafe_map_3":
            """
            8--------------------9
            |                    |
            |                    |
            |           11------10
            |           |         
            |           |         
            |           |         
            |           12-----13=1     
            |                    |
            |                    |
            7--------6           |
                     |           |
                     |           |
                     |           |
            4--------5           |
            |                    |
            |                    |
            3--------------------2
            """
            safety_point_13 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 + 18)
            safety_point_12 = Point(0.03229626534308827 + 1.5, -0.06590457330324587 + 18)
            safety_point_11 = Point(0.03229626534308827 + 1.5, -0.06590457330324587 + 30)
            safety_point_10 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 + 30)
            safety_point_9 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 + 34)
            safety_point_8 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 34)
            safety_point_7 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 14)
            safety_point_6 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 14)
            safety_point_5 = Point(0.03229626534308827 + 14, -0.06590457330324587 + 2)
            safety_point_4 = Point(0.03229626534308827 - 2, -0.06590457330324587 + 2)
            safety_point_3 = Point(0.03229626534308827 - 2, -0.06590457330324587 - 2)
            safety_point_2 = Point(0.03229626534308827 + 17.5, -0.06590457330324587 - 2)
            safety_point_1 = safety_point_13

            xs = []
            ys = []
            # usafe states
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs.append((safety_point_5.x - 1, safety_point_5.y + 1))
            xs.append((safety_point_6.x - 1, safety_point_6.y - 1))
            xs.append((safety_point_7.x - 1, safety_point_7.y - 1))

            xs.append((safety_point_8.x - 1, safety_point_8.y + 1))
            xs.append((safety_point_9.x - 1, safety_point_9.y + 1))
            xs.append((safety_point_10.x - 1, safety_point_10.y - 1))
            xs.append((safety_point_11.x - 1, safety_point_11.y - 1))

            xs.append((safety_point_12.x - 1, safety_point_12.y + 1))
            xs.append((safety_point_13.x + 1, safety_point_13.y + 1))

            xs.append((safety_point_2.x + 1, safety_point_2.y - 1))
            xs.append((safety_point_3.x - 1, safety_point_3.y - 1))
            xs.append((safety_point_4.x - 1, safety_point_4.y + 1))
            xs = extrapolate_points(xs)
            for i in range(len(xs)):
                ys.append(1)
            num_unsafe_states = len(xs)

            # safe states
            # test
            xs_safe = []
            xs_safe.append((safety_point_3.x + 1, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_4.y + safety_point_3.y) / 2))
            xs_safe.append(((safety_point_5.x + safety_point_2.x) / 2, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe.append((safety_point_7.x + 1, (safety_point_7.y + safety_point_8.y) / 2))
            xs_safe = extrapolate_points(xs_safe)
            xs.extend(xs_safe)
            for i in range(len(xs_safe)):
                ys.append(0)
            dataset = [xs, ys]

            safety_boundary = [safety_point_1,
                            safety_point_2, safety_point_3, 
                            safety_point_4, safety_point_5, 
                            safety_point_6, safety_point_7,
                            safety_point_8, safety_point_9,
                            safety_point_10, safety_point_11,
                            safety_point_12, safety_point_13]


        else:
            assert 1 == 0

        
        if get_safe_unsafe_dataset:
            assert len(dataset[0]) == len(dataset[1])
            return safety_boundary, dataset
        else:
            return safety_boundary
