from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

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


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, args):
        self.args = args
        self.cost_func = None
        self.setted_cost_func = False
        self.num_timesteps = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.args.safe_env_hazards:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher_hazards.xml' % dir_path, 4)
        else:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def set_cost_func(self, cost_func):
        self.cost_func = cost_func
        self.setted_cost_func = True

    def step(self, a):
        self.num_timesteps += 1
        self.do_simulation(a, self.frame_skip)
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        reward_ctrl = 0.001 * -np.square(a).sum()

        fail = True
        if np.sqrt(np.sum(np.square(vec_2))) <= 0.25:
            fail = False
        ob = self._get_obs()
        if self.args.pusher_four_goal_dim:
            self.ac_goal_pos = np.concatenate((self.get_body_com("object").copy()[:2], 
                                               self.get_body_com("tips_arm").copy()[:2]))
            self.goal = np.concatenate((self.get_body_com("goal").copy()[:2], 
                                        self.get_body_com("object").copy()[:2]))
        elif self.args.pusher_three_goal_dim:
            self.ac_goal_pos = self.get_body_com("object").copy()
            self.goal = self.get_body_com("goal").copy()
        elif self.args.pusher_two_goal_dim:
            self.ac_goal_pos = self.get_body_com("object").copy()[:2]
            self.goal = self.get_body_com("goal").copy()[:2]
        else:
            self.ac_goal_pos = np.concatenate((self.get_body_com("object").copy(), self.get_body_com("tips_arm").copy()))
            self.goal = np.concatenate((self.get_body_com("goal").copy(), self.get_body_com("object").copy()))

        return ob, - float(fail) + reward_ctrl, self.num_timesteps >= 100, {'is_success': not fail}


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        if self.args.pusher_hard_task:
            self.goal_pos = np.asarray([-0.2, 0.3])
            self.cylinder_pos = np.array([-0.2, -1.1]) + np.random.normal(0, 0.025, [2])
        elif self.args.pusher_random_obj_start_poses:
            #l_u = (-0.45, -0.05)
            #l_d = (-0.45, -0.4)
            #r_d = (0.6, -0.4)
            #r_u = (0.6, -0.05)
            """
            l_d-------------l_u
            |               |
            |               |
            |               |
            r_d-------------r_u
            """

            l_u = (0.1, 0.2)
            l_d = (0.1, -0.8)
            r_d = (-0.3, -0.8)
            r_u = (-0.3, -0.2)

            x_min = min(l_u[0], l_d[0], r_d[0], r_u[0])
            x_max = max(l_u[0], l_d[0], r_d[0], r_u[0])
            y_min = min(l_u[1], l_d[1], r_d[1], r_u[1])
            y_max = max(l_u[1], l_d[1], r_d[1], r_u[1])

            def generate_random_point():
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                return (x, y)

            self.goal_pos = np.asarray(generate_random_point())
            self.cylinder_pos = np.asarray(generate_random_point())

        else:
            self.goal_pos = np.asarray([0, 0])
            self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        # testing
        #self.goal_pos = np.asarray([0, -0.6])
        #self.cylinder_pos = np.array([0, 0.15])
            
        #self.goal_pos = np.asarray([0.1, -0.8])
        #self.cylinder_pos = np.array([-0.1, 0.0])
            
        #self.goal_pos = np.asarray([-0.1, -0.8])
        #self.cylinder_pos = np.array([-0.1, 0.0])
            
        #self.goal_pos = np.asarray([-0.1, 0.0])
        #self.cylinder_pos = np.array([-0.1, -0.8])
            
        #self.goal_pos = np.asarray([-0.1, 0.0])
        #self.cylinder_pos = np.array([0.1, -0.8])

        #self.goal_pos = np.asarray([0.1, 0.0])
        #self.cylinder_pos = np.array([0.1, -0.8])
            
        #self.goal_pos = np.asarray([0.0, 0.0])
        #self.cylinder_pos = np.array([-0.1, -0.5])

        # safe task 1    
        #self.cylinder_pos = np.asarray([-0.3, 0.0])
        #self.goal_pos = np.array([0.1, -0.8])

        # safe task 1    
        #self.cylinder_pos = np.asarray([-0.3, 0.0])
        #self.goal_pos = np.array([0.4, 0.2])
            
        """
        safe zone:
            8------------------------------------------------------7
            |                                                      |
            |                                                      |
            |                                                      |
            |                                                      |
            |                    3---------------4                 |
            |                    |               |                 |
            |                    |               |                 |
            |                    |               |                 |
            |                    |               |                 |
            1--------------------2               5-----------------6
        """
        if self.args.pusher_safe_env:
            assert self.args.pusher_random_obj_start_poses
            safe_pos_1 = Point(-0.35, -0.8)
            safe_pos_2 = Point(-0.35, -0.6)
            safe_pos_3 = Point(-0.1, -0.6)
            safe_pos_4 = Point(-0.1, -0.1)
            safe_pos_5 = Point(-0.35, -0.1)
            safe_pos_6 = Point(-0.35, 0.2)
            safe_pos_7 = Point(0.1, 0.2)
            safe_pos_8 = Point(0.1, -0.8)
            safe_points = [None, safe_pos_1, safe_pos_2, safe_pos_3, 
                                safe_pos_4, safe_pos_5, safe_pos_6, 
                                safe_pos_7, safe_pos_8]
            def is_safe_state(state, safe_points):
                if state[0] < safe_points[1].x or state[0] > safe_points[8].x:
                    return False
                if state[1] < safe_points[1].y or state[1] > safe_points[7].y:
                    return False
                if state[0] < safe_points[3].x and state[1] > safe_points[3].y and state[1] < safe_points[4].y:
                    return False
                return True

            while not is_safe_state(self.cylinder_pos, safe_points) or not is_safe_state(self.goal_pos, safe_points):
                self.goal_pos = np.asarray(generate_random_point())
                self.cylinder_pos = np.asarray(generate_random_point())

        if self.args.safe_env_hazards:
            self.hazard_pos = np.asarray([0.3, 0.0])

        if self.args.safe_env_hazards:
            qpos[-6:-4] = self.hazard_pos
        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                    high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        if self.args.pusher_four_goal_dim:
            self.ac_goal_pos = np.concatenate((self.get_body_com("object").copy()[:2], 
                                               self.get_body_com("tips_arm").copy()[:2]))
            self.goal = np.concatenate((self.get_body_com("goal").copy()[:2], 
                                        self.get_body_com("object").copy()[:2]))
        elif self.args.pusher_three_goal_dim:
            self.ac_goal_pos = self.get_body_com("object").copy()
            self.goal = self.get_body_com("goal").copy()
        elif self.args.pusher_two_goal_dim:
            self.ac_goal_pos = self.get_body_com("object").copy()[:2]
            self.goal = self.get_body_com("goal").copy()[:2]
        else:
            self.ac_goal_pos = np.concatenate((self.get_body_com("object").copy(), self.get_body_com("tips_arm").copy()))
            self.goal = np.concatenate((self.get_body_com("goal").copy(), self.get_body_com("object").copy()))

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:7].copy(),
            self.data.qvel.flat[:7].copy(),
            self.get_body_com("tips_arm").copy(),
            self.get_body_com("object").copy(),
        ])

    def reset(self):
        self.num_timesteps = 0
        return super().reset()


if __name__ == '__main__':
    env = PusherEnv()
    done = False
    obs = env.reset()
    counter = 0
    import pdb;

    pdb.set_trace()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        counter += 1
        print(obs, reward, done, info)
    print(counter)