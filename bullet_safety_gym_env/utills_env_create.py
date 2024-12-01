import sys
import os 

import numpy as np
import matplotlib.pylab as plt

from .bullet_safety_gym_wrapper import GCBulletCarRun


class CustomVideoRendered:
    def __init__(self, env, plot_subgoal=True):
        # config
        self.add_subgoal_values = False
        self.add_mesurements = True
        self.plot_safe_dataset = False    
        self.plot_subgoal = plot_subgoal  

        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        self.env = env
        self.render_info["env_min_x"], self.render_info["env_max_x"] = -20, 20
        self.render_info["env_min_y"], self.render_info["env_max_y"] = -20, 20
        self.render_info["grid_resolution_x"] = 20
        self.render_info["grid_resolution_y"] = 20
        self.render_info["state_dim"] = env.state_dim
    
    def setup_renderer(self):
        pass
    
    def delete_data(self):
        pass

    def custom_render(self, current_step_info, positions_render=False, 
                      plot_goal=True, debug_info={}, shape=(600, 600), 
                      env_name="", safe_model=None):    
        assert "robot_pos" in current_step_info and \
               "goal_pos" in current_step_info and \
               "robot_radius" in current_step_info
        if self.plot_subgoal:
            assert "subgoal_pos" in current_step_info

        env_min_x, env_max_x = self.render_info["env_min_x"], self.render_info["env_max_x"]
        env_min_y, env_max_y = self.render_info["env_min_y"], self.render_info["env_max_y"]
        if self.render_info["fig"] is None:
            self.render_info["fig"] = plt.figure(figsize=[6.4, 4.8])
            self.render_info["ax_states"] = self.render_info["fig"].add_subplot(111)
        self.render_info["ax_states"].set_ylim(bottom=env_min_y, top=env_max_y)
        self.render_info["ax_states"].set_xlim(left=env_min_x, right=env_max_x)
        
        # robot pose
        x = current_step_info["robot_pos"][0]
        y = current_step_info["robot_pos"][1]
        circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="g", alpha=0.5)
        self.render_info["ax_states"].add_patch(circle_robot) 
        self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s")

        # subgoal
        if self.plot_subgoal:
            x = current_step_info["subgoal_pos"][0]
            y = current_step_info["subgoal_pos"][1]
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="orange", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot)
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s_g")
            if self.add_subgoal_values:
                self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_s_sg"])), debug_info["v_s_sg"])
                self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_sg_g"])), debug_info["v_sg_g"])

        # goal
        if env_name != "AntGather" and env_name != "AntMazeSparse" and plot_goal:
            x = current_step_info["goal_pos"][0]
            y = current_step_info["goal_pos"][1]
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="y", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "g")  
        """
        if self.add_mesurements: 
            assert "acc_reward" in debug_info 
            assert "acc_cost" in debug_info
            assert "t" in debug_info
            if len(debug_info) != 0:
                # main
                acc_reward = debug_info["acc_reward"]
                acc_cost = debug_info["acc_cost"]
                t = debug_info["t"]
                # option
                if "acc_controller_reward" in debug_info:
                    acc_controller_reward = debug_info["acc_controller_reward"]
                    self.render_info["ax_states"].text(env_max_x - 18.5, env_max_y - 2, f"Rc:{int(acc_controller_reward*100)/100}")
                if "dist_a_net_s_sg" in debug_info:
                    dist_a_net_s_sg = debug_info["dist_a_net_s_sg"]
                if "dist_a_net_s_g" in debug_info:
                    dist_a_net_s_g = debug_info["dist_a_net_s_g"]
                if "imagine_subgoal_safety" in debug_info:
                    imagine_subgoal_safety = debug_info["imagine_subgoal_safety"]
                    self.render_info["ax_states"].text(env_max_x - 34.5, env_max_y - 2, f"Is:{int(imagine_subgoal_safety*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 26.5, env_max_y - 2, f"Cm:{int(acc_cost*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 8.5, env_max_y - 2, f"Rm:{int(acc_reward*10)/10}")
        """

        # render img
        self.render_info["fig"].canvas.draw()
        data = np.frombuffer(self.render_info["fig"].canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.render_info["fig"].canvas.get_width_height()[::-1] + (3,))
        self.render_info["ax_states"].clear()
        return data



def create_bullet_safety_gym_env(args, renderer_args={}):

    goal_dim = 4
    subgoal_dim = 4
    if args.env_name == "SafeBulletCarRun":
        env = GCBulletCarRun(goal_dim)
    else:
        assert 1 == 0

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space["observation"].shape[0]
    renderer = None
    env.max_len = 500
    env.state_dim = state_dim

    renderer = CustomVideoRendered(env, **renderer_args)
    
    return env, state_dim, goal_dim, subgoal_dim, action_dim, renderer
