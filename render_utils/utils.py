import numpy as np
import matplotlib.pylab as plt

from .plots import plot_values


class CustomVideoRendered:
    def __init__(self, env, args, controller_safe_model=False, 
                 world_model_comparsion=False, 
                 plot_world_model_state=False,
                 plot_subgoal=False, 
                 plot_safety_boundary=False,
                 plot_cost_model_heatmap=False):
        # config
        self.add_subgoal_values = False
        self.add_mesurements = True
        self.plot_safe_dataset = False    
        self.plot_subgoal = plot_subgoal  
        self.plot_safety_boundary = plot_safety_boundary  
        self.plot_world_model_state = plot_world_model_state
        self.plot_cost_model_heatmap = plot_cost_model_heatmap

        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        self.env = env
        self.world_model_comparsion = world_model_comparsion
        self.controller_safe_model = controller_safe_model
        self.render_info["env_min_x"], self.render_info["env_max_x"] = -7, 7
        self.render_info["env_min_y"], self.render_info["env_max_y"] = -7, 7
        self.render_info["grid_resolution_x"] = 40
        self.render_info["grid_resolution_y"] = 40
        self.render_info["state_dim"] = env.state_dim
        if self.plot_world_model_state or self.world_model_comparsion:
            self.robot_poses = None
            self.world_model_poses = None
        if self.add_subgoal_values:
            assert 1 == 0, "didnt implement"
    
    def setup_renderer(self):
        if self.plot_world_model_state or self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            self.robot_poses = []
            self.world_model_poses = []
    
    def delete_data(self):
        if self.plot_world_model_state or self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            del self.robot_poses
            del self.world_model_poses

    def custom_render(self, current_step_info, positions_render=False, 
                      plot_goal=True, debug_info={}, shape=(600, 600), 
                      env_name="", safe_model=None):    
        assert "robot_pos" in current_step_info and \
               "goal_pos" in current_step_info and \
               "robot_radius" in current_step_info and \
               "hazards" in current_step_info and \
               "hazards_radius" in current_step_info
               
        if self.plot_subgoal:
            assert "subgoal_pos" in current_step_info
        if self.plot_world_model_state:
            assert "imagined_robot_pos" in current_step_info

        if env_name == "SafeAntMaze":
            safety_boundary, safe_dataset = self.env.get_safety_bounds(get_safe_unsafe_dataset=True)
            debug_info["safety_boundary"] = safety_boundary
            debug_info["safe_dataset"] = safe_dataset

        env_min_x, env_max_x = self.render_info["env_min_x"], self.render_info["env_max_x"]
        env_min_y, env_max_y = self.render_info["env_min_y"], self.render_info["env_max_y"]
        if self.render_info["fig"] is None:
            if self.add_subgoal_values:
                self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                self.render_info["ax_subgoal_values"] = self.render_info["fig"].add_subplot(122)
            elif self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
                self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                self.render_info["ax_world_model_robot_trajectories"] = self.render_info["fig"].add_subplot(122)
            else:
                self.render_info["fig"] = plt.figure(figsize=[6.4, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(111)
        self.render_info["ax_states"].set_ylim(bottom=env_min_y, top=env_max_y)
        self.render_info["ax_states"].set_xlim(left=env_min_x, right=env_max_x)
        if self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            self.render_info["ax_world_model_robot_trajectories"].set_ylim(bottom=env_min_y, top=env_max_y)
            self.render_info["ax_world_model_robot_trajectories"].set_xlim(left=env_min_x, right=env_max_x)

        # robot pose
        x = current_step_info["robot_pos"][0]
        y = current_step_info["robot_pos"][1]
        circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="g", alpha=0.5)
        self.render_info["ax_states"].add_patch(circle_robot) 
        self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s")
        # world model comparsion
        if self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            self.robot_poses.append((x, y))   

        # robot imagined pose
        if self.plot_world_model_state or self.world_model_comparsion:
            x = current_step_info["imagined_robot_pos"][0]
            y = current_step_info["imagined_robot_pos"][1]
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"] / 2, color="r", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x, y + 0.05, "i_s")
            self.world_model_poses.append((x, y))  

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

        # world model comparsion
        if self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            #xA, yA = zip(*self.robot_poses)
            #self.render_info["ax_world_model_robot_trajectories"].plot(xA, yA, 'g', label='robot poses')
            x = current_step_info["robot_pos"][0]
            y = current_step_info["robot_pos"][1]
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="r", alpha=1, fill=False)
            self.render_info["ax_world_model_robot_trajectories"].add_patch(circle_robot) 
            if env_name == "SafeGym":
                for hazard in current_step_info["hazards"]:
                    x = hazard[0]
                    y = hazard[1]
                    circle_robot = plt.Circle((x, y), radius=current_step_info["hazards_radius"], color="b", alpha=1, fill=False)
                    self.render_info["ax_world_model_robot_trajectories"].add_patch(circle_robot) 
            if self.world_model_comparsion:
                xB, yB = zip(*self.world_model_poses)
                self.render_info["ax_world_model_robot_trajectories"].plot(xB, yB, 'r', label='wm poses')
                xA, yA = zip(*self.robot_poses)
                self.render_info["ax_world_model_robot_trajectories"].plot(xA, yA, 'g', label='robot poses')

        if self.plot_cost_model_heatmap and self.controller_safe_model:
            assert "agent_full_obs" in current_step_info
            cb = plot_values(self.render_info["fig"], 
                        self.render_info["ax_world_model_robot_trajectories"], 
                        safe_model, render_info=self.render_info, 
                        current_step_info=current_step_info,
                        return_cb=True)

        # safety boundary
        if self.plot_safety_boundary:
            safety_boundary = debug_info["safety_boundary"]
            if (self.controller_safe_model and self.plot_cost_model_heatmap):
                safe_dataset = debug_info["safe_dataset"]
            xs = [point.render_x for point in safety_boundary]
            ys = [point.render_y for point in safety_boundary]
            self.render_info["ax_states"].plot(xs, ys, 'b')
            if self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
                xs = [point.x for point in safety_boundary]
                ys = [point.y for point in safety_boundary]
                self.render_info["ax_world_model_robot_trajectories"].plot(xs, ys, 'b')
                # safe dataset check
                if (self.controller_safe_model and self.plot_cost_model_heatmap) and self.plot_safe_dataset:
                    xs_dataset = safe_dataset[0]
                    ys_dataset = safe_dataset[1]
                    x1s_unsafe = []
                    x2s_unsafe = []
                    x1s_safe = []
                    x2s_safe = []
                    for i in range(len(ys_dataset)):
                        if ys_dataset[i] == 1:
                            x1s_unsafe.append(xs_dataset[i][0])
                            x2s_unsafe.append(xs_dataset[i][1])
                        else:
                            x1s_safe.append(xs_dataset[i][0])
                            x2s_safe.append(xs_dataset[i][1])
                    self.render_info["ax_world_model_robot_trajectories"].plot(x1s_unsafe, x2s_unsafe, 'r')
                    self.render_info["ax_world_model_robot_trajectories"].plot(x1s_safe, x2s_safe, 'g')

        if env_name == "SafeGym":
            for hazard in current_step_info["hazards"]:
                x = hazard[0]
                y = hazard[1]
                circle_robot = plt.Circle((x, y), radius=current_step_info["hazards_radius"], color="b", alpha=0.5)
                self.render_info["ax_states"].add_patch(circle_robot) 
        
        if self.add_mesurements: 
            assert "acc_reward" in debug_info 
            assert "acc_cost" in debug_info
            assert "t" in debug_info
            if len(debug_info) != 0:
                # main
                acc_reward = debug_info["acc_reward"]
                acc_cost = debug_info["acc_cost"]
                t = debug_info["t"]
                self.render_info["ax_states"].text(env_max_x - 9.5, env_max_y - 1, f"t:{t}")
                # option
                if "acc_controller_reward" in debug_info:
                    acc_controller_reward = debug_info["acc_controller_reward"]
                    self.render_info["ax_states"].text(env_max_x - 18.5, env_max_y - 2, f"Rc:{int(acc_controller_reward*100)/100}")
                if "dist_a_net_s_sg" in debug_info:
                    dist_a_net_s_sg = debug_info["dist_a_net_s_sg"]
                if "dist_a_net_s_g" in debug_info:
                    dist_a_net_s_g = debug_info["dist_a_net_s_g"]
                if "dist_to_goal" in debug_info:
                    dist_to_goal = debug_info["dist_to_goal"]
                    self.render_info["ax_states"].text(env_max_x - 7.5, env_max_y - 1, f"d_g:{int(dist_to_goal*100)/100}")
                if "goals_achieved" in debug_info:
                    goals_achieved = debug_info["goals_achieved"]
                    self.render_info["ax_states"].text(env_max_x - 5.5, env_max_y - 1, f"g_a:{goals_achieved}")

                if "imagine_subgoal_safety" in debug_info:
                    imagine_subgoal_safety = debug_info["imagine_subgoal_safety"]
                    self.render_info["ax_states"].text(env_max_x - 5, env_max_y - 1, f"Is:{int(imagine_subgoal_safety*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 3.5, env_max_y - 1, f"Cm:{int(acc_cost*100)/100}")
                self.render_info["ax_states"].text(env_max_x - 1.5, env_max_y - 1, f"Rm:{int(acc_reward*10)/10}")

        # render img
        self.render_info["fig"].canvas.draw()
        data = np.frombuffer(self.render_info["fig"].canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.render_info["fig"].canvas.get_width_height()[::-1] + (3,))
        self.render_info["ax_states"].clear()
        if self.world_model_comparsion or (self.controller_safe_model and self.plot_cost_model_heatmap):
            if self.plot_cost_model_heatmap and (self.controller_safe_model and self.plot_cost_model_heatmap):
                cb.remove()
            self.render_info["ax_world_model_robot_trajectories"].clear()
        if self.add_subgoal_values:
            self.render_info["ax_subgoal_values"].clear()
        return data
    


def get_renderer(env, args, renderer_args):
    renderer = CustomVideoRendered(env,  
                                   args,
                                   **renderer_args)
    
    return renderer