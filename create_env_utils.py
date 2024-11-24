import numpy as np
import matplotlib.pylab as plt

from envs import EnvWithGoal, GatherEnv, MultyEnvWithGoal, SafeMazeAnt
from envs.create_gather_env import create_gather_env
from envs.create_maze_env import create_maze_env
from envs.plots import plot_values


class CustomVideoRendered:
    def __init__(self, env, controller_safe_model=False, 
                 world_model_comparsion=True, plot_subgoal=True, 
                 plot_safety_boundary=True):
        # config
        self.add_subgoal_values = False
        self.add_mesurements = True
        self.plot_safe_dataset = False    
        self.plot_subgoal = plot_subgoal  
        self.plot_safety_boundary = plot_safety_boundary  

        self.render_info = {}
        self.render_info["fig"] = None
        self.render_info["ax_states"] = None
        self.env = env
        self.world_model_comparsion = world_model_comparsion
        self.controller_safe_model = controller_safe_model
        self.shift_x = env.render_info["shift_x"]
        self.shift_y = env.render_info["shift_y"]
        self.render_info["env_min_x"], self.render_info["env_max_x"] = -20, 20
        self.render_info["env_min_y"], self.render_info["env_max_y"] = -20, 20
        self.render_info["grid_resolution_x"] = 20
        self.render_info["grid_resolution_y"] = 20
        self.render_info["state_dim"] = env.state_dim
        if self.world_model_comparsion:
            self.robot_poses = None
            self.world_model_poses = None
        if self.add_subgoal_values:
            assert 1 == 0, "didnt implement"
    
    def setup_renderer(self):
        if self.world_model_comparsion or self.controller_safe_model:
            self.robot_poses = []
            self.world_model_poses = []
    
    def delete_data(self):
        if self.world_model_comparsion or self.controller_safe_model:
            del self.robot_poses
            del self.world_model_poses

    def custom_render(self, current_step_info, positions_render=False, 
                      plot_goal=True, debug_info={}, shape=(600, 600), 
                      env_name="", safe_model=None):    
        assert "robot_pos" in current_step_info and \
               "goal_pos" in current_step_info and \
               "robot_radius" in current_step_info
        if self.plot_subgoal:
            assert "subgoal_pos" in current_step_info

        if "SafeAntMaze" in env_name:
            safety_boundary, safe_dataset = self.env.get_safety_bounds(get_safe_unsafe_dataset=True)
            debug_info["safety_boundary"] = safety_boundary
            debug_info["safe_dataset"] = safe_dataset

        shift_x, shift_y = self.shift_x, self.shift_y
        env_min_x, env_max_x = self.render_info["env_min_x"], self.render_info["env_max_x"]
        env_min_y, env_max_y = self.render_info["env_min_y"], self.render_info["env_max_y"]
        if self.render_info["fig"] is None:
            if self.add_subgoal_values:
                self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                self.render_info["ax_subgoal_values"] = self.render_info["fig"].add_subplot(122)
            elif self.world_model_comparsion or self.controller_safe_model:
                self.render_info["fig"] = plt.figure(figsize=[6.4*2, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(121)
                self.render_info["ax_world_model_robot_trajectories"] = self.render_info["fig"].add_subplot(122)
            else:
                self.render_info["fig"] = plt.figure(figsize=[6.4, 4.8])
                self.render_info["ax_states"] = self.render_info["fig"].add_subplot(111)
        self.render_info["ax_states"].set_ylim(bottom=env_min_y, top=env_max_y)
        self.render_info["ax_states"].set_xlim(left=env_min_x, right=env_max_x)
        if self.world_model_comparsion or self.controller_safe_model:
            self.render_info["ax_world_model_robot_trajectories"].set_ylim(bottom=env_min_y, top=env_max_y)
            self.render_info["ax_world_model_robot_trajectories"].set_xlim(left=env_min_x, right=env_max_x)

        # robot pose
        x = current_step_info["robot_pos"][0] + shift_x
        y = current_step_info["robot_pos"][1] + shift_y
        circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="g", alpha=0.5)
        self.render_info["ax_states"].add_patch(circle_robot) 
        self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s")
        # world model comparsion
        if self.world_model_comparsion or self.controller_safe_model:
            self.robot_poses.append((x - shift_x, y - shift_y))   

        # robot imagined pose
        if self.world_model_comparsion:
            x = current_step_info["imagined_robot_pos"][0] + shift_x
            y = current_step_info["imagined_robot_pos"][1] + shift_y
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"] / 2, color="r", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x, y + 0.05, "i_s")
            self.world_model_poses.append((x - shift_x, y - shift_y))   

        # subgoal
        if self.plot_subgoal:
            x = current_step_info["subgoal_pos"][0] + shift_x
            y = current_step_info["subgoal_pos"][1] + shift_y
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="orange", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot)
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "s_g")
            if self.add_subgoal_values:
                self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_s_sg"])), debug_info["v_s_sg"])
                self.render_info["ax_subgoal_values"].plot(range(len(debug_info["v_sg_g"])), debug_info["v_sg_g"])

        # goal
        if env_name != "AntGather" and env_name != "AntMazeSparse" and plot_goal:
            x = current_step_info["goal_pos"][0] + shift_x
            y = current_step_info["goal_pos"][1] + shift_y
            circle_robot = plt.Circle((x, y), radius=current_step_info["robot_radius"], color="y", alpha=0.5)
            self.render_info["ax_states"].add_patch(circle_robot) 
            self.render_info["ax_states"].text(x + 0.05, y + 0.05, "g")  

        # world model comparsion
        if self.world_model_comparsion or self.controller_safe_model:
            xA, yA = zip(*self.robot_poses)
            self.render_info["ax_world_model_robot_trajectories"].plot(xA, yA, 'g', label='robot poses')
            if self.world_model_comparsion:
                xB, yB = zip(*self.world_model_poses)
                self.render_info["ax_world_model_robot_trajectories"].plot(xB, yB, 'r', label='wm poses')

        if self.controller_safe_model:
            cb = plot_values(self.render_info["fig"], 
                        self.render_info["ax_world_model_robot_trajectories"], 
                        safe_model, render_info=self.render_info, return_cb=True)

        # safety boundary
        if self.plot_safety_boundary:
            safety_boundary = debug_info["safety_boundary"]
            if self.controller_safe_model:
                safe_dataset = debug_info["safe_dataset"]
            xs = [point.x + shift_x for point in safety_boundary]
            ys = [point.y + shift_y for point in safety_boundary]
            self.render_info["ax_states"].plot(xs, ys, 'b')
            if self.world_model_comparsion or self.controller_safe_model:
                xs = [point.x for point in safety_boundary]
                ys = [point.y for point in safety_boundary]
                self.render_info["ax_world_model_robot_trajectories"].plot(xs, ys, 'b')
                # safe dataset check
                if self.controller_safe_model and self.plot_safe_dataset:
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

            
        # print maze
        if env_name != "AntGather":
            env_map = self.env.get_maze()
            for i in env_map:
                for indx, val in enumerate(i):
                    if val == 'r':
                        i[indx] = 2

            self.render_info["ax_states"].imshow(env_map, cmap='viridis', 
                                                 interpolation='nearest', 
                                                 extent=[env_min_x, env_max_x, env_min_y, env_max_y])
        else:
            # add apples & bombs
            apples_and_bombs = current_step_info["apples_and_bombs"]

            apples = [(x, y) for (x, y, type_) in apples_and_bombs if type_ == 0]
            apples = [plt.Circle([obs[0] + shift_x, obs[1] + shift_y], radius=current_step_info["apple_bomb_radius"],  # noqa
                        color="y", alpha=0.5) for obs in apples]
            
            bombs = [(x, y) for (x, y, type_) in apples_and_bombs if type_ == 1]
            bombs = [plt.Circle([obs[0] + shift_x, obs[1] + shift_y], radius=current_step_info["apple_bomb_radius"],  # noqa
                        color="r", alpha=0.5) for obs in bombs]
            
            for item_ in apples:
                self.render_info["ax_states"].add_patch(item_)
            for item_ in bombs:
                self.render_info["ax_states"].add_patch(item_)
                
        
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

        # render img
        self.render_info["fig"].canvas.draw()
        data = np.frombuffer(self.render_info["fig"].canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.render_info["fig"].canvas.get_width_height()[::-1] + (3,))
        self.render_info["ax_states"].clear()
        if self.world_model_comparsion or self.controller_safe_model:
            if self.controller_safe_model:
                cb.remove()
            self.render_info["ax_world_model_robot_trajectories"].clear()
        if self.add_subgoal_values:
            self.render_info["ax_subgoal_values"].clear()
        return data


def create_env(args, renderer_args={}):
    # Env initialization
    ## Ant envs
    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)   
    elif args.env_name in ["SafeAntMazeC", "SafeAntMazeW", "SafeAntMazeS", "AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        if args.env_name == "AntMaze":
            maze_id = "Maze"
        if args.env_name == "SafeAntMazeC":
            maze_id = "MazeSafe_map_1"
        elif args.env_name == "SafeAntMazeW":
            maze_id = "MazeSafe_map_2"
        elif args.env_name == "SafeAntMazeS":
            maze_id = "MazeSafe_map_3"
        elif args.env_name == "AntMazeSparse":
            maze_id = "Maze2"
        elif args.env_name == "AntPush":
            maze_id = "Push"
        elif args.env_name == "AntFall":
            maze_id = "Fall"
        else:
            assert 1 == 0
        if args.env_name == "SafeAntMazeC" or args.env_name == "SafeAntMazeW" or args.env_name == "SafeAntMazeS":
            env = SafeMazeAnt(EnvWithGoal(create_maze_env("AntMaze", args.seed, maze_id=maze_id), "AntMaze", maze_id=maze_id))
            if args.random_start_pose:
                env.set_train_start_pose_to_random()
        else:
            env = EnvWithGoal(create_maze_env(args.env_name, args.seed, maze_id=maze_id), args.env_name, maze_id=maze_id)
        env.seed(args.seed)
    elif args.env_name == "AntMazeMultiMap":    
        maze_ids = ["Maze_map_1", "Maze_map_2", "Maze_map_3", "Maze_map_4"]
        envs = []
        for maze_id in maze_ids:
            env = EnvWithGoal(create_maze_env(args.env_name, args.seed, maze_id=maze_id), args.env_name, maze_id=maze_id)
            env.seed(args.seed)
            envs.append(env)
            env = MultyEnvWithGoal(envs)
        env.seed(args.seed)
    ## Safety gym envs
    elif "Point" in args.env_name:
        # test
        DEFAULT_ENV_CONFIG_POINT = dict(
            action_repeat=1,
            max_episode_length=750,
            use_dist_reward=False,
            stack_obs=False,
        )
        robot = 'Point'
        eplen = 750
        num_steps = 4.5e5
        steps_per_epoch = 30000
        epochs = 60
        DEFAULT_ENV_CONFIG_POINT['max_episode_length'] = eplen
        env_config=DEFAULT_ENV_CONFIG_POINT
        #env = SafetyGymEnv(robot=robot, task="goal", level='2', seed=10, config=env_config)
        #state_dim, action_dim = env.observation_size, env.action_size
        assert 1 == 0
    else:
        raise NotImplementedError
    
    # Reset env
    obs = env.reset()
    goal = obs["desired_goal"]
    state = obs["observation"]

    action_dim = env.action_space.shape[0]
    state_dim = state.shape[0]
    if args.env_name in ["SafeAntMazeC", "SafeAntMazeW", "SafeAntMazeS", "AntMaze", 
                         "AntPush", "AntFall", "AntMazeMultiMap"]:
        goal_dim = goal.shape[0]
    else:
        goal_dim = 0
    env.set_state_dim(state_dim)
    env.set_goal_dim(goal_dim)

    renderer = CustomVideoRendered(env, **renderer_args)

    env.max_len = 500

    return env, state_dim, goal_dim, action_dim, renderer