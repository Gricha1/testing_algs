import numpy as np

from safety_ant_maze_pusher_envs import EnvWithGoal, GatherEnv, MultyEnvWithGoal, SafeMazeAnt, SafeFetch
from safety_ant_maze_pusher_envs.create_gather_env import create_gather_env
from safety_ant_maze_pusher_envs.create_maze_env import create_maze_env
from safety_ant_maze_pusher_envs.pusher import PusherEnv
from safety_ant_maze_pusher_envs.create_fetch_env import create_fetch_env
from safety_ant_maze_pusher_envs.render_utils.utils import get_renderer


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
    elif "SafePusher" in args.env_name:
        from gym.envs.registration import register
        register(
            id='Pusher-v0',
            entry_point='safety_ant_maze_pusher_envs.create_fetch_env:create_fetch_env',
            kwargs={'env_name': 'Pusher-v0', "args": args},
            max_episode_steps=100
        )
        import gym 
        env = SafeFetch(gym.make("Pusher-v0", reward_shaping="dense"), args)
    else:
        raise NotImplementedError
    
    # Reset env
    obs = env.reset()
    goal = obs["desired_goal"]
    state = obs["observation"]

    action_dim = env.action_space.shape[0]
    state_dim = state.shape[0]
    if args.env_name in ["SafeAntMazeC", "SafeAntMazeW", "SafeAntMazeS", "AntMaze", 
                         "AntPush", "AntFall", "AntMazeMultiMap", "SafePusher"]:
        goal_dim = goal.shape[0]
    else:
        goal_dim = 0

    env.set_state_dim(state_dim)
    env.set_goal_dim(goal_dim)

    renderer = get_renderer(env, args.env_name, renderer_args)

    env.max_len = 500

    return env, state_dim, goal_dim, action_dim, renderer