import sys
import os 

#sys.path.append(os.path.dirname(__file__))
#import bullet_safety_gym
from .bullet_safety_gym_wrapper import GCBulletCarRun


def create_bullet_safety_gym_env(args):

    if args.env_name == "SafeBulletCarRun":
        env = GCBulletCarRun()
    else:
        assert 1 == 0

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space["observation"].shape[0]
    goal_dim = 1
    renderer = None
    env.max_len = 500
    
    return env, state_dim, goal_dim, action_dim, renderer
