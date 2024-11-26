import argparse
from env import make_safety

def main(args):
    assert not args.goal_conditioned or (args.goal_conditioned and args.vector_env), "goal conditioned implemented only for vec obs"
    env = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', 
                            image_size=args.image_size, 
                            use_pixels=not args.vector_env, 
                            action_repeat=args.action_repeat,
                            goal_conditioned=args.goal_conditioned)
    env.seed(args.seed)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # action.shape = (2,)
        # obs.shape = (3, 2, 2) if not args.vector_env
        # obs.shape = (44,) if args.vector_env
        # obs["observation"], obs["desired_goal"], obs["achieved_goal"] if args.goal_conditioned
        # info = {"cost_hazards": ... , "cost": ..., 'safety_cost' ... }
        print("action shape:", action.shape)
        print("reward:", reward)
        if args.goal_conditioned:
            print("obs keys:", obs.keys())
            print("observation:", obs["observation"].shape) # (30,)
            print("desired_goal:", obs["desired_goal"].shape) # (2,)
            print("achieved_goal:", obs["achieved_goal"].shape) # (2,)
        else:
            print("obs shape:", obs.shape)
        print("info:", info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--image_size", type=int, default=2)
    parser.add_argument("--vector_env", default=False, action="store_true")
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--domain_name", type=str, default="Safexp", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--goal_conditioned", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    args = parser.parse_args()
    main(args)

