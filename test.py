import argparse
from env import make_safety

def main(args):
    env = make_safety(f'{args.domain_name}{"-" if len(args.domain_name) > 0 else ""}{args.task_name}-v0', 
                            image_size=args.image_size, 
                            use_pixels=not args.vector_env, 
                            action_repeat=args.action_repeat)
    env.seed(args.seed)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # action.shape = (2,)
        # obs.shape = (3, 2, 2)
        # info = {"cost_hazards": ... , "cost": ...}
        print("action shape:", action.shape)
        print("reward:", reward)
        print("obs shape:", obs.shape)
        print("info:", info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--image_size", type=int, default=2)
    parser.add_argument("--vector_env", default=False, action="store_true")
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--domain_name", type=str, default="Safexp", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="PointGoal1", help="Name of the task")
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    args = parser.parse_args()
    main(args)

