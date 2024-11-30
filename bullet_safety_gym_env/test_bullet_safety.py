import gym
#import bullet_safety_gym

env = gym.make('SafetyCarRun-v0')

print("reset:", env.reset())
#print("render:", env.render().shape)
print("action space:", env.action_space.shape[0])
print("observation space:", env.observation_space.shape[0])
"""
while True:
    done = False
    #env.render()  # make GUI of PyBullet appear
    print("reset env")
    x = env.reset()
    steps = 0
    max_vel = 0
    while not done:
        random_action = env.action_space.sample()
        x, reward, done, info = env.step(random_action)
        #done = done or turn
        #print("action:", random_action)
        #print("new x:", x)
        #print("r:", reward)
        #print("info:", info)
        #print("x:", x.shape, "env:", type(env))
        #if x[2] > max_vel:
        #    max_vel = x[2]
        #print("max episode vel:", max_vel)
        #print("max vel:", env.agent.velocity_constraint)
        steps += 1
    print("episode steps:", steps)
"""