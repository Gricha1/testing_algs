import numpy as np


def get_safetydataset_as_random_experience(env):
    states = []
    costs = []

    ## Collect transitions with random policy for world model, cost model
    unsafe_state = []
    safe_state = []
    done = True
    while len(safe_state) < 2_000 or len(unsafe_state) < 2_000:
        if done:
            obs = env.reset()
            state = obs["observation"]
            done = False
        action = env.action_space.sample()
        next_tup, manager_reward, done, info = env.step(action)   
        next_state = next_tup["observation"]
        cost = info["safety_cost"]        
        state = next_state
        if cost >= 1:
            if len(unsafe_state) < 2_000:
                unsafe_state.append(state)
        else:
            if len(safe_state) < 2_000:
                safe_state.append(state)

    states.extend(unsafe_state)
    states.extend(safe_state)
    costs.extend([1 for i in range(len(unsafe_state))])
    costs.extend([0 for i in range(len(safe_state))])

    assert len(np.unique(costs)) <= 2, f"unique: {np.unique(costs)}"
    assert len(states) == len(costs)
    return states, costs