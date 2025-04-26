import numpy as np


def get_safetydataset_as_random_experience(env, frame_stack_num=1, phi=None):
    state_goal_pairs = []
    costs = []

    ## Collect transitions with random policy for world model, cost model
    unsafes = []
    safes = []
    done = True
    # current_trajectory = [(goal, state, cost), (goal, state, cost), ...]
    current_trajectory = []
    states_count = 16_000
    while len(safes) < states_count or len(unsafes) < states_count:
        if done:
            obs = env.reset()
            state = obs["observation"]
            achieved_goal = obs["achieved_goal"]
            done = False
            if len(current_trajectory) != 0:
                for i in range(len(current_trajectory)):
                    for j in range(len(current_trajectory)):

                        if frame_stack_num > 1:
                            assert 1 == 0

                        _ = current_trajectory[i][0]
                        state_i = current_trajectory[i][1]
                        _ = current_trajectory[i][2]
                        goal_j = current_trajectory[j][0]
                        _ = current_trajectory[j][1]
                        cost_j = current_trajectory[j][2]
                        if cost_j >= 1: # test could be [0, 1, 2]
                            if len(unsafes) < states_count:
                                unsafes.append((goal_j, state_i))                                
                        else:
                            if len(safes) < states_count:
                                safes.append((goal_j, state_i))
            current_trajectory = []

        action = env.action_space.sample()
        next_tup, manager_reward, done, info = env.step(action)   
        next_state = next_tup["observation"]
        next_achieved_goal = next_tup["achieved_goal"]
        cost = info["safety_cost"]        
        state = next_state
        achieved_goal = next_achieved_goal
        current_trajectory.append((achieved_goal, state, cost))

    state_goal_pairs.extend(unsafes)
    state_goal_pairs.extend(safes)
    goals = []
    states = []
    for (goal, state) in state_goal_pairs:
        goals.append(goal)
        states.append(state)
    costs.extend([1 for i in range(len(unsafes))])
    costs.extend([0 for i in range(len(safes))])

    assert len(np.unique(costs)) <= 2, f"unique: {np.unique(costs)}"
    return goals, states, costs