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

                        #if frame_stack_num > 1:
                        #    frame_stack_states_i = [sc_pair[0] for sc_pair in current_trajectory[i-frame_stack_num+1:i+1]]
                        #else:
                        _ = current_trajectory[i][0]
                        state_i = current_trajectory[i][1]
                        _ = current_trajectory[i][2]
                        goal_j = current_trajectory[j][0]
                        _ = current_trajectory[j][1]
                        cost_j = current_trajectory[j][2]
                        #hazards_i = current_trajectory[i][2]

                        #manager_absolute_goal = state_j[:2]
                        """
                        part_of_state = []
                        if frame_stack_num > 1:
                            agent_poses = [state_i[:2] for state_i in frame_stack_states_i]
                            obstacle_datas = [state_i[-16:] for state_i in frame_stack_states_i]
                            # if current i < self.frame_stack_num, fill posses, obstacle_datas with zeros
                            while len(agent_poses) < frame_stack_num:
                                agent_poses.append([0 for i in range(2)])
                                obstacle_datas.append([0 for i in range(16)])
                            for agent_pose, obstacle_data in zip(agent_poses, obstacle_datas):
                                part_of_state.extend(agent_pose)
                                part_of_state.extend(obstacle_data)
                        else:
                            agent_pose = state_i[:2]
                            obstacle_data = state_i[-16:]                        
                            part_of_state.extend(agent_pose)
                            part_of_state.extend(obstacle_data)
                        state = []
                        state.extend(manager_absolute_goal)
                        state.extend(part_of_state)
                        """
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
        #if cost >= 1:
        #    if len(unsafe_state) < states_count:
        #        unsafe_state.append(state)
        #else:
        #    if len(safe_state) < states_count:
        #        safe_state.append(state)

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