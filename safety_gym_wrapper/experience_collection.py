import numpy as np


def get_safetydataset_as_random_experience(env, frame_stack_num=1):
    states = []
    costs = []

    ## Collect transitions with random policy for world model, cost model
    unsafe_state = []
    safe_state = []
    unsafe_hazard_poses = []
    safe_hazard_poses = []
    done = True
    # current_trajectory = [(state, cost), (state, cost), ...]
    current_trajectory = []
    states_count = 16_000
    while len(safe_state) < states_count or len(unsafe_state) < states_count:
        if done:
            obs = env.reset()
            state = obs["observation"]
            done = False
            if len(current_trajectory) != 0:
                for i in range(len(current_trajectory)):
                    for j in range(len(current_trajectory)):
                        if frame_stack_num > 1:
                            frame_stack_states_i = [sc_pair[0] for sc_pair in current_trajectory[i-frame_stack_num+1:i+1]]
                        else:
                            state_i = current_trajectory[i][0]
                        _ = current_trajectory[i][1]
                        state_j = current_trajectory[j][0]
                        cost_j = current_trajectory[j][1]
                        hazards_i = current_trajectory[i][2]

                        manager_absolute_goal = state_j[:2]
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
                        if cost_j >= 1: # test could be [0, 1, 2]
                            if len(unsafe_state) < states_count:
                                unsafe_state.append(state)
                                unsafe_hazard_poses.append(hazards_i)
                        else:
                            if len(safe_state) < states_count:
                                safe_state.append(state)
                                safe_hazard_poses.append(hazards_i)
            current_trajectory = []

        action = env.action_space.sample()
        next_tup, manager_reward, done, info = env.step(action)   
        next_state = next_tup["observation"]
        cost = info["safety_cost"]        
        state = next_state
        current_hazards = np.array([hazard[:2] for hazard in env.hazards_pos])
        current_trajectory.append((state, cost, current_hazards))
        #if cost >= 1:
        #    if len(unsafe_state) < states_count:
        #        unsafe_state.append(state)
        #else:
        #    if len(safe_state) < states_count:
        #        safe_state.append(state)

    states.extend(unsafe_state)
    states.extend(safe_state)
    costs.extend([1 for i in range(len(unsafe_state))])
    costs.extend([0 for i in range(len(safe_state))])

    hazard_poses = []
    hazard_poses.extend(unsafe_hazard_poses)
    hazard_poses.extend(safe_hazard_poses)

    assert len(np.unique(costs)) <= 2, f"unique: {np.unique(costs)}"
    assert len(states) == len(costs) == len(hazard_poses)
    return states, costs, hazard_poses