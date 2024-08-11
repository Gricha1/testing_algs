import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_values(fig, ax_values, safe_model, render_info={}, current_step_info={}, return_cb=True):
    env_max_x = render_info["env_max_x"]
    env_min_x = render_info["env_min_x"]
    env_max_y = render_info["env_max_y"]
    env_min_y = render_info["env_min_y"]
    grid_resolution_x = render_info["grid_resolution_x"]
    grid_resolution_y = render_info["grid_resolution_y"]
    agent_full_obs = current_step_info["agent_full_obs"]
    cm_frame_stack_num = current_step_info["cm_frame_stack_num"]
    prev_agent_full_observations = current_step_info["prev_agent_full_observations"]

    grid_states = []              
    grid_dx = (env_max_x - env_min_x) / grid_resolution_x
    grid_dy = (env_max_y - env_min_y) / grid_resolution_y
    for grid_state_y in np.linspace(env_min_y + grid_dy/2, env_max_y - grid_dy/2, grid_resolution_y):
        for grid_state_x in np.linspace(env_min_x + grid_dx/2, env_max_x - grid_dx/2, grid_resolution_x):
            grid_state = [grid_state_x, grid_state_y]            
            if cm_frame_stack_num > 1:
                for i in range(cm_frame_stack_num):
                    if abs(-i-1) > len(prev_agent_full_observations):
                        agent_full_obs_temp = [0 for i in range(30)]
                    else:
                        agent_full_obs_temp = prev_agent_full_observations[-i-1]
                    grid_state = np.concatenate([grid_state, agent_full_obs_temp[:2], agent_full_obs_temp[-16:]])
            else:
                grid_state = np.concatenate([grid_state, agent_full_obs[:2], agent_full_obs[-16:]])
            grid_states.append(grid_state.tolist())
    grid_states = torch.FloatTensor(np.array(grid_states)).to(device)
    grid_vs = safe_model(grid_states)
    grid_vs = grid_vs.detach().cpu().numpy().reshape(grid_resolution_x, grid_resolution_y)[::-1]
    #mask = grid_vs >= 0.5
    #grid_vs[mask] = 1
    #grid_vs[1 - mask] = 0
    img = ax_values.imshow(grid_vs, extent=[env_min_x,env_max_x, env_min_y,env_max_y])
    if return_cb:
        cb = fig.colorbar(img)
    else:
        cb = None
    #ax_values.scatter([np.linspace(env_max_x - 3.5, env_max_x - 3.5 + car_length*np.cos(theta), 100)], 
    #                [np.linspace(env_max_y - 1.5, env_max_y - 1.5 + car_length*np.sin(theta), 100)], 
    #                color="black", s=5)
    #ax_values.scatter([env_max_x - 3.5], [env_max_y - 1.5], color="black", s=40)
    #ax_values.scatter([x_agent], [y_agent], color="green", s=100)
    #ax_values.scatter([np.linspace(x_agent, x_agent + car_length*np.cos(theta_agent), 100)], 
    #                [np.linspace(y_agent, y_agent + car_length*np.sin(theta_agent), 100)], 
    #                color="black", s=5)
    #ax_values.scatter([x_goal], [y_goal], color="yellow", s=100)
    #ax_values.scatter([np.linspace(x_goal, x_goal + car_length*np.cos(theta_goal), 100)], 
    #                [np.linspace(y_goal, y_goal + car_length*np.sin(theta_goal), 100)], 
    #                color="black", s=5)

    return cb
