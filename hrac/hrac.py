import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hrac.models import ControllerActor, ControllerCritic, \
    ManagerActor, ManagerCritic, ControllerSafeModel

from hrac.world_models import EnsembleDynamicsModel, PredictEnv

"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/hiro.py
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("******")
print("******")
print("training device:", device)
print("******")
print("******")


def var(tensor, to_device=True):
    if to_device:
        return tensor.to(device)
    else:
        return tensor


def get_tensor(z, to_device=True):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy()), to_device).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()), to_device)


class Manager(object):
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5, goal_loss_coeff=0, absolute_goal=False,
                 wm_no_xy=False, img_horizon=10, 
                 modelfree_safety=False, testing_mean_wm=False,
                 subgoal_grad_clip=0,
                 coef_safety_modelbased=1.0,
                 coef_safety_modelfree=1.0,
                 lidar_observation=False,
                 use_lagrange=False,
                 pid_kp=1e-6,
                 pid_ki=1e-7,
                 pid_kd=1e-7,
                 pid_d_delay=10,
                 pid_delta_p_ema_alpha=0.95,
                 pid_delta_d_ema_alpha=0.95,
                 penalty_max=100.,
                 lagrangian_multiplier_init=0.0,
                 cost_limit=25.,):
        self.scale = scale
        self.actor = ManagerActor(state_dim, goal_dim, action_dim,
                                  scale=scale, absolute_goal=absolute_goal).to(device)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim,
                                         scale=scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr, weight_decay=0.0001)

        self.action_norm_reg = 0

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal

        # WorldModel
        self.predict_env = None
        self.no_xy = wm_no_xy
        self.testing_mean_wm = testing_mean_wm

        # Safety
        self.lidar_observation = lidar_observation   
        self.img_horizon = img_horizon
        self.modelfree_safety = modelfree_safety
        self.subgoal_grad_clip = subgoal_grad_clip
        self.coef_safety_modelbased = coef_safety_modelbased
        self.coef_safety_modelfree = coef_safety_modelfree

        # Lagrange
        self.use_lagrange = use_lagrange
        if self.use_lagrange:
            self.cost_criterion = nn.SmoothL1Loss()
            self.cost_critic = ManagerCritic(state_dim, goal_dim, action_dim).to(device)
            self.cost_critic_target = ManagerCritic(state_dim, goal_dim, action_dim).to(device)

            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                                                    lr=critic_lr, weight_decay=0.0001)
            
            self._pid_kp = pid_kp
            self._pid_ki = pid_ki
            self._pid_kd = pid_kd
            self._pid_d_delay = pid_d_delay
            self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
            self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
            self._penalty_max = penalty_max
            self._pid_i = lagrangian_multiplier_init
            self._cost_ds = deque(maxlen=self._pid_d_delay)
            self._cost_ds.append(0.0)
            self._delta_p = 0.0
            self._cost_d = 0.0
            self._cost_limit = cost_limit
            self._cost_penalty = 0.0

    def set_predict_env(self, predict_env):
        self.predict_env = predict_env

    def imagine_state(self, prev_imagined_state, prev_action, current_state, current_step, imagined_state_freq):
        with torch.no_grad():
            if prev_imagined_state is None or current_step % imagined_state_freq == 0:
                imagined_state = current_state
            else:
                imagined_state = self.predict_env.step(prev_imagined_state, 
                                                       prev_action, 
                                                       deterministic=True, 
                                                       testing_mean_pred=self.testing_mean_wm)
        return imagined_state

    def train_world_model(self, replay_buffer, batch_size=256):
        if replay_buffer.cost_memmory:
            x, y, sg, u, r, c, d, _, _ = replay_buffer.sample(len(replay_buffer))
        else:
            x, y, sg, u, r, d, _, _ = replay_buffer.sample(len(replay_buffer))
        state = get_tensor(x, to_device=False)
        state_device = state.to(device)
        action = get_tensor(u, to_device=False)
        next_state = get_tensor(y, to_device=False)

        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)

        labels = delta_state.numpy()
        epoch, loss = self.predict_env.model.train(inputs, labels, batch_size=batch_size, holdout_ratio=0.2)
        del state, action, next_state
        
        return loss

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        if to_numpy:
            return self.actor(state, goal).cpu().data.numpy().squeeze()
        else:
            return self.actor(state, goal).squeeze()

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)
    
    def state_safety_on_horizon(self, state, actions, 
                                controller_policy, 
                                safety_cost, 
                                all_steps_safety=False, 
                                train=False):

        assert not(self.predict_env is None), "world model must be initialized"
        manager_proposed_goal = actions.clone()
        next_img_state = state.clone()

        h = 0
        if all_steps_safety:
            safeties = []    
            horizon = self.img_horizon
        else:
            horizon = random.randint(1, self.img_horizon)
        while h < horizon:
            img_state = next_img_state
            ctrl_actions = controller_policy.actor(controller_policy.clean_obs(img_state), manager_proposed_goal) 
            next_img_state = self.predict_env.step(img_state, ctrl_actions, 
                                                    deterministic=True, 
                                                    torch_deviced=True,
                                                    testing_mean_pred=self.testing_mean_wm)
            if all_steps_safety:
                if self.lidar_observation:
                    agent_pose = next_img_state[:, :2]
                    obstacle_data = next_img_state[:, -16:]
                    part_of_state = torch.cat((agent_pose, obstacle_data), dim=1)
                    manager_absolute_goal = agent_pose
                    manager_absolute_goal = torch.cat((manager_absolute_goal, part_of_state), dim=1)
                    safety = safety_cost(manager_absolute_goal)
                else:
                    safety = safety_cost(next_img_state)
                safeties.append(safety)
            manager_proposed_goal = controller_policy.subgoal_transition(img_state, 
                                                                         manager_proposed_goal, 
                                                                         next_img_state)
            h += 1
        if not all_steps_safety:
            if self.lidar_observation:
                agent_pose = next_img_state[:, :2]
                obstacle_data = next_img_state[:, -16:]
                part_of_state = torch.cat((agent_pose, obstacle_data), dim=1)
                manager_absolute_goal = agent_pose
                manager_absolute_goal = torch.cat((manager_absolute_goal, part_of_state), dim=1)
                safety = safety_cost(manager_absolute_goal)
            else:
                safety = safety_cost(next_img_state)
        else:
            safety = 0
            for el in safeties:
                safety += el
            if train:
                safety /= self.img_horizon
        return safety
            
    def actor_loss(self, state, goal, a_net, r_margin, cost_model=None):
        actions = self.actor(state, goal)
        
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        goal_loss = None
        safety_loss = None
        cost_loss = 0
        if not(a_net is None):
            goal_loss = torch.clamp(F.pairwise_distance(
                a_net(state[:, :self.action_dim]), a_net(state[:, :self.action_dim] + actions)) - r_margin, min=0.).mean()
        safety_model_free_loss = 0
        if self.modelfree_safety:
            copy_state = state.detach()
            manager_absolute_goal = actions.clone()
            manager_absolute_goal += copy_state[:, :actions.shape[1]]
            if self.lidar_observation:
                # test copy states
                if cost_model.frame_stack_num > 1:
                    agent_pose = copy_state[:, :2]
                    obstacle_data = copy_state[:, -16:]
                    part_of_state = torch.cat((agent_pose, obstacle_data), dim=1)
                    for i in range(cost_model.frame_stack_num - 1):
                        part_of_state = torch.cat((part_of_state, agent_pose, obstacle_data), dim=1)
                else:
                    agent_pose = copy_state[:, :2]
                    obstacle_data = copy_state[:, -16:]
                    part_of_state = torch.cat((agent_pose, obstacle_data), dim=1)
                manager_absolute_goal = torch.cat((manager_absolute_goal, part_of_state), dim=1)
            else:
                zeros_to_add = torch.zeros(actions.shape[0], 
                                            state.shape[1] - actions.shape[1]).to(device)
                manager_absolute_goal = torch.cat((manager_absolute_goal, zeros_to_add), dim=1)
            safety_model_free_loss = cost_model.safe_model(manager_absolute_goal)
            safety_model_free_loss = safety_model_free_loss.mean()
        safety_loss = self.coef_safety_modelfree * safety_model_free_loss
        if self.use_lagrange:
            cost_loss = self.cost_critic.Q1(state, goal, actions).mean() 
        return eval, norm, goal_loss, safety_loss, cost_loss

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq):
        first_x = [x[0] for x in x_seq]
        last_x = [x[-1] for x in x_seq]

        # Shape: (batchsz, 1, subgoal_dim)
        diff_goal = (np.array(last_x) - np.array(first_x))[:, np.newaxis, :self.action_dim]

        # Shape: (batchsz, 1, subgoal_dim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :self.action_dim],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale[:self.action_dim], self.scale[:self.action_dim])

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # print(np.array(x_seq).shape)
        x_seq = np.array(x_seq)[:, :-1, :]
        a_seq = np.array(a_seq)
        seq_len = len(x_seq[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = candidates.shape[1]

        true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
        observations = x_seq.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(x_seq, candidates[:, c])
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, cost_model, 
              iterations, batch_size=100, discount=0.99,
              tau=0.005, a_net=None, r_margin=None):
        avg_act_loss, avg_crit_loss = 0., 0.
        avg_eval_loss, avg_cost_loss, avg_cost_critic_loss = 0., 0., 0.
        debug_maganer_info = {"sum_manager_actor_grad_norm": 0}
        if a_net is not None:
            avg_goal_loss = 0.
        if self.modelfree_safety:
            avg_safety_subgoals_loss = 0.
        else:
            avg_safety_subgoals_loss = None
        for it in range(iterations):
            # Sample replay buffer
            if self.use_lagrange:
                x, y, g, sgorig, r, c, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            else:
                x, y, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            batch_size = min(batch_size, x.shape[0])

            if self.correction and not self.absolute_goal:
                sg = self.off_policy_corrections(controller_policy, batch_size,
                                                 sgorig, xobs_seq, a_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            # print(g)
            goal = get_tensor(g)
            subgoal = get_tensor(sg)
            if self.use_lagrange:
                cost = get_tensor(c)
            reward = get_tensor(r)
            done = get_tensor(1 - d)

            noise = torch.FloatTensor(sgorig).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, goal,
                                          next_action)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                          self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            cost_critic_loss = 0
            if self.use_lagrange:
                # Cost critic
                target_C1, target_C2 = self.cost_critic_target(next_state, goal,
                                            next_action)
                # TODO: check max 
                target_C = torch.max(target_C1, target_C2)
                target_C = cost + (done * discount * target_C)
                target_C_no_grad = target_C.detach()

                # Get current C estimate
                current_C1, current_C2 = self.cost_critic(state, goal, subgoal)

                # Compute critic loss
                cost_critic_loss = self.cost_criterion(current_C1, target_C_no_grad) +\
                            self.cost_criterion(current_C2, target_C_no_grad)

                # Optimize the critic
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            # Compute actor loss
            eval_loss, norm_loss, goal_loss, safety_subgoals_loss, cost_loss = self.actor_loss(state, goal, a_net, r_margin, cost_model)
            if self.use_lagrange:
                actor_loss = norm_loss + (eval_loss + self._cost_penalty * cost_loss) / (1 + self._cost_penalty)
            else:
                actor_loss = eval_loss + norm_loss
            if not(a_net is None):
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
            if self.modelfree_safety:
                actor_loss = actor_loss + safety_subgoals_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()

            # test
            #actor_loss.retain_grad() 
            #goal_loss.retain_grad() 
            #safety_subgoals_loss.retain_grad() 

            actor_loss.backward()

            if self.subgoal_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                      max_norm=self.subgoal_grad_clip)

            with torch.no_grad():
                manager_actor_grad_norm = (
                    sum(p.grad.data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None) ** 0.5
                )
                debug_maganer_info["sum_manager_actor_grad_norm"] += manager_actor_grad_norm                

            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            avg_eval_loss += eval_loss
            avg_cost_critic_loss += cost_critic_loss
            avg_cost_loss += cost_loss
            if a_net is not None:
                avg_goal_loss += goal_loss
            if self.modelfree_safety:
                avg_safety_subgoals_loss += safety_subgoals_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if self.use_lagrange:
                for param, target_param in zip(self.cost_critic.parameters(),
                                           self.cost_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        debug_maganer_info["sum_manager_actor_grad_norm"] /= iterations
        if self.modelfree_safety:
            avg_safety_subgoals_loss = avg_safety_subgoals_loss / iterations
        avg_act_loss /= iterations
        avg_crit_loss /= iterations
        avg_goal_loss /= iterations
        avg_eval_loss /= iterations
        avg_cost_critic_loss /= iterations
        avg_cost_loss /= iterations
        return avg_act_loss, avg_crit_loss, avg_goal_loss, avg_safety_subgoals_loss,\
            debug_maganer_info, avg_eval_loss, avg_cost_critic_loss, avg_cost_loss
    
    def pid_update(self, ep_cost_avg):
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        self._cost_ds.append(self._cost_d)

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir, env_name, algo, exp_num):
        torch.save(self.actor.state_dict(), "{}/{}/{}_{}_ManagerActor.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}/{}_{}_ManagerCritic.pth".format(dir, exp_num, env_name, algo))
        
        torch.save(self.actor_target.state_dict(), "{}/{}/{}_{}_ManagerActorTarget.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}/{}_{}_ManagerCriticTarget.pth".format(dir, exp_num, env_name, algo))
        if self.use_lagrange:
            torch.save(self.cost_critic.state_dict(), "{}/{}/{}_{}_ManagerCostCritic.pth".format(dir, exp_num, env_name, algo))
            torch.save(self.cost_critic_target.state_dict(), "{}/{}/{}_{}_ManagerCostCriticTarget.pth".format(dir, exp_num, env_name, algo))
        if not(self.predict_env is None):
            # save as pkl
            torch.save(self.predict_env.model, "{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
            # save as state dict + scaler
            torch.save(self.predict_env.model.ensemble_model.state_dict(), "{}/{}/{}_{}_env_model.pth".format(dir, exp_num, env_name, algo))
            # save: scalar mu, scalar std, model elite idxs 
            mu = self.predict_env.model.scaler.mu
            std = self.predict_env.model.scaler.std
            elite_model_idxes = np.array(self.predict_env.model.elite_model_idxes)
            np.save("{}/{}/{}_{}_wm_scaler_mu.npy".format(dir, exp_num, env_name, algo), mu)
            np.save("{}/{}/{}_{}_wm_scaler_std.npy".format(dir, exp_num, env_name, algo), std)
            np.save("{}/{}/{}_{}_wm_elite_model_idxes.npy".format(dir, exp_num, env_name, algo), elite_model_idxes)
        
    def load(self, dir, env_name, algo, exp_num, load_wm_as_pkl=True):
        self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActor.pth".format(dir, exp_num, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCritic.pth".format(dir, exp_num, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActorTarget.pth".format(dir, exp_num, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCriticTarget.pth".format(dir, exp_num, env_name, algo)))
        if not(self.predict_env is None):
            temp_env_name = 'safepg2'
            temp_model_type='pytorch'
            if load_wm_as_pkl:
                env_model = torch.load("{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
                predict_env = PredictEnv(env_model, temp_env_name, temp_model_type)
                self.set_predict_env(predict_env)
            else:
                self.predict_env.model.ensemble_model.load_state_dict(torch.load("{}/{}/{}_{}_env_model.pth".format(dir, exp_num, env_name, algo)))
                mu = np.load("{}/{}/{}_{}_wm_scaler_mu.npy".format(dir, exp_num, env_name, algo))
                std = np.load("{}/{}/{}_{}_wm_scaler_std.npy".format(dir, exp_num, env_name, algo))
                self.predict_env.model.scaler.set_mu_std(mu, std)
                elite_model_idxes = np.load("{}/{}/{}_{}_wm_elite_model_idxes.npy".format(dir, exp_num, env_name, algo))
                self.predict_env.model.set_elite_model_idxes(elite_model_idxes)
                
class CostModel(object):
    def __init__(self, state_dim, goal_dim, lidar_observation, 
                       frame_stack_num, 
                       safe_model_loss_coef, lr):
        self.lidar_observation = lidar_observation
        self.safe_model_loss_coef = safe_model_loss_coef        
        self.frame_stack_num = frame_stack_num
        if self.lidar_observation:
            agent_xy = 2     
            lidar_obs = 16       
            self.safe_model = ControllerSafeModel(goal_dim + (agent_xy + lidar_obs) * frame_stack_num).to(device)
        else:
            self.safe_model = ControllerSafeModel(state_dim).to(device)
        
        self.safe_model_criterion = nn.BCELoss()
        self.safe_model_optimizer = torch.optim.Adam(self.safe_model.parameters(),
                                                lr=lr, weight_decay=0.0001)
        

    def train_cost_model(self, replay_buffer, 
                         cost_model_iterations=10, 
                         cost_model_batch_size=128,
                         train_on_dataset=False,
                         dataset=None):
        debug_info = {}
        debug_info["safe_model_loss"] = []
        debug_info["safe_model_mean_true"] = []
        debug_info["safe_model_mean_pred"] = []
        for i in range(cost_model_iterations):
            if train_on_dataset:
                x = random.sample(dataset[0], cost_model_batch_size)
                x_np = np.array(x, dtype=np.float32)
                x_tensor = torch.tensor(x_np)
                state_device = x_tensor.to(device)

                cost = random.sample(dataset[1], cost_model_batch_size)
                cost_np = np.array(cost, dtype=np.float32)
                cost_tensor = torch.tensor(cost_np)
                cost_device = cost_tensor.to(device)
            else:
                if replay_buffer.name == "cost_trajectory_buffer":
                    x, c = replay_buffer.sample(cost_model_batch_size)
                    state = get_tensor(x, to_device=False) # cost for the next_state
                    state_device = state.to(device)
                    cost = get_tensor(c, to_device=False)
                    cost_device = cost.to(device)
                elif replay_buffer.cost_memmory:
                    x, y, sg, u, r, c, d, _, _ = replay_buffer.sample(cost_model_batch_size)
                    state = get_tensor(y, to_device=False) # cost for the next_state
                    state_device = state.to(device)
                    cost = get_tensor(c, to_device=False)
                    cost_device = cost.to(device)
                else:
                    x, y, sg, u, r, d, _, _ = replay_buffer.sample(cost_model_batch_size)
                    state = get_tensor(x, to_device=False)
                    state_device = state.to(device)
                    cost_device = None
            safe_model_loss, true, pred = self.train_batch_cost_model(state_device, cost=cost_device)
            debug_info["safe_model_loss"].append(safe_model_loss.mean().cpu().detach())
            debug_info["safe_model_mean_true"].append(true.float().mean().cpu().detach())
            debug_info["safe_model_mean_pred"].append(pred.float().mean().cpu().detach())
        return debug_info
    
    def train_batch_cost_model(self, init_state, cost=None):
        pred = self.safe_model(init_state)
        numpy_b_xy = init_state.cpu().detach().numpy()[:, :2]
        if not(cost is None):
            true = cost
        else:
            true = torch.tensor(self.cost_function(numpy_b_xy), dtype=torch.float).to(device).unsqueeze(1)

        # Compute safet_model loss
        safe_model_loss = self.safe_model_loss_coef * self.safe_model_criterion(pred, true)

        # Optimize the safe_model
        self.safe_model_optimizer.zero_grad()
        safe_model_loss.backward()
        self.safe_model_optimizer.step()

        return safe_model_loss, true, pred
    

    def save(self, dir, env_name, algo, exp_num):
        torch.save(self.safe_model.state_dict(), "{}/{}/{}_{}_SafeModel.pth".format(dir, exp_num, env_name, algo))
        
    def load(self, dir, env_name, algo, exp_num):
        self.safe_model.load_state_dict(torch.load("{}/{}/{}_{}_SafeModel.pth".format(dir, exp_num, env_name, algo)))


class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr,
                 critic_lr, repr_dim=15, no_xy=True, policy_noise=0.2, noise_clip=0.5,
                 absolute_goal=False, 
                 cost_function=None,
                 controller_imagination_safety_loss=False,
                 manager=None, controller_grad_clip=0, controller_safety_coef=0, 
                 use_lagrange=False,
                 pid_kp=1e-6,
                 pid_ki=1e-7,
                 pid_kd=1e-7,
                 pid_d_delay=10,
                 pid_delta_p_ema_alpha=0.95,
                 pid_delta_d_ema_alpha=0.95,
                 penalty_max=100.,
                 lagrangian_multiplier_init=0.0,
                 cost_limit=25.,):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal
        self.criterion = nn.SmoothL1Loss()            

        self.manager = manager
        self.controller_imagination_safety_loss = controller_imagination_safety_loss
        self.controller_safety_coef = controller_safety_coef
        self.controller_grad_clip = controller_grad_clip
        if self.controller_imagination_safety_loss:
            assert self.manager
        self.cost_function = cost_function

        self.actor = ControllerActor(state_dim, goal_dim, action_dim,
                                    scale=max_action).to(device)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim,
                                            scale=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
            lr=critic_lr, weight_decay=0.0001)
        
        # Lagrange
        self.use_lagrange = use_lagrange
        if self.use_lagrange:
            self.cost_criterion = nn.SmoothL1Loss()
            self.cost_critic = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
            self.cost_critic_target = ControllerCritic(state_dim, goal_dim, action_dim).to(device)

            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                lr=critic_lr, weight_decay=0.0001)
            
            self._pid_kp = pid_kp
            self._pid_ki = pid_ki
            self._pid_kd = pid_kd
            self._pid_d_delay = pid_d_delay
            self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
            self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
            self._penalty_max = penalty_max
            self._pid_i = lagrangian_multiplier_init
            self._cost_ds = deque(maxlen=self._pid_d_delay)
            self._cost_ds.append(0.0)
            self._delta_p = 0.0
            self._cost_d = 0.0
            self._cost_limit = cost_limit
            self._cost_penalty = 0.0

    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def get_value(self, state, sg):
        assert not sg is None
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.agent.get_value(state, sg)

    def select_action_logprob_value(self, state, sg, evaluation=False):
        assert not sg is None
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.agent.get_action_and_value(state, sg)

    def select_action(self, state, sg, evaluation=False):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.actor(state, sg).cpu().data.numpy().squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg, init_state, cost_model):
        actions = self.actor(state, sg)
        eval_loss = -self.critic.Q1(state, sg, actions).mean()
        safety_loss = 0
        if self.controller_imagination_safety_loss:
            safety_loss = self.manager.state_safety_on_horizon(init_state, sg, 
                                                                controller_policy=self, 
                                                                safety_cost=cost_model.safe_model,
                                                                all_steps_safety=False,
                                                                train=True).mean()
        cost_loss = 0
        if self.use_lagrange:
            cost_loss = self.cost_critic.Q1(state, sg, actions).mean() 
        return eval_loss, safety_loss, cost_loss

    def subgoal_transition(self, state, subgoal, next_state):
        if self.absolute_goal:
            return subgoal
        else:
            if len(state.shape) == 1:  # check if batched
                return state[:self.goal_dim] + subgoal - next_state[:self.goal_dim]
            else:
                return state[:, :self.goal_dim] + subgoal -\
                       next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals

    def train(self, replay_buffer, cost_model, iterations, batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.
        avg_eval_loss, avg_cost_loss, avg_cost_critic_loss = 0., 0., 0.
        debug_info = {}
        for it in range(iterations):              
            if self.use_lagrange:
                x, y, sg, u, r, c, d, _, _ = replay_buffer.sample(batch_size)
                cost = get_tensor(c)
            else:
                x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            init_state = get_tensor(x)
            next_g = get_tensor(self.subgoal_transition(x, sg, y))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y)) 

            
            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(state, sg, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) +\
                        self.criterion(current_Q2, target_Q_no_grad)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            cost_critic_loss = 0
            if self.use_lagrange:
                # Cost critic
                target_C1, target_C2 = self.cost_critic_target(next_state, next_g, next_action)
                # TODO: check max 
                target_C = torch.max(target_C1, target_C2)
                target_C = cost + (done * discount * target_C)
                target_C_no_grad = target_C.detach()

                # Get current C estimate
                current_C1, current_C2 = self.cost_critic(state, sg, action)

                # Compute critic loss
                cost_critic_loss = self.cost_criterion(current_C1, target_C_no_grad) +\
                            self.cost_criterion(current_C2, target_C_no_grad)

                # Optimize the critic
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            # Compute actor loss
            eval_loss, safety_loss, cost_loss = self.actor_loss(state, sg, init_state, cost_model)
            if self.use_lagrange:
                actor_loss = (eval_loss + self._cost_penalty * cost_loss) / (1 + self._cost_penalty)
            else:
                actor_loss = eval_loss
            actor_loss += self.controller_safety_coef * safety_loss
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                    max_norm=self.controller_grad_clip)
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            avg_eval_loss += eval_loss
            avg_cost_critic_loss += cost_critic_loss
            avg_cost_loss += cost_loss
            
            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if self.use_lagrange:
                for param, target_param in zip(self.cost_critic.parameters(),
                                           self.cost_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations, debug_info, \
                avg_eval_loss / iterations, avg_cost_critic_loss / iterations, avg_cost_loss / iterations
    
    def pid_update(self, ep_cost_avg):
        delta = float(ep_cost_avg - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        self._cost_ds.append(self._cost_d)

    def save(self, dir, env_name, algo, exp_num):
        torch.save(self.actor.state_dict(), "{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.actor_target.state_dict(), "{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo))
        if self.use_lagrange:
            torch.save(self.cost_critic.state_dict(), "{}/{}/{}_{}_ControllerCostCritic.pth".format(dir, exp_num, env_name, algo))
            torch.save(self.cost_critic_target.state_dict(), "{}/{}/{}_{}_ControllerCostCriticTarget.pth".format(dir, exp_num, env_name, algo))

    def load(self, dir, env_name, algo, exp_num):
        self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo)))