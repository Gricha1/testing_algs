import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ites.models import ControllerActor, ControllerCritic, \
    ManagerActor, ManagerCritic, ControllerSafeModel

from ites.world_models import EnsembleDynamicsModel, PredictEnv
import ites.utils as utils

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
    
def subgoal_transition(achieved_goal, subgoal, next_achieved_goal, absolute_goal=False):
    if absolute_goal:
        return subgoal
    else:
        if len(achieved_goal.shape) == 1:  # check if batched
            return achieved_goal + subgoal - next_achieved_goal
        else:
            return achieved_goal + subgoal -\
                    next_achieved_goal

class Manager(object):
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5, goal_loss_coeff=0, absolute_goal=False, 
                 testing_mean_wm=False,
                 subgoal_grad_clip=0,
                 coef_safety_modelbased=1.0,
                 coef_safety_modelfree=1.0,
                 lidar_observation=False, algo=None,
                 hidden_size=300,
                 phi=None,
                 args=None
                 ):
        
        self.algo = algo
        self.device = device
        self.args = args

        self.phi = phi
        self.scale = scale
        self.torch_scale = torch.tensor(self.scale).type('torch.FloatTensor').to(device)
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal

        # Safety
        self.lidar_observation = lidar_observation   
        self.subgoal_grad_clip = subgoal_grad_clip
        self.coef_safety_modelbased = coef_safety_modelbased
        self.coef_safety_modelfree = coef_safety_modelfree
        if "high_lag" in self.args.manager_algo:
            self._cost_penalty = 0.0
            # Cost Critic
            self.cost_critic = ControllerCritic(
                state_dim, goal_dim, action_dim, hidden_size
            ).to(device)
            self.cost_critic_target = ControllerCritic(
                state_dim, goal_dim, action_dim, hidden_size
            ).to(device)
            self.cost_critic_target.load_state_dict(
                self.cost_critic.state_dict()
            )
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(), lr=critic_lr, weight_decay=0.0001
            )
            self.cost_criterion = nn.SmoothL1Loss() 
            # Lagrangian
            self.safe_threshold = torch.tensor(args.cost_budget)
            self._pid_kp = args.ctrl_pid_kp
            self._pid_ki = args.ctrl_pid_ki
            self._pid_kd = args.ctrl_pid_kd
            self._pid_d_delay = args.ctrl_pid_d_delay
            self._pid_delta_p_ema_alpha = args.ctrl_pid_delta_p_ema_alpha
            self._pid_delta_d_ema_alpha = args.ctrl_pid_delta_d_ema_alpha
            self._pid_i = args.ctrl_lagrangian_multiplier_init
            self._cost_ds = deque(maxlen=self._pid_d_delay)
            self._cost_ds.append(0.0)
            self._delta_p = 0.0
            self._cost_d = 0.0
            self._cost_penalty = 0.0
        elif "low_lag" in self.args.manager_algo:
            self._cost_penalty = 0.0
    
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
    
    def pid_update(self, ep_cost_avg):
        delta = float(ep_cost_avg - self.safe_threshold)
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

    def actor_loss(self, state, achieved_goal, goal, a_net, r_margin, 
                   cost_model=None, selected_landmark=None, controller_policy=None):
        actions = self.actor(state, goal)

        # HRAC high level reward loss
        if self.args.noise_man_training:
            actions = torch.clamp(actions + torch.normal(mean=0, 
                                                std=self.args.man_safe_noise_sigma, 
                                  size=actions.shape).type('torch.FloatTensor').to(device), 
                                  min=-self.torch_scale, max=self.torch_scale)
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        actor_loss = eval + norm

        scaled_norm_direction = var(torch.FloatTensor([0.] * self.action_dim))
        gen_subgoal = actions if self.absolute_goal else achieved_goal + actions
        if not(a_net is None):
            goal_loss = torch.clamp(F.pairwise_distance(
                a_net(achieved_goal), a_net(gen_subgoal)) - r_margin, min=0.).mean()
            
        # ITES high level cost loss
        safety_losses = {}
        if "safe_cls" in self.args.manager_algo:
            if not self.absolute_goal:
                safety_subgoal_cls_loss = cost_model.safe_model(actions + self.phi(state), state)
            else:
                safety_subgoal_cls_loss = cost_model.safe_model(actions, state)
            safety_subgoal_cls_loss = safety_subgoal_cls_loss.mean()
            safety_subgoal_cls_loss = self.coef_safety_modelfree * safety_subgoal_cls_loss
            safety_losses["safety_subgoal_cls_loss"] = safety_subgoal_cls_loss
        if "high_lag" in self.args.manager_algo:
            safety_lag_loss = self.cost_critic.Q1(state, goal, actions).mean()
            safety_losses["safety_lag_loss"] = safety_lag_loss
        elif "low_lag" in self.args.manager_algo:            
            ctrl_actions = controller_policy.actor(controller_policy.clean_obs(state), actions) 
            safety_lag_loss = controller_policy.cost_critic.Q1(state, actions, ctrl_actions).mean()
            safety_losses["safety_lag_loss"] = safety_lag_loss

        return actor_loss, \
               goal_loss, \
               safety_losses, \
               None, scaled_norm_direction

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

    def train(self, controller_policy, replay_buffer, controller_replay_buffer, cost_model, 
              iterations, batch_size=100, discount=0.99,
              tau=0.005, a_net=None, r_margin=None, total_timesteps=None, ep_cost=None):
        avg_act_loss, avg_crit_loss, avg_ld_loss = 0., 0., 0.
        avg_scaled_norm_direction = get_tensor(np.array([0.] * self.action_dim)).squeeze()
        debug_maganer_info = {"sum_manager_actor_grad_norm": 0}
        if a_net is not None:
            avg_goal_loss = 0.
        if "safe_cls" in self.args.manager_algo:
            avg_safety_subgoals_loss = 0.
        else:
            avg_safety_subgoals_loss = None

        for it in range(iterations):
            if "high_lag" in self.args.manager_algo:        
                x, y, x_ag, y_ag, g, sgorig, r, d, c, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            else:
                x, y, x_ag, y_ag, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            batch_size = min(batch_size, x.shape[0])

            if self.correction and not self.absolute_goal:
                sg = self.off_policy_corrections(controller_policy, batch_size,
                                                 sgorig, xobs_seq, a_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            achieved_goal = get_tensor(x_ag)
            next_achieved_goal = get_tensor(y_ag)
        
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            reward = get_tensor(r)
            if "high_lag" in self.args.manager_algo:
                cost = get_tensor(c)
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
            if "td3_adj_safe_cls_high_lag" in self.args.manager_algo:
                # Cost critic
                target_C1, target_C2 = self.cost_critic_target(
                    next_state, goal, next_action
                )
                target_C = torch.max(target_C1, target_C2)
                target_C = cost + (done * discount * target_C)
                target_C_no_grad = target_C.detach()

                # Get current C estimate
                current_C1, current_C2 = self.cost_critic(state, goal, subgoal)

                # Compute cost critic loss
                cost_critic_loss = self.cost_criterion(
                    current_C1, target_C_no_grad
                ) + self.cost_criterion(current_C2, target_C_no_grad)

                # Optimize the cost critic
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            # Compute actor loss
            assert a_net is not None
            actor_loss, goal_loss, safety_loss, _, _ = \
                self.actor_loss(state, achieved_goal, goal, a_net, r_margin, cost_model, 
                                selected_landmark=None, controller_policy=controller_policy)

            if not(a_net is None):
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
            if "lag" in self.args.manager_algo: 
                actor_loss = (actor_loss + safety_loss["safety_lag_loss"] * self._cost_penalty) / (1 + self._cost_penalty)
            if "safe_cls" in self.args.manager_algo:
                actor_loss = actor_loss + safety_loss["safety_subgoal_cls_loss"]

            # Optimize the actor
            self.actor_optimizer.zero_grad()
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
            if a_net is not None:
                avg_goal_loss += goal_loss
            if "safe_cls" in self.args.manager_algo:
                avg_safety_subgoals_loss += sum([loss_s.item() for _, loss_s in safety_loss.items()])

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if "high_lag" in self.args.manager_algo:
                for param, target_param in zip(
                    self.cost_critic.parameters(),
                    self.cost_critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if "high_lag" in self.args.manager_algo:
                self.pid_update(ep_cost)

        if "high_lag" in self.args.manager_algo:
            debug_maganer_info["man_lagrangian"] = self._cost_penalty

        debug_maganer_info["sum_manager_actor_grad_norm"] /= iterations
        if "safe_cls" in self.args.manager_algo:
            avg_safety_subgoals_loss = avg_safety_subgoals_loss / iterations
        return avg_act_loss / iterations, avg_crit_loss / iterations, avg_goal_loss / iterations, avg_safety_subgoals_loss, debug_maganer_info

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
        
    def load(self, dir, env_name, algo, exp_num, load_wm_as_pkl=True):
        self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActor.pth".format(dir, exp_num, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCritic.pth".format(dir, exp_num, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActorTarget.pth".format(dir, exp_num, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCriticTarget.pth".format(dir, exp_num, env_name, algo)))
                
class CostModel(object):
    """
        cost_model: G x S -> [0, 1]
        phi: S -> G
    """
    def __init__(self, state_dim, goal_dim, lidar_observation, 
                       frame_stack_num, 
                       safe_model_loss_coef, lr, cm_hidden_size,
                       regression_cost_model=False,
                       phi=None):
        self.lidar_observation = lidar_observation
        self.safe_model_loss_coef = safe_model_loss_coef        
        self.frame_stack_num = frame_stack_num
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.safe_model = ControllerSafeModel(self.goal_dim, 
                                              self.state_dim * self.frame_stack_num, 
                                              cm_hidden_size).to(device)
        self.phi = phi

        class BinaryFocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                super(BinaryFocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
            def forward(self, inputs, targets):
                p = torch.sigmoid(inputs)
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                pt = torch.where(targets == 1, p, 1 - p)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        if regression_cost_model:
            self.safe_model_criterion = nn.MSELoss()
        else:
            self.safe_model_criterion = nn.BCELoss()
        #self.safe_model_criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.safe_model_optimizer = torch.optim.Adam(self.safe_model.parameters(), lr=lr, weight_decay=0.0001)
        

    def train_cost_model(self, replay_buffer, 
                         cost_model_iterations=10, 
                         cost_model_batch_size=128):
        debug_info = {}
        debug_info["safe_model_loss"] = []
        debug_info["safe_model_mean_true"] = []
        debug_info["safe_model_mean_pred"] = []
        for _ in range(cost_model_iterations):
            assert replay_buffer.name == "cost_trajectory_buffer"
            ac_g, x, c = replay_buffer.sample(cost_model_batch_size)
            ac_goal = get_tensor(ac_g, to_device=False)
            ac_goal_goal_device = ac_goal.to(device)
            state = get_tensor(x, to_device=False)
            state_device = state.to(device)
            cost = get_tensor(c, to_device=False)
            cost_device = cost.to(device)

            safe_model_loss, true, pred = self.train_batch_cost_model(ac_goal_goal_device, state_device, cost_device)
            debug_info["safe_model_loss"].append(safe_model_loss.mean().cpu().detach())
            debug_info["safe_model_mean_true"].append(true.float().mean().cpu().detach())
            debug_info["safe_model_mean_pred"].append(pred.float().mean().cpu().detach())
        return debug_info
    
    def train_batch_cost_model(self, ac_goal, state, cost):
        pred = self.safe_model(ac_goal, state)
        true = cost
        safe_model_loss = self.safe_model_loss_coef * self.safe_model_criterion(pred, true)
        
        self.safe_model_optimizer.zero_grad()
        safe_model_loss.backward()
        self.safe_model_optimizer.step()

        return safe_model_loss, true, pred
    

    def save(self, dir, env_name, algo, exp_num):
        torch.save(self.safe_model.state_dict(), "{}/{}/{}_{}_SafeModel.pth".format(dir, exp_num, env_name, algo))
        
    def load(self, dir, env_name, algo, exp_num):
        self.safe_model.load_state_dict(torch.load("{}/{}/{}_{}_SafeModel.pth".format(dir, exp_num, env_name, algo)))


class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr, hidden_size,
                 critic_lr, repr_dim=15, no_xy=True, policy_noise=0.2, noise_clip=0.5,
                 absolute_goal=False, 
                 controller_grad_clip=0, controller_safety_coef=0, 
                 controller_cumul_img_safety=False,
                 img_horizon=10,
                 safe_threshold=None,
                 algo="td3",
                 sac_alpha=None,
                 phi=None,
                 pose=None,
                 args=None,

    ):
        self.device = device
        self.args = args
        self.phi = phi
        self.pose = pose
        self.torch_scale = torch.tensor(max_action).type('torch.FloatTensor').to(device)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.criterion = nn.SmoothL1Loss()  
        self.controller_cumul_img_safety = controller_cumul_img_safety
        self.algo = algo

        self.sac_alpha = sac_alpha

        self.controller_safety_coef = controller_safety_coef
        self.img_horizon = img_horizon
        self.controller_grad_clip = controller_grad_clip
        if "lag" in self.algo or "low_lag" in self.args.manager_algo:
            self.safe_threshold = torch.tensor(safe_threshold)
            self._pid_kp = args.ctrl_pid_kp
            self._pid_ki = args.ctrl_pid_ki
            self._pid_kd = args.ctrl_pid_kd
            self._pid_d_delay = args.ctrl_pid_d_delay
            self._pid_delta_p_ema_alpha = args.ctrl_pid_delta_p_ema_alpha
            self._pid_delta_d_ema_alpha = args.ctrl_pid_delta_d_ema_alpha
            self._pid_i = args.ctrl_lagrangian_multiplier_init
            self._cost_ds = deque(maxlen=self._pid_d_delay)
            self._cost_ds.append(0.0)
            self._delta_p = 0.0
            self._cost_d = 0.0
            self._cost_penalty = 0.0

        self.actor = ControllerActor(state_dim, goal_dim, action_dim, hidden_size,
                                    scale=max_action, sac="sac" in self.algo).to(device)
        if "td3" in self.algo:
            self.actor_target = ControllerActor(state_dim, goal_dim, action_dim, hidden_size,
                                                scale=max_action, sac="sac" in self.algo).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim, hidden_size).to(device)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.0001)
        
        if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:
            class FocalLossRegression(nn.Module):
                def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
                    super(FocalLossRegression, self).__init__()
                    self.gamma = gamma
                    self.alpha = alpha
                    self.reduction = reduction
                def forward(self, y_pred, y_true):
                    error = torch.abs(y_pred - y_true)
                    weights = torch.pow(error, self.gamma)
                    loss = self.alpha * weights * error
                    if self.reduction == 'mean':
                        return loss.mean()
                    elif self.reduction == 'sum':
                        return loss.sum()
                    elif self.reduction == 'none':
                        return loss
                    else:
                        raise ValueError(f"Unsupported reduction: {self.reduction}")
            self.cost_critic = ControllerCritic(state_dim, goal_dim, action_dim, hidden_size).to(device)
            self.cost_critic_target = ControllerCritic(state_dim, goal_dim, action_dim, hidden_size).to(device)
            self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(), lr=critic_lr, weight_decay=0.0001
            )
            self.cost_criterion = nn.SmoothL1Loss() 
            #self.cost_criterion = FocalLossRegression(gamma=2.0, alpha=1.0)


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
        if "td3" in self.algo:
            action = self.actor(state, sg).cpu().data.numpy().squeeze()
        elif "sac" in self.algo:
            action, _ = self.actor.sample_action_logprob(state, sg)
            action = action.cpu().data.numpy().squeeze()

        return action

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)
    
    def state_safety_on_horizon(self, state, actions, 
                                controller_policy, 
                                cost_model, 
                                all_steps_safety=False, 
                                train=False,
                                predict_env=None,
                                return_img_states=False,
                                manager_policy=None):

        assert not(predict_env is None), "world model must be initialized"
        manager_proposed_goal = actions.clone()
        next_img_state = state.clone()
        safety_cost = cost_model.safe_model
        
        if self.algo in ["td3_img_safe_c_cost"]:
            safeties_c_cost = []
        h = 0
        if all_steps_safety:
            safeties = []    
            horizon = self.img_horizon
        else:
            horizon = random.randint(1, self.img_horizon)
        img_states = []
        while h < horizon:
            img_state = next_img_state
            img_states.append(img_state)
            ctrl_actions = controller_policy.actor(controller_policy.clean_obs(img_state), manager_proposed_goal) 
            if self.args.noise_ctr_training:
                ctrl_actions = torch.clamp(ctrl_actions + torch.normal(mean=0, 
                                                    std=self.args.ctr_safe_noise_sigma, 
                                           size=actions.shape).type('torch.FloatTensor').to(device), 
                                           min=-self.torch_scale, max=self.torch_scale)

            next_img_state = predict_env.step(img_state, ctrl_actions, 
                                              deterministic=True, 
                                              torch_deviced=True)
            if self.algo in ["td3_img_safe_c_cost"]:
                safety_c_cost = self.cost_critic.Q1(img_state, manager_proposed_goal, ctrl_actions).mean()
                safeties_c_cost.append(safety_c_cost)
            if all_steps_safety:
                safety = safety_cost(self.phi(next_img_state), next_img_state)
                safeties.append(safety)
            if self.args.manager_algo == "none":
                absolute_goal = False
            else:
                absolute_goal = manager_policy.absolute_goal
            manager_proposed_goal = subgoal_transition(self.phi(img_state), 
                                                        manager_proposed_goal, 
                                                        self.phi(next_img_state),
                                                        absolute_goal)
            h += 1
        if not all_steps_safety:
            safety = safety_cost(self.phi(next_img_state), next_img_state)
        else:
            safety = 0
            for el in safeties:
                safety = safety + el
            if train:
                safety /= self.img_horizon

        if self.algo in ["td3_img_safe_c_cost"]:
            c_cost_safety = 0
            for el in safeties_c_cost:
                c_cost_safety = c_cost_safety + el            
            safety = safety + c_cost_safety
        if return_img_states:
            return safety, img_states
        return safety

    def actor_loss(self, state, sg, init_state, cost_model, predict_env, manager_policy):
        # HRAC low level reward loss
        if "td3" in self.algo:
            action = self.actor(state, sg)
            actor_loss = -self.critic.Q1(state, sg, action).mean()
        elif "sac" in self.algo:
            action, log_prob = self.actor.sample_action_logprob(state, sg)
            actor_loss = (self.sac_alpha * log_prob - self.critic.Q1(state, sg, action)).mean()
        else:
            assert 1 == 0

        # ITES low level cost loss
        if self.algo in ["td3_lag", "sac_lag"]:
            safety_loss = self.cost_critic.Q1(state, sg, action).mean()
            actor_loss = (actor_loss + safety_loss * self._cost_penalty) / (1 + self._cost_penalty)
        elif "img_safe" in self.algo:
            safety_loss = self.state_safety_on_horizon(init_state, sg, 
                                                       controller_policy=self, 
                                                       cost_model=cost_model,
                                                       all_steps_safety=self.controller_cumul_img_safety,
                                                       train=self.controller_cumul_img_safety,
                                                       predict_env=predict_env, 
                                                       manager_policy=manager_policy)
            if "lag" in self.algo:
                actor_loss = (actor_loss + self._cost_penalty * safety_loss.mean()) / (1 + self._cost_penalty)
            else:
                actor_loss += self.controller_safety_coef * safety_loss.mean()
   
        return actor_loss

    def multi_subgoal_transition(self, states, subgoal):
        assert 1 == 0, "achieved goal instead of states should be"
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals
    
    def pid_update(self, ep_cost_avg):
        delta = float(ep_cost_avg - self.safe_threshold)
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

    def train(self, replay_buffer, cost_model, predict_env, manager_policy, iterations, 
              batch_size=100, discount=0.99, tau=0.005, ep_cost=None):
        avg_act_loss, avg_crit_loss = 0., 0.
        if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:
            avg_cost_loss = 0.
        debug_info = {}
        for _ in range(iterations):      
            if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:        
                x, y, x_ag, y_ag, sg, u, r, d, c, _, _ = replay_buffer.sample(batch_size)
            else:
                x, y, x_ag, y_ag, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            init_state = get_tensor(x)
            if self.args.manager_algo == "none":
                absolute_goal = False
            else:
                absolute_goal = manager_policy.absolute_goal
            next_g = get_tensor(subgoal_transition(x_ag, sg, y_ag, absolute_goal))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y)) 
            if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo: 
                cost = get_tensor(c)
            noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
            if "td3" in self.algo:
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state, next_g) + noise)
                next_action = torch.min(next_action, self.actor.scale)
                next_action = torch.max(next_action, -self.actor.scale)
            elif "sac" in self.algo:
                with torch.no_grad():
                    next_action, next_log_prob = self.actor.sample_action_logprob(next_state, next_g)

            # Reward Critic
            target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if "sac" in self.algo:
                target_Q = target_Q - self.sac_alpha * next_log_prob
            target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()
            current_Q1, current_Q2 = self.critic(state, sg, action)
            critic_loss = self.criterion(current_Q1, target_Q_no_grad) \
                                + self.criterion(current_Q2, target_Q_no_grad)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            cost_critic_loss = 0
            if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:
                # Cost Critic
                target_C1, target_C2 = self.cost_critic_target(
                    next_state, next_g, next_action
                )
                target_C = torch.max(target_C1, target_C2)
                target_C = cost + (done * discount * target_C)
                if "sac" in self.algo:
                    target_C = target_C - self.sac_alpha * next_log_prob
                target_C_no_grad = target_C.detach()
                current_C1, current_C2 = self.cost_critic(state, sg, action)
                cost_critic_loss = self.cost_criterion(current_C1, target_C_no_grad) \
                                        + self.cost_criterion(current_C2, target_C_no_grad)
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, sg, init_state, cost_model, predict_env, manager_policy)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                    max_norm=self.controller_grad_clip)
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:
                avg_cost_loss += cost_critic_loss
            
            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"] or "low_lag" in self.args.manager_algo:
                for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if "td3" in self.algo:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if "lag" in self.algo and ep_cost is not None:
                self.pid_update(ep_cost)

        if self.algo in ["td3_img_safe_c_cost", "td3_lag", "sac_lag"]:
            debug_info["controller_safe_critic_loss"] = avg_cost_loss / iterations
        if "lag" in self.algo and ep_cost is not None:
            debug_info["lagrangian"] = self._cost_penalty
        debug_info["controller_actor_loss"] = avg_act_loss / iterations
        debug_info["controller_critic_loss"] = avg_crit_loss / iterations

        return debug_info

    def save(self, dir, env_name, algo, exp_num):
        torch.save(self.actor.state_dict(), "{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo))
        if "td3" in self.algo:
            torch.save(self.actor_target.state_dict(), "{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo))

    def load(self, dir, env_name, algo, exp_num):
        self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo)))
        if "td3" in self.algo:
            self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo)))

    def pairwise_value(self, obs, ag, goal, absolute_goal):
        assert ag.shape[0] == goal.shape[0]
        with torch.no_grad():
            if not absolute_goal:
                relative_goal = goal - ag
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, relative_goal)
                dist1, dist2 = self.critic(cleaned_obs, relative_goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)
            else:
                cleaned_obs = self.clean_obs(obs)
                actions = self.actor(cleaned_obs, goal)
                dist1, dist2 = self.critic(cleaned_obs, goal, actions)
                dist = torch.min(dist1, dist2)
                return dist.squeeze(-1)
