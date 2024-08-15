from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .models import ControllerActor, ControllerCritic, CostCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Controller(object):
    def __init__(
        self, state_dim, goal_dim, action_dim, max_action, actor_lr, critic_lr,
        no_xy=True, policy_noise=0.2, noise_clip=0.5, absolute_goal=False,
        pid_kp=1e-6, pid_ki=1e-7, pid_kd=1e-7, pid_d_delay=10,
        pid_delta_p_ema_alpha=0.95, pid_delta_d_ema_alpha=0.95,
        lagrangian_multiplier_init=0., cost_limit=25., use_lagrange=False
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal
        self.criterion = nn.SmoothL1Loss() 
        self.cost_criterion = nn.SmoothL1Loss() 
        # self.criterion = nn.MSELoss()

        self.actor = ControllerActor(state_dim, goal_dim, action_dim,
                                     scale=max_action).to(device)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim,
                                            scale=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(
            state_dim, goal_dim, action_dim
        ).to(device)
        self.critic_target = ControllerCritic(
            state_dim, goal_dim, action_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=0.0001
        )
        
        self.use_lagrange = use_lagrange
        if self.use_lagrange:
            self.cost_critic = ControllerCritic(
                state_dim, goal_dim, action_dim
            ).to(device)
            self.cost_critic_target = ControllerCritic(
                state_dim, goal_dim, action_dim
            ).to(device)
            self.cost_critic_target.load_state_dict(
                self.cost_critic.state_dict()
            )
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(), lr=critic_lr, weight_decay=0.0001
            )

        # Lagrangian
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
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

    def select_action(self, state, sg, evaluation=False):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.actor(state, sg).cpu().data.numpy().squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        action = self.actor(state, sg)
        loss_r = -self.critic.Q1(state, sg, action).mean()
        if self.use_lagrange:
            loss_c = self.cost_critic.Q1(state, sg, action).mean()
            loss = (
                loss_r + loss_c * self._cost_penalty
            ) / (1 + self._cost_penalty)
        else:
            loss = loss_r
        return loss

    def subgoal_transition(self, state, subgoal, next_state):
        if self.absolute_goal:
            return subgoal
        else:
            if len(state.shape) == 1:  # check if batched
                return state[:self.goal_dim] + subgoal - \
                    next_state[:self.goal_dim]
            else:
                return state[:, :self.goal_dim] + subgoal -\
                       next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals

    def train(
        self, replay_buffer, iterations, batch_size=100, discount=0.99,
        tau=0.005
    ):
        avg_act_loss, avg_crit_loss, avg_cost_loss = 0., 0., 0
        for it in range(iterations):
            x, y, sg, u, r, d, c = replay_buffer.sample(batch_size)
            next_g = get_tensor(self.subgoal_transition(x, sg, y))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            cost = get_tensor(c)
            next_state = self.clean_obs(get_tensor(y))

            noise = torch.FloatTensor(u).data.normal_(
                0, self.policy_noise
            ).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_g) + noise)
            next_action = torch.min(next_action, self.actor.scale)
            next_action = torch.max(next_action, -self.actor.scale)

            target_Q1, target_Q2 = self.critic_target(
                next_state, next_g, next_action
            )
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
                target_C1, target_C2 = self.cost_critic_target(
                    next_state, next_g, next_action
                )
                target_C = torch.max(target_C1, target_C2)
                target_C = cost + (done * discount * target_C)
                target_C_no_grad = target_C.detach()

                # Get current C estimate
                current_C1, current_C2 = self.cost_critic(state, sg, action)

                # Compute cost critic loss
                cost_critic_loss = self.cost_criterion(
                    current_C1, target_C_no_grad
                ) + self.cost_criterion(current_C2, target_C_no_grad)

                # Optimize the cost critic
                self.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.cost_critic_optimizer.step()

            # Compute actor loss
            actor_loss = self.actor_loss(state, sg)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            avg_cost_loss += cost_critic_loss

            # Update the target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            
            if self.use_lagrange:
                for param, target_param in zip(
                    self.cost_critic.parameters(),
                    self.cost_critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        return avg_act_loss / iterations, avg_crit_loss / iterations, \
            avg_cost_loss / iterations
    
    def update_lag(self, episode_cost):
        delta = float(episode_cost - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)

        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta

        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(episode_cost)

        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)

        self._cost_ds.append(self._cost_d)

    def lagrangian(self):
        return self._cost_penalty


    def save(self, dir, env_name, algo):
        torch.save(
            self.actor.state_dict(),
            "{}/{}_{}_ControllerActor.pth".format(dir, env_name, algo)
        )
        torch.save(
            self.critic.state_dict(),
            "{}/{}_{}_ControllerCritic.pth".format(dir, env_name, algo)
        )
        torch.save(
            self.actor_target.state_dict(),
            "{}/{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo)
        )
        torch.save(
            self.critic_target.state_dict(),
            "{}/{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo)
        )
        if self.use_lagrange:
            torch.save(
                self.cost_critic.state_dict(),
                "{}/{}_{}_CostCritic.pth".format(dir, env_name, algo)
            )
            torch.save(
                self.cost_critic_target.state_dict(),
                "{}/{}_{}_CostCriticTarget.pth".format(dir, env_name, algo)
            )
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo):
        self.actor.load_state_dict(torch.load(
            "{}/{}_{}_ControllerActor.pth".format(dir, env_name, algo)
        ))
        self.critic.load_state_dict(torch.load(
            "{}/{}_{}_ControllerCritic.pth".format(dir, env_name, algo)
        ))
        self.actor_target.load_state_dict(torch.load(
            "{}/{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo)
        ))
        self.critic_target.load_state_dict(torch.load(
            "{}/{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo)
        ))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo)))