import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=300, scale=1):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim + goal_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x, g):
        x = torch.cat([x, g], 1)
        return self.critic(x)

    def get_action_and_value(self, x, g, action=None):
        x = torch.cat([x, g], -1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
    
    def forward(self, x, g=None, nonlinearity='tanh'):
        if g is not None:
            x = F.relu(self.l1(torch.cat([x, g], 1)))
        else:
            x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if nonlinearity == 'tanh':
            x = self.max_action * torch.tanh(self.l3(x)) 
        elif nonlinearity == 'sigmoid':
            x = self.max_action * torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, g=None, u=None):
        if g is not None:
            xu = torch.cat([x, g, u], 1)
        else:
            xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, g=None, u=None):
        if g is not None:
            xu = torch.cat([x, g, u], 1)
        else:
            xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class ControllerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=1):
        super().__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        self.scale = nn.Parameter(torch.tensor(scale).float(),
                                  requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)
    
    def forward(self, x, g):
        return self.scale*self.actor(x, g)


class ControllerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()

        self.critic = Critic(state_dim, goal_dim, action_dim)
    
    def forward(self, x, sg, u):
        return self.critic(x, sg, u)

    def Q1(self, x, sg, u):
        return self.critic.Q1(x, sg, u)


class ManagerActor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None, absolute_goal=False):
        super().__init__()
        if scale is None:
            scale = torch.ones(action_dim)
        self.scale = nn.Parameter(torch.tensor(scale[:action_dim]).float(), requires_grad=False)
        self.actor = Actor(state_dim, goal_dim, action_dim, 1)
        self.absolute_goal = absolute_goal
    
    def forward(self, x, g):
        if self.absolute_goal:
            return self.scale * self.actor(x, g, nonlinearity='sigmoid')
        else:
            return self.scale * self.actor(x, g)


class ManagerCritic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.critic = Critic(state_dim, goal_dim, action_dim)

    def forward(self, x, g, u):
        return self.critic(x, g, u)

    def Q1(self, x, g, u):
        return self.critic.Q1(x, g, u)


class ANet(nn.Module):

    def __init__(self, state_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
