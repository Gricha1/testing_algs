import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from hrac.models import ControllerActor, ControllerCritic, \
    ManagerActor, ManagerCritic, PPOAgent, ControllerSafeModel

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
                 wm_no_xy=False, safety_subgoals=False, safety_loss_coef=1, img_horizon=10, 
                 cost_function=None, testing_safety_subgoal=False):
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

        self.predict_env = None
        self.no_xy = wm_no_xy
        self.safety_subgoals = safety_subgoals
        self.safety_loss_coef = safety_loss_coef
        self.img_horizon = img_horizon
        self.cost_function = cost_function
        self.testing_safety_subgoal = testing_safety_subgoal

    def set_predict_env(self, predict_env):
        self.predict_env = predict_env

    def imagine_state(self, prev_imagined_state, prev_action, current_state, current_step, imagined_state_freq):
        with torch.no_grad():
            if prev_imagined_state is None or current_step % imagined_state_freq == 0:
                imagined_state = current_state
            else:
                imagined_state = self.predict_env.step(prev_imagined_state, prev_action)
        return imagined_state

    def train_world_model(self, replay_buffer):
        x, y, sg, u, r, d, _, _ = replay_buffer.sample(len(replay_buffer))
        state = get_tensor(x, to_device=False)
        action = get_tensor(u, to_device=False)
        next_state = get_tensor(y, to_device=False)

        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)

        labels = delta_state.numpy()

        epoch, loss = self.predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
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

    def actor_loss(self, state, goal, a_net, r_margin, controller_policy=None):
        actions = self.actor(state, goal)
        
        eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        goal_loss = None
        safety_loss = None
        if not(a_net is None):
            goal_loss = torch.clamp(F.pairwise_distance(
                a_net(state[:, :self.action_dim]), a_net(state[:, :self.action_dim] + actions)) - r_margin, min=0.).mean()
        if self.safety_subgoals:
            assert self.testing_safety_subgoal or (not self.testing_safety_subgoal and not(self.predict_env is None)), "world model must be initialized"
            if self.testing_safety_subgoal:
                copy_state = state.cpu().detach().numpy()
                copy_actions = actions.cpu().detach().numpy()
                if self.absolute_goal:
                        manager_absolute_goal = copy_actions[:, :self.action_dim]
                else:
                    manager_absolute_goal = copy_state[:, :self.action_dim] + copy_actions[:, :self.action_dim]
                cost_indexes = torch.tensor(self.cost_function(manager_absolute_goal), dtype=torch.int)
                cost_violate_indexes = (cost_indexes).bool()
                safe_indexes = (1 - cost_indexes).bool()
                # if unsafe state loss = 1, loss = 0 otherwise
                safety_loss = torch.abs(actions[:, 0]) + torch.abs(actions[:, 1])
                zero_indexes = (safety_loss == 0).cpu().detach()
                cost_violate_indexes[zero_indexes] = False
                safe_indexes[zero_indexes] = True
                safety_loss[cost_violate_indexes] = safety_loss[cost_violate_indexes] / safety_loss[cost_violate_indexes]
                safety_loss[safe_indexes] = safety_loss[safe_indexes] * 0
                safety_loss = safety_loss.mean()
            else:
                safety_loss = 0
                h = 0
                img_state = state
                acc_costs = np.zeros(img_state.shape[0]) # (batch_size,)
                while h < self.img_horizon:
                    # subgoals = actions
                    ctrl_actions = controller_policy.actor(img_state, actions) 
                    img_state = img_state.cpu().numpy()
                    img_actions = img_actions.cpu().numpy()
                    #acc_costs += self.cost_function(img_state)
                    img_state = self.predict_env.step(img_state, ctrl_actions)
                    img_state = torch.from_numpy(img_state).float().to(device)
                    h += 1
                safety_loss = (acc_costs/self.img_horizon).mean()
        return eval + norm, goal_loss, safety_loss

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

    def train(self, controller_policy, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, a_net=None, r_margin=None):
        print("train subgoal policy")
        avg_act_loss, avg_crit_loss = 0., 0.
        if a_net is not None:
            avg_goal_loss = 0.
        if self.safety_subgoals:
            avg_safety_subgoals_loss = 0.
        else:
            avg_safety_subgoals_loss = None
        for it in range(iterations):
            # Sample replay buffer
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

            # Compute actor loss
            actor_loss, goal_loss, safety_subgoals_loss = self.actor_loss(state, goal, a_net, r_margin, controller_policy)
            if not(a_net is None):
                actor_loss = actor_loss + self.goal_loss_coeff * goal_loss
            if self.safety_subgoals:
                actor_loss = actor_loss + self.safety_loss_coef * safety_subgoals_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()

            # test
            #actor_loss.retain_grad() 
            #goal_loss.retain_grad() 
            #safety_subgoals_loss.retain_grad() 

            actor_loss.backward()
            
            self.actor_optimizer.step()

            avg_act_loss += actor_loss
            avg_crit_loss += critic_loss
            if a_net is not None:
                avg_goal_loss += goal_loss
            if self.safety_subgoals:
                avg_safety_subgoals_loss += safety_subgoals_loss

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.safety_subgoals:
            avg_safety_subgoals_loss = avg_safety_subgoals_loss / iterations
        return avg_act_loss / iterations, avg_crit_loss / iterations, avg_goal_loss / iterations, avg_safety_subgoals_loss

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
        if not(self.predict_env is None):
            torch.save(self.predict_env.model, "{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, exp_num):
        self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActor.pth".format(dir, exp_num, env_name, algo)))
        self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCritic.pth".format(dir, exp_num, env_name, algo)))
        self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerActorTarget.pth".format(dir, exp_num, env_name, algo)))
        self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ManagerCriticTarget.pth".format(dir, exp_num, env_name, algo)))
        if not(self.predict_env is None):
            temp_env_name = 'safepg2'
            temp_model_type='pytorch'
            env_model = torch.load("{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
            predict_env = PredictEnv(env_model, temp_env_name, temp_model_type)
            self.set_predict_env(predict_env)
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo)))


class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr,
                 critic_lr, repr_dim=15, no_xy=True, policy_noise=0.2, noise_clip=0.5,
                 absolute_goal=False, PPO=False, ppo_lr=None, hidden_dim_ppo=300,
                 weight_decay_ppo=None, safe_model=False, cost_function=None,
    ):
        self.PPO = PPO
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal
        self.criterion = nn.SmoothL1Loss()    
        # self.criterion = nn.MSELoss()

        self.safe_model = safe_model
        if safe_model:
            assert not(cost_function is None)
            self.cost_function = cost_function
            self.safe_model = ControllerSafeModel(state_dim).to(device)
            self.safe_model_criterion = nn.CrossEntropyLoss()
            self.safe_model_optimizer = torch.optim.Adam(self.safe_model.parameters(),
                                                 lr=critic_lr, weight_decay=0.0001)

        if self.PPO:
            self.agent = PPOAgent(state_dim, goal_dim, action_dim,
                                  hidden_dim=hidden_dim_ppo, scale=max_action).to(device)
            if not(weight_decay_ppo is None): 
                self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=ppo_lr, eps=1e-5,
                                                weight_decay=weight_decay_ppo)  
            else:
                self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=ppo_lr, eps=1e-5)  
        else:
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
        if self.PPO: assert 1 == 0
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        return self.actor(state, sg).cpu().data.numpy().squeeze()

    def value_estimate(self, state, sg, action):
        if self.PPO: assert 1 == 0
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        if self.PPO: assert 1 == 0
        return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()

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

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, 
              minibatch_size=None, clip_coef=None, clip_vloss=None,
              norm_adv=None, max_grad_norm=None, vf_coef=None, ent_coef=None, target_kl=None,
              num_minibatches=None):
        avg_act_loss, avg_crit_loss = 0., 0.
        debug_info = {}
        debug_batch_data = True
        if self.safe_model:
            debug_info["safe_model_loss"] = []
        if self.PPO:
            x, _, sg, u, r, d, l, v, _, _ = replay_buffer.sample(batch_size)
            b_obs = self.clean_obs(torch.FloatTensor(x).to(device))
            b_goals = torch.FloatTensor(sg).to(device)
            b_logprobs = torch.FloatTensor(l).to(device)
            b_actions = torch.FloatTensor(u).to(device)
            b_advantages = torch.FloatTensor(replay_buffer.advantages).to(device)
            b_returns = torch.FloatTensor(replay_buffer.returns).to(device)
            b_values = torch.FloatTensor(v).to(device)
            b_done = torch.FloatTensor(d).to(device)

            if debug_batch_data:
                debug_info["batch_return"] = b_returns.mean().cpu()
                debug_info["batch_values"] = b_values.mean().cpu()
                debug_info["batch_advantages"] = b_advantages.mean().cpu()
                debug_info["batch_done"] = b_done.mean().cpu()
                debug_info["batch_norm_advantages"] = ((b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)).mean().cpu()

            b_inds = np.arange(batch_size)
            clipfracs = []
        for it in range(iterations):
            if not self.PPO:
                x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
                init_state = get_tensor(x)
                next_g = get_tensor(self.subgoal_transition(x, sg, y))
                state = self.clean_obs(get_tensor(x))
                action = get_tensor(u)
                sg = get_tensor(sg)
                done = get_tensor(1 - d)
                reward = get_tensor(r)
                next_state = self.clean_obs(get_tensor(y))                

            if self.PPO:
                 # Optimizing the policy and value network
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_goals[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    debug_info["value"] = newvalue.mean().cpu()
                    debug_info["action_logprob"] = newlogprob.mean().cpu()
                    debug_info["ratio"] = ratio.mean().cpu()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    debug_info["pg_loss1"] = pg_loss1.mean().cpu()
                    debug_info["pg_loss2"] = pg_loss2.mean().cpu()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                    self.optimizer.step()

                    avg_act_loss += pg_loss
                    avg_crit_loss += v_loss

                if target_kl is not None and approx_kl > target_kl:
                    break
            else:
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

                if self.safe_model:
                    pred = self.safe_model(init_state)
                    numpy_b_xy = init_state.cpu().detach().numpy()[:, :2]
                    true = torch.tensor(self.cost_function(numpy_b_xy), dtype=torch.long).to(device)

                    # Compute safet_model loss
                    safe_model_loss = self.safe_model_criterion(pred, true)

                    # Optimize the safe_model
                    self.safe_model_optimizer.zero_grad()
                    safe_model_loss.backward()
                    self.safe_model_optimizer.step()

                    debug_info["safe_model_loss"].append(safe_model_loss.mean().cpu().detach())

                # Compute actor loss
                actor_loss = self.actor_loss(state, sg)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                avg_act_loss += actor_loss
                avg_crit_loss += critic_loss

            if not self.PPO:
                # Update the target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if self.PPO:
                avg_act_loss = avg_act_loss / num_minibatches
                avg_crit_loss = avg_crit_loss / num_minibatches

        if self.safe_model:
            debug_info["safe_model_loss"] = np.mean(debug_info["safe_model_loss"])

        return avg_act_loss / iterations, avg_crit_loss / iterations, debug_info

    def save(self, dir, env_name, algo, exp_num):
        if self.PPO:
            pass
        else:
            torch.save(self.actor.state_dict(), "{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo))
            torch.save(self.critic.state_dict(), "{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo))
            torch.save(self.actor_target.state_dict(), "{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo))
            torch.save(self.critic_target.state_dict(), "{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo))
            # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo))
            # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, exp_num):
        if self.PPO:
            pass
        else:
            self.actor.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActor.pth".format(dir, exp_num, env_name, algo)))
            self.critic.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCritic.pth".format(dir, exp_num, env_name, algo)))
            self.actor_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerActorTarget.pth".format(dir, exp_num, env_name, algo)))
            self.critic_target.load_state_dict(torch.load("{}/{}/{}_{}_ControllerCriticTarget.pth".format(dir, exp_num, env_name, algo)))
            # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo)))
            # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo)))
