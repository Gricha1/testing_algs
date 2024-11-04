import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

#torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorWrapper:
    def __init__(self):
        pass

    def __enter__(self):
        self.prev_type = torch.get_default_dtype()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_tensor_type(torch.FloatTensor)


class StandardScaler(object):
    def __init__(self):
        pass

    def set_mu_std(self, mu, std):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data, torch_deviced=False):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if torch_deviced:
            mu_tensor = torch.tensor(self.mu, device=device)
            std_tensor = torch.tensor(self.std, device=device)
            return (data - mu_tensor) / std_tensor
        else:
            return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #reporter = MemReporter()
        #reporter.report()
    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #print("input device:", input.device)
        #print("self.weight device:", self.weight.device)
        #assert 1 == 0
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    #@profile
    def __init__(self, state_size, action_size, reward_size, cost_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = state_size # + reward_size + cost_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights)
        self.swish = Swish()
        #reporter = MemReporter()
        #reporter.report()
    #@profile
    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class EnsembleDynamicsModel():
    #@profile
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=0, cost_size=0, hidden_size=200, learning_rate=1e-3, use_decay=False):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.cost_size = cost_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, cost_size, network_size, hidden_size, learning_rate=learning_rate, use_decay=use_decay)
        self.scaler = StandardScaler()

    def set_elite_model_idxes(self, elite_model_idxes):
        self.elite_model_idxes = elite_model_idxes

    #@profile
    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        #print(train_inputs.shape, holdout_inputs.shape)
        # holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        # holdout_labels = torch.from_numpy(holdout_labels).float().to(device)


        # holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        # holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        #print(holdout_inputs.shape)
        for epoch in itertools.count():
            #--------training------------
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            #print(train_idx)
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            losses = []
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                #print("idx[0]:", idx[0])
                #print("idx shape:", idx.shape)
                #print("train_inputs[0]:", train_inputs[0])
                #print("train_inputs shape:", train_inputs.shape)
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                #print(train_input.shape)
                #print("Size occupied on cuda",sys.getsizeof(train_input))
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)

                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, mtrain = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(mtrain)
            #-----validation------------------
            val_idx = np.vstack([np.random.permutation(holdout_inputs.shape[0]) for _ in range(self.network_size)])
            val_batch_size = 512
            val_losses_list = []
            len_valid = 0
            for start_pos in range(0, holdout_inputs.shape[0], val_batch_size):
                with torch.no_grad():
                    idx = val_idx[:, start_pos: start_pos + val_batch_size]
                    val_input = torch.from_numpy(holdout_inputs[idx]).float().to(device)
                    val_label = torch.from_numpy(holdout_labels[idx]).float().to(device)
                    holdout_mean, holdout_logvar = self.ensemble_model(val_input, ret_log_var=True)
                    _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, val_label, inc_var_loss=False)
                    holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                    val_losses_list.append(holdout_mse_losses)
                len_valid+=1
            #print(val_losses)
            val_losses = np.array(val_losses_list)
            val_losses = np.sum(val_losses,axis=0)/len_valid
            sorted_loss_idx = np.argsort(val_losses)
            self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
            break_train = self._save_best(epoch, val_losses)
            if break_train:
                break
            train_mse_losses = []
            for i in losses:
                train_mse_losses.append(i.detach().cpu().numpy())

            #print('epoch: {}, train mse losses: {}'.format(epoch, np.mean(train_mse_losses,axis=0)))
            #print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))

            return epoch, np.mean(np.mean(train_mse_losses,axis=0))

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True, torch_deviced=False):
        inputs = self.scaler.transform(inputs, torch_deviced=torch_deviced)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            if torch_deviced:
                input = inputs[i:min(i + batch_size, inputs.shape[0]), :]
            else:
                input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            if torch_deviced:
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            else:
                ensemble_mean.append(b_mean.detach().cpu().numpy())
                ensemble_var.append(b_var.detach().cpu().numpy())
        if torch_deviced:
            ensemble_mean = torch.cat(ensemble_mean, dim=-1)
            ensemble_var = torch.cat(ensemble_var, dim=-1)
        else:
            ensemble_mean = np.hstack(ensemble_mean)
            ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

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
    

class PredictEnv:
    def __init__(self, model, env_name, model_type, testing_mean_wm):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.testing_mean_wm = testing_mean_wm

    def train_world_model(self, replay_buffer, batch_size=256):
        if replay_buffer.cost_memmory:
            x, y, _, u, _, c, _, _, _ = replay_buffer.sample(len(replay_buffer))
        else:
            x, y, _, u, _, _, _, _ = replay_buffer.sample(len(replay_buffer))
        state = get_tensor(x, to_device=False)
        action = get_tensor(u, to_device=False)
        next_state = get_tensor(y, to_device=False)

        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)

        labels = delta_state.numpy()
        _, loss = self.model.train(inputs, labels, batch_size=batch_size, holdout_ratio=0.2)
        del state, action, next_state
        
        return loss

    def imagine_state(self, prev_imagined_state, prev_action, current_state, current_step, imagined_state_freq):
        with torch.no_grad():
            if prev_imagined_state is None or current_step % imagined_state_freq == 0:
                imagined_state = current_state
            else:
                imagined_state = self.step(prev_imagined_state, 
                                            prev_action, 
                                            deterministic=True)
        return imagined_state

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        else:
            return False

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, single=False, deterministic=False, torch_deviced=False):
        testing_mean_pred = self.testing_mean_wm

        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        if torch_deviced:
            inputs = torch.cat((obs, act), dim=-1)
        else:
            inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, torch_deviced=torch_deviced)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        #print(ensemble_model_means.shape, ensemble_model_vars.shape)
        #random_idx = np.random.randint(5,size=1)#
        # ensemble_model_means[random_idx] += obs
        # test
        if not deterministic:
            if torch_deviced:
                ensemble_model_stds = torch.sqrt(ensemble_model_vars)
            else:
                ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        if testing_mean_pred:
            samples = ensemble_samples.mean(0)
        else:
            samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        # test
        #model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        # test
        #log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        next_obs_delta = samples
        next_obs = next_obs_delta + obs
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # batch_size = model_means.shape[0]
        # return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        # return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            # reward = rewards[0]
            # cost = costs[0]
            #return_means = return_means[0]
            #return_stds = return_stds[0]

            #terminals = terminals[0]

        #info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return  next_obs #, reward, cost

    def step_elite(self, obs, act, idx, single=False, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        #print(ensemble_model_means.shape, ensemble_model_vars.shape)
        #random_idx = np.random.randint(5,size=1)#
        # ensemble_model_means[random_idx] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice([idx], size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        next_obs_delta = samples
        #print(obs, samples)
        next_obs = next_obs_delta + obs
        #print(next_obs)
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # batch_size = model_means.shape[0]
        # return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        # return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            # reward = rewards[0]
            # cost = costs[0]
            #return_means = return_means[0]
            #return_stds = return_stds[0]

            #terminals = terminals[0]

        #info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return  next_obs
    


    def save(self, dir, env_name, algo, exp_num):
        # save as pkl
        torch.save(self.model, "{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
        # save as state dict + scaler
        torch.save(self.model.ensemble_model.state_dict(), "{}/{}/{}_{}_env_model.pth".format(dir, exp_num, env_name, algo))
        # save: scalar mu, scalar std, model elite idxs 
        mu = self.model.scaler.mu
        std = self.model.scaler.std
        elite_model_idxes = np.array(self.model.elite_model_idxes)
        np.save("{}/{}/{}_{}_wm_scaler_mu.npy".format(dir, exp_num, env_name, algo), mu)
        np.save("{}/{}/{}_{}_wm_scaler_std.npy".format(dir, exp_num, env_name, algo), std)
        np.save("{}/{}/{}_{}_wm_elite_model_idxes.npy".format(dir, exp_num, env_name, algo), elite_model_idxes)


    def load(self, dir, env_name, algo, exp_num, load_wm_as_pkl=True):
        temp_env_name = 'safepg2'
        temp_model_type='pytorch'
        if load_wm_as_pkl:
            env_model = torch.load("{}/{}/{}_{}_env_model.pkl".format(dir, exp_num, env_name, algo))
            #predict_env = PredictEnv(env_model, temp_env_name, temp_model_type)
            #self = predict_env
            self.model = env_model
        else:
            self.model.ensemble_model.load_state_dict(torch.load("{}/{}/{}_{}_env_model.pth".format(dir, exp_num, env_name, algo)))
            mu = np.load("{}/{}/{}_{}_wm_scaler_mu.npy".format(dir, exp_num, env_name, algo))
            std = np.load("{}/{}/{}_{}_wm_scaler_std.npy".format(dir, exp_num, env_name, algo))
            self.model.scaler.set_mu_std(mu, std)
            elite_model_idxes = np.load("{}/{}/{}_{}_wm_elite_model_idxes.npy".format(dir, exp_num, env_name, algo))
            self.model.set_elite_model_idxes(elite_model_idxes)