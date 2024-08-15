import numpy as np


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6):
        self.storage = [[] for _ in range(7)]
        self.maxsize = maxsize
        self.next_idx = 0

    def clear(self):
        self.storage = [[] for _ in range(7)]
        self.next_idx = 0

    # Expects tuples of (x, x', g, u, r, d, cost)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            for array, datapoint in zip(self.storage, data):
                array.append(datapoint)
        else:
            for array, datapoint in zip(self.storage, data):
                array.__setitem__(self.next_idx, datapoint) 

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        if len(self.storage[0]) <= batch_size:
            ind = np.arange(len(self.storage[0]))
        else:
            ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, g, u, r, d, c = [], [], [], [], [], [], []         

        for i in ind: 
            X, Y, G, U, R, D, C = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            c.append(np.array(C, copy=False))
        
        return np.array(x), np.array(y), np.array(g), np.array(u), \
            np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
            np.array(c).reshape(-1, 1)

    def save(self, file):
        np.savez_compressed(
            file, idx=np.array([self.next_idx]), x=self.storage[0],
            y=self.storage[1], g=self.storage[2], u=self.storage[3],
            r=self.storage[4], d=self.storage[5], c=self.storage[6]
        )

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [
                data['x'], data['y'], data['g'], data['u'], data['r'],
                data['d'], data['c']
            ]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class HERBuffer(object):
    def __init__(
            self, maxep_length, rew_func, absolute_goal, maxsize=1e6,
            strategy='future', ratio=0.8
        ):
        self.maxep_length = maxep_length
        self.rew_func = rew_func
        self.absolute_goal = absolute_goal
        self.storage = [[] for _ in range(7)]
        self.maxsize = maxsize
        self.next_idx = 0
        self.strategy = strategy
        self.her_ratio = ratio
        self.steps = []
        self.step = 0

    def clear(self):
        self.storage = [[] for _ in range(7)]
        self.next_idx = 0
        self.steps = []
        self.step = 0

    # Expects tuples of (x, x', g, u, r, d, cost)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            for array, datapoint in zip(self.storage, data):
                array.append(datapoint)
            self.steps.append(self.step)
        else:
            for array, datapoint in zip(self.storage, data):
                array.__setitem__(self.next_idx, datapoint) 
            self.steps.__setitem__(self.next_idx, self.step)

        self.next_idx = (self.next_idx + 1) % self.maxsize
        self.step = int((self.step + 1) * (1 - data[5]))

    def sample(self, batch_size):
        if len(self.storage[0]) <= batch_size:
            ind = np.arange(len(self.storage[0]))
        else:
            ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        nb_virtual = int(self.her_ratio * batch_size)
        virtual_ind, real_ind = np.split(ind, [nb_virtual])
        x, y, g, u, r, d, c = [], [], [], [], [], [], []          

        for i in real_ind: 
            data = (array[i] for array in self.storage)
            X, Y, G, U, R, D, C = data
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            c.append(np.array(C, copy=False))

        for i in virtual_ind: 
            data = (array[i] for array in self.storage)
            X, Y, _, U, _, D, C = data

            step = self.steps[i]
            end = self.maxep_length - step
            if self.strategy == 'future':
                begin = 0
            elif self.strategy == 'episode':
                begin = - step
            else:
                raise NotImplementedError
            
            while True:
                shift = np.random.randint(begin,  end)
                new_ind = i + shift
                new_step = step + shift
                if new_ind < len(self.steps):
                    if new_step == self.steps[new_ind]:
                        break
            # G = self.storage[2][new_ind]
            G = np.copy(self.storage[0][new_ind][:2])
            if not self.absolute_goal:
                G = G - X[:2]
            R = self.rew_func(X, G, Y, 1.)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            c.append(np.array(C, copy=False))
        return np.array(x), np.array(y), np.array(g), np.array(u),\
            np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
            np.array(c).reshape(-1, 1)

    def save(self, file):
        np.savez_compressed(
            file, idx=np.array([self.next_idx]), x=self.storage[0],
            y=self.storage[1], g=self.storage[2], u=self.storage[3],
            r=self.storage[4], d=self.storage[5], c=self.storage[6]
        )

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [
                data['x'], data['y'], data['g'], data['u'], data['r'],
                data['d'], data['c']
            ]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape)).clip(min_action, max_action)
        return action
