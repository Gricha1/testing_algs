from __future__ import annotations
from typing import Any, ClassVar
import os
from dataclasses import dataclass

# import comet_ml
import numpy as np
import torch
from gymnasium import spaces
import omnisafe
from omnisafe.envs.core import CMDP, env_register

from safety_ant_maze_pusher_envs.create_env_utils import create_env


@dataclass
class Args:
    env_name: str
    random_start_pose: bool
    seed = 0


@env_register
class SafeAntMaze(CMDP):
    _support_envs: ClassVar[list[str]] = [
        'SafeAntMazeC-Fixed', 'SafeAntMazeW-Fixed',
        'SafeAntMazeC-Rand', 'SafeAntMazeW-Rand',
        'SafePusher-Rand',
    ]  # Supported task names

    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = False  # Whether `TimeLimit` Wrapper is needed

    def __init__(self, env_id: str, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1

        env_name, start = env_id.split("-")
        self.long_horizon_env_name = env_name
        args = Args(env_name=env_name, random_start_pose=start=="Rand")
        if env_name == "SafePusher":
            args.pusher_safe_env_dangerous_circle = True
            args.pusher_safe_env_hazards = False
            args.safe_env_hazards = False
            args.pusher_hard_goal_dist = False
            args.pusher_always_random_obj_start_poses = True
            args.pusher_hard_task = False
            args.pusher_sparse_reward = False
            args.pusher_two_goal_dim = False
            args.pusher_four_goal_dim = False
            args.pusher_three_goal_dim = True
            args.pusher_safe_env_safe_zone = False
            args.pusher_random_obj_start_poses  = True
        env, state_dim, goal_dim, action_dim, renderer = create_env(args)
        self._env = env
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim+goal_dim,))
        self._action_space = spaces.Box(low=env.action_space.low, high=env.action_space.high, shape=(action_dim,))
        self._device = kwargs.get('device', 'cpu')

        # for eval
        self.do_eval = False
        self.eval_episode_index = 0
        self.successes = []

    def activate_eval(self, activate):
        if activate:
            self._env.evaluate = True
            self.do_eval = True
            self.eval_episode_index = 0
            self.successes = []
        else:
            self._env.evaluate = False
            self.do_eval = False
            self.eval_episode_index = 0
            self.successes = []
        

    def set_seed(self, seed: int) -> None:
        self._env.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        if self.do_eval and not self.long_horizon_env_name == "SafePusher":
            obs = self._env.reset(eval_idx=self.eval_episode_index)
            self.eval_episode_index += 1
        else:
            obs = self._env.reset()
        obs = np.concatenate([obs['observation'], obs['desired_goal']]).astype(np.float32)
        obs = torch.as_tensor(obs, device=self._device)
        self._count = 0
        return obs, {}

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        if self.long_horizon_env_name == "SafePusher":
            return 100
        elif self.long_horizon_env_name == "SafeAntMazeC" or self.long_horizon_env_name == "SafeAntMazew":
            return 500
        else:
            assert 1 == 0
            return 

    def render(self) -> Any:
        pass

    def close(self) -> None:
        pass

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs, reward, done, info = self._env.step(action.cpu().numpy()) 
        obs = np.concatenate([obs['observation'], obs['desired_goal']]).astype(np.float32)
        obs = torch.as_tensor(obs, device=self._device)
        reward = torch.as_tensor(reward, device=self._device, dtype=torch.float)
        cost = torch.as_tensor(info["safety_cost"], device=self._device, dtype=torch.float)
        terminated = torch.as_tensor(done if self._count < self.max_episode_steps else False, device=self._device)
        truncated = torch.as_tensor(self._count >= self.max_episode_steps, device=self._device)

        if self.do_eval:
            if self.long_horizon_env_name == "SafePusher":
                terminated = torch.as_tensor(info["is_success"], device=self._device)
            else:
                terminated = torch.as_tensor(self._env.success_fn(reward), device=self._device)
            if torch.logical_or(terminated, truncated):
                self.successes.append(terminated)

        return obs, reward, cost, terminated, truncated, {'final_observation': obs}


def train():
    # experiment = comet_ml.Experiment(
    #     project_name="ites"
    # )
    custom_cfgs = {
        'seed': 224424,
        'train_cfgs': {
            'total_steps': 4_020_000,
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cuda:0'
        },
        'algo_cfgs': {
            'steps_per_epoch': 30000,
        },
        'logger_cfgs': {
            'use_wandb': True,
            'use_tensorboard': True,
            'log_dir': 'logs', # /logdir/omnisafe
            'save_model_freq': 1 
        },
        "lagrange_cfgs": {
            'cost_limit': 25
        }
    }

    #agent = omnisafe.Agent('PPOLag', 'SafeAntMazeC-Rand', custom_cfgs=custom_cfgs) 
    # SafeAntMazeC-Rand
    # SafeAntMazeW-Rand
    # SafePusher-Rand
    #agent = omnisafe.Agent('FOCOPS', 'SafePusher-Rand', custom_cfgs=custom_cfgs)
    #agent = omnisafe.Agent('CUP', 'SafePusher-Rand', custom_cfgs=custom_cfgs)
    agent = omnisafe.Agent('PPOLag', 'SafePusher-Rand', custom_cfgs=custom_cfgs) 
    # experiment.log_parameters(agent.cfgs)
    agent.learn()


def eval(log_dir, al_name):
    import tqdm
    LOG_DIR = log_dir + al_name
    evaluator = omnisafe.Evaluator()
    all_items = []
    for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            all_items.append(item.name)

    all_items = sorted(all_items, key=lambda name: int(name.split('-')[-1].split('.')[0]))
    all_items = all_items

    data = {
        'mean_cost': [],
        'std_cost': [],
        'mean_return': [],
        'std_return': [],
        'mean_success': [],
        'std_success': [],
        'step': [],
    }
    for i, item in enumerate(tqdm.tqdm(all_items)):
        print("weights:", item)
        evaluator.load_saved(save_dir=LOG_DIR, model_name=item)
        evaluator._env._env.activate_eval(True)
        # evaluator._env._env._time_limit = 500
        rews, cs = evaluator.evaluate(num_episodes=40)
        ss = evaluator._env._env.successes
        data['mean_success'].append(np.mean(ss))
        data['std_success'].append(np.std(ss))
        evaluator._env._env.activate_eval(False)
        data['mean_return'].append(np.mean(rews))
        data['std_return'].append(np.std(rews))
        data['mean_cost'].append(np.mean(cs))
        data['std_cost'].append(np.std(cs))
        data['step'].append(i * 30_000)

    data = {k: np.array(v) for k, v in data.items()}
    last_weights_data = {key: val[-1] for key, val in data.items()}
    print("eval results:", last_weights_data)
    """
    np.savez(al_name + '.npz', **data)

    import matplotlib.pyplot as plt
    plt.plot(data['step'], data['mean_success'])
    plt.savefig(al_name + '_sucs.png')
    plt.close()

    plt.plot(data['step'], data['mean_return'])
    plt.savefig(al_name + '_rewards.png')
    plt.close()

    plt.plot(data['step'], data['mean_cost'])
    plt.savefig(al_name + '_costs.png')
    plt.close()
    """

def plot():
    import matplotlib.pyplot as plt
    names = {
        'TD3': ['RandMazeW', 'TD3RandMazeW_1', 'TD3RandMazeW_2'],
        'TD3PID': ['TD3PID40RandMazeW_1', 'TD3PID40RandMazeW_2', 'TD3PID40RandMazeW_3']
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for name, files in names.items():
        costs = []
        sucs = []
        for file in files:
            with np.load(file + '.npz') as data:
                costs.append(data['mean_cost'][:134])
                sucs.append(data['mean_success'][:134])
                steps = data['step'][:134]
            
        ax[0].fill_between(
            steps,
            np.mean(sucs, 0)-np.std(sucs, 0),
            np.mean(sucs, 0)+np.std(sucs, 0), alpha=0.5
        )
        ax[0].plot(steps, np.mean(sucs, 0), label=name)
        ax[1].fill_between(
            steps,
            np.mean(costs, 0)-np.std(costs, 0),
            np.mean(costs, 0)+np.std(costs, 0), alpha=0.5
        )
        ax[1].plot(steps, np.mean(costs, 0), label=name)
    ax[0].set_title('Success')
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[1].set_title('Cost')
    ax[1].legend()
    plt.savefig('rand_maze_w.png')
    plt.close()

if __name__ == "__main__":
    #train()
    #log_dir = "logs/"
    log_dir = "/logdir/omnisafe"
    al_name = "PPOLag-{SafePusher-Rand}/seed-224424-2025-05-17-12-30-33"
    #al_name = "PPOLag-{SafePusher-Rand}/seed-224424-2025-05-17-12-30-33"
    eval(log_dir, al_name)






