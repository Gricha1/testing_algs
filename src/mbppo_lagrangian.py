#This code is modified for research purpose from Open AI Spinning Up implementation of PPO https://github.com/openai/spinningup (MIT LICENSE which allows for private use, attached  in ./src folder)
import sys
import collections

import numpy as np
import torch
import wandb
from torch.optim import Adam
import safety_gym
import gym
import time
import core
from torch.nn.functional import softplus
from mpi4py import MPI

from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from env_utils import SafetyGymEnv
from envs.create_env_utils import create_env
from aux import dist_xy, get_reward_cost, get_goal_flag, ego_xy, obs_lidar_pseudo, make_observation, generate_lidar
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from replay_memory import ReplayMemory
from tqdm import tqdm
from stable_baselines3.common.logger import Video

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        #buf.store(   o, a, r, c, v,vc, logp)
    def store(self, obs, act, rew, crew, val,cval, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)


        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)
        #print(adv_mean, adv_std, cadv_mean, cadv_std)
        eps = 1e-5
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean) # / (cadv_std+eps)
        cpudevice = torch.device('cpu')
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
                    adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v,device=cpudevice, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, num_steps, cost_limit, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.1, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=None, save_freq=1,exp_name='default', beta=1,
        args=None, state_dim=0, goal_dim=0, action_dim=0, renderer=None, img_rollout_H=80):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    logger = EpochLogger(**logger_kwargs)
    if not logger.use_wandb:
        logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    #obs_dim = env.observation_space.shape
    #-----------------This dimension size is specific for our use-case in our modified Safety Gym envs-----------------------
    if args.env_name == "SafeAntMaze":
        obs_dim = (state_dim + goal_dim, )
        act_dim = action_dim
    else:
        obs_dim = (26,)
        act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(obs_dim, env.action_space, **ac_kwargs)
    cpudevice = torch.device('cpu')
    if args.load:
        ac.load(exp_name)
        print("load from:", exp_name)
    ac.to(cpudevice)
    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    gradnorm_list = []
    cost_limit = cost_limit
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):        
        obs, act, adv, cadv,  logp_old = data['obs'], data['act'], data['adv'], data['cadv'] ,data['logp']

        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']

        # Policy loss
        pi, logp = ac.pi(obs, act.to(cpudevice))

        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()

        clip_cadv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * cadv
        loss_cpi = (torch.max(ratio * cadv, clip_cadv)).mean()
        loss_cpi = loss_cpi.mean()

        p = softplus(penalty_param)
        penalty_item = p.item()

        pi_objective = loss_rpi - penalty_item*loss_cpi
        ent_coef = 0.0 #1*pi.entropy().mean()
        pi_objective = (pi_objective + ent_coef) /(1+penalty_item)
        loss_pi = -pi_objective
        #to tackle underestimation of cost due to truncated horizon we use 'beta_safety' hyperparameter
        beta_safety = beta
        cost_deviation = (cur_cost - cost_limit*beta_safety)


        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, device=cpudevice, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        return ((ac.v(obs) - ret)**2).mean(),((ac.vc(obs) - cret)**2).mean()


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = torch.tensor(0.5, device=cpudevice,requires_grad=True).float()
    penalty = softplus(penalty_param)


    penalty_lr = 5e-2
    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(),lr=vf_lr)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

#----------------------------------------------------------------------------------------------------------
    def get_param_values(model):
        trainable_params = list(model.parameters()) #+ [self.log_std]
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in trainable_params])
        return params.copy()

    def set_param_values(new_params,model,set_new=True):
        trainable_params = list(model.parameters())

        param_shapes = [p.data.numpy().shape for p in trainable_params]
        #print("param shapes",len(param_shapes))
        param_sizes = [p.data.numpy().size for p in trainable_params]
        if set_new:
            current_idx = 0
            for idx, param in enumerate(trainable_params):
                vals = new_params[current_idx:current_idx + param_sizes[idx]]
                vals = vals.reshape(param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += param_sizes[idx]

#---------------------------------------------------------------------------------------------------------------
    def update():
        cur_cost = logger.get_stats('DynaEpCost')[0]
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)

        loss_penalty = -penalty_param*cost_dev


        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()

        data['cur_penalty'] = penalty_param
        print("penal=",softplus(penalty_param))


        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item()


        # Train policy with multiple steps of gradient descent
        train_epoch_eigvals=[]

        train_pi_iters=80
        this_epoch_gradnorms = []
        for i in range(train_pi_iters):

            loss_pi, _,pi_info = compute_loss_pi(data)


            kl = mpi_avg(pi_info['kl'])


            if kl > 1.2 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break


            pi_optimizer.zero_grad()
            loss_pi.backward()
            mpi_avg_grads(ac.pi)
            pi_optimizer.step()


        # Value function learning
        train_v_iters=80
        for i in range(train_v_iters):
            loss_v, loss_vc = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)   # average grads across MPI processes
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        #wandb.log({'losspi':pi_l_old,'kl':kl})
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     LossCV=cv_l_old,
                     Lambda=penalty_param,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

#----------------------TRAIN DYNAMICS----------------------------------------------------------
    def train_predict_model(env_pool, predict_env):
        print("***********")
        print("train world model")
        # Get all samples from environment
        state, action, reward, cost, next_state, done = env_pool.sample(len(env_pool))

        #print("memory occupied by  sampled info = ",sys.getsizeof(state))
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)

        labels = delta_state

        epoch, loss = predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
        logger.store(LossModel=loss)
        del state, action, reward, next_state, done
        print("model loss:", loss)
        print("***********")
#-----------------------------------------------------------------------------------------------

    if args.validate:
        validate_world_model = True
        start_time = time.time()

        print("validating .....")
        val_episode = 0
        screens = []
        ep_ret, ep_len =  0, 0
        ep_cost = 0
        violations = 0
        epoch = 0
        megaiter = 0
        max_training_steps = 0

        if args.env_name == "SafeAntMaze":
            obs = env.reset(eval_idx=val_episode)
            goal_pos = obs["desired_goal"]
            o = obs["observation"]
            o = np.concatenate((o, goal_pos))
        else:
            o, static = env.reset()
            goal_pos = static['goal']
            hazards_pos = static['hazards']
        screens = []
        ep_ret = 0
        pep_ret = 0
        ep_cost = 0
        pep_cost = 0
        ep_len = 0
        t = 0
        while True:
            t += 1
            if args.env_name == "SafeAntMaze":
                obs_vec = o
            else:
                #generate hazard lidar
                obs_vec = generate_lidar(o, hazards_pos)
                obs_vec = np.array(obs_vec)
            
            ot = torch.as_tensor(obs_vec, device=cpudevice, dtype=torch.float32)
            a, v, vc, logp = ac.step(ot)
            if validate_world_model:
                with torch.no_grad():
                    pass
            del ot

            next_o, r, d, info = env.step(a)

            if args.env_name == "SafeAntMaze": 
                next_o = next_o["observation"]  
                next_o = np.concatenate((next_o, goal_pos))
                c = info['safety_cost']
            else:
                c = info['cost']
            violations += c
            ep_ret += r
            ep_cost += c
            ep_len += 1

            debug_info = {"acc_reward": ep_ret,
                          "acc_cost": ep_cost,
                          "t": ep_len}
            current_step_info = {}
            current_step_info["robot_pos"] = np.array(o[:2])
            if args.env_name != "AntGather" and args.env_name != "AntMazeSparse":
                current_step_info["goal_pos"] = np.array(goal_pos[:2])
            else:
                current_step_info["goal_pos"] = None
            current_step_info["robot_radius"] = 1.5
            if args.vizualize_validation:
                if args.env_name == "SafeAntMaze":
                    if t == 1:
                        renderer.setup_renderer()
                    screen = renderer.custom_render(current_step_info, 
                                                    debug_info=debug_info, 
                                                    plot_goal=True,
                                                    env_name=args.env_name,
                                                    safe_model=None)  
                else:
                    screen = env.custom_render(positions_render=True, debug_info=debug_info)
                screens.append(screen)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout

            if terminal:
                logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost, 
                             PEPRet=pep_ret, PEPCost=pep_cost)
                break
                o,static= env.reset()
                goal_pos = static['goal']
                hazards_pos = static['hazards']
                #ld = dist_xy(o[40:], goal_pos)
                ep_ret, ep_len =  0, 0
                pep_ret,pep_cost  = 0,0
                ep_cost = 0
                val_episode += 1

        if args.env_name == "SafeAntMaze" and args.vizualize_validation:
            logger.save_video(screens)
            del screens
            renderer.delete_data()
        elif args.vizualize_validation:
            logger.save_video(screens)
            del screens
        logging(logger, epoch, megaiter, 
                start_time, violations, 
                max_training_steps, validate=True)
        print("end validation")
        return

    # Prepare for interaction with environment
    start_time = time.time()
    ep_ret, ep_len =  0, 0
    ep_cost = 0
    violations = 0
    # Main loop: collect experience in env to train dynamics models
    # 450k environment steps
    # Collecting data from real environment
    max_training_steps = int(10000 / num_procs())
    outer_loop_epochs = int(num_steps / max_training_steps)
    validation_episode = 1
    validate_each_epoch = 5
    validate_world_model = True

    # Start Training
    for epoch in range(outer_loop_epochs):
        if args.env_name == "SafeAntMaze":
            obs = env.reset()
            goal_pos = obs["desired_goal"]
            o = obs["observation"]  
            o_without_goal = o
            o = np.concatenate((o, goal_pos))
        else:
            o, static = env.reset()
            goal_pos = static['goal']
            hazards_pos = static['hazards']
            ld = dist_xy(o[40:], goal_pos)
        train_episode = 0
        screens = []
        ep_ret = 0
        pep_ret = 0
        ep_cost = 0
        pep_cost = 0
        ep_len = 0
        mix_real = int(1500/num_procs())
        max_ep_len2 = img_rollout_H
        print("interacting with real environment")
        ## Collect Trajectories From Env
        for t in tqdm(range(max_training_steps)):

            ### Generate hazard lidar
            if args.env_name == "SafeAntMaze":
                obs_vec = o
            else:
                obs_vec = generate_lidar(o, hazards_pos)
                obs_vec = np.array(obs_vec)

            ot = torch.as_tensor(obs_vec, device=cpudevice, dtype=torch.float32)
            a, v, vc, logp = ac.step(ot)
            if validate_world_model:
                with torch.no_grad():
                    pass
            del ot

            next_o, r, d, info = env.step(a)

            if args.env_name == "SafeAntMaze":
                next_o_without_goal = next_o["observation"]                  
                next_o = np.concatenate((next_o_without_goal, goal_pos))  
                c = info['safety_cost']
            else:
                c = info['cost']

            if args.env_name == "SafeAntMaze":
                # test probably here is a problem with goal met
                if not d:
                    env_pool.push(o_without_goal, a, r, info['safety_cost'], next_o_without_goal, d)
            else:
                if not d and not info['goal_met']:
                    env_pool.push(o, a, r, info['cost'], next_o, d)
            
            violations += c
            ep_ret += r
            ep_cost += c
            ep_len += 1
            if train_episode == validation_episode and (epoch % validate_each_epoch) == 0:
                debug_info = {"acc_reward": ep_ret,
                              "acc_cost": ep_cost,
                              "t": ep_len}
                if args.vizualize_validation:
                    screen = env.custom_render(positions_render=True, debug_info=debug_info)
                    screens.append(screen)

            ## Mixing some real environment samples
            if t <= mix_real - 1:
                buf.store(obs_vec, a, r, c, v, vc, logp)

            o = next_o
            o_without_goal = next_o_without_goal 

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == max_training_steps - 1

            timeout_mixer = ep_len == max_ep_len2
            terminal_mixer = d or timeout_mixer
            epoch_ended_mixer = t == mix_real - 1

            if t <= mix_real - 1:
                if timeout_mixer or epoch_ended_mixer:
                    ott = torch.as_tensor(obs_vec, device=cpudevice, dtype=torch.float32)
                    _, v,vc, _ = ac.step(ott)
                    del ott
                    buf.finish_path(v,vc)
                elif d:
                    v = 0
                    vc = 0
                    buf.finish_path(v,vc)

            if terminal:
                logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost, PEPRet=pep_ret, PEPCost=pep_cost)
            if terminal or epoch_ended:
                if args.env_name == "SafeAntMaze":
                    obs = env.reset()
                    goal_pos = obs["desired_goal"]
                    o = obs["observation"]
                    o_without_goal = o   
                    o = np.concatenate((o, goal_pos))        
                else:
                    o, static = env.reset()
                    goal_pos = static['goal']
                    hazards_pos = static['hazards']
                    ld = dist_xy(o[40:], goal_pos)
                ep_ret, ep_len =  0, 0
                pep_ret,pep_cost = 0, 0
                ep_cost = 0
                train_episode += 1
        if args.vizualize_validation and (epoch % validate_each_epoch) == 0:
            logger.save_video(screens)

        ## Train Dynamic Model
        train_predict_model(env_pool, predict_env)
        torch.save(env_model,exp_name + "/" + "env_model.pkl")

        ## Train Actor Critic
        env_name = "safepg2"
        model_type = 'pytorch'
        megaiter = 0
        perf_flag = True
        while perf_flag:
            if megaiter == 0:
                last_dynaret = 0
                last_valid_rets = [0]*args.num_elites

            env_model2 = torch.load(exp_name + "/" + "env_model.pkl")
            predict_env2 = PredictEnv(env_model2, env_name, model_type)

            if args.env_name == "SafeAntMaze":
                obs = env.reset()
                goal_pos = obs["desired_goal"]
                o_without_goal = obs["observation"] 
                o = np.concatenate((o_without_goal, goal_pos))               
            else:
                o, static = env.reset()
                goal_pos = static['goal']
                hazards_pos = static['hazards']
                ld = dist_xy(o[40:], goal_pos)
            dep_ret = 0
            dep_cost = 0
            dep_len = 0

            print("get imagination steps:", local_steps_per_epoch - mix_real)
            ## Get Imagination Steps
            ## Mixing real environment data in first outer loop
            if megaiter == 0:
                mix_real = 1500
            else:
                mix_real = 0
            for t in tqdm(range(local_steps_per_epoch - mix_real)):
                ## generate hazard lidar
                if args.env_name == "SafeAntMaze":
                    obs_vec = o
                    robot_pos = o[:2]
                else:
                    obs_vec = generate_lidar(o, hazards_pos)
                    robot_pos = o[40:]
                    obs_vec = np.array(obs_vec)

                otensor = torch.as_tensor(obs_vec, device=cpudevice, dtype=torch.float32)
                a, v, vc, logp = ac.step(otensor)
                del otensor

                next_o_without_goal = predict_env2.step(o_without_goal, a)
                next_o = np.concatenate((next_o_without_goal, goal_pos))
                
                if args.env_name == "SafeAntMaze":
                    r, c, goal_flag = env.get_reward_cost(robot_pos, 
                                                          goal_pos)
                else:
                    r, c, ld, goal_flag = get_reward_cost(ld, robot_pos, hazards_pos, goal_pos)

                dep_ret += r
                dep_cost += c
                dep_len += 1

                # save and log
                buf.store(obs_vec, a, r, c , v, vc, logp)

                o = next_o

                # Model horizon (H)  = max_ep_len2
                timeout = dep_len == max_ep_len2
                terminal = timeout
                epoch_ended = t == local_steps_per_epoch - 1
                if terminal or epoch_ended or goal_flag:
                    # if epoch_ended and not(terminal):
                    #     print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended or goal_flag:
                        otensort = torch.as_tensor(obs_vec, device=cpudevice, dtype=torch.float32)
                        _, v, vc, _ = ac.step(otensort)
                        del otensort
                    else:
                        v = 0
                        vc = 0
                    buf.finish_path(v, vc)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(DynaEpRet=dep_ret, DynaEpCost=dep_cost)
                    if args.env_name == "SafeAntMaze":
                        obs = env.reset()
                        goal_pos = obs["desired_goal"]
                        o_without_goal = obs["observation"]
                        o = np.concatenate((o_without_goal, goal_pos))                   
                    else:
                        o, static  = env.reset()
                        goal_pos = static['goal']
                        hazards_pos = static['hazards']
                        ld = dist_xy(o[40:], goal_pos)
                    dep_ret, dep_len, dep_cost = 0, 0, 0

            if (epoch % save_freq == 0) or (epoch == outer_loop_epochs - 1):
                logger.save_state({'env': env}, None)
            new_dynaret = logger.get_stats('DynaEpRet')[0]

            ## Validation
            if megaiter > 0:
                old_params_pi = get_param_values(ac.pi)
                old_params_v = get_param_values(ac.v)
                old_params_vc = get_param_values(ac.vc)
                update()
                # 6 ELITE MODELS OUT OF 8
                valid_rets = [0]*args.num_elites
                winner = 0
                print("validating............")
                for va in tqdm(range(len(valid_rets))):
                    if args.env_name == "SafeAntMaze":
                        obs = env.reset()
                        goal_posv = obs["desired_goal"]
                        ov_without_goal = obs["observation"]
                        ov = np.concatenate((ov_without_goal, goal_pos))                
                    else:
                        ov, staticv = env.reset()
                        goal_posv = staticv['goal']
                        hazards_posv = staticv['hazards']
                        # test probably bug with o instead of ov
                        ldv = dist_xy(o[40:], goal_posv)
                    for step_iter in range(75):
                        if args.env_name == "SafeAntMaze":
                            obs_vecv = ov
                            robot_posv = ov[:2]
                        else:
                            obs_vecv = generate_lidar(ov, hazards_posv)
                            robot_posv = ov[40:]
                            obs_vecv = np.array(obs_vecv)
                        ovt = torch.as_tensor(obs_vecv, device=cpudevice, dtype=torch.float32)
                        av, _, _, _ = ac.step(ovt)
                        del ovt
                        next_ov_without_goal = predict_env2.step_elite(ov_without_goal, av, va)
                        next_ov = np.concatenate((next_ov_without_goal, goal_posv))

                        if args.env_name == "SafeAntMaze":
                            rv, cv, goal_flagv = env.get_reward_cost(robot_posv, 
                                                                     goal_posv)
                        else:
                            rv, cv, ldv, goal_flagv = get_reward_cost(ldv, robot_posv, hazards_posv, goal_posv)

                        valid_rets[va] += rv
                        ov = next_ov
                        if goal_flagv:
                            if args.env_name == "SafeAntMaze":
                                obs = env.reset()
                                goal_posv = obs["desired_goal"]
                                ov_without_goal = obs["observation"] 
                                ov = np.concatenate((ov_without_goal, goal_pos))        
                            else:
                                ov, staticv = env.reset()
                                goal_posv = staticv['goal']
                                hazards_posv = staticv['hazards']
                                ldv = dist_xy(ov[40:], goal_posv)
                    if valid_rets[va] > last_valid_rets[va]:
                        winner += 1
                print(valid_rets, last_valid_rets)
                performance_ratio = winner / args.num_elites
                print("Performance ratio=", performance_ratio)
                thresh = 4/6  #BETTER THAN 50%
                if performance_ratio < thresh:
                    perf_flag = False
                    print("backtracking.........")
                    #------------backtrack-----------------
                    set_param_values(old_params_pi, ac.pi)
                    set_param_values(old_params_v, ac.v)
                    set_param_values(old_params_vc, ac.vc)
                    print("done!")
                    megaiter += 1
                    del predict_env2
                    del env_model2
                    break
                else:
                    del predict_env2
                    del env_model2
                    megaiter += 1
                    last_valid_rets = valid_rets
            else:
                # Perform PPO-agrangian update!
                megaiter+=1
                update()

        ## Save Actor Critic Weights and Logging
        ac.save(exp_name)
        logging(logger, epoch, megaiter, 
                start_time, violations, 
                max_training_steps)

def logging(logger, epoch, megaiter, 
            start_time, violations, 
            max_training_steps, validate=False):
    logger.store(Megaiter=megaiter)
    # Log info about epoch
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('PEPRet', with_min_and_max=True)
    logger.log_tabular('PEPCost', with_min_and_max=True)
    # logger.log_tabular('VVals', with_min_and_max=True)
    #-------------no. of environment interaction per epoch = max_training_steps = 1e4 --------------------
    if not validate:
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*max_training_steps)
        logger.log_tabular('DynaEpRet', with_min_and_max=True)
        logger.log_tabular('DynaEpCost',with_min_and_max=True)
        logger.log_tabular('Megaiter', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossModel', average_only=True)
        logger.log_tabular('LossCV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
    #logger.log_tabular('StopIter', average_only=True)
    logger.log_tabular('Time', time.time()-start_time)
    logger.log_tabular('Violations', violations)

    logger.dump_tabular(epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--vizualize_validation', action='store_true', default=False)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--loaded_exp_num", default=0, type=str)

    # environment
    parser.add_argument('--env_name', type=str, default='Safexp-PointGoal2-v0')
    parser.add_argument("--random_start_pose", action="store_true", default=False)

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
     #---------Don't use multiprocessing as data collection and environment model training is not modified for that-----------#
    parser.add_argument('--cpu', type=int, default=1)
    #--------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--cost_limit', type=int, default=18) # 18
    parser.add_argument('--beta', type=float, default=1)

    # world model
    parser.add_argument("--world_model", default=True)
    parser.add_argument('--pred_hidden_size', default=200, type=int)
    parser.add_argument('--num_networks', default=8, type=int)
    parser.add_argument('--num_elites', default=6, type=int)
    parser.add_argument('--img_rollout_H', default=80, type=int)
    
    # actor critic
    parser.add_argument('--pi_lr', default=3e-4, type=float) # 3e-4
    parser.add_argument('--vf_lr', default=1e-3, type=float) # 1e-3

    # logger
    parser.add_argument("--log_dir", default="./data", type=str)
    parser.add_argument("--not_use_wandb", action='store_true', default=False)
    parser.add_argument("--tensorboard_descript", default="", type=str)

    args = parser.parse_args()
    print("config:", args)
    mpilist = []
    comm = MPI.COMM_WORLD
    mpi_fork(args.cpu)  # run parallel code with mpi
    # mpilist.append(proc_id())
    # newdata = comm.gather(mpilist,root=0)


    #if rank>0:
    #=================safety gym benchmarks defaults==============================
    num_steps = 4.5e5
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    #=============================================================================
    #---modified safety_gym-------------------------------------------------------
    DEFAULT_ENV_CONFIG_POINT = dict(
        action_repeat=1,
        max_episode_length=750,
        use_dist_reward=False,
        stack_obs=False,
    )
    if "Point" in args.env_name or "Car" in args.env_name:
        if "Point" in args.env_name:
            robot = 'Point'
            eplen = 750
            num_steps = 4.5e5
            steps_per_epoch = 30000
            epochs = 60
            DEFAULT_ENV_CONFIG_POINT['max_episode_length'] = eplen
        elif "Car" in args.env_name:
            robot = 'Car'
            eplen = 750
            DEFAULT_ENV_CONFIG_POINT['max_episode_length'] = eplen
            num_steps = 4.5e5
            steps_per_epoch = 30000
            epochs = 60
        env_config=DEFAULT_ENV_CONFIG_POINT
        env = SafetyGymEnv(robot=robot, task="goal", level='2', seed=10, config=env_config)
        state_dim, action_dim = env.observation_size, env.action_size
        renderer = None

    else:
        num_steps = 1.8e6
        renderer_args = {"plot_subgoal": False, 
                         "world_model_comparsion": False,
                         "plot_safety_boundary": True}
        env, state_dim, goal_dim, action_dim, renderer = create_env(args, renderer_args=renderer_args)

    if proc_id()==0:
        num_networks = args.num_networks
        num_elites = args.num_elites
        pred_hidden_size = args.pred_hidden_size
        use_decay = True
        replay_size = 1000000
        env_name = 'safepg2'
        model_type='pytorch'
        reward_size = 0
        cost_size = 0
        #state_dim = 8
        env_model = EnsembleDynamicsModel(num_networks, num_elites, state_dim, action_dim, reward_size, cost_size, pred_hidden_size,
                                          use_decay=use_decay)
        env_pool = ReplayMemory(replay_size)
        predict_env = PredictEnv(env_model, env_name, model_type)
    else:
        assert 1 == 0

    # Set up logger and save configuration
    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs["config"] = args
    logger_kwargs["use_wandb"] = not args.not_use_wandb
    logger_kwargs["tensorboard_log_dir"] = args.log_dir
    logger_kwargs["tensorboard_env_name"] = args.env_name
    logger_kwargs["tensorboard_descript"] = args.tensorboard_descript
    logger_kwargs["vizualize_validation"] = args.vizualize_validation
    # test
    logger_kwargs["exp_num"] = 0

    #-----------------------------------------------------------------------------

    ppo(lambda : env, num_steps, args.cost_limit, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=steps_per_epoch, epochs=epochs, max_ep_len=750,
        logger_kwargs=logger_kwargs,exp_name="data/" + args.exp_name,beta=args.beta,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, args=args, state_dim=state_dim, goal_dim=goal_dim, 
        action_dim=action_dim,
        renderer=renderer, img_rollout_H=args.img_rollout_H)
