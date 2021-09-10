import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.epg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a epg agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val1_buf = np.zeros(size, dtype=np.float32)
        self.val2_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val1, val2, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val1_buf[self.ptr] = val1
        self.val2_buf[self.ptr] = val2
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, omega, last_val=0):
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
        vals1 = np.append(self.val1_buf[path_slice], last_val)
        vals2 = np.append(self.val2_buf[path_slice], last_val)

        _gamma_list = np.array(
            [self.gamma**i for i in range(self.ptr - self.path_start_idx + 1)])

        # the next two lines implement EPG advantage calculation
        # TODO add omega
        self.adv_buf[path_slice] = (vals1 * np.exp(omega * rews * _gamma_list) *
                                    vals2)[:-1]

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # calculate forward values

        self.ret2_buf = np.zeros(len(rews) - 1)
        for i in range(self.ptr - self.path_start_idx):
            self.ret2_buf[
                i] = self.ret_buf[0] - _gamma_list[i] * self.ret_buf[i]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    ret2=self.ret2_buf,
                    rew=self.rew_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32).cuda()
            for k, v in data.items()
        }


def epg(env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        omega_lr=1e-4,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=10):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

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
            you provided to epg.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space,
                      **ac_kwargs).cuda()

    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(
        core.count_vars(module) for module in [ac.pi, ac.v1, ac.v2])
    logger.log('\nNumber of parameters: \t pi: %d, \t v1: %d, \t v2: %d\n' %
               var_counts)

    # Set up experience buffer
    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    local_steps_per_epoch = steps_per_epoch
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing EPG policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data[
            'logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v1(data, loss_type='td'):
        obs, rew, ret = data['obs'], data['rew'], data['ret']
        if loss_type == 'regression':
            return ((ac.v1.get_exponential_value(obs, ac.omega) -
                     torch.exp(ac.omega * ret))**2).mean()
        elif loss_type == 'td':
            backup = torch.exp(ac.omega * rew) * ac.v1.get_exponential_value(
                obs, ac.omega * gamma)
            print('backup', backup)
            print('ac.v1.get_exponential_value(obs, ac.omega)',
                  ac.v1.get_exponential_value(obs, ac.omega))
            return ((ac.v1.get_exponential_value(obs, ac.omega)[:-1] -
                     backup[1:])**2).mean()

    # Set up function for computing truncated value loss
    def compute_loss_v2(data, loss_type='td'):
        obs, rew, ret = data['obs'], data['rew'], data['ret2']
        obs0 = torch.empty_like(obs)
        obs0[:] = obs[0]
        if loss_type == 'regression':
            return ((ac.v2.get_exponential_value(obs, obs0, ac.omega) -
                     torch.exp(ac.omega * ret))**2).mean()
        elif loss_type == 'td':
            _gamma_list = torch.tensor([gamma**i for i in range(len(rew))
                                       ]).cuda()
            backup = torch.exp(
                ac.omega * _gamma_list * rew) * ac.v2.get_exponential_value(
                    obs, obs0, ac.omega * gamma)
            return ((ac.v2.get_exponential_value(obs, obs0, ac.omega)[1:] -
                     backup[:-1])**2).mean()

    # Set up function for computing omega loss
    def compute_loss_omega(data):
        obs, rew = data['obs'], data['rew']
        obs0 = torch.empty_like(obs)
        obs0[:] = obs[0]
        _gamma_list = torch.tensor([gamma**i for i in range(len(rew))]).cuda()
        return (ac.v2.get_exponential_value(obs, obs0, ac.omega) *
                torch.exp(rew * ac.omega * _gamma_list) *
                ac.v1.get_exponential_value(obs, ac.omega)).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf1_optimizer = Adam(ac.v1.parameters(), lr=vf_lr)
    vf2_optimizer = Adam(ac.v2.parameters(), lr=vf_lr)
    omega_optimizer = Adam([ac.omega], lr=omega_lr, betas=(0.9, 0.999))

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v1_l_old = compute_loss_v1(data).item()
        v2_l_old = compute_loss_v2(data).item()
        omega_l_old = compute_loss_omega(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        # mpi_avg_grads(ac.pi)  # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf1_optimizer.zero_grad()
            vf2_optimizer.zero_grad()
            loss_v1 = compute_loss_v1(data)
            loss_v2 = compute_loss_v2(data)
            loss_v1.backward()
            loss_v2.backward()
            vf1_optimizer.step()
            vf2_optimizer.step()

            # Optimize omega
            omega_optimizer.zero_grad()
            loss_omega = compute_loss_omega(data)
            loss_omega.backward()
            # mpi_avg_grads(ac.omega)
            omega_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old,
                     LossV1=v1_l_old,
                     LossV2=v2_l_old,
                     LossOmega=loss_omega.item(),
                     KL=kl,
                     Entropy=ent,
                     Omega=ac.omega.item(),
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV1=(loss_v1.item() - v1_l_old),
                     DeltaLossV2=(loss_v2.item() - v2_l_old),
                     DeltaLossOmega=(loss_omega.item() - omega_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    first_o = o

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v1, v2, logp = ac.step(
                torch.as_tensor(o, dtype=torch.float32).cuda(),
                torch.as_tensor(first_o, dtype=torch.float32).cuda())

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v1, v2, logp)
            logger.store(V1Vals=v1)
            logger.store(V2Vals=v2)
            adv = v1 * v2 * np.exp(
                ac.omega.detach().cpu().numpy() * r * gamma**ep_len)
            logger.store(AdvVals=adv)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                          ep_len,
                          flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v1, _, _ = ac.step(
                        torch.as_tensor(o, dtype=torch.float32).cuda(),
                        torch.as_tensor(first_o, dtype=torch.float32).cuda())
                else:
                    v1 = 0
                buf.finish_path(ac.omega.detach().cpu().numpy(), v1)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0
                first_o = o

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform epg update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('V1Vals', with_min_and_max=True)
        logger.log_tabular('V2Vals', with_min_and_max=True)
        logger.log_tabular('AdvVals', with_min_and_max=True)
        logger.log_tabular('Omega', with_min_and_max=True)
        # logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV1', average_only=True)
        logger.log_tabular('LossV2', average_only=True)
        logger.log_tabular('LossOmega', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV1', average_only=True)
        # logger.log_tabular('DeltaLossV2', average_only=True)
        # logger.log_tabular('DeltaLossOmega', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='epg')
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    epg(lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs)