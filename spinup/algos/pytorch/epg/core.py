import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                              activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                          activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs),
                             -1)  # Critical to ensure v has right shape.


class MLPCategoricalCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, v_min, v_max, num_atoms,
                 activation):
        super().__init__()
        self.head = mlp([obs_dim] + list(hidden_sizes) + [num_atoms],
                        activation)
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms).cuda()

    def forward(self, obs):
        return self.head(obs)

    def get_probs(self, obs):
        return F.softmax(self.forward(obs), dim=-1)

    def get_exponential_value(self, obs, omega):
        probs = self.get_probs(obs)
        return (torch.exp(omega * self.z_atoms) * probs).sum(-1)


class MLPCategoricalTrauncatedCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, v_min, v_max, num_atoms,
                 activation):
        super().__init__()
        self.head = mlp([obs_dim * 2] + list(hidden_sizes) + [num_atoms],
                        activation)
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms).cuda()

    def forward(self, obs1, obs2):
        return self.head(torch.cat([obs1, obs2], dim=-1))

    def get_probs(self, obs1, obs2):
        return F.softmax(self.forward(obs1, obs2), dim=-1)

    def get_exponential_value(self, obs1, obs2, omega):
        probs = self.get_probs(obs1, obs2)
        return (torch.exp(omega * self.z_atoms) * probs).sum(-1)


class MLPActorCritic(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(64, 64),
                 v_max=10.,
                 v_min=-10,
                 num_atoms=51,
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0],
                                       hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes,
                                          activation)

        # build value function
        self.v1 = MLPCategoricalCritic(obs_dim, hidden_sizes, v_max, v_min,
                                       num_atoms, activation)
        self.v2 = MLPCategoricalTrauncatedCritic(obs_dim, hidden_sizes, v_max,
                                                 v_min, num_atoms, activation)
        self.omega = nn.Parameter(torch.tensor(0.1).cuda(), requires_grad=True)

    def step(self, obs, obs0):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v1 = self.v1.get_exponential_value(obs, self.omega)
            v2 = self.v2.get_exponential_value(obs, obs0, self.omega)
        return a.cpu().numpy(), v1.cpu().numpy(), v2.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
        return a