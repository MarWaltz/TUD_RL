import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class GaussianActor(nn.Module):
    """Defines stochastic actor based on a Gaussian distribution."""
    def __init__(self, action_dim, state_dim, net_struc_actor, log_std_min=-20, log_std_max=2):
        super(GaussianActor, self).__init__()

        assert net_struc_actor is None, "The net structure cannot be controlled in this way for the SAC-agent."

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mu      = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, s, deterministic, with_logprob):
        """Returns action and it's logprob for given states. s is a torch tensor. Args:

        s:             torch.Size([batch_size, state_dim])
        deterministic: bool (whether to use mean as a sample, only at test time)
        with_logprob:  bool (whether to return logprob of sampled action as well, else second tuple element below will be 'None')

        returns:       (torch.Size([batch_size, action_dim]), torch.Size([batch_size, action_dim]))
        """
        # forward through shared part of net
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))

        # compute mean, log_std and std of Gaussian
        mu      = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std     = torch.exp(log_std)
        
        # construct pre-squashed distribution
        pi_distribution = Normal(mu, std)

        # sample action, deterministic only used for evaluating policy at test time
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        # compute logprob from Gaussian and then correct it for the Tanh squashing
        if with_logprob:

            # this does not exactly match the expression given in Appendix C in the paper, but it is 
            # equivalent and according to SpinningUp OpenAI numerically much more stable
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

            # logp_pi sums in both prior steps over all actions,
            # since these are assumed to be independent Gaussians and can thus be factorized into their margins
            # however, shape is now torch.Size([batch_size]), but we want torch.Size([batch_size, 1])
            logp_pi = logp_pi.reshape((-1, 1))

        else:
            logp_pi = None

        # squash action to [-1, 1]
        pi_action = torch.tanh(pi_action)

        # return squashed action and its logprob
        return pi_action, logp_pi


class Critic(nn.Module):
    """Defines critic network to compute Q-values."""
    def __init__(self, action_dim, state_dim, net_struc_critic):
        super(Critic, self).__init__()

        assert net_struc_critic is None, "The net structure cannot be controlled in this way for the SAC-agent."

        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, s, a):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])
        a:       torch.Size([batch_size, action_dim])
        returns: torch.Size([batch_size, 1])
        """
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Double_Critic(nn.Module):
    """Defines two critic network to compute Q-values in the SAC algorithm."""
    def __init__(self, action_dim, state_dim, net_struc_critic):
        super(Double_Critic, self).__init__()
        
        assert net_struc_critic is None, "The net structure cannot be controlled in this way for the SAC-agent."

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

	    # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, s, a):
        """s and a are torch tensors. Shapes:
        s:        torch.Size([batch_size, state_dim])
        a:        torch.Size([batch_size, action_dim])
        returns: (torch.Size([batch_size, 1]), torch.Size([batch_size, 1]))
        """
        sa = torch.cat([s, a], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
