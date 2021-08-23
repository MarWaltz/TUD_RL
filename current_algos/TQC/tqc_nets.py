import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np 


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        self.layers = []
        tmp_size = input_size
        
        # Add layers recursively
        for i, next_size in enumerate(hidden_sizes):
            layer = Linear(tmp_size, next_size)
            self.add_module(f"layer{i}",layer)
            self.layers.append(layer)
            tmp_size = next_size
        self.last_layer = Linear(tmp_size, output_size)

    def forward(self, input):
        
        h = input

        # Propagate input forward recusively        
        for layer in self.layers:
            h = relu(layer(h))
        last_layer = self.last_layer(h)
        return last_layer

# The critics receive state and action as input and output n quantiles
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_critics):
        super().__init__()

        self.n_quantiles = n_quantiles
        self.n_critics = n_critics

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,n_quantiles)
        )
        #for i in range(n_critics):
        #    net = MLP(state_dim + action_dim, [512,512,512], n_quantiles)
        #    net = self.net
        #    self.add_module(f"critic {i}",net)
         #   self.nets.append(net)
        self.nets = [self.net] * n_critics
        
    def forward(self, state, action):
        sa = torch.cat((state,action),dim = 1).unsqueeze(1)

        quantiles = torch.cat(tuple(net(sa) for net in self.nets),dim=1)
        return quantiles

# Modeled as in the SAC paper. Gaussian ditribution
class Actor(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()

        self.log_std_min_max = (-20,2)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # self.net = MLP(state_dim, [256,256], 2 * action_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,2 * action_dim)
        )

    def forward(self, state):

        # Construct mean, log_std and std of Gaussian distribution
        mu, log_std = self.net(state).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*self.log_std_min_max)
        std = torch.exp(log_std)

        # Define Gaussian distribution with mean and std given by neural net
        pi_distr = Normal(mu, std)

        if self.training:
            action = pi_distr.rsample()
            logprob = pi_distr.log_prob(action).sum(axis=1) # Appendix C in the paper, but rewritten
            logprob = logprob - (2*(np.log(2) - action - F.softplus(-2 * action))).sum(axis=1) # Eq. 21 (SAC | Haarnoja, 2019)
            logprob = logprob.reshape((-1,1)) # Reshape to ([batch_size, 1])
        else:
            action = pi_distr.mean()
            logprob = None

        action = torch.tanh(action)

        return action, logprob

