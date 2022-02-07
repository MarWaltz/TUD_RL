import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

ACTIVATIONS = {"relu"     : F.relu,
               "identity" : nn.Identity(),
               "tanh"     : torch.tanh}


# --------------------------- MLP ---------------------------------
class MLP(nn.Module):
    """Generically defines a multi-layer perceptron."""

    def __init__(self, in_size, out_size, net_struc):
        super(MLP, self).__init__()

        self.struc = net_struc

        assert isinstance(self.struc, list), "net should be a list,  e.g. [[64, 'relu'], [64, 'relu'], 'identity']."
        assert len(self.struc) >= 2, "net should have at least one hidden layer and a final activation."
        assert isinstance(self.struc[-1], str), "Final element of net should only be the activation string."
   
        self.layers = nn.ModuleList()

        # create input-hidden_1
        self.layers.append(nn.Linear(in_size, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

        # create hidden_n-out
        self.layers.append(nn.Linear(self.struc[-2][0], out_size))

    def forward(self, x):
        """x is a torch tensor. Shapes:
        x:       torch.Size([batch_size, in_size]), e.g., in_size = state_dim

        returns: torch.Size([batch_size, out_size]), e.g., out_size = num_actions
        """

        for layer_idx, layer in enumerate(self.layers):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = ACTIVATIONS[act_str]

            # forward
            x = act_f(layer(x))

        return x


class Double_MLP(nn.Module):
    """Maintains two MLPs of identical structure as, e.g., in the TD3 author's original implementation."""

    def __init__(self, in_size, out_size, net_struc):
        super().__init__()

        self.MLP1 = MLP(in_size   = in_size, 
                        out_size  = out_size, 
                        net_struc = net_struc)

        self.MLP2 = MLP(in_size   = in_size, 
                        out_size  = out_size, 
                        net_struc = net_struc)
    
    def forward(self, x):
        return self.MLP1(x), self.MLP2(x)

    def single_forward(self, x):
        return self.MLP1(x)


# --------------------------- MinAtar ---------------------------------
class MinAtar_CoreNet(nn.Module):
    """Defines the CNN part for MinAtar games."""

    def __init__(self, in_channels, height, width):
        super().__init__()

        # CNN hyperparams
        self.out_channels = 16
        self.kernel_size  = 3
        self.stride       = 1
        self.padding      = 0

        # define CNN
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, \
                              stride=self.stride, padding=self.padding)

        # calculate input size for FC layer (which is size of a single feature map multiplied by number of out_channels)
        self.in_size_FC = self._output_size_filter(height) * self._output_size_filter(width) * self.out_channels

    def _output_size_filter(self, size, kernel_size=3, stride=1):
        """Computes for given height or width (here named 'size') of ONE input channel, given
        kernel (or filter) size and stride, the resulting size (again: height or width) of the feature map.
        Note: This formula assumes padding = 0 and dilation = 1."""

        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, s):
        """s: torch.Size([batch_size, in_channels, height, width])

        returns: torch.Size([batch_size, in_size_FC])
        """
        # CNN
        x = F.relu(self.conv(s))

        # reshape from torch.Size([batch_size, out_channels, out_height, out_width]) to 
        # torch.Size([batch_size, out_channels * out_height * out_width])
        x = x.view(x.shape[0], -1)

        return x


class MinAtar_DQN(nn.Module):
    """Defines the DQN consisting of the CNN part and a fully-connected layer."""
    def __init__(self, in_channels, height, width, num_actions):
        super().__init__()
        
        self.core = MinAtar_CoreNet(in_channels = in_channels,
                                    height      = height,
                                    width       = width)

        self.head = MLP(in_size   = self.core.in_size_FC, 
                        out_size  = num_actions,
                        net_struc = [[128, "relu"], "identity"])
    
    def forward(self, s):
        x = self.core(s)
        return self.head(x)


class MinAtar_BootDQN(nn.Module):
    """Defines the BootDQN consisting of the common CNN part and K different heads."""
    def __init__(self, in_channels, height, width, num_actions, K):
        super().__init__()
        
        self.core = MinAtar_CoreNet(in_channels = in_channels,
                                    height      = height,
                                    width       = width)

        self.heads = nn.ModuleList([MLP(in_size   = self.core.in_size_FC, 
                                        out_size  = num_actions,
                                        net_struc = [[128, "relu"], "identity"]) for _ in range(K)])

    def forward(self, s, head=None):
        """Returns for a state s all Q(s,a) for each k. Args:
        s: torch.Size([batch_size, in_channels, height, width])

        returns:
        list of length K with each element being torch.Size([batch_size, num_actions]) if head is None,
        torch.Size([batch_size, num_actions]) else."""

        # CNN part
        x = self.core(s)

        # K heads
        if head is None:
            return [head_net(x) for head_net in self.heads]
        else:
            return self.heads[head](x)


# --------------------------- LSTM ---------------------------------
class LSTM_Actor(nn.Module):
    """Defines recurrent deterministic actor."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Actor, self).__init__()
        
        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(obs_dim, 128)
        self.curr_fe_dense2 = nn.Linear(128, 128)
        
        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(obs_dim + action_dim, 128)
        else:
            self.mem_dense = nn.Linear(obs_dim, 128)
        self.mem_LSTM = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1, batch_first = True)
        
        # post combination
        self.post_comb_dense1 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2 = nn.Linear(128, action_dim)


    def forward(self, o, o_hist, a_hist, hist_len) -> tuple:
        """o, o_hist, hist_len are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        
        returns: output with shape torch.Size([batch_size, action_dim]), act_net_info (dict)
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #------ current feature extraction ------
        curr_fe = F.relu(self.curr_fe_dense1(o))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))

        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(o_hist))
        
        # LSTM
        #self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]
        
        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0

        #------ post combination ------
        # concate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)

        # final dense layers
        x = F.relu(self.post_comb_dense1(x))
        x = torch.tanh(self.post_comb_dense2(x))
        
        # create dict for logging
        act_net_info = dict(Actor_CurFE = curr_fe.detach().mean().cpu().numpy(),
                            Actor_ExtMemory = hidden_mem.detach().mean().cpu().numpy())
        
        # return output
        return x, act_net_info


class LSTM_Critic(nn.Module):
    """Defines recurrent critic network to compute Q-values."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Critic, self).__init__()
        
        self.use_past_actions = use_past_actions

        # current feature extraction
        self.curr_fe_dense1 = nn.Linear(obs_dim + action_dim, 128)
        self.curr_fe_dense2 = nn.Linear(128, 128)
        
        # memory
        if use_past_actions:
            self.mem_dense = nn.Linear(obs_dim + action_dim, 128)
        else:
            self.mem_dense = nn.Linear(obs_dim, 128)
        self.mem_LSTM = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1, batch_first = True)
        
        # post combination
        self.post_comb_dense1 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2 = nn.Linear(128, 1)
        

    def forward(self, o, a, o_hist, a_hist, hist_len, log_info=True) -> tuple:
        """o, o_hist, a_hist are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        a:        torch.Size([batch_size, action_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        log_info: Bool, whether to return logging dict
        
        returns: output with shape torch.Size([batch_size, 1]), critic_net_info (dict) (if log_info)
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe = F.relu(self.curr_fe_dense1(oa))
        curr_fe = F.relu(self.curr_fe_dense2(curr_fe))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem = F.relu(self.mem_dense(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem = F.relu(self.mem_dense(o_hist))
        
        # LSTM
        #self.mem_LSTM.flatten_parameters()
        extracted_mem, (_, _) = self.mem_LSTM(x_mem)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1

        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem = extracted_mem[torch.arange(extracted_mem.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        x = torch.cat([curr_fe, hidden_mem], dim=1)
        
        # final dense layers
        x = F.relu(self.post_comb_dense1(x))
        x = self.post_comb_dense2(x)

        # create dict for logging
        if log_info:
            critic_net_info = dict(Critic_CurFE = curr_fe.detach().mean().cpu().numpy(),
                                   Critic_ExtMemory = hidden_mem.detach().mean().cpu().numpy())
            return x, critic_net_info
        else:
            return x


class LSTM_Double_Critic(nn.Module):
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Double_Critic, self).__init__()

        self.LSTM_Q1 = LSTM_Critic(action_dim       = action_dim, 
                                   obs_dim          = obs_dim,
                                   use_past_actions = use_past_actions)

        self.LSTM_Q2 = LSTM_Critic(action_dim       = action_dim, 
                                   obs_dim          = obs_dim,
                                   use_past_actions = use_past_actions)

    def forward(self, o, a, o_hist, a_hist, hist_len) -> tuple:
        q1                  = self.LSTM_Q1(o, a, o_hist, a_hist, hist_len, log_info=False)
        q2, critic_net_info = self.LSTM_Q2(o, a, o_hist, a_hist, hist_len, log_info=True)

        return q1, q2, critic_net_info


    def single_forward(self, o, a, o_hist, a_hist, hist_len):
        q1 = self.LSTM_Q1(o, a, o_hist, a_hist, hist_len, log_info=False)

        return q1


#-------------------------- SAC: GaussianActor ----------------------------

class GaussianActor(nn.Module):
    """Defines stochastic actor based on a Gaussian distribution."""
    def __init__(self, action_dim, state_dim, log_std_min=-20, log_std_max=2):
        super(GaussianActor, self).__init__()

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
