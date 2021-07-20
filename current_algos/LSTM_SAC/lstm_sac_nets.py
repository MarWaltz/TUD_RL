import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class LSTM_GaussianActor(nn.Module):
    """Defines recurrent, stochastic actor based on a Gaussian distribution."""
    def __init__(self, action_dim, obs_dim, use_past_actions, log_std_min = -20, log_std_max = 2):
        super(LSTM_GaussianActor, self).__init__()

        self.use_past_actions = use_past_actions
        self.log_std_min      = log_std_min
        self.log_std_max      = log_std_max

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
        self.post_comb_dense = nn.Linear(128 + 128, 128)

        # output mu and log_std
        self.mu      = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, o, o_hist, a_hist, hist_len, deterministic, with_logprob):
        """Returns action and it's logprob for given obs and history. o, o_hist, a_hist, hist_len are torch tensors. Args:

        o:        torch.Size([batch_size, obs_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)

        deterministic: bool (whether to use mean as a sample, only at test time)
        with_logprob:  bool (whether to return logprob of sampled action as well, else second tuple element below will be 'None')

        returns:       (torch.Size([batch_size, action_dim]), torch.Size([batch_size, action_dim]), act_net_info (dict))

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

        # final dense layer
        x = F.relu(self.post_comb_dense(x))

        # compute mean, log_std and std of Gaussian
        mu      = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std     = torch.exp(log_std)
        
        #------ having mu and std, compute actions and log_probs -------
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

        #------ return ---------
        # create dict for logging
        act_net_info = dict(Actor_CurFE = curr_fe.detach().mean().cpu().numpy(),
                            Actor_ExtMemory = hidden_mem.detach().mean().cpu().numpy())
        
        # return squashed action, it's logprob and logging info
        return pi_action, logp_pi, act_net_info


class LSTM_Double_Critic(nn.Module):
    """Defines two recurrent critic networks to compute Q-values."""
    
    def __init__(self, action_dim, obs_dim, use_past_actions) -> None:
        super(LSTM_Double_Critic, self).__init__()
        
        self.use_past_actions = use_past_actions

        # ----------------------- Q1 ---------------------------
        # current feature extraction
        self.curr_fe_dense1_q1 = nn.Linear(obs_dim + action_dim, 128)
        self.curr_fe_dense2_q1 = nn.Linear(128, 128)
        
        # memory
        if use_past_actions:
            self.mem_dense_q1 = nn.Linear(obs_dim + action_dim, 128)
        else:
            self.mem_dense_q1 = nn.Linear(obs_dim, 128)
        self.mem_LSTM_q1 = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1, batch_first = True)
        
        # post combination
        self.post_comb_dense1_q1 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2_q1 = nn.Linear(128, 1)
        
        # ----------------------- Q2 ---------------------------
        # current feature extraction
        self.curr_fe_dense1_q2 = nn.Linear(obs_dim + action_dim, 128)
        self.curr_fe_dense2_q2 = nn.Linear(128, 128)
        
        # memory
        if use_past_actions:
            self.mem_dense_q2 = nn.Linear(obs_dim + action_dim, 128)
        else:
            self.mem_dense_q2 = nn.Linear(obs_dim, 128)
        self.mem_LSTM_q2 = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1, batch_first = True)
        
        # post combination
        self.post_comb_dense1_q2 = nn.Linear(128 + 128, 128)
        self.post_comb_dense2_q2 = nn.Linear(128, 1)
        

    def forward(self, o, a, o_hist, a_hist, hist_len) -> tuple:
        """o, a, o_hist, a_hist, hist_len are torch tensors. Shapes:
        o:        torch.Size([batch_size, obs_dim])
        a:        torch.Size([batch_size, action_dim])
        o_hist:   torch.Size([batch_size, history_length, obs_dim])
        a_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len: torch.Size(batch_size)
        
        returns: tuple with two elements, each having shape torch.Size([batch_size, 1])
        
        Note: 
        The one-layer LSTM is defined with batch_first=True, hence it expects input in form of:
        x = (batch_size, seq_length, obs_dim)
        
        The call <out, (hidden, cell) = LSTM(x)> results in: 
        out:    Output (= hidden state) of LSTM for each time step with shape (batch_size, seq_length, hidden_size).
        hidden: The hidden state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        cell:   The cell state of the last time step in each sequence with shape (1, batch_size, hidden_size).
        """

        #-------------------- Q1 ------------------------
        
        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe_q1 = F.relu(self.curr_fe_dense1_q1(oa))
        curr_fe_q1 = F.relu(self.curr_fe_dense2_q1(curr_fe_q1))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem_q1 = F.relu(self.mem_dense_q1(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem_q1 = F.relu(self.mem_dense_q1(o_hist))
        
        # LSTM
        #self.mem_LSTM_q1.flatten_parameters()
        extracted_mem_q1, (_, _) = self.mem_LSTM_q1(x_mem_q1)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem_q1 = extracted_mem_q1[torch.arange(extracted_mem_q1.size(0)), h_idx]

        # mask no-history cases to yield zero extracted memory
        hidden_mem_q1[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        q1 = torch.cat([curr_fe_q1, hidden_mem_q1], dim=1)
        
        # final dense layers
        q1 = F.relu(self.post_comb_dense1_q1(q1))
        q1 = self.post_comb_dense2_q1(q1)
        
        #-------------------- Q2 ------------------------
        
        #------ current feature extraction ------
        # concatenate obs and act
        oa = torch.cat([o, a], dim=1)
        curr_fe_q2 = F.relu(self.curr_fe_dense1_q2(oa))
        curr_fe_q2 = F.relu(self.curr_fe_dense2_q2(curr_fe_q2))
        
        #------ memory ------
        # dense layer
        if self.use_past_actions:
            x_mem_q2 = F.relu(self.mem_dense_q2(torch.cat([o_hist, a_hist], dim=2)))
        else:
            x_mem_q2 = F.relu(self.mem_dense_q2(o_hist))
        
        # LSTM
        #self.mem_LSTM_q2.flatten_parameters()
        extracted_mem_q2, (_, _) = self.mem_LSTM_q2(x_mem_q2)

        # get selection index according to history lengths (no-history cases will be masked later)
        h_idx = copy.deepcopy(hist_len)
        h_idx[h_idx == 0] = 1
        h_idx -= 1
        
        # select LSTM output, resulting shape is (batch_size, hidden_dim)
        hidden_mem_q2 = extracted_mem_q2[torch.arange(extracted_mem_q2.size(0)), h_idx]
        
        # mask no-history cases to yield zero extracted memory
        hidden_mem_q2[hist_len == 0] = 0
        
        #------ post combination ------
        # concatenate current feature extraction with generated memory
        q2 = torch.cat([curr_fe_q2, hidden_mem_q2], dim=1)
        
        # final dense layers
        q2 = F.relu(self.post_comb_dense1_q2(q2))
        q2 = self.post_comb_dense2_q2(q2)

        #--------------- return output -----------------
        # create dict for logging
        critic_net_info = dict(Q1_CurFE = curr_fe_q1.detach().mean().cpu().numpy(),
                               Q2_CurFE = curr_fe_q2.detach().mean().cpu().numpy(),
                               Q1_ExtMemory = hidden_mem_q1.detach().mean().cpu().numpy(),
                               Q2_ExtMemory = hidden_mem_q2.detach().mean().cpu().numpy())
        
        return q1, q2, critic_net_info
