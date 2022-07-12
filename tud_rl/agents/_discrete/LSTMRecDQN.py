import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl import logger
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.exploration import LinearDecayEpsilonGreedy


class LSTMRecDQNAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.lr              = c.lr
        self.dqn_weights     = c.dqn_weights
        self.eps_init        = c.eps_init
        self.eps_final       = c.eps_final
        self.eps_decay_steps = c.eps_decay_steps
        self.tgt_update_freq = c.tgt_update_freq
        self.net_struc       = c.net_struc
        self.history_length  = getattr(c.Agent, agent_name)["history_length"]

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)), "Need prior weights in test mode."

        assert self.state_type == "feature", "LSTMRecDQN is currently based on features."

        if self.net_struc is not None:
            logger.warning("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        assert self.history_length == 2, "Currently, only 'history_length = 2' is available."

        # linear epsilon schedule
        self.exploration = LinearDecayEpsilonGreedy(eps_init        = self.eps_init, 
                                                    eps_final       = self.eps_final,
                                                    eps_decay_steps = self.eps_decay_steps)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer_LSTM(state_type     = self.state_type, 
                                                                 state_shape    = self.state_shape, 
                                                                 buffer_length  = self.buffer_length,
                                                                 batch_size     = self.batch_size,
                                                                 device         = self.device,
                                                                 disc_actions   = True,
                                                                 history_length = self.history_length)

        # init DQN
        if self.state_type == "feature":
            self.DQN = nets.LSTMRecDQN(num_actions = self.num_actions).to(self.device)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.DQN)

        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights, map_location=self.device))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)


    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Epsilon-greedy based action selection for a given state.

        Arg s:   np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
        returns: int for the action
        """

        # get current epsilon
        curr_epsilon = self.exploration.get_epsilon(self.mode)

        # random
        if np.random.binomial(1, curr_epsilon) == 1:
            a = np.random.randint(low=0, high=self.num_actions, size=1, dtype=int).item()
            
        # greedy
        else:
            a = self._greedy_action(s, s_hist, a_hist, hist_len)
        return a


    @torch.no_grad()
    def _greedy_action(self, s, s_hist, a_hist, hist_len, with_Q=False):
        """Selects a greedy action.

        Args:
            s:        np.array with shape (state_shape,)
            s_hist:   np.array with shape (history_length, state_shape)
            a_hist:   np.array with shape (history_length, 1)
            hist_len: int
        
        Returns: 
            int
        """

        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(s_hist, dtype=torch.float32).view(1, self.history_length, self.state_shape).to(self.device)
        a_hist = torch.tensor(a_hist, dtype=torch.int32).view(1, self.history_length, 1).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        q = self.DQN(s, s_hist, a_hist, hist_len).to(self.device)
        
        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, q[0][a].item()
        return a


    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)


    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
 
        with torch.no_grad():
            Q_next_main = self.DQN(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            a2 = torch.argmax(Q_next_main, dim=1).reshape(self.batch_size, 1)

            Q_next_tgt = self.target_DQN(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.gather(input=Q_next_tgt, dim=1, index=a2)
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)


    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d = batch

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # Q-estimates
        Q = self.DQN(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        Q = torch.gather(input=Q, dim=1, index=a)
 
        # calculate targets
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

        # calculate loss
        loss = self._compute_loss(Q=Q, y=y)

        # compute gradients
        loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_optimizer.step()
        
        # log critic training
        self.logger.store(Loss=loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        self._target_update()


    @torch.no_grad()
    def _target_update(self):

        # increase target-update cnt
        self.tgt_up_cnt += 1

        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())
