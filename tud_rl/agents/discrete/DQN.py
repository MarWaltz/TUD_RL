import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tud_rl.agents.BaseAgent import BaseAgent
from tud_rl.common.buffer import UniformReplayBuffer
from tud_rl.common.exploration import LinearDecayEpsilonGreedy
from tud_rl.common.logging_func import *
from tud_rl.common.nets import MLP, MinAtar_DQN


class DQNAgent(BaseAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.lr              = c["lr"]
        self.dqn_weights     = c["dqn_weights"]
        self.eps_init        = c["eps_init"]
        self.eps_final       = c["eps_final"]
        self.eps_decay_steps = c["eps_decay_steps"]
        self.tgt_update_freq = c["tgt_update_freq"]
        self.net_struc       = c["net_struc"]

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image" and self.net_struc is not None:
            raise Exception("For CNN-based nets, the specification of 'net_struc_dqn' should be 'None'.")

        # linear epsilon schedule
        self.exploration = LinearDecayEpsilonGreedy(eps_init        = self.eps_init, 
                                                    eps_final       = self.eps_final,
                                                    eps_decay_steps = self.eps_decay_steps)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = UniformReplayBuffer(state_type    = self.state_type, 
                                                     state_shape   = self.state_shape, 
                                                     buffer_length = self.buffer_length,
                                                     batch_size    = self.batch_size,
                                                     device        = self.device,
                                                     disc_actions  = True)

        # init DQN
        if self.state_type == "image":
            self.DQN = MinAtar_DQN(in_channels = self.state_shape[0],
                                   height      = self.state_shape[1],
                                   width       = self.state_shape[2],
                                   num_actions = self.num_actions).to(self.device)

        elif self.state_type == "feature":
            self.DQN = MLP(in_size   = self.state_shape,
                           out_size  = self.num_actions, 
                           net_struc = self.net_struc).to(self.device)

        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str, info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params: {self._count_params(self.DQN)}")
            print("--------------------------------------------")
        
        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights))

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
    def select_action(self, s):
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
            a = self._greedy_action(s)
        return a


    @torch.no_grad()
    def _greedy_action(self, s):
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN(s).to(self.device)

        # greedy
        return torch.argmax(q).item()


    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            Q_next = self.target_DQN(s2)
            Q_next = torch.max(Q_next, dim=1).values.reshape(self.batch_size, 1)
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)


    def train(self):
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # Q estimates
        Q = self.DQN(s)
        Q = torch.gather(input=Q, dim=1, index=a)
 
        # targets
        y = self._compute_target(r, s2, d)

        # loss
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

        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())

        # increase target-update cnt
        self.tgt_up_cnt += 1
