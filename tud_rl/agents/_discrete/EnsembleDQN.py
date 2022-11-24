import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim

import tud_rl.common.nets as nets
from tud_rl.agents._discrete.DQN import DQNAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.logging_func import *


class EnsembleDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.N           = getattr(c.Agent, agent_name)["N"]
        self.N_to_update = getattr(c.Agent, agent_name)["N_to_update"]

        # init EnsembleDQN
        self.DQN = nn.ModuleList().to(self.device)

        for _ in range(self.N):
            if self.state_type == "image":
                self.DQN.append(nets.MinAtar_DQN(in_channels = self.state_shape[0],
                                                 height      = self.state_shape[1],
                                                 width       = self.state_shape[2],
                                                 num_actions = self.num_actions).to(self.device))

            elif self.state_type == "feature":
                self.DQN.append(nets.MLP(in_size   = self.state_shape,
                                         out_size  = self.num_actions, 
                                         net_struc = self.net_struc).to(self.device))
        # parameter number of net
        self.n_params = self._count_params(self.DQN)

        # prior weights
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights, map_location=self.device))

        # target net and counter for target update
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

    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.mean(q_ens, dim=0)

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action by maximizing over the reduced ensemble.
        
        Args:
            s:      np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
            with_Q: bool, whether to return the associate ensemble average of Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)"""

        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward through ensemble
        q_ens = [net(s).to(self.device) for net in self.DQN] # list of torch.Size([batch_size, num_actions])
        q_ens = torch.stack(q_ens).to(self.device)                   # torch.Size([N, batch_size, num_actions])   

        # reduction over ensemble
        q = self._ensemble_reduction(q_ens).to(self.device)

        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, q[0][a].item()
        return a


    def _compute_target(self, r, s2, d):
        with torch.no_grad():

            # forward through ensemble
            Q_next_ens = [net(s2).to(self.device) for net in self.target_DQN]
            Q_next_ens = torch.stack(Q_next_ens).to(self.device)
            
            # reduction over ensemble
            Q_next = self._ensemble_reduction(Q_next_ens)

            # maximization and target
            Q_next = torch.max(Q_next, dim=1).values.reshape(self.batch_size, 1)
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def train(self):
        """Samples from replay_buffer, updates critic and the target networks.""" 
       
        #-------- train EnsembleDQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad(set_to_none=True)
        
        for _ in range(self.N_to_update):
            
            # ensemble member to update
            i = np.random.choice(self.N)

            # sample batch
            batch = self.replay_buffer.sample()
        
            # unpack batch
            s, a, r, s2, d = batch
            
            # Q estimates
            Q = self.DQN[i](s)
            Q = torch.gather(input=Q, dim=1, index=a)
 
            # targets
            y = self._compute_target(r, s2, d)

            # loss
            loss = self._compute_loss(Q=Q, y=y)
            
            # compute gradients
            loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.DQN[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.DQN[i].parameters(), max_norm=10)
            
            # perform optimizing step
            self.DQN_optimizer.step()
            
            # log critic training
            self.logger.store(Loss=loss.detach().cpu().numpy().item())
            self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        self._target_update()


    def _target_update(self):
        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())

        # increase target-update cnt
        self.tgt_up_cnt += 1
