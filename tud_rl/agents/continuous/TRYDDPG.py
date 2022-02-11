import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tud_rl.agents.continuous.DDPG import DDPGAgent
from tud_rl.common.logging_func import *


class TRYDDPGAgent(DDPGAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True, init_critic=True)

        # attributes and hyperparameters
        self.MC_batch_size   = c["agent"][agent_name]["MC_batch_size"]
        self.tgt_to_upd_bias = c["agent"][agent_name]["tgt_to_upd_bias"]

        self.bias_net = copy.deepcopy(self.critic).to(self.device)

        # define critic optimizer
        if self.optimizer == "Adam":
            self.bias_optimizer = optim.Adam(self.bias_net.parameters(), lr=self.lr)
        else:
            self.bias_optimizer = optim.RMSprop(self.bias_net.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    def greedy_action_Q(self, s, with_bias=True):
        # greedy action
        a = self._greedy_action(s)

        # evaluate
        sa = torch.cat([s, a], dim=1)

        if with_bias:
            return self.critic(sa).item(), self.bias_net(sa).item()
        else:
            return self.critic(sa).item()

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            target_a = self.target_actor(s2)

            # next Q-estimate
            s2_tgta = torch.cat([s2, target_a], dim=1)
            Q_next = self.target_critic(s2_tgta) - self.bias_net(s2_tgta)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def train_bias_net(self, s, a, MC):
        """Updates the bias network based on recent on-policy rollouts.

        Args:
            s:  np.array([MC_batch_size, in_channels, height, width]))
            a:  np.array([MC_batch_size, 1]))
            MC: np.array([MC_batch_size, 1])
        """
        s  = torch.tensor(s.astype(np.float32))
        a  = torch.tensor(a.astype(np.int64))
        MC = torch.tensor(MC.astype(np.float32))

        sa = torch.cat([s, a], dim=1)

        # clear gradients
        self.bias_optimizer.zero_grad()

        # bias estimate
        B = self.bias_net(sa)

        # get target
        with torch.no_grad():

            if self.tgt_to_upd_bias:
                Q = self.target_critic(sa)
            else:
                Q = self.critic(sa)

            y_bias = Q - MC
        
        # loss
        bias_loss = self._compute_loss(B, y_bias)

        # compute gradients
        bias_loss.backward()

        # perform optimizing step
        self.bias_optimizer.step()

        # log critic training
        self.logger.store(Bias_loss=bias_loss.detach().cpu().numpy().item())
        self.logger.store(Bias_val=B.detach().mean().cpu().numpy().item())
