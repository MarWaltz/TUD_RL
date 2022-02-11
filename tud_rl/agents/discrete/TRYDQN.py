import copy

import torch
import torch.optim as optim
from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.common.logging_func import *


class TRYDQNAgent(DQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True)

        # attributes and hyperparameters
        self.MC_batch_size   = c["agent"][agent_name]["MC_batch_size"]
        self.tgt_to_upd_bias = c["agent"][agent_name]["tgt_to_upd_bias"]

        self.bias_net = copy.deepcopy(self.DQN).to(self.device)

        # define optimizer
        if self.optimizer == "Adam":
            self.bias_optimizer = optim.Adam(self.bias_net.parameters(), lr=self.lr)
        else:
            self.bias_optimizer = optim.RMSprop(self.bias_net.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    def greedy_action_Q(self, s):
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN(s).to(self.device)

        # greedy
        return torch.max(q).item()

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            Q_next = self.target_DQN(s2) - self.bias_net(s2)
            Q_next = torch.max(Q_next, dim=1).values.reshape(self.batch_size, 1)
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

        # clear gradients
        self.bias_optimizer.zero_grad()

        # bias estimate
        B = self.bias_net(s)
        B = torch.gather(input=B, dim=1, index=a)

        # get target
        with torch.no_grad():

            if self.tgt_to_upd_bias:
                Q = self.target_DQN(s)
            else:
                Q = self.DQN(s)
            Q = torch.gather(input=Q, dim=1, index=a)

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
