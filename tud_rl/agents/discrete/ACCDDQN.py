import math

import torch
import torch.nn as nn
import torch.optim as optim

import tud_rl.common.nets as nets

from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.common.configparser import ConfigFile

class ACCDDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.AC_K = getattr(c.Agent, agent_name)["AC_K"]

        # checks
        assert self.AC_K <= self.num_actions, "ACC-K cannot exceed number of actions."

        # replace DQN + target by DQN_A + DQN_B
        self.DQN_A = self.DQN
        del self.target_DQN

        if self.state_type == "image":
            self.DQN_B = nets.MinAtar_DQN(in_channels=self.state_shape[0],
                                          height=self.state_shape[1],
                                          width=self.state_shape[2],
                                          num_actions=self.num_actions).to(self.device)
        elif self.state_type == "feature":
            self.DQN_B = nets.MLP(in_size=self.state_shape,
                                  out_size=self.num_actions,
                                  net_struc=self.net_struc).to(self.device)

        # Number of Parameters of Net
        self.n_params = self._count_params(self.DQN_A)

        # prior weights
        if self.dqn_weights is not None:
            raise NotImplementedError(
                "Weight loading for AC_CDDQN is not implemented yet.")

        #  optimizer
        del self.DQN_optimizer

        if self.optimizer == "Adam":
            self.DQN_A_optimizer = optim.Adam(
                self.DQN_A.parameters(), lr=self.lr)
            self.DQN_B_optimizer = optim.Adam(
                self.DQN_B.parameters(), lr=self.lr)
        else:
            self.DQN_A_optimizer = optim.RMSprop(
                self.DQN_A.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)
            self.DQN_B_optimizer = optim.RMSprop(
                self.DQN_B.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action.
        Args:
            s:      np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
            with_Q: bool, whether to return the associated Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)
        """

        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN_A(s).to(self.device) + self.DQN_B(s).to(self.device)

        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, 0.5 * q[0][a].item()
        return a

    def train(self):
        """Samples from replay_buffer and updates DQN."""

        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s, a, r, s2, d = batch

        # -------- train DQN_A & DQN_B --------
        # Note: The description of the training process is not completely clear, see Section 'Deep Version' of Jiang et. al (2021).
        #       Here, both nets will be updated towards the same target, stemming from Equation (12). Alternatively, one could compute
        #       two distinct targets based on different buffer samples and train each net separately.

        # clear gradients
        self.DQN_A_optimizer.zero_grad()
        self.DQN_B_optimizer.zero_grad()

        # Q-values
        QA = self.DQN_A(s)
        QA = torch.gather(input=QA, dim=1, index=a)

        QB = self.DQN_B(s)
        QB = torch.gather(input=QB, dim=1, index=a)

        # targets
        with torch.no_grad():

            # compute candidate set based on QB
            QB_v2 = self.DQN_B(s2)
            M_K = torch.argsort(QB_v2, dim=1, descending=True)[:, :self.AC_K]

            # get a_star_K
            QA_v2 = self.DQN_A(s2)
            a_star_K = torch.empty((self.batch_size, 1),
                                   dtype=torch.int64).to(self.device)

            for bat_idx in range(self.batch_size):

                # get best action of the candidate set
                act_idx = torch.argmax(QA_v2[bat_idx][M_K[bat_idx]])

                # store its index
                a_star_K[bat_idx] = M_K[bat_idx][act_idx]

            # evaluate a_star_K on B
            Q_next = torch.gather(QB_v2, dim=1, index=a_star_K)

            # clip to ME
            ME = torch.max(QA_v2, dim=1).values.reshape(self.batch_size, 1)
            Q_next = torch.min(Q_next, ME)

            # target
            y = r + self.gamma * Q_next * (1 - d)

        # calculate loss
        loss_A = self._compute_loss(QA, y)
        loss_B = self._compute_loss(QB, y)

        # compute gradients
        loss_A.backward()
        loss_B.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN_A.parameters():
                p.grad *= 1 / math.sqrt(2)
            for p in self.DQN_B.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN_A.parameters(), max_norm=10)
            nn.utils.clip_grad_norm_(self.DQN_B.parameters(), max_norm=10)

        # perform optimizing step
        self.DQN_A_optimizer.step()
        self.DQN_B_optimizer.step()

        # log critic training
        self.logger.store(Loss=loss_A.detach().cpu().numpy().item())
        self.logger.store(Q_val=QA.detach().mean().cpu().numpy().item())