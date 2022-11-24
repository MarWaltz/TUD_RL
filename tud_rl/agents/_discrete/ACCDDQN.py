import math

import torch
import torch.nn as nn
import torch.optim as optim

import tud_rl.common.nets as nets
from tud_rl.agents._discrete.DQN import DQNAgent
from tud_rl.common.configparser import ConfigFile


class ACCDDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.AC_K = getattr(c.Agent, agent_name)["AC_K"]

        # checks
        assert self.AC_K <= self.num_actions, "ACC-K cannot exceed number of actions."

        # init two nets
        self.DQN = nn.ModuleList().to(self.device)

        if self.state_type == "image":
            for _ in range(2):
                self.DQN.append(nets.MinAtar_DQN(in_channels = self.state_shape[0],
                                                height      = self.state_shape[1],
                                                width       = self.state_shape[2],
                                                num_actions = self.num_actions).to(self.device))
        elif self.state_type == "feature":
            for _ in range(2):
                self.DQN.append(nets.MLP(in_size   = self.state_shape,
                                         out_size  = self.num_actions, 
                                         net_struc = self.net_struc).to(self.device))

        # number of parameters of net
        self.n_params = self._count_params(self.DQN)

        # prior weights
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights, map_location=self.device))

        #  optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_A_optimizer = optim.RMSprop(self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

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
        q = self.DQN[0](s).to(self.device) + self.DQN[1](s).to(self.device)

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

        #-------- train both nets --------
        # Note: The description of the training process is not completely clear, see Section 'Deep Version' of Jiang et. al (2021).
        #       Here, both nets will be updated towards the same target, stemming from Equation (12). Alternatively, one could compute
        #       two distinct targets based on different buffer samples and train each net separately. 

        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # Q-values
        QA = self.DQN[0](s)
        QA = torch.gather(input=QA, dim=1, index=a)

        QB = self.DQN[1](s)
        QB = torch.gather(input=QB, dim=1, index=a)
 
        # targets
        with torch.no_grad():

            # compute candidate set based on QB
            QB_v2 = self.DQN[1](s2)
            M_K = torch.argsort(QB_v2, dim=1, descending=True)[:, :self.AC_K]

            # get a_star_K
            QA_v2 = self.DQN[0](s2)
            a_star_K = torch.empty((self.batch_size, 1), dtype=torch.int64).to(self.device)

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
        loss = self._compute_loss(QA, y) + self._compute_loss(QB, y)
       
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
        self.logger.store(Q_val=QA.detach().mean().cpu().numpy().item())
