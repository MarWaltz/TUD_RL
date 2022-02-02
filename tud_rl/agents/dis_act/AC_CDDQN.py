import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tud_rl.agents.dis_act.DQN import DQNAgent
from tud_rl.common.logging_func import *
from tud_rl.common.nets import MLP, MinAtar_DQN


class AC_CDDQN_Agent(DQNAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name, logging=False)

        # attributes and hyperparameters
        self.AC_K = c["agent"][agent_name]["AC_K"]

        # replace DQN + target by DQN_A + DQN_B
        self.DQN_A = self.DQN
        del self.target_DQN

        if self.state_type == "image":
            self.DQN_B = MinAtar_DQN(in_channels = self.state_shape[0],
                                     height      = self.state_shape[1],
                                     width       = self.state_shape[2],
                                     num_actions = self.num_actions).to(self.device)
        elif self.state_type == "feature":
            self.DQN_B = MLP(in_size   = self.state_shape,
                             out_size  = self.num_actions, 
                             net_struc = self.net_struc).to(self.device)

        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params DQN: {self._count_params(self.DQN_A)}")
            print("--------------------------------------------")
        
        # prior weights
        if self.dqn_weights is not None:
            raise NotImplementedError("Weight loading for AC_CDDQN is not implemented yet.")

        #  optimizer
        del self.DQN_optimizer

        if self.optimizer == "Adam":
            self.DQN_A_optimizer = optim.Adam(self.DQN_A.parameters(), lr=self.lr)
            self.DQN_B_optimizer = optim.Adam(self.DQN_B.parameters(), lr=self.lr)
        else:
            self.DQN_A_optimizer = optim.RMSprop(self.DQN_A.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)
            self.DQN_B_optimizer = optim.RMSprop(self.DQN_B.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)


    @torch.no_grad()
    def _greedy_action(self, s):
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN_A(s).to(self.device) + self.DQN_B(s).to(self.device)

        # greedy
        return torch.argmax(q).item()


    def train(self):
        raise NotImplementedError("Not yet.")
        self.train_A()
        self.train_B()

    def train_A(self):
        """Samples from replay_buffer and updates DQN."""        

        #-------- train DQN_A --------
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        # clear gradients
        self.DQN_A_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        QA_v = self.DQN_A(s)
        QA_v = torch.gather(input=QA_v, dim=1, index=a)
 
        # calculate targets
        with torch.no_grad():

            # compute candidate set based on QB
            QB_v2 = self.DQN_B(s2)
            M_K = torch.argsort(QB_v2, dim=1, descending=True)[:, :self.action_candidate_K]

            # get a_star_K
            QA_v2 = self.DQN_A(s2)
            a_star_K = torch.empty((self.batch_size, 1), dtype=torch.int64).to(self.device)

            for bat_idx in range(self.batch_size):
                
                # get best action of the candidate set
                act_idx = torch.argmax(QA_v2[bat_idx][M_K[bat_idx]])

                # store its index
                a_star_K[bat_idx] = M_K[bat_idx][act_idx]

            # evaluate a_star_K on B
            target_Q_next = torch.gather(QB_v2, dim=1, index=a_star_K)

            # clip to ME
            ME = torch.max(QA_v2, dim=1).values.reshape(self.batch_size, 1)
            target_Q_next = torch.min(target_Q_next, ME)

            # target
            target_Q = r + self.gamma * target_Q_next * (1 - d)

        # calculate loss
        if self.loss == "MSELoss":
            loss_A = F.mse_loss(QA_v, target_Q)

        elif self.loss == "SmoothL1Loss":
            loss_A = F.smooth_l1_loss(QA_v, target_Q)
        
        # compute gradients
        loss_A.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN_A.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN_A.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_A_optimizer.step()
        
        # log critic training
        self.logger.store(Loss=loss_A.detach().cpu().numpy().item())
        self.logger.store(Q_val=QA_v.detach().mean().cpu().numpy().item())

    def train_B(self):
        """Samples from replay_buffer and updates DQN."""        

        #-------- train DQN_B --------
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        # clear gradients
        self.DQN_B_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        QB_v = self.DQN_B(s)
        QB_v = torch.gather(input=QB_v, dim=1, index=a)
 
        # calculate targets
        with torch.no_grad():

            # compute candidate set based on QA
            QA_v2 = self.DQN_A(s2)
            M_K = torch.argsort(QA_v2, dim=1, descending=True)[:, :self.action_candidate_K]

            # get a_star_K
            QB_v2 = self.DQN_B(s2)
            a_star_K = torch.empty((self.batch_size, 1), dtype=torch.int64).to(self.device)

            for bat_idx in range(self.batch_size):
                
                # get best action of the candidate set
                act_idx = torch.argmax(QB_v2[bat_idx][M_K[bat_idx]])

                # store its index
                a_star_K[bat_idx] = M_K[bat_idx][act_idx]

            # evaluate a_star_K on A
            target_Q_next = torch.gather(QA_v2, dim=1, index=a_star_K)

            # clip to ME
            ME = torch.max(QB_v2, dim=1).values.reshape(self.batch_size, 1)
            target_Q_next = torch.min(target_Q_next, ME)

            # target
            target_Q = r + self.gamma * target_Q_next * (1 - d)

        # calculate loss
        if self.loss == "MSELoss":
            loss_B = F.mse_loss(QB_v, target_Q)

        elif self.loss == "SmoothL1Loss":
            loss_B = F.smooth_l1_loss(QB_v, target_Q)
        
        # compute gradients
        loss_B.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN_B.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN_B.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_B_optimizer.step()
