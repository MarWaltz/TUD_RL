import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.common.logging_func import *
from tud_rl.common.nets import MLP, MinAtar_DQN


class EnsembleDQNAgent(DQNAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name, logging=False)

        # attributes and hyperparameters
        self.N           = c["agent"][agent_name]["N"]
        self.N_to_update = c["agent"][agent_name]["N_to_update"]

        # init EnsembleDQN
        del self.DQN
        self.EnsembleDQN = [None] * self.N

        for i in range(self.N):
            if self.state_type == "image":
                self.EnsembleDQN[i] = MinAtar_DQN(in_channels = self.state_shape[0],
                                                  height      = self.state_shape[1],
                                                  width       = self.state_shape[2],
                                                  num_actions = self.num_actions).to(self.device)

            elif self.state_type == "feature":
                self.EnsembleDQN[i] = MLP(in_size   = self.state_shape,
                                          out_size  = self.num_actions, 
                                          net_struc = self.net_struc).to(self.device)
        

        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str)
            self.logger.save_config({"agent_name" : self.name, **c})

            print("--------------------------------------------")
            print(f"n_params: {self.N * self._count_params(self.EnsembleDQN[0])}")
            print("--------------------------------------------")

        # prior weights
        if self.dqn_weights is not None:
            raise NotImplementedError("Prior weights not implemented so far for EnsembleDQN.")

        # target net and counter for target update
        del self.target_DQN

        self.target_EnsembleDQN = copy.deepcopy(self.EnsembleDQN).to(self.device)
        self.tgt_up_cnt = 0
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for net in self.target_EnsembleDQN:
            for p in net.parameters():
                p.requires_grad = False

        # define optimizer
        del self.DQN_optimizer

        self.EnsembleDQN_optimizer = [None] * self.N

        for i in range(self.N):
            if self.optimizer == "Adam":
                self.EnsembleDQN_optimizer[i] = optim.Adam(self.EnsembleDQN[i].parameters(), lr=self.lr)

            else:
                self.EnsembleDQN_optimizer[i] = optim.RMSprop(self.EnsembleDQN[i].parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

#---------------- CONTINUE -----------------------

    @torch.no_grad()
    def select_action(self, s):
        """Greedy action selection using the active head for a given state.
        s:           np.array with shape (in_channels, height, width)
        active_head: int 

        returns: int for the action
        """
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        if self.mode == "train":
            q = self.DQN(s, self.active_head)

            # greedy
            a = torch.argmax(q).item()
        
        # majority vote
        else:

            # push through all heads
            q = self.DQN(s)

            # get favoured action of each head
            actions = [torch.argmax(head_q).item() for head_q in q]

            # choose majority vote
            actions = Counter(actions)
            a = actions.most_common(1)[0][0]

        return a


    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d, m = batch

        #-------- train BootDQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # current and next Q-values
        Q_main = self.DQN(s)
        Q_s2_tgt = self.target_DQN(s2)
        Q_s2_main = self.DQN(s2)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):
            
            # gather actions
            Q = torch.gather(input=Q_main[k], dim=1, index=a)

            # targets
            with torch.no_grad():

                a2 = torch.argmax(Q_s2_main[k], dim=1).reshape(self.batch_size, 1)
                Q_next = torch.gather(input=Q_s2_tgt[k], dim=1, index=a2)

                y = r + self.gamma * Q_next * (1 - d)

            # calculate (Q - y)**2
            loss_k = self._compute_loss(Q, y, reduction="none")

            # use only relevant samples for given head
            loss_k = loss_k * m[:, k].unsqueeze(1)

            # append loss
            losses.append(torch.sum(loss_k) / torch.sum(m[:, k]))
       
        # compute gradients
        loss = sum(losses)
        loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN.core.parameters():
                p.grad *= 1/float(self.K)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_optimizer.step()
        
        # log critic training
        self.logger.store(Loss=loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        self._target_update()
