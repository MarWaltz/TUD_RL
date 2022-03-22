import math

import scipy.stats
import torch
import torch.nn as nn
from tud_rl.agents.discrete.EnsembleDQN import EnsembleDQNAgent
from tud_rl.common.logging_func import *


class KEEnsembleDQNAgent(EnsembleDQNAgent):
    """This agent performs action selection like the EnsembleDQN (epsilon-greedy over average of ensemble). 
    Only the target computation differs."""

    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True)

        # attributes and hyperparameter
        self.kernel       = c["agent"][agent_name]["kernel"]
        self.kernel_param = c["agent"][agent_name]["kernel_param"]

        # checks
        assert self.kernel in ["test", "gaussian_cdf"], "Unknown kernel."

        # kernel funcs
        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(u, scale=self.kernel_param), dtype=torch.float32)


    def _compute_target(self, r, s2, d, i):
        with torch.no_grad():

            # forward through target ensemble
            Q_s2_tgt = [net(s2).to(self.device) for net in self.target_EnsembleDQN]
            Q_s2_tgt_stacked = torch.stack(Q_s2_tgt).to(self.device)    # torch.Size([N, batch_size, num_actions])
            
            # compute variances over the ensemble, gives torch.Size([batch_size, num_actions])
            Q_s2_var = torch.var(Q_s2_tgt_stacked, dim=0, unbiased=True)

            # select ensemble member that is trained
            Q_tgt = Q_s2_tgt[i]

            # get values and action indices for ME
            ME_values, ME_a_indices = torch.max(Q_tgt, dim=1)

            # reshape indices
            ME_a_indices = ME_a_indices.reshape(self.batch_size, 1)

            # get variance of ME
            ME_var = torch.gather(Q_s2_var, dim=1, index=ME_a_indices).reshape(self.batch_size)

            # compute weights
            w = torch.zeros((self.batch_size, self.num_actions)).to(self.device)

            for a_idx in range(self.num_actions):
                u = (Q_tgt[:, a_idx] - ME_values) / torch.sqrt(Q_s2_var[:, a_idx] + ME_var)
                w[:, a_idx] = self.g(u)

            # compute weighted mean
            Q_next = torch.sum(Q_tgt * w, dim=1) / torch.sum(w, dim=1)
            Q_next = Q_next.reshape(self.batch_size, 1)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def train(self):
        """Samples from replay_buffer, updates critic and the target networks.""" 
       
        #-------- train KEEnsembleDQN --------
        for _ in range(self.N_to_update):
            
            # ensemble member to update
            i = np.random.choice(self.N)

            # sample batch
            batch = self.replay_buffer.sample()
        
            # unpack batch
            s, a, r, s2, d = batch

            # clear gradients
            self.EnsembleDQN_optimizer[i].zero_grad()
            
            # Q estimates
            Q = self.EnsembleDQN[i](s)
            Q = torch.gather(input=Q, dim=1, index=a)
 
            # targets
            y = self._compute_target(r, s2, d, i)

            # loss
            loss = self._compute_loss(Q=Q, y=y)
            
            # compute gradients
            loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.EnsembleDQN[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.EnsembleDQN[i].parameters(), max_norm=10)
            
            # perform optimizing step
            self.EnsembleDQN_optimizer[i].step()
            
            # log critic training
            self.logger.store(Loss=loss.detach().cpu().numpy().item())
            self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        self._target_update()
