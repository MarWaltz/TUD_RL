import scipy.stats
import torch
import torch.nn as nn
from tud_rl.agents._discrete.BootDQN import BootDQNAgent
from tud_rl.common.configparser import ConfigFile


class KEBootDQNAgent(BootDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameter
        self.kernel       = getattr(c.Agent, agent_name)["kernel"]
        self.kernel_param = getattr(c.Agent, agent_name)["kernel_param"]

        # checks
        assert self.kernel in ["test", "gaussian_cdf"], "Unknown kernel."

        # kernel funcs
        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(u, scale=self.kernel_param), dtype=torch.float32)


    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d, m = batch

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # current and next Q-values
        Q_main = self.DQN(s)
        Q_s2_tgt = self.target_DQN(s2)
       
        # stack list into torch.Size([K, batch_size, num_actions])
        Q_s2_tgt_stacked = torch.stack(Q_s2_tgt)

        # compute variances over the K heads, gives torch.Size([batch_size, num_actions])
        Q_s2_var = torch.var(Q_s2_tgt_stacked, dim=0, unbiased=True)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):
            
            # gather actions
            Q = torch.gather(input=Q_main[k], dim=1, index=a)

            # targets
            with torch.no_grad():

                # get easy access to relevant target Q
                Q_tgt = Q_s2_tgt[k].to(self.device)

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
