import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim

import tud_rl.common.nets as nets
from tud_rl.agents._continuous.MADDPG import MADDPGAgent
from tud_rl.common.configparser import ConfigFile


class MATD3Agent(MADDPGAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name, init_critic=False)

        # attributes and hyperparameters
        self.tgt_noise      = getattr(c.Agent, agent_name)["tgt_noise"]
        self.tgt_noise_clip = getattr(c.Agent, agent_name)["tgt_noise_clip"]
        self.pol_upd_delay  = getattr(c.Agent, agent_name)["pol_upd_delay"]

        # init double critic
        if self.state_type == "feature":
            self.critic = nn.ModuleList().to(self.device)
            for _ in range(self.N_agents):
                self.critic.append(nets.Double_MLP(in_size   = (self.state_shape + self.num_actions) * self.N_agents,
                                                   out_size  = 1,
                                                   net_struc = self.net_struc_critic).to(self.device))

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior critic weights if available
        if self.critic_weights is not None:
            self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

        # redefine target critic
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        # counter for policy update delay
        self.pol_upd_cnt = 0
        
        # freeze target critic nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define critic optimizer
        if self.optimizer == "Adam":
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

    def _compute_target(self, r, s2, d, i):
        with torch.no_grad():

            # we need target actions from all agents
            target_a = torch.zeros((self.batch_size, self.N_agents, self.num_actions), dtype=torch.float32)
            for j in range(self.N_agents):
                s2_j = s2[:, j]
                
                if self.is_continuous:
                    target_a[:, j] = self.target_actor[j](s2_j)
                else:
                    target_a[:, j] = self._onehot(self.target_actor[j](s2_j))

            # target policy smoothing (only in continuous case)
            if self.is_continuous:
                eps = torch.randn_like(target_a) * self.tgt_noise
                eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
                target_a += eps
                target_a = torch.clamp(target_a, -1, 1)

            # next Q-estimate
            s2a2_for_Q = torch.cat([s2.reshape(self.batch_size, -1), target_a.reshape(self.batch_size, -1)], dim=1)
            Q_next1, Q_next2 = self.target_critic[i](s2a2_for_Q)
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r[:, i] + self.gamma * Q_next * (1 - d)
        return y

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        for i in range(self.N_agents):

            # sample batch
            batch = self.replay_buffer.sample()

            # unpack batch
            s, a, r, s2, d = batch
            sa_for_Q = torch.cat([s.reshape(self.batch_size, -1), a.reshape(self.batch_size, -1)], dim=1)

            #-------- train critic --------
            # clear gradients
            self.critic_optimizer.zero_grad(set_to_none=True)
            
            # Q-estimates
            Q1, Q2 = self.critic[i](sa_for_Q)
    
            # targets
            y = self._compute_target(r, s2, d, i)

            # loss
            critic_loss = self._compute_loss(Q1, y) + self._compute_loss(Q2, y)
            
            # compute gradients
            critic_loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.critic[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic[i].parameters(), max_norm=10)

            # perform optimizing step
            self.critic_optimizer.step()
            
            # log critic training
            self.logger.store(**{f"Critic_loss_{i}" : critic_loss.detach().cpu().numpy().item()})
            self.logger.store(**{f"Q_val_{i}" : Q1.detach().mean().cpu().numpy().item()})

            #-------- train actor --------
            if self.pol_upd_cnt % self.pol_upd_delay == 0:

                # freeze critic so no gradient computations are wasted while training actor
                for param in self.critic[i].parameters():
                    param.requires_grad = False

                # clear gradients
                self.actor_optimizer.zero_grad(set_to_none=True)

                # get current actions via actor
                curr_a = self.actor[i](s[:, i])

                # compute loss, which is negative Q-values from critic
                if self.is_continuous:
                    a[:, i] = curr_a
                else:
                    a[:, i] = self._gumbel_softmax(curr_a, hard=True)

                sa_for_Q_new = torch.cat([s.reshape(self.batch_size, -1), a.reshape(self.batch_size, -1)], dim=1)
                actor_loss = -self.critic[i].single_forward(sa_for_Q_new).mean()

                # compute gradients
                actor_loss.backward()

                # gradient scaling and clipping
                if self.grad_rescale:
                    for p in self.actor[i].parameters():
                        p.grad *= 1 / math.sqrt(2)
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.actor[i].parameters(), max_norm=10)
                
                # perform step with optimizer
                self.actor_optimizer.step()

                # unfreeze critic so it can be trained in next iteration
                for param in self.critic[i].parameters():
                    param.requires_grad = True
                
                # log actor training
                self.logger.store(**{f"Actor_loss_{i}" : actor_loss.detach().cpu().numpy().item()})

                #------- Update target networks -------
                self.polyak_update()
        
        self.pol_upd_cnt += 1
