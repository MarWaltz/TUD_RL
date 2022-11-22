import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
import tud_rl.common.nets as nets
from tud_rl.agents._continuous.DDPG import DDPGAgent
from tud_rl.common.configparser import ConfigFile


class TD3Agent(DDPGAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name, init_critic=False)

        # attributes and hyperparameters
        self.tgt_noise      = getattr(c.Agent, agent_name)["tgt_noise"]
        self.tgt_noise_clip = getattr(c.Agent, agent_name)["tgt_noise_clip"]
        self.pol_upd_delay  = getattr(c.Agent, agent_name)["pol_upd_delay"]

        # init double critic
        if self.state_type == "feature":
            self.critic = nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                          out_size  = 1,
                                          net_struc = self.net_struc_critic).to(self.device)

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

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            target_a = self.target_actor(s2)

            # target policy smoothing
            eps = torch.randn_like(target_a) * self.tgt_noise
            eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
            target_a += eps
            target_a = torch.clamp(target_a, -1, 1)

            # next Q-estimate
            Q_next1, Q_next2 = self.target_critic(torch.cat([s2, target_a], dim=1))
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch
        sa = torch.cat([s, a], dim=1)

        #-------- train critics --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # Q-estimates
        Q1, Q2 = self.critic(sa)
 
        # targets
        y = self._compute_target(r, s2, d)

        # loss
        critic_loss = self._compute_loss(Q1, y) + self._compute_loss(Q2, y)
        
        # compute gradients
        critic_loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.critic.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        
        # perform optimizing step
        self.critic_optimizer.step()
        
        # log critic training
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())

        #-------- train actor --------
        if self.pol_upd_cnt % self.pol_upd_delay == 0:

            # freeze critic so no gradient computations are wasted while training actor
            for param in self.critic.parameters():
                param.requires_grad = False
            
            # clear gradients
            self.actor_optimizer.zero_grad()

            # get current actions via actor
            curr_a = self.actor(s)
            
            # compute loss, which is negative Q-values from critic
            actor_loss = -self.critic.single_forward(torch.cat([s, curr_a], dim=1)).mean()

            # compute gradients
            actor_loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.actor.parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            
            # perform step with optimizer
            self.actor_optimizer.step()

            # unfreeze critic so it can be trained in next iteration
            for param in self.critic.parameters():
                param.requires_grad = True
            
            # log actor training
            self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item())

            #------- Update target networks -------
            self.polyak_update()
        
        self.pol_upd_cnt += 1
