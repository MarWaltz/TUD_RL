import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
from tud_rl.agents.continuous.LSTMDDPG import LSTMDDPGAgent
from tud_rl.common.logging_func import *
from tud_rl.common.nets import LSTM_Double_Critic


class LSTMTD3Agent(LSTMDDPGAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name, logging=False, init_critic=False)

        # attributes and hyperparameters
        self.tgt_noise      = c["agent"][agent_name]["tgt_noise"]
        self.tgt_noise_clip = c["agent"][agent_name]["tgt_noise_clip"]
        self.pol_upd_delay  = c["agent"][agent_name]["pol_upd_delay"]

        # init double critic
        if self.state_type == "feature":
            self.critic = LSTM_Double_Critic(obs_dim          = self.state_shape,
                                             action_dim       = self.num_actions,
                                             use_past_actions = self.use_past_actions).to(self.device)

        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str, info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  |  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior critic weights if available
        if self.critic_weights is not None:
            self.critic.load_state_dict(torch.load(self.critic_weights))

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


    def _compute_target(self, o2_hist, a2_hist, hist_len2, r, o2, d):
 
        with torch.no_grad():
            target_a, _ = self.target_actor(o=o2, o_hist=o2_hist, a_hist=a2_hist, hist_len=hist_len2)
            
            # target policy smoothing
            eps = torch.randn_like(target_a) * self.tgt_noise
            eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
            target_a += eps
            target_a = torch.clamp(target_a, -1, 1)
            
            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(o=o2, a=target_a, o_hist=o2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        o_hist, a_hist, hist_len, o2_hist, a2_hist, hist_len2, o, a, r, o2, d = batch

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # Q-estimates
        Q1, Q2, critic_net_info = self.critic(o=o, a=a, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
 
        # calculate targets
        y = self._compute_target(o2_hist, a2_hist, hist_len2, r, o2, d)

        # calculate loss
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
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item(), **critic_net_info)
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())
        
        #-------- train actor --------
        if self.pol_upd_cnt % self.pol_upd_delay == 0:

            # freeze critic so no gradient computations are wasted while training actor
            for param in self.critic.parameters():
                param.requires_grad = False
            
            # clear gradients
            self.actor_optimizer.zero_grad()
            
            # get current actions via actor
            curr_a, act_net_info = self.actor(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
            
            # compute loss, which is negative Q-values from critic
            actor_loss = -self.critic.single_forward(o=o, a=curr_a, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len).mean()

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
            self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info)
            
            #------- Update target networks -------
            self.polyak_update()
        
        self.pol_upd_cnt += 1
