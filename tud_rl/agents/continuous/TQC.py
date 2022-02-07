import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from tud_rl.agents.continuous.SAC import SACAgent
from tud_rl.common.logging_func import *
from tud_rl.common.nets import TQC_Critics


class TQCAgent(SACAgent):
    def __init__(self, c, agent_name, logging=True):
        super().__init__(c, agent_name, logging=False, init_critic=False)

        # attributes and hyperparameters
        self.top_qs_to_drop = c["agent"][agent_name]["top_qs_to_drop"]
        self.n_qs           = c["agent"][agent_name]["n_qs"]
        self.n_critics      = c["agent"][agent_name]["n_critics"]

        # calculate total number of quantiles in use
        self.total_qs = self.n_critics * self.n_qs

        # checks
        if self.net_struc_critic is not None:
            warnings.warn("The net structure of the Critic-Ensemble cannot be controlled via the config-spec for TQC.")

        # init critic
        if self.state_type == "feature":
            self.critic = TQC_Critics(state_shape = self.state_shape,
                                      action_dim  = self.num_actions,
                                      n_quantiles = self.n_qs,
                                      n_critics   = self.n_critics).to(self.device)
        
        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str, info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  |  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior weights if available
        if self.critic_weights is not None:
            self.critic.load_state_dict(torch.load(self.critic_weights))

        # init target net
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)


    def _compute_target(self, r, s2, d):
        with torch.no_grad():

            # target actions come from current policy (no target actor)
            next_new_action, next_log_pi = self.actor(s2, deterministic=False, with_logprob=True)

            # compute the quantiles - Eq.(10) in Kuznetsov et al. 2020
            next_z = self.target_critic(s2, next_new_action)

            # sort in ascending order
            sorted_z, _ = torch.sort(next_z.reshape(self.batch_size, -1))

            # cut the top n quantiles - Eq.(11)
            sorted_z_part = sorted_z[:,:self.total_qs - self.top_qs_to_drop]

            # targets from the target distribution - Eq.(12)
            y = r + (1-d) * self.gamma * (sorted_z_part - self.temperature * next_log_pi)
        return y


    def _quantile_huber_loss(self, quantiles, y):
        """Compute the quantile Huber loss to approximate the 1-Wasserstein distance between quantiles."""

        pairwise_delta = y[:,None,None,:] - quantiles[:,:,:,None] # Reshape to
        abs_pairwise_delta = torch.abs(pairwise_delta)

        # huber loss as in Dabney, 2018
        huber_loss = torch.where(abs_pairwise_delta >1,
                abs_pairwise_delta - 0.5,
                pairwise_delta * 0.5 ** 2)

        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device=self.device).float() / n_quantiles + 1/2 / n_quantiles
        loss = (torch.abs(tau[None,None,:,None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss


    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated quantiles 
        # shape: torch.Size([batch_size, n_critics, n_quantiles])
        current_z = self.critic(s, a)
 
        # calculate targets
        y = self._compute_target(r, s2, d)

        # calculate loss
        critic_loss = self._quantile_huber_loss(current_z, y)
 
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
        self.logger.store(Q_val=current_z.detach().mean().cpu().numpy().item())

        #-------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob = self.actor(s, deterministic=False, with_logprob=True)
        
        # evaluate via critics
        current_mean_z = self.critic(s, curr_a).mean(2).mean(1, keepdim=True)

        # consider entropy regularization in loss
        actor_loss = (self.temperature * curr_a_logprob - current_mean_z).mean()
        
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

        #------- update temperature --------
        if self.temp_tuning:

            # clear gradients
            self.temp_optimizer.zero_grad()

            # calculate loss
            temperature_loss = -self.log_temperature * (curr_a_logprob + self.target_entropy).detach().mean()

            # compute gradients
            temperature_loss.backward()

            # perform optimizer step
            self.temp_optimizer.step()

        #------- Update target network -------
        self.polyak_update()

    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""

        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
