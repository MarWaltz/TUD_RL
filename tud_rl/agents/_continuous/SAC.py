import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile


class SACAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name, init_critic=True):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic

        self.lr_temp     = getattr(c.Agent, agent_name)["lr_temp"]
        self.temp_tuning = getattr(c.Agent, agent_name)["temp_tuning"]
        self.init_temp   = getattr(c.Agent, agent_name)["init_temp"]

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise NotImplementedError("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None:
            warnings.warn("The net structure of the Gaussian actor cannot be controlled via the config-spec for SAC.")

        # dynamic or static temperature
        if self.temp_tuning:

            # define target entropy
            self.target_entropy = -self.num_actions

            # optimize log(temperature) instead of temperature
            self.log_temperature = torch.zeros(1, requires_grad=True, device=self.device)

            # define temperature optimizer
            self.temp_optimizer = optim.Adam([self.log_temperature], lr=self.lr_temp)

        else:
            self.temperature = self.init_temp

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer(state_type    = self.state_type, 
                                                            state_shape   = self.state_shape, 
                                                            buffer_length = self.buffer_length,
                                                            batch_size    = self.batch_size,
                                                            device        = self.device,
                                                            disc_actions  = False,
                                                            action_dim    = self.num_actions)
        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.GaussianActor(state_shape = self.state_shape,
                                            action_dim  = self.num_actions).to(self.device)
            
            if init_critic:
                self.critic = nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                              out_size  = 1,
                                              net_struc = self.net_struc_critic).to(self.device)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))

            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

        # init target net
        if init_critic:
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        if init_critic:
            for p in self.target_critic.parameters():
                p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state.
        Arg s:   np.array with shape (state_shape,)
        returns: np.array with shape (action_dim,)
        """        
        # reshape obs
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _ = self.actor(s, deterministic=False, with_logprob=False)
        else:
            a, _ = self.actor(s, deterministic=True, with_logprob=False)
        
        # reshape actions
        return a.cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, target_logp_a = self.actor(s2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            Q_next1, Q_next2 = self.target_critic(torch.cat([s2, target_a], dim=1))
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r + self.gamma * (1 - d) * (Q_next - self.temperature * target_logp_a)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch
        sa = torch.cat([s, a], dim=1)

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        Q1, Q2 = self.critic(sa)
 
        # calculate targets
        y = self._compute_target(r, s2, d)

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
        self.logger.store(Critic_loss=critic_loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q1.detach().mean().cpu().numpy().item())
        
        #-------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob = self.actor(s, deterministic=False, with_logprob=True)
        
        # compute Q1, Q2 values for current state and actor's actions
        Q1_curr_a, Q2_curr_a = self.critic(torch.cat([s, curr_a], dim=1))
        Q_curr_a = torch.min(Q1_curr_a, Q2_curr_a)

        # compute policy loss (which is based on min Q1, Q2 instead of just Q1 as in TD3, plus consider entropy regularization)
        actor_loss = (self.temperature * curr_a_logprob - Q_curr_a).mean()
        
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
