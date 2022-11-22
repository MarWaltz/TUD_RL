import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl import logger
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile


class LSTMSACAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name):
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

        self.needs_history    = True
        self.history_length   = getattr(c.Agent, agent_name)["history_length"]
        self.use_past_actions = getattr(c.Agent, agent_name)["use_past_actions"]

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None or self.net_struc_critic is not None:
            logger.warning("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

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
            self.replay_buffer = buffer.UniformReplayBuffer_LSTM(state_type     = self.state_type, 
                                                                 state_shape    = self.state_shape, 
                                                                 buffer_length  = self.buffer_length,
                                                                 batch_size     = self.batch_size,
                                                                 device         = self.device,
                                                                 disc_actions   = False,
                                                                 action_dim     = self.num_actions,
                                                                 history_length = self.history_length)
        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.LSTM_GaussianActor(state_shape = self.state_shape,
                                                 action_dim  = self.num_actions,
                                                 use_past_actions = self.use_past_actions).to(self.device)
            
            self.critic = nets.LSTM_Double_Critic(state_shape      = self.state_shape,
                                                  action_dim       = self.num_actions,
                                                  use_past_actions = self.use_past_actions).to(self.device)

        # number of parameters for actor and critic
        self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))            
            self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

        # init target net
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(s_hist, dtype=torch.float32).view(1, self.history_length, self.state_shape).to(self.device)
        a_hist = torch.tensor(a_hist, dtype=torch.float32).view(1, self.history_length, self.num_actions).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=False, with_logprob=False)
        else:
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=True, with_logprob=False)
        
        # reshape actions
        return a.cpu().numpy().reshape(self.num_actions)

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
        with torch.no_grad():
            # target actions come from current policy (no target actor)
            target_a, target_logp_a, _ = self.actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
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
        s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d = batch

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        Q1, Q2, critic_net_info = self.critic(s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
 
        # calculate targets
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

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
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob, act_net_info = self.actor(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, deterministic=False, with_logprob=True)

        # compute Q1, Q2 values for current state and actor's actions
        Q1_curr_a, Q2_curr_a, _ = self.critic(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
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
        self.logger.store(Actor_loss=actor_loss.detach().cpu().numpy().item(), **act_net_info)

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

        #------- Update target networks -------
        self.polyak_update()

    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""

        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
