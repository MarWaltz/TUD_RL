import copy
import math
import warnings
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets

from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import Configfile
from tud_rl.common.logging_func import *
from tud_rl.common.exploration import Gaussian_Noise
from tud_rl.common.normalizer import Action_Normalizer


class DDPGAgent(BaseAgent):
    def __init__(self, c: Configfile, agent_name, logging=True, init_critic=True):
        super().__init__(c, agent_name,logging)

        # attributes and hyperparameters
        self.action_high      = c.action_high
        self.action_low       = c.action_low
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

        # noise
        self.noise = Gaussian_Noise(action_dim = self.num_actions)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer(
                state_type    = self.state_type, 
                state_shape   = self.state_shape, 
                buffer_length = self.buffer_length,
                batch_size    = self.batch_size,
                device        = self.device,
                disc_actions  = False,
                action_dim    = self.num_actions)
        # action normalizer
        self.act_normalizer = Action_Normalizer(action_high = self.action_high, action_low = self.action_low)      

        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.MLP(
                in_size   = self.state_shape,
                out_size  = self.num_actions,
                net_struc = self.net_struc_actor).to(self.device)
            
            if init_critic:
                self.critic = nets.MLP(
                    in_size   = self.state_shape + self.num_actions,
                    out_size  = 1,
                    net_struc = self.net_struc_critic).to(self.device)
        
        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights))            
            
            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights))

        # init target nets
        self.target_actor  = copy.deepcopy(self.actor).to(self.device)
        
        if init_critic:
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_actor.parameters():
            p.requires_grad = False
        
        if init_critic:
            for p in self.target_critic.parameters():
                p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(
                    self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(
                self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(
                    self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)


    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        Arg s:   np.array with shape (state_shape,)
        returns: np.array with shape (num_actions,)
        """        
        # greedy
        a = self._greedy_action(s).to(self.device)
       
        # exploration noise
        if self.mode == "train":
            a += torch.tensor(self.noise.sample()).to(self.device)
        
        # clip actions in [-1,1]
        a = torch.clamp(a, -1, 1).cpu().numpy().reshape(self.num_actions)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)


    def _greedy_action(self, s):
        # reshape obs (namely, to torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward pass
        return self.actor(s)


    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer.
        Note: Action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
        self.replay_buffer.add(s, a, r, s2, d)


    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            target_a = self.target_actor(s2)

            # next Q-estimate
            Q_next = self.target_critic(torch.cat([s2, target_a], dim=1))

            # target
            y = r + self.gamma * Q_next * (1 - d)
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

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # Q-estimates
        Q = self.critic(sa)
 
        # targets
        y = self._compute_target(r, s2, d)

        # loss
        critic_loss = self._compute_loss(Q, y)
        
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
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        #-------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a = self.actor(s)
        
        # compute loss, which is negative Q-values from critic
        actor_loss = -self.critic(torch.cat([s, curr_a], dim=1)).mean()

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


    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""
        for target_p, main_p in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)


class LSTMDDPGAgent(BaseAgent):
    def __init__(self, c: Configfile, agent_name, logging=True, init_critic=True):
        super().__init__(c, agent_name,logging)

        # attributes and hyperparameters
        self.action_high      = c.action_high
        self.action_low       = c.action_low
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic
        self.history_length   = getattr(c.Agent,agent_name)["history_length"]
        self.use_past_actions = getattr(c.Agent,agent_name)["use_past_actions"]

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None or self.net_struc_critic is not None:
            warnings.warn("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        # noise
        self.noise = Gaussian_Noise(action_dim = self.num_actions)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer_LSTM(
                state_type     = self.state_type, 
                state_shape    = self.state_shape, 
                buffer_length  = self.buffer_length,
                batch_size     = self.batch_size,
                device         = self.device,
                disc_actions   = False,
                action_dim     = self.num_actions,
                history_length = self.history_length)
        # action normalizer
        self.act_normalizer = Action_Normalizer(action_high = self.action_high, action_low = self.action_low)      

        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.LSTM_Actor(state_shape = self.state_shape,
                                    action_dim       = self.num_actions,
                                    use_past_actions = self.use_past_actions).to(self.device)
            
            if init_critic:
                self.critic = nets.LSTM_Critic(state_shape = self.state_shape,
                                          action_dim       = self.num_actions,
                                          use_past_actions = self.use_past_actions).to(self.device)

        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights))            
            
            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights))

        # init target nets
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        
        if init_critic:
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_actor.parameters():
            p.requires_grad = False
        
        if init_critic:
            for p in self.target_critic.parameters():
                p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(
                    self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(
                self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(
                    self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)


    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # past actions will be on application scale
        a_hist = self.act_normalizer.action_to_norm(a_hist)

        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(
            s_hist, dtype=torch.float32).view(
                1, self.history_length, self.state_shape).to(self.device)
        a_hist = torch.tensor(
            a_hist, dtype=torch.float32).view(
                1, self.history_length, self.num_actions).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        a, _ = self.actor(s, s_hist, a_hist, hist_len)
        
        # add noise
        if self.mode == "train":
            a += torch.tensor(self.noise.sample()).to(self.device)
        
        # clip actions in [-1,1]
        a = torch.clamp(a, -1, 1).cpu().numpy().reshape(self.num_actions)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)


    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer.
        Note: Action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
        self.replay_buffer.add(s, a, r, s2, d)


    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
 
        with torch.no_grad():
            target_a, _ = self.target_actor(
                s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
                        
            # next Q-estimate
            Q_next = self.target_critic(
                s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2, log_info=False)

            # target
            y = r + self.gamma * Q_next * (1 - d)
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

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # Q-estimates
        Q, critic_net_info = self.critic(
            s=s, a=a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, log_info=True)
 
        # calculate targets
        y = self._compute_target(s2_hist, a2_hist, hist_len2, r, s2, d)

        # calculate loss
        critic_loss = self._compute_loss(Q, y)

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
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())
        
        #-------- train actor --------
        
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()
        
        # get current actions via actor
        curr_a, act_net_info = self.actor(
            s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
        
        # compute loss, which is negative Q-values from critic
        actor_loss = -self.critic(
            s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len, log_info=False).mean()

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

    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""
        for target_p, main_p in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)


class LSTMSACAgent(BaseAgent):
    def __init__(self, c: Configfile, agent_name, logging=True):
        super().__init__(c, agent_name,logging)

        # attributes and hyperparameters
        self.action_high      = c.action_high
        self.action_low       = c.action_low
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

        self.history_length   = getattr(c.Agent, agent_name)["history_length"]
        self.use_past_actions = getattr(c.Agent, agent_name)["use_past_actions"]

        # checks
        assert not (self.mode == "test" and\
            (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

        if self.net_struc_actor is not None or self.net_struc_critic is not None:
            warnings.warn("The net structure cannot be controlled via the config-spec for LSTM-based agents.")

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
            self.replay_buffer = buffer.UniformReplayBuffer_LSTM(
                state_type     = self.state_type, 
                state_shape    = self.state_shape, 
                buffer_length  = self.buffer_length,
                batch_size     = self.batch_size,
                device         = self.device,
                disc_actions   = False,
                action_dim     = self.num_actions,
                history_length = self.history_length)
        # action normalizer
        self.act_normalizer = Action_Normalizer(action_high = self.action_high, action_low = self.action_low)      

        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.LSTM_GaussianActor(
                state_shape = self.state_shape,
                action_dim  = self.num_actions,
                use_past_actions = self.use_past_actions).to(self.device)
            
            self.critic = nets.LSTM_Double_Critic(
                state_shape      = self.state_shape,
                action_dim       = self.num_actions,
                use_past_actions = self.use_past_actions).to(self.device)

        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights))            
            self.critic.load_state_dict(torch.load(self.critic_weights))

        # init target net
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.lr_critic)
        else:
            self.actor_optimizer = optim.RMSprop(
                self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(
                self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def select_action(self, s, s_hist, a_hist, hist_len):
        """Selects action via actor network for a given state. 
        Adds exploration bonus from noise and clips to action scale.
        s:        np.array with shape (state_shape,)
        s_hist:   np.array with shape (history_length, state_shape)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        
        returns: np.array with shape (action_dim,)
        """
        # past actions will be on application scale
        a_hist = self.act_normalizer.action_to_norm(a_hist)

        # reshape arguments and convert to tensors
        s = torch.tensor(s, dtype=torch.float32).view(1, self.state_shape).to(self.device)
        s_hist = torch.tensor(s_hist, 
                              dtype=torch.float32
                              ).view(1, 
                                     self.history_length, 
                                     self.state_shape).to(self.device)
        a_hist = torch.tensor(a_hist, 
                              dtype=torch.float32
                              ).view(1, 
                                     self.history_length, 
                                     self.num_actions).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=False, with_logprob=False)
        else:
            a, _, _ = self.actor(s, s_hist, a_hist, hist_len, deterministic=True, with_logprob=False)
        
        # reshape actions
        a = a.cpu().numpy().reshape(self.num_actions)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)


    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer.
        Note: Action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
        self.replay_buffer.add(s, a, r, s2, d)


    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
 
        with torch.no_grad():

            # target actions come from current policy (no target actor)
            target_a, target_logp_a, _ = self.actor(
                s=s2, s_hist=s2_hist, a_hist=a2_hist, 
                hist_len=hist_len2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(
                s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
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


class LSTMTD3Agent(LSTMDDPGAgent):
    def __init__(self, c: Configfile, agent_name, logging=True):
        super().__init__(c, agent_name, logging, init_critic=False)

        # attributes and hyperparameters
        self.tgt_noise      = getattr(c.Agent,agent_name)["tgt_noise"]
        self.tgt_noise_clip = getattr(c.Agent,agent_name)["tgt_noise_clip"]
        self.pol_upd_delay  = getattr(c.Agent,agent_name)["pol_upd_delay"]

        # init double critic
        if self.state_type == "feature":
            self.critic = nets.LSTM_Double_Critic(state_shape = self.state_shape,
                                             action_dim       = self.num_actions,
                                             use_past_actions = self.use_past_actions).to(self.device)

        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
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


    def _compute_target(self, s2_hist, a2_hist, hist_len2, r, s2, d):
 
        with torch.no_grad():
            target_a, _ = self.target_actor(s=s2, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            
            # target policy smoothing
            eps = torch.randn_like(target_a) * self.tgt_noise
            eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
            target_a += eps
            target_a = torch.clamp(target_a, -1, 1)
            
            # Q-value of next state-action pair
            Q_next1, Q_next2, _ = self.target_critic(s=s2, a=target_a, s_hist=s2_hist, a_hist=a2_hist, hist_len=hist_len2)
            Q_next = torch.min(Q_next1, Q_next2)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y


    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s_hist, a_hist, hist_len, s2_hist, a2_hist, hist_len2, s, a, r, s2, d = batch

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # Q-estimates
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
        if self.pol_upd_cnt % self.pol_upd_delay == 0:

            # freeze critic so no gradient computations are wasted while training actor
            for param in self.critic.parameters():
                param.requires_grad = False
            
            # clear gradients
            self.actor_optimizer.zero_grad()
            
            # get current actions via actor
            curr_a, act_net_info = self.actor(s=s, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len)
            
            # compute loss, which is negative Q-values from critic
            actor_loss = -self.critic.single_forward(s=s, a=curr_a, s_hist=s_hist, a_hist=a_hist, hist_len=hist_len).mean()

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


class SACAgent(BaseAgent):
    def __init__(self, c: Configfile, agent_name, logging=True, init_critic=True):
        super().__init__(c, agent_name,logging)

        # attributes and hyperparameters
        self.action_high      = c.action_high
        self.action_low       = c.action_low
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic

        self.lr_temp     = getattr(c.Agent,agent_name)["lr_temp"]
        self.temp_tuning = getattr(c.Agent,agent_name)["temp_tuning"]
        self.init_temp   = getattr(c.Agent,agent_name)["init_temp"]

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

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
        # action normalizer
        self.act_normalizer = Action_Normalizer(action_high = self.action_high, action_low = self.action_low)      

        # init actor and critic
        if self.state_type == "feature":
            self.actor = nets.GaussianActor(state_shape = self.state_shape,
                                       action_dim  = self.num_actions).to(self.device)
            
            if init_critic:
                self.critic =nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                         out_size  = 1,
                                         net_struc = self.net_struc_critic).to(self.device)
        
        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
            print("--------------------------------------------")

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights))

            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights))

        # init target net
        if init_critic:
            self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        if init_critic:
            for p in self.target_critic.parameters():
                p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(
                self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(
                    self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(
                self.actor.parameters(), 
                lr=self.lr_actor, 
                alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(
                    self.critic.parameters(), 
                    lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)

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
        a = a.cpu().numpy().reshape(self.num_actions)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)


    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer.
        Note: Action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
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


class TD3Agent(DDPGAgent):
    def __init__(self, c: Configfile, agent_name, logging=True):
        super().__init__(c, agent_name, logging, init_critic=False)

        # attributes and hyperparameters
        self.tgt_noise      = getattr(c.Agent,agent_name)["tgt_noise"]
        self.tgt_noise_clip = getattr(c.Agent,agent_name)["tgt_noise_clip"]
        self.pol_upd_delay  = getattr(c.Agent,agent_name)["pol_upd_delay"]

        # init double critic
        if self.state_type == "feature":
            self.critic = nets.Double_MLP(in_size   = self.state_shape + self.num_actions,
                                     out_size  = 1,
                                     net_struc = self.net_struc_critic).to(self.device)
        
        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
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


class TQCAgent(SACAgent):
    def __init__(self, c: Configfile, agent_name, logging=True):
        super().__init__(c, agent_name, logging, init_critic=False)

        # attributes and hyperparameters
        self.top_qs_to_drop = getattr(c.Agent,agent_name)["top_qs_to_drop"]
        self.n_qs           = getattr(c.Agent,agent_name)["n_qs"]
        self.n_critics      = getattr(c.Agent,agent_name)["n_critics"]

        # calculate total number of quantiles in use
        self.total_qs = self.n_critics * self.n_qs

        # checks
        if self.net_struc_critic is not None:
            warnings.warn("The net structure of the Critic-Ensemble "
                          "cannot be controlled via the config-spec for TQC.")

        # init critic
        if self.state_type == "feature":
            self.critic = nets.TQC_Critics(state_shape = self.state_shape,
                                      action_dim  = self.num_actions,
                                      n_quantiles = self.n_qs,
                                      n_critics   = self.n_critics).to(self.device)
        
        # init logger and save config
        if logging:
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  "
                  f"|  n_params_critic: {self._count_params(self.critic)}")
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
