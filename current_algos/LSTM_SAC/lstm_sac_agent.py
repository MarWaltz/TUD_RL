import sys

import copy
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from current_algos.LSTM_SAC.lstm_sac_buffer import UniformReplayBuffer
from current_algos.LSTM_SAC.lstm_sac_nets import LSTM_GaussianActor, LSTM_Double_Critic
from common.normalizer import Action_Normalizer, Input_Normalizer
from common.logging_func import *

class LSTM_SAC_Agent:
    def __init__(self, 
                 mode,
                 action_dim,  
                 action_high,
                 action_low,
                 obs_dim,
                 actor_weights,
                 critic_weights, 
                 input_norm,
                 input_norm_prior,
                 gamma,
                 tau,
                 net_struc_actor,
                 net_struc_critic,
                 lr_actor,
                 lr_critic,
                 lr_temperature,
                 buffer_length,
                 grad_clip,
                 grad_rescale,
                 act_start_step,
                 upd_start_step,
                 upd_every,
                 batch_size,
                 history_length,
                 use_past_actions,
                 temp_tuning,
                 temperature,
                 device,
                 seed):
        """Initializes agent. Agent can select actions based on his model, memorize and replay to train his model.

        Args:
            mode ([type]): [description]
            action_dim ([type]): [description]
            obs_dim ([type]): [description]
            action_high ([type]): [description]
            action_low ([type]): [description]
            actor_weights ([type], optional): [description]. Defaults to None.
            critic_weights ([type], optional): [description]. Defaults to None.
            input_norm (bool, optional): [description]. Defaults to False.
            input_norm_prior ([type], optional): [description]. Defaults to None.
            gamma (float, optional): [description]. Defaults to 0.99.
            tau (float, optional): [description]. Defaults to 0.005.
            lr_actor (float, optional): [description]. Defaults to 0.001.
            lr_critic (float, optional): [description]. Defaults to 0.001.
            buffer_length (int, optional): [description]. Defaults to 1000000.
            grad_clip (bool, optional): [description]. Defaults to False.
            grad_rescale (bool, optional): [description]. Defaults to False.
            act_start_step (int, optional): Number of steps with random actions before using own decisions. Defaults to 10000.
            upd_start_step (int, optional): Steps to perform in environment before starting updates. Defaults to 1000.
            upd_every (int, optional): Frequency of performing updates. However, ratio between environment and gradient steps is always 1.
            batch_size (int, optional): [description]. Defaults to 100.
            device (str, optional): [description]. Defaults to "cpu".
        """

        # store attributes and hyperparameters
        assert mode in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."
        assert not (mode == "test" and (actor_weights is None or critic_weights is None)), "Need prior weights in test mode."
        self.mode = mode
        
        self.name             = "lstm_sac_agent"
        self.action_dim       = action_dim
        self.action_high      = action_high
        self.action_low       = action_low
        self.obs_dim          = obs_dim
        self.actor_weights    = actor_weights
        self.critic_weights   = critic_weights 
        self.input_norm       = input_norm
        self.input_norm_prior = input_norm_prior
        self.gamma            = gamma
        self.tau              = tau
        self.net_struc_actor  = net_struc_actor,
        self.net_struc_critic = net_struc_critic,
        self.lr_actor         = lr_actor
        self.lr_critic        = lr_critic
        self.lr_temperature   = lr_temperature
        self.buffer_length    = buffer_length
        self.grad_clip        = grad_clip
        self.grad_rescale     = grad_rescale
        self.act_start_step   = act_start_step
        self.upd_start_step   = upd_start_step
        self.upd_every        = upd_every
        self.batch_size       = batch_size
        self.history_length   = history_length
        self.use_past_actions = use_past_actions

        # history_length
        assert history_length >= 1, "'history_length' should not be smaller than 1."

        # gpu support
        assert device in ["cpu", "cuda"], "Unknown device."

        if device == "cpu":    
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print("Using GPU support.")
        
        # dynamic or static temperature
        self.temp_tuning = temp_tuning

        if self.temp_tuning:

            # define target entropy
            self.target_entropy = -action_dim

            # optimize log(temperature) instead of temperature
            self.log_temperature = torch.zeros(1, requires_grad=True, device=self.device)

            # define temperature optimizer
            self.temp_optimizer = optim.Adam([self.log_temperature], lr=self.lr_temperature)

        else:
            self.temperature = temperature
        
        # init logger and save config
        self.logger = EpochLogger(alg_str = self.name)
        self.logger.save_config(locals())
        
        # init replay buffer
        if mode == "train":
            self.replay_buffer = UniformReplayBuffer(action_dim=action_dim, obs_dim=obs_dim, gamma=gamma, history_length=history_length,
                                                     buffer_length=buffer_length, batch_size=batch_size, device=self.device)

        # init input, action normalizer
        if input_norm:
            assert not (mode == "test" and input_norm_prior is None), "Please supply 'input_norm_prior' in test mode with input normalization."
            
            if input_norm_prior is not None:
                with open(input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(obs_dim=obs_dim, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(obs_dim=obs_dim, prior=None)
        
        self.act_normalizer = Action_Normalizer(action_high=action_high, action_low=action_low)
        
        # init actor, critic
        self.actor = LSTM_GaussianActor(action_dim=action_dim, obs_dim=obs_dim, use_past_actions=use_past_actions, 
                                        net_struc_actor=net_struc_actor).to(self.device)
        self.critic = LSTM_Double_Critic(action_dim=action_dim, obs_dim=obs_dim, use_past_actions=use_past_actions,
                                         net_struc_critic=net_struc_critic).to(self.device)

        print("--------------------------------------------")
        print(f"n_params actor: {self._count_params(self.actor)}, n_params critic: {self._count_params(self.critic)}")
        print("--------------------------------------------")
        
        # load prior weights if available
        if actor_weights is not None and critic_weights is not None:
            self.actor.load_state_dict(torch.load(actor_weights))
            self.critic.load_state_dict(torch.load(critic_weights))

        # init target critic
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    @torch.no_grad()
    def select_action(self, o, o_hist, a_hist, hist_len):
        """Selects action via actor network for a given state.
        o:        np.array with shape (obs_dim,)
        o_hist:   np.array with shape (history_length, obs_dim)
        a_hist:   np.array with shape (history_length, action_dim)
        hist_len: int
        returns: np.array with shape (action_dim,)
        """        
        # reshape arguments and convert to tensors
        o = torch.tensor(o.astype(np.float32)).view(1, self.obs_dim).to(self.device)
        o_hist = torch.tensor(o_hist.astype(np.float32)).view(1, self.history_length, self.obs_dim).to(self.device)
        a_hist = torch.tensor(a_hist.astype(np.float32)).view(1, self.history_length, self.action_dim).to(self.device)
        hist_len = torch.tensor(hist_len).to(self.device)

        # forward pass
        if self.mode == "train":
            a, _, _ = self.actor(o, o_hist, a_hist, hist_len, deterministic=False, with_logprob=False)
        else:
            a, _, _ = self.actor(o, o_hist, a_hist, hist_len, deterministic=True, with_logprob=False)
        
        # reshape actions
        a = a.cpu().numpy().reshape(self.action_dim)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)
    
    def memorize(self, o, a, r, o2, d):
        """Stores current transition in replay buffer.
        Note: action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
        self.replay_buffer.add(o, a, r, o2, d)

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        o_hist, a_hist, hist_len, o2_hist, a2_hist, hist_len2, o, a, r, o2, d = batch

        # get current temperature
        if self.temp_tuning:
            self.temperature = torch.exp(self.log_temperature).detach()

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        Q_v1, Q_v2, critic_net_info = self.critic(o=o, a=a, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
 
        # calculate targets
        with torch.no_grad():

            # target actions come from current policy (no target actor)
            target_a, target_logp_a, _ = self.actor(o=o2, o_hist=o2_hist, a_hist=a2_hist, hist_len=hist_len2, deterministic=False, with_logprob=True)

            # Q-value of next state-action pair
            target_Q_next1, target_Q_next2, _ = self.target_critic(o=o2, a=target_a, o_hist=o2_hist, a_hist=a2_hist, hist_len=hist_len2)
            target_Q_next = torch.min(target_Q_next1, target_Q_next2)

            # target
            target_Q = r + self.gamma * (1 - d) * (target_Q_next - self.temperature * target_logp_a)

        # calculate loss
        critic_loss = F.mse_loss(Q_v1, target_Q) + F.mse_loss(Q_v2, target_Q)
 
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
        self.logger.store(Q1_val=Q_v1.detach().mean().cpu().numpy().item(), Q2_val=Q_v2.detach().mean().cpu().numpy().item())
        
        #-------- train actor --------
        # freeze critic so no gradient computations are wasted while training actor
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # clear gradients
        self.actor_optimizer.zero_grad()

        # get current actions via actor
        curr_a, curr_a_logprob, act_net_info = self.actor(o=o, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len, deterministic=False, with_logprob=True)
        
        # compute Q1, Q2 values for current state and actor's actions
        Q1_val_curr_a, Q2_val_curr_a, _ = self.critic(o=o, a=curr_a, o_hist=o_hist, a_hist=a_hist, hist_len=hist_len)
        Q_val_curr_a = torch.min(Q1_val_curr_a, Q2_val_curr_a)

        # compute policy loss (which is based on min Q1, Q2 instead of just Q1 as in TD3, plus consider entropy regularization)
        actor_loss = (self.temperature * curr_a_logprob - Q_val_curr_a).mean()
        
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
