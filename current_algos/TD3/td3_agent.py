import sys

import copy
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from current_algos.TD3.td3_buffer import UniformReplayBuffer
from current_algos.TD3.td3_nets import Actor, Critic, Double_Critic
from common.noise import Gaussian_Noise
from common.normalizer import Action_Normalizer, Input_Normalizer
from common.logging_func import *

class TD3_Agent:
    def __init__(self, 
                 mode,
                 action_dim, 
                 state_dim, 
                 action_high,
                 action_low,
                 actor_weights    = None,
                 critic_weights   = None, 
                 input_norm       = False,
                 input_norm_prior = None,
                 double_critic    = True,
                 tgt_pol_smooth   = True,
                 tgt_noise        = 0.2,
                 tgt_noise_clip   = 0.5,
                 pol_upd_delay    = 2,
                 gamma            = 0.99,
                 n_steps          = 1,
                 tau              = 0.001,
                 lr_actor         = 0.001,
                 lr_critic        = 0.001,
                 l2_reg_actor     = 0.0,
                 l2_reg_critic    = 0.0,
                 buffer_length    = 100000,
                 grad_clip        = False,
                 grad_rescale     = False,
                 act_start_step   = 10000,
                 upd_start_step   = 1000,
                 upd_every        = 1,
                 batch_size       = 32,
                 device           = "cpu"):
        """Initializes agent. Agent can select actions based on his model, memorize and replay to train his model.

        Args:
            mode ([type]): [description]
            action_dim ([type]): [description]
            state_dim ([type]): [description]
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
        
        self.name             = "TD3_Agent" if double_critic == True and tgt_pol_smooth == True and pol_upd_delay != 1 else "DDPG_Agent"
        self.action_dim       = action_dim
        self.state_dim        = state_dim
        self.action_high      = action_high
        self.action_low       = action_low
        self.actor_weights    = actor_weights
        self.critic_weights   = critic_weights 
        self.input_norm       = input_norm
        self.input_norm_prior = input_norm_prior
        self.double_critic    = double_critic
        self.tgt_pol_smooth   = tgt_pol_smooth
        self.tgt_noise        = tgt_noise
        self.tgt_noise_clip   = tgt_noise_clip
        self.pol_upd_delay    = pol_upd_delay
        self.gamma            = gamma
        self.n_steps          = n_steps
        self.tau              = tau
        self.lr_actor         = lr_actor
        self.lr_critic        = lr_critic
        self.l2_reg_actor     = l2_reg_actor
        self.l2_reg_critic    = l2_reg_critic
        self.buffer_length    = buffer_length
        self.grad_clip        = grad_clip
        self.grad_rescale     = grad_rescale
        self.act_start_step   = act_start_step
        self.upd_start_step   = upd_start_step
        self.upd_every        = upd_every
        self.batch_size       = batch_size

        # n_step
        assert n_steps >= 1, "'n_steps' should not be smaller than 1."

        # gpu support
        assert device in ["cpu", "cuda"], "Unknown device."

        if device == "cpu":    
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print("Using GPU support.")
        
        # init logger and save config
        self.logger = EpochLogger(alg_str = self.name)
        self.logger.save_config(locals())
        
        # init replay buffer and noise
        if mode == "train":
            self.replay_buffer = UniformReplayBuffer(action_dim=action_dim, state_dim=state_dim, n_steps=n_steps, gamma=gamma,
                                                     buffer_length=buffer_length, batch_size=batch_size, device=self.device)
            self.noise = Gaussian_Noise(action_dim=action_dim)

        # init input, action normalizer
        if input_norm:
            assert not (mode == "test" and input_norm_prior is None), "Please supply 'input_norm_prior' in test mode with input normalization."
            
            if input_norm_prior is not None:
                with open(input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(state_dim=state_dim, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(state_dim=state_dim, prior=None)
        
        self.act_normalizer = Action_Normalizer(action_high=action_high, action_low=action_low)
        
        # init actor, critic
        self.actor  = Actor(action_dim=action_dim, state_dim=state_dim).to(self.device)
        if self.double_critic:
            self.critic = Double_Critic(action_dim=action_dim, state_dim=state_dim).to(self.device)
        else:
            self.critic = Critic(action_dim=action_dim, state_dim=state_dim).to(self.device)

        print("--------------------------------------------")
        print(f"n_params actor: {self._count_params(self.actor)}, n_params critic: {self._count_params(self.critic)}")
        print("--------------------------------------------")
        
        # load prior weights if available
        if actor_weights is not None and critic_weights is not None:
            self.actor.load_state_dict(torch.load(actor_weights))            
            self.critic.load_state_dict(torch.load(critic_weights))

        # init target nets and counter for polyak-updates
        self.target_actor  = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.pol_upd_cnt = 0
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # define optimizer
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=l2_reg_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=l2_reg_critic)
    
    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        Arg s:   np.array with shape (state_dim,)
        returns: np.array with shape (action_dim,)
        """        
        # reshape obs
        s = torch.tensor(s.astype(np.float32)).view(1, self.state_dim).to(self.device)

        # forward pass
        a = self.actor(s) 
        
        # add noise
        if self.mode == "train":
            a += torch.tensor(self.noise.sample()).to(self.device)
        
        # clip actions in [-1,1]
        a = torch.clamp(a, -1, 1).cpu().numpy().reshape(self.action_dim)
        
        # transform [-1,1] to application scale
        return self.act_normalizer.norm_to_action(a)
    
    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer.
        Note: action is transformed from application scale to [-1,1]."""
        a = self.act_normalizer.action_to_norm(a)
        self.replay_buffer.add(s, a, r, s2, d)

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        #-------- train critic --------
        # clear gradients
        self.critic_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        if self.double_critic:
            Q_v1, Q_v2 = self.critic(s, a)
        else:
            Q_v = self.critic(s, a)
 
        # calculate targets
        with torch.no_grad():
            target_a = self.target_actor(s2)

            # target policy smoothing
            if self.tgt_pol_smooth:
                eps = torch.randn_like(target_a) * self.tgt_noise
                eps = torch.clamp(eps, -self.tgt_noise_clip, self.tgt_noise_clip)
                target_a += eps
                target_a = torch.clamp(target_a, -1, 1)

            # Q-value of next state-action pair
            if self.double_critic:
                target_Q_next1, target_Q_next2 = self.target_critic(s2, target_a)
                target_Q_next = torch.min(target_Q_next1, target_Q_next2)
            else:
                target_Q_next = self.target_critic(s2, target_a)

            # target
            target_Q = r + (self.gamma ** self.n_steps) * target_Q_next * (1 - d)

        # calculate loss
        if self.double_critic:
            critic_loss = F.mse_loss(Q_v1, target_Q) + F.mse_loss(Q_v2, target_Q)
        else:
            critic_loss = F.mse_loss(Q_v, target_Q)
        
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
        if self.double_critic:
            self.logger.store(Q1_val=Q_v1.detach().mean().cpu().numpy().item(), Q2_val=Q_v2.detach().mean().cpu().numpy().item())
        else:
            self.logger.store(Q_val=Q_v.detach().mean().cpu().numpy().item())

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
            if self.double_critic:
                actor_loss = -self.critic.Q1(s, curr_a).mean()
            else:
                actor_loss = -self.critic(s, curr_a).mean()
            
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

        # increase polyak-update cnt
        self.pol_upd_cnt += 1
    
    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""
        for target_p, main_p in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
