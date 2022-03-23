import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tud_rl.agents.BaseAgent import BaseAgent
from tud_rl.common.buffer import UniformReplayBuffer
from tud_rl.common.logging_func import *
from tud_rl.common.nets import MLP
from tud_rl.common.exploration import Gaussian_Noise
from tud_rl.common.normalizer import Action_Normalizer


class DDPGAgent(BaseAgent):
    def __init__(self, c, agent_name, logging=True, init_critic=True):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.action_high      = c["action_high"]
        self.action_low       = c["action_low"]
        self.lr_actor         = c["lr_actor"]
        self.lr_critic        = c["lr_critic"]
        self.tau              = c["tau"]
        self.actor_weights    = c["actor_weights"]
        self.critic_weights   = c["critic_weights"]
        self.net_struc_actor  = c["net_struc_actor"]
        self.net_struc_critic = c["net_struc_critic"]

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise Exception("Currently, image input is not supported for continuous action spaces.")

        # noise
        self.noise = Gaussian_Noise(action_dim = self.num_actions)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = UniformReplayBuffer(state_type    = self.state_type, 
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
            self.actor = MLP(in_size   = self.state_shape,
                            out_size  = self.num_actions,
                            net_struc = self.net_struc_actor).to(self.device)
            
            if init_critic:
                self.critic = MLP(in_size   = self.state_shape + self.num_actions,
                                  out_size  = 1,
                                  net_struc = self.net_struc_critic).to(self.device)
        
        # init logger and save config
        if logging:
            self.logger = EpochLogger(alg_str = self.name, env_str = self.env_str, info = self.info)
            self.logger.save_config({"agent_name" : self.name, **c})
            
            print("--------------------------------------------")
            print(f"n_params_actor: {self._count_params(self.actor)}  |  n_params_critic: {self._count_params(self.critic)}")
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
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            if init_critic:
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, alpha=0.95, centered=True, eps=0.01)
            if init_critic:
                self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, alpha=0.95, centered=True, eps=0.01)


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
