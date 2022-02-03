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
    def __init__(self, c, agent_name, logging=True):
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
            self.critic.load_state_dict(torch.load(self.critic_weights))

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
        if self.optimizer == "Adam":
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        else:
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    
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
            target_Q = r + self.gamma * target_Q_next * (1 - d)

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
