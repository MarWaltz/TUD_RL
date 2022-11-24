import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tud_rl.common.buffer as buffer
import tud_rl.common.nets as nets
from tud_rl.agents.base import BaseAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.exploration import Gaussian_Noise


class MADDPGAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name, init_critic=True):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.N_agents         = getattr(c.Env, "env_kwargs")["N_agents"]
        self.lr_actor         = c.lr_actor
        self.lr_critic        = c.lr_critic
        self.tau              = c.tau
        self.actor_weights    = c.actor_weights
        self.critic_weights   = c.critic_weights
        self.net_struc_actor  = c.net_struc_actor
        self.net_struc_critic = c.net_struc_critic
        self.is_multi         = True

        # checks
        assert not (self.mode == "test" and (self.actor_weights is None or self.critic_weights is None)), "Need prior weights in test mode."

        if self.state_type == "image":
            raise NotImplementedError("Currently, image input is not supported for MADDPG.")

        # noise
        self.noise = Gaussian_Noise(action_dim = self.num_actions)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.MultiAgentUniformReplayBuffer(N_agents      = self.N_agents,
                                                                      state_type    = self.state_type, 
                                                                      state_shape   = self.state_shape, 
                                                                      buffer_length = self.buffer_length,
                                                                      batch_size    = self.batch_size,
                                                                      device        = self.device,
                                                                      action_dim    = self.num_actions)
        # init N actors and N critics
        if self.state_type == "feature":

            self.actor  = nn.ModuleList().to(self.device)
            for _ in range(self.N_agents):
                self.actor.append(nets.MLP(in_size   = self.state_shape,
                                           out_size  = self.num_actions,
                                           net_struc = self.net_struc_actor).to(self.device))
            if init_critic:
                self.critic = nn.ModuleList().to(self.device)
                for _ in range(self.N_agents):
                    self.critic.append(nets.MLP(in_size   = (self.state_shape + self.num_actions) * self.N_agents,
                                                out_size  = 1,
                                                net_struc = self.net_struc_critic).to(self.device))

        # number of parameters for actor and critic
        if init_critic:
            self.n_params = self._count_params(self.actor), self._count_params(self.critic)

        # load prior weights if available
        if self.actor_weights is not None and self.critic_weights is not None:
            self.actor.load_state_dict(torch.load(self.actor_weights, map_location=self.device))            
            
            if init_critic:
                self.critic.load_state_dict(torch.load(self.critic_weights, map_location=self.device))

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
        Arg s:   np.array with shape (N_agents, state_shape)
        returns: np.array with shape (N_agents, num_actions)
        """        
        # greedy
        a = self._greedy_action(s).to(self.device)
       
        # exploration noise
        if self.mode == "train":
            for i in range(self.N_agents):
                a[i] += torch.tensor(self.noise.sample().flatten()).to(self.device)
        
        # clip actions in [-1,1]
        return torch.clamp(a, -1, 1).cpu().numpy()

    def _greedy_action(self, s):
        a = torch.zeros((self.N_agents, self.num_actions), dtype=torch.float32).to(self.device)

        for i in range(self.N_agents):

            # reshape obs (namely, to torch.Size([1, state_shape]))
            s_i = torch.tensor(s[i], dtype=torch.float32).unsqueeze(0).to(self.device)

            # forward pass
            a[i] = self.actor[i](s_i)

        return a

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def _compute_target(self, r, s2, d, i):
        with torch.no_grad():

            # we need target actions from all agents
            target_a = torch.zeros((self.batch_size, self.N_agents, self.num_actions), dtype=torch.float32)
            for j in range(self.N_agents):
                s2_j = s2[:, j]
                target_a[:, j] = self._tar_act_transform(self.target_actor[j](s2_j))

            # next Q-estimate
            s2a2_for_Q = torch.cat([s2.reshape(self.batch_size, -1), target_a.reshape(self.batch_size, -1)], dim=1)
            Q_next = self.target_critic[i](s2a2_for_Q)

            # target
            y = r[:, i] + self.gamma * Q_next * (1 - d)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def _onehot_to_int(self, a):
        """Transform one-hot encoded vector to integers.
        Args:
            a: np.array([N_agents, action_dim])
        Returns:
            np.array(N_agents,)
        Example input:
            [[0., 0., 0., 1., 0.],
             [0., 0., 1., 0., 0.],
             [1., 0., 0., 0., 0.]]
        Example output:
            [3, 2, 0]"""
        return np.where(a==1)[1]

    def _int_to_onehot(self, a):
        """Transform to integers to one-hot encoded vectors. 
        Args:
            a: np.array(N_agents,)
        Returns:
            np.array([N_agents, action_dim])
        Example input:
            [3, 2, 0]
        Example output:
            [[0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0]]"""
        out = np.zeros((self.N_agents, self.num_actions), dtype=np.int64)
        for i, a_i in enumerate(a):
            out[i, a_i] = 1
        return out

    def _onehot(self, x):
        """Transform non-normalized actions to one-hot-encoded form.
        Args:
            x: [batch_size, num_actions]
        Returns:
               [batch_size, num_actions]"""
        return (x == x.max(-1, keepdim=True)[0]).float()

    def _gumbel_softmax(self, x, temp=1.0, hard=False, eps=1e-20):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            x:     [batch_size, num_actions], non-normalized actions
            temp:  non-negative scalar
            hard:  if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class], sample from the Gumbel-Softmax distribution
        
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes."""

        # sample from Gumbel (0,1)
        u = torch.rand_like(x, requires_grad=False).to(self.device)
        g = -torch.log(-torch.log(u + eps) + eps)

        # softmax
        y = F.softmax((x+g)/temp, dim=1)

        if hard:
            y_hard = self._onehot(y)
            y = (y_hard - y).detach() + y
        return y

    def _cur_act_transform(self, curr_a):
        """Transformation for the current actions during the actor training. Required for the discrete MADDPG class."""
        return curr_a

    def _tar_act_transform(self, tar_a):
        """Transformation for the target actions during the critic training. Required for the discrete MADDPG class."""
        return tar_a

    def train(self):
        """Samples from replay_buffer, updates actor, critic and their target networks."""
        for i in range(self.N_agents):

            # sample batch
            batch = self.replay_buffer.sample()

            # unpack batch
            s, a, r, s2, d = batch
            sa_for_Q = torch.cat([s.reshape(self.batch_size, -1), a.reshape(self.batch_size, -1)], dim=1)

            #-------- train critic --------
            # clear gradients
            self.critic_optimizer.zero_grad(set_to_none=True)
            
            # Q-estimates
            Q = self.critic[i](sa_for_Q)
    
            # targets
            y = self._compute_target(r, s2, d, i)

            # loss
            critic_loss = self._compute_loss(Q, y)
            
            # compute gradients
            critic_loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.critic[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.critic[i].parameters(), max_norm=10)

            # perform optimizing step
            self.critic_optimizer.step()
            
            # log critic training
            self.logger.store(**{f"Critic_loss_{i}" : critic_loss.detach().cpu().numpy().item()})
            self.logger.store(**{f"Q_val_{i}" : Q.detach().mean().cpu().numpy().item()})

            #-------- train actor --------
            # freeze critic so no gradient computations are wasted while training actor
            for param in self.critic[i].parameters():
                param.requires_grad = False

            # clear gradients
            self.actor_optimizer.zero_grad(set_to_none=True)

            # get current actions via actor
            curr_a = self.actor[i](s[:, i])

            # possibly transform current actions (used in discrete case)
            curr_a = self._cur_act_transform(curr_a)
            
            # compute loss, which is negative Q-values from critic
            a[:, i] = curr_a
            sa_for_Q_new = torch.cat([s.reshape(self.batch_size, -1), a.reshape(self.batch_size, -1)], dim=1)
            actor_loss = -self.critic[i](sa_for_Q_new).mean()

            # compute gradients
            actor_loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.actor[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.actor[i].parameters(), max_norm=10)
            
            # perform step with optimizer
            self.actor_optimizer.step()

            # unfreeze critic so it can be trained in next iteration
            for param in self.critic[i].parameters():
                param.requires_grad = True
            
            # log actor training
            self.logger.store(**{f"Actor_loss_{i}" : actor_loss.detach().cpu().numpy().item()})

        #------- Update target networks -------
        self.polyak_update()


    @torch.no_grad()
    def polyak_update(self):
        """Soft update of target network weights."""
        for target_p, main_p in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
        
        for target_p, main_p in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * main_p.data + (1-self.tau) * target_p.data)
