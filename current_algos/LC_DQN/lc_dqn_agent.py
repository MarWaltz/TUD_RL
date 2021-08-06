import copy
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from current_algos.LC_DQN.lc_dqn_buffer import UniformReplayBuffer
from current_algos.LC_DQN.lc_dqn_nets import DQN
from current_algos.common.normalizer import Input_Normalizer
from current_algos.common.logging_func import *

class LC_DQN_Agent:
    def __init__(self, 
                 mode,
                 num_actions, 
                 state_dim,
                 dqn_weights      = None, 
                 input_norm       = False,
                 input_norm_prior = None,
                 N                = 4,
                 act_softmax      = True,
                 gamma            = 0.99,
                 eps_decay        = 0.99995,
                 eps_final        = 0.001,
                 n_steps          = 1,
                 tgt_update_freq  = 256,
                 lr               = 0.001,
                 l2_reg           = 0.0,
                 buffer_length    = 10000,
                 grad_clip        = False,
                 grad_rescale     = False,
                 act_start_step   = 10000,
                 upd_start_step   = 1000,
                 upd_every        = 1,
                 batch_size       = 128,
                 device           = "cpu"):
        """Initializes agent. Agent can select actions based on his model, memorize and replay to train his model.

        Args:
            mode ([type]): [description]
            num_actions ([type]): [description]
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
        assert not (mode == "test" and (dqn_weights is None)), "Need prior weights in test mode."
        self.mode = mode
        
        self.name             = "LC_DQN_Agent with Double Softmax"
        self.num_actions      = num_actions
        self.state_dim        = state_dim
        self.dqn_weights      = dqn_weights
        self.input_norm       = input_norm
        self.input_norm_prior = input_norm_prior
        self.N                = N
        self.act_softmax      = act_softmax
        self.gamma            = gamma
        self.epsilon          = 1.0
        self.eps_decay        = eps_decay
        self.eps_final        = eps_final
        self.n_steps          = n_steps
        self.tgt_update_freq  = tgt_update_freq
        self.lr               = lr
        self.l2_reg           = l2_reg
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
        self.logger = EpochLogger()
        self.logger.save_config(locals())
        
        # init replay buffer and noise
        if mode == "train":
            self.replay_buffer = UniformReplayBuffer(state_dim=state_dim, n_steps=n_steps, gamma=gamma,
                                                     buffer_length=buffer_length, batch_size=batch_size, device=self.device)
            self.replay_buffer_w = copy.deepcopy(self.replay_buffer)

        # init input normalizer
        if input_norm:
            assert not (mode == "test" and input_norm_prior is None), "Please supply 'input_norm_prior' in test mode with input normalization."
            
            if input_norm_prior is not None:
                with open(input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(state_dim=state_dim, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(state_dim=state_dim, prior=None)
        
        # init DQN-ensemble
        self.DQN = [DQN(num_actions=num_actions, state_dim=state_dim).to(self.device) for _ in range(N)]
        
        print("--------------------------------------------")
        print(f"n_params of one DQN: {self._count_params(self.DQN[0])}")
        print("--------------------------------------------")
        
        # load prior weights if available
        if dqn_weights is not None:
            for n in range(N):
                self.DQN[n].load_state_dict(torch.load(dqn_weights))

        # init target net and counter for target update
        self.target_DQN = [copy.deepcopy(net).to(self.device) for net in self.DQN]
        self.tgt_up_cnt = 0
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for target_net in self.target_DQN:
            for p in target_net.parameters():
                p.requires_grad = False

        # init linear combination
        self.w = torch.tensor([1/N for _ in range(N)], requires_grad=True)

        # define optimizer
        self.DQN_optimizer = [optim.Adam(main_net.parameters(), lr=lr, weight_decay=l2_reg) for main_net in self.DQN]
        self.w_optimizer = optim.Adam([self.w], lr=lr)

    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    def _get_combined_Q(self, s, use_target):
        """Computes for a given state Q_combined(s,a) which is defined to be the exponentially-w-weighted sum over all Q for each action.
        
        s:          torch.Size([batch_size, in_channels, height, width])
        use_target: bool

        returns:    torch.Size([batch_size, num_actions])"""

        # transform w to sum up to 1
        w = torch.exp(self.w) / torch.sum(torch.exp(self.w))

        # use main nets for Q_comb
        if use_target == False:
            
            Q_comb = w[0] * self.DQN[0](s)
            for Q_idx in range(1, self.N):
                Q_comb += w[Q_idx] * self.DQN[Q_idx](s)
        
        # or target nets
        else:
            
            Q_comb = w[0] * self.target_DQN[0](s)
            for Q_idx in range(1, self.N):
                Q_comb += w[Q_idx] * self.target_DQN[Q_idx](s)

        return Q_comb

    @torch.no_grad()
    def select_action(self, s):
        """Epsilon-greedy based action selection for a given state.
        Arg s:   np.array with shape (state_dim,)
        returns: int for the action
        """
        # random action
        if (np.random.binomial(1, self.epsilon) == 1) and (self.mode == "train"):
            a = np.random.randint(low=0, high=self.num_actions, size=1, dtype=int).item()
            
        # greedy action
        else:
            # reshape obs
            s = torch.tensor(s.astype(np.float32)).view(1, self.state_dim).to(self.device)

            # compute Q_comb
            Q_comb = self._get_combined_Q(s, use_target=False)

            # greedy
            a = torch.argmax(Q_comb).item()

        # decay epsilon
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_final)

        return a
    
    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        if np.random.binomial(1, 0.5) == 1:
            self.replay_buffer.add(s, a, r, s2, d)
        else:
            self.replay_buffer_w.add(s, a, r, s2, d)

    def train(self):
        """Train Q and w."""
        self.train_Q()
        self.train_w()

    def train_Q(self):
        """Samples from replay_buffer, updates Q and the target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        # sample Q to update
        Q_idx = np.random.choice(self.N)

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer[Q_idx].zero_grad()
        
        # calculate current estimated Q-values
        Q_v = self.DQN[Q_idx](s)
        Q_v = torch.gather(input=Q_v, dim=1, index=a)
 
        # calculate targets
        with torch.no_grad():

            # Q-value of next state-action pair
            target_Qcomb_next = self._get_combined_Q(s2, use_target=True)

            if self.act_softmax:
                softmax = torch.sum(F.softmax(target_Qcomb_next, dim=1) * target_Qcomb_next, dim=1)
                target_Q_next = softmax.reshape(self.batch_size, 1)
            else:
                target_Q_next = torch.max(target_Qcomb_next, dim=1).values.reshape(self.batch_size, 1)

            # target
            target_Q = r + self.gamma * target_Q_next * (1 - d)

        # calculate loss
        loss = F.mse_loss(Q_v, target_Q)
        
        # compute gradients
        loss.backward()
        
        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN[Q_idx].parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN[Q_idx].parameters(), max_norm=10)
      
        # perform optimizing step
        self.DQN_optimizer[Q_idx].step()

        # log critic training
        self.logger.store(Loss=loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q_v.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_update()

        # increase target-update cnt
        self.tgt_up_cnt += 1
    
    def train_w(self):
        """Samples from replay_buffer_w and updates w."""

        # sample batch
        batch = self.replay_buffer_w.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        # sample Q used in the update
        Q_idx = np.random.choice(self.N)

        #------------- train w ----------------
        # clear gradients
        self.w_optimizer.zero_grad()

        # calculate current estimated Q-values
        with torch.no_grad():
            Q_v = self.DQN[Q_idx](s)
            Q_v = torch.gather(input=Q_v, dim=1, index=a)

        # Q-value of next state-action pair
        target_Qcomb_next = self._get_combined_Q(s2, use_target=True)
        if self.act_softmax:
            softmax = torch.sum(F.softmax(target_Qcomb_next, dim=1) * target_Qcomb_next, dim=1)
            target_Q_next = softmax.reshape(self.batch_size, 1)
        else:
            target_Q_next = torch.max(target_Qcomb_next, dim=1).values.reshape(self.batch_size, 1)
        
        # target
        target_Q = r + self.gamma * target_Q_next * (1 - d)

        # calculate loss
        loss = F.mse_loss(Q_v, target_Q)
        
        # compute gradients
        loss.backward()
        
        # perform optimizing step
        self.w_optimizer.step()

        # log new normalized w
        w_dict = dict()
        for n in range(self.N):
            w_dict[f"w{n}"] = torch.exp(self.w[n]).item()/torch.sum(torch.exp(self.w)).item()
        self.logger.store(**w_dict)

    @torch.no_grad()
    def target_update(self):
        for i in range(self.N):
            self.target_DQN[i].load_state_dict(self.DQN[i].state_dict())
