import copy
import math
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.logging_func import *
from common.normalizer import Input_Normalizer
from current_algos.DQN.dqn_buffer import UniformReplayBuffer
from current_algos.DQN.dqn_nets import CNN_DQN, DQN


class DQN_Agent:
    def __init__(self, 
                 mode,
                 num_actions, 
                 state_shape,
                 state_type,
                 dqn_weights, 
                 input_norm,
                 input_norm_prior,
                 double,
                 self_correcting_beta,
                 gamma,
                 eps_init,
                 eps_final,
                 eps_decay_steps,
                 tgt_update_freq,
                 net_struc_dqn,
                 optimizer,
                 loss,
                 lr,
                 buffer_length,
                 grad_clip,
                 grad_rescale,
                 act_start_step,
                 upd_start_step,
                 upd_every,
                 batch_size,
                 device,
                 env_str,
                 seed):
        """Initializes agent. Agent can select actions based on his model, memorize and replay to train his model.

        Args:
            mode ([type]): [description]
            num_actions ([type]): [description]
            state_shape ([type]): [description]
            state_type (str, optional): [description]. Defaults to "image".
            dqn_weights ([type], optional): [description]. Defaults to None.
            input_norm (bool, optional): [description]. Defaults to False.
            input_norm_prior ([type], optional): [description]. Defaults to None.
            double (bool, optional): [description]. Defaults to False.
            gamma (float, optional): [description]. Defaults to 0.99.
            eps_init (float, optional): [description]. Defaults to 1.0.
            eps_final (float, optional): [description]. Defaults to 0.1.
            eps_decay_steps (int, optional): [description]. Defaults to 100000.
            tgt_update_freq (int, optional): [description]. Defaults to 1000.
            optimizer (str, optional): [description]. Defaults to "RMSprop".
            loss (str, optional): [description]. Defaults to "SmoothL1Loss".
            lr (float, optional): [description]. Defaults to 0.00025.
            buffer_length ([type], optional): [description]. Defaults to int(1e5).
            grad_clip (bool, optional): [description]. Defaults to False.
            grad_rescale (bool, optional): [description]. Defaults to False.
            act_start_step (int, optional): [description]. Defaults to 5000.
            upd_start_step (int, optional): [description]. Defaults to 5000.
            upd_every (int, optional): [description]. Defaults to 1.
            batch_size (int, optional): [description]. Defaults to 32.
            device (str, optional): [description]. Defaults to "cpu".
            env_str ([type], optional): [description]. Defaults to None.
        """

        # store attributes and hyperparameters
        assert mode in ["train", "test"], "Unknown mode. Should be 'train' or 'test'."
        assert not (mode == "test" and (dqn_weights is None)), "Need prior weights in test mode."
        self.mode = mode
        
        assert not (double and self_correcting_beta is not None), "Specify either 'double' or 'self_correcting_beta', not both."

        if double:
            self.name = "ddqn_agent"
        elif self_correcting_beta is not None:
            self.name = f"scdqn_{self_correcting_beta}_agent"
        else:
            self.name = "dqn_agent"

        self.num_actions = num_actions
 
        # state type and shape
        self.state_type = state_type
        self.state_shape = state_shape

        assert self.state_type in ["image", "feature"], "'state_type' can be either 'Image' or 'Vector'."

        if state_type == "image":
            assert len(state_shape) == 3 and type(state_shape) == tuple, "'state_shape' should be: (in_channels, height, width) for images."

        self.dqn_weights          = dqn_weights
        self.input_norm           = input_norm
        self.input_norm_prior     = input_norm_prior
        self.double               = double
        self.self_correcting_beta = self_correcting_beta
        self.gamma                = gamma

        # linear epsilon schedule
        self.eps_init         = eps_init
        self.epsilon          = eps_init
        self.eps_final        = eps_final
        self.eps_decay_steps  = eps_decay_steps
        self.eps_inc          = (eps_final - eps_init) / eps_decay_steps
        self.eps_t            = 0

        self.tgt_update_freq  = tgt_update_freq
        self.net_struc_dqn    = net_struc_dqn

        if state_type == "image" and net_struc_dqn is not None:
            warnings.warn("For CNN-based nets, your specification of 'net_struc_dqn' is not considered.")

        self.optimizer        = optimizer
        self.loss             = loss

        assert self.loss in ["SmoothL1Loss", "MSELoss"], "Pick 'SmoothL1Loss' or 'MSELoss', please."
        assert self.optimizer in ["Adam", "RMSprop"], "Pick 'Adam' or 'RMSprop' as optimizer, please."
        
        self.lr               = lr
        self.buffer_length    = buffer_length
        self.grad_clip        = grad_clip
        self.grad_rescale     = grad_rescale
        self.act_start_step   = act_start_step
        self.upd_start_step   = upd_start_step
        self.upd_every        = upd_every
        self.batch_size       = batch_size

        # gpu support
        assert device in ["cpu", "cuda"], "Unknown device."

        if device == "cpu":    
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            print("Using GPU support.")

        # init logger and save config
        self.logger = EpochLogger(alg_str = self.name, env_str = env_str)
        self.logger.save_config(locals())
        
        # init replay buffer and noise
        if mode == "train":
            self.replay_buffer = UniformReplayBuffer(state_type=state_type, state_shape=state_shape, 
                                                     buffer_length=buffer_length, batch_size=batch_size, device=self.device)

        # init input normalizer
        if input_norm:
            assert not (mode == "test" and input_norm_prior is None), "Please supply 'input_norm_prior' in test mode with input normalization."
            
            if input_norm_prior is not None:
                with open(input_norm_prior, "rb") as f:
                    prior = pickle.load(f)
                self.inp_normalizer = Input_Normalizer(state_dim=state_shape, prior=prior)
            else:
                self.inp_normalizer = Input_Normalizer(state_dim=state_shape, prior=None)
        
        # init DQN
        if self.state_type == "image":
            self.DQN = CNN_DQN(in_channels=state_shape[0], height=state_shape[1], width=state_shape[2], num_actions=num_actions).to(self.device)

        elif self.state_type == "feature":
            self.DQN = DQN(num_actions=num_actions, state_dim=state_shape, net_struc_dqn=net_struc_dqn).to(self.device)

        print("--------------------------------------------")
        print(f"n_params DQN: {self._count_params(self.DQN)}")
        print("--------------------------------------------")
        
        # load prior weights if available
        if dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(dqn_weights))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0
        
        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=lr)
        else:
            self.DQN_optimizer = optim.RMSprop(self.DQN.parameters(), lr=lr, alpha=0.95, centered=True, eps=0.01)

    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    @torch.no_grad()
    def select_action(self, s):
        """Epsilon-greedy based action selection for a given state.
        Arg s:   np.array with shape (in_channels, height, width)
        returns: int for the action
        """
        # random action
        if (np.random.binomial(1, self.epsilon) == 1) and (self.mode == "train"):
            a = np.random.randint(low=0, high=self.num_actions, size=1, dtype=int).item()
            
        # greedy action
        else:
            # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
            s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

            # forward pass
            q = self.DQN(s).to(self.device)

            # greedy
            a = torch.argmax(q).item()

        # anneal epsilon linearly
        if self.mode == "train":
            self.eps_t += 1
            self.epsilon = max(self.eps_inc * self.eps_t + self.eps_init, self.eps_final)

        return a

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d = batch

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # calculate current estimated Q-values
        Q_v = self.DQN(s)
        Q_v = torch.gather(input=Q_v, dim=1, index=a)
 
        # calculate targets
        with torch.no_grad():

            # Q-value of next state-action pair
            if self.double:
                a2 = torch.argmax(self.DQN(s2), dim=1).reshape(self.batch_size, 1)
                target_Q_next = torch.gather(input=self.target_DQN(s2), dim=1, index=a2)

            elif self.self_correcting_beta is not None:
                target_Q_beta = (1 - self.self_correcting_beta) * self.target_DQN(s2) + self.self_correcting_beta * self.DQN(s2)
                a2 = torch.argmax(target_Q_beta, dim=1).reshape(self.batch_size, 1)
                target_Q_next = torch.gather(input=self.target_DQN(s2), dim=1, index=a2)

            else:
                target_Q_next = self.target_DQN(s2)
                target_Q_next = torch.max(target_Q_next, dim=1).values.reshape(self.batch_size, 1)

            # target
            target_Q = r + self.gamma * target_Q_next * (1 - d)

        # calculate loss
        if self.loss == "MSELoss":
            loss = F.mse_loss(Q_v, target_Q)

        elif self.loss == "SmoothL1Loss":
            loss = F.smooth_l1_loss(Q_v, target_Q)
        
        # compute gradients
        loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_optimizer.step()
        
        # log critic training
        self.logger.store(Loss=loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q_v.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_update()

        # increase target-update cnt
        self.tgt_up_cnt += 1
    
    @torch.no_grad()
    def target_update(self):
        """Hard update of target network weights."""
        self.target_DQN.load_state_dict(self.DQN.state_dict())
