import copy
import math
import pickle
from collections import Counter

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from current_algos.Bootstrapped_DQN.bootstrapped_dqn_buffer import UniformReplayBuffer_Bootstrapped_DQN
from current_algos.Bootstrapped_DQN.bootstrapped_dqn_nets import Bootstrapped_DQN, CNN_Bootstrapped_DQN
from common.logging_func import *
from common.normalizer import Input_Normalizer


class Bootstrapped_DQN_Agent:
    def __init__(self, 
                 mode,
                 num_actions, 
                 state_shape,
                 state_type       = "image",
                 dqn_weights      = None, 
                 input_norm       = False,
                 input_norm_prior = None,
                 double           = False,
                 kernel           = None,
                 kernel_param     = None,
                 K                = 10,
                 mask_p           = 1.0,
                 gamma            = 0.99,
                 tgt_update_freq  = 1000,
                 optimizer        = "Adam",
                 loss             = "MSELoss",
                 lr               = 0.00025,
                 buffer_length    = int(1e5),
                 grad_clip        = False,
                 grad_rescale     = True,
                 act_start_step   = 5000,
                 upd_start_step   = 5000,
                 upd_every        = 1,
                 batch_size       = 32,
                 device           = "cpu",
                 env_str          = None):
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
            kernel ([type], optional): [description]. Defaults to None.
            kernel_param ([type], optional): [description]. Defaults to None.
            K (int, optional): [description]. Defaults to 10.
            mask_p (float, optional): [description]. Defaults to 1.0.
            gamma (float, optional): [description]. Defaults to 0.99.
            tgt_update_freq (int, optional): [description]. Defaults to 1000.
            optimizer (str, optional): [description]. Defaults to "RMSprop".
            loss (str, optional): [description]. Defaults to "SmoothL1Loss".
            lr (float, optional): [description]. Defaults to 0.00025.
            buffer_length ([type], optional): [description]. Defaults to int(1e5).
            grad_clip (bool, optional): [description]. Defaults to False.
            grad_rescale (bool, optional): [description]. Defaults to True.
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
        
        if kernel is not None:
            self.name = f"our_bootstrapped_dqn_agent_{kernel}"
        elif double:
            self.name = "bootstrapped_ddqn_agent"
        else:
            self.name = "bootstrapped_dqn_agent"

        self.num_actions = num_actions
 
        # state type and shape
        self.state_type = state_type
        self.state_shape = state_shape

        assert self.state_type in ["image", "feature"], "'state_type' can be either 'Image' or 'Vector'."

        if state_type == "image":
            assert len(state_shape) == 3 and type(state_shape) == tuple, "'state_shape' should be: (in_channels, height, width) for images."

        self.dqn_weights      = dqn_weights
        self.input_norm       = input_norm
        self.input_norm_prior = input_norm_prior

        # estimator
        assert not (double and kernel is not None), "Can pick either the double or a kernel estimator, not both."
        self.double = double
        
        assert kernel in [None, "test", "gaussian_cdf"], "Unknown kernel."
        self.kernel = kernel
        self.kernel_param = kernel_param

        if kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(kernel_param)
            self.g = lambda u: u >= self.critical_value

        elif kernel == "gaussian_cdf":
            self.g = scipy.stats.norm(scale=kernel_param).cdf

        # further params
        self.gamma            = gamma
        self.K                = K
        self.mask_p           = mask_p

        self.tgt_update_freq  = tgt_update_freq
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
            self.replay_buffer = UniformReplayBuffer_Bootstrapped_DQN(state_type=state_type, state_shape=state_shape, K=K, mask_p=mask_p,
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
            self.DQN = CNN_Bootstrapped_DQN(in_channels=state_shape[0], height=state_shape[1], width=state_shape[2], num_actions=num_actions, K=K).to(self.device)
        
        elif self.state_type == "feature":
            self.DQN = Bootstrapped_DQN(state_dim=state_shape, num_actions=num_actions, K=K)

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
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=lr, weight_decay=l2_reg)
        else:
            self.DQN_optimizer = optim.RMSprop(self.DQN.parameters(), lr=lr, alpha=0.95, centered=True, eps=0.01)

    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    @torch.no_grad()
    def select_action(self, s, active_head):
        """Greedy action selection using the active head for a given state.
        s:           np.array with shape (in_channels, height, width)
        active_head: int 

        returns: int for the action
        """
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        if active_head is not None:
            q = self.DQN(s, active_head)

            # greedy
            a = torch.argmax(q).item()
        
        # majority vote
        else:

            # push through all heads
            q = self.DQN(s)

            # get favoured action of each head
            actions = [torch.argmax(head_q).item() for head_q in q]

            # choose majority vote
            actions = Counter(actions)
            a = actions.most_common(1)[0][0]

        return a

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""        
        # sample batch
        batch = self.replay_buffer.sample()
        
        # unpack batch
        s, a, r, s2, d, m = batch

        #-------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()
        
        # calculate current estimated Q-values and next Q-values
        Q_s_main = self.DQN(s)
        Q_s2_tgt = self.target_DQN(s2)

        if self.double:
            Q_s2_main = self.DQN(s2)
        
        if self.kernel is not None:

            # stack list into torch.Size([K, batch_size, num_actions])
            Q_s2_tgt_stacked = torch.stack(Q_s2_tgt)

            # compute variances over the K heads, gives torch.Size([batch_size, num_actions])
            Q_s2_var = torch.var(Q_s2_tgt_stacked, dim=0, unbiased=True)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):
            
            # gather actions
            Q_s = torch.gather(input=Q_s_main[k], dim=1, index=a)

            # calculate targets
            with torch.no_grad():

                # Q-value of next state-action pair
                if self.double:
                    a2 = torch.argmax(Q_s2_main[k], dim=1).reshape(self.batch_size, 1)
                    target_Q_next = torch.gather(input=Q_s2_tgt[k], dim=1, index=a2)

                elif self.kernel is not None:

                    # get easy access to relevant target Q
                    Q_tgt = Q_s2_tgt[k].to(self.device)

                    # get values and action indices for ME
                    ME_values, ME_a_indices = torch.max(Q_tgt, dim=1)

                    # reshape indices
                    ME_a_indices = ME_a_indices.reshape(self.batch_size, 1)

                    # get variance of ME
                    ME_var = torch.gather(Q_s2_var, dim=1, index=ME_a_indices).reshape(self.batch_size)

                    # compute weights
                    w = torch.empty((self.batch_size, self.num_actions)).to(self.device)

                    for a_idx in range(self.num_actions):
                        u = (Q_tgt[:, a_idx] - ME_values) / torch.sqrt(Q_s2_var[:, a_idx] + ME_var)
                        w[:, a_idx] = torch.tensor(self.g(u))

                    # compute weighted mean
                    target_Q_next = torch.sum(Q_tgt * w, dim=1) / torch.sum(w, dim=1)
                    target_Q_next = target_Q_next.reshape(self.batch_size, 1)

                else:
                    target_Q_next = torch.max(Q_s2_tgt[k], dim=1).values.reshape(self.batch_size, 1)

                # target
                target_Q = r + self.gamma * target_Q_next * (1 - d)

            # calculate (Q - y)**2
            if self.loss == "MSELoss":
                loss_k = F.mse_loss(Q_s, target_Q, reduction="none")

            elif self.loss == "SmoothL1Loss":
                loss_k = F.smooth_l1_loss(Q_s, target_Q, reduction="none")

            # use only relevant samples for given head
            loss_k = loss_k * m[:, k].unsqueeze(1)

            # append loss
            losses.append(torch.sum(loss_k) / torch.sum(m[:, k]))
       
        # compute gradients
        loss = sum(losses)
        loss.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN.core.parameters():
                p.grad *= 1/float(self.K)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN.parameters(), max_norm=10)
        
        # perform optimizing step
        self.DQN_optimizer.step()
        
        # log critic training
        self.logger.store(Loss=loss.detach().cpu().numpy().item())
        self.logger.store(Q_val=Q_s.detach().mean().cpu().numpy().item())

        #------- Update target networks -------
        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_update()

        # increase target-update cnt
        self.tgt_up_cnt += 1
    
    @torch.no_grad()
    def target_update(self):
        """Hard update of target network weights."""
        self.target_DQN.load_state_dict(self.DQN.state_dict())
