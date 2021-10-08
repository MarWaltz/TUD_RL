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
from current_algos.CNN_based.Bootstrapped_DQN_MinAtar.bootstrapped_dqn_buffer_MinAtar import \
    UniformReplayBuffer_Bootstrap_CNN
from current_algos.CNN_based.Bootstrapped_DQN_MinAtar.bootstrapped_dqn_nets_MinAtar import \
    CNN_Bootstrapped_DQN
from current_algos.common.logging_func import *
from current_algos.common.normalizer import Input_Normalizer


class CNN_Bootstrapped_DQN_Agent:
    def __init__(self, 
                 mode,
                 num_actions, 
                 state_shape,
                 dqn_weights      = None, 
                 input_norm       = False,
                 input_norm_prior = None,
                 double           = False,
                 our_estimator    = True,
                 our_alpha        = 0.1,
                 K                = 10,
                 mask_p           = 1.0,
                 gamma            = 0.99,
                 n_steps          = 1,
                 tgt_update_freq  = 1000,
                 optimizer        = "Adam",
                 lr               = 0.00025,
                 grad_momentum    = 0.95,
                 sq_grad_momentum = 0.95,
                 min_sq_grad      = 0.01,
                 l2_reg           = 0.0,
                 buffer_length    = int(10e5),
                 grad_clip        = False,
                 grad_rescale     = True,
                 act_start_step   = 5000,
                 upd_start_step   = 5000,
                 upd_every        = 1,
                 batch_size       = 32,
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
            K (float, optional): Number of heads. Defaults to 5.
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
        
        self.name        = "CNN_Bootstrapped_DQN_Agent" if our_estimator == False else f"CNN_OurBootstrapped_DQN_Agent_{our_alpha}"
        self.num_actions = num_actions
 
        # CNN shape
        assert len(state_shape) == 3 and type(state_shape) == tuple, "'state_shape' should be: (in_channels, height, width)."
        self.state_shape = state_shape

        self.dqn_weights      = dqn_weights
        self.input_norm       = input_norm
        self.input_norm_prior = input_norm_prior

        assert not (double and our_estimator), "Can pick either the double or our estimator, not both."

        self.double        = double
        self.our_estimator = our_estimator

        if self.our_estimator:
            self.critical_value = scipy.stats.norm().ppf(our_alpha)

        self.gamma            = gamma
        self.K                = K
        self.mask_p           = mask_p

        self.n_steps          = n_steps
        self.tgt_update_freq  = tgt_update_freq
        self.optimizer        = optimizer

        assert self.optimizer in ["Adam", "RMSprop"], "Pick 'Adam' or 'RMSprop' as optimizer, please."
        self.lr               = lr
        self.grad_momentum    = grad_momentum
        self.sq_grad_momentum = sq_grad_momentum
        self.min_sq_grad      = min_sq_grad
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
        self.logger = EpochLogger(alg_str = self.name)
        self.logger.save_config(locals())
        
        # init replay buffer and noise
        if mode == "train":
            self.replay_buffer = UniformReplayBuffer_Bootstrap_CNN(state_shape=state_shape, n_steps=n_steps, gamma=gamma, K=K, mask_p=mask_p,
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
        
        # init convolutional DQN
        self.DQN = CNN_Bootstrapped_DQN(in_channels=state_shape[0], height=state_shape[1], width=state_shape[2], num_actions=num_actions, K=K).to(self.device)
        
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
            self.DQN_optimizer = optim.RMSprop(self.DQN.parameters(), lr=lr, momentum=grad_momentum, alpha=sq_grad_momentum, centered=True, eps=min_sq_grad)

    def _count_params(self, net):
        return sum([np.prod(p.shape) for p in net.parameters()])

    def _mean_test(self, mean1, mean2, var_mean1, var_mean2):
        """Returns True if the H_0: mu1 >= mu2 was not rejected, else False. 
        Note: mean2 should be the ME.

        Args:
            mean1: mean of X1
            mean2: mean of X2
            var_mean1: variance estimate of mean1 
            var_mean2: variance estimate of mean2
        """
        T = (mean1 - mean2) / torch.sqrt(var_mean1 + var_mean2)
        return T >= self.critical_value

    @torch.no_grad()
    def select_action(self, s, active_head):
        """Greedy action selection using the active head for a given state.
        s:           np.array with shape (in_channels, height, width)
        active_head: int 

        returns: int for the action
        """
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]))
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
        Q_v_all      = self.DQN(s)
        Q_v2_all_tgt = self.target_DQN(s2)

        if self.double:
            Q_v2_all_main = self.DQN(s2)
        
        if self.our_estimator:

            # stack list into torch.Size([K, batch_size, num_actions])
            Q_v2_all_stacked = torch.stack(Q_v2_all_tgt)

            # compute variances over the K heads, gives torch.Size([batch_size, num_actions])
            Q_v2_var = torch.var(Q_v2_all_stacked, dim=0, unbiased=True)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):
            
            # gather actions
            Q_v = torch.gather(input=Q_v_all[k], dim=1, index=a)

            # calculate targets
            with torch.no_grad():

                # Q-value of next state-action pair
                if self.double:
                    a2 = torch.argmax(Q_v2_all_main[k], dim=1).reshape(self.batch_size, 1)
                    target_Q_next = torch.gather(input=Q_v2_all_tgt[k], dim=1, index=a2)

                elif self.our_estimator:

                    # get easy access to relevant target Q
                    Q_tgt = Q_v2_all_tgt[k].to(self.device)

                    # get values and action indices for ME
                    ME_values, ME_a_indices = torch.max(Q_tgt, dim=1)

                    # reshape indices
                    ME_a_indices = ME_a_indices.reshape(self.batch_size, 1)

                    # get variance of ME
                    ME_var = torch.gather(Q_v2_var, dim=1, index=ME_a_indices)[0]   # torch.Size([batch_size])

                    # perform pairwise tests
                    keep = torch.empty((self.batch_size, self.num_actions)).to(self.device)

                    for a_idx in range(self.num_actions):
                        keep[:, a_idx] = self._mean_test(mean1=Q_tgt[:, a_idx], mean2=ME_values, var_mean1=Q_v2_var[:, a_idx], var_mean2=ME_var)

                    # compute mean of kept Qs
                    target_Q_next = torch.sum(Q_tgt * keep, dim=1) / torch.sum(keep, dim = 1)
                    target_Q_next = target_Q_next.reshape(self.batch_size, 1)

                else:
                    target_Q_next = torch.max(Q_v2_all_tgt[k], dim=1).values.reshape(self.batch_size, 1)

                # target
                target_Q = r + (self.gamma ** self.n_steps) * target_Q_next * (1 - d)

            # calculate (Q - y)**2
            loss_k = F.mse_loss(Q_v, target_Q, reduction="none")

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
