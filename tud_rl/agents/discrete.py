import math
import copy
import warnings
import scipy.stats
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tud_rl.common.nets as nets
import tud_rl.common.buffer as buffer

from tud_rl.agents.base import BaseAgent, _BootAgent

from collections import Counter

from tud_rl.common.helper_fnc import get_MC_ret_from_rew
from tud_rl.common.exploration import LinearDecayEpsilonGreedy
from tud_rl.common.configparser import ConfigFile


class DQNAgent(BaseAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.lr = c.lr
        self.dqn_weights = c.dqn_weights
        self.eps_init = c.eps_init
        self.eps_final = c.eps_final
        self.eps_decay_steps = c.eps_decay_steps
        self.tgt_update_freq = c.tgt_update_freq
        self.net_struc = c.net_struc

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)
                    ), "Need prior weights in test mode."

        if self.state_type == "image" and self.net_struc is not None:
            raise Exception(
                "For CNN-based nets, the specification of 'net_struc_dqn' should be 'None'.")

        # linear epsilon schedule
        self.exploration = LinearDecayEpsilonGreedy(eps_init=self.eps_init,
                                                    eps_final=self.eps_final,
                                                    eps_decay_steps=self.eps_decay_steps)

        # replay buffer
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer(state_type=self.state_type,
                                                            state_shape=self.state_shape,
                                                            buffer_length=self.buffer_length,
                                                            batch_size=self.batch_size,
                                                            device=self.device,
                                                            disc_actions=True)

        # init DQN
        if self.state_type == "image":
            self.DQN = nets.MinAtar_DQN(in_channels=self.state_shape[0],
                                        height=self.state_shape[1],
                                        width=self.state_shape[2],
                                        num_actions=self.num_actions).to(self.device)

        elif self.state_type == "feature":
            self.DQN = nets.MLP(in_size=self.state_shape,
                                out_size=self.num_actions,
                                net_struc=self.net_struc).to(self.device)

        # Number of Parameters of Net
        self.n_params = self._count_params(self.DQN)

        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(
                self.dqn_weights, map_location=self.device))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(
                self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(
                self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d)

    @torch.no_grad()
    def select_action(self, s):
        """Epsilon-greedy based action selection for a given state.

        Arg s:   np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
        returns: int for the action
        """

        # get current epsilon
        curr_epsilon = self.exploration.get_epsilon(self.mode)

        # random
        if np.random.binomial(1, curr_epsilon) == 1:
            a = np.random.randint(
                low=0, high=self.num_actions, size=1, dtype=int).item()

        # greedy
        else:
            a = self._greedy_action(s)
        return a

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action.
        Args:
            s:      np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
            with_Q: bool, whether to return the associated Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)
        """
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN(s).to(self.device)

        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, q[0][a].item()
        return a

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            Q_next = self.target_DQN(s2)
            Q_next = torch.max(Q_next, dim=1).values.reshape(
                self.batch_size, 1)
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def _compute_loss(self, Q, y, reduction="mean"):
        if self.loss == "MSELoss":
            return F.mse_loss(Q, y, reduction=reduction)

        elif self.loss == "SmoothL1Loss":
            return F.smooth_l1_loss(Q, y, reduction=reduction)

    def train(self):
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s, a, r, s2, d = batch

        # -------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()

        # Q estimates
        Q = self.DQN(s)
        Q = torch.gather(input=Q, dim=1, index=a)

        # targets
        y = self._compute_target(r, s2, d)

        # loss
        loss = self._compute_loss(Q=Q, y=y)

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
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        # ------- Update target networks -------
        self._target_update()

    @torch.no_grad()
    def _target_update(self):

        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            self.target_DQN.load_state_dict(self.DQN.state_dict())

        # increase target-update cnt
        self.tgt_up_cnt += 1


class ACCDDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.AC_K = getattr(c.Agent, agent_name)["AC_K"]

        # checks
        assert self.AC_K <= self.num_actions, "ACC-K cannot exceed number of actions."

        # replace DQN + target by DQN_A + DQN_B
        self.DQN_A = self.DQN
        del self.target_DQN

        if self.state_type == "image":
            self.DQN_B = nets.MinAtar_DQN(in_channels=self.state_shape[0],
                                          height=self.state_shape[1],
                                          width=self.state_shape[2],
                                          num_actions=self.num_actions).to(self.device)
        elif self.state_type == "feature":
            self.DQN_B = nets.MLP(in_size=self.state_shape,
                                  out_size=self.num_actions,
                                  net_struc=self.net_struc).to(self.device)

        # Number of Parameters of Net
        self.n_params = self._count_params(self.DQN_A)

        # prior weights
        if self.dqn_weights is not None:
            raise NotImplementedError(
                "Weight loading for AC_CDDQN is not implemented yet.")

        #  optimizer
        del self.DQN_optimizer

        if self.optimizer == "Adam":
            self.DQN_A_optimizer = optim.Adam(
                self.DQN_A.parameters(), lr=self.lr)
            self.DQN_B_optimizer = optim.Adam(
                self.DQN_B.parameters(), lr=self.lr)
        else:
            self.DQN_A_optimizer = optim.RMSprop(
                self.DQN_A.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)
            self.DQN_B_optimizer = optim.RMSprop(
                self.DQN_B.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action.
        Args:
            s:      np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
            with_Q: bool, whether to return the associated Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)
        """

        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN_A(s).to(self.device) + self.DQN_B(s).to(self.device)

        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, 0.5 * q[0][a].item()
        return a

    def train(self):
        """Samples from replay_buffer and updates DQN."""

        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s, a, r, s2, d = batch

        # -------- train DQN_A & DQN_B --------
        # Note: The description of the training process is not completely clear, see Section 'Deep Version' of Jiang et. al (2021).
        #       Here, both nets will be updated towards the same target, stemming from Equation (12). Alternatively, one could compute
        #       two distinct targets based on different buffer samples and train each net separately.

        # clear gradients
        self.DQN_A_optimizer.zero_grad()
        self.DQN_B_optimizer.zero_grad()

        # Q-values
        QA = self.DQN_A(s)
        QA = torch.gather(input=QA, dim=1, index=a)

        QB = self.DQN_B(s)
        QB = torch.gather(input=QB, dim=1, index=a)

        # targets
        with torch.no_grad():

            # compute candidate set based on QB
            QB_v2 = self.DQN_B(s2)
            M_K = torch.argsort(QB_v2, dim=1, descending=True)[:, :self.AC_K]

            # get a_star_K
            QA_v2 = self.DQN_A(s2)
            a_star_K = torch.empty((self.batch_size, 1),
                                   dtype=torch.int64).to(self.device)

            for bat_idx in range(self.batch_size):

                # get best action of the candidate set
                act_idx = torch.argmax(QA_v2[bat_idx][M_K[bat_idx]])

                # store its index
                a_star_K[bat_idx] = M_K[bat_idx][act_idx]

            # evaluate a_star_K on B
            Q_next = torch.gather(QB_v2, dim=1, index=a_star_K)

            # clip to ME
            ME = torch.max(QA_v2, dim=1).values.reshape(self.batch_size, 1)
            Q_next = torch.min(Q_next, ME)

            # target
            y = r + self.gamma * Q_next * (1 - d)

        # calculate loss
        loss_A = self._compute_loss(QA, y)
        loss_B = self._compute_loss(QB, y)

        # compute gradients
        loss_A.backward()
        loss_B.backward()

        # gradient scaling and clipping
        if self.grad_rescale:
            for p in self.DQN_A.parameters():
                p.grad *= 1 / math.sqrt(2)
            for p in self.DQN_B.parameters():
                p.grad *= 1 / math.sqrt(2)
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.DQN_A.parameters(), max_norm=10)
            nn.utils.clip_grad_norm_(self.DQN_B.parameters(), max_norm=10)

        # perform optimizing step
        self.DQN_A_optimizer.step()
        self.DQN_B_optimizer.step()

        # log critic training
        self.logger.store(Loss=loss_A.detach().cpu().numpy().item())
        self.logger.store(Q_val=QA.detach().mean().cpu().numpy().item())


class BootDQNAgent(DQNAgent, _BootAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.K = getattr(c.Agent, agent_name)["K"]
        self.mask_p = getattr(c.Agent, agent_name)["mask_p"]
        self.grad_rescale = getattr(c.Agent, agent_name)["grad_rescale"]
        c.grad_rescale = self.grad_rescale   # for correct logging

        # checks
        assert self.state_type == "image", "Currently, BootDQN is only available with 'image' input."

        # replay buffer with masks
        if self.mode == "train":
            self.replay_buffer = buffer.UniformReplayBuffer_BootDQN(state_type=self.state_type,
                                                                    state_shape=self.state_shape,
                                                                    buffer_length=self.buffer_length,
                                                                    batch_size=self.batch_size,
                                                                    device=self.device,
                                                                    K=self.K,
                                                                    mask_p=self.mask_p)
        # init BootDQN
        if self.state_type == "image":
            self.DQN = nets.MinAtar_BootDQN(in_channels=self.state_shape[0],
                                            height=self.state_shape[1],
                                            width=self.state_shape[2],
                                            num_actions=self.num_actions,
                                            K=self.K).to(self.device)

        # Parameter number of net
        self.n_params = self._count_params(self.DQN)

        # prior weights
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights))

        # target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(
                self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

        # init active head
        self.reset_active_head()

    def reset_active_head(self):
        self.active_head = np.random.choice(self.K)

    @torch.no_grad()
    def select_action(self, s):
        """Greedy action selection for a given state using the active head (train) or majority vote (test).
        s:           np.array with shape (in_channels, height, width)
        active_head: int 

        returns: int for the action
        """

        # forward pass
        if self.mode == "train":

            # reshape obs (namely, to torch.Size([1, in_channels, height, width]))
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(
                0).to(self.device)

            # forward
            q = self.DQN(s, self.active_head)

            # greedy
            a = torch.argmax(q).item()

        # majority vote
        else:
            a = self._greedy_action(s)
        return a

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action via majority vote of the bootstrap heads.
        Args:
            s:      np.array with shape (in_channels, height, width)
            with_Q: bool, whether to return the associate ensemble average of Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)
        """

        # reshape obs (namely, to torch.Size([1, in_channels, height, width]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # push through all heads (generates list of length K with torch.Size([1, num_actions]))
        q = self.DQN(s)

        # get favoured action of each head
        actions = [torch.argmax(head_q).item() for head_q in q]

        # choose majority vote
        actions = Counter(actions)
        a = actions.most_common(1)[0][0]

        if with_Q:
            Q_avg = np.mean([q_head[0][a].item() for q_head in q])
            return a, Q_avg
        return a

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s, a, r, s2, d, m = batch

        # -------- train BootDQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()

        # current and next Q-values
        Q_main = self.DQN(s)
        Q_s2_tgt = self.target_DQN(s2)
        Q_s2_main = self.DQN(s2)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):

            # gather actions
            Q = torch.gather(input=Q_main[k], dim=1, index=a)

            # targets
            with torch.no_grad():

                a2 = torch.argmax(Q_s2_main[k], dim=1).reshape(
                    self.batch_size, 1)
                Q_next = torch.gather(input=Q_s2_tgt[k], dim=1, index=a2)

                y = r + self.gamma * Q_next * (1 - d)

            # calculate (Q - y)**2
            loss_k = self._compute_loss(Q, y, reduction="none")

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
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        # ------- Update target networks -------
        self._target_update()


class KEBootDQNAgent(BootDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameter
        self.kernel = getattr(c.Agent, agent_name)["kernel"]
        self.kernel_param = getattr(c.Agent, agent_name)["kernel_param"]

        # checks
        assert self.kernel in ["test", "gaussian_cdf"], "Unknown kernel."

        # kernel funcs
        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(
                u, scale=self.kernel_param), dtype=torch.float32)

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""
        # sample batch
        batch = self.replay_buffer.sample()

        # unpack batch
        s, a, r, s2, d, m = batch

        # -------- train DQN --------
        # clear gradients
        self.DQN_optimizer.zero_grad()

        # current and next Q-values
        Q_main = self.DQN(s)
        Q_s2_tgt = self.target_DQN(s2)

        # stack list into torch.Size([K, batch_size, num_actions])
        Q_s2_tgt_stacked = torch.stack(Q_s2_tgt)

        # compute variances over the K heads, gives torch.Size([batch_size, num_actions])
        Q_s2_var = torch.var(Q_s2_tgt_stacked, dim=0, unbiased=True)

        # set up losses
        losses = []

        # calculate loss for each head
        for k in range(self.K):

            # gather actions
            Q = torch.gather(input=Q_main[k], dim=1, index=a)

            # targets
            with torch.no_grad():

                # get easy access to relevant target Q
                Q_tgt = Q_s2_tgt[k].to(self.device)

                # get values and action indices for ME
                ME_values, ME_a_indices = torch.max(Q_tgt, dim=1)

                # reshape indices
                ME_a_indices = ME_a_indices.reshape(self.batch_size, 1)

                # get variance of ME
                ME_var = torch.gather(
                    Q_s2_var, dim=1, index=ME_a_indices).reshape(self.batch_size)

                # compute weights
                w = torch.zeros(
                    (self.batch_size, self.num_actions)).to(self.device)

                for a_idx in range(self.num_actions):
                    u = (Q_tgt[:, a_idx] - ME_values) / \
                        torch.sqrt(Q_s2_var[:, a_idx] + ME_var)
                    w[:, a_idx] = self.g(u)

                # compute weighted mean
                Q_next = torch.sum(Q_tgt * w, dim=1) / torch.sum(w, dim=1)
                Q_next = Q_next.reshape(self.batch_size, 1)

                # target
                y = r + self.gamma * Q_next * (1 - d)

            # calculate (Q - y)**2
            loss_k = self._compute_loss(Q, y, reduction="none")

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
        self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        # ------- Update target networks -------
        self._target_update()


class AdaKEBootDQNAgent(KEBootDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameter
        self.env_max_episode_steps = c.Env.max_episode_steps
        #self.kernel_param      = torch.tensor(self.kernel_param, dtype=torch.float32, requires_grad=True, device=self.device)
        self.kernel_batch_size = getattr(c.Agent, agent_name)[
            "kernel_batch_size"]
        self.kernel_lr = getattr(c.Agent, agent_name)["kernel_lr"]

        self._set_g()

        # checks
        assert "MinAtar" in c.Env.name, "Currently, AdaKEBootDQN is only available for MinAtar environments."

        # optimizer
        # if self.optimizer == "Adam":
        #    self.kernel_optimizer = optim.Adam([self.kernel_param], lr=self.kernel_lr)
        # else:
        #    self.kernel_optimizer = optim.RMSprop([self.kernel_param], lr=self.kernel_lr, alpha=0.95, centered=True, eps=0.01)

        # bounds
        if self.kernel == "test":
            self.kernel_param_l, self.kernel_param_u = 1e-6, 0.5

        elif self.kernel == "gaussian_cdf":
            self.kernel_param_l, self.kernel_param_u = 0.0, np.inf

        # new buffer since we store envs
        self.replay_buffer = buffer.UniformReplayBufferEnvs_BootDQN(
            state_type=self.state_type,
            state_shape=self.state_shape,
            buffer_length=self.buffer_length,
            batch_size=self.batch_size,
            device=self.device,
            K=self.K,
            mask_p=self.mask_p)

    def _set_g(self):
        """Sets the kernel function depending on the current kernel param."""

        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(
                u, scale=self.kernel_param), dtype=torch.float32)

    def memorize(self, s, a, r, s2, d, env):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, env)

    def _target_update(self):
        if self.tgt_up_cnt % self.tgt_update_freq == 0:

            with torch.no_grad():

                # target
                self.target_DQN.load_state_dict(self.DQN.state_dict())

                # kernel param update
                self._train_kernel()

                # update kernel function
                self._set_g()

        # increase target-update cnt
        self.tgt_up_cnt += 1

    def _train_kernel(self):
        """Updates the kernel param based on recent on-policy rollouts."""

        # perform rollouts
        s, a, MC = self._get_s_a_MC()

        # convert to tensors
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        MC = torch.tensor(MC, dtype=torch.float32)

        # estimate Q for each (s,a) pair as average of bootstrap heads
        # forward through all heads, creates list of length K containing torch.Size([MC_batch_size, num_actions])
        Q = self.DQN(s)

        # gather relevant action for each head, creates list of length K containing torch.Size([MC_batch_size, 1])
        Q = [torch.gather(input=Q_head, dim=1, index=a) for Q_head in Q]

        # average over ensemble
        Q = torch.stack(Q)
        Q = torch.mean(Q, dim=0)

        # get difference term
        delta = torch.sum(MC - Q).item()

        # update kernel param
        self.kernel_param += self.kernel_lr * delta

        # clip it
        self.kernel_param = np.clip(
            self.kernel_param, self.kernel_param_l, self.kernel_param_u)

    def _get_s_a_MC(self):
        """Samples random initial env-specifications and acts greedy wrt current ensemble opinion (majority vote).

        Returns:
            s:  np.array([MC_batch_size, in_channels, height, width]))
            a:  np.array([MC_batch_size, 1]))
            MC: np.array([MC_batch_size, 1])"""

        # go greedy
        self.mode = "test"

        # s and a of ALL episodes
        s_all_eps = []
        a_all_eps = []

        # MC-vals of all (s,a) pairs of ALL episodes
        MC_ret_all_eps = []

        # init epi steps and rewards for ONE episode
        epi_steps = 0
        r_one_eps = []

        # get env and current state | Note: This selection is MinAtar specific.
        sampled_env = self.replay_buffer.sample_env()
        s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
        if self.input_norm:
            s = self.inp_normalizer.normalize(s, mode=self.mode)

        for _ in range(self.kernel_batch_size):

            epi_steps += 1

            # select action
            a = self.select_action(s)

            # perform step
            s2, r, d, _ = sampled_env.step(a)

            # save s, a, r
            s_all_eps.append(s)
            a_all_eps.append(a)
            r_one_eps.append(r)

            # potentially normalize s2
            if self.input_norm:
                s2 = self.inp_normalizer.normalize(s2, mode=self.mode)

            # s becomes s2
            s = s2

            # end of episode: for artificial time limit in env, we need to correct final reward to be a return
            if epi_steps == self.env_max_episode_steps:

                # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
                r_one_eps[-1] += self.gamma * \
                    self._greedy_action(s2, with_Q=True)[1]

            # end of episode: artificial or true done signal
            if epi_steps == self.env_max_episode_steps or d:

                # transform rewards to returns and store them
                MC_ret_all_eps += get_MC_ret_from_rew(
                    rews=r_one_eps, gamma=self.gamma)

                # reset
                epi_steps = 0
                r_one_eps = []

                # get another initial state
                sampled_env = self.replay_buffer.sample_env()
                s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
                if self.input_norm:
                    s = self.inp_normalizer.normalize(s, mode=self.mode)

        # store MC from final unfinished episode
        if len(r_one_eps) > 0:

            # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
            r_one_eps[-1] += self.gamma * \
                self._greedy_action(s2, with_Q=True)[1]

            # transform rewards to returns and store them
            MC_ret_all_eps += get_MC_ret_from_rew(
                rews=r_one_eps, gamma=self.gamma)

        # continue training
        self.mode = "train"

        return np.stack(s_all_eps), np.expand_dims(a_all_eps, 1), np.expand_dims(MC_ret_all_eps, 1)


class EnsembleDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.N = getattr(c.Agent, agent_name)["N"]
        self.N_to_update = getattr(c.Agent, agent_name)["N_to_update"]

        # init EnsembleDQN
        del self.DQN
        self.EnsembleDQN = [None] * self.N

        for i in range(self.N):
            if self.state_type == "image":
                self.EnsembleDQN[i] = nets.MinAtar_DQN(in_channels=self.state_shape[0],
                                                       height=self.state_shape[1],
                                                       width=self.state_shape[2],
                                                       num_actions=self.num_actions).to(self.device)

            elif self.state_type == "feature":
                self.EnsembleDQN[i] = nets.MLP(in_size=self.state_shape,
                                               out_size=self.num_actions,
                                               net_struc=self.net_struc).to(self.device)

        # Parameter number of Net
        self.n_params = self.N * self._count_params(self.EnsembleDQN[0])

        # prior weights
        if self.dqn_weights is not None:
            raise NotImplementedError(
                "Prior weights not implemented so far for EnsembleDQN.")

        # target net and counter for target update
        del self.target_DQN
        self.target_EnsembleDQN = [copy.deepcopy(net).to(
            self.device) for net in self.EnsembleDQN]
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for net in self.target_EnsembleDQN:
            for p in net.parameters():
                p.requires_grad = False

        # define optimizer
        del self.DQN_optimizer
        self.EnsembleDQN_optimizer = [None] * self.N

        for i in range(self.N):
            if self.optimizer == "Adam":
                self.EnsembleDQN_optimizer[i] = optim.Adam(
                    self.EnsembleDQN[i].parameters(), lr=self.lr)

            else:
                self.EnsembleDQN_optimizer[i] = optim.RMSprop(
                    self.EnsembleDQN[i].parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.mean(q_ens, dim=0)

    @torch.no_grad()
    def _greedy_action(self, s, with_Q=False):
        """Selects a greedy action by maximizing over the reduced ensemble.

        Args:
            s:      np.array with shape (in_channels, height, width) or, for feature input, (state_shape,)
            with_Q: bool, whether to return the associate ensemble average of Q-estimates for the selected action
        Returns:
            int for action, float for Q (if with_Q)"""

        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)

        # forward through ensemble
        # list of torch.Size([batch_size, num_actions])
        q_ens = [net(s).to(self.device) for net in self.EnsembleDQN]
        # torch.Size([N, batch_size, num_actions])
        q_ens = torch.stack(q_ens).to(self.device)

        # reduction over ensemble
        q = self._ensemble_reduction(q_ens).to(self.device)

        # greedy
        a = torch.argmax(q).item()

        if with_Q:
            return a, q[0][a].item()
        return a

    def _compute_target(self, r, s2, d):
        with torch.no_grad():

            # forward through ensemble
            Q_next_ens = [net(s2).to(self.device)
                          for net in self.target_EnsembleDQN]
            Q_next_ens = torch.stack(Q_next_ens).to(self.device)

            # reduction over ensemble
            Q_next = self._ensemble_reduction(Q_next_ens)

            # maximization and target
            Q_next = torch.max(Q_next, dim=1).values.reshape(
                self.batch_size, 1)
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""

        # -------- train EnsembleDQN --------
        for _ in range(self.N_to_update):

            # ensemble member to update
            i = np.random.choice(self.N)

            # sample batch
            batch = self.replay_buffer.sample()

            # unpack batch
            s, a, r, s2, d = batch

            # clear gradients
            self.EnsembleDQN_optimizer[i].zero_grad()

            # Q estimates
            Q = self.EnsembleDQN[i](s)
            Q = torch.gather(input=Q, dim=1, index=a)

            # targets
            y = self._compute_target(r, s2, d)

            # loss
            loss = self._compute_loss(Q=Q, y=y)

            # compute gradients
            loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.EnsembleDQN[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(
                    self.EnsembleDQN[i].parameters(), max_norm=10)

            # perform optimizing step
            self.EnsembleDQN_optimizer[i].step()

            # log critic training
            self.logger.store(Loss=loss.detach().cpu().numpy().item())
            self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        # ------- Update target networks -------
        self._target_update()

    def _target_update(self):
        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            for i in range(self.N):
                self.target_EnsembleDQN[i].load_state_dict(
                    self.EnsembleDQN[i].state_dict())

        # increase target-update cnt
        self.tgt_up_cnt += 1


class KEEnsembleDQNAgent(EnsembleDQNAgent):
    """This agent performs action selection like the EnsembleDQN (epsilon-greedy over average of ensemble). 
    Only the target computation differs."""

    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameter
        self.kernel = getattr(c.Agent, agent_name)["kernel"]
        self.kernel_param = getattr(c.Agent, agent_name)["kernel_param"]

        # checks
        assert self.kernel in ["test", "gaussian_cdf"], "Unknown kernel."

        # kernel funcs
        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(
                u, scale=self.kernel_param), dtype=torch.float32)

    def _compute_target(self, r, s2, d, i):
        with torch.no_grad():

            # forward through target ensemble
            Q_s2_tgt = [net(s2).to(self.device)
                        for net in self.target_EnsembleDQN]
            # torch.Size([N, batch_size, num_actions])
            Q_s2_tgt_stacked = torch.stack(Q_s2_tgt).to(self.device)

            # compute variances over the ensemble, gives torch.Size([batch_size, num_actions])
            Q_s2_var = torch.var(Q_s2_tgt_stacked, dim=0, unbiased=True)

            # select ensemble member that is trained
            Q_tgt = Q_s2_tgt[i]

            # get values and action indices for ME
            ME_values, ME_a_indices = torch.max(Q_tgt, dim=1)

            # reshape indices
            ME_a_indices = ME_a_indices.reshape(self.batch_size, 1)

            # get variance of ME
            ME_var = torch.gather(
                Q_s2_var, dim=1, index=ME_a_indices).reshape(self.batch_size)

            # compute weights
            w = torch.zeros((self.batch_size, self.num_actions)
                            ).to(self.device)

            for a_idx in range(self.num_actions):
                u = (Q_tgt[:, a_idx] - ME_values) / \
                    torch.sqrt(Q_s2_var[:, a_idx] + ME_var)
                w[:, a_idx] = self.g(u)

            # compute weighted mean
            Q_next = torch.sum(Q_tgt * w, dim=1) / torch.sum(w, dim=1)
            Q_next = Q_next.reshape(self.batch_size, 1)

            # target
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def train(self):
        """Samples from replay_buffer, updates critic and the target networks."""

        # -------- train KEEnsembleDQN --------
        for _ in range(self.N_to_update):

            # ensemble member to update
            i = np.random.choice(self.N)

            # sample batch
            batch = self.replay_buffer.sample()

            # unpack batch
            s, a, r, s2, d = batch

            # clear gradients
            self.EnsembleDQN_optimizer[i].zero_grad()

            # Q estimates
            Q = self.EnsembleDQN[i](s)
            Q = torch.gather(input=Q, dim=1, index=a)

            # targets
            y = self._compute_target(r, s2, d, i)

            # loss
            loss = self._compute_loss(Q=Q, y=y)

            # compute gradients
            loss.backward()

            # gradient scaling and clipping
            if self.grad_rescale:
                for p in self.EnsembleDQN[i].parameters():
                    p.grad *= 1 / math.sqrt(2)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(
                    self.EnsembleDQN[i].parameters(), max_norm=10)

            # perform optimizing step
            self.EnsembleDQN_optimizer[i].step()

            # log critic training
            self.logger.store(Loss=loss.detach().cpu().numpy().item())
            self.logger.store(Q_val=Q.detach().mean().cpu().numpy().item())

        # ------- Update target networks -------
        self._target_update()


class AdaKEEnsembleDQNAgent(KEEnsembleDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameter
        self.env_max_episode_steps = c.Env.max_episode_steps
        #self.kernel_param      = torch.tensor(self.kernel_param, dtype=torch.float32, requires_grad=True, device=self.device)
        self.kernel_batch_size = getattr(c.Agent, agent_name)[
            "kernel_batch_size"]
        self.kernel_lr = getattr(c.Agent, agent_name)["kernel_lr"]

        self._set_g()

        # checks
        assert "MinAtar" in c.Env.name, "Currently, AdaKEEnsembleDQN is only available for MinAtar environments."

        # optimizer
        # if self.optimizer == "Adam":
        #    self.kernel_optimizer = optim.Adam([self.kernel_param], lr=self.kernel_lr)
        # else:
        #    self.kernel_optimizer = optim.RMSprop([self.kernel_param], lr=self.kernel_lr, alpha=0.95, centered=True, eps=0.01)

        # bounds
        if self.kernel == "test":
            self.kernel_param_l, self.kernel_param_u = 1e-6, 0.5

        elif self.kernel == "gaussian_cdf":
            self.kernel_param_l, self.kernel_param_u = 0.0, np.inf

        # new buffer since we store envs
        self.replay_buffer = buffer.UniformReplayBufferEnvs(state_type=self.state_type,
                                                            state_shape=self.state_shape,
                                                            buffer_length=self.buffer_length,
                                                            batch_size=self.batch_size,
                                                            device=self.device,
                                                            disc_actions=True)

    def _set_g(self):
        """Sets the kernel function depending on the current kernel param."""

        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(
                u, scale=self.kernel_param), dtype=torch.float32)

    def memorize(self, s, a, r, s2, d, env):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, env)

    def _target_update(self):
        if self.tgt_up_cnt % self.tgt_update_freq == 0:

            with torch.no_grad():

                # target
                for i in range(self.N):
                    self.target_EnsembleDQN[i].load_state_dict(
                        self.EnsembleDQN[i].state_dict())

                # kernel param update
                self._train_kernel()

                # update kernel function
                self._set_g()

        # increase target-update cnt
        self.tgt_up_cnt += 1

    def _train_kernel(self):
        """Updates the kernel param based on recent on-policy rollouts."""

        # perform rollouts
        s, a, MC = self._get_s_a_MC()

        # convert to tensors
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        MC = torch.tensor(MC, dtype=torch.float32)

        # estimate Q for each (s,a) pair as average of ensemble
        # forward through ensemble, creates list of length N containing torch.Size([MC_batch_size, num_actions])
        Q = [net(s).to(self.device) for net in self.EnsembleDQN]

        # gather relevant action for each ensemble net, creates list of length N containing torch.Size([MC_batch_size, 1])
        Q = [torch.gather(input=Q_net, dim=1, index=a) for Q_net in Q]

        # average over ensemble
        Q = torch.stack(Q)
        Q = self._ensemble_reduction(Q)

        # get difference term
        delta = torch.sum(MC - Q).item()

        # update kernel param
        self.kernel_param += self.kernel_lr * delta

        # clip it
        self.kernel_param = np.clip(
            self.kernel_param, self.kernel_param_l, self.kernel_param_u)

    def _get_s_a_MC(self):
        """Samples random initial env-specifications and acts greedy wrt current ensemble opinion (majority vote).

        Returns:
            s:  np.array([MC_batch_size, in_channels, height, width]))
            a:  np.array([MC_batch_size, 1]))
            MC: np.array([MC_batch_size, 1])"""

        # go greedy
        self.mode = "test"

        # s and a of ALL episodes
        s_all_eps = []
        a_all_eps = []

        # MC-vals of all (s,a) pairs of ALL episodes
        MC_ret_all_eps = []

        # init epi steps and rewards for ONE episode
        epi_steps = 0
        r_one_eps = []

        # get env and current state | Note: This selection is MinAtar specific.
        sampled_env = self.replay_buffer.sample_env()
        s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
        if self.input_norm:
            s = self.inp_normalizer.normalize(s, mode=self.mode)

        for _ in range(self.kernel_batch_size):

            epi_steps += 1

            # select action
            a = self.select_action(s)

            # perform step
            s2, r, d, _ = sampled_env.step(a)

            # save s, a, r
            s_all_eps.append(s)
            a_all_eps.append(a)
            r_one_eps.append(r)

            # potentially normalize s2
            if self.input_norm:
                s2 = self.inp_normalizer.normalize(s2, mode=self.mode)

            # s becomes s2
            s = s2

            # end of episode: for artificial time limit in env, we need to correct final reward to be a return
            if epi_steps == self.env_max_episode_steps:

                # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
                r_one_eps[-1] += self.gamma * \
                    self._greedy_action(s2, with_Q=True)[1]

            # end of episode: artificial or true done signal
            if epi_steps == self.env_max_episode_steps or d:

                # transform rewards to returns and store them
                MC_ret_all_eps += get_MC_ret_from_rew(
                    rews=r_one_eps, gamma=self.gamma)

                # reset
                epi_steps = 0
                r_one_eps = []

                # get another initial state
                sampled_env = self.replay_buffer.sample_env()
                s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
                if self.input_norm:
                    s = self.inp_normalizer.normalize(s, mode=self.mode)

        # store MC from final unfinished episode
        if len(r_one_eps) > 0:

            # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
            r_one_eps[-1] += self.gamma * \
                self._greedy_action(s2, with_Q=True)[1]

            # transform rewards to returns and store them
            MC_ret_all_eps += get_MC_ret_from_rew(
                rews=r_one_eps, gamma=self.gamma)

        # continue training
        self.mode = "train"

        return np.stack(s_all_eps), np.expand_dims(a_all_eps, 1), np.expand_dims(MC_ret_all_eps, 1)


class ComboDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):

        # Note: Since the 'dqn_weights' in the config-file
        # wouldn't fit the plain DQN, we need to artificially
        # provide a 'None' entry for them and set the mode to 'train'.
        c_cpy = copy.deepcopy(c)
        c_cpy.dqn_weights = None
        c_cpy.mode = "train"

        # now we can instantiate the parent class and correct the overwritten information, rest as usual
        super().__init__(c_cpy, agent_name)
        self.dqn_weights = c.dqn_weights
        self.mode = c.mode

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)
                    ), "Need prior weights in test mode."

        #assert self.state_type == "image", "ComboDQN is not setup to take feature vectors."

        if self.net_struc is not None:
            warnings.warn(
                "The net structure cannot be controlled via the config-spec for this agent.")

        # init DQN
        self.DQN = nets.ComboDQN(
            n_actions=self.num_actions,
            height=c.img_height,
            width=c.img_width
        )

        # Common name for param count
        self.n_params = self._count_params(self.DQN)

        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(
                self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)


class DDQNAgent(DQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name)

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            a2 = torch.argmax(self.DQN(s2), dim=1).reshape(self.batch_size, 1)
            Q_next = torch.gather(input=self.target_DQN(s2), dim=1, index=a2)

            y = r + self.gamma * Q_next * (1 - d)
        return y


class MaxMinDQNAgent(EnsembleDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

    def _ensemble_reduction(self, q_ens):
        """
        Input:  torch.Size([N, batch_size, num_actions])
        Output: torch.Size([batch_size, num_actions])
        """
        return torch.min(q_ens, dim=0).values


class RecDQNAgent(DDQNAgent):
    def __init__(self, c: ConfigFile, agent_name):

        # Note: Since the 'dqn_weights' in the config-file wouldn't fit the plain DQN,
        #       we need to artificially provide a 'None' entry for them and set the mode to 'train'.
        c_cpy = copy.copy(c)
        c_cpy.dqn_weights = None
        c_cpy.mode = "train"

        # now we can instantiate the parent class and correct the overwritten information, rest as usual
        super().__init__(c_cpy, agent_name)
        self.dqn_weights = c.dqn_weights
        self.mode = c.mode

        # checks
        assert not (self.mode == "test" and (self.dqn_weights is None)
                    ), "Need prior weights in test mode."

        assert self.state_type == "feature", "RecDQN is currently based on features."

        if self.net_struc is not None:
            warnings.warn(
                "The net structure cannot be controlled via the config-spec for LSTM-based agents.")

        # init DQN
        if self.state_type == "feature":
            self.DQN = nets.RecDQN(num_actions=self.num_actions,
                                   N_TSs=c.Env.env_kwargs["N_TSs"]).to(self.device)

        # Common name for param count
        self.n_params = self._count_params(self.DQN)

        # load prior weights if available
        if self.dqn_weights is not None:
            self.DQN.load_state_dict(torch.load(self.dqn_weights))

        # init target net and counter for target update
        self.target_DQN = copy.deepcopy(self.DQN).to(self.device)
        self.tgt_up_cnt = 0

        # freeze target nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_DQN.parameters():
            p.requires_grad = False

        # define optimizer
        if self.optimizer == "Adam":
            self.DQN_optimizer = optim.Adam(self.DQN.parameters(), lr=self.lr)
        else:
            self.DQN_optimizer = optim.RMSprop(
                self.DQN.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)


class SCDQNAgent(DQNAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)

        # attributes and hyperparameters
        self.sc_beta = getattr(c.Agent, agent_name)["sc_beta"]

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            tgt_s2 = self.target_DQN(s2)

            target_Q_beta = (1 - self.sc_beta) * tgt_s2 + \
                self.sc_beta * self.DQN(s2)
            a2 = torch.argmax(target_Q_beta, dim=1).reshape(self.batch_size, 1)

            Q_next = torch.gather(input=tgt_s2, dim=1, index=a2)
            y = r + self.gamma * Q_next * (1 - d)

        return y
