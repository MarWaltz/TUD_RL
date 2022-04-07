import scipy.stats
import torch

import tud_rl.common.buffer as buffer

from tud_rl.agents.discrete.KEBootDQN import KEBootDQNAgent
from tud_rl.common.configparser import ConfigFile
from tud_rl.common.helper_fnc import get_MC_ret_from_rew
from tud_rl.common.logging_func import *


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