import scipy.stats
import torch
from tud_rl.agents.discrete.KEBootDQN import KEBootDQNAgent
from tud_rl.common.buffer import UniformReplayBufferEnvs_BootDQN
from tud_rl.common.helper_fnc import get_MC_ret_from_rew
from tud_rl.common.logging_func import *


class AdaKEBootDQNAgent(KEBootDQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True)      

        # attributes and hyperparameter
        self.kernel_batch_size     = c["agent"][agent_name]["kernel_batch_size"]
        self.kernel_lr             = c["agent"][agent_name]["kernel_lr"]

        self._set_g()

        # checks
        assert self.kernel == "test", "Currently, AdaKEBootDQN is only available for adjusting the significance level of the TE."
        assert "MinAtar" in c["env"]["name"], "Currently, AdaKEBootDQN is only available for MinAtar environments."

        # bounds
        if self.kernel == "test":
            self.kernel_param_l, self.kernel_param_u = 1e-6, 0.5

        elif self.kernel == "gaussian_cdf":
            self.kernel_param_l, self.kernel_param_u = 0.0, np.inf

        # new buffer since we store envs
        self.replay_buffer = UniformReplayBufferEnvs_BootDQN(state_type    = self.state_type, 
                                                             state_shape   = self.state_shape,
                                                             buffer_length = self.buffer_length, 
                                                             batch_size    = self.batch_size, 
                                                             device        = self.device,
                                                             K             = self.K, 
                                                             mask_p        = self.mask_p)

    def _set_g(self):
        """Sets the kernel function depending on the current kernel param."""

        if self.kernel == "test":
            self.critical_value = scipy.stats.norm().ppf(self.kernel_param)
            self.g = lambda u: (u >= self.critical_value) + 0.0

        elif self.kernel == "gaussian_cdf":
            self.g = lambda u: torch.tensor(scipy.stats.norm.cdf(u, scale=self.kernel_param), dtype=torch.float32)


    def memorize(self, s, a, r, s2, d, env):
        """Stores current transition in replay buffer."""
        self.replay_buffer.add(s, a, r, s2, d, env)


    @torch.no_grad()
    def _target_update(self):
        if self.tgt_up_cnt % self.tgt_update_freq == 0:

            # target
            self.target_DQN.load_state_dict(self.DQN.state_dict())

            # get delta's between Q and MC rollouts
            delta = 0.0
            for k in range(self.K):
                delta += self._get_MC_Q_delta(k)
            delta /= self.K

            # update kernel param
            self._upd_kernel_param(delta)

            # update kernel function
            self._set_g()

        # increase target-update cnt
        self.tgt_up_cnt += 1


    def _upd_kernel_param(self, delta):
        """Updates the kernel param based on a delta.
        Args:
            delta (float): Difference between MC-rollouts and estimated Q's."""

        # update kernel param
        self.kernel_param += self.kernel_lr * delta

        # clip it
        self.kernel_param = np.clip(self.kernel_param, self.kernel_param_l, self.kernel_param_u)


    def _get_MC_Q_delta(self, k):
        """Samples random initial env-specifications and acts greedy wrt k-th bootstrap head. Computes the delta between
        the actual observed MC return and the estimated Q-values for each encountered (s,a) pair.

        Args:
            k (int): Index of the bootstrap head serving as an update basis

        Returns:
            Delta as a float."""

        # MC-vals, Q-vals of all (s,a) pairs of ALL episodes
        MC_ret_all_eps, Q_all_eps = [], []

        # init rewards for ONE episode
        r_one_eps = []

        # get env and current state | Note: This selection is MinAtar specific.
        sampled_env = self.replay_buffer.sample_env()
        s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
        if self.input_norm:
            s = self.inp_normalizer.normalize(s, mode=self.mode)

        # main loop
        for _ in range(self.kernel_batch_size):

            # get action and Q
            a, Q = self._greedy_action(s, k, with_Q=True)
            Q_all_eps.append(Q)

            # perform step
            s2, r, d, _ = sampled_env.step(a)

            # save r
            r_one_eps.append(r)

            # potentially normalize s2
            if self.input_norm:
                s2 = self.inp_normalizer.normalize(s2, mode=self.mode)

            # s becomes s2
            s = s2

            # end of episode
            if d:

                # when done signal is artifical, correct final reward to be a return | Note: Again, MinAtar specific.
                if sampled_env.env.game.game_name() == "freeway":
                    if sampled_env.env.game.env.terminate_timer < 0:
                    
                        # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
                        r_one_eps[-1] += self.gamma * self._greedy_action(s2, active_head=k, with_Q=True)[1]

                # transform rewards to returns and store them
                MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=self.gamma)

                # reset
                r_one_eps = []

                # get another initial state
                sampled_env = self.replay_buffer.sample_env()
                s = np.moveaxis(sampled_env.game.env.state(), -1, 0)
                if self.input_norm:
                    s = self.inp_normalizer.normalize(s, mode=self.mode)

        # store MC from final unfinished episode
        if len(r_one_eps) > 0:

            # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
            r_one_eps[-1] += self.gamma * self._greedy_action(s2, active_head=k, with_Q=True)[1]

            # transform rewards to returns and store them
            MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=self.gamma)

        return sum(np.array(MC_ret_all_eps) - np.array(Q_all_eps))
