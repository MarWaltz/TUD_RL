import torch
import torch.optim as optim
from tud_rl.agents.discrete.DQN import DQNAgent
from tud_rl.common.helper_fnc import get_MC_ret_from_rew
from tud_rl.common.logging_func import *
from tud_rl.common.nets import MLP, MinAtar_DQN


class TRYDQNAgent(DQNAgent):
    def __init__(self, c, agent_name):
        super().__init__(c, agent_name, logging=True)

        # attributes and hyperparameters
        self.env_max_episode_steps = c["env"]["max_episode_steps"]
        
        self.MC_batch_size = c["agent"][agent_name]["MC_batch_size"]
        self.MC_num_upd    = c["agent"][agent_name]["MC_num_upd"]

        # init bias net
        if self.state_type == "image":
            self.bias_net = MinAtar_DQN(in_channels = self.state_shape[0],
                                        height      = self.state_shape[1],
                                        width       = self.state_shape[2],
                                        num_actions = self.num_actions).to(self.device)

        elif self.state_type == "feature":
            self.bias_net = MLP(in_size   = self.state_shape,
                                out_size  = self.num_actions, 
                                net_struc = self.net_struc).to(self.device)

        # define optimizer
        if self.optimizer == "Adam":
            self.bias_optimizer = optim.Adam(self.bias_net.parameters(), lr=self.lr)
        else:
            self.bias_optimizer = optim.RMSprop(self.bias_net.parameters(), lr=self.lr, alpha=0.95, centered=True, eps=0.01)

    def greedy_action_Q(self, s):
        # reshape obs (namely, to torch.Size([1, in_channels, height, width]) or torch.Size([1, state_shape]))
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0).to(self.device)

        # forward pass
        q = self.DQN(s).to(self.device)

        # greedy
        return torch.max(q).item()

    def _compute_target(self, r, s2, d):
        with torch.no_grad():
            Q_next = self.target_DQN(s2) - self.bias_net(s2)
            Q_next = torch.max(Q_next, dim=1).values.reshape(self.batch_size, 1)
            y = r + self.gamma * Q_next * (1 - d)
        return y

    def _target_update(self):

        if self.tgt_up_cnt % self.tgt_update_freq == 0:
            with torch.no_grad():
                self.target_DQN.load_state_dict(self.DQN.state_dict())

            for _ in range(self.MC_num_upd):
                self._train_bias_net()

        # increase target-update cnt
        self.tgt_up_cnt += 1

    def _train_bias_net(self):
        """Updates the bias network based on recent on-policy rollouts.

        Args:
            s:  np.array([MC_batch_size, in_channels, height, width]))
            a:  np.array([MC_batch_size, 1]))
            MC: np.array([MC_batch_size, 1])
        """
        # perform rollouts
        s, a, MC = self._get_s_a_MC()

        # convert to tensors
        s  = torch.tensor(s.astype(np.float32))
        a  = torch.tensor(a.astype(np.int64))
        MC = torch.tensor(MC.astype(np.float32))

        # clear gradients
        self.bias_optimizer.zero_grad()

        # bias estimate
        B = self.bias_net(s)
        B = torch.gather(input=B, dim=1, index=a)

        # get target
        with torch.no_grad():

            Q = self.target_DQN(s)
            Q = torch.gather(input=Q, dim=1, index=a)

            y_bias = Q - MC
        
        # loss
        bias_loss = self._compute_loss(B, y_bias)

        # compute gradients
        bias_loss.backward()

        # perform optimizing step
        self.bias_optimizer.step()

        # log critic training
        self.logger.store(Bias_loss=bias_loss.detach().cpu().numpy().item())
        self.logger.store(Bias_val=B.detach().mean().cpu().numpy().item())

    def _get_s_a_MC(self):

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

        # get initial state
        s = self.MC_env.reset()
        if self.input_norm:
            s = self.inp_normalizer.normalize(s, mode=self.mode)

        for _ in range(self.MC_batch_size):

            epi_steps += 1

            # select action
            a = self.select_action(s)

            # perform step
            s2, r, d, _ = self.MC_env.step(a)

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
                r_one_eps[-1] += self.gamma * self.greedy_action_Q(s2)

            # end of episode: artificial or true done signal
            if epi_steps == self.env_max_episode_steps or d:

                # transform rewards to returns and store them
                MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=self.gamma)

                # reset
                epi_steps = 0
                r_one_eps = []

                # get initial state
                s = self.MC_env.reset()
                if self.input_norm:
                    s = self.inp_normalizer.normalize(s, mode=self.mode)

        # store MC from final unfinished episode
        if len(r_one_eps) > 0:

            # backup from current Q-net: r + gamma * Q(s2, pi(s2)) with greedy pi
            r_one_eps[-1] += self.gamma * self.greedy_action_Q(s2)

            # transform rewards to returns and store them
            MC_ret_all_eps += get_MC_ret_from_rew(rews=r_one_eps, gamma=self.gamma)

        # continue training
        self.mode = "train"

        return np.stack(s_all_eps), np.expand_dims(a_all_eps, 1), np.expand_dims(MC_ret_all_eps, 1)
