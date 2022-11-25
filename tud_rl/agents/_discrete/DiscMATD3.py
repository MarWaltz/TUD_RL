import torch

from tud_rl.agents._continuous.MATD3 import MATD3Agent
from tud_rl.common.configparser import ConfigFile


class DiscMATD3Agent(MATD3Agent):
    """MATD3 agent for discrete action spaces based on the Gumbel-Softmax trick."""
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name)
        self.is_continuous = False

    @torch.no_grad()
    def select_action(self, s):
        """Selects action via actor network for a given state. Adds exploration bonus from noise and clips to action scale.
        Arg s:   np.array with shape (N_agents, state_shape)
        returns: np.array with shape (N_agents,)
        """
        a = torch.zeros((self.N_agents, self.num_actions), dtype=torch.float32).to(self.device)

        for i in range(self.N_agents):

            # reshape obs (namely, to torch.Size([1, state_shape]))
            s_i = torch.tensor(s[i], dtype=torch.float32).unsqueeze(0).to(self.device)

            # forward pass
            a_i = self.actor[i](s_i)

            # gumbel-softmax in exploration, hard one-hot in testing
            if self.mode == "train":
                a[i] = self._gumbel_softmax(a_i, hard=True)
            else:
                a[i] = self._onehot(a_i)

        # go to integers instead of one-hot for env compatibility
        a = a.cpu().numpy()
        return self._onehot_to_int(a)

    def memorize(self, s, a, r, s2, d):
        """Stores current transition in replay buffer. Transform integer actions to one-hot encodings"""
        a = self._int_to_onehot(a)
        self.replay_buffer.add(s, a, r, s2, d)
