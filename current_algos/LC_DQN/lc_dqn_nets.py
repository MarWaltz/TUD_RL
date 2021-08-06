import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Defines critic network to compute Q-values."""
    def __init__(self, num_actions, state_dim):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, num_actions)

    def forward(self, s):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])

        returns: torch.Size([batch_size, num_actions])
        """

        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        q = self.linear3(x)
        return q
