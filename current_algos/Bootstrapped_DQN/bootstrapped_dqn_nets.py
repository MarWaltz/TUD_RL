import torch
import torch.nn as nn
import torch.nn.functional as F
from common.net_activations import activations

# --------------- CNN-based nets ---------------------

class CNN_HeadNet(nn.Module):
    """Defines a single head for the Bootstrapped DQN architecture."""
    def __init__(self, in_size_FC, num_actions):
        super().__init__()

        self.linear1 = nn.Linear(in_size_FC, 128)
        self.linear2 = nn.Linear(128, num_actions)

    def forward(self, x):
        """x : torch.Size([batch_size, in_size_FC])
        
        returns: torch.Size([batch_size, num_actions])"""

        x = F.relu(self.linear1(x))
        q = self.linear2(x)

        return q


class CNN_CoreNet(nn.Module):
    """Defines the CNN part of a Bootstrapped DQN. Suitable, e.g., for MinAtar games."""
    def __init__(self, in_channels, height, width):
        super(CNN_CoreNet, self).__init__()

        # CNN hyperparams
        self.out_channels = 16
        self.kernel_size  = 3
        self.stride       = 1
        self.padding      = 0

        # define CNN
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, \
                              stride=self.stride, padding=self.padding)

        # calculate input size for FC layer (which is size of a single feature map multiplied by number of out_channels)
        self.in_size_FC = self._output_size_filter(height) * self._output_size_filter(width) * self.out_channels

    def _output_size_filter(self, size, kernel_size=3, stride=1):
        """Computes for given height or width (here named 'size') of ONE input channel, given
        kernel (or filter) size and stride, the resulting size (again: height or width) of the feature map.
        Note: This formula assumes padding = 0 and dilation = 1."""

        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, s):
        """s: torch.Size([batch_size, in_channels, height, width])

        returns: torch.Size([batch_size, in_size_FC])
        """
        # CNN
        x = F.relu(self.conv(s))

        # reshape from torch.Size([batch_size, out_channels, out_height, out_width]) to 
        # torch.Size([batch_size, out_channels * out_height * out_width])
        x = x.view(x.shape[0], -1)

        return x


class CNN_Bootstrapped_DQN(nn.Module):
    """Defines the Bootstrapped DQN consisting of the common CNN part and K different heads."""
    def __init__(self, in_channels, height, width, num_actions, K):
        super().__init__()
        
        self.core = CNN_CoreNet(in_channels=in_channels, height=height, width=width)
        self.heads = nn.ModuleList([CNN_HeadNet(in_size_FC=self.core.in_size_FC, num_actions=num_actions) for _ in range(K)])

    def forward(self, s, head=None):
        """Returns for a state s all Q(s,a) for each k. Args:
        s: torch.Size([batch_size, in_channels, height, width])

        returns:
        list of length K with each element being torch.Size([batch_size, num_actions]) if head is None,
        torch.Size([batch_size, num_actions]) else."""

        # CNN part
        x = self.core(s)

        # K heads
        if head is None:
            return [head_net(x) for head_net in self.heads]
        else:
            return self.heads[head](x)


# --------------- Dense nets ---------------------

class HeadNet(nn.Module):
    """Defines a single head for the Bootstrapped DQN architecture."""
    def __init__(self, num_actions, net_struc_dqn):
        super().__init__()

        self.struc = net_struc_dqn

        self.linear1 = nn.Linear(net_struc_dqn[-2][0], num_actions)

    def forward(self, x):
        """x : torch.Size([batch_size, hid_size])
        
        returns: torch.Size([batch_size, num_actions])"""

        act_str = self.struc[-1]
        act_f = activations[act_str]

        q = act_f(self.linear1(x))

        return q


class CoreNet(nn.Module):
    """Defines the core for the Bootstrapped DQN architecture."""
    def __init__(self, state_dim, net_struc_dqn):
        super().__init__()

        self.struc = net_struc_dqn      
        self.layers = nn.ModuleList()

        # create input-hidden_1
        self.layers.append(nn.Linear(state_dim, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

    def forward(self, x):
        """x : torch.Size([batch_size, state_dim])
        
        returns: torch.Size([batch_size, out_size])"""
        
        for layer_idx, layer in enumerate(self.layers):

            # get activation fnc
            act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            x = act_f(layer(x))

        return x


class Bootstrapped_DQN(nn.Module):
    """Defines the Bootstrapped DQN consisting of a common part and K different heads."""
    def __init__(self, state_dim, num_actions, K, net_struc_dqn):
        super().__init__()

        assert isinstance(net_struc_dqn, list), "net should be a list,  e.g. [[64, 'relu'], [64, 'relu'], 'identity']."
        assert len(net_struc_dqn) >= 2, "net should have at least one hidden layer and a final activation."
        assert isinstance(net_struc_dqn[-1], str), "Final element of net should only be the activation string."
        
        self.core = CoreNet(state_dim=state_dim, net_struc_dqn=net_struc_dqn)
        self.heads = nn.ModuleList([HeadNet(num_actions=num_actions, net_struc_dqn=net_struc_dqn) for _ in range(K)])

    def forward(self, s, head=None):
        """Returns for a state s all Q(s,a) for each k. Args:
        s: torch.Size([batch_size, state_dim])

        returns:
        list of length K with each element being torch.Size([batch_size, num_actions]) if head is None,
        torch.Size([batch_size, num_actions]) else."""

        # common part
        x = self.core(s)

        # K heads
        if head is None:
            return [head_net(x) for head_net in self.heads]
        else:
            return self.heads[head](x)