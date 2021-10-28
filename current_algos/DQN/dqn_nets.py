import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Defines critic network to compute Q-values."""
    def __init__(self, num_actions, state_dim, num_hid_layers, hid_size):
        super(DQN, self).__init__()

        assert num_hid_layers >= 1, "Please specify at least one hidden layer."
        
        self.layers = nn.ModuleList()

        # create input-hidden_1
        self.layers.append(nn.Linear(state_dim, hid_size))

        # create hidden_1-...-hidden_n
        for _ in range(num_hid_layers - 1):
            self.layers.append(nn.Linear(hid_size, hid_size))

        # create hidden_n-out
        self.layers.append(nn.Linear(hid_size, num_actions))

    def forward(self, s):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])

        returns: torch.Size([batch_size, num_actions])
        """

        for layer in self.layers[:-1]:
            x = F.relu(layer(s))
        q = self.layers[-1](x)

        return q

class CNN_DQN(nn.Module):
    """Defines a deep Q-network with a convolutional first layer. Suitable, e.g., for MinAtar games."""
    def __init__(self, in_channels, height, width, num_actions):
        super(CNN_DQN, self).__init__()

        # CNN hyperparams
        self.out_channels = 16
        self.kernel_size  = 3
        self.stride       = 1
        self.padding      = 0

        # define CNN
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, \
                              stride=self.stride, padding=self.padding)

        # calculate input size for FC layer (which is size of a single feature map multiplied by number of out_channels)
        num_linear_units = self._output_size_filter(height) * self._output_size_filter(width) * self.out_channels

        # define FC layers
        self.linear1 = nn.Linear(num_linear_units, 128)
        self.linear2 = nn.Linear(128, num_actions)

    def _output_size_filter(self, size, kernel_size=3, stride=1):
        """Computes for given height or width (here named 'size') of ONE input channel, given
        kernel (or filter) size and stride, the resulting size (again: height or width) of the feature map.
        Note: This formula assumes padding = 0 and dilation = 1."""

        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, s):
        """Returns for a state s all the Q(s,a). Args:
        s: torch.Size([batch_size, in_channels, height, width])

        returns: torch.Size([batch_size, num_actions])
        """
        # CNN
        x = F.relu(self.conv(s))

        # reshape from torch.Size([batch_size, out_channels, out_height, out_width]) to 
        # torch.Size([batch_size, out_channels * out_height * out_width])
        x = x.view(x.shape[0], -1)

        # FC layers
        x = F.relu(self.linear1(x))
        q = self.linear2(x)

        return q
