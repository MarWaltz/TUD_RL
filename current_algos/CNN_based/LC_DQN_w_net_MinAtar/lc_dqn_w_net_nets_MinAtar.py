import torch.nn as nn
import torch.nn.functional as F


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

class W_NET(nn.Module):
    """Defines a network with a convolutional first layer, a FC and a softmax layer to compute the weights for the ensemble."""
    def __init__(self, in_channels, height, width, N):
        super(W_NET, self).__init__()

        # CNN hyperparams
        self.out_channels = 8
        self.kernel_size  = 3
        self.stride       = 1
        self.padding      = 0

        # define CNN
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, \
                              stride=self.stride, padding=self.padding)

        # calculate input size for FC layer (which is size of a single feature map multiplied by number of out_channels)
        num_linear_units = self._output_size_filter(height) * self._output_size_filter(width) * self.out_channels

        # define FC layers
        self.linear1 = nn.Linear(num_linear_units, 64)
        self.linear2 = nn.Linear(64, N)
        self.softmax = nn.Softmax(dim=1)

    def _output_size_filter(self, size, kernel_size=3, stride=1):
        """Computes for given height or width (here named 'size') of ONE input channel, given
        kernel (or filter) size and stride, the resulting size (again: height or width) of the feature map.
        Note: This formula assumes padding = 0 and dilation = 1."""

        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, s):
        """Returns for a state s the vector [w_1, ..., w_N]. Args:
        s: torch.Size([batch_size, in_channels, height, width])

        returns: torch.Size([batch_size, N])
        """
        # CNN
        x = F.relu(self.conv(s))

        # reshape from torch.Size([batch_size, out_channels, out_height, out_width]) to 
        # torch.Size([batch_size, out_channels * out_height * out_width])
        x = x.view(x.shape[0], -1)

        # FC layers
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        # softmax
        w = self.softmax(x)

        return w
