import torch
import torch.nn as nn
import torch.nn.functional as F
#from common.net_activations import activations




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
