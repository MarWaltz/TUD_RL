import torch
import torch.nn as nn
from common.net_activations import activations


class Actor(nn.Module):
    """Defines deterministic actor."""
    def __init__(self, action_dim, state_dim, net_struc_actor):

        super(Actor, self).__init__()

        self.struc = net_struc_actor

        assert isinstance(self.struc, list), "net should be a list,  e.g. [[64, 'relu'], [64, 'relu'], 'tanh']."
        assert len(self.struc) >= 2, "net should have at least one hidden layer and a final activation."
        assert isinstance(self.struc[-1], str), "Final element of net should only be the activation string."

        self.layers = nn.ModuleList()

        # create input-hidden_1
        self.layers.append(nn.Linear(state_dim, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

        # create hidden_n-out
        self.layers.append(nn.Linear(self.struc[-2][0], action_dim))
        
    def forward(self, s):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])
        returns: torch.Size([batch_size, action_dim])
        """

        for layer_idx, layer in enumerate(self.layers):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            s = act_f(layer(s))

        return s


class Critic(nn.Module):
    """Defines critic network to compute Q-values."""
    def __init__(self, action_dim, state_dim, net_struc_critic):
        
        super(Critic, self).__init__()

        self.struc = net_struc_critic

        assert isinstance(self.struc, list), "net should be a list,  e.g. [[64, 'relu'], [64, 'relu'], 'identity']."
        assert len(self.struc) >= 2, "net should have at least one hidden layer and a final activation."
        assert isinstance(self.struc[-1], str), "Final element of net should only be the activation string."

        self.layers = nn.ModuleList()

        # create input-hidden_1
        self.layers.append(nn.Linear(state_dim + action_dim, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

        # create hidden_n-out
        self.layers.append(nn.Linear(self.struc[-2][0], 1))

    def forward(self, s, a):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])
        a:       torch.Size([batch_size, action_dim])
        returns: torch.Size([batch_size, 1])
        """
        x = torch.cat([s, a], dim=1)

        for layer_idx, layer in enumerate(self.layers):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            x = act_f(layer(x))

        return x


class Double_Critic(nn.Module):
    """Defines two critic network to compute Q-values in TD3 algorithm."""
    def __init__(self, action_dim, state_dim, net_struc_critic):
        super(Double_Critic, self).__init__()
        
        self.struc = net_struc_critic

        assert isinstance(self.struc, list), "net should be a list,  e.g. [[64, 'relu'], [64, 'relu'], 'identity']."
        assert len(self.struc) >= 2, "net should have at least one hidden layer and a final activation."
        assert isinstance(self.struc[-1], str), "Final element of net should only be the activation string."


        #---- Q1 architecture ---- 
        self.layers1 = nn.ModuleList()

        # create input-hidden_1
        self.layers1.append(nn.Linear(state_dim + action_dim, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers1.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

        # create hidden_n-out
        self.layers1.append(nn.Linear(self.struc[-2][0], 1))


		#----- Q2 architecture ---- 
        self.layers2 = nn.ModuleList()

        # create input-hidden_1
        self.layers2.append(nn.Linear(state_dim + action_dim, self.struc[0][0]))

        # create hidden_1-...-hidden_n
        for idx in range(len(self.struc) - 2):
            self.layers2.append(nn.Linear(self.struc[idx][0], self.struc[idx+1][0]))

        # create hidden_n-out
        self.layers2.append(nn.Linear(self.struc[-2][0], 1))

    def forward(self, s, a):
        """s and a are torch tensors. Shapes:
        s:        torch.Size([batch_size, state_dim])
        a:        torch.Size([batch_size, action_dim])
        returns: (torch.Size([batch_size, 1]), torch.Size([batch_size, 1]))
        """
        x1 = torch.cat([s, a], dim=1)
        x2 = torch.cat([s, a], dim=1)

        # Q1 forward
        for layer_idx, layer in enumerate(self.layers1):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            x1 = act_f(layer(x1))

        # Q2 forward
        for layer_idx, layer in enumerate(self.layers2):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            x2 = act_f(layer(x2))

        return x1, x2

    def Q1(self, s, a):
        """As forward, but returns only Q-value of critic 1."""
        x1 = torch.cat([s, a], dim=1)
        
        for layer_idx, layer in enumerate(self.layers1):

            # get activation fnc
            if layer_idx == len(self.struc)-1:
                act_str = self.struc[layer_idx]
            else:
                act_str = self.struc[layer_idx][1]
            act_f = activations[act_str]

            # forward
            x1 = act_f(layer(x1))

        return x1
