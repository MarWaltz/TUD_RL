import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Defines deterministic actor."""
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_dim)
        
    def forward(self, s):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])
        returns: torch.Size([batch_size, action_dim])
        """
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

class Critic(nn.Module):
    """Defines critic network to compute Q-values."""
    def __init__(self, action_dim, state_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, s, a):
        """s is a torch tensor. Shapes:
        s:       torch.Size([batch_size, state_dim])
        a:       torch.Size([batch_size, action_dim])
        returns: torch.Size([batch_size, 1])
        """
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Double_Critic(nn.Module):
    """Defines two critic network to compute Q-values in TD3 algorithm."""
    def __init__(self, action_dim, state_dim):
        super(Double_Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

		# Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, s, a):
        """s and a are torch tensors. Shapes:
        s:        torch.Size([batch_size, state_dim])
        a:        torch.Size([batch_size, action_dim])
        returns: (torch.Size([batch_size, 1]), torch.Size([batch_size, 1]))
        """
        sa = torch.cat([s, a], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, s, a):
        """As forward, but returns only Q-value of critic 1."""
        sa = torch.cat([s, a], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
