import numpy as np
import torch


class UniformReplayBuffer:
    def __init__(self, action_dim, state_dim, gamma, buffer_length, batch_size, device):
        """A simple replay buffer with uniform sampling."""

        self.max_size   = buffer_length
        self.gamma      = gamma
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device
        
        self.s  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        """s and s2 are np.arrays of shape (state_dim,)."""
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, state_dim])
        a:  torch.Size([batch_size, action_dim])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, state_dim])
        d:  torch.Size([batch_size, 1])"""

        # sample index
        ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)

        return (torch.tensor(self.s[ind]).to(self.device), 
                torch.tensor(self.a[ind]).to(self.device), 
                torch.tensor(self.r[ind]).to(self.device), 
                torch.tensor(self.s2[ind]).to(self.device), 
                torch.tensor(self.d[ind]).to(self.device))
