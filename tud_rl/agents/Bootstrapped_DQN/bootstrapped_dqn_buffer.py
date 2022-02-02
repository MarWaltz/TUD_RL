import numpy as np
import torch


class UniformReplayBuffer_Bootstrapped_DQN:
    def __init__(self, state_type, state_shape, K, mask_p, buffer_length, batch_size, device):
        """A simple replay buffer with uniform sampling. Incorporates bootstrapping masks."""

        self.max_size   = buffer_length
        self.K          = K
        self.mask_p     = mask_p
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device

        if state_type == "image":
            self.s  = np.zeros((self.max_size, *state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, *state_shape), dtype=np.float32)

        elif state_type == "feature":
            self.s  = np.zeros((self.max_size, state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, state_shape), dtype=np.float32)

        self.a  = np.zeros((self.max_size, 1), dtype=np.int64)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.m  = np.zeros((self.max_size, K), dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        """s and s2 are np.arrays of shape (in_channels, height, width)  or (state_shape,)."""
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        while True:
            m = np.random.binomial(1, self.mask_p, size=self.K)
            if 1 in m:
                break
        self.m[self.ptr] = m

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        a:  torch.Size([batch_size, 1])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        d:  torch.Size([batch_size, 1])
        m:  torch.Size([batch_size, K])"""

        # sample index
        ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)

        return (torch.tensor(self.s[ind]).to(self.device), 
                torch.tensor(self.a[ind]).to(self.device), 
                torch.tensor(self.r[ind]).to(self.device), 
                torch.tensor(self.s2[ind]).to(self.device), 
                torch.tensor(self.d[ind]).to(self.device),
                torch.tensor(self.m[ind]).to(self.device))
