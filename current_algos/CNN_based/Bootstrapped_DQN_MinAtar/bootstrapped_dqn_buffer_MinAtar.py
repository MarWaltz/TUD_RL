import numpy as np
import torch


class UniformReplayBuffer_Bootstrap_CNN:
    def __init__(self, state_shape, n_steps, gamma, K, mask_p, buffer_length, batch_size, device):
        """A simple replay buffer with uniform sampling. Incorporates bootstrapping masks."""

        self.max_size   = buffer_length
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.K          = K
        self.mask_p     = mask_p
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device
        
        self.s  = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.a  = np.zeros((self.max_size, 1), dtype=np.int64)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.m  = np.zeros((self.max_size, K), dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        """s and s2 are np.arrays of shape (in_channels, height, width)."""
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
        s:  torch.Size([batch_size, in_channels, height, width])
        a:  torch.Size([batch_size, 1])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, in_channels, height, width])
        d:  torch.Size([batch_size, 1])
        m:  torch.Size([batch_size, K])"""

        if self.n_steps == 1:
            # sample index
            ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)

            return (torch.tensor(self.s[ind]).to(self.device), 
                    torch.tensor(self.a[ind]).to(self.device), 
                    torch.tensor(self.r[ind]).to(self.device), 
                    torch.tensor(self.s2[ind]).to(self.device), 
                    torch.tensor(self.d[ind]).to(self.device),
                    torch.tensor(self.m[ind]).to(self.device))
        else:
            # sample index
            ind = np.random.randint(low = 0, high = self.size - (self.n_steps - 1), size = self.batch_size)

            # get s, a
            s = self.s[ind]
            a = self.a[ind]

            # get s', d
            s_n = self.s2[ind + (self.n_steps - 1)]
            d   = self.d[ind + (self.n_steps - 1)]

            # compute reward part of n-step return
            r_n = np.zeros((self.batch_size, 1), dtype=np.float32)

            for i, idx in enumerate(ind):
                for j in range(self.n_steps):
                    
                    # add discounted reward
                    r_n[i] += (self.gamma ** j) * self.r[idx + j]
                    
                    # if done appears, break and set done which will be returned True (to avoid incorrect Q addition)
                    if self.d[idx + j] == 1:
                        d[i] = 1
                        break

            return (torch.tensor(s).to(self.device), 
                    torch.tensor(a).to(self.device), 
                    torch.tensor(r_n).to(self.device), 
                    torch.tensor(s_n).to(self.device), 
                    torch.tensor(d).to(self.device))
