import numpy as np
import torch


class UniformReplayBuffer:
    def __init__(self, state_type, state_shape, 
                 buffer_length, batch_size, 
                 device, disc_actions, action_dim=None):
        """A simple replay buffer with uniform sampling."""

        self.state_type  = state_type
        self.state_shape = state_shape
        self.max_size    = buffer_length
        self.batch_size  = batch_size
        self.ptr         = 0
        self.size        = 0
        self.device      = device
        
        if state_type == "image":
            self.s  = np.zeros((self.max_size, *state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, *state_shape), dtype=np.float32)

        elif state_type == "feature":
            self.s  = np.zeros((self.max_size, state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, state_shape), dtype=np.float32)
        
        if disc_actions:
            self.a = np.zeros((self.max_size, 1), dtype=np.int64)
        else:
            self.a = np.zeros((self.max_size, action_dim), dtype=np.float32)

        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        """s and s2 are np.arrays of shape (in_channels, height, width) or (state_shape,)."""
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        a:  torch.Size([batch_size, 1])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        d:  torch.Size([batch_size, 1])"""

        # sample index
        ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)

        return (torch.tensor(self.s[ind]).to(self.device), 
                torch.tensor(self.a[ind]).to(self.device), 
                torch.tensor(self.r[ind]).to(self.device), 
                torch.tensor(self.s2[ind]).to(self.device), 
                torch.tensor(self.d[ind]).to(self.device))


class UniformReplayBuffer_BootDQN(UniformReplayBuffer):
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, K, mask_p):
        super().__init__(state_type    = state_type,
                         state_shape   = state_shape, 
                         buffer_length = buffer_length, 
                         batch_size    = batch_size, 
                         device        = device,
                         disc_actions  = True)
        """A simple replay buffer with uniform sampling. Incorporates bootstrapping masks."""

        self.K          = K
        self.mask_p     = mask_p
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


class UniformReplayBuffer_LSTM(UniformReplayBuffer):
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim, history_length):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim)
        
        self.action_dim = action_dim
        self.history_length = history_length

    def sample(self) -> tuple:
        """Returns tuple of past experiences with elements:

        s_hist:    torch.Size([batch_size, history_length, state_shape])
        a_hist:    torch.Size([batch_size, history_length, action_dim])
        hist_len:  torch.Size(batch_size)

        s2_hist:   torch.Size([batch_size, history_length, state_shape])
        a2_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len2: torch.Size(batch_size)

        s:         torch.Size([batch_size, state_shape])
        a:         torch.Size([batch_size, action_dim])
        r:         torch.Size([batch_size, 1])
        s2:        torch.Size([batch_size, state_shape])
        d:         torch.Size([batch_size, 1])
        
        E.g., hist_len says how long the actual history of the respective batch element of s_hist and a_hist is. Rest is filled with zeros.
        """

        # sample indices
        bat_indices = np.random.randint(low = self.history_length, high = self.size, size = self.batch_size)
        
        # ---------- direct extraction ---------

        s  = self.s[bat_indices]
        a  = self.a[bat_indices]
        r  = self.r[bat_indices]
        s2 = self.s2[bat_indices]
        d  = self.d[bat_indices]
        
        # ---------- hist generation  --------

        # create empty histories
        s_hist   = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)
        a_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        hist_len = np.ones(self.batch_size, dtype=np.int64) * self.history_length

        # fill histories
        for i, b_idx in enumerate(bat_indices):
            
            # take data
            s_hist[i, :, :] = self.s[(b_idx - self.history_length) : b_idx, :]
            a_hist[i, :, :] = self.a[(b_idx - self.history_length) : b_idx, :]

            # truncate if done appeared
            for j in range(1, self.history_length + 1):
                
                if self.d[b_idx - j] == True:
                    
                    # set history lengths
                    hist_len[i]  = j - 1

                    # set prior entries to zero when done appeared
                    s_hist[i, : (self.history_length - j + 1) ,:]  = 0.0
                    a_hist[i, : (self.history_length - j + 1) ,:]  = 0.0

                    # move non-zero experiences to the beginning
                    s_hist[i] = np.roll(s_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    a_hist[i] = np.roll(a_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    break
        
        # ---------- hist2 generation  --------

        # create empty histories
        s2_hist   = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)
        a2_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        hist_len2 = np.ones(self.batch_size, dtype=np.int64) * self.history_length

        # fill histories
        for i, b_idx in enumerate(bat_indices):
            
            # take data
            s2_hist[i, :, :] = self.s[(b_idx - self.history_length + 1) : (b_idx + 1), :]
            a2_hist[i, :, :] = self.a[(b_idx - self.history_length + 1) : (b_idx + 1), :]
            
            # truncate if done appeared
            for j in range(1, self.history_length):
                
                if self.d[b_idx - j] == True:
                    
                    # set history lengths
                    hist_len2[i] = j

                    # set prior entries to zero when done appeared
                    s2_hist[i, : (self.history_length - j) ,:] = 0.0
                    a2_hist[i, : (self.history_length - j) ,:]  = 0.0

                    # move non-zero experiences to the beginning
                    s2_hist[i] = np.roll(s2_hist[i], shift= -(self.history_length - j), axis=0)
                    a2_hist[i] = np.roll(a2_hist[i], shift= -(self.history_length - j), axis=0)
                    break

        #print({"s": self.s, "a": self.a, "r": self.r, "s2" : self.s2, "d" : self.d})
        #print({"s_hist": s_hist, "a_hist": a_hist, "hist_len":hist_len, "s2_hist":s2_hist, "a2_hist":a2_hist,\
        #"hist_len2":hist_len2, "s":s, "a":a, "r":r, "s2": s2, "d": d, "idx": bat_indices})
        #print("--------------------------")

        return (torch.tensor(s_hist).to(self.device), 
                torch.tensor(a_hist).to(self.device), 
                torch.tensor(hist_len).to(self.device),
                torch.tensor(s2_hist).to(self.device), 
                torch.tensor(a2_hist).to(self.device), 
                torch.tensor(hist_len2).to(self.device),
                torch.tensor(s).to(self.device),
                torch.tensor(a).to(self.device),
                torch.tensor(r).to(self.device),
                torch.tensor(s2).to(self.device),
                torch.tensor(d).to(self.device))


class UniformReplayBufferEnvs(UniformReplayBuffer):
    """This buffer additionally stores a copy of the current env-object at each time step, which might be necessary when the state
    of an environment alone is not sufficient to fully characterize its internals, as, e.g., in the MinAtar environments, and one
    wants episodes starting from a random initial state in the buffer. Memory-wise this is not too expensive since a MinAtar 
    environment typically requires 48 bytes."""
    
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim=None):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim)
        self.envs = [None] * buffer_length
    
    def add(self, s, a, r, s2, d, env):
        self.envs[self.ptr] = env
        super().add(s, a, r, s2, d)

    def sample_env(self):
        ind = np.random.choice(self.size)
        return self.envs[ind]


class UniformReplayBufferEnvs_BootDQN(UniformReplayBuffer_BootDQN):
    """Corresponds to 'UniformReplayBufferEnvs' with bootstrapping masks."""

    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, K, mask_p):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, K, mask_p)
        self.envs = [None] * buffer_length
    
    def add(self, s, a, r, s2, d, env):
        self.envs[self.ptr] = env
        super().add(s, a, r, s2, d)
    
    def sample_env(self):
        ind = np.random.choice(self.size)
        return self.envs[ind]
