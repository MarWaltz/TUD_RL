import numpy as np
import torch


class UniformReplayBuffer:
    """A simple replay buffer with uniform sampling."""
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim=None):
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
        a:  torch.Size([batch_size, 1]) or torch.Size([batch_size, action_dim])
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


class UniformReplayBuffer_SRL(UniformReplayBuffer):
    """A simple replay buffer with uniform sampling."""
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim=None):
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
        self.info = [None] * self.max_size
    
    def add(self, s, a, r, s2, d, info):
        """s and s2 are np.arrays of shape (in_channels, height, width) or (state_shape,)."""
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d
        self.info[self.ptr]  = info

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        a:  torch.Size([batch_size, 1]) or torch.Size([batch_size, action_dim])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, in_channels, height, width]) or torch.Size([batch_size, state_shape])
        d:  torch.Size([batch_size, 1])"""

        # sample index
        ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)
        info = [self.info[i] for i in ind]
        
        return (torch.tensor(self.s[ind]).to(self.device), 
                torch.tensor(self.a[ind]).to(self.device), 
                torch.tensor(self.r[ind]).to(self.device), 
                torch.tensor(self.s2[ind]).to(self.device), 
                torch.tensor(self.d[ind]).to(self.device),
                info)

class MultiAgentUniformReplayBuffer(UniformReplayBuffer):
    """A simple replay buffer with uniform sampling for multi-agent scenarios."""
    def __init__(self, N_agents, state_type, state_shape, buffer_length, batch_size, device, action_dim) -> None:
        super().__init__(state_type=state_type, state_shape=state_shape, buffer_length=buffer_length, batch_size=batch_size,\
             device=device, disc_actions=False, action_dim=action_dim)

        self.N_agents = N_agents

        if state_type == "image":
            self.s  = np.zeros((self.max_size, self.N_agents, *state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, self.N_agents, *state_shape), dtype=np.float32)

        elif state_type == "feature":
            self.s  = np.zeros((self.max_size, self.N_agents, state_shape), dtype=np.float32)
            self.s2 = np.zeros((self.max_size, self.N_agents, state_shape), dtype=np.float32)
        
        self.a = np.zeros((self.max_size, self.N_agents, action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_size, self.N_agents, 1), dtype=np.float32)
        self.d = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        """Args:
            s/s2: np.arrays of shape (N_agents, in_channels, height, width) or (N_agents, state_shape)
            a:    np.array of shape (N_agents, action_dim)
            r:    np.array of shape (N_agents, 1)
            d:    bool
        """
        super().add(s, a, r, s2, d)

    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, N_agents, in_channels, height, width]) or torch.Size([batch_size, N_agents, state_shape])
        a:  torch.Size([batch_size, N_agents, action_dim])
        r:  torch.Size([batch_size, N_agents, 1])
        s2: torch.Size([batch_size, N_agents, in_channels, height, width]) or torch.Size([batch_size, N_agents, state_shape])
        d:  torch.Size([batch_size, 1])"""
        return super().sample()


class UniformReplayBuffer_BootDQN(UniformReplayBuffer):
    """A simple replay buffer with uniform sampling. Incorporates bootstrapping masks."""
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, K, mask_p):
        super().__init__(state_type    = state_type,
                         state_shape   = state_shape, 
                         buffer_length = buffer_length, 
                         batch_size    = batch_size, 
                         device        = device,
                         disc_actions  = True)
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
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, history_length, action_dim=None):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim)
        
        self.action_dim     = action_dim
        self.disc_actions   = disc_actions
        self.history_length = history_length

    def sample(self) -> tuple:
        """Returns tuple of past experiences with elements:

        s_hist:    torch.Size([batch_size, history_length, state_shape])
        a_hist:    torch.Size([batch_size, history_length, action_dim or 1])
        hist_len:  torch.Size(batch_size)

        s2_hist:   torch.Size([batch_size, history_length, state_shape])
        a2_hist:   torch.Size([batch_size, history_length, action_dim or 1])
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
        s_hist = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)
        
        if self.disc_actions:
            a_hist = np.zeros((self.batch_size, self.history_length, 1), dtype=np.int64)
        else:
            a_hist = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        
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
                    s_hist[i, : (self.history_length - j + 1) ,:] = 0.0
                    a_hist[i, : (self.history_length - j + 1) ,:] = 0

                    # move non-zero experiences to the beginning
                    s_hist[i] = np.roll(s_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    a_hist[i] = np.roll(a_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    break
        
        # ---------- hist2 generation  --------

        # create empty histories
        s2_hist   = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)

        if self.disc_actions:
            a2_hist = np.zeros((self.batch_size, self.history_length, 1), dtype=np.int64)
        else:
            a2_hist = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)

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
                    a2_hist[i, : (self.history_length - j) ,:] = 0

                    # move non-zero experiences to the beginning
                    s2_hist[i] = np.roll(s2_hist[i], shift= -(self.history_length - j), axis=0)
                    a2_hist[i] = np.roll(a2_hist[i], shift= -(self.history_length - j), axis=0)
                    break

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



class PrioritizedReplayBuffer_LSTM_SRL(UniformReplayBuffer):
    def __init__(self, state_type, state_shape, buffer_length, batch_size,
                 device, disc_actions, history_length, action_dim=None,
                 alpha=0.8, beta_start=0.4, beta_frames=100000):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim)

        self.action_dim     = action_dim
        self.disc_actions   = disc_actions
        self.history_length = history_length
        self.max_size = buffer_length

        self.info = [None] * self.max_size

        # PER-specific
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # priorities and sum tree pointers
        self.priorities = np.zeros((buffer_length,), dtype=np.float32)
        self.max_priority = 1.0



    def add(self, s, a, r, s2, d, info):
        # store transition
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.info[self.ptr] = info

        # set new priority to max so new samples are likely to be used
        self.priorities[self.ptr] = self.max_priority

        # advance pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_beta(self):
        # anneal beta to 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        return beta
    


    def sample_ranked(self):
        if self.size == 0:
            raise ValueError("The buffer is empty!")

        # 1) Compute ranks of current priorities
        p = self.priorities[:self.size]
        desc_order = np.argsort(-p)
        ranks = np.empty_like(desc_order)
        ranks[desc_order] = np.arange(1, self.size + 1)

        # 2) Build sampling distribution: 1 / rank^alpha
        probs = (1.0 / (ranks.astype(np.float32) ** self.alpha))
        probs /= probs.sum()

        # 3) Sample indices
        indices = np.random.choice(self.size, self.batch_size, p=probs)

        # 4) Importance-sampling weights
        beta = self._get_beta()
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32).reshape(-1, 1)

        # 5) Gather your transitions & histories as before
        s_batch = self.s[indices]
        a_batch = self.a[indices]
        r_batch = self.r[indices]
        s2_batch = self.s2[indices]
        d_batch = self.d[indices]
        info_batch = [self.info[i] for i in indices]

        s_hist, a_hist, hist_len = self._build_histories(indices, self.s, self.a, self.d)
        s2_hist, a2_hist, hist_len2 = self._build_histories(indices, self.s, self.a, self.d, offset=1)

        # 6) Return tensors just like your original sample()
        return (
            torch.tensor(s_hist).to(self.device),
            torch.tensor(a_hist).to(self.device),
            torch.tensor(hist_len).to(self.device),
            torch.tensor(s2_hist).to(self.device),
            torch.tensor(a2_hist).to(self.device),
            torch.tensor(hist_len2).to(self.device),
            torch.tensor(s_batch).to(self.device),
            torch.tensor(a_batch).to(self.device),
            torch.tensor(r_batch).to(self.device),
            torch.tensor(s2_batch).to(self.device),
            torch.tensor(d_batch).to(self.device),
            torch.tensor(weights).to(self.device),
            torch.tensor(indices).to(self.device),
            info_batch
        )


    def sample(self):
        # ensure we don't divide by zero
        if self.size == 0:
            raise ValueError("The buffer is empty!")

        # calculate sampling probabilities
        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        # sample indices
        indices = np.random.choice(self.size, self.batch_size, p=probs)

        # compute importance-sampling weights
        beta = self._get_beta()
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)

        # extract transitions and histories
        # base extractions
        s_batch = self.s[indices]
        a_batch = self.a[indices]
        r_batch = self.r[indices]
        s2_batch = self.s2[indices]
        d_batch = self.d[indices]
        info_batch = [self.info[i] for i in indices]

        # generate histories (similar to uniform buffer)
        s_hist, a_hist, hist_len = self._build_histories(indices, self.s, self.a, self.d)
        s2_hist, a2_hist, hist_len2 = self._build_histories(indices, self.s, self.a, self.d,
                                                             offset=1)

        # convert to tensors
        return (
            torch.tensor(s_hist).to(self.device),
            torch.tensor(a_hist).to(self.device),
            torch.tensor(hist_len).to(self.device),
            torch.tensor(s2_hist).to(self.device),
            torch.tensor(a2_hist).to(self.device),
            torch.tensor(hist_len2).to(self.device),
            torch.tensor(s_batch).to(self.device),
            torch.tensor(a_batch).to(self.device),
            torch.tensor(r_batch).to(self.device),
            torch.tensor(s2_batch).to(self.device),
            torch.tensor(d_batch).to(self.device),
            torch.tensor(weights).to(self.device),
            torch.tensor(indices).to(self.device),
            info_batch
        )

    def update_priorities_ranked(self, indices, td_errors):
        sorted_errors = sorted([(i, abs(e)) for i, e in zip(indices, td_errors)],
                       key=lambda x: -x[1])
        for rank, (idx, _) in enumerate(sorted_errors):
            priority = 1.0 / (rank + 1)  # highest rank gets highest priority
            self.priorities[idx] = priority
        

    def update_priorities(self, indices, td_errors):
        # update priorities based on latest td_errors
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def _build_histories(self, indices, s_array, a_array, d_array, offset=0):

        s_hist = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)
        
        if self.disc_actions:
            a_hist = np.zeros((self.batch_size, self.history_length, 1), dtype=np.int64)
        else:
            a_hist = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        
        hist_len = np.ones(self.batch_size, dtype=np.int64) * self.history_length



        for i, idx in enumerate(indices):
            start = idx - self.history_length + 1 - offset
            end = idx + 1 - offset
            if start < 0:
                pad = -start
                s_seq = np.vstack([np.zeros((pad, self.state_shape)),
                                   s_array[0:end]])
                a_seq = np.vstack([np.zeros((pad, a_array.shape[-1])),
                                   a_array[0:end]])
            else:
                s_seq = s_array[start:end]
                a_seq = a_array[start:end]

            # handle episode boundaries
            for j in range(len(s_seq)):
                if d_array[start + j] and j < self.history_length:
                    hist_len[i] = self.history_length - j - offset
                    s_seq[:j+1-offset] = 0
                    a_seq[:j+1-offset] = 0
                    s_seq = np.roll(s_seq, shift=-(j+1-offset), axis=0)
                    a_seq = np.roll(a_seq, shift=-(j+1-offset), axis=0)
                    break

            s_hist[i] = s_seq
            a_hist[i] = a_seq

        return s_hist, a_hist, hist_len



class UniformReplayBuffer_LSTM_SRL(UniformReplayBuffer):
    def __init__(self, state_type, state_shape, buffer_length, batch_size, device, disc_actions, history_length, action_dim=None):
        super().__init__(state_type, state_shape, buffer_length, batch_size, device, disc_actions, action_dim)
        
        self.action_dim     = action_dim
        self.disc_actions   = disc_actions
        self.history_length = history_length
        self.info = [None] * self.max_size


    def add(self, s, a, r, s2, d, info):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d
        self.info[self.ptr]  = info

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        

    def sample(self) -> tuple:
        """Returns tuple of past experiences with elements:

        s_hist:    torch.Size([batch_size, history_length, state_shape])
        a_hist:    torch.Size([batch_size, history_length, action_dim or 1])
        hist_len:  torch.Size(batch_size)

        s2_hist:   torch.Size([batch_size, history_length, state_shape])
        a2_hist:   torch.Size([batch_size, history_length, action_dim or 1])
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
        info = [self.info[i] for i in bat_indices]
        
        # ---------- hist generation  --------

        # create empty histories
        s_hist = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)
        
        if self.disc_actions:
            a_hist = np.zeros((self.batch_size, self.history_length, 1), dtype=np.int64)
        else:
            a_hist = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        
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
                    s_hist[i, : (self.history_length - j + 1) ,:] = 0.0
                    a_hist[i, : (self.history_length - j + 1) ,:] = 0

                    # move non-zero experiences to the beginning
                    s_hist[i] = np.roll(s_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    a_hist[i] = np.roll(a_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    break
        
        # ---------- hist2 generation  --------

        # create empty histories
        s2_hist   = np.zeros((self.batch_size, self.history_length, self.state_shape), dtype=np.float32)

        if self.disc_actions:
            a2_hist = np.zeros((self.batch_size, self.history_length, 1), dtype=np.int64)
        else:
            a2_hist = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)

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
                    a2_hist[i, : (self.history_length - j) ,:] = 0

                    # move non-zero experiences to the beginning
                    s2_hist[i] = np.roll(s2_hist[i], shift= -(self.history_length - j), axis=0)
                    a2_hist[i] = np.roll(a2_hist[i], shift= -(self.history_length - j), axis=0)
                    break

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
                torch.tensor(d).to(self.device),
                info)


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
