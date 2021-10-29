import numpy as np
import torch

class UniformReplayBuffer:
    def __init__(self, action_dim, obs_dim, gamma, history_length, buffer_length, batch_size, device) -> None:

        self.action_dim     = action_dim
        self.obs_dim        = obs_dim
        self.gamma          = gamma
        self.history_length = history_length
        self.buffer_length  = buffer_length
        self.batch_size     = batch_size
        self.device         = device
        
        self.ptr  = 0
        self.size = 0
        
        self.o_buff  = np.zeros((buffer_length, obs_dim), dtype=np.float32)
        self.a_buff  = np.zeros((buffer_length, action_dim), dtype=np.float32)
        self.r_buff  = np.zeros((buffer_length, 1), dtype=np.float32)
        self.o2_buff = np.zeros((buffer_length, obs_dim), dtype=np.float32)
        self.d_buff  = np.zeros((buffer_length, 1), dtype=np.float32)
    
    def add(self, o, a, r, o2, d) -> None:
        """Add transistions to buffer.
        
        o:  np.array of shape (obs_dim,)
        a:  float32/float64
        r:  float64
        o2: np.array of shape (obs_dim)
        d:  bool
        """
        
        self.o_buff[self.ptr]  = o
        self.a_buff[self.ptr]  = a
        self.r_buff[self.ptr]  = r
        self.o2_buff[self.ptr] = o2
        self.d_buff[self.ptr]  = d

        self.ptr  = (self.ptr + 1) % self.buffer_length
        self.size = min(self.size + 1, self.buffer_length)
    
    def sample(self) -> tuple:
        """Returns tuple of past experiences with elements:

        o_hist:    torch.Size([batch_size, history_length, obs_dim])
        a_hist:    torch.Size([batch_size, history_length, action_dim])
        hist_len:  torch.Size(batch_size)

        o2_hist:   torch.Size([batch_size, history_length, obs_dim])
        a2_hist:   torch.Size([batch_size, history_length, action_dim])
        hist_len2: torch.Size(batch_size)

        o:         torch.Size([batch_size, obs_dim])
        a:         torch.Size([batch_size, action_dim])
        r:         torch.Size([batch_size, 1])
        o2:        torch.Size([batch_size, obs_dim])
        d:         torch.Size([batch_size, 1])
        
        E.g., hist_len says how long the actual history of the respective batch element of o_hist and a_hist is. Rest is filled with zeros.
        """

        # sample indices
        bat_indices = np.random.randint(low = self.history_length, high = self.size, size = self.batch_size)
        
        # ---------- direct extraction ---------

        o  = self.o_buff[bat_indices]
        a  = self.a_buff[bat_indices]
        r  = self.r_buff[bat_indices]
        o2 = self.o2_buff[bat_indices]
        d = self.d_buff[bat_indices]
        
        # ---------- hist generation  --------

        # create empty histories
        o_hist   = np.zeros((self.batch_size, self.history_length, self.obs_dim), dtype=np.float32)
        a_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        hist_len = np.ones(self.batch_size, dtype=np.int64) * self.history_length

        # fill histories
        for i, b_idx in enumerate(bat_indices):
            
            # take data
            o_hist[i, :, :] = self.o_buff[(b_idx - self.history_length) : b_idx, :]
            a_hist[i, :, :] = self.a_buff[(b_idx - self.history_length) : b_idx, :]

            # truncate if done appeared
            for j in range(1, self.history_length + 1):
                
                if self.d_buff[b_idx - j] == True:
                    
                    # set history lengths
                    hist_len[i]  = j - 1

                    # set prior entries to zero when done appeared
                    o_hist[i, : (self.history_length - j + 1) ,:]  = 0.0
                    a_hist[i, : (self.history_length - j + 1) ,:]  = 0.0

                    # move non-zero experiences to the beginning
                    o_hist[i] = np.roll(o_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    a_hist[i] = np.roll(a_hist[i], shift = -(self.history_length - j + 1), axis=0)
                    break
        
        # ---------- hist2 generation  --------

        # create empty histories
        o2_hist   = np.zeros((self.batch_size, self.history_length, self.obs_dim), dtype=np.float32)
        a2_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
        hist_len2 = np.ones(self.batch_size, dtype=np.int64) * self.history_length

        # fill histories
        for i, b_idx in enumerate(bat_indices):
            
            # take data
            o2_hist[i, :, :] = self.o_buff[(b_idx - self.history_length + 1) : (b_idx + 1), :]
            a2_hist[i, :, :] = self.a_buff[(b_idx - self.history_length + 1) : (b_idx + 1), :]
            
            # truncate if done appeared
            for j in range(1, self.history_length):
                
                if self.d_buff[b_idx - j] == True:
                    
                    # set history lengths
                    hist_len2[i] = j

                    # set prior entries to zero when done appeared
                    o2_hist[i, : (self.history_length - j) ,:] = 0.0
                    a2_hist[i, : (self.history_length - j) ,:]  = 0.0

                    # move non-zero experiences to the beginning
                    o2_hist[i] = np.roll(o2_hist[i], shift= -(self.history_length - j), axis=0)
                    a2_hist[i] = np.roll(a2_hist[i], shift= -(self.history_length - j), axis=0)
                    break

        #print({"o_buff":self.o_buff, "a_buff":self.a_buff, "r_buff":self.r_buff, "o2_buff":self.o2_buff, "d_buff":self.d_buff})
        #print({"o_hist": o_hist, "a_hist": a_hist, "hist_len":hist_len, "o2_hist":o2_hist, "a2_hist":a2_hist,\
        #"hist_len2":hist_len2, "o":o, "a":a, "r":r, "o2": o2, "d": d, "idx": bat_indices})
        #print("--------------------------")

        return (torch.tensor(o_hist).to(self.device), 
                torch.tensor(a_hist).to(self.device), 
                torch.tensor(hist_len).to(self.device),
                torch.tensor(o2_hist).to(self.device), 
                torch.tensor(a2_hist).to(self.device), 
                torch.tensor(hist_len2).to(self.device),
                torch.tensor(o).to(self.device),
                torch.tensor(a).to(self.device),
                torch.tensor(r).to(self.device),
                torch.tensor(o2).to(self.device),
                torch.tensor(d).to(self.device))
