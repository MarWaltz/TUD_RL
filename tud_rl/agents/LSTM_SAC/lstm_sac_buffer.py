import numpy as np
import torch

class UniformReplayBuffer:
    def __init__(self, action_dim, obs_dim, gamma, history_length, buffer_length, batch_size, device, n_steps=1) -> None:

        self.action_dim     = action_dim
        self.obs_dim        = obs_dim
        self.gamma          = gamma
        self.history_length = history_length
        self.buffer_length  = buffer_length
        self.batch_size     = batch_size
        self.device         = device
        self.n_steps        = n_steps
        
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
        if self.n_steps == 1:

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

        else:

            # sample indices
            bat_indices = np.random.randint(low = self.history_length, high = self.size - (self.n_steps - 1), size = self.batch_size)

            # ---------- direct extraction ---------

            # get o, a
            o  = self.o_buff[bat_indices]
            a  = self.a_buff[bat_indices]
            
            # get o', d
            o2 = self.o2_buff[bat_indices + (self.n_steps - 1)]
            d = self.d_buff[bat_indices + (self.n_steps - 1)]

            # ---------- hist generation  --------

            # create zero o_hist, a_hist
            o_hist   = np.zeros((self.batch_size, self.history_length, self.obs_dim), dtype=np.float32)
            a_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
            hist_len = np.ones(self.batch_size, dtype=np.int64) * self.history_length

            # fill o_hist, a_hist
            for i, b_idx in enumerate(bat_indices):

                # take data
                o_hist[i, :, :] = self.o_buff[(b_idx - self.history_length) : b_idx, :]
                a_hist[i, :, :] = self.a_buff[(b_idx - self.history_length) : b_idx, :]

                # truncate if done appeared
                for j in range(1, self.history_length + 1):
                    
                    if self.d_buff[b_idx - j] == True:
                        
                        # set history length
                        hist_len[i] = j - 1

                        # set prior entries to zero when done appeared
                        o_hist[i, : (self.history_length - j + 1) ,:]  = 0.0
                        a_hist[i, : (self.history_length - j + 1) ,:]  = 0.0
                        
                        # move non-zero experiences to the beginning
                        o_hist[i] = np.roll(o_hist[i], shift = -(self.history_length - j + 1), axis=0)
                        a_hist[i] = np.roll(a_hist[i], shift = -(self.history_length - j + 1), axis=0)

                        break

            # ---------- hist2 generation  --------

            # create zero o2_hist, a2_hist
            o2_hist   = np.zeros((self.batch_size, self.history_length, self.obs_dim), dtype=np.float32)
            a2_hist   = np.zeros((self.batch_size, self.history_length, self.action_dim), dtype=np.float32)
            hist_len2 = np.ones(self.batch_size, dtype=np.int64) * self.history_length

            # fill o2_hist, a2_hist
            for i, b_idx in enumerate(bat_indices):

                # take data
                o2_hist[i, :, :] = self.o_buff[(b_idx - self.history_length + self.n_steps) : (b_idx + self.n_steps), :]
                a2_hist[i, :, :] = self.a_buff[(b_idx - self.history_length + self.n_steps) : (b_idx + self.n_steps), :]

                # truncate if done appeared
                for j in range(1, self.history_length):

                    if self.d_buff[(b_idx + self.n_steps - 1) - j] == True:
                        
                        # set history length
                        hist_len2[i] = j

                        # set prior entries to zero when done appeared
                        o2_hist[i, : (self.history_length - j) ,:] = 0.0
                        a2_hist[i, : (self.history_length - j) ,:]  = 0.0
                        
                        # move non-zero experiences to the beginning
                        o2_hist[i] = np.roll(o2_hist[i], shift= -(self.history_length - j), axis=0)
                        a2_hist[i] = np.roll(a2_hist[i], shift= -(self.history_length - j), axis=0)
                        break
                
                # Note: 
                # Truncation at this point is irrelevant for the case n_steps > history_length, as the computed target-Q value 
                # is not regarded if a done appears during the n-step return calculation anyways. However, this truncation can 
                # become relevant if history_length >= n_steps, hence we perform it as in the o-hist case.

            # ----------- n-step return -----------

            # create reward part of n-step return
            r_n = np.zeros((self.batch_size, 1), dtype=np.float32)

            # fill return
            for i, b_idx in enumerate(bat_indices):
                for j in range(self.n_steps):
                    
                    # add discounted reward
                    r_n[i] += (self.gamma ** j) * self.r_buff[b_idx + j]
                    
                    # if done appears, break and set done which will be returned True (to avoid incorrect Q addition)
                    if self.d_buff[b_idx + j] == 1:
                        d[i] = 1
                        break
            
            #print({"o_buff":self.o_buff, "a_buff":self.a_buff, "r_buff":self.r_buff, "o2_buff":self.o2_buff, "d_buff":self.d_buff})
            #print({"o_hist": o_hist, "a_hist": a_hist, "hist_len":hist_len, "o2_hist":o2_hist, "a2_hist":a2_hist,\
            #    "hist_len2":hist_len2, "o":o, "a":a, "r":r_n, "o2": o2, "d": d, "idx": bat_indices})
            #print("--------------------------")

            return (torch.tensor(o_hist).to(self.device), 
                    torch.tensor(a_hist).to(self.device), 
                    torch.tensor(hist_len).to(self.device),
                    torch.tensor(o2_hist).to(self.device), 
                    torch.tensor(a2_hist).to(self.device), 
                    torch.tensor(hist_len2).to(self.device),
                    torch.tensor(o).to(self.device),
                    torch.tensor(a).to(self.device),
                    torch.tensor(r_n).to(self.device),
                    torch.tensor(o2).to(self.device),
                    torch.tensor(d).to(self.device))

"""
np.random.seed(100)
my_buff = UniformReplayBuffer(action_dim=1, obs_dim=1, n_steps=1, gamma=0.99, history_length=5, buffer_length=11, batch_size=1, device="cpu")

o = np.array([.3, .8, .7, 1.5, 1.7, 5.3, 4.7, 9.0, 3.2, 1.7, 0.3])
a = np.array([.5, .7, .4, .2, -0.9, 1.3, 1.2, -1.4, -1.5, 0.1, 0.2])
r = np.array([-0.2, 0.7, -0.6, 0.5, -1.2, -1.7, -2.3, 2.4, -3.4, -5.3, 1.1])
o2= np.array([.8, .7, .9, 1.7, -0.9, 4.7, 9.0, 3.2, 1.7, 4.2, 4.7])
d = np.array([False] * 11)
d[[2, 4, 9]] = True

for i in range(len(o)):
    my_buff.add(o[i], a[i], r[i], o2[i], d[i])

my_buff.sample()
"""
