import torch
import numpy as np 

class UniformReplayBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_length, device):
        self.max_size = buffer_length
        self.ptr = 0
        self.size = 0
        self.device = device

        self.transition_names = ('s', 'a', 'r', 's2', 'd')
        sizes = (state_dim, action_dim, 1, state_dim, 1)
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.zeros((self.max_size, size)))

    def add(self, state, action, next_state, reward, done):
        values = (state, action, next_state, reward, done)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        names = self.transition_names
        return (torch.FloatTensor(getattr(self, name)[ind]).to(self.device) for name in names)
 