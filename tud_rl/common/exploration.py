import copy
import numpy as np


class LinearDecayEpsilonGreedy:
    def __init__(self, eps_init, eps_final, eps_decay_steps):
        self.eps_init        = eps_init
        self.eps_final       = eps_final
        self.eps_decay_steps = eps_decay_steps

        self.eps_inc = (eps_final - eps_init) / eps_decay_steps
        self.eps_t   = 0

    def get_epsilon(self, mode):
        "Returns the current epsilon based on linear scheduling."

        if mode == "train":
            self.current_eps = max(self.eps_init + self.eps_inc * self.eps_t, self.eps_final)
            self.eps_t += 1
        
        else:
            self.current_eps = 0

        return self.current_eps


class OU_Noise:
    """Create Ornstein-Uhlenbeck process for exploration."""
    def __init__(self, action_dim, mu=0.0, theta=0.15, dt=0.01, sigma=0.1):
        self.action_dim   = action_dim
        
        self.mu    = np.ones(shape=(1,action_dim)) * mu
        self.theta = theta
        self.dt    = dt
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.x = copy.deepcopy(self.mu)
            
    def sample(self):
        """Samples from process.
        returns: np.array with shape (1, action_dim)."""
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(1,self.action_dim)
        self.x += dx
        return self.x


class Gaussian_Noise:
    """Create white noise process for exploration."""
    def __init__(self, action_dim, mu=0.0, sigma=0.1):
        self.action_dim = action_dim
        
        self.mu    = np.ones(shape=(1,action_dim)) * mu
        self.sigma = sigma

    def sample(self):
        return self.mu + self.sigma * np.random.randn(1,self.action_dim)
    
    def reset(self):
        pass
