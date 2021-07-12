import numpy as np
import copy

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
