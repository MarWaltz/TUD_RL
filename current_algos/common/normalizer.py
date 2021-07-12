import numpy as np

class Input_Normalizer:
    """Normalizes the input to have zero mean and unit variance in each observation dimension."""
    
    def __init__(self, state_dim, prior=None) -> None:
        """prior should be tuple of length 4 containing old n, mean, mean_diff, var."""
        self.state_dim = state_dim
        
        if prior is not None:
            if len(prior) != 4:
                raise Exception("Invalide values for input normalizer.")
            self.n, self.mean, self.mean_diff, self.var = prior
        else:
            self.n         = 0
            self.mean      = np.zeros(self.state_dim)
            self.mean_diff = np.zeros(self.state_dim)
            self.var       = np.zeros(self.state_dim)

    def normalize(self, x, mode) -> np.array:
        """Takes input, updates tracked mean and variance (online) and returns normalized output.
        x:      np.array with shape (state_dim,)
        mode:   str, either 'train' or 'test', to decide whether mean and var are updated

        output: np.array with shape (state_dim,)
        """
        if mode not in ["train", "test"]:
            raise Exception("Input normalizer's mode should be either 'train' or 'test'.")
        
        if mode == "train":
            # update mean
            self.n += 1.
            last_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.n
            
            # update variance
            self.mean_diff += (x - last_mean) * (x - self.mean)
            self.var = (self.mean_diff / self.n).clip(min = 0.01)
        
        # return normalized input
        return (x - self.mean) / np.sqrt(self.var)
    
    def get_for_save(self) -> tuple:
        """Returns tracked values."""
        return (self.n, self.mean, self.mean_diff, self.var)


class Action_Normalizer:
    """Normalizer to transform actions from application scale to [-1,1] and vice versa. 
    Critical: Assumes all action dimensions have the same application scale."""
    
    def __init__(self, action_high, action_low):
        self.action_high = action_high
        self.action_low  = action_low
        
        self.b     = (action_high + action_low)/ 2.
        self.m     = (action_high - action_low)/ 2.
        self.m_inv = 2./(action_high - action_low)
    
    def norm_to_action(self, action):
        """Takes float action input in [-1,1] and transforms it to application scale."""
        return self.m * action + self.b

    def action_to_norm(self, action):
        """Transforms float action input in application scale to [-1,1]."""
        return self.m_inv * (action - self.b)
