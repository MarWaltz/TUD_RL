import numpy as np

def exponential_smoothing(x, alpha=0.05):
    s = np.zeros_like(x)

    for idx, x_val in enumerate(x):
        if idx == 0:
            s[idx] = x[idx]
        else:
            s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

    return s

def get_MC_ret_from_rew(rews, gamma):
    """Returns for a given episode of rewards (list) the corresponding list of MC-returns under a specified discount factor."""

    MC = 0
    MC_list = []
    
    for r in reversed(rews):
        # compute one-step backup
        backup = r + gamma * MC
        
        # add to MCs
        MC_list.append(backup)
        
        # update MC
        MC = backup
    
    return list(reversed(MC_list))
