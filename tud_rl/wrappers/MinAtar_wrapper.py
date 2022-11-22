import numpy as np
import gym

class MinAtar_wrapper(gym.ObservationWrapper):
    """Changes observation to be of shape (in_channels, height, width) instead of (height, width, in_channels)"""
    def __init__(self, env):
        env.observation_space._shape = (env.observation_space.shape[2], *env.observation_space.shape[0:2])
        super().__init__(env)

    def observation(self, obs):        
        obs = np.moveaxis(obs, -1, 0)
        return obs
