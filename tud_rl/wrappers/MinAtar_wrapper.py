import numpy as np
import gym

class MinAtar_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # change obs to be of shape (in_channels, height, width) instead of (height, width, in_channels)
        obs = np.moveaxis(obs, -1, 0)

        return obs
