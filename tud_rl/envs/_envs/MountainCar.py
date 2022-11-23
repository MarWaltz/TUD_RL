import numpy as np
import gym
from gym import spaces


class MountainCar(gym.Env):
    """The MountainCar environment following the description of p.245 in Sutton & Barto (2018).
    Methods: __init__, step, reset. State consists of [position, velocity]."""

    def __init__(self, rewardStd=1):
        # gym inherits
        super(MountainCar, self).__init__()
        self.observation_space = spaces.Box(low=np.array([-1.2, -0.07], dtype=np.float32),
                                            high=np.array([0.5, 0.07], dtype=np.float32))
        self.action_space = spaces.Discrete(3)

        # reward
        self.rewardStd = rewardStd

        # step cnt
        self._max_episode_steps = 200
        self.made_steps = 0

    def reset(self):
        self.made_steps = 0
        self.position   = -0.6 + np.random.random()*0.2
        self.velocity   = 0.0
        return np.array([self.position, self.velocity])

    def step(self, a):
        """Updates internal state for given action and returns tuple (s2, r, d, None)."""

        assert a in [0, 1, 2], "Invalid action."
        
        # increment step cnt
        self.made_steps += 1

        # update velocity
        self.velocity += 0.001*(a-1) - 0.0025*np.cos(3*self.position)

        if self.velocity < -0.07:
            self.velocity = -0.07
        elif self.velocity >= 0.07:
            self.velocity = 0.06999999
        
        # update position
        self.position += self.velocity
        if self.position < -1.2:
            self.position = -1.2
            self.velocity = 0.0
        
        # calculate done flag and sample reward
        done = True if (self.position >= 0.5 or self.made_steps == self._max_episode_steps) else False
        r = np.random.normal(-1.0, self.rewardStd)
 
        return np.array([self.position, self.velocity]), r, done, {}
    
    def seed(self, seed):
        pass

    def render(self, mode=None):
        """Note: The 'mode' argument is needed since a recent update of the 'gym' package."""
        pass
