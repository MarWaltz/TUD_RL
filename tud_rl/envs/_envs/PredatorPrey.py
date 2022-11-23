import math
from typing import Union

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt


class Pred:
    def __init__(self, x, y, vx, vy, dt, x_max, y_max) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.dt = dt
        self.x_max = x_max
        self.y_max = y_max
        self.color = "red"
        self.size = 4
        self.is_pred = True
    
    def move(self):
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

    def is_off_map(self):
        if self.x <= 0 or self.x >= self.x_max or self.y <= 0 or self.y >= self.y_max:
            return True
        return False

class Prey(Pred):
    def __init__(self, x, y, vx, vy, dt, x_max, y_max) -> None:
        super().__init__(x, y, vx, vy, dt, x_max, y_max)
        self.color = "green"
        self.size = 2
        self.is_pred = False


class PredatorPrey(gym.Env):
    """Implements the classic predator-prey simulation game. There is one predator and N_agents-1 preys."""
    def __init__(self, N_agents, N_preds, N_preys):
        super(PredatorPrey, self).__init__()

        # simulation setup
        self.x_max = 100
        self.y_max = 100
        self.v_max_pred = 3
        self.v_max_prey = 6

        # config
        assert N_agents >= 2, "Need at least two agents."
        assert N_preys + N_preds == N_agents, "Sum of preds and preys should equals number of agents."
        self.N_agents = N_agents
        self.N_preds = N_preds
        self.N_preys = N_preys

        # obs and act size for a single agent
        obs_size = 2 + (N_agents-1) * 3
        act_size = 2
        self.observation_space = spaces.Box(low  = np.full(obs_size, 0.0, dtype=np.float32), 
                                            high = np.full(obs_size, 1.0, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                       high = np.full(act_size,  1.0, dtype=np.float32))
        self.dt = 1.0
        self._max_episode_steps = 500

    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # randomly generate pred and preys
        self.agents : Union[Pred, Prey] = []
        for _ in range(self.N_preds):
            self.agents.append(Pred(x  = np.random.uniform(low=0.0, high=self.x_max),
                                    y  = np.random.uniform(low=0.0, high=self.y_max),
                                    vx = np.random.uniform(low=-self.v_max_pred, high=self.v_max_pred),
                                    vy = np.random.uniform(low=-self.v_max_pred, high=self.v_max_pred),
                                    dt = self.dt, x_max = self.x_max, y_max = self.y_max))
        for _ in range(self.N_preys):
            self.agents.append(Prey(x  = np.random.uniform(low=0.0, high=self.x_max),
                                    y  = np.random.uniform(low=0.0, high=self.y_max),
                                    vx = np.random.uniform(low=-self.v_max_prey, high=self.v_max_prey),
                                    vy = np.random.uniform(low=-self.v_max_prey, high=self.v_max_prey),
                                    dt = self.dt, x_max = self.x_max, y_max = self.y_max))
        # init state
        self._set_state()
        self.state_init = self.state
        return self.state
  
    def _set_state(self):
        """State contains own normalized position, relative normalized positions of others, and flag whether other is predator. 
        Overall 2 + (N_agents-1) * 3 features. Thus, state is np.array([N_agents, 2 + (N_agents-1) * 3])."""
        self.state = np.zeros((self.N_agents, 2 + (self.N_agents-1) * 3), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            s_i = np.array([agent.x/self.x_max, agent.y/self.y_max])

            for j, other in enumerate(self.agents):
                if i != j:
                    s_other = np.array([(other.x-agent.x)/self.x_max, 
                                        (other.y-agent.y)/self.y_max, 
                                        1.0 if other.is_pred else -1.0], dtype=np.float32)
                    s_i = np.concatenate((s_i, s_other))
            self.state[i] = s_i

    def step(self, a):
        """a is np.array([N_agents, action_dim]), where action_dim = 2 in our case."""
        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.dt
 
        # update pred/prey velocities
        for i in range(self.N_agents):
            if self.agents[i].is_pred:
                v_max = self.v_max_pred
            else:
                v_max = self.v_max_prey
            self.agents[i].vx = a[i, 0] * v_max
            self.agents[i].vy = a[i, 1] * v_max

        # update positions
        [agent.move() for agent in self.agents]

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, self.r, d, {}
 
    def _calculate_reward(self):
        self.r = np.zeros((self.N_agents, 1), dtype=np.float32)

        for i, agent in enumerate(self.agents):
            r_i = 0.0
            
            # leave the simulation area
            if agent.is_off_map():
                r_i -= 10.0

            # predators get rewarded for catching preys and based on distance
            if agent.is_pred:
                for other in self.agents:
                    if not other.is_pred:
                        d = math.sqrt((agent.x-other.x)**2 + (agent.y-other.y)**2)
                        if d <= (agent.size + other.size):
                            r_i += 50.0
                        else:
                            r_i += math.exp(-d/self.x_max)

            # preys get rewarded for having large distance two predators
            else:
                for other in self.agents:
                    if other.is_pred:
                        d = math.sqrt((agent.x-other.x)**2 + (agent.y-other.y)**2)
                        if d <= (agent.size + other.size):
                            r_i -= 10.0
                        else:
                            r_i += math.exp(d/self.x_max)
            self.r[i] = r_i

    def _done(self):
        # someone left the map
        if any([agent.is_off_map() for agent in self.agents]):
            return True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False

    def render(self, mode=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 10 == 0: 
            
            # init figure
            if len(plt.get_fignums()) == 0:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))
                plt.ion()
                plt.show()           
            
            # set screen
            self.ax1.clear()
            self.ax1.set_xlim(-5, self.x_max+5)
            self.ax1.set_ylim(-5, self.y_max+5)
            self.ax1.set_xlabel("x")
            self.ax1.set_ylabel("y")

            # plot preds/preys
            for agent in self.agents:
                circ = plt.Circle((agent.x, agent.y), agent.size, color=agent.color)
                self.ax1.add_patch(circ)
            plt.pause(0.1)
