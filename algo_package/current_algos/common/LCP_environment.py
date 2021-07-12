import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import gym
from gym import spaces

class LCP_Environment(gym.Env):
    """Class environment with initializer, step, reset and render method."""
    
    def __init__(self):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------
        # river size and number of vessels
        self.x_max = 3000
        self.y_max = 500
        self.n_vessels  = 5
        self.over_coast = 10

        # maximum sight of agent
        self.delta_x_max = 3000
        self.delta_y_max = 500

        # initial agent position, speed and acceleration
        self.start_x_agent = 0
        self.start_y_agent = 250
        self.v_x_agent = 10
        self.v_y_agent = 0
        self.a_x_agent = 0
        self.a_y_agent = 0

        # speed and acceleration of vessels
        self.v_x_max_vessel = 2.5
        self.v_y_max_vessel = 0
        self.a_x_max_vessel = 0
        self.a_y_max_vessel = 0

        # time step and max episode steps
        self.delta_t = 3
        self._max_episode_steps = int(self.x_max / (self.v_x_agent * self.delta_t))

        # rendering
        self.plot_delay = 0.001
        self.agent_color = "red"
        self.vessel_color = ["purple", "blue", "green", "green", "orange", "purple", "blue", "green", "green", "orange"][0:self.n_vessels]
        self.line_color = "black"

        # reward config
        self.variance_x = 12000
        self.variance_y = 500
        
        # for rendering only
        self.reward_lines   = [0.5, 0.1, 0.001] 
        self.ellipse_width  = [2 * np.sqrt(-2 * self.variance_x * np.log(reward)) for reward in self.reward_lines]
        self.ellipse_height = [2 * np.sqrt(-2 * self.variance_y * np.log(reward)) for reward in self.reward_lines]
        
        # --------------------------------  gym inherits ---------------------------------------------------
        super(LCP_Environment, self).__init__()
        self.observation_space = spaces.Box(low=np.full((1, 2 * self.n_vessels + 2), -1, dtype=np.float32)[0],
                                            high=np.full((1, 2 * self.n_vessels + 2), 1, dtype=np.float32)[0])
        self.action_space = spaces.Box(low=np.array([-10], dtype=np.float32), 
                                       high=np.array([10], dtype=np.float32))
        
        # --------------------------------- custom inits ---------------------------------------------------
        self.agent_pos  = None
        self.agent_v    = None
        self.agent_a    = None
        self.vessel_pos = None
        self.vessel_v   = None
        self.vessel_a   = None
        self.delta      = None # delta are truncated delta x and y, but NOT normalized to [-1,1]
        self.state      = None # state is normalized, flattened delta AND agent's y and y-y_max to have information about coast distance
        
    def reset(self):
        """Resets environment to initial state."""
        self._set_pos_v_a()
        self._set_delta_from_pos()
        self._set_state_from_delta()
        return self.state
    
    def _set_pos_v_a(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        self.agent_pos = np.array([self.start_x_agent, self.start_y_agent], dtype=np.float32)
        self.agent_v   = np.array([self.v_x_agent, self.v_y_agent], dtype=np.float32)
        self.agent_a   = np.array([self.a_x_agent, self.a_y_agent], dtype=np.float32)
        
        self.vessel_pos = np.empty((0,2), dtype=np.float32)
        for _ in range(self.n_vessels):
            self.vessel_pos = np.vstack((self.vessel_pos,
                                         (np.random.uniform(100, self.x_max), np.random.uniform(100, self.y_max-100)))) # x,y coordinates
        #self.vessel_pos = np.array([[1000, 100], [1200, 150], [1800, 350], [2200, 300], [2500, 300]], dtype=np.float32)
        
        self.vessel_v = np.empty((0,2), dtype=np.float32)
        for _ in range(self.n_vessels):
            self.vessel_v = np.vstack((self.vessel_v, 
                                       (np.random.uniform(-self.v_x_max_vessel, self.v_x_max_vessel),
                                        np.random.uniform(-self.v_y_max_vessel, self.v_y_max_vessel))))
        
        self.vessel_a = np.empty((0,2), dtype=np.float32)
        for _ in range(self.n_vessels):
            self.vessel_a = np.vstack((self.vessel_a, 
                                       (np.random.uniform(-self.a_x_max_vessel, self.a_x_max_vessel),
                                        np.random.uniform(-self.a_y_max_vessel, self.a_y_max_vessel))))

    def _set_delta_from_pos(self):
        """Sets based on the positions of the agent and the vessel all delta x and delta y and 
        truncates them to agent's sight."""
        self.delta = np.apply_along_axis(lambda ves: ves - self.agent_pos, 1, self.vessel_pos)
        
        # truncate delta to self.delta_x_max and self.delta_y_max
        for i in range(self.delta.shape[1]):
            tmp = self.delta[:,i].copy()
            if i == 0:
                tmp[tmp > self.delta_x_max] = self.delta_x_max
                tmp[tmp < -self.delta_x_max] = -self.delta_x_max
                self.delta[:,i] = tmp
            else:
                tmp[tmp > self.delta_y_max] = self.delta_y_max
                tmp[tmp < -self.delta_y_max] = -self.delta_y_max
                self.delta[:,i] = tmp
        
        # ADDITION: if DELTA_X is exceeded, the vessel is not in sight
        # --> delta_y should be set to DELTA_Y as well (and vice versa)
        for i in range(self.n_vessels):
            if abs(self.delta[i][0]) == self.delta_x_max:
                self.delta[i][1] = self.delta_y_max
            if abs(self.delta[i][1]) == self.delta_y_max:
                self.delta[i][0] = self.delta_x_max
    
    def _set_state_from_delta(self):
        """Sets state which is flattened, ordered delta normalized to [-1,1] and the normalized agent's y and y_max-y coordinate."""
        # normalize delta
        delta_normalized = self.delta.copy()
        delta_normalized[:,0] = delta_normalized[:,0] / self.delta_x_max
        delta_normalized[:,1] = delta_normalized[:,1] / self.delta_y_max
        
        # order delta based on the euclidean distance and get state
        eucl_dist = np.apply_along_axis(lambda x: np.sqrt(x[0]**2 + x[1]**2), 1, delta_normalized)
        idx = np.argsort(eucl_dist)
        self.state = np.append(delta_normalized[idx].transpose().flatten(), np.array([self.agent_pos[1], self.y_max-self.agent_pos[1]]) / self.y_max).astype(np.float32)
    
    def step(self, action):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done."""
        self._move_vessel()
        self._move_agent(action)
        self._set_delta_from_pos()
        self._set_state_from_delta()
        reward = self._calculate_reward()
        done = self._done()
        
        return self.state, reward, done, {}
    
    def _move_vessel(self):
        """Updates positions, velocities and accelerations of vessels. For now accelerations are constant.
        Used approximation: Euler-Cromer method, that is v_(n+1) = v_n + a_n * t and x_(n+1) = x_n + v_(n+1) * t."""
        for i in range(self.n_vessels):
            # get current values
            pos = self.vessel_pos[i].copy()
            v = self.vessel_v[i].copy()
            a = self.vessel_a[i].copy()
            
            # update v
            new_v_x = v[0] + a[0] * self.delta_t
            new_v_y = v[1] + a[1] * self.delta_t
            if new_v_x > self.v_x_max_vessel:
                new_v_x = self.v_x_max_vessel
            if new_v_y > self.v_y_max_vessel:
                new_v_y = self.v_y_max_vessel
            
            # update position
            new_x = pos[0] + new_v_x * self.delta_t
            new_y = pos[1] + new_v_y * self.delta_t
            if new_y >= self.y_max + self.over_coast:
                new_y = self.y_max + self.over_coast
            if new_y <= -self.over_coast:
                new_y = -self.over_coast
            
            # set new values
            self.vessel_pos[i] = np.array([new_x, new_y], dtype=np.float32)
            self.vessel_v[i] = np.array([new_v_x, new_v_y], dtype=np.float32)
    
    def _move_agent(self, action):
        """Update self.agent_pos using a given action. For now: a_x, a_y = 0.
        Actions corresponds to delta y."""
        self.agent_pos[0] += self.agent_v[0] * self.delta_t
        self.agent_pos[1] += action
        
        if self.agent_pos[1] >= self.y_max + self.over_coast:
            self.agent_pos[1] = self.y_max + self.over_coast
        if self.agent_pos[1] <= -self.over_coast:
            self.agent_pos[1] = -self.over_coast
    
    def _calculate_reward(self):
        """Returns reward of the current state."""
        reward = 0
        
        # reward from other vessels
        vess_reward = 0
        for i in range(self.n_vessels):
            vess_reward = max(np.exp(-0.5 * ((self.agent_pos[0] - self.vessel_pos[i][0])**2 / self.variance_x  + 
                                             (self.agent_pos[1] - self.vessel_pos[i][1])**2 / self.variance_y)), vess_reward)
           #vess_reward = vess_reward * 2 if vess_reward > 0.5 else vess_reward
        reward -= vess_reward
        
        #for i in range(self.n_vessels):
        #    vess_reward = np.exp(-0.5 * ((self.agent_pos[0] - self.vessel_pos[i][0])**2 / VARIANCE_X  + 
        #                                (self.agent_pos[1] - self.vessel_pos[i][1])**2 / VARIANCE_Y))
        #    vess_reward = 0 if vess_reward < 0.001 else vess_reward
        #    reward -= vess_reward
                
        # reward from top coast
        if self.agent_pos[1] >= self.y_max:
            reward -= 1
        elif self.agent_pos[1] >= self.y_max-100:
            reward -= (self.agent_pos[1] - (self.y_max-100)) * 0.01
        
        # reward from bottom coast
        if self.agent_pos[1] <= 0:
            reward -= 1
        elif self.agent_pos[1] <= 100:
            reward -= 1 - (self.agent_pos[1]) * 0.01
               
        return reward
    
    def _done(self):
        """Returns boolean flag whether episode is over."""
        return True if self.agent_pos[0] >= self.x_max else False
    
    def render(self, agent_name=None, reward=None, episode_timestep=None):
        """Renders the current environment."""
        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            self.fig = plt.figure(figsize=(17, 10))
            self.gs  = self.fig.add_gridspec(2, 2)
            self.ax0 = self.fig.add_subplot(self.gs[0, :]) # ship
            self.ax1 = self.fig.add_subplot(self.gs[1, 0]) # state
            self.ax2 = self.fig.add_subplot(self.gs[1, 1]) # reward
            self.ax2.old_time = 0
            self.ax2.old_reward = 0
            plt.ion()
            plt.show()
        
        # ---- ACTUAL SHIP MOVEMENT ----
        # clear prior axes, set limits and add labels and title
        self.ax0.clear()
        self.ax0.set_xlim(-30, self.x_max)
        self.ax0.set_ylim(0, self.y_max)
        self.ax0.set_xlabel("x")
        self.ax0.set_ylabel("y")
        if agent_name is not None:
            self.ax0.set_title(agent_name)

        # set agent and vessels
        self.ax0.scatter(self.agent_pos[0], self.agent_pos[1], color = self.agent_color)
        self.ax0.scatter(self.vessel_pos[:,0], self.vessel_pos[:,1], color = self.vessel_color)

        # draw contour lines around vessels
        for i in range(self.n_vessels):
            for j in range(len(self.reward_lines)):
                ellipse = Ellipse(xy=(self.vessel_pos[i][0], self.vessel_pos[i][1]), 
                                  width=self.ellipse_width[j], height=self.ellipse_height[j], color = self.line_color, fill = False)
                self.ax0.add_patch(ellipse)
        
        # connect agent and "in sight" vessels
        for i in range(self.n_vessels):
            if all(abs(self.delta[i]) < np.array([self.delta_x_max, self.delta_y_max])):
                self.ax0.plot(np.array([self.agent_pos[0], self.vessel_pos[i,0]]),
                              np.array([self.agent_pos[1], self.vessel_pos[i,1]]),
                              color = self.line_color, alpha = 0.15)
        
        # connect agent with coasts
        self.ax0.plot(np.array([self.agent_pos[0], self.agent_pos[0]]),
                      np.array([self.agent_pos[1], 0]),
                      color = self.line_color, alpha = 0.25)
        self.ax0.plot(np.array([self.agent_pos[0], self.agent_pos[0]]),
                      np.array([self.agent_pos[1], self.y_max]),
                      color = self.line_color, alpha = 0.25)
        
        # ---- STATE PLOT ----
        # clear prior axes, set limits
        self.ax1.clear()
        self.ax1.set_xlim(-1, 1)
        self.ax1.set_ylim(-1, 1)
        
        # add agent and states
        self.ax1.scatter(0, 0, color = self.agent_color)
        self.ax1.set_xlabel("Normalized delta x")
        self.ax1.set_ylabel("Normalized delta y")
        for i in range(self.n_vessels):
            self.ax1.scatter(self.state[i], self.state[i + self.n_vessels], color = self.vessel_color[i])

        # ---- REWARD PLOT ----
        if episode_timestep == 0:
            self.ax2.clear()
            self.ax2.old_time = 0
            self.ax2.old_reward = 0
        self.ax2.set_xlim(1, self.x_max / (self.v_x_agent * self.delta_t))
        self.ax2.set_ylim(-1.1, 0.1)
        self.ax2.set_xlabel("Timestep in episode")
        self.ax2.set_ylabel("Reward")
        self.ax2.plot([self.ax2.old_time, episode_timestep], [self.ax2.old_reward, reward], color = self.line_color)
        self.ax2.old_time = episode_timestep
        self.ax2.old_reward = reward
        
        # delay plotting for ease of user
        plt.pause(self.plot_delay)
