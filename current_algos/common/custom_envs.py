import numpy as np
from scipy.stats import norm 
from scipy.signal import savgol_filter
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
        self.y_max = 300
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


class ObstacleAvoidance_Env(gym.Env):
    """Class environment with initializer, step, reset and render method."""
    
    def __init__(self):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------
        self.hide_velocity = False
        self.sort_obs_ttc = True
        self.polygon_reward = False

        # river size and vessel characteristics   
        self.y_max = 500
        self.n_vessels  = 10
        self.n_vessels_half  = int(self.n_vessels/2)
        self.over_coast = 10
        self.max_temporal_dist = 300 # maximal temporal distance when placing new vessel

        # maximum sight of agent
        self.delta_x_max = 3000
        self.delta_y_max = 300

        # initial agent position, speed and acceleration
        self.start_x_agent = 0
        self.start_y_agent = 0
        self.v_x_agent = 0
        self.v_y_agent = 0
        self.a_x_agent = 0
        self.a_y_agent = 0

        # speed and acceleration of vessels
        self.vx_max = 6
        self.vy_max = 4
        self.ax_max = 0
        self.ay_max = 0.01

        # time step, max episode steps and length of river
        self.delta_t = 5
        self.current_timestep = 0 
        self._max_episode_steps = 500
        

        # rendering
        self.plot_delay = 0.001
        self.agent_color = "red"
        #self.vessel_color = ["purple", "blue", "green", "green", "orange", "purple", "blue", "green", "green", "orange"][0:self.n_vessels]
        self.vessel_color =  np.full(self.n_vessels,"green")
        self.vessel_color[0:self.n_vessels_half] = "blue"
        self.line_color = "black"

        # reward config
        self.variance_x = 12000
        self.variance_y = 25
        self.variance_ttc = 25
        
        # for rendering only
        # self.reward_lines   = [0.5, 0.1, 0.001] 
        # self.ellipse_width  = [2 * np.sqrt(-2 * self.variance_x * np.log(reward)) for reward in self.reward_lines]
        # self.ellipse_height = [2 * np.sqrt(-2 * self.variance_y * np.log(reward)) for reward in self.reward_lines]
        
        # --------------------------------  gym inherits ---------------------------------------------------
        if self.hide_velocity:
            num_vessel_obs = 2
        else:
            num_vessel_obs = 4
        super(ObstacleAvoidance_Env, self).__init__()
        self.observation_space = spaces.Box(low=np.full((1, num_vessel_obs * self.n_vessels + 2), -1, dtype=np.float32)[0],
                                            high=np.full((1, num_vessel_obs * self.n_vessels + 2), 1, dtype=np.float32)[0])
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        
        # --------------------------------- custom inits ---------------------------------------------------
        self.agent_x      = None
        self.agent_y      = None
        self.agent_vx     = None
        self.agent_vy     = None
        self.agent_ax     = None
        self.agent_ay     = None
        self.agent_ay_old = None

        self.vessel_x   = None
        self.vessel_y   = None
        self.vessel_vx  = None
        self.vessel_vy  = None
        self.vessel_ax  = None
        self.vessel_ay  = None
        self.vessel_ttc = None

        self.state      = None 
        
    def reset(self):
        """Resets environment to initial state."""
        self.current_timestep = 0
        self.reward = 0
        self._set_AR1()
        self._set_dynamics()
        self._set_state()
        return self.state
    
    def _set_AR1(self):
        """Sets the AR1 Array containing the desired lateral trajectory for all episode steps"""
        self.AR1 = np.zeros(self._max_episode_steps+2000, dtype=np.float32) 
        for i in range(self.AR1.size-1):
            self.AR1[i+1] = self.AR1[i] * 0.99 + np.random.normal(0,np.sqrt(400))

        # smooth data
        self.AR1 = savgol_filter(self.AR1,125,2)    


    def _set_dynamics(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        self.agent_x = self.start_x_agent
        self.agent_y = self.start_y_agent
        self.agent_vx = np.random.uniform(1,self.vx_max)
        self.x_max = np.ceil(self.agent_vx * self._max_episode_steps * self.delta_t) # set window length by chosen longitudinal speed
        self.agent_vy = self.v_y_agent
        self.agent_ax = self.a_x_agent
        self.agent_ay = self.a_y_agent 
        self.agent_ay_old = self.a_y_agent
        
        self.vessel_x = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_y = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_vx = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_vy = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_ttc = np.empty((self.n_vessels), dtype=np.float32)

        # find initial vessel position
        self._place_vessel(True,-1)
        self._place_vessel(True, 1)


        for i in range(int(self.n_vessels/2-1)):
            self._place_vessel(False,-1)
            self._place_vessel(False, 1)

    
    def _place_vessel(self, initial_placement, vessel_direction):
        if vessel_direction == -1:
            ttc = self.vessel_ttc[:self.n_vessels_half].copy()
            x = self.vessel_x[:self.n_vessels_half].copy()
            y = self.vessel_y[:self.n_vessels_half].copy()
            vx = self.vessel_vx[:self.n_vessels_half].copy()
            vy = self.vessel_vy[:self.n_vessels_half].copy()
        else:
            ttc = self.vessel_ttc[self.n_vessels_half:].copy()
            x = self.vessel_x[self.n_vessels_half:].copy()
            y = self.vessel_y[self.n_vessels_half:].copy()
            vx = self.vessel_vx[self.n_vessels_half:].copy()
            vy = self.vessel_vy[self.n_vessels_half:].copy()

        # compute new ttc
        if initial_placement:
            new_ttc = np.random.uniform(-self.max_temporal_dist, -1)
        else:
            new_ttc = np.maximum(1,ttc[-1] +  np.random.uniform(0,self.max_temporal_dist))

        # compute new vessel dynamics
        y_future = self.AR1[abs(int(self.current_timestep + new_ttc/self.delta_t))] + vessel_direction * np.maximum(20, np.random.normal(100,70))
        new_vx = np.random.uniform(-self.vx_max, self.vx_max)
        new_x = (self.agent_vx - new_vx) * new_ttc + self.agent_x
        new_vy = np.abs(new_vx/5) * np.random.uniform(-1,1)
        new_y = y_future - new_vy * new_ttc

        # rotate dynamic arrays to place new vessel at the end
        ttc = np.roll(ttc,-1)
        x = np.roll(x,-1)
        y = np.roll(y,-1)
        vx = np.roll(vx,-1)
        vy = np.roll(vy,-1)

        # set new vessel dynamics
        ttc[-1] = new_ttc
        x[-1] = new_x
        y[-1] = new_y
        vx[-1] = new_vx
        vy[-1] = new_vy

        if vessel_direction == -1:
            self.vessel_ttc[:self.n_vessels_half] = ttc
            self.vessel_x[:self.n_vessels_half] = x
            self.vessel_y[:self.n_vessels_half] = y
            self.vessel_vx[:self.n_vessels_half] = vx
            self.vessel_vy[:self.n_vessels_half] = vy
        else:
            self.vessel_ttc[self.n_vessels_half:] = ttc
            self.vessel_x[self.n_vessels_half:] = x
            self.vessel_y[self.n_vessels_half:] = y
            self.vessel_vx[self.n_vessels_half:] = vx
            self.vessel_vy[self.n_vessels_half:] = vy
    
    
    def _set_state(self):
        """Sets state which is flattened, ordered with ascending TTC, normalized and clipped to [-1, 1]"""
        if self.sort_obs_ttc:
            # arrays are already sorted according ascending ttc
            x = self.vessel_x.copy()
            y = self.vessel_y.copy()
            vx = self.vessel_vx.copy()
            vy = self.vessel_vy.copy()
        else:
            # compute sorting index array for asceding x-value
            idx = np.concatenate([np.argsort(self.vessel_x[:self.n_vessels_half]),np.argsort(self.vessel_x[self.n_vessels_half:]) + self.n_vessels_half])

            x = self.vessel_x[idx].copy()
            y = self.vessel_y[idx].copy()
            vx = self.vessel_vx[idx].copy()
            vy = self.vessel_vy[idx].copy()


        self.state = np.empty(0, dtype=np.float32)
        self.state = np.append(self.state, np.clip(self.agent_ay/self.ay_max, -1, 1))
        self.state = np.append(self.state, np.clip(self.agent_vy/self.vy_max, -1, 1))
        self.state = np.append(self.state, np.clip((self.agent_x  - x)/self.delta_x_max,-1, 1))
        self.state = np.append(self.state, np.clip((self.agent_y  - y)/self.delta_y_max,-1, 1))

        if not self.hide_velocity:
            self.state = np.append(self.state, np.clip((self.agent_vx - vx)/(2*self.vx_max),-1, 1))
            self.state = np.append(self.state, np.clip((self.agent_vy - vy)/(2*self.vy_max),-1, 1))

    
    def step(self, action):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done."""
        self.crash_flag = False
        self._move_vessel()
        self._move_agent(action)
        self._set_state()
        self._calculate_reward()
        done = self._done()
        self.current_timestep += 1
        
        return self.state, self.reward, done, {}
    
    def _move_vessel(self):
        """Updates positions, velocities and accelerations of vessels. For now accelerations are constant.
        Used approximation: Euler-Cromer method, that is v_(n+1) = v_n + a_n * t and x_(n+1) = x_n + v_(n+1) * t."""
        for i in range(self.n_vessels):
            # lateral dynamics for ships with positive TTC
            if self.vessel_ttc[i] > 0:
                self.vessel_y[i] = self.vessel_y[i] + self.vessel_vy[i] * self.delta_t

            # longitudinal dynamics     
            self.vessel_x[i] = self.vessel_x[i] + self.vessel_vx[i] * self.delta_t
            self.vessel_ttc[i] -= self.delta_t
            
            
            # replace vessel if necessary       
            while self.vessel_ttc[1] < 0:
                self._place_vessel(False,-1)
                if self.agent_y < self.vessel_y[0]:
                    self.agent_y = self.vessel_y[0] # agent crahsed! 
                    self.crash_flag = True

            while self.vessel_ttc[self.n_vessels_half+1] < 0:
                self._place_vessel(False, 1)           
                if self.agent_y > self.vessel_y[self.n_vessels_half]:
                    self.agent_y = self.vessel_y[self.n_vessels_half] # agent crahsed! 
                    self.crash_flag = True
    
    def _move_agent(self, action):
        """Update self.agent_pos using a given action. For now: a_x = 0."""
        self.agent_ay_old = self.agent_ay

        # update lateral dynamics
        self.agent_ay = self.ay_max * action

        agent_vy_new = np.clip(self.agent_vy + self.agent_ay * self.delta_t,-self.vy_max, self.vy_max)

        agent_y_new = self.agent_y + 0.5 * (self.agent_vy + agent_vy_new) * self.delta_t

        self.agent_vy = agent_vy_new
        self.agent_y = agent_y_new

        # update longitudinal dynamics
        self.agent_x= self.agent_x + self.agent_vx * self.delta_t
        

    
    def _calculate_reward(self):
        """Returns reward of the current state."""   
        # compute jerk reward
        jerk_reward = -40 * (((self.agent_ay_old - self.agent_ay)/0.1)**2)/3600

        if self.polygon_reward:                
        
            # create vertices between closest upper and lower vessels
            alpha1 = -self.vessel_ttc[0]/(self.vessel_ttc[1] - self.vessel_ttc[0]) 
            alpha2 = -self.vessel_ttc[self.n_vessels_half]/(self.vessel_ttc[self.n_vessels_half+1] - self.vessel_ttc[self.n_vessels_half])
            
            delta_y1 = self.vessel_y[0] + alpha1 * (self.vessel_y[1] - self.vessel_y[0]) - self.agent_y
            delta_y2 = self.agent_y - (self.vessel_y[self.n_vessels_half] + alpha2 * (self.vessel_y[self.n_vessels_half+1] - self.vessel_y[self.n_vessels_half]))

            # compute vessel reward based on distance to vertices
            if delta_y1 > 0: 
                vess_reward1 = -1
            else:
                vess_reward1 = -norm.pdf(delta_y1,0,self.variance_y)/norm.pdf(0,0,self.variance_y)

            if delta_y2 > 0: 
                vess_reward2 = -1
            else:
                vess_reward2 = -norm.pdf(delta_y2,0,self.variance_y)/norm.pdf(0,0,self.variance_y)
        

            
            # final reward
            if vess_reward1 == -1 and vess_reward2 == -1:
                self.reward = jerk_reward
            else:
                self.reward = jerk_reward - (np.maximum(vess_reward1,vess_reward2) - np.minimum(vess_reward1,vess_reward2))/(np.maximum(vess_reward1,vess_reward2)+1)

        else: # point reward for all vessels            
            vess_reward1 = 0
            vess_reward2 = 0
            for i in range(self.n_vessels_half):
                vess_reward1 = np.maximum(vess_reward1, 
                                        norm.pdf(self.vessel_ttc[i],0,self.variance_ttc)/norm.pdf(0,0,self.variance_ttc) *
                                        norm.pdf(np.maximum(0,self.agent_y-self.vessel_y[i]),0,self.variance_y)/norm.pdf(0,0,self.variance_y))
                vess_reward2 = np.maximum(vess_reward2, 
                                        norm.pdf(self.vessel_ttc[self.n_vessels_half+i],0,self.variance_ttc)/norm.pdf(0,0,self.variance_ttc) *
                                        norm.pdf(np.maximum(0,self.vessel_y[self.n_vessels_half+i]-self.agent_y),0,self.variance_y)/norm.pdf(0,0,self.variance_y))
            self.reward = jerk_reward - (np.maximum(-vess_reward1,-vess_reward2) - np.minimum(-vess_reward1,-vess_reward2))/(np.maximum(-vess_reward1,-vess_reward2)+1)

        #if self.crash_flag: 
            # self.reward -=6
    
    def _done(self):
        """Returns boolean flag whether episode is over."""
        return True if self.current_timestep >= self._max_episode_steps else False
    
    def render(self, agent_name=None):
        """Renders the current environment."""

        # plot every nth timestep
        if self.current_timestep % 1 == 0: 

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
            self.ax0.set_xlim(-1500, self.x_max + 1500)
            self.ax0.set_ylim(-self.y_max, self.y_max)
            self.ax0.set_xlabel("x")
            self.ax0.set_ylabel("y")
            if agent_name is not None:
                self.ax0.set_title(agent_name)

            # set agent and vessels
            self.ax0.scatter(self.agent_x, self.agent_y, color = self.agent_color)
            self.ax0.scatter(self.vessel_x, self.vessel_y, color = self.vessel_color)
            
            # ---- STATE PLOT ----
            # clear prior axes, set limits
            self.ax1.clear()
            self.ax1.set_xlim(-200, 1000)
            self.ax1.set_ylim(-self.y_max, self.y_max)
            
            # add agent and states
            self.ax1.scatter(0, self.agent_y, color = self.agent_color)
            self.ax1.set_xlabel("Normalized delta x")
            self.ax1.set_ylabel("Normalized delta y")
            self.ax1.scatter(self.vessel_ttc, self.vessel_y, color = self.vessel_color)

            # ---- REWARD PLOT ----
            if self.current_timestep == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
            self.ax2.set_xlim(1, self.x_max / (self.agent_vx * self.delta_t))
            self.ax2.set_ylim(-1.1, 0.1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward")
            self.ax2.plot([self.ax2.old_time, self.current_timestep], [self.ax2.old_reward, self.reward], color = self.line_color)
            self.ax2.old_time = self.current_timestep
            self.ax2.old_reward = self.reward
            
            # delay plotting for ease of user
            plt.pause(self.plot_delay)
        


class MountainCar(gym.Env):
    """The MountainCar environment following the description of p.245 in Sutton & Barto (2018).
    Methods: __init__, step, reset. State consists of [position, velocity]."""

    def __init__(self, rewardStd):
        # gym inherits
        super(MountainCar, self).__init__()
        self.observation_space = spaces.Box(low=np.array([-1.2, -0.07], dtype=np.float32),
                                            high=np.array([0.5, 0.07], dtype=np.float32))
        self.action_space = spaces.Discrete(3)
        self._max_episode_steps = 500

        # reward
        self.rewardStd = rewardStd

        # step cnt
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
 
        return np.array([self.position, self.velocity]), r, done, None
    
    def seed(self, seed):
        pass
