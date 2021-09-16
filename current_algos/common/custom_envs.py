import numpy as np
from scipy.stats import norm 
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import gym
from gym import spaces


class ObstacleAvoidance_Env(gym.Env):
    """Class environment with initializer, step, reset and render method."""
    
    def __init__(self, POMDP_type="MDP"):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------

        assert POMDP_type in ["MDP", "RV", "FL"], "Unknown MDP/POMDP specification."

        self.POMDP_type     = POMDP_type
        self.FL_prob        = 0.2
        self.sort_obs_ttc   = False
        self.polygon_reward = False

        # river size and vessel characteristics   
        self.y_max = 500 # only for plotting
        self.n_vessels  = 12
        self.n_vessels_half  = int(self.n_vessels/2)
        self.max_temporal_dist = 300 # maximal temporal distance when placing new vessel

        # maximum sight of agent
        self.delta_x_max = 3000
        self.delta_y_max = 500

        # initial agent position, speed and acceleration
        self.start_x_agent = 0
        self.start_y_agent = 0
        self.v_x_agent = 0
        self.v_y_agent = 0
        self.a_x_agent = 0
        self.a_y_agent = 0

        # speed and acceleration of vessels
        self.vx_max = 6
        self.vy_max = 1
        self.ax_max = 0
        self.ay_max = 0.1

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
        if self.POMDP_type == "RV":
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
        y_future = self.AR1[abs(int(self.current_timestep + new_ttc/self.delta_t))] + vessel_direction * np.maximum(10, np.random.normal(40,60))
        new_vx = np.random.uniform(-self.vx_max, self.vx_max)
        new_x = (self.agent_vx - new_vx) * new_ttc + self.agent_x
        new_vy = np.abs(new_vx/self.vx_max) * np.random.uniform(-1,1)
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
            # compute sorting index array for asceding euclidean distance
            eucl_dist = (self.vessel_x - self.agent_x)**2 + (self.vessel_y - self.agent_y)**2
            idx1 = np.argsort(eucl_dist[:self.n_vessels_half])
            idx2 = np.argsort(eucl_dist[self.n_vessels_half:]) + self.n_vessels_half
            idx = np.concatenate([idx1, idx2])

            x = self.vessel_x[idx].copy()
            y = self.vessel_y[idx].copy()
            vx = self.vessel_vx[idx].copy()
            vy = self.vessel_vy[idx].copy()

        self.state = np.empty(0, dtype=np.float32)
        self.state = np.append(self.state, self.agent_ay/self.ay_max)
        self.state = np.append(self.state, self.agent_vy/self.vy_max)
        self.state = np.append(self.state, (self.agent_x  - x)/self.delta_x_max)
        #self.state = np.append(self.state, self.vessel_ttc/1200)
        self.state = np.append(self.state, (self.agent_y  - y)/self.delta_y_max)

        if self.POMDP_type in ["MDP", "FL"]:
            self.state = np.append(self.state, (self.agent_vx - vx)/(2*self.vx_max))
            self.state = np.append(self.state, (self.agent_vy - vy)/(2*self.vy_max))
        if self.POMDP_type == "FL" and np.random.binomial(1, self.FL_prob) == 1:
            self.state = np.zeros_like(self.state)

        # order delta based on the euclidean distance and get state
        #eucl_dist = np.apply_along_axis(lambda x: np.sqrt(x[0]**2 + x[1]**2), 1, delta_normalized)
        #idx = np.argsort(eucl_dist)

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
                #if self.agent_y < self.vessel_y[0]:
                #    self.agent_y = self.vessel_y[0] # agent crahsed! 
                #    self.crash_flag = True

            while self.vessel_ttc[self.n_vessels_half+1] < 0:
                self._place_vessel(False, 1)           
                #if self.agent_y > self.vessel_y[self.n_vessels_half]:
                #    self.agent_y = self.vessel_y[self.n_vessels_half] # agent crahsed! 
                #    self.crash_flag = True
    
    def _move_agent(self, action):
        """Update self.agent_pos using a given action. For now: a_x = 0."""
        self.agent_ay_old = self.agent_ay

        # update lateral dynamics
        self.agent_ay = self.ay_max * action.item()

        agent_vy_new = np.clip(self.agent_vy + self.agent_ay * self.delta_t,-self.vy_max, self.vy_max)

        agent_y_new = self.agent_y + 0.5 * (self.agent_vy + agent_vy_new) * self.delta_t

        self.agent_vy = agent_vy_new
        self.agent_y = agent_y_new

        # update longitudinal dynamics
        self.agent_x= self.agent_x + self.agent_vx * self.delta_t
        
    def _calculate_reward(self):
        """Returns reward of the current state."""   
        # compute jerk reward
        jerk_reward = -10*(self.agent_ay_old - self.agent_ay)**2
        #jerk_reward = 0

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
            self.reward = self.reward.item()

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
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                self.ax1 = self.fig.add_subplot(self.gs[1, 0]) # state
                self.ax2 = self.fig.add_subplot(self.gs[1, 1]) # reward
                self.ax3 = self.fig.add_subplot(self.gs[0, 1]) # action
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
                self.ax3.old_time = 0
                self.ax3.old_action = 0
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
            self.ax1.set_xlabel("TTC-x")
            self.ax1.set_ylabel("y")
            self.ax1.scatter(self.vessel_ttc, self.vessel_y, color = self.vessel_color)
            y = self.AR1[self.current_timestep:]
            self.ax1.plot([i * self.delta_t for i in range(len(y))], y)

            # ---- REWARD PLOT ----
            if self.current_timestep == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
            self.ax2.hlines(-1, 1, self.x_max / (self.agent_vx * self.delta_t), linestyles="dashed")
            self.ax2.set_xlim(1, self.x_max / (self.agent_vx * self.delta_t))
            self.ax2.set_ylim(-1.1, 0.1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward")
            self.ax2.plot([self.ax2.old_time, self.current_timestep], [self.ax2.old_reward, self.reward], color = self.line_color)
            self.ax2.old_time = self.current_timestep
            self.ax2.old_reward = self.reward

            # ---- ACTION PLOT ----
            if self.current_timestep == 0:
                self.ax3.clear()
                self.ax3.old_time = 0
                self.ax3.old_action = 0
            self.ax3.set_xlim(1, self.x_max / (self.agent_vx * self.delta_t))
            self.ax3.set_ylim(-self.ay_max, self.ay_max)
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Action")
            self.ax3.plot([self.ax3.old_time, self.current_timestep], [self.ax3.old_action, self.agent_ay], color = self.line_color)
            self.ax3.old_time = self.current_timestep
            self.ax3.old_action = self.agent_ay
            
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
