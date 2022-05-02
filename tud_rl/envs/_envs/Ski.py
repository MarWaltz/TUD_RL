import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt


class Ski(gym.Env):
    """Class environment with initializer, step, reset and render method."""
    
    def __init__(self, POMDP_type="MDP", frame_stack=1):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------
        assert POMDP_type in ["MDP", "RV", "FL"], "Unknown MDP/POMDP specification."
        assert frame_stack >= 1, "Frame stacking must be positive."
        
        self.POMDP_type  = POMDP_type
        self.frame_stack = frame_stack

        self.goalwidth = 80
        self.max_goal_end_y = 200
        self.max_temporal_dist = 300 # maximal temporal distance when placing new vessel

        # speed and acceleration of vessels
        self.vx_max = 5
        self.vy_max = 5
        self.ax_max = 0
        self.ay_max = 0.01
        self.jerk_max = 0.001

        # maximum sight of agent
        self.delta_x_max = self.max_temporal_dist *  self.vx_max 
        self.delta_y_max = self.max_temporal_dist *  self.vy_max + self.max_goal_end_y

        # time step, max episode steps and length of river
        self.delta_t = 5
        self.current_timestep = 0
        
        # rendering
        self.plot_delay = 0.001

        # reward config
        #self.variance_x = 12000
        self.variance_y = 25
        #self.variance_ttc = 25
        
        # --------------------------------  gym inherits ---------------------------------------------------
        if self.POMDP_type == "RV":
            num_vessel_obs = 2
        else:
            num_vessel_obs = 4

        super(Ski, self).__init__()
        self.observation_space = spaces.Box(low=np.full((1, self.frame_stack * (num_vessel_obs + 2)), -1, dtype=np.float32)[0],
                                            high=np.full((1, self.frame_stack * (num_vessel_obs + 2)), 1, dtype=np.float32)[0])
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        
        # --------------------------------- custom inits ---------------------------------------------------

    def reset(self):
        """Resets environment to initial state."""
        self.current_timestep = 0
        self.reward = 0
        self._set_dynamics()

        # CHANGE START
        self.epi_info = np.empty(shape=(600, 4))
        # CHANGE END

        if self.frame_stack > 1:
            self.frame_hist_cnt = 0
            self.frame_array    = np.zeros((self.frame_stack, int(self.observation_space.shape[0] / self.frame_stack)))

        self._set_state()
        return self.state
   
    def _set_dynamics(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        self.agent_x = 0
        self.agent_y = 0
        self.agent_vx = np.random.uniform(1,self.vx_max)
        self.agent_vy = 0
        self.agent_ax = 0
        self.agent_ay = 0 
        self.agent_ay_old = 0
        
        TTC = (np.random.uniform(-20, 20) + self.max_temporal_dist)
        self.obst_x =  self.agent_vx * TTC
        self.obst_y_future = np.random.uniform(-self.max_goal_end_y, self.max_goal_end_y )
        self.obst_vy = np.random.uniform(-self.vy_max, self.vy_max)
        self.obst_y = self.obst_y_future - self.obst_vy * TTC
    
    def _set_state(self):
        """Sets state which is flattened, ordered with ascending TTC, normalized and clipped to [-1, 1]"""        
        self.state = np.array([self.agent_ay/self.ay_max,
                               self.agent_vy/self.vy_max,
                               (self.agent_x - self.obst_x)/self.delta_x_max,
                               (self.agent_y - self.obst_y)/self.delta_y_max])

        # POMDP specs
        if self.POMDP_type == "MDP":
            v_obs = np.array([(self.agent_vy - self.obst_vy)/(2*self.vy_max),
                               self.agent_vx/self.vx_max])
            self.state = np.append(self.state, v_obs)
        
        # frame stacking
        if self.frame_stack > 1:
            
            if self.frame_hist_cnt == self.frame_stack:
                self.frame_array = np.roll(self.frame_array, shift = -1, axis = 0)
                self.frame_array[self.frame_stack - 1, :] = self.state
            else:
                self.frame_array[self.frame_hist_cnt] = self.state
                self.frame_hist_cnt += 1
            
            self.state = self.frame_array.flatten()

    def step(self, action):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done."""
        self._move_vessel()
        self._move_agent(action)
        self._set_state()
        done = self._done()
        if done:
            self._calculate_reward()

        # CHANGES START
        self.epi_info[self.current_timestep] = np.array([self.agent_x, self.agent_y, self.obst_x, self.obst_y])
        #np.savetxt("epi_info_Ski.csv", self.epi_info, delimiter=" ")
        # CHANGES END

        self.current_timestep += 1
        
        return self.state, self.reward, done, {}
    
    def _move_vessel(self):
        """Updates positions, velocities and accelerations of vessels. For now accelerations are constant.
        Used approximation: Euler-Cromer method, that is v_(n+1) = v_n + a_n * t and x_(n+1) = x_n + v_(n+1) * t."""
        
        # lateral dynamics    
        self.obst_y = self.obst_y + self.obst_vy * self.delta_t
    
    def _move_agent(self, action):
        """Update self.agent_pos using a given action. For now: a_x = 0."""
        self.agent_ay_old = self.agent_ay

        # update lateral dynamics
        self.agent_ay = action.item() * self.ay_max
        agent_vy_new = np.clip(self.agent_vy + self.agent_ay * self.delta_t,-self.vy_max, self.vy_max)

        agent_y_new = self.agent_y + 0.5 * (self.agent_vy + agent_vy_new) * self.delta_t

        self.agent_vy = agent_vy_new
        self.agent_y = agent_y_new

        # update longitudinal dynamics
        self.agent_x = self.agent_x + self.agent_vx * self.delta_t

    def _calculate_reward(self):
        """Returns reward of the current state."""   
        if abs(self.agent_y - self.obst_y) < self.goalwidth/2:
            self.reward = 100
        else:
            self.reward = -100
    
    def _done(self):
        """Returns boolean flag whether episode is over."""
        return True if self.agent_x >= self.obst_x else False
    
    def render(self, agent_name=None, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep
        if self.current_timestep % 2 == 0: 

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
            self.ax0.set_xlim(0, self.delta_x_max)
            self.ax0.set_ylim(-self.max_goal_end_y*1.1, self.max_goal_end_y*1.1)
            self.ax0.set_xlabel("x")
            self.ax0.set_ylabel("y")
            if agent_name is not None:
                self.ax0.set_title(agent_name)

            # set agent and vessels
            self.ax0.scatter(self.agent_x, self.agent_y, color = "red")
            self.ax0.scatter(self.obst_x, self.obst_y + self.goalwidth/2, color = "green")
            self.ax0.scatter(self.obst_x, self.obst_y - self.goalwidth/2, color = "green")
            self.ax0.scatter(self.obst_x, self.obst_y_future + self.goalwidth/2, color = "yellow")
            self.ax0.scatter(self.obst_x, self.obst_y_future - self.goalwidth/2, color = "yellow")            
            
            # ---- STATE PLOT ----
            # clear prior axes, set limits
            self.ax1.clear()
            self.ax1.set_xlim(-200, 1000)
            self.ax0.set_ylim(-self.max_goal_end_y*1.1, self.max_goal_end_y*1.1)
            
            # add agent and states
            self.ax1.scatter(0, self.agent_y, color = "red")
            self.ax1.set_xlabel("TTC-x")
            self.ax1.set_ylabel("y")
 

            # ---- REWARD PLOT ----
            if self.current_timestep == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_reward = 0
            self.ax2.set_xlim(1, self.max_temporal_dist + 20)
            self.ax2.set_ylim(0, 100)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward")
            self.ax2.plot([self.ax2.old_time, self.current_timestep], [self.ax2.old_reward, self.reward], color = "black")
            self.ax2.old_time = self.current_timestep
            self.ax2.old_reward = self.reward

            # ---- ACTION PLOT ----
            if self.current_timestep == 0:
                self.ax3.clear()
                self.ax3.old_time = 0
                self.ax3.old_action = 0
            self.ax2.set_xlim(1, self.max_temporal_dist + 20)
            self.ax3.set_ylim(-self.ay_max, self.ay_max)
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Agent a_y")
            self.ax3.plot([self.ax3.old_time, self.current_timestep], [self.ax3.old_action, self.agent_ay], color = "black")
            self.ax3.old_time = self.current_timestep
            self.ax3.old_action = self.agent_ay
            
            # delay plotting for ease of user
            plt.pause(self.plot_delay)
