import math

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from collections import deque


class ComplexOA(gym.Env):
    """Complex-OA Environment of Hart, Waltz, Okhrin (2021)."""
    
    def __init__(self, POMDP_type="MDP", frame_stack=1, n_vessels=12, max_temporal_dist=300, obst_traj = "stochastic"):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------

        assert POMDP_type in ["MDP", "RV", "FL"], "Unknown MDP/POMDP specification."
        assert frame_stack >= 1, "Frame stacking must be positive."
        assert obst_traj in ["constant", "stochastic"], "Unknown obstacle trajectory specification."

        self.POMDP_type   = POMDP_type
        self.frame_stack  = frame_stack
        self.obst_traj    = obst_traj
        self.FL_prob      = 0.1
        self.sort_obs_ttc = False

        # river size and vessel characteristics   
        self.y_max             = 500  # only for plotting
        self.n_vessels         = n_vessels
        self.n_vessels_half    = int(self.n_vessels/2)
        self.max_temporal_dist = max_temporal_dist  # maximum temporal distance when placing new vessel

        # initial agent position, speed and acceleration
        self.start_x_agent = 0
        self.start_y_agent = 0
        self.v_x_agent = 0
        self.v_y_agent = 0
        self.a_x_agent = 0
        self.a_y_agent = 0

        # speed and acceleration of vessels
        self.vx_max = 5
        self.vy_max = 5
        self.ax_max = 0
        self.ay_max = 0.01
        self.jerk_max = 0.001

        # maximum sight of agent
        self.delta_x_max = self.max_temporal_dist * self.vx_max
        self.delta_y_max = self.max_temporal_dist * self.vy_max
        self.R_scale = math.sqrt(self.delta_x_max**2 + self.delta_y_max**2)
        self.u_scale = math.sqrt(self.vx_max**2 + self.vy_max**2)

        # time step, max episode steps and length of river
        self.delta_t = 5
        self.current_timestep = 0
        self.max_episode_steps = 500
        
        # rendering
        self.plot_delay   = 0.001
        self.agent_color  = "red"
        self.vessel_color = np.full(self.n_vessels, "green")
        self.vessel_color[0:self.n_vessels_half] = "blue"
        self.line_color = "black"

        # reward config
        self.sd_y   = 25
        self.sd_ttc = 25

        # --------------------------------  gym inherits ---------------------------------------------------
        num_vessel_obs = 2 if self.POMDP_type == "RV" else 4

        super(ComplexOA, self).__init__()
        self.observation_space = spaces.Box(low =np.full((1, self.frame_stack * (num_vessel_obs * self.n_vessels + 2)), -1, dtype=np.float32)[0],
                                            high=np.full((1, self.frame_stack * (num_vessel_obs * self.n_vessels + 2)), 1, dtype=np.float32)[0])
        self.action_space = spaces.Box(low =np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))


    def reset(self):
        """Resets environment to initial state."""
        self.current_timestep = 0
        self.reward = 0
        self._set_AR1()
        self._set_dynamics()

        # CHANGE START
        # self.epi_info = np.empty(shape=(600, 39))
        # CHANGE END

        if self.frame_stack > 1:
            self.frame_hist_cnt = 0
            self.frame_array    = np.zeros((self.frame_stack, int(self.observation_space.shape[0] / self.frame_stack)))

        self._set_state()
        return self.state
    
    def _exponential_smoothing(self, x, alpha=0.03):
        s = np.zeros_like(x)

        for idx, x_val in enumerate(x):
            if idx == 0:
                s[idx] = x[idx]
            else:
                s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

        return s

    def _set_AR1(self):
        """Sets the AR1 array containing the desired lateral trajectory for all episode steps."""
        self.AR1 = np.zeros(self.max_episode_steps + int(self.n_vessels_half * self.max_temporal_dist / self.delta_t), dtype=np.float32) 
        for i in range(self.AR1.size-1):
            self.AR1[i+1] = self.AR1[i] * 0.99 + np.random.normal(0, math.sqrt(800))

        # smooth data
        self.AR1 = self._exponential_smoothing(self.AR1, alpha=0.03)

    def _set_dynamics(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        self.agent_x = self.start_x_agent
        self.agent_y = self.start_y_agent
        self.agent_vx = np.random.uniform(1,self.vx_max)
        self.x_max = np.ceil(self.agent_vx * self.max_episode_steps * self.delta_t) # set window length by chosen longitudinal speed
        self.agent_vy = self.v_y_agent
        self.agent_ax = self.a_x_agent
        self.agent_ay = self.a_y_agent 
        self.agent_ay_old = self.a_y_agent
        
        self.vessel_x = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_y = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_vx = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_vy = np.empty((self.n_vessels), dtype=np.float32)
        self.vessel_ttc = np.empty((self.n_vessels), dtype=np.float32)

        if self.obst_traj == "stochastic":
            self.vessel_x_traj_low  = deque([None]*self.n_vessels_half)
            self.vessel_y_traj_low  = deque([None]*self.n_vessels_half)
            self.vessel_vx_traj_low = deque([None]*self.n_vessels_half)
            self.vessel_vy_traj_low = deque([None]*self.n_vessels_half)
            self.vessel_x_traj_up   = deque([None]*self.n_vessels_half)
            self.vessel_y_traj_up   = deque([None]*self.n_vessels_half)
            self.vessel_vx_traj_up  = deque([None]*self.n_vessels_half)
            self.vessel_vy_traj_up  = deque([None]*self.n_vessels_half)

        # find initial vessel position
        self._place_vessel(True,-1)
        self._place_vessel(True, 1)

        for _ in range(int(self.n_vessels/2-1)):
            self._place_vessel(False,-1)
            self._place_vessel(False, 1)

    def _place_vessel(self, initial_placement, vessel_direction):
        if vessel_direction == -1:
            ttc = self.vessel_ttc[:self.n_vessels_half]
            x = self.vessel_x[:self.n_vessels_half]
            y = self.vessel_y[:self.n_vessels_half]
            vx = self.vessel_vx[:self.n_vessels_half]
            vy = self.vessel_vy[:self.n_vessels_half]

            if self.obst_traj == "stochastic":
                x_traj = self.vessel_x_traj_low
                y_traj = self.vessel_y_traj_low
                vx_traj = self.vessel_vx_traj_low
                vy_traj = self.vessel_vy_traj_low
        else:
            ttc = self.vessel_ttc[self.n_vessels_half:]
            x = self.vessel_x[self.n_vessels_half:]
            y = self.vessel_y[self.n_vessels_half:]
            vx = self.vessel_vx[self.n_vessels_half:]
            vy = self.vessel_vy[self.n_vessels_half:]

            if self.obst_traj == "stochastic":
                x_traj = self.vessel_x_traj_up
                y_traj = self.vessel_y_traj_up
                vx_traj = self.vessel_vx_traj_up
                vy_traj = self.vessel_vy_traj_up        

        # compute new ttc
        if initial_placement:
            new_ttc = np.random.uniform(-self.max_temporal_dist, -1)
        else:
            new_ttc = np.maximum(1,ttc[-1] +  np.random.uniform(0,self.max_temporal_dist))

        # compute new vessel dynamics
        y_future = self.AR1[abs(int(self.current_timestep + new_ttc/self.delta_t))] + vessel_direction * np.maximum(40, np.random.normal(100,50))
        
        new_vx = np.random.uniform(-self.vx_max, self.vx_max)
        new_vy = np.random.uniform(-self.vy_max, self.vy_max)

        if self.obst_traj == "constant":
            new_x = (self.agent_vx - new_vx) * new_ttc + self.agent_x
            new_y = y_future - new_vy * new_ttc
        
        else:
            x_future = self.agent_x + self.agent_vx * new_ttc
            timestepsToCollision = int(np.maximum(0,new_ttc/self.delta_t))
            x_AR1 = np.zeros(self.max_temporal_dist + timestepsToCollision, dtype=np.float32) 
            y_AR1 = np.zeros(self.max_temporal_dist + timestepsToCollision, dtype=np.float32)
            x_const = np.zeros(self.max_temporal_dist + timestepsToCollision, dtype=np.float32) 
            y_const = np.zeros(self.max_temporal_dist + timestepsToCollision, dtype=np.float32)             
            for i in range(x_AR1.size-1):
                x_AR1[i+1] = x_AR1[i] * 0.99 + np.random.normal(0,np.sqrt(800))
                y_AR1[i+1] = y_AR1[i] * 0.99 + np.random.normal(0,np.sqrt(800))
                x_const[i+1] = x_const[i] + new_vx * self.delta_t
                y_const[i+1] = y_const[i] + new_vy * self.delta_t

            x_traj.rotate(-1) 
            y_traj.rotate(-1) 
            vx_traj.rotate(-1) 
            vy_traj.rotate(-1)   
            x_AR1 = self._exponential_smoothing(x_AR1, alpha=0.03)
            y_AR1 = self._exponential_smoothing(y_AR1, alpha=0.03)
            temp_x = x_AR1 + x_future - x_AR1[timestepsToCollision] + x_const - x_const[timestepsToCollision]
            temp_y = y_AR1 + y_future - y_AR1[timestepsToCollision] + y_const - y_const[timestepsToCollision]
            x_traj[-1] = temp_x
            y_traj[-1] = temp_y
            vx_traj[-1] = np.diff(temp_x)/self.delta_t
            vy_traj[-1] = np.diff(temp_y)/self.delta_t

            new_x = x_traj[-1][0]
            new_y = y_traj[-1][0]
            new_vx = vx_traj[-1][0]
            new_vy = vy_traj[-1][0]
 

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
            if self.obst_traj == "stochastic":
               self.vessel_x_traj_low  = x_traj  
               self.vessel_y_traj_low  = y_traj  
               self.vessel_vx_traj_low = vx_traj  
               self.vessel_vy_traj_low = vy_traj             
        else:
            self.vessel_ttc[self.n_vessels_half:] = ttc
            self.vessel_x[self.n_vessels_half:] = x
            self.vessel_y[self.n_vessels_half:] = y
            self.vessel_vx[self.n_vessels_half:] = vx
            self.vessel_vy[self.n_vessels_half:] = vy
            if self.obst_traj == "stochastic":
               self.vessel_x_traj_up  = x_traj  
               self.vessel_y_traj_up  = y_traj  
               self.vessel_vx_traj_up = vx_traj  
               self.vessel_vy_traj_up = vy_traj  
    
    def _set_state(self):
        """Sets state which is flattened, ordered with ascending TTC, normalized and clipped to [-1, 1]"""
#        if self.sort_obs_ttc:
#            # arrays are already sorted according ascending ttc
#            x = self.vessel_x.copy()
#            y = self.vessel_y.copy()
#            vx = self.vessel_vx.copy()
#            vy = self.vessel_vy.copy()
        
        # compute sorting index array for asceding euclidean distance
        eucl_dist = (self.vessel_x - self.agent_x)**2 + (self.vessel_y - self.agent_y)**2
        idx1 = np.argsort(eucl_dist[:self.n_vessels_half])
        idx2 = np.argsort(eucl_dist[self.n_vessels_half:]) + self.n_vessels_half
        idx = np.concatenate([idx1, idx2])

        eucl_dist = eucl_dist[idx].copy()
        x = self.vessel_x[idx].copy()
        y = self.vessel_y[idx].copy()
        vx = self.vessel_vx[idx].copy()
        vy = self.vessel_vy[idx].copy()

        ############## CARTEESIAN STATE INPUT ###############

        #  # state definition
        # self.state = np.array([self.agent_ay/self.ay_max, self.agent_vy/self.vy_max])
        # self.state = np.append(self.state, (self.agent_x  - x)/self.delta_x_max)
        # self.state = np.append(self.state, (self.agent_y  - y)/self.delta_y_max)

        # # POMDP specs
        # if self.POMDP_type in ["MDP", "FL"]:
        #     v_obs = (self.agent_vx - vx)/(2*self.vx_max)
        #     v_obs = np.append(v_obs, (self.agent_vy - vy)/(2*self.vy_max))
                              
        #     self.state = np.append(self.state, v_obs)

        # if self.POMDP_type == "FL" and np.random.binomial(1, self.FL_prob) == 1:
        #     self.state = np.zeros_like(self.state)


        ############## POLAR STATE INPUT ###############
        
        phi = np.arctan2(vy,vx)                                     # heading of obstacles
        agent_phi = np.arctan2(self.agent_y,self.agent_x)           # heading of agent
        u = np.sqrt(vx**2 + vy**2)                                  # absolute speed of obstacles
        agent_u = np.sqrt(self.agent_vx**2 + self.agent_vy**2)      # absolute speed of agent

        theta  = np.arctan2(y-self.agent_y, x-self.agent_x) - agent_phi    # direction of obstacle position in agent's body frame
        theta2 = np.arctan2(self.agent_y-y, self.agent_x-x) - phi          # moving direction of agent with respect to obstacle


        # state definition
        self.state = np.array([self.agent_ay/self.ay_max,
                               agent_u/self.u_scale])
        self.state = np.append(self.state, eucl_dist/self.R_scale)
        self.state = np.append(self.state,  theta/np.pi)

        # POMDP specs
        if self.POMDP_type in ["MDP", "FL"]:
            v_obs = np.array([u/self.u_scale,
                               theta2/np.pi])
            self.state = np.append(self.state, v_obs)

        if self.POMDP_type == "FL" and np.random.binomial(1, self.FL_prob) == 1:
            self.state = np.zeros_like(self.state)

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
        self.crash_flag = False
        self._move_vessel()
        self._move_agent(action)
        self._set_state()
        self._calculate_reward()
        done = self._done()

        # CHANGES START
        #tmp = np.append(np.append(np.array([self.agent_x, self.agent_y]), np.append(self.vessel_x, self.vessel_y)), self.vessel_ttc)
        #self.epi_info[self.current_timestep] = np.append(tmp, np.array([self.reward]))
        #np.savetxt("epi_info.csv", self.epi_info, delimiter=" ")
        # CHANGES END
        
        self.current_timestep += 1

        return self.state, self.reward, done, {}
    
    def _move_vessel(self):
        """Updates positions, velocities and accelerations of vessels. For now accelerations are constant.
        Used approximation: Euler-Cromer method, that is v_(n+1) = v_n + a_n * t and x_(n+1) = x_n + v_(n+1) * t."""

        for i in range(self.n_vessels):
            if self.obst_traj == "constant":
                # lateral dynamics
                self.vessel_y[i] = self.vessel_y[i] + self.vessel_vy[i] * self.delta_t

                # longitudinal dynamics     
                self.vessel_x[i] = self.vessel_x[i] + self.vessel_vx[i] * self.delta_t
            else:
                if i < self.n_vessels_half:
                    self.vessel_x[i]  = self.vessel_x_traj_low[i][0]
                    self.vessel_y[i]  = self.vessel_y_traj_low[i][0]
                    self.vessel_vx[i] = self.vessel_vx_traj_low[i][0]
                    self.vessel_vy[i] = self.vessel_vy_traj_low[i][0]
                    self.vessel_x_traj_low[i]  = np.delete(self.vessel_x_traj_low[i], 0)
                    self.vessel_y_traj_low[i]  = np.delete(self.vessel_y_traj_low[i], 0)
                    self.vessel_vx_traj_low[i] = np.delete(self.vessel_vx_traj_low[i], 0)
                    self.vessel_vy_traj_low[i] = np.delete(self.vessel_vy_traj_low[i], 0)

                else:
                    temp_index = i-self.n_vessels_half
                    self.vessel_x[i]  = self.vessel_x_traj_up[temp_index][0]
                    self.vessel_y[i]  = self.vessel_y_traj_up[temp_index][0]
                    self.vessel_vx[i] = self.vessel_vx_traj_up[temp_index][0]
                    self.vessel_vy[i] = self.vessel_vy_traj_up[temp_index][0]  
                    self.vessel_x_traj_up[temp_index]  = np.delete(self.vessel_x_traj_up[temp_index], 0)
                    self.vessel_y_traj_up[temp_index]  = np.delete(self.vessel_y_traj_up[temp_index], 0)
                    self.vessel_vx_traj_up[temp_index] = np.delete(self.vessel_vx_traj_up[temp_index], 0)
                    self.vessel_vy_traj_up[temp_index] = np.delete(self.vessel_vy_traj_up[temp_index], 0)                                      
            self.vessel_ttc[i] -= self.delta_t
            
            # replace vessel if necessary       
            while self.vessel_ttc[1] < 0:
                self._place_vessel(False,-1)

            while self.vessel_ttc[self.n_vessels_half+1] < 0:
                self._place_vessel(False, 1)           

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

        vess_reward1 = 0
        vess_reward2 = 0

        for i in range(self.n_vessels_half):
            vess_reward1 = np.maximum(vess_reward1, 
                                      self._norm_pdf(self.vessel_ttc[i],0,self.sd_ttc)/self._norm_pdf(0,0,self.sd_ttc) *
                                      self._norm_pdf(np.maximum(0,self.agent_y-self.vessel_y[i]),0,self.sd_y)/self._norm_pdf(0,0,self.sd_y))
            vess_reward2 = np.maximum(vess_reward2, 
                                      self._norm_pdf(self.vessel_ttc[self.n_vessels_half+i],0,self.sd_ttc)/self._norm_pdf(0,0,self.sd_ttc) *
                                      self._norm_pdf(np.maximum(0,self.vessel_y[self.n_vessels_half+i]-self.agent_y),0,self.sd_y)/self._norm_pdf(0,0,self.sd_y))
        #self.reward = - (np.abs(vess_reward1-vess_reward2))/(np.maximum(-vess_reward1,-vess_reward2)+1)
        self.reward = - np.maximum(vess_reward1, vess_reward2)
        self.reward = self.reward.item()

    def _norm_pdf(self, x, mu, sd):
        return 1 / math.sqrt(2 * math.pi * sd**2) * math.exp(-(x - mu)**2 / (2*sd**2))

    def _done(self):
        """Returns boolean flag whether episode is over."""
        return False

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
            #self.ax0.clear()
            self.ax0.set_xlim(-1500, 5000)
            self.ax0.set_ylim(-2000, 2000)
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
            self.ax3.set_ylabel("Agent a_y")
            self.ax3.plot([self.ax3.old_time, self.current_timestep], [self.ax3.old_action, self.agent_ay], color = self.line_color)
            self.ax3.old_time = self.current_timestep
            self.ax3.old_action = self.agent_ay
            
            # delay plotting for ease of user
            plt.pause(self.plot_delay)
