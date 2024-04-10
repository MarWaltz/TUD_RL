import math
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from matplotlib import pyplot as plt


class CPANet(nn.Module):
    """Defines a recurrent network to predict DCPA and TCPA."""
    
    def __init__(self) -> None:
        super(CPANet, self).__init__()

        # memory
        self.mem_LSTM = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(64, 64)
        
        # post combination
        self.dense2 = nn.Linear(64, 2)

    def forward(self, x0, y0) -> tuple:
        #------ memory ------
        # LSTM
        _, (mem, _) = self.mem_LSTM(torch.cat([x0, y0], dim=2))
        mem = mem[0]

        # dense
        x = F.relu(self.dense1(mem))
        
        # final dense layers
        x = self.dense2(x)
        return x


class ObstacleAvoidance(gym.Env):
    def __init__(self, 
                 setting="STD", 
                 frame_stack=1, 
                 n_vessels=10, 
                 max_temporal_dist=400, 
                 min_channel_width=100, 
                 amplitude_factor=1.0,
                 obst_traj = "stochastic"):
        
        # ----------------------------- settings and hyperparameter -----------------------------------------
        assert setting in ["STD", "EST"], "Unknown setting."
        assert frame_stack >= 1, "Frame stacking must be positive."
        assert obst_traj in ["constant", "stochastic", "sinus"], "Unknown obstacle trajectory specification."

        self.setting     = setting
        self.frame_stack    = frame_stack
        self.obst_traj      = obst_traj
        self.FL_prob        = 0.1
        self.sort_obs_ttc   = False

        # river size and vessel characteristics   
        self.y_max = 500 # only for plotting
        self.n_vessels  = n_vessels
        self.n_vessels_half  = int(self.n_vessels/2)
        self.max_temporal_dist = max_temporal_dist # maximal temporal distance when placing new vessel
        self.min_channel_width = min_channel_width
        self.amplitude_factor = amplitude_factor

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

        # CPA estimator parameters
        self.FORECAST_WIN = 100 # window length of obstacle trajectory estimation (past and future)
        self.N_OBS_MAX = 50 # max last observations to consider when estimating obstacle trajectories
        self.dcpa_scale  = 500
        self.tcpa_scale = 400

        # agent observation parameters
        self.delta_x_max = self.max_temporal_dist * self.vx_max
        self.delta_y_max = self.max_temporal_dist * self.vy_max
        self.R_scale = np.sqrt(self.delta_x_max**2 + self.delta_y_max**2)
        self.u_scale = np.sqrt(self.vx_max**2 + self.vy_max**2)
        self.N_pastObs = 20 # to get N_pastObs-1 differential observations

        # time step, max episode steps and length of river
        self.delta_t = 5
        self.current_timestep = 0
        self.max_episode_steps = 500
        
        # rendering
        self.plot_delay   = 0.001
        self.agent_color  = "red"
        self.vessel_color = np.full(self.n_vessels,"green")
        self.vessel_color[0:self.n_vessels_half] = "blue"
        self.line_color = "black"

        # reward config
        self.sd_y = 25
        self.sd_ttc = 25
        
        # --------------------------------  gym inherits ---------------------------------------------------
        if self.setting == "EST":
            num_vessel_obs = 4
        else:
            num_vessel_obs = 2

        super(ObstacleAvoidance, self).__init__()
        self.observation_space = spaces.Box(low=np.full((1, self.frame_stack * (num_vessel_obs * self.n_vessels + 3)), -1, dtype=np.float32)[0],
                                            high=np.full((1, self.frame_stack * (num_vessel_obs * self.n_vessels + 3)), 1, dtype=np.float32)[0])
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        
        # --------------------------------- custom inits ---------------------------------------------------

    def reset(self):
        """Resets environment to initial state."""
        self.current_timestep = 0
        self.reward = 0
        self._set_AR1()
        self._set_dynamics()

        if self.setting == "EST":
            self.net = CPANet()

            if self.obst_traj == "stochastic":
                self.net.load_state_dict(torch.load("cpa_weights_AR1.pth"))
            else:
                self.net.load_state_dict(torch.load("cpa_weights_sinus.pth"))

            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad = False

        # save data to file
        #self.epi_info = np.empty(shape=(600, 33))

        if self.frame_stack > 1:
            self.frame_hist_cnt = 0
            self.frame_array = np.zeros((self.frame_stack, int(self.observation_space.shape[0] / self.frame_stack)))

        self._set_state()
        return self.state
    
    def _exponential_smoothing(self, x : np.ndarray, alpha=0.03) -> np.ndarray:
        s = np.zeros_like(x)
        for idx, x_val in enumerate(x):
            if idx == 0:
                s[idx] = x[idx]
            else:
                s[idx] = alpha * x_val + (1-alpha) * s[idx-1]
        return s

    def _set_AR1(self):
        """Sets the AR1 Array containing the desired lateral trajectory for all episode steps."""
        self.AR1 = np.zeros(self.max_episode_steps + int(self.n_vessels_half * self.max_temporal_dist / self.delta_t), dtype=np.float32) 
        for i in range(self.AR1.size-1):
            self.AR1[i+1] = self.AR1[i] * 0.99 + np.random.normal(0,np.sqrt(800))

        # smooth data
        self.AR1 = self._exponential_smoothing(self.AR1)

    def _set_dynamics(self):
        """Initializes positions, velocity and acceleration of agent and vessels."""
        self.agent_x = self.start_x_agent
        self.agent_y = self.start_y_agent
        #self.agent_vx = np.random.uniform(1,self.vx_max)
        self.agent_vx = 5
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
        self.tcpa_est = np.empty((self.n_vessels), dtype=np.float32)
        self.dcpa_est = np.empty((self.n_vessels), dtype=np.float32)

        if self.obst_traj == "stochastic" or self.obst_traj == "sinus":
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
            ttc = self.vessel_ttc[:self.n_vessels_half].copy()
            x = self.vessel_x[:self.n_vessels_half].copy()
            y = self.vessel_y[:self.n_vessels_half].copy()
            vx = self.vessel_vx[:self.n_vessels_half].copy()
            vy = self.vessel_vy[:self.n_vessels_half].copy()
            tcpa_est = self.tcpa_est[:self.n_vessels_half].copy()
            dcpa_est = self.dcpa_est[:self.n_vessels_half].copy()
            if self.obst_traj == "stochastic" or self.obst_traj == "sinus":
                x_traj = self.vessel_x_traj_low
                y_traj = self.vessel_y_traj_low
                vx_traj = self.vessel_vx_traj_low
                vy_traj = self.vessel_vy_traj_low
        else:
            ttc = self.vessel_ttc[self.n_vessels_half:].copy()
            x = self.vessel_x[self.n_vessels_half:].copy()
            y = self.vessel_y[self.n_vessels_half:].copy()
            vx = self.vessel_vx[self.n_vessels_half:].copy()
            vy = self.vessel_vy[self.n_vessels_half:].copy()
            tcpa_est = self.tcpa_est[self.n_vessels_half:].copy()
            dcpa_est = self.dcpa_est[self.n_vessels_half:].copy()            
            if self.obst_traj == "stochastic" or self.obst_traj == "sinus":
                x_traj = self.vessel_x_traj_up
                y_traj = self.vessel_y_traj_up
                vx_traj = self.vessel_vx_traj_up
                vy_traj = self.vessel_vy_traj_up        

        # compute new ttc
        if initial_placement:
            new_ttc = np.random.uniform(-self.max_temporal_dist, -1)
        else:
            new_ttc = np.maximum(1,ttc[-1] +  np.random.uniform(self.max_temporal_dist*0.25,self.max_temporal_dist))

        # compute new vessel dynamics
        y_future = self.AR1[abs(int(self.current_timestep + new_ttc/self.delta_t))] + vessel_direction * np.maximum(self.min_channel_width, np.random.normal(100,50))
        
        new_vx = 0
        new_vy = 0
        while np.sqrt(new_vx**2 + new_vy**2) < 2:
            new_vx = np.random.uniform(-self.vx_max, 0)
            new_vy = np.random.uniform(-self.vy_max, self.vy_max)

        if self.obst_traj == "constant":
            new_x = (self.agent_vx - new_vx) * new_ttc + self.agent_x
            new_y = y_future - new_vy * new_ttc
        
        else:
             # stochastic trajectory
            x_future = self.agent_x + self.agent_vx * new_ttc
            timestepsToCollision = int(np.maximum(0,new_ttc/self.delta_t))

             # constant part
            temp_vector = np.arange(0,self.max_temporal_dist + timestepsToCollision + self.N_pastObs,1, dtype=np.float32)
            x_const = temp_vector * self.delta_t * new_vx 
            y_const = temp_vector * self.delta_t * new_vy 

            if self.obst_traj == "stochastic":
                # initialize trajectory
                x_AR1 = np.zeros(self.max_temporal_dist + timestepsToCollision + self.N_pastObs, dtype=np.float32) 
                y_AR1 = np.zeros(self.max_temporal_dist + timestepsToCollision + self.N_pastObs, dtype=np.float32)                          

                # compute stochastic trajectory   
                for i in range(x_AR1.size-1):
                    # stochastic part
                    x_AR1[i+1] = x_AR1[i] * 0.9 + np.random.normal(0,15)
                    y_AR1[i+1] = y_AR1[i] * 0.9 + np.random.normal(0,15)  

                # smoothing and scaling trajectory
                alpha = np.sqrt(new_vx**2+new_vy**2)/ np.sqrt(self.vx_max**2+self.vy_max**2)
                x_stoch = alpha * self._exponential_smoothing(x_AR1, 0.8) 
                y_stoch = alpha * self._exponential_smoothing(y_AR1, 0.8)
            
            elif self.obst_traj == "sinus":
                 x_stoch =  self.amplitude_factor * 15 * new_vy *np.sin(temp_vector*2*np.pi / (2*10)) + (new_vx * np.random.normal(0,4, temp_vector.size)).astype('float32')
                 y_stoch = -self.amplitude_factor * 15 * new_vx *np.sin(temp_vector*2*np.pi / (2*10)) + (new_vy * np.random.normal(0,4, temp_vector.size)).astype('float32')

            # move whole trajectory to fit with current position and add constant and stochastic part
            temp_x = x_stoch + x_future - y_stoch[timestepsToCollision + self.N_pastObs] + x_const - x_const[timestepsToCollision + self.N_pastObs]
            temp_y = y_stoch + y_future - x_stoch[timestepsToCollision + self.N_pastObs] + y_const - y_const[timestepsToCollision + self.N_pastObs]

            x_traj.rotate(-1) 
            y_traj.rotate(-1) 
            vx_traj.rotate(-1) 
            vy_traj.rotate(-1)  
            
            # save current trajectory at the end of traj array
            x_traj[-1] = temp_x
            y_traj[-1] = temp_y
            vx_traj[-1] = np.diff(temp_x)/self.delta_t
            vy_traj[-1] = np.diff(temp_y)/self.delta_t

            new_x = x_traj[-1][self.N_pastObs-1]
            new_y = y_traj[-1][self.N_pastObs-1]
            new_vx = vx_traj[-1][self.N_pastObs-1]
            new_vy = vy_traj[-1][self.N_pastObs-1]
 
        # rotate dynamic arrays to place new vessel at the end
        ttc = np.roll(ttc,-1)
        x = np.roll(x,-1)
        y = np.roll(y,-1)
        vx = np.roll(vx,-1)
        vy = np.roll(vy,-1)
        tcpa_est = np.roll(tcpa_est,-1)
        dcpa_est = np.roll(dcpa_est,-1)

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
            self.tcpa_est[:self.n_vessels_half] = tcpa_est
            self.dcpa_est[:self.n_vessels_half] = dcpa_est
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
            self.tcpa_est[self.n_vessels_half:] = tcpa_est
            self.dcpa_est[self.n_vessels_half:] = dcpa_est            
            if self.obst_traj == "stochastic":
               self.vessel_x_traj_up  = x_traj  
               self.vessel_y_traj_up  = y_traj  
               self.vessel_vx_traj_up = vx_traj  
               self.vessel_vy_traj_up = vy_traj  
    
    def _set_state(self):
        """Sets state which is flattened, ordered with ascending TTC, normalized and clipped to [-1, 1]"""
        if self.setting == "EST":

            # NN estimator   
            #x = torch.cat((torch.tensor([el[:self.N_pastObs] for el in self.vessel_x_traj_low]),
            #               torch.tensor([el[:self.N_pastObs] for el in self.vessel_x_traj_up])),0)
            #y = torch.cat((torch.tensor([el[:self.N_pastObs] for el in self.vessel_y_traj_low]),
            #               torch.tensor([el[:self.N_pastObs] for el in self.vessel_y_traj_up])),0)  
            x = torch.tensor(np.concatenate(
                (np.array([el[:self.N_pastObs] for el in self.vessel_x_traj_low]),
                 np.array([el[:self.N_pastObs] for el in self.vessel_x_traj_up]))
                )
            )
            y = torch.tensor(np.concatenate(
                (np.array([el[:self.N_pastObs] for el in self.vessel_y_traj_low]),
                 np.array([el[:self.N_pastObs] for el in self.vessel_y_traj_up]))
                )
            )

            agent_vx = torch.tensor(self.agent_vx)
            agent_vy = torch.tensor(self.agent_vy)
           
            t_vector = 1 + torch.arange(-self.FORECAST_WIN - self.N_pastObs, self.FORECAST_WIN).unsqueeze(0)

            x_agent = self.agent_x + agent_vx * self.delta_t * t_vector
            y_agent = self.agent_y + agent_vy * self.delta_t * t_vector

            # NN based forcasting
            if self.current_timestep == 1 or self.current_timestep % 10 == 0: 
                for i in range(self.FORECAST_WIN):

                    # extract last N_OBS trajectory points
                    n_input_estimator = np.minimum(x.size(0)-1,self.N_OBS_MAX)
                    x_diff_in = torch.diff(x[-(n_input_estimator+1):])
                    y_diff_in = torch.diff(y[-(n_input_estimator+1):])  

                    # forward
                    est = self.net(x_diff_in.unsqueeze(2), y_diff_in.unsqueeze(2))
                    
                    # add point to trajectory tensor
                    new_x = x[:,-1] + est[:,0]
                    new_y = y[:,-1] + est[:,1]
                    x = torch.cat([x, new_x.unsqueeze(1)],1)
                    y = torch.cat([y, new_y.unsqueeze(1)],1)

                    # estimate additional point in the past
                    x_diff_in = -torch.flip(x_diff_in,[1,])
                    y_diff_in = -torch.flip(y_diff_in,[1,])

                    # forward
                    est = self.net(x_diff_in.unsqueeze(2), y_diff_in.unsqueeze(2))
                    
                    # add point to trajectory tensor
                    new_x = x[:, 0] + est[:,0]
                    new_y = y[:, 0] + est[:,1]
                    x = torch.cat([new_x.unsqueeze(1), x],1)
                    y = torch.cat([new_y.unsqueeze(1), y],1)              

                # to numpy 
                x = x.detach().numpy()
                y = y.detach().numpy()       
                x_agent = x_agent.detach().numpy()
                y_agent = y_agent.detach().numpy()

                # compute dcpa and tcpa between agent and estimates
                dist = np.sqrt((x_agent - x)**2 + (y_agent - y)**2)

                # smoothing dist curve by using moving average (np.convolve)
                kernel_size = 21
                smoothed_dist = np.ones_like(dist)*self.dcpa_scale*10  # init with high values 
                for i,d in enumerate(dist):
                    smoothed_dist[i][int((kernel_size-1)/2):-int((kernel_size-1)/2)] = np.convolve(d, np.ones(kernel_size)/kernel_size, 'valid') # valid -> convolution only for valid overlap -> loosing kernel_size-1 datapoints

                self.dcpa_est = np.min(smoothed_dist,1)
                self.tcpa_est = self.delta_t * (np.argmin(smoothed_dist,1) - self.FORECAST_WIN - self.N_pastObs)
            else: 
                self.tcpa_est = self.tcpa_est - self.delta_t
 
            idx1 = np.argsort(self.vessel_ttc[:self.n_vessels_half])
            idx2 = np.argsort(self.vessel_ttc[self.n_vessels_half:]) + self.n_vessels_half              

            idx = np.concatenate([idx1, idx2])
            dcpa_est_sorted = self.dcpa_est[idx].copy()
            tcpa_est_sorted = self.tcpa_est[idx].copy()
        
        else:
            # compute sorting index array for asceding euclidean distance
            eucl_dist = (self.vessel_x - self.agent_x)**2 + (self.vessel_y - self.agent_y)**2
            idx1 = np.argsort(eucl_dist[:self.n_vessels_half])
            idx2 = np.argsort(eucl_dist[self.n_vessels_half:]) + self.n_vessels_half
            idx = np.concatenate([idx1, idx2])

        x = self.vessel_x[idx].copy()
        y = self.vessel_y[idx].copy()
        
        # state definition
        self.state = np.array([self.agent_ay/self.ay_max,
                               self.agent_vx/self.vx_max,
                               self.agent_vy/self.vy_max])

        self.state = np.append(self.state, (self.agent_x  - x)/self.delta_x_max)
        self.state = np.append(self.state, (self.agent_y  - y)/self.delta_y_max)
        
        if self.setting == "EST":          
            self.state = np.append(self.state, dcpa_est_sorted/self.dcpa_scale)
            self.state = np.append(self.state, tcpa_est_sorted/self.tcpa_scale)

        # frame stacking
        if self.frame_stack > 1:
            
            if self.frame_hist_cnt == self.frame_stack:
                self.frame_array = np.roll(self.frame_array, shift = -1, axis = 0)
                self.frame_array[self.frame_stack - 1, :] = self.state
            else:
                self.frame_array[self.frame_hist_cnt] = self.state
                self.frame_hist_cnt += 1
            
            self.state = self.frame_array.flatten()
        
        # Write data to file
        #tmp = np.append(np.append(np.array([self.agent_x, self.agent_y]), np.append(self.vessel_x, self.vessel_y)), self.vessel_ttc)
        #self.epi_info[self.current_timestep] = np.append(tmp, np.array([self.reward]))
        #np.savetxt("epi_info.csv", self.epi_info, delimiter=" ")

    def step(self, action):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done."""
        self.crash_flag = False
        self._move_vessel()
        self._move_agent(action)
        self._set_state()
        self._calculate_reward()
        done = self._done()
        
        #self.render()

        self.current_timestep += 1
        return self.state, self.reward, done, {}
    
    def _move_vessel(self):
        """Updates positions, velocities and accelerations of vessels. Accelerations are constant.
        Used approximation: Euler-Cromer method, that is v_(n+1) = v_n + a_n * t and x_(n+1) = x_n + v_(n+1) * t."""

        for i in range(self.n_vessels):
            if self.obst_traj == "constant":
                # lateral dynamics
                self.vessel_y[i] = self.vessel_y[i] + self.vessel_vy[i] * self.delta_t

                # longitudinal dynamics     
                self.vessel_x[i] = self.vessel_x[i] + self.vessel_vx[i] * self.delta_t
            else:
                if i < self.n_vessels_half:
                    self.vessel_x[i]  = self.vessel_x_traj_low[i][self.N_pastObs-1]
                    self.vessel_y[i]  = self.vessel_y_traj_low[i][self.N_pastObs-1]
                    self.vessel_vx[i] = self.vessel_vx_traj_low[i][self.N_pastObs-1]
                    self.vessel_vy[i] = self.vessel_vy_traj_low[i][self.N_pastObs-1]
                    self.vessel_x_traj_low[i]  = np.delete(self.vessel_x_traj_low[i], 0)
                    self.vessel_y_traj_low[i]  = np.delete(self.vessel_y_traj_low[i], 0)
                    self.vessel_vx_traj_low[i] = np.delete(self.vessel_vx_traj_low[i], 0)
                    self.vessel_vy_traj_low[i] = np.delete(self.vessel_vy_traj_low[i], 0)

                else:
                    temp_index = i-self.n_vessels_half
                    self.vessel_x[i]  = self.vessel_x_traj_up[temp_index][self.N_pastObs-1]
                    self.vessel_y[i]  = self.vessel_y_traj_up[temp_index][self.N_pastObs-1]
                    self.vessel_vx[i] = self.vessel_vx_traj_up[temp_index][self.N_pastObs-1]
                    self.vessel_vy[i] = self.vessel_vy_traj_up[temp_index][self.N_pastObs-1]  
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
        self.reward = 0
        for i in range(self.n_vessels_half):
            if self.vessel_ttc[i]< 0 and self.vessel_ttc[i] > -self.delta_t and (self.agent_y - self.vessel_y[i]) < 0:
                self.reward = -1
            if self.vessel_ttc[self.n_vessels_half+i]< 0 and self.vessel_ttc[self.n_vessels_half+i] > -self.delta_t and (self.agent_y - self.vessel_y[self.n_vessels_half+i]) > 0: 
                self.reward = -1            

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
            self.ax0.set_xlim(-500, 4000)
            self.ax0.set_ylim(-1000, 1000)
            self.ax0.set_xlabel("x")
            self.ax0.set_ylabel("y")

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
            if self.setting == "EST":
                for i in range(self.n_vessels):
                    self.ax1.text(self.vessel_ttc[i] + 50, self.vessel_y[i] + 50, "TCPA: "+ np.array2string(self.tcpa_est[i], precision=0), horizontalalignment='left', verticalalignment='center', color='red', size = 6)    
                    self.ax1.text(self.vessel_ttc[i] + 50, self.vessel_y[i] + 20, "DCPA: "+ np.array2string(np.abs(self.dcpa_est[i]), precision=0), horizontalalignment='left', verticalalignment='center', color='blue', size = 6)    

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
