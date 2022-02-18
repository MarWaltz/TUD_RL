import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt


class FossenCS2(gym.Env):
    """Class environment with initializer, step, reset, and render method."""
    
    def __init__(self):
        super().__init__()

        # to be filled
        self.observation_space  = spaces.Box(low=np.array([-1], dtype=np.float32), high=np.array([1], dtype=np.float32))
        self.action_space       = spaces.Box(low=np.array([-1], dtype=np.float32), high=np.array([1], dtype=np.float32))
        self._max_episode_steps = 1e8

        # CS2 parameters (from Skjetne et al. (2005) in Automatica) | Note: Adjustet according to Fabian.
        self.length = 1.3 * 20  # to see something
        self.width  = 0.2 * 20 
        
        self.m      =  23.8
        self.I_z    =  1.76
        self.x_g    =  0.046
        self.X_u    = -0.72253
        self.X_lulu = -1.32742
        self.X_uuu  = -5.86643
        self.Y_v    = -0.88965
        self.Y_lvlv = -36.47287

        #self.Y_r    =  0.1079
        #self.N_v    =  0.1052
        #self.N_lvlv =  5.0437
        self.X_dotu = -2.
        self.Y_dotv = -10.
        self.Y_dotr =  0.
        self.N_dotv =  0.
        self.N_dotr = -1.

        # further CS2 parameters (from Skjetne et al. (2004) in CAMS)
        # Note: This paper has slightly different values for some of the parameters above.
        self.Y_lrlv = -0.805
        self.Y_r    = -7.250
        self.Y_lvlr = -0.845
        self.Y_lrlr = -3.450

        self.N_lrlv =  0.130
        self.N_r    = -1.900
        self.N_lvlr =  0.080
        self.N_lrlr =  0.750 # SIGN ?
        self.N_lvlv =  3.95645
        self.N_v    =  0.03130

        # mass matrix (rigid body + added mass) and its inverse
        self.M_RB = np.array([[self.m, 0, 0],
                              [0, self.m, self.m * self.x_g],
                              [0, self.m * self.x_g, self.I_z]])
        self.M_A = np.array([[-self.X_dotu, 0, 0],
                              [0, -self.Y_dotv, -self.Y_dotr],
                              [0, -self.N_dotv, -self.N_dotr]])
        self.M = self.M_RB + self.M_A
        self.M_inv = np.linalg.inv(self.M)

        # simulation settings
        self.delta_t     = .5               # simulation time interval (in s)
        self.x_max       = 500              # maximum x-coordinate (in m)
        self.y_max       = 500              # maximum y-coordinate (in m)
        self.delta_tau_u = .5               # thrust change in u-direction (in N)
        self.tau_u_max   = 5.               # maximum tau in u-direction (in N)
        self.tau_r       = .5               # base moment to rudder (in Nm)


    def __str__(self) -> str:
        ste = f"Step: {self.step_cnt}"
        pos = f"x: {np.round(self.eta[0], 3)}, y: {np.round(self.eta[1], 3)}, " + r"$\psi$: " + f"{np.round(self._rad_to_deg(self.eta[2]), 3)}°"
        vel = f"u: {np.round(self.nu[0], 3)}, v: {np.round(self.nu[1], 3)}, r: {np.round(self.nu[2], 3)}"
        return ste + "\n" + pos + "\n" + vel


    def _C_of_nu(self, nu):
        """Computes centripetal/coriolis matrix for given velocities. Source: Xu et. al (2022)."""

        # unpacking
        u, v, r = nu

        # rigid-body
        C_RB = np.array([[0, 0, -self.m * (self.x_g * r + v)],
                         [0, 0,  self.m * u],
                         [self.m * (self.x_g * r + v), - self.m * u, 0]])
        
        # added mass
        C_A = np.array([[0, 0, self.Y_dotv * v + self.Y_dotr * r],
                        [0, 0, - self.X_dotu * u],
                        [-self.Y_dotv * v - self.Y_dotr * r, self.X_dotu * u, 0]])
        
        return C_RB + C_A


    def _D_of_nu(self, nu):
        """Computes damping matrix for given velocities. Source: Xu et. al (2022)."""
        
        # unpacking
        u, v, r = nu

        # components
        d11 = -self.X_u - self.X_lulu * np.abs(u) - self.X_uuu * u**2
        d22 = -self.Y_v - self.Y_lvlv * np.abs(v) - self.Y_lrlv * np.abs(r)
        d23 = -self.Y_r - self.Y_lvlr * np.abs(v) - self.Y_lrlr * np.abs(r)
        d32 = -self.N_v - self.N_lvlv * np.abs(v) - self.N_lrlv * np.abs(r)
        d33 = -self.N_r - self.N_lvlr * np.abs(v) - self.N_lrlr * np.abs(r)

        return np.array([[d11, 0, 0],
                         [0, d22, d23],
                         [0, d32, d33]])


    def _g_of_nu(self, nu):
        """Computes hydrostatic forces for velocities. Might be not feasible for 3 DOF model according to Fossen (2021). Source: Xu et. al (2022)."""
        
        # unpacking
        u, v, r = nu

        # components
        g1 = 0.0279 * u * v**2 + 0.0342 * v**2 * r
        g2 = 0.0912 * u**2 * v
        g3 = 0.0156 * u * r**2 + 0.0278 *u * r * v**3

        return np.array([g1, g2, g3])


    def _tau_w_of_t(self, t):
        """Computes time-varying environmental disturbances. Source: Xu et. al (2022)."""
        
        tau_w1 = 2 * np.cos(0.5*t) * np.cos(t) + 0.3 * np.cos(0.5*t) * np.sin(0.5*t) - 3
        tau_w2 = 0.01 * np.sin(0.1*t)
        tau_w3 = 0.6 * np.sin(1.1*t) * np.cos(0.3*t)

        return np.array([tau_w1, tau_w2, tau_w3])


    def _T_of_psi(self, psi):
        """Computes rotation matrix for given heading (in rad)."""
        return np.array([[np.cos(psi), -np.sin(psi), 0],
                         [np.sin(psi),  np.cos(psi), 0],
                         [0, 0, 1]])

    def _deg_to_rad(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * np.pi / 180


    def _rad_to_deg(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * 180 / np.pi


    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        self._set_dynamics()
        self._set_state()

        return self.state
   

    def _set_dynamics(self):
        """Initializes positions and velocities of agent and vessels."""
        
        # motion init (for own ship)
        self.eta = np.array([100., 100., self._deg_to_rad(0.)])        # x (in m),   y (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([0.55, 0., 0.])                            # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system

        # thrust init for OS
        self.tau = np.array([2., 0., 0.])   # thrust in u-direction, thrust in v-direction, force moment for r


    def _upd_dynamics(self):
        """Updates positions and velocities for next simulation step. Uses basic Euler method."""

        # calculate nu_dot by solving Fossen's equation
        nu = self.nu
        M_nu_dot = self.tau - np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu)# - self._g_of_nu(nu)# + self._tau_w_of_t(self.sim_t)
        nu_dot = np.dot(self.M_inv, M_nu_dot)

        # get new velocity (BODY-system)
        self.nu += nu_dot * self.delta_t

        # calculate eta_dot via rotation
        eta_dot = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # get new positions (NE-system)
        self.eta += eta_dot * self.delta_t

        # clip heading to [-2pi, 2pi]
        if np.abs(self.eta[2]) > 2*np.pi:
            self.eta[2] = np.sign(self.eta[2]) * (abs(self.eta[2]) - 2*np.pi)

        # increase overall simulation time
        self.sim_t += self.delta_t


    def _set_state(self):      
        self.state = None


    def _upd_tau(self, a):
        """Action 'a' is a np.array([a1, a2]). Both a1, a2 take values in 0, 1, 2. They correspond to:
        
        a1: 0 - keep thrust as is | 1 - decrease thrust            | 2 - increase thrust
        a2: 0 - no rudder force   | 1 - rudder force from one side | 2 - rudder force from other side  
        """
        a1, a2 = a
        
        assert a1 in [0, 1, 2] and a2 in [0, 1, 2], "Unknown actions."

        # thrust for surge (u)
        if a1 == 0:
            pass
        elif a1 == 1:
            self.tau[0] -= self.delta_tau_u
        else:
            self.tau[0] += self.delta_tau_u
        
        # clip it to surge max
        if self.tau[0] > self.tau_u_max:
            self.tau[0] = self.tau_u_max
        
        elif self.tau[0] < -self.tau_u_max:
            self.tau[0] = -self.tau_u_max

        # yaw moment
        if a2 == 0:
            pass
        elif a2 == 1:
            self.tau[2] = -self.tau_r
        else:
            self.tau[2] = self.tau_r

  
    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # update control tau
        self._upd_tau(a)

        # update update dynamics
        self._upd_dynamics()

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt
        self.step_cnt += 1
        
        return self.state, self.r, d, {}
    

    def _calculate_reward(self):
        """Returns reward of the current state."""   
        self.r = None
    

    def _done(self):
        """Returns boolean flag whether episode is over."""
        return None


    def render(self):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(10, 10))
                self.gs  = self.fig.add_gridspec(1, 1)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                #self.ax1 = self.fig.add_subplot(self.gs[1, 0]) # empty
                #self.ax2 = self.fig.add_subplot(self.gs[1, 1]) # empty
                #self.ax3 = self.fig.add_subplot(self.gs[0, 1]) # empty
                plt.ion()
                plt.show()
            
            # ---- ship movement ----
            # clear prior axes, set limits and add labels and title
            self.ax0.clear()
            self.ax0.set_xlim(0, self.x_max)
            self.ax0.set_ylim(0, self.y_max)
            self.ax0.set_xlabel("x")
            self.ax0.set_ylabel("y")

            # set ship
            rect = patches.Rectangle((self.eta[0]-self.length/2, self.eta[1]-self.width/2), self.length, self.width, self._rad_to_deg(self.eta[2]), linewidth=1, edgecolor='r', facecolor='none')
            self.ax0.add_patch(rect)
            self.ax0.text(self.eta[0] + 10, self.eta[1] + 10, self.__str__())
            plt.pause(0.1)

            """
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
            """
