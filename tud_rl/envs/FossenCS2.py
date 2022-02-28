import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt


class FossenCS2(gym.Env):
    """Class environment with initializer, step, reset, and render method."""
    
    def __init__(self):
        super().__init__()

        # gym definitions
        self.observation_space  = spaces.Box(low  = np.array([-np.inf, -np.inf, -np.inf, -1, -1, -np.inf], dtype=np.float32), 
                                             high = np.array([ np.inf,  np.inf,  np.inf,  1,  1,  np.inf], dtype=np.float32))
        self.action_space = spaces.Discrete(3) #9

        # custom inits
        self._max_episode_steps = 5e3
        self.r = 0
        self.r_head = 0
        self.r_dist = 0
        self.action = 0
        self.state_names = ["u", "v", "r", r"$\Psi$", r"$\Psi_e$", "ED"]

        # CS2 parameters (from Skjetne et al. (2004) in CAMS)
        self.length = 1.255 * 20     # to see something
        self.width  = 0.29 * 20      # to see something
        
        self.m      =  23.8
        self.I_z    =  1.76
        self.x_g    =  0.046
        self.X_u    = -0.72253
        self.X_lulu = -1.32742
        self.X_uuu  = -5.86643
        self.Y_v    = -0.88965
        self.Y_lvlv = -36.47287

        self.X_dotu = -2.
        self.Y_dotv = -10.
        self.Y_dotr =  0.
        self.N_dotv =  0.
        self.N_dotr = -1.

        self.Y_lrlv = -0.805
        self.Y_r    = -7.250
        self.Y_lvlr = -0.845
        self.Y_lrlr = -3.450

        self.N_lrlv =  0.130
        self.N_r    = -1.900
        self.N_lvlr =  0.080
        self.N_lrlr = -0.750
        self.N_lvlv =  3.95645
        self.N_v    =  0.03130

        # CS2 parameters for translating revolutions per second (n = n1 = n2) and rudder angle (delta = delta1 = delta2) into tau
        # Bow thruster of CS2 is assumed to yield zero force
        # (from Skjetne et al. (2004) in MIC)
        self.l_xT1 = -0.499
        self.l_yT1 = -0.078
        self.l_xT2 = -0.499
        self.l_yT2 =  0.078
        self.l_xT3 =  0.466
        self.l_yT3 =  0.000
        self.l_xR1 = -0.549
        self.l_yR1 = -0.078
        self.l_xR2 = -0.549
        self.l_yR2 =  0.078

        self.L_d_p  = 6.43306
        self.L_dd_p = 5.83594
        self.L_d_m  = 3.19573
        self.L_dd_m = 2.34356
        self.T_nn_p = 3.65034e-3
        self.T_nu_p = 1.52468e-4
        self.T_nn_m = 5.10256e-3
        self.T_nu_m = 4.55822e-2
        self.T_n3n3 = 1.56822e-4

        self.B = np.array([[1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1],
                           [np.abs(self.l_yT1), -np.abs(self.l_yT2), np.abs(self.l_xT3), -np.abs(self.l_xR1), -np.abs(self.l_xR2)]])

        self.d_prop = 60e-3   # diameter of propellers in m (from PhD-thesis of Karl-Petter W. Lindegaard)
        self.ku  = 0.5        # induced velocity factor
        self.rho = 1014       # density of sea water (in kg/m3)

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
        self.delta_t = 0.1              # simulation time interval (in s)
        self.x_max   = 400              # maximum x-coordinate (in m)
        self.y_max   = 400              # maximum y-coordinate (in m)
        self.n_prop  = 100               # revolutions per second of the two main propellers
        self.n_bow   = 0                # revolutions per second of the bow thruster

        #self.delta_tau_u = .5                    # thrust change in u-direction (in N)
        #self.tau_u_max     = 5.                  # maximum tau in u-direction (in N)
        #self.tau_r       = .5                    # base moment to rudder (in Nm)

        # clipping values (from Xu et al. (2022) in Neurocomputing)
        self.u_max = 3.5                          # maximum surge velocity (in m/s) 
        self.u_dot_max = 0.4                      # maximum surge acceleration (in m/s2)
        self.rud_angle_max = self._deg_to_rad(30) # maximum rudder angle (in rad)
        


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

        self._spawn_objects()
        self._spawn_agent()
        self._set_state()
        self.state_init = self.state

        return self.state


    def _spawn_objects(self):
        """Initializes objects such as the target, vessels, static obstacles, etc."""
        
        #self.goal = np.array([200., 200.])
        self.goal = np.array([np.random.uniform(self.x_max - 300, self.x_max),
                              np.random.uniform(self.y_max - 300, self.y_max)])


    def _spawn_agent(self):
        """Initializes positions and velocities of agent."""
        
        # motion init (for own ship)
        self.eta = np.array([100., 100., self._deg_to_rad(0.)])        # x (in m),   y (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([0., 0., 0.])                              # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system

        # thrust init for OS
        #self.tau = np.array([2., 0., 0.])   # thrust in u-direction, thrust in v-direction, force moment for r
        self.rud_angle = 0
        self._set_tau_from_n_delta()

        # initial euclidean distance to goal
        self.ED_goal_init = np.sqrt((self.goal[0] - self.eta[0])**2 + (self.goal[1] - self.eta[1])**2)


    def _set_tau_from_n_delta(self):
        """Translates revolutions per second (n) and rudder angle (delta in rad) into tau. Currently, n is fixed."""

        u = self.nu[0]
        n = self.n_prop

        # compute T1, T2 (same value since n = n1 = n2)
        n_bar_u = np.max([0, self.T_nu_p / self.T_nn_p * u])
        n_bar_l = np.min([0, self.T_nu_m / self.T_nn_m * u])

        if n >= n_bar_u:
            T12 = self.T_nn_p * np.abs(n) * n - self.T_nu_p * np.abs(n) * u
        elif n <= n_bar_l:
            T12 = self.T_nn_m * np.abs(n) * n - self.T_nu_m * np.abs(n) * u
        else:
            T12 = 0

        # compute T3
        T3 = self.T_n3n3 * np.abs(self.n_bow) * self.n_bow

        # compute u_rud12
        if u >= 0:
            u_rud12 = u + self.ku * (np.sqrt(np.max([0, 8 / (np.pi * self.rho * self.d_prop**2) * T12 + u**2])) - u)
        else:
            u_rud12 = u
        
        # compute L12
        if u_rud12 >= 0:
            L12 = (self.L_d_p * self.rud_angle - self.L_dd_p * np.abs(self.rud_angle) * self.rud_angle) * np.abs(u_rud12) * u_rud12
        else:
            L12 = (self.L_d_m * self.rud_angle - self.L_dd_m * np.abs(self.rud_angle) * self.rud_angle) * np.abs(u_rud12) * u_rud12
        
        # compute tau
        self.tau = np.dot(self.B, np.array([T12, T12, T3, L12, L12]))


    def _set_state(self):      
        """State consists of: u, v, r, heading, heading_error, ED_goal."""
        self.state = np.array([self.nu[0], 
                               self.nu[1],
                               self.nu[2], 
                               self.eta[2] / (2*np.pi), 
                               self._get_sign_psi_e() * self._get_abs_psi_e() / (np.pi), 
                               self._ED_to_goal() / self.ED_goal_init])


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # update control tau
        self._upd_tau(a)

        # update dynamics
        self._upd_dynamics()

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt
        self.step_cnt += 1
        
        return self.state, self.r, d, {}


    def _upd_dynamics(self):
        """Updates positions and velocities for next simulation step. Uses basic Euler method."""

        # calculate nu_dot by solving Fossen's equation
        nu = self.nu
        M_nu_dot = self.tau - np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu)# - self._g_of_nu(nu)# + self._tau_w_of_t(self.sim_t)
        nu_dot = np.dot(self.M_inv, M_nu_dot)

        # clip u_dot to maximum acceleration
        nu_dot[0] = np.clip(nu_dot[0], -self.u_dot_max, self.u_dot_max)

        # get new velocity (BODY-system)
        self.nu += nu_dot * self.delta_t

        # calculate eta_dot via rotation
        eta_dot = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # get new positions (NE-system)
        self.eta += eta_dot * self.delta_t

        # clip surge velocity
        self.nu[0] = np.clip(self.nu[0], -self.u_max, self.u_max)

        # transform heading to [-2pi, 2pi]
        if np.abs(self.eta[2]) > 2*np.pi:
            self.eta[2] = np.sign(self.eta[2]) * (abs(self.eta[2]) - 2*np.pi)

        # increase overall simulation time
        self.sim_t += self.delta_t


    def _upd_tau(self, a):
        """Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:
        
        0 - keep rudder angle as is
        1 - increase rudder angle by 5 degree
        2 - decrease rudder angle by 5 degree
        """
        assert a in range(3), "Unknown action."

        # store action for rendering
        self.action = a

        # update angle
        if a == 1:
            self.rud_angle += self._deg_to_rad(5)
        elif a == 2:
            self.rud_angle -= self._deg_to_rad(5)
        
        # clip it
        self.rud_angle = np.clip(self.rud_angle, -self.rud_angle_max, self.rud_angle_max)

        # update the control tau
        self._set_tau_from_n_delta()

        # clip thrust to surge max
        #self.tau[0] = np.clip(self.tau[0], -self.tau_u_max, self.tau_u_max)


    def _calculate_reward(self):
        """Returns reward of the current state."""

        # ---- Path planning reward (Xu et al. 2022) -----

        # 1. Distance reward
        ED = self._ED_to_goal()
        r_dist = - ED / self.ED_goal_init

        # 2. Heading reward
        r_head = - self._get_abs_psi_e() / np.pi

        # 3. Goal reach reward
        #r_goal = 10 if self._ED_to_goal() <= 25 else 0
        
        # overall reward
        self.r_dist = r_dist
        self.r_head = r_head
        self.r = r_dist + r_head #+ r_goal


    def _get_abs_psi_e(self):
        """Calculates the absolute value of the heading error (goal_angle - heading)."""
        
        psi_e_abs = np.abs(self._get_psi_d() - self._clip_angle(self.eta[2]))

        if psi_e_abs <= np.pi:
            return psi_e_abs
        else:
            return 2*np.pi - psi_e_abs
    

    def _get_sign_psi_e(self):
        """Calculates the sign of the heading error."""

        psi_d = self._get_psi_d()
        psi   = self._clip_angle(self.eta[2])

        if psi_d <= np.pi:

            if psi_d <= psi <= psi_d + np.pi:
                return -1
            else:
                return 1
        
        else:
            if psi_d - np.pi <= psi <= psi_d:
                return 1
            else:
                return -1


    def _clip_angle(self, angle):
        """Clips an angle from [-2pi, 2pi] to [0, 2pi]."""

        if angle < 0:
            return 2*np.pi + angle
        return angle


    def _get_psi_d(self):
        """Calculates the heading angle of the agent towards the goal. Perspective as in unit circle."""
        
        ED = self._ED_to_goal()
        psi_d = np.arccos(np.abs(self.goal[0] - self.eta[0]) / ED)

        # x_goal < x_agent
        if self.goal[0] < self.eta[0]:
            
            if self.goal[1] >= self.eta[1]:
                psi_d = np.pi - psi_d
            else:
                psi_d = np.pi + psi_d
        
        # x_goal >= x_agent
        else:
            if self.goal[1] < self.eta[1]:
                psi_d = 2*np.pi - psi_d
        
        return psi_d


    def _done(self):
        """Returns boolean flag whether episode is over."""
        if any([self._ED_to_goal() <= 25,
                self.eta[0] < 0,
                self.eta[0] >= self.x_max,
                self.eta[1] < 0,
                self.eta[1] >= self.y_max,
                self.step_cnt >= self._max_episode_steps]):

                return True

        return False


    def _ED_to_goal(self):
        """Computes the euclidean distance to the goal."""
        return np.sqrt((self.goal[0] - self.eta[0])**2 + (self.goal[1] - self.eta[1])**2)


    def render(self):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 2 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(10, 7))
                self.gs  = self.fig.add_gridspec(2, 2)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                self.ax1 = self.fig.add_subplot(self.gs[0, 1]) # reward
                self.ax2 = self.fig.add_subplot(self.gs[1, 0]) # state
                self.ax3 = self.fig.add_subplot(self.gs[1, 1]) # action
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
            self.ax0.text(self.goal[0] - 40, self.goal[1] - 40, r"$\psi_d$" + f": {np.round(self._rad_to_deg(self._get_psi_d()),3)}°")

            # set goal
            self.ax0.scatter(self.goal[0], self.goal[1], color="blue")

            # ----- reward plot ----
            if self.step_cnt == 0:
                self.ax1.clear()
                self.ax1.old_time = 0
                self.ax1.old_r_head = 0
                self.ax1.old_r_dist = 0

            self.ax1.set_xlim(0, self._max_episode_steps)
            self.ax1.set_ylim(-1.25, 0.1)
            self.ax1.set_xlabel("Timestep in episode")
            self.ax1.set_ylabel("Reward")

            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_head, self.r_head], color = "blue", label="Heading")
            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_dist, self.r_dist], color = "black", label="Distance")
            
            if self.step_cnt == 0:
                self.ax1.legend()

            self.ax1.old_time = self.step_cnt
            self.ax1.old_r_head = self.r_head
            self.ax1.old_r_dist = self.r_dist


            # ---- state plot ----
            if self.step_cnt == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_state = self.state_init

            self.ax2.set_xlim(0, self._max_episode_steps)
            self.ax2.set_ylim(-1, 1.1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("State information")

            for i in range(len(self.state)):
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_state[i], self.state[i]], 
                               color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i], 
                               label=self.state_names[i])          
            if self.step_cnt == 0:
                self.ax2.legend()

            self.ax2.old_time = self.step_cnt
            self.ax2.old_state = self.state


            # ---- action plot ----
            if self.step_cnt == 0:
                self.ax3.clear()
                self.ax3.old_time = 0
                self.ax3.old_action = 0

            self.ax3.set_xlim(0, self._max_episode_steps)
            self.ax3.set_ylim(-0.1, self.action_space.n - 1 + 0.1)
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Action (discrete)")

            self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_action, self.action], color="black")

            self.ax3.old_time = self.step_cnt
            self.ax3.old_action = self.action

            plt.pause(0.01)

    """def _upd_tau(self, a):
        Action 'a' is an integer taking values in [0, 1, 2, ..., 8]. They correspond to:
        
        0 - keep thrust as is | no rudder force 
        1 - keep thrust as is | rudder force from one side
        2 - keep thrust as is | rudder force from other side
        
        3 - decrease thrust   | no rudder force 
        4 - decrease thrust   | rudder force from one side
        5 - decrease thrust   | rudder force from other side
        
        6 - increase thrust   | no rudder force 
        7 - increase thrust   | rudder force from one side
        8 - increase thrust   | rudder force from other side

        assert a in range(4), "Unknown action."

        # store action for rendering
        self.action = a

        # keep thrust as is
        if a == 0:
            pass

        elif a == 1:
            self.tau[2] = -self.tau_r
        
        elif a == 2:
            self.tau[2] = self.tau_r
        
        # decrease thrust
        elif a == 3:
            self.tau[0] -= self.delta_tau_u
        
        elif a == 4:
            self.tau[0] -= self.delta_tau_u
            self.tau[2] = -self.tau_r
        
        elif a == 5:
            self.tau[0] -= self.delta_tau_u
            self.tau[2] = self.tau_r
        
        # increase thrust:
        elif a == 6:
            self.tau[0] += self.delta_tau_u
        
        elif a == 7:
            self.tau[0] += self.delta_tau_u
            self.tau[2] = -self.tau_r

        elif a == 8:
            self.tau[0] += self.delta_tau_u
            self.tau[2] = self.tau_r

        # clip thrust to surge max
        if self.tau[0] > self.tau_u_max:
            self.tau[0] = self.tau_u_max
        
        elif self.tau[0] < -self.tau_u_max:
            self.tau[0] = -self.tau_u_max"""

x = FossenCS2()
x.reset()
while True:
    x.render()
    s2, r, d, _ = x.step(0)
    #print(x)
    
    #import sys
    #import time
    #time.sleep(0.5)
