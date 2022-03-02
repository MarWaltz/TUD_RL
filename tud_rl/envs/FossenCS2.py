import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class CyberShipII:
    """This class provides a vessel behaving according to the nonlinear ship manoeuvering model (3 DOF) proposed in 
    Skjetne et al. (2004) in Modeling, Identification and Control."""

    def __init__(self, delta_t) -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation step size and dummy action for rendering
        self.delta_t = delta_t
        self.action = 0

        # CyberShip II parameters
        self.length = 1.255
        self.width  = 0.29
        
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
        self.Y_dotr = -0.
        self.N_dotv = -0.
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

        # parameters for translating revolutions per second (n = n1 = n2) and rudder angle (delta = delta1 = delta2) into tau
        # Bow thruster of CS2 is assumed to yield zero force
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

        self.d_prop = 60e-3                          # diameter of propellers in m (from PhD-thesis of Karl-Petter W. Lindegaard)
        self.ku  = 0.5                               # induced velocity factor
        self.rho = 1014                              # density of sea water (in kg/m3)
        self.n_prop  = 2000 / 60 * self.delta_t      # revolutions per second of the two main propellers
        self.n_bow   =    0 / 60 * self.delta_t      # revolutions per second of the bow thruster

        # mass matrix (rigid body + added mass) and its inverse
        self.M_RB = np.array([[self.m, 0, 0],
                              [0, self.m, self.m * self.x_g],
                              [0, self.m * self.x_g, self.I_z]])
        self.M_A = np.array([[-self.X_dotu, 0, 0],
                              [0, -self.Y_dotv, -self.Y_dotr],
                              [0, -self.N_dotv, -self.N_dotr]])
        self.M = self.M_RB + self.M_A
        self.M_inv = np.linalg.inv(self.M)   
        
        # rudder angle max (in rad) and increment (in rad/s)
        self.rud_angle_max = self._deg_to_rad(35)
        self.rud_angle_inc = self._deg_to_rad(5) * self.delta_t


        #------------------------- Motion Initialization -----------------------------------
        self.eta = np.array([10., 10., 0.]) #np.random.uniform(0, np.pi)])   # N (in m),   E (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([0., 0., 0.])                              # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system

        self.rud_angle = 0
        self._set_tau_from_n_delta()


    def _C_of_nu(self, nu):
        """Computes centripetal/coriolis matrix for given velocities."""

        # unpacking
        u, v, r = nu

        # rigid-body
        C_RB = np.array([[0, 0, -self.m * (self.x_g * r + v)],
                         [0, 0,  self.m * u],
                         [self.m * (self.x_g * r + v), - self.m * u, 0]])
        
        # added mass
        C_A = np.array([[0, 0, self.Y_dotv * v + 0.5 * (self.N_dotv + self.Y_dotr) * r],
                        [0, 0, - self.X_dotu * u],
                        [-self.Y_dotv * v - 0.5 * (self.N_dotv + self.Y_dotr) * r, self.X_dotu * u, 0]])

        return C_RB + C_A


    def _D_of_nu(self, nu):
        """Computes damping matrix for given velocities."""
        
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


    def _T_of_psi(self, psi):
        """Computes rotation matrix for given heading (in rad)."""

        return np.array([[np.cos(psi), -np.sin(psi), 0],
                         [np.sin(psi),  np.cos(psi), 0],
                         [0, 0, 1]])


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


    def _ship_dynamic(self, t, y):
        """Returns eta_dot and nu_dot according to the nonlinear ship manoeuvering model. Forms the basis for ODE integration.

        Args:
            t (time):    integration time
            y (array):   array containing [N, E, psi, u, v, r] (which is [eta, nu])
        """

        # unpack values
        eta = y[:3]
        nu  = y[3:]

        # find nu_dot
        M_nu_dot = self.tau - np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu)
        nu_dot = np.dot(self.M_inv, M_nu_dot)

        # find eta_dot
        eta_dot = np.dot(self._T_of_psi(eta[2]), nu)

        return np.concatenate([eta_dot, nu_dot])


    def _upd_dynamics(self, euler=False):
        """Updates positions and velocities for next simulation step. Uses basic Euler method."""

        if euler:

            # calculate nu_dot by solving Fossen's equation
            M_nu_dot = self.tau - np.dot(self._C_of_nu(self.nu) + self._D_of_nu(self.nu), self.nu)
            nu_dot = np.dot(self.M_inv, M_nu_dot)

            # get new velocity (BODY-system)
            self.nu += nu_dot * self.delta_t

            # calculate eta_dot via rotation
            eta_dot = np.dot(self._T_of_psi(self.eta[2]), self.nu)

            # get new positions (NE-system)
            self.eta += eta_dot * self.delta_t
           
        else:

            # solve ODE
            sol = solve_ivp(fun    = self._ship_dynamic, 
                            t_span = (0.0, self.delta_t), 
                            y0     = np.concatenate([self.eta, self.nu]),
                            method = "RK45",
                            t_eval = np.array([self.delta_t]))

            # store new eta and nu
            self.eta = sol.y[0:3, 0]
            self.nu  = sol.y[3:, 0]

        # transform heading to [-2pi, 2pi]
        if np.abs(self.eta[2]) > 2*np.pi:
            self.eta[2] = np.sign(self.eta[2]) * (np.abs(self.eta[2]) - 2*np.pi)


    def _upd_tau(self, a):
        """Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:
        
        0 - keep rudder angle as is
        1 - increase rudder angle by 5 degree per second
        2 - decrease rudder angle by 5 degree per second
        """
        assert a in range(3), "Unknown action."

        # store action for rendering
        self.action = a

        # update angle
        if a == 1:
            self.rud_angle += self.rud_angle_inc
        elif a == 2:
            self.rud_angle -= self.rud_angle_inc
        
        # clip it
        self.rud_angle = np.clip(self.rud_angle, -self.rud_angle_max, self.rud_angle_max)

        # update the control tau
        self._set_tau_from_n_delta()


    def _deg_to_rad(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * np.pi / 180


    def _rad_to_deg(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * 180 / np.pi


    def _clip_angle(self, angle):
        """Clips an angle from [-2pi, 2pi] to [0, 2pi]."""

        if angle < 0:
            return 2*np.pi + angle
        return angle



class FossenCS2(gym.Env):
    """This environment implements the nonlinear ship manoeuvering model (3 DOF) proposed in Skjetne et al. (2004) 
    in Modeling, Identification and Control."""

    def __init__(self):
        super().__init__()

        # gym definitions
        self.observation_space  = spaces.Box(low  = np.array([-np.inf, -np.inf, -np.inf, -1, -1, -np.inf], dtype=np.float32), 
                                             high = np.array([ np.inf,  np.inf,  np.inf,  1,  1,  np.inf], dtype=np.float32))
        self.action_space = spaces.Discrete(3)

        # custom inits
        self._max_episode_steps = 1e3
        self.r = 0
        self.r_head = 0
        self.r_dist = 0
        self.state_names = ["u", "v", "r", r"$\Psi$", r"$\Psi_e$", "ED"]

        # simulation settings
        self.delta_t = 0.5       # simulation time interval (in s)
        self.N_max   = 50        # maximum N-coordinate (in m)
        self.E_max   = 50        # maximum E-coordinate (in m)


    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init objects
        self._spawn_objects()
        
        # init agent (OS for 'Own Ship') and calculate initial distance to goal
        self.OS = CyberShipII(delta_t = self.delta_t)
        self.OS_goal_ED_init = self._ED(N0 = self.OS.eta[0], E0 = self.OS.eta[1], N1 = self.goal["N"], E1 = self.goal["E"])
        
        # init state
        self._set_state()
        self.state_init = self.state

        return self.state


    def _spawn_objects(self):
        """Initializes objects such as the goal, vessels, static obstacles, etc."""
        
        self.goal = {"N" : np.random.uniform(self.N_max - 25, self.N_max),
                     "E" : np.random.uniform(self.E_max - 25, self.E_max)}
        #self.goal = {"N" : 20, "E" : 20}

    def _set_state(self):
        """State consists of: u, v, r, heading, heading_error, ED_goal (all from agent's perspective)."""

        OS_goal_ED = self._ED(N0 = self.OS.eta[0], E0 = self.OS.eta[1], N1 = self.goal["N"], E1 = self.goal["E"])

        self.state = np.array([self.OS.nu[0], 
                               self.OS.nu[1],
                               self.OS.nu[2], 
                               self.OS.eta[2] / (2*np.pi), 
                               self._get_sign_psi_e() * self._get_abs_psi_e() / (np.pi), 
                               OS_goal_ED / self.OS_goal_ED_init])


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # update control tau
        self.OS._upd_tau(a)

        # update agent dynamics
        self.OS._upd_dynamics()

        # update environmental dynamics, e.g., other vessels
        pass

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t
        
        return self.state, self.r, d, {}


    def _calculate_reward(self):
        """Returns reward of the current state."""

        # ---- Path planning reward (Xu et al. 2022) -----

        # 1. Distance reward
        OS_goal_ED = self._ED(N0 = self.OS.eta[0], E0 = self.OS.eta[1], N1 = self.goal["N"], E1 = self.goal["E"])
        r_dist = - OS_goal_ED / self.OS_goal_ED_init

        # 2. Heading reward
        r_head = - self._get_abs_psi_e() / np.pi

        # overall reward
        self.r_dist = r_dist
        self.r_head = r_head
        self.r = r_dist + r_head


    def _get_abs_psi_e(self):
        """Calculates the absolute value of the heading error (goal_angle - heading)."""
        
        psi_e_abs = np.abs(self._get_psi_g() - self.OS._clip_angle(self.OS.eta[2]))

        if psi_e_abs <= np.pi:
            return psi_e_abs
        else:
            return 2*np.pi - psi_e_abs
    

    def _get_sign_psi_e(self):
        """Calculates the sign of the heading error."""

        psi_g = self._get_psi_g()
        psi   = self.OS._clip_angle(self.OS.eta[2])

        if psi_g <= np.pi:

            if psi_g <= psi <= psi_g + np.pi:
                return -1
            else:
                return 1
        
        else:
            if psi_g - np.pi <= psi <= psi_g:
                return 1
            else:
                return -1


    def _get_psi_g(self):
        """Calculates the heading angle of the agent towards the goal."""
        
        # compute ED to goal
        ED = self._ED(N0 = self.OS.eta[0], E0 = self.OS.eta[1], N1 = self.goal["N"], E1 = self.goal["E"])

        # differentiate between goal position from NE perspective centered at the agent
        if self.OS.eta[0] <= self.goal["N"]:

            # I. quadrant
            if self.OS.eta[1] <= self.goal["E"]:
                psi_g = np.arccos((self.goal["N"] - self.OS.eta[0]) / ED)

            # II. quadrant
            else:
                psi_g = 2*np.pi - np.arccos((self.goal["N"] - self.OS.eta[0]) / ED)

        else:

            # III. quadrant
            if self.OS.eta[1] >= self.goal["E"]:
                psi_g = np.pi + np.arccos((self.OS.eta[0] - self.goal["N"]) / ED)
            
            # IV. quadrant
            else:
                psi_g = np.pi - np.arccos((self.OS.eta[0] - self.goal["N"]) / ED)

        return psi_g


    def _done(self):
        """Returns boolean flag whether episode is over."""

        OS_goal_ED = self._ED(N0 = self.OS.eta[0], E0 = self.OS.eta[1], N1 = self.goal["N"], E1 = self.goal["E"])

        if any([OS_goal_ED <= 2.5,
                self.OS.eta[0] < 0,
                self.OS.eta[0] >= self.N_max,
                self.OS.eta[1] < 0,
                self.OS.eta[1] >= self.E_max,
                self.step_cnt >= self._max_episode_steps]):

                return True

        return False


    def _ED(self, N0, E0, N1, E1):
        """Computes the euclidean distance between two coordinates in the NE-system."""
        return np.sqrt((N0 - N1)**2 + (E0 - E1)**2)


    def __str__(self) -> str:
        ste = f"Step: {self.step_cnt}"
        pos = f"N: {np.round(self.OS.eta[0], 3)}, E: {np.round(self.OS.eta[1], 3)}, " + r"$\psi$: " + f"{np.round(self.OS._rad_to_deg(self.OS.eta[2]), 3)}°"
        vel = f"u: {np.round(self.OS.nu[0], 3)}, v: {np.round(self.OS.nu[1], 3)}, r: {np.round(self.OS.nu[2], 3)}"
        return ste + "\n" + pos + "\n" + vel


    def render(self):
        """Renders the current environment."""

        # plot every nth timestep
        if self.step_cnt % 1 == 0: 

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
            self.ax0.set_xlim(0, self.E_max)
            self.ax0.set_ylim(0, self.N_max)
            self.ax0.set_xlabel("East")
            self.ax0.set_ylabel("North")

            # set OS
            N = self.OS.eta[0]
            E = self.OS.eta[1]
            
            rect = patches.Rectangle((E-self.OS.width/2, N-self.OS.length/2), self.OS.width, self.OS.length, -self.OS._rad_to_deg(self.OS.eta[2]), 
                                      linewidth=1, edgecolor='r', facecolor='none')
            # Note: negate angle since we define it clock-wise, contrary to plt
            
            self.ax0.add_patch(rect)
            self.ax0.text(E + 2.5, N + 2.5, self.__str__())
            self.ax0.text(self.E_max - 5, self.N_max - 5, r"$\psi_g$" + f": {np.round(self.OS._rad_to_deg(self._get_psi_g()),3)}°")

            # set goal (stored as NE)
            self.ax0.scatter(self.goal["E"], self.goal["N"], color="blue")

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
            #self.ax2.set_ylim(-1, 1.1)
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
                self.ax3_twin = self.ax3.twinx()
                self.ax3.old_time = 0
                self.ax3.old_action = 0
                self.ax3.old_rud_angle = 0

            self.ax3.set_xlim(0, self._max_episode_steps)
            self.ax3.set_ylim(-0.1, self.action_space.n - 1 + 0.1)
            self.ax3.set_yticks(range(self.action_space.n))
            self.ax3.set_yticklabels(range(self.action_space.n))
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("Action (discrete)")

            self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_action, self.OS.action], color="black", alpha=0.5)

            # add rudder angle plot
            self.ax3_twin.plot([self.ax3.old_time, self.step_cnt], [self.OS._rad_to_deg(self.ax3.old_rud_angle), self.OS._rad_to_deg(self.OS.rud_angle)], color="blue")
            self.ax3_twin.set_ylim(-self.OS._rad_to_deg(self.OS.rud_angle_max) - 5, self.OS._rad_to_deg(self.OS.rud_angle_max) + 5)
            self.ax3_twin.set_yticks(range(-int(self.OS._rad_to_deg(self.OS.rud_angle_max)), int(self.OS._rad_to_deg(self.OS.rud_angle_max)) + 5, 5))
            self.ax3_twin.set_yticklabels(range(-int(self.OS._rad_to_deg(self.OS.rud_angle_max)), int(self.OS._rad_to_deg(self.OS.rud_angle_max)) + 5, 5))
            self.ax3_twin.set_ylabel("Rudder angle (in °, blue)")

            self.ax3.old_time = self.step_cnt
            self.ax3.old_action = self.OS.action
            self.ax3.old_rud_angle = self.OS.rud_angle

            plt.pause(0.001)

"""
x = FossenCS2()
x.reset()
x.render()
x.OS.rud_angle = x.OS._deg_to_rad(5)

for _ in range(10000):
    x.step(0)
    print(x)
    x.render()
"""
