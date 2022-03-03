import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class CyberShipII:
    """This class provides a vessel behaving according to the nonlinear ship manoeuvering model (3 DOF) proposed in 
    Skjetne et al. (2004) in Modeling, Identification and Control."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, delta_t, N_max, E_max, domain_size, cnt_approach="f123") -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation settings and dummy action for rendering
        self.delta_t     = delta_t
        self.N_max       = N_max
        self.E_max       = E_max
        self.domain_size = domain_size
        self.action      = 0

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

        # mass matrix (rigid body + added mass) and its inverse
        self.M_RB = np.array([[self.m, 0, 0],
                              [0, self.m, self.m * self.x_g],
                              [0, self.m * self.x_g, self.I_z]], dtype=np.float32)
        self.M_A = np.array([[-self.X_dotu, 0, 0],
                              [0, -self.Y_dotv, -self.Y_dotr],
                              [0, -self.N_dotv, -self.N_dotr]], dtype=np.float32)
        self.M = self.M_RB + self.M_A
        self.M_inv = np.linalg.inv(self.M)

        # control approaches
        assert cnt_approach in ["rps_angle", "f123"], "Unknown control approach."
        self.cnt_approach = cnt_approach

        # ---------------------- Approach 1: Control rps and angle (following Skjetne et al. (2004)) ----------------------

        if cnt_approach == "rps_angle":

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
                               [np.abs(self.l_yT1), -np.abs(self.l_yT2), np.abs(self.l_xT3), -np.abs(self.l_xR1), -np.abs(self.l_xR2)]], dtype=np.float32)

            self.d_prop = 60e-3                          # diameter of propellers in m (from PhD-thesis of Karl-Petter W. Lindegaard)
            self.ku  = 0.5                               # induced velocity factor
            self.rho = 1014                              # density of sea water (in kg/m3)
            self.n_prop  = 2000 / 60 * self.delta_t      # revolutions per second of the two main propellers
            self.n_bow   =    0 / 60 * self.delta_t      # revolutions per second of the bow thruster

            # rudder angle max (in rad) and increment (in rad/s)
            self.rud_angle_max = self._dtr(35)
            self.rud_angle_inc = self._dtr(5) * self.delta_t

            # init rudder angle
            self.rud_angle = 0


        # ---------------------- Approach 2: Control three thrusters (following Kordobad et al. (2021)) ----------------------
        elif cnt_approach == "f123":

            # the reference assumes that the vessel is fully actuated; we impose the restriction tau_v = 0 by setting f1 = 0
            self.lx = 0.5
            self.ly = 0.1

            self.B = np.array([[0, 1, 1],
                               [1, 0, 0],
                               [self.lx, -self.ly, self.ly]], dtype=np.float32)
            self.f1 = 0.0
            self.f2 = 0.0
            self.f3 = 0.0

            self.df23 = 1       # increment (in N)
            self.f23_max = 8    # maximum (in N)


        #------------------------- Motion Initialization -----------------------------------
        self.eta = np.array([N_init, E_init, psi_init], dtype=np.float32)        # N (in m),   E (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([u_init, v_init, r_init], dtype=np.float32)          # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system
        self._set_tau()


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


    def _set_tau(self):
        """
        Control Approach 1:
            Translates revolutions per second (n) and rudder angle (delta in rad) into tau. Currently, n is fixed.
        
        Control Approach 2:
            Translates actuator forces into tau."""

        if self.cnt_approach == "rps_angle":

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
        
        elif self.cnt_approach == "f123":

            self.tau = np.dot(self.B, np.array([self.f1, self.f2, self.f3], dtype=np.float32))


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


    def _upd_dynamics(self, euler=False, mirrow=False):
        """Updates positions and velocities for next simulation step. Uses basic Euler method.
        Arguments:

        euler (bool):  Whether to use Euler integration or, if false, the RK45 procedure.
        mirrow (bool): Whether the vessel should by mirrowed if it hits the boundary of the simulation area. 
                       Inspired by Xu et al. (2022, Neurocomputing).
        """

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

        # transform heading to [0, 2pi]
        self.eta[2] = self._clip_angle(self.eta[2])

        # stay on map: either mirrow mode or by adjusting NE
        if self._is_off_map():
            
            if mirrow:
                # quick access
                psi = self.eta[2]

                # right or left bound (E-axis)
                if self.eta[1] <= 0 or self.eta[1] >= self.E_max:
                    self.eta[2] = 2*np.pi - psi
                
                # upper and lower bound (N-axis)
                else:
                    self.eta[2] = np.pi - psi
            
            else:
                self.eta[0] = np.clip(self.eta[0], 0, self.N_max)
                self.eta[1] = np.clip(self.eta[1], 0, self.E_max)


    def _upd_tau(self, a):
        """
        Control Approach 1:
            Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:
            
            0 - keep rudder angle as is
            1 - increase rudder angle by 5 degree per second
            2 - decrease rudder angle by 5 degree per second
        
        Control Approach 2:
            Action 'a' is an integer taking values in [0, 1, 2, ..., 8]. They correspond to:

            0 - increase f2, increase f3
            1 - increase f2, keep f3
            2 - increase f2, decrease f3
            
            3 - keep f2, increase f3
            4 - keep f2, keep f3
            5 - keep f2, decrease f3

            6 - decrease f2, increase f3
            7 - decrease f2, keep f3
            8 - decrease f2, decrease f3
        """
        # store action for rendering
        self.action = a

        if self.cnt_approach == "rps_angle":

            assert a in range(3), "Unknown action."

            # update angle
            if a == 1:
                self.rud_angle += self.rud_angle_inc
            elif a == 2:
                self.rud_angle -= self.rud_angle_inc
            
            # clip it
            self.rud_angle = np.clip(self.rud_angle, -self.rud_angle_max, self.rud_angle_max)
        

        elif self.cnt_approach == "f123":

            assert a in range(9), "Unknown action."

            # increase f2
            if a == 0:
                self.f2 += self.df23
                self.f3 += self.df23
            
            elif a == 1:
                self.f2 += self.df23
            
            elif a == 2:
                self.f2 += self.df23
                self.f3 -= self.df23
            
            # keep f2
            elif a == 3:
                self.f3 += self.df23
            
            elif a == 4:
                pass

            elif a == 5:
                self.f3 -= self.df23
            
            # decrease f2
            elif a == 6:
                self.f2 -= self.df23
                self.f3 += self.df23
            
            elif a == 7:
                self.f2 -= self.df23

            elif a == 8:
                self.f2 -= self.df23
                self.f3 -= self.df23
            
            # clip both
            self.f2 = np.clip(self.f2, 0, self.f23_max)
            self.f3 = np.clip(self.f3, 0, self.f23_max)

        # update the control tau
        self._set_tau()


    def _dtr(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * np.pi / 180


    def _rtd(self, angle):
        """Takes angle in degree an transforms it to radiant."""
        return angle * 180 / np.pi


    def _clip_angle(self, angle):
        """Clips an angle to [0, 2pi]."""

        # clip it to [-2pi, 2pi]
        if np.abs(angle) > 2*np.pi:
            angle = np.sign(angle) * (np.abs(angle) - 2*np.pi)

        # clip it to [0, 2pi]
        if angle < 0:
            return 2*np.pi + angle
        return angle


    def _is_off_map(self):
        """Checks whether vessel left the map."""

        if self.eta[0] <= 0 or self.eta[0] >= self.N_max or self.eta[1] <= 0 or self.eta[1] >= self.E_max:
            return True
        return False


class StaticObstacle:
    """A static circle-shaped obstacle."""
    
    def __init__(self, N_init, E_init, max_radius) -> None:
        
        # spawning point
        self.N = N_init
        self.E = E_init

        # size
        self.radius = np.random.uniform(1, max_radius)
        self.radius_norm = self.radius / max_radius


class FossenCS2(gym.Env):
    """This environment contains an agent steering a CyberShip II."""

    def __init__(self, cnt_approach="f123"):
        super().__init__()

        # simulation settings
        self.delta_t      = 0.5              # simulation time interval (in s)
        self.N_max        = 50               # maximum N-coordinate (in m)
        self.E_max        = 50               # maximum E-coordinate (in m)
        self.N_statO      = 3                # number of static obstacles
        self.N_TSs        = 0                # number of other vessels
        self.domain_size  = 15               # size of the simplified ship domain (in m, circle around the agent and vessels)
        self.cnt_approach = cnt_approach     # whether to control actuator forces or rudder angle and rps directly

        # gym definitions
        obs_size = 8 + self.N_statO * 2
        self.observation_space  = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                             high = np.full(obs_size,  np.inf, dtype=np.float32))
        
        if cnt_approach == "rps_angle":
            self.action_space = spaces.Discrete(3)

        elif cnt_approach == "f123":
            self.action_space = spaces.Discrete(9)

        # custom inits
        self._max_episode_steps = 1e3
        self.r = 0
        self.r_head = 0
        self.r_dist = 0
        self.r_coll = 0
        self.r_coll_sigma = 5
        self.state_names = ["u", "v", "r", "N_rel", "E_rel", r"$\Psi$", r"$\Psi_e$", "ED"]


    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init goal
        self.goal = {"N" : np.random.uniform(self.N_max - 25, self.N_max),
                     "E" : np.random.uniform(self.E_max - 25, self.E_max)}

        # init static obstacles
        self.statOs = [StaticObstacle(N_init = np.random.uniform(15, self.N_max),
                                      E_init = np.random.uniform(15, self.E_max),
                                      max_radius = 5) for _ in range(self.N_statO)]

        # init other vessels
        self.TSs = [CyberShipII(N_init       = np.random.uniform(15, self.N_max), 
                                E_init       = np.random.uniform(15, self.E_max), 
                                psi_init     = np.random.uniform(0, np.pi),
                                u_init       = np.random.uniform(0, 1),
                                v_init       = 0.0,
                                r_init       = 0.0,
                                delta_t      = self.delta_t,
                                N_max        = self.N_max,
                                E_max        = self.E_max,
                                domain_size  = 7.5,
                                cnt_approach = self.cnt_approach) for _ in range(self.N_TSs)]

        # init agent (OS for 'Own Ship') and calculate initial distance to goal
        self.OS = CyberShipII(N_init      = 10.0, 
                              E_init      = 10.0, 
                              psi_init    = np.random.uniform(0, np.pi),
                              u_init      = np.random.uniform(0, 1),
                              v_init      = 0.0,
                              r_init      = 0.0,
                              delta_t     = self.delta_t,
                              N_max       = self.N_max,
                              E_max       = self.E_max,
                              domain_size = self.domain_size,
                              cnt_approach = self.cnt_approach)

        self.OS_goal_ED_init = self._ED(N1=self.goal["N"], E1=self.goal["E"])
        
        # init state
        self._set_state()
        self.state_init = self.state

        return self.state


    def _set_state(self):
        """State consists of (all from agent's perspective): 
        
        OS related:
        u, v, r, N_rel, E_rel, heading

        Goal related:
        heading_error, ED_goal

        Static obstacle related (for each, sorted by ED):
        euclidean distance to closest point
        heading error from agent's view
        """

        #--- OS related ---
        state_OS = np.concatenate([self.OS.nu, np.array([self.OS.eta[0] / self.N_max, 
                                                         self.OS.eta[1] / self.E_max, 
                                                         self.OS.eta[2] / (2*np.pi)])])

        #--- goal related ---
        OS_goal_ED = self._ED(N1=self.goal["N"], E1=self.goal["E"])

        state_goal = np.array([self._get_psi_e_to_point(N1=self.goal["N"], E1=self.goal["E"]) / (np.pi), 
                               OS_goal_ED / self.OS_goal_ED_init])

        #--- static obstacle related ---
        state_statOs = []

        for obs in self.statOs:

            # normalized distance to clostest point (ED - radius)
            ED_norm = (self._ED(N1=obs.N, E1=obs.E) - obs.radius_norm)/ self.OS_goal_ED_init
            
            # heading error from agent's view
            head_err = self._get_psi_e_to_point(N1=obs.N, E1=obs.E) / (np.pi)
            
            # store it
            state_statOs.append([ED_norm, head_err])
        
        # sort according to ascending euclidean distance to agent
        state_statOs = np.array(sorted(state_statOs, key=lambda x: x[0]))
        state_statOs = state_statOs.flatten(order="F")

        #--- combine state ---
        self.state = np.concatenate([state_OS, state_goal, state_statOs])


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # update control tau
        self.OS._upd_tau(a)

        # update agent dynamics
        self.OS._upd_dynamics(mirrow=False)

        # update environmental dynamics, e.g., other vessels
        [TS._upd_dynamics(mirrow=True) for TS in self.TSs]

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t
        
        return self.state, self.r, d, {}


    def _calculate_reward(self, w_dist=3, w_head=1, w_coll=1, w_map=1):
        """Returns reward of the current state."""

        # ---- Path planning reward (Xu et al. 2022) -----

        # 1. Distance reward
        OS_goal_ED = self._ED(N1=self.goal["N"], E1=self.goal["E"])
        r_dist = - OS_goal_ED / self.OS_goal_ED_init

        # 2. Heading reward
        r_head = - self._get_abs_psi_e_to_point(N1=self.goal["N"], E1=self.goal["E"]) / np.pi

        # --- Collision reward ----
        #r_coll = -10 if any([self._ED(N1=obs.N, E1=obs.E) <= obs.radius for obs in self.statOs]) else 0
        r_coll = 0

        for obs in self.statOs:
            num = np.exp(-0.5 * self._ED(N1=obs.N, E1=obs.E, sqrt=False) / self.r_coll_sigma**2)
            den = np.exp(-0.5 * obs.radius**2 / self.r_coll_sigma**2)
            r_coll -= num/den

        # --- Leave-the-map reward ---
        r_map = -10 if self.OS._is_off_map() else 0

        # overall reward
        self.r_dist = r_dist
        self.r_head = r_head
        self.r_coll = r_coll
        self.r_map  = r_map
        self.r = w_dist * r_dist + w_head * r_head + w_coll * r_coll + w_map * r_map


    def _get_psi_e_to_point(self, N1, E1):
        """Computes heading error of the agent towards a point (N1, E1)."""

        value = self._get_abs_psi_e_to_point(N1=N1, E1=E1)
        sign  = self._get_sign_psi_e_to_point(N1=N1, E1=E1)

        return sign * value


    def _get_abs_psi_e_to_point(self, N1, E1):
        """Calculates the absolute value of the heading error of the agent towards the given point (N1, E1) (target_angle - heading)."""

        # compute angle of (N1, E1)
        psi_g = self._get_psi_to_point(N1=N1, E1=E1)

        # compute absolute heading error
        psi_e_abs = np.abs(psi_g - self.OS.eta[2])

        # keep it in [0, pi]
        if psi_e_abs <= np.pi:
            return psi_e_abs
        else:
            return 2*np.pi - psi_e_abs


    def _get_sign_psi_e_to_point(self, N1, E1):
        """Calculates the sign of the heading error of the agent towards the given point (N1, E1)."""
        
        # compute angle of (N1, E1)
        psi_g = self._get_psi_to_point(N1=N1, E1=E1)
        
        # get heading of agent
        psi = self.OS.eta[2]

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


    def _get_psi_to_point(self, N1, E1):
        """Calculates the angle in [0, 2pi] between agent and a point (N1, E1) in the NE-system. 
        True zero is along the N-axis. Perspective centered at the agent."""
        
        # quick access
        N0 = self.OS.eta[0]
        E0 = self.OS.eta[1]

        # compute ED to goal
        ED = self._ED(N1=N1, E1=E1)

        # differentiate between goal position from NE perspective centered at the agent
        if N0 <= N1:

            # I. quadrant
            if E0 <= E1:
                psi_2P = np.arccos((N1 - N0) / ED)

            # II. quadrant
            else:
                psi_2P = 2*np.pi - np.arccos((N1 - N0) / ED)

        else:

            # III. quadrant
            if E0 >= E1:
                psi_2P = np.pi + np.arccos((N0 - N1) / ED)
            
            # IV. quadrant
            else:
                psi_2P = np.pi - np.arccos((N0 - N1) / ED)

        return psi_2P


    def _done(self):
        """Returns boolean flag whether episode is over."""

        # goal reached
        OS_goal_ED = self._ED(N1=self.goal["N"], E1=self.goal["E"])
        if OS_goal_ED <= 2.5:
            return True

        # out of the simulation area
        #if self.OS._is_off_map():
        #    return True

        # collision with a static obstacle
        #if any([self._ED(N1=obs.N, E1=obs.E) <= obs.radius for obs in self.statOs]):
        #    return True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True

        return False


    def _ED(self, N1, E1, sqrt=True):
        """Computes the euclidean distance of the agent to a point in the NE-system."""
        d_sq = (self.OS.eta[0] - N1)**2 + (self.OS.eta[1] - E1)**2

        if sqrt:
            return np.sqrt(d_sq)
        return d_sq


    def __str__(self) -> str:
        ste = f"Step: {self.step_cnt}"
        pos = f"N: {np.round(self.OS.eta[0], 3)}, E: {np.round(self.OS.eta[1], 3)}, " + r"$\psi$: " + f"{np.round(self.OS._rtd(self.OS.eta[2]), 3)}°"
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
            self.ax0.set_xlim(-5, self.E_max)
            self.ax0.set_ylim(-5, self.N_max)
            self.ax0.set_xlabel("East")
            self.ax0.set_ylabel("North")

            # set OS
            N = self.OS.eta[0]
            E = self.OS.eta[1]
            
            rect = patches.Rectangle((E-self.OS.width/2, N-self.OS.length/2), self.OS.width, self.OS.length, -self.OS._rtd(self.OS.eta[2]), 
                                      linewidth=1, edgecolor='red', facecolor='none')
            # Note: negate angle since we define it clock-wise, contrary to plt
            
            self.ax0.add_patch(rect)
            self.ax0.text(E + 2.5, N + 2.5, self.__str__())

            # set ship domain
            circ = patches.Circle((E, N), radius=self.OS.domain_size, edgecolor='red', facecolor='none', alpha=0.75)
            self.ax0.add_patch(circ)

            # set goal (stored as NE)
            self.ax0.scatter(self.goal["E"], self.goal["N"], color="blue")
            self.ax0.text(self.goal["E"], self.goal["N"] + 2,
                          r"$\psi_g$" + f": {np.round(self.OS._rtd(self._get_psi_to_point(N1=self.goal['N'], E1=self.goal['E'])),3)}°",
                          horizontalalignment='center', verticalalignment='center', color='blue')

            # set other vessels
            for TS in self.TSs:
                N = TS.eta[0]
                E = TS.eta[1]

                # vessel
                rect = patches.Rectangle((E-TS.width/2, N-TS.length/2), TS.width, TS.length, -TS._rtd(TS.eta[2]), 
                                          linewidth=1, edgecolor='darkred', facecolor='none')
                self.ax0.add_patch(rect)

                # domain
                #circ = patches.Circle((E, N), radius=TS.domain_size, edgecolor='darkred', facecolor='none', alpha=0.75)
                #self.ax0.add_patch(circ)

            # set static obstacles
            for obs_id, obs in enumerate(self.statOs):
                circ = patches.Circle((obs.E, obs.N), radius=obs.radius, edgecolor='green', facecolor='none', alpha=0.75)
                self.ax0.add_patch(circ)
                self.ax0.text(obs.E, obs.N, str(obs_id), horizontalalignment='center', verticalalignment='center', color='green')
                self.ax0.text(obs.E, obs.N - 3, rf"$\psi_{obs_id}$" + f": {np.round(self.OS._rtd(self._get_psi_to_point(N1=obs.N, E1=obs.E)),3)}°",
                              horizontalalignment='center', verticalalignment='center', color='green')

            # ----- reward plot ----
            if self.step_cnt == 0:
                self.ax1.clear()
                self.ax1.old_time = 0
                self.ax1.old_r_head = 0
                self.ax1.old_r_dist = 0
                self.ax1.old_r_coll = 0

            self.ax1.set_xlim(0, self._max_episode_steps)
            #self.ax1.set_ylim(-1.25, 0.1)
            self.ax1.set_xlabel("Timestep in episode")
            self.ax1.set_ylabel("Reward")

            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_head, self.r_head], color = "blue", label="Heading")
            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_dist, self.r_dist], color = "black", label="Distance")
            self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_coll, self.r_coll], color = "green", label="Collision")
            
            if self.step_cnt == 0:
                self.ax1.legend()

            self.ax1.old_time = self.step_cnt
            self.ax1.old_r_head = self.r_head
            self.ax1.old_r_dist = self.r_dist
            self.ax1.old_r_coll = self.r_coll

            # ---- state plot ----
            if self.step_cnt == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_state = self.state_init

            self.ax2.set_xlim(0, self._max_episode_steps)
            #self.ax2.set_ylim(-1, 1.1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("State information")

            for i in range(8):
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
            if self.cnt_approach == "rps_angle":
                self.ax3_twin.plot([self.ax3.old_time, self.step_cnt], [self.OS._rtd(self.ax3.old_rud_angle), self.OS._rtd(self.OS.rud_angle)], color="blue")
                self.ax3_twin.set_ylim(-self.OS._rtd(self.OS.rud_angle_max) - 5, self.OS._rtd(self.OS.rud_angle_max) + 5)
                self.ax3_twin.set_yticks(range(-int(self.OS._rtd(self.OS.rud_angle_max)), int(self.OS._rtd(self.OS.rud_angle_max)) + 5, 5))
                self.ax3_twin.set_yticklabels(range(-int(self.OS._rtd(self.OS.rud_angle_max)), int(self.OS._rtd(self.OS.rud_angle_max)) + 5, 5))
                self.ax3_twin.set_ylabel("Rudder angle (in °, blue)")
                self.ax3.old_rud_angle = self.OS.rud_angle

            self.ax3.old_time = self.step_cnt
            self.ax3.old_action = self.OS.action

            plt.pause(0.001)
