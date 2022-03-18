import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from tud_rl.envs.FossenFnc import dtr, angle_to_2pi


class CyberShipII:
    """This class provides a vessel behaving according to the nonlinear ship manoeuvering model (3 DOF) proposed in 
    Skjetne et al. (2004) in Modeling, Identification and Control."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, delta_t, N_max, E_max, cnt_approach, tau_u) -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation settings and dummy action for rendering
        self.delta_t     = delta_t
        self.N_max       = N_max
        self.E_max       = E_max
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
        assert cnt_approach in ["tau", "rps_angle", "f123"], "Unknown control approach."
        self.cnt_approach = cnt_approach


        # ---------------------- Approach 1: Control force vector directly (following Cheng & Zhang (2007)) ----------------------
        if cnt_approach == "tau":
            
            # system is underactuated in v-direction, u-force will be constant, r-tau compoment controlled
            self.tau_u         = tau_u         # initialization in Nm
            self.tau_cnt_r     = 0.0           # initialization in Nm
            self.tau_cnt_r_inc = 0.25          # increment in Nm
            self.tau_cnt_r_max = 1.0           # maximum (absolute value) in Nm           


        # ---------------------- Approach 2: Control rps and angle (following Skjetne et al. (2004)) ----------------------
        elif cnt_approach == "rps_angle":

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
            self.rud_angle_max = dtr(35)
            self.rud_angle_inc = dtr(5) * self.delta_t

            # init rudder angle
            self.rud_angle = 0


        # ---------------------- Approach 3: Control three thrusters (following Kordobad et al. (2021)) ----------------------
        elif cnt_approach == "f123":

            # the reference assumes that the vessel is fully actuated; we impose the restriction tau_v = 0 by setting f1 = 0
            self.lx = 0.5
            self.ly = 0.1

            self.B = np.array([[0, 1, 1],
                               [1, 0, 0],
                               [self.lx, -self.ly, self.ly]], dtype=np.float32)
            self.f1 = 0.0
            self.f2 = 2.0
            self.f3 = 2.0

            self.df23 = 1       # increment (in N)
            self.f23_max = 5    # maximum (in N)


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
            Control tau directly. Precisely, force in u-direction is fixed to 2 N, force in v-direction is fixed to 0 N, tau in r is controlled.

        Control Approach 2:
            Translates revolutions per second (n) and rudder angle (delta in rad) into tau. Currently, n is fixed.
        
        Control Approach 3:
            Translates actuator forces into tau."""

        if self.cnt_approach == "tau":
            self.tau = np.array([self.tau_u, 0.0, self.tau_cnt_r])

        elif self.cnt_approach == "rps_angle":

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


    def _upd_dynamics(self, euler=False):
        """Updates positions and velocities for next simulation step.
        Args:
            euler (bool):  Whether to use Euler integration or, if false, the RK45 procedure.
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

        # transform heading to [0, 2pi)
        self.eta[2] = angle_to_2pi(self.eta[2])


    def _control(self, a):
        """
        Control Approach 1:
            Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:

            0 - keep tau-r as is
            1 - increase tau-r by 0.25 Nm
            2 - decrease tau-r by 0.25 Nm

        Control Approach 2:
            Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:
            
            0 - keep rudder angle as is
            1 - increase rudder angle by 5 degree per second
            2 - decrease rudder angle by 5 degree per second
        
        Control Approach 3:
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

        if self.cnt_approach == "tau":
            
            assert a in range(3), "Unknown action."

            # update tau-r
            if a == 0:
                pass
            elif a == 1:
                self.tau_cnt_r += self.tau_cnt_r_inc
            elif a == 2:
                self.tau_cnt_r -= self.tau_cnt_r_inc
            
            # clip it
            self.tau_cnt_r = np.clip(self.tau_cnt_r, -self.tau_cnt_r_max, self.tau_cnt_r_max)


        elif self.cnt_approach == "rps_angle":

            assert a in range(3), "Unknown action."

            # update angle
            if a == 0:
                pass
            elif a == 1:
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


    def _get_sideslip(self):
        """Returns the sideslip angle in radiant."""
        u = self.nu[0]
        v = self.nu[1]

        frac = np.arctan(np.abs(v/u))

        if u >= 0 and v >= 0:
            return frac
        
        elif u >= 0 and v < 0:
            return -frac

        elif u < 0 and v >= 0:
            return np.pi - frac
        
        elif u < 0 and v < 0:
            return - (np.pi - frac)


    def _get_course(self):
        """Returns the course angle in radiant, which is heading + sideslip."""
        return angle_to_2pi(self.eta[2] + self._get_sideslip())


    def _get_V(self):
        """Returns the aggregated velocity."""
        return np.sqrt(self.nu[0]**2 + self.nu[1]**2)


    def _is_off_map(self):
        """Checks whether vessel left the map."""

        if self.eta[0] <= 0 or self.eta[0] >= self.N_max or self.eta[1] <= 0 or self.eta[1] >= self.E_max:
            return True
        return False


    def _tau_u_from_u(self, u):
        """Returns the thrust for in u-direction (in N) for maintaining a given longitudinal speed."""
        
        nu  = np.array([u, 0.0, 0.0])
        tau = np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu)
        return tau[0]


    def _u_from_tau_u(self, tau_u):
        """Returns the final longitudinal speed of a CSII under constant tau_u and zero other thrust."""

        tau = np.array([tau_u, 0.0, 0.0])

        def to_find_root_of(u):
            nu = np.array([u, 0.0, 0.0])
            vec = np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu) - tau
            return vec[0]

        return newton(func=to_find_root_of, x0=0.5)
