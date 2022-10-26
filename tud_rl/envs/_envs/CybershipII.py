import math

import numpy as np
from scipy.optimize import newton
from tud_rl.envs._envs.VesselFnc import angle_to_2pi


class CyberShipII:
    """This class provides a vessel behaving according to the nonlinear ship manoeuvering model (3 DOF) proposed in 
    Skjetne et al. (2004) in Modeling, Identification and Control."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, delta_t, N_max, E_max) -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation settings and dummy action for rendering
        self.delta_t     = delta_t
        self.N_max       = N_max
        self.E_max       = E_max
        self.action      = 0

        # CyberShip II parameters
        self.length = 1.255
        self.width  = 0.29
        
        self.m   = 23.8
        self.I_z = 1.76
        self.x_g = 0.046

        # for tug forces
        self.l1 = self.length*0.5 + self.x_g
        self.l2 = self.length*0.5 - self.x_g

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

        #------------------------- Motion Initialization -----------------------------------
        self.eta = np.array([N_init, E_init, psi_init], dtype=np.float32)        # N (in m),   E (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([u_init, v_init, r_init], dtype=np.float32)          # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system

        self.nu_dot = np.array([0.0, 0.0, 0.0])
        self.tau    = np.array([0.0, 0.0, 0.0])

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

        return np.array([[math.cos(psi), -math.sin(psi), 0],
                         [math.sin(psi),  math.cos(psi), 0],
                         [0, 0, 1]])

    def _ship_dynamic(self, nu):
        """Returns nu_dot and nu_dot according to the nonlinear ship manoeuvering model."""
        M_nu_dot = self.tau - np.dot(self._C_of_nu(nu) + self._D_of_nu(nu), nu)
        nu_dot = np.dot(self.M_inv, M_nu_dot)
        return nu_dot

    def _upd_dynamics(self):
        """Updates positions and velocities for next simulation step. Uses the ballistic approach of Treiber, Kanagaraj (2015)."""

        # store current values
        eta_dot_old = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # euler update of velocities
        self.nu_dot = self._ship_dynamic(nu = self.nu)
        self.nu += self.nu_dot * self.delta_t

        # find new eta_dot via rotation
        eta_dot_new = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # trapezoidal update of positions
        self.eta += 0.5 * (eta_dot_old + eta_dot_new) * self.delta_t

        # transform heading to [0, 2pi)
        self.eta[2] = angle_to_2pi(self.eta[2])

    def _get_sideslip(self):
        """Returns the sideslip angle in radiant."""
        u = self.nu[0]
        v = self.nu[1]
        return math.atan2(v, u)


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
