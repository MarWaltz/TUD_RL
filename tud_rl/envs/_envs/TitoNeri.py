import math

import numpy as np
from scipy.optimize import newton
from tud_rl.envs._envs.VesselFnc import angle_to_2pi


class TitoNeri:
    """This class provides a tug boat behaving according to the nonlinear ship manoeuvering model (3 DOF) outline in 
    Haseltalab & Negenborn (2019) in Applied Energy."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, delta_t, N_max, E_max, tug1) -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation settings and dummy action for rendering
        self.delta_t     = delta_t
        self.N_max       = N_max
        self.E_max       = E_max
        self.action      = 0

        # TitoNeri parameters
        self.length = 0.97
        self.width  = 0.3      
        self.m =  16.9

        # mass matrix (rigid body + added mass) and its inverse
        self.M_RB = np.array([[16.9,  0.0,  0.0],
                              [0.0,  16.9,  0.0],
                              [0.0,   0.0, 0.51]], dtype=np.float32)
        self.M_A = np.array([[1.2,  0.0, 0.0],
                             [0.0, 49.2, 0.0],
                             [0.0,  0.0, 1.8]], dtype=np.float32)
        self.M = self.M_RB + self.M_A
        self.M_inv = np.linalg.inv(self.M)

        # parameters of the towing system
        assert isinstance(tug1, bool), "Either tug1 or tug2."
        
        if tug1:
            self.li          = 0.67
            self.Fimax       = 3.0
            self.tauimax     = 5.0
            self.alpha_bar_i = 5.0 * self.delta_t
            self.F_bar_i     = 0.3 * self.delta_t
        else:
            self.li          = 0.585
            self.Fimax       = 3.0
            self.tauimax     = 5.0
            self.alpha_bar_i = 5.0 * self.delta_t
            self.F_bar_i     = 0.3 * self.delta_t

        # control parameters
        self.tau = np.array([0.0, 0.0, 0.0])      # control force applied to tug
        self.tau_max = np.array([5.0, 5.0, 2.5])  # maximum control force
        self.F_rope = 0.0                         # rope force

        #------------------------- Motion Initialization -----------------------------------
        self.eta = np.array([N_init, E_init, psi_init], dtype=np.float32)        # N (in m),   E (in m),   psi (in rad)   in NE-system
        self.nu  = np.array([u_init, v_init, r_init], dtype=np.float32)          # u (in m/s), v in (m/s), r (in rad/s)   in BODY-system

        self.nu_dot  = np.array([0.0, 0.0, 0.0], dtype=np.float32)


    def _C_of_nu(self, nu):
        """Computes centripetal/coriolis matrix for given velocities."""

        # unpacking
        u, v, _ = nu

        C = np.array([[     0.0,       0.0, -self.m*v],
                      [     0.0,       0.0,  self.m*u],
                      [self.m*v, -self.m*u,       0.0]])
        return C


    def _T_of_psi(self, psi):
        """Computes rotation matrix for given heading (in rad)."""

        return np.array([[math.cos(psi), -math.sin(psi), 0],
                         [math.sin(psi),  math.cos(psi), 0],
                         [0, 0, 1]])

    def _tug_dynamic(self, nu, ship_heading):
        """Returns nu_dot and nu_dot according to the nonlinear ship manoeuvering model."""
        # tug config matrix
        beta = ship_heading + self.alpha_rope - self.eta[2]
        B = np.array([math.cos(beta), math.sin(beta), self.lTi * math.sin(beta)])

        # dynamics
        M_nu_dot = self.tau + self.F * B - np.dot(self._C_of_nu(nu), nu)
        nu_dot = np.dot(self.M_inv, M_nu_dot)
        return nu_dot

    def _upd_dynamics(self, ship_heading):
        """Updates positions and velocities for next simulation step. Uses the ballistic approach of Treiber, Kanagaraj (2015).
        Args:
            ship_heading (float) : heading of the ship which is pulled, NOT the tug boat's heading"""

        # store current values
        eta_dot_old = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # euler update of velocities
        self.nu_dot = self._tug_dynamic(nu=self.nu, ship_heading=ship_heading)
        self.nu += self.nu_dot * self.delta_t

        # find new eta_dot via rotation
        eta_dot_new = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # trapezoidal update of positions
        self.eta += 0.5 * (eta_dot_old + eta_dot_new) * self.delta_t

        # transform heading to [0, 2pi)
        self.eta[2] = angle_to_2pi(self.eta[2])


    def _control(self, a):
        """Action-space is five-dimensional. First three actions correspond to change in control force of the tug,
        while fourth and fifth action control rope force and angle, respectively."""
        pass

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
