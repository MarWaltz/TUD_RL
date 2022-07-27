import math

import numpy as np
from scipy.optimize import newton
from tud_rl.envs._envs.VesselFnc import (angle_to_2pi, angle_to_pi, dtr,
                                         polar_from_xy, xy_from_polar)


class KVLCC2:
    """This class provides a KVLCC2 tanker behaving according to the MMG standard model of Yasukawa, Yoshimura (2015)."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, nps, delta_t, N_max, E_max, full_ship=True) -> None:

        #------------------------- Parameter/Settings -----------------------------------

        # store simulation settings and dummy action for rendering
        self.delta_t = delta_t
        self.N_max   = N_max
        self.E_max   = E_max
        self.action  = 0

        # KVLCC2 parameters
        if full_ship:

            self.kvlcc2 = {
                "C_b":          0.810,          # Block Coefficient
                "Lpp":          320.0,          # Length over perpendiculars (m)
                "B":            58.,            # Overall width
                "m":            312_600*1020,   # Mass of ship as calculated by ▽*rho (displacement * water density)
                "w_P0":         0.35,           # Assumed wake fraction coefficient
                "J_int":        0.4,            # Intercept for the calculation of K_T (https://doi.org/10.1615/ICHMT.2012.ProcSevIntSympTurbHeatTransfPal.500)
                "J_slo":       -0.5,            # Slope for the calculation of K_T
                "x_G":          11.2,           # X-Coordinate of the center of gravity (m)
                "x_P":         -160.0,          # X-Coordinate of the propeller (-0.5*Lpp)
                "D_p":          9.86,           # Diameter of propeller (m)
                "k_0":          0.2931,         # Same value as "J_int" | Propeller open water coefficients. 
                "k_1":         -0.2753,
                "k_2":         -0.1359,    
                "C_1":          2.0,
                "C_2_plus":     1.6,
                "C_2_minus":    1.1,
                "l_R":         -0.710,          # correction of flow straightening factor to yaw-rate
                "gamma_R":      None,      
                "gamma_R_plus": 0.640,          # Flow straightening coefficient for positive rudder angles
                "gamma_R_minus":0.395,          # Flow straightening coefficient for negative rudder angles
                "eta_param":    0.626,          # Ratio of propeller diameter to rudder span
                "kappa":        0.50,           # An experimental constant for expressing "u_R"
                "A_R":          112.5,          # Moveable rudder area
                "epsilon":      1.09,           # Ratio of wake fraction at propeller and rudder positions ((1 - w_R) / (1 - w_P))
                "A_R_Ld_em":    1/46.8,         # Fraction of moveable Rudder area to length*draft
                "f_alpha":      2.747,          # Rudder lift gradient coefficient (assumed rudder aspect ratio = 2)
                "rho":          1020,           # Water density of seawater
                "t_R":          0.387,          # Steering resistance deduction factor
                "t_P":          0.220,          # Thrust deduction factor. TODO give this more than an arbitrary value
                "x_H_dash":    -0.464,          # Longitudinal coordinate of acting point of the additional lateral force
                "d":            20.8,           # Ship draft (Tiefgang)
                "m_x_dash":     0.022,          # Non dimensionalized added masses coefficient in x direction
                "m_y_dash":     0.223,          # Non dimensionalized added masses coefficient in y direction
                "R_0_dash":     0.022,          # frictional resistance coefficient TODO Estimate this via Schoenherr's formula
                "X_vv_dash":   -0.040,          # Hull derivatives
                "X_vr_dash":    0.002,          # Hull derivatives
                "X_rr_dash":    0.011,          # Hull derivatives
                "X_vvvv_dash":  0.771,          # Hull derivatives
                "Y_v_dash":    -0.315,          # Hull derivatives
                "Y_r_dash":     0.083,          # Hull derivatives
                "Y_vvv_dash":  -1.607,          # Hull derivatives
                "Y_vvr_dash":   0.379,          # Hull derivatives
                "Y_vrr_dash":  -0.391,          # Hull derivatives
                "Y_rrr_dash":   0.008,          # Hull derivatives
                "N_v_dash":    -0.137,          # Hull derivatives
                "N_r_dash":    -0.049,          # Hull derivatives
                "N_vvv_dash":  -0.030,          # Hull derivatives
                "N_vvr_dash":  -0.294,          # Hull derivatives
                "N_vrr_dash":   0.055,          # Hull derivatives
                "N_rrr_dash":  -0.013,          # Hull derivatives
                "I_zG":         2e12,           # Moment of inertia of ship around center of gravity (m*(0.25*Lpp)**2) (Point mass Inertia)
                "J_z_dash":     0.011,          # Added moment of inertia coefficient
                "a_H":          0.312           # Rudder force increase factor
            }

        else:
            # scale 1:5 replica of the original tanker
            self.kvlcc2 = {
                "C_b":          0.810,          # Block Coeffiient
                "Lpp":          64,             # Length over pependiculars (m)
                "B":            11.6,           # Overall width
                "displ":        2500.8,         # Displacement in [m³]
                "w_P0":         0.35,           # Assumed wake fraction coefficient
                "J_int":        0.4,            # Intercept for the calculation of K_T (https://doi.org/10.1615/ICHMT.2012.ProcSevIntSympTurbHeatTransfPal.500)
                "J_slo":       -0.5,            # Slope for the calculation of K_T
                "x_G":          2.24,           # X-Coordinate of the center of gravity (m)
                "x_P":         -32.0,           # X-Coordinate of the propeller (-0.5*Lpp)
                "D_p":          1.972,          # Diameter of propeller (m)
                "k_0":          0.2931,         # Same value as "J_int" | Propeller open water coefficients. 
                "k_1":         -0.2753,
                "k_2":         -0.1359,
                "C_1":          2.0,
                "C_2_plus":     1.6,
                "C_2_minus":    1.1,
                "l_R":         -0.710,          # correction of flow straightening factor to yaw-rate
                "gamma_R":      None,
                "gamma_R_plus": 0.640,          # Flow straightening coefficient for positive rudder angles
                "gamma_R_minus":0.395,          # Flow straightening coefficient for negative rudder angles
                "eta_param":    0.626,          # Ratio of propeller diameter to rudder span
                "kappa":        0.50,           # An experimental constant for expressing "u_R"
                "A_R":          4.5,            # Moveable rudder area
                "epsilon":      1.09,           # Ratio of wake fraction at propeller and rudder positions ((1 - w_R) / (1 - w_P))
                "f_alpha":      2.747,          # Rudder lift gradient coefficient (assumed rudder aspect ratio = 2)
                "rho":          1020,           # Water density of seawater
                "t_R":          0.387,          # Steering resistance deduction factor
                "t_P":          0.220,          # Thrust deduction factor. TODO give this more than an arbitrary value
                "x_H_dash":    -0.464,          # Longitudinal coordinate of acting point of the additional lateral force
                "d":            4.16,           # Ship draft (Tiefgang)
                "m_x_dash":     0.022,          # Non dimensionalized added masses coefficient in x direction
                "m_y_dash":     0.223,          # Non dimensionalized added masses coefficient in y direction
                "R_0_dash":     0.022,          # frictional resistance coefficient TODO Estimate this via Schoenherr's formula
                "X_vv_dash":   -0.040,          # Hull derivatives
                "X_vr_dash":    0.002,          # Hull derivatives
                "X_rr_dash":    0.011,          # Hull derivatives
                "X_vvvv_dash":  0.771,          # Hull derivatives
                "Y_v_dash":    -0.315,          # Hull derivatives
                "Y_r_dash":     0.083,          # Hull derivatives
                "Y_vvv_dash":  -1.607,          # Hull derivatives
                "Y_vvr_dash":   0.379,          # Hull derivatives
                "Y_vrr_dash":  -0.391,          # Hull derivatives
                "Y_rrr_dash":   0.008,          # Hull derivatives
                "N_v_dash":    -0.137,          # Hull derivatives
                "N_r_dash":    -0.049,          # Hull derivatives
                "N_vvv_dash":  -0.030,          # Hull derivatives
                "N_vvr_dash":  -0.294,          # Hull derivatives
                "N_vrr_dash":   0.055,          # Hull derivatives
                "N_rrr_dash":  -0.013,          # Hull derivatives
                "J_z_dash":     0.011,          # Added moment of inertia coefficient
                "a_H":          0.312           # Rudder force increase factor
            }
            self.kvlcc2["m"] = self.kvlcc2["displ"] * self.kvlcc2["rho"]
            self.kvlcc2["I_zG"] = self.kvlcc2["m"] * (0.25* self.kvlcc2["Lpp"])**2

        for key, value in self.kvlcc2.items():
            setattr(self, key, value)

        # in [m]
        self.ship_domain_A = 3 * self.Lpp + 0.5 * self.Lpp
        self.ship_domain_B = 1 * self.Lpp + 0.5 * self.B   
        self.ship_domain_C = 1 * self.Lpp + 0.5 * self.Lpp
        self.ship_domain_D = 3 * self.Lpp + 0.5 * self.B

        #------------------------- Motion Initialization -----------------------------------
        # Propeller revolutions [s⁻¹]
        self.nps = nps

        # rudder angle max (in rad) and increment (in rad/s)
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)

        # init rudder angle
        self.rud_angle = 0.0

        # eta, nu
        self.eta = np.array([N_init, E_init, psi_init], dtype=np.float32)  # N (in m),   E (in m),    psi (in rad)   in NE-system
        self.nu  = np.array([u_init, v_init, r_init], dtype=np.float32)    # u (in m/s), vm in (m/s), r (in rad/s)   in (midship-centered) BODY-system

        self.nu_dot  = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _T_of_psi(self, psi):
        """Computes rotation matrix for given heading (in rad)."""
        return np.array([[math.cos(psi), -math.sin(psi), 0.],
                         [math.sin(psi),  math.cos(psi), 0.],
                         [0., 0., 1.]])

    def _mmg_dynamics(self, nu, psi, rud_angle, nps, beta_c, V_c) -> np.ndarray:
        """System of ODEs after Yasukawa & Yoshimura (2015, Journal of Marine Science and Technology) for the MMG standard model. 
        Current forces are taken from Fossen (2021) and Budak & Beji (2020, Ocean Engineering). See also Chen et al. (2013, Ocean Engineering).
        Returns nu_dot."""          

        #----------------------------- preparation ------------------------------
        # unpack values
        u, vm, r = nu

        # current adjustments
        u_c = V_c * math.cos(angle_to_2pi(beta_c - psi - math.pi))
        v_c = V_c * math.sin(angle_to_2pi(beta_c - psi - math.pi))

        u -= u_c
        vm -= v_c

        U = math.sqrt(u**2 + vm**2)

        if U == 0.0:
            beta = 0.0
            v_dash = 0.0
            r_dash = 0.0
        else:
            beta = math.atan2(-vm, u)   # drift angle at midship position
            v_dash = vm / U             # non-dimensionalized lateral velocity
            r_dash = r * self.Lpp / U   # non-dimensionalized yaw rate

        #---------------- hydrodynamic forces acting on ship hull ----------------
        X_H = (0.5 * self.rho * self.Lpp * self.d * (U**2) * (
            - self.R_0_dash
            + self.X_vv_dash * (v_dash**2)
            + self.X_vr_dash * v_dash * r_dash
            + self.X_rr_dash * (r_dash**2)
            + self.X_vvvv_dash * (v_dash**4)
        )
        )

        Y_H = (0.5 * self.rho * self.Lpp * self.d * (U**2) * (
            self.Y_v_dash * v_dash
            + self.Y_r_dash * r_dash
            + self.Y_vvv_dash * (v_dash**3)
            + self.Y_vvr_dash * (v_dash**2) * r_dash
            + self.Y_vrr_dash * v_dash * (r_dash**2)
            + self.Y_rrr_dash * (r_dash**3)
        )
        )

        N_H = (0.5 * self.rho * (self.Lpp**2) * self.d * (U**2) * (
            self.N_v_dash * v_dash
            + self.N_r_dash * r_dash
            + self.N_vvv_dash * (v_dash**3)
            + self.N_vvr_dash * (v_dash**2) * r_dash
            + self.N_vrr_dash * v_dash * (r_dash**2)
            + self.N_rrr_dash * (r_dash**3)
        )
        )

        #---------------- longitudinal surge force due to propeller ----------------
        # redefine
        beta_P = beta - (self.x_P/self.Lpp) * r_dash

        if all([key in self.kvlcc2.keys() for key in ["C_1","C_2_plus","C_2_minus"]]):
            C_2 = self.C_2_plus if beta_P >= 0 else self.C_2_minus

            tmp = 1-math.exp(-self.C_1*abs(beta_P))*(C_2-1)
            w_P = 1-(1-self.w_P0)*(1+tmp)
        else:
            w_P = self.w_P0 * math.exp(-4.0 * (beta_P)**2)

        if nps == 0.0:  # no propeller movement, no advance ratio
            J = 0.0
        else:
            J = (1 - w_P) * u / (nps * self.D_p)  # propeller advance ratio

        if all([key in self.kvlcc2.keys() for key in ["k_0", "k_1", "k_2"]]):
            # propeller thrust open water characteristic
            K_T = self.k_0 + (self.k_1 * J) + (self.k_2 * J**2)
        else:
            # inferred slope + intercept dependent on J (empirical)
            K_T = self.J_slo * J + self.J_int

        X_P = (1 - self.t_P) * self.rho * K_T * nps**2 * self.D_p**4


        #--------------------- hydrodynamic forces by steering ----------------------
        # effective inflow angle to rudder in maneuvering motions
        beta_R = beta - self.l_R * r_dash

        # flow straightening coefficient
        if self.gamma_R is not None:
            gamma_R = self.gamma_R
        else:
            if beta_R < 0.0:
                gamma_R = self.gamma_R_minus
            else:
                gamma_R = self.gamma_R_plus

        # lateral inflow velocity components to rudder
        v_R = U * gamma_R * beta_R

        # longitudinal inflow velocity components to rudder
        if J == 0.0:
            u_R = 0.0
        else:
            u_R = u * (1 - w_P) * self.epsilon * math.sqrt(
                self.eta_param * (1.0 + self.kappa * (math.sqrt(1.0 + 8.0 * K_T / (np.pi * J**2)) - 1))**2 + (1 - self.eta_param)
            )
        # rudder inflow velocity
        U_R = math.sqrt(u_R**2 + v_R**2)

        # rudder inflow angle
        alpha_R = rud_angle - math.atan2(v_R, u_R)

        # normal force on rudder
        F_N = 0.5 * self.A_R * self.rho * self.f_alpha * (U_R**2) * math.sin(alpha_R)

        # longitudinal surge force around midship by steering
        X_R = -(1 - self.t_R) * F_N * math.sin(rud_angle)

        # lateral surge force by steering
        Y_R = -(1 + self.a_H) * F_N * math.cos(rud_angle)

        # redimensionalize x_H
        x_H = self.x_H_dash * self.Lpp

        # yaw moment around midship by steering
        N_R = -(-0.5 + self.a_H * x_H) * F_N * math.cos(rud_angle)

        #------------------------- forces related to currents --------------------------
        if V_c is not None and V_c != 0.:

            # Longitudinal velocity of current dependent on ship heading
            u_c = V_c * math.cos(angle_to_2pi(beta_c - psi))
            u_rc = u - u_c

            # Lateral velocity of current dependent on ship heading
            v_c = V_c * math.sin(angle_to_2pi(beta_c - psi))
            v_rc = vm - v_c

            V_rc_sq = u_rc**2 + v_rc**2
            g_rc = -math.atan2(v_rc,u_rc)

            # Longitudinal current force
            A_Fc = self.B * self.d * self.C_b
            X_C = 0.5 * self.rho * A_Fc * self._C_X(g_rc) * V_rc_sq

            # Lateral current force
            A_Lc = self.Lpp * self.d * self.C_b
            Y_C = 0.5 * self.rho * A_Lc * self._C_Y(g_rc) * V_rc_sq

            # Current Moment
            N_C = 0.5 * self.rho * A_Lc * self.Lpp * self._C_N(g_rc)* V_rc_sq

            #--- not considered at the moment ---
            X_C, Y_C, N_C = 0.0, 0.0, 0.0

        else:
            X_C, Y_C, N_C = 0.0, 0.0, 0.0

        #-------------------------- Equation solving ----------------------------
        # added masses and added moment of inertia
        m_x = self.m_x_dash * (0.5 * self.rho * (self.Lpp**2) * self.d)
        m_y = self.m_y_dash * (0.5 * self.rho * (self.Lpp**2) * self.d)
        J_z = self.J_z_dash * (0.5 * self.rho * (self.Lpp**4) * self.d)
        m = self.m
        I_zG = self.I_zG

        X = X_H + X_R + X_P + X_C
        Y = Y_H + Y_R + Y_C
        N_M = N_H + N_R + N_C

        # longitudinal acceleration
        d_u = (X + (m + m_y) * vm * r + self.x_G * m * (r**2)) / (m + m_x)

        # lateral acceleration
        f = I_zG + J_z + (self.x_G**2) * m

        d_vm_nom = Y - (m + m_x)*u*r - self.x_G * m * N_M/f + self.x_G**2 * m**2 * u * r/f
        d_vm_den = m + m_y - (self.x_G**2 * m**2)/f
        d_vm = d_vm_nom / d_vm_den

        # yaw rate acceleration
        d_r = (N_M - self.x_G * m * (d_vm + u * r)) / f

        return np.array([r*v_c + d_u, -r*u_c + d_vm, d_r])


    def _C_X(self, g_rc: float) -> float:
        """Polynomial fit of Budak & Beji (2020, Ocean Engineering)."""

        return (
        -0.0665*g_rc**5 + 
            0.5228*g_rc**4 - 
            1.4365*g_rc**3 + 
            1.6024*g_rc**2 - 
            0.2967*g_rc - 
            0.4691)


    def _C_Y(self, g_rc: float) -> float:
        """Polynomial fit of Budak & Beji (2020, Ocean Engineering)."""

        # return (
        #     0.1273*g_rc**4 - 
        #     0.8020*g_rc**3 + 
        #     1.3216*g_rc**2 - 
        #     0.1799*g_rc)
        return (
        0.05930686*g_rc**4 -
        0.37522028*g_rc**3 +
        0.46812233*g_rc**2 +
        0.39114522*g_rc -
        0.00273578
        )

    def _C_N(self, g_rc: float) -> float:
        """Polynomial fit of Budak & Beji (2020, Ocean Engineering)."""

        return (
        -0.0140*g_rc**5 + 
            0.1131*g_rc**4 -
            0.2757*g_rc**3 + 
            0.1617*g_rc**2 + 
            0.0728*g_rc)


    def _upd_dynamics(self, beta_c=0.0, V_c=0.0):
        """Updates positions and velocities for next simulation step. Uses the ballistic approach of Treiber, Kanagaraj (2015)."""

        # store current values
        eta_dot_old = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # euler update of velocities
        self.nu_dot = self._mmg_dynamics(nu        = self.nu, 
                                         psi       = self.eta[2],
                                         rud_angle = self.rud_angle,
                                         nps       = self.nps, 
                                         beta_c    = beta_c, 
                                         V_c       = V_c)
        self.nu += self.nu_dot * self.delta_t

        # find new eta_dot via rotation
        eta_dot_new = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # trapezoidal update of positions
        self.eta += 0.5 * (eta_dot_old + eta_dot_new) * self.delta_t

        # transform heading to [0, 2pi)
        self.eta[2] = angle_to_2pi(self.eta[2])


    def _control(self, a):
        """
        Action 'a' is an integer taking values in [0, 1, 2]. They correspond to:

        0 - keep rudder angle as is
        1 - increase rudder angle
        2 - decrease rudder angle
        """
        assert a in range(3), "Unknown action."

        # store action for rendering
        self.action = a
        
        # update angle
        if a == 0:
            pass
        elif a == 1:
            self.rud_angle += self.rud_angle_inc
        elif a == 2:
            self.rud_angle -= self.rud_angle_inc
        
        # clip it
        self.rud_angle = np.clip(self.rud_angle, -self.rud_angle_max, self.rud_angle_max)


    def _get_sideslip(self):
        """Returns the sideslip angle in radiant."""
        u, vm, _ = self.nu
        return math.atan2(-vm, u)


    def _get_course(self):
        """Returns the course angle in radiant, which is heading - sideslip for the MMG model."""
        return angle_to_2pi(self.eta[2] - self._get_sideslip())


    def _get_V(self):
        """Returns the aggregated velocity."""
        u, vm, _ = self.nu
        return math.sqrt(u**2 + vm**2)


    def _is_off_map(self):
        """Checks whether vessel left the map."""
        if self.eta[0] <= 0 or self.eta[0] >= self.N_max or self.eta[1] <= 0 or self.eta[1] >= self.E_max:
            return True
        return False


    def _get_u_from_nps(self, nps, psi=0.0, V_c=0.0, beta_c=0.0):
        """Returns the converged u-velocity for given revolutions per second if rudder angle is 0.0."""

        def to_find_root_of(u):
            nu = np.array([u, 0.0, 0.0])
            return self._mmg_dynamics(nu=nu, psi=psi, rud_angle=0.0, nps=nps, beta_c=beta_c, V_c=V_c)[0]

        return newton(func=to_find_root_of, x0=5.0)


    def _get_nps_from_u(self, u, psi=0.0, V_c=0.0, beta_c=0.0):
        """Returns the revolutions per second for a given u-velocity if rudder angle is 0.0.
        Note: Heading (psi) is irrelevant since we do not consider currents."""

        def to_find_root_of(nps):
            nu = np.array([u, 0.0, 0.0])
            return self._mmg_dynamics(nu=nu, psi=psi, rud_angle=0.0, nps=nps, beta_c=beta_c, V_c=V_c)[0]

        return newton(func=to_find_root_of, x0=2.0)
