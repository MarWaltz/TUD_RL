import math

import numpy as np
from scipy.optimize import newton
from tud_rl.envs._envs.VesselFnc import angle_to_2pi, bng_abs, dtr


class KVLCC2:
    """This class provides a KVLCC2 tanker behaving according to the MMG standard model of Yasukawa, Yoshimura (2015)."""

    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, nps, delta_t, N_max, E_max, full_ship=True, cont_acts=False) -> None:

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
                "rho_air":      1.269,          # Air density at 5°C
                "A_Fw":         1200,           # Frontal wind area [m²]
                "A_Lw":         3600,           # Lateral wind area [m²]
                "A_Fc":         977.184,        # Frontal current area [m²]
                "A_Lc":         5391.36,        # Lateral current area [m²]
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
                "rho_air":      1.269,          # Air density at 5°C
                "A_Fw":         240,            # Frontal wind area [m²]
                "A_Lw":         720,            # Lateral wind area [m²]
                "A_Fc":         39.08736,       # Frontal current area [m²]
                "A_Lc":         215.6544,       # Lateral current area [m²]
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

        # minimum water depth which can be operated on
        self.critical_depth = 1.2 * self.d

        # construct ship hull
        self.hull = Hull(Lpp=self.Lpp, B=self.B)

        # in [m]
        if cont_acts:
            self.ship_domain_A = 1 * self.Lpp + 0.5 * self.Lpp
            self.ship_domain_B = 1 * self.B + 0.5 * self.B   
            self.ship_domain_C = 1 * self.B + 0.5 * self.Lpp
            self.ship_domain_D = self.ship_domain_B
        else:
            self.ship_domain_A = 3 * self.Lpp + 0.5 * self.Lpp
            self.ship_domain_B = 1 * self.Lpp + 0.5 * self.B   
            self.ship_domain_C = 1 * self.Lpp + 0.5 * self.Lpp
            self.ship_domain_D = 3 * self.Lpp + 0.5 * self.B

        #---------------------------- Dynamic inits ----------------------------------------
        self.m_x = self.m_x_dash * (0.5 * self.rho * (self.Lpp**2) * self.d)
        self.m_y = self.m_y_dash * (0.5 * self.rho * (self.Lpp**2) * self.d)
        self.J_z = self.J_z_dash * (0.5 * self.rho * (self.Lpp**4) * self.d)
        
        self.M_RB = np.array([[self.m, 0.0, 0.0],
                              [0.0, self.m, self.m * self.x_G],
                              [0.0, self.m * self.x_G, self.I_zG]])
        self.M_A = np.array([[self.m_x, 0.0, 0.0],
                             [0.0, self.m_y, 0.0],
                             [0.0, 0.0, self.J_z + (self.x_G**2) * self.m]])
        self.M_inv = np.linalg.inv(self.M_RB + self.M_A)

        #------------------------- Motion Initialization -----------------------------------
        # Propeller revolutions [s⁻¹]
        self.nps = nps

        # actions
        self.cont_acts = cont_acts
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

    def _C_RB(self, r):
        return np.array([[0.0, -self.m * r, -self.m * self.x_G * r],
                         [self.m * r, 0.0, 0.0],
                         [self.m * self.x_G * r, 0.0, 0.0]])

    def _C_A(self, u, vm):
        return np.array([[0.0, 0.0, -self.m_y * vm],
                         [0.0, 0.0, self.m_x * u],
                         [0.0, 0.0, 0.0]])

    def _C_X_wind(self, g_w, cx=0.9):
        return -cx*math.cos(g_w)

    def _C_Y_wind(self, g_w, cy=0.95):
        return cy*math.sin(g_w)

    def _C_N_wind(self, g_w, cn=0.20):
        return cn*math.sin(2*g_w)

    def _mmg_dynamics(self, nu, rud_angle, nps, psi, V_w=0.0, beta_w=0.0, V_c=0.0, beta_c=0.0, H=None, 
                      beta_wave=None, eta_wave=None, T_0_wave=None, lambda_wave=None) -> np.ndarray:
        """System of ODEs after Yasukawa & Yoshimura (2015, Journal of Marine Science and Technology) for the MMG standard model. 
        Wind and currents follow Fossen (2021), short waves follow Taimuri et al. (2020).
        Wind, wave or current angle = 0.0 means that wind, waves or currents flow from N to E.
        Shallow water is considered by specifying the water depth. 

        Args:
            nu:          np.array of u, v, r
            rud_angle:   rudder angle in radiant (float)
            nps:         revolutions per seconds (float)
            psi:         heading in radiant (float)
            V_w:         wind speed in m/s (float)
            beta_w:      wind angle in radiant (float)
            V_c:         current speed in m/s (float)
            beta_c:      current angle in radiant (float)
            H:           water depth in meter (float)
            beta_wave:   incident wave angle in rad (float)
            eta_wave:    incident wave amplitude in m (float)
            T_0_wave:    modal period of waves in s (float)
            lambda_wave: wave length in m (float)

        Returns:
            np.array with nu_dot"""

        # shallow water
        if H is not None:
            assert H >= self.critical_depth, "Insufficient water depth!"

        X_vv_dash, X_vr_dash, X_rr_dash, X_vvvv_dash,\
             Y_v_dash, Y_r_dash, Y_vvv_dash, Y_vvr_dash, Y_vrr_dash, Y_rrr_dash,\
                 N_v_dash, N_r_dash, N_vvv_dash, N_vvr_dash, N_vrr_dash, N_rrr_dash, \
                    w_P0, t_P, gamma_R_minus, gamma_R_plus = self._shallow_water(H=H)

        # unpack values
        u, vm, r = nu

        if V_c != 0.0:
            # current velocities in ship's body frame
            u_c = V_c * math.cos(angle_to_2pi(beta_c - psi - math.pi))
            v_c = V_c * math.sin(angle_to_2pi(beta_c - psi - math.pi))

            # relative velocities
            u_r = u - u_c
            v_r = vm - v_c

            # calculate hydrodynamic forces with relative velocities
            u_pure = u
            vm_pure = vm
            u = u_r
            vm = v_r

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
            + X_vv_dash * (v_dash**2)
            + X_vr_dash * v_dash * r_dash
            + X_rr_dash * (r_dash**2)
            + X_vvvv_dash * (v_dash**4)
        )
        )

        Y_H = (0.5 * self.rho * self.Lpp * self.d * (U**2) * (
            Y_v_dash * v_dash
            + Y_r_dash * r_dash
            + Y_vvv_dash * (v_dash**3)
            + Y_vvr_dash * (v_dash**2) * r_dash
            + Y_vrr_dash * v_dash * (r_dash**2)
            + Y_rrr_dash * (r_dash**3)
        )
        )

        N_H = (0.5 * self.rho * (self.Lpp**2) * self.d * (U**2) * (
            N_v_dash * v_dash
            + N_r_dash * r_dash
            + N_vvv_dash * (v_dash**3)
            + N_vvr_dash * (v_dash**2) * r_dash
            + N_vrr_dash * v_dash * (r_dash**2)
            + N_rrr_dash * (r_dash**3)
        )
        )

        #---------------- longitudinal surge force due to propeller ----------------
        # redefine
        beta_P = beta - (self.x_P/self.Lpp) * r_dash

        if all([key in self.kvlcc2.keys() for key in ["C_1","C_2_plus","C_2_minus"]]):
            C_2 = self.C_2_plus if beta_P >= 0 else self.C_2_minus

            tmp = 1-math.exp(-self.C_1*abs(beta_P))*(C_2-1)
            w_P = 1-(1-w_P0)*(1+tmp)
        else:
            w_P = w_P0 * math.exp(-4.0 * (beta_P)**2)

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

        X_P = (1 - t_P) * self.rho * K_T * nps**2 * self.D_p**4

        #--------------------- hydrodynamic forces by steering ----------------------
        # effective inflow angle to rudder in maneuvering motions
        beta_R = beta - self.l_R * r_dash

        # flow straightening coefficient
        if self.gamma_R is not None:
            gamma_R = self.gamma_R
        else:
            if beta_R < 0.0:
                gamma_R = gamma_R_minus
            else:
                gamma_R = gamma_R_plus

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
        N_R = -(-0.5*self.Lpp + self.a_H * x_H) * F_N * math.cos(rud_angle)

        #-------------------------------- Wind ---------------------------------
        if V_w != 0.0:

            # relative velocity computations
            beta_w -= math.pi   # wind flows in the opposite direction
            u_w = V_w * math.cos(angle_to_2pi(beta_w - psi))
            v_w = V_w * math.sin(angle_to_2pi(beta_w - psi))

            if V_c != 0.0:
                u_rw = u_pure - u_w
                v_rw = vm_pure - v_w
            else:
                u_rw = u - u_w
                v_rw = vm - v_w

            V_rw_sq = u_rw**2 + v_rw**2
            g_rw = -math.atan2(v_rw, u_rw)

            # forces
            X_W = 0.5 * self.rho_air * V_rw_sq * self._C_X_wind(g_rw) * self.A_Fw
            Y_W = 0.5 * self.rho_air * V_rw_sq * self._C_Y_wind(g_rw) * self.A_Lw
            N_W = 0.5 * self.rho_air * V_rw_sq * self._C_N_wind(g_rw) * self.A_Lw * self.Lpp

        else:
            X_W, Y_W, N_W = 0.0, 0.0, 0.0

        #-------------------------------- Short waves ---------------------------------
        if all([ele is not None for ele in (beta_wave, eta_wave, T_0_wave, lambda_wave)]):
            X_SW, Y_SW, N_SW = self.hull.get_wave_XYN(U=U, psi=psi, T=self.d, beta_wave=beta_wave, eta_wave=eta_wave, 
                                                      T_0_wave=T_0_wave, lambda_wave=lambda_wave, rho=self.rho)
        else:
            X_SW, Y_SW, N_SW = 0.0, 0.0, 0.0
        
        #------------------------------ Equation solving ------------------------------
        X = X_H + X_R + X_P + X_W + X_SW
        Y = Y_H + Y_R + Y_W + Y_SW
        N_M = N_H + N_R + N_W + N_SW

        F = np.array([X, Y, N_M])

        if V_c != 0.0:
            nu_c_dot = np.array([v_c*r, -u_c*r, 0.0])
            return np.dot(self.M_inv, F - np.dot(self._C_RB(r), np.array([u_pure, vm_pure, r]))     
                                        - np.dot(self._C_A(u_r, v_r), np.array([u_r, v_r, r])  
                                        + np.dot(self.M_A, nu_c_dot)))
        else:
            return np.dot(self.M_inv, F - np.dot((self._C_RB(r) + self._C_A(u, vm)), np.array([u, vm, r])))

    def _upd_dynamics(self, V_w=0.0, beta_w=0.0, V_c=0.0, beta_c=0.0, H=None, beta_wave=None, eta_wave=None, T_0_wave=None, lambda_wave=None):
        """Updates positions and velocities for next simulation step. Uses the ballistic approach of Treiber, Kanagaraj (2015)."""

        # store current values
        eta_dot_old = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # euler update of velocities
        self.nu_dot = self._mmg_dynamics(nu          = self.nu,
                                         rud_angle   = self.rud_angle,
                                         nps         = self.nps,
                                         psi         = self.eta[2],
                                         V_w         = V_w,
                                         beta_w      = beta_w,
                                         V_c         = V_c,
                                         beta_c      = beta_c,
                                         H           = H,
                                         beta_wave   = beta_wave,
                                         eta_wave    = eta_wave,
                                         T_0_wave    = T_0_wave,
                                         lambda_wave = lambda_wave)
        self.nu += self.nu_dot * self.delta_t

        # find new eta_dot via rotation
        eta_dot_new = np.dot(self._T_of_psi(self.eta[2]), self.nu)

        # trapezoidal update of positions
        self.eta += 0.5 * (eta_dot_old + eta_dot_new) * self.delta_t

        # transform heading to [0, 2pi)
        self.eta[2] = angle_to_2pi(self.eta[2])

    def _shallow_water(self, H=None):
        """Updates hydrodynamic derivatives and further coefficients depending on the water depth. 
        Follows Taimuri et al. (2020, Ocean Engineering)."""

        if H is None:
            return (
                self.X_vv_dash, self.X_vr_dash, self.X_rr_dash, self.X_vvvv_dash,\
                self.Y_v_dash, self.Y_r_dash, self.Y_vvv_dash, self.Y_vvr_dash, self.Y_vrr_dash, self.Y_rrr_dash,\
                self.N_v_dash, self.N_r_dash, self.N_vvv_dash, self.N_vvr_dash, self.N_vrr_dash, self.N_rrr_dash,\
                self.w_P0, self.t_P, self.gamma_R_minus, self.gamma_R_plus
            )
        else:
            # preparation
            frac = lambda x,y: x/y
            L, B, T, Cb = self.Lpp, self.B, self.d, self.C_b

            HT = H/T -1
            TH = T/H
            K0 = 1+frac(0.0775,HT**2)-frac(0.011,HT**3)+frac(0.000068,HT**5)
            K1 = -frac(0.0643,HT)+frac(0.0724,HT**2)-frac(0.0113,HT**3)+frac(0.0000767,HT**5)
            K2 = frac(0.0342,HT) if B/T <= 4 else frac(0.137*B,HT*T)
            B1 = Cb*B*(1+frac(B,L))**2

            A1Yr = -5.5*(frac(Cb*B,T))**2+26*frac(Cb*B,T)-31.5
            A2Yr = 37*frac(Cb*B,T)**2-185*frac(Cb*B,T)+230
            A3Yr = -38*frac(Cb*B,T)**2+197*frac(Cb*B,T)-250

            A1Nvvr = 91*Cb*frac(T,B)-25
            A2Nvvr = -515*Cb*frac(T,B)+144
            A3Nvvr = 508*Cb*frac(T,B)-143

            A1Nvrr = 40*Cb*frac(B,T)-88
            A2Nvrr = -295*Cb*frac(B,T)+645
            A3Nvrr = 312*Cb*frac(B,T)-678

            gnr = K0+frac(8,15)*K1*frac(B1,T)+frac(40,105)*K2*(frac(B1,T))**2
            fyr = K0+frac(2,5)*K1*frac(B1,T)+frac(24,105)*K2*(frac(B1,T))**2
            fnr = K0+frac(1,2)*K1*frac(B1,T)+frac(1,3)*K2*(frac(B1,T))**2
            fyv = 1.5*fnr-0.5
            fnv = K0+K1*frac(B1,T)+K2*frac(B1,T)**2

            # corrections for hydrodynamic derivatives
            X_vv_dash = fyv * self.X_vv_dash
            X_vr_dash = fyr * self.X_vr_dash
            X_rr_dash = fnr * self.X_rr_dash
            X_vvvv_dash = fyv * self.X_vvvv_dash

            Y_v_dash = (-TH+frac(1,(1-TH)**(frac(0.4*Cb*B,T)))) * self.Y_v_dash
            Y_r_dash = (1+A1Yr*TH+A2Yr*TH**2+A3Yr*TH**3) * self.Y_r_dash
            Y_vvv_dash = fyv * self.Y_vvv_dash
            Y_vvr_dash = fyv * self.Y_vvr_dash
            Y_vrr_dash = fyv * self.Y_vrr_dash
            Y_rrr_dash = gnr * self.Y_rrr_dash

            N_v_dash = fnv * self.N_v_dash
            N_r_dash = (-TH+frac(1,(1-TH)**(frac(-14.28*T,L)+1.5))) * self.N_r_dash
            N_vvv_dash = fyv * self.N_vvv_dash
            N_vvr_dash = (1+A1Nvvr*TH+A2Nvvr*TH**2+A3Nvvr*TH**3) * self.N_vvr_dash
            N_vrr_dash = (1+A1Nvrr*TH+A2Nvrr*TH**2+A3Nvrr*TH**3) * self.N_vrr_dash
            N_rrr_dash = gnr * self.N_rrr_dash

            # corrections for wake fraction, thrust deduction, and flow-straightening coefficients
            w_P0 = (1+(-4.932+0.6425*frac(Cb*L,T)-0.0165*(frac(Cb*L,T)**2))*TH**1.655) * self.w_P0
            ctp = 1+((29.495-14.089*frac(Cb*L,B)+1.6486*frac(Cb*L,B)**2)*(frac(1,250)-frac(7*TH,200)-frac(13*TH**2,125)))
            t_P = 1-ctp*(1-self.t_P)

            cgr1 = 1+((frac(-5129,500)+178.207*frac(Cb*B,L)-frac(2745,4)*frac(Cb*B,L)**2)*(frac(-1927,500)+frac(2733*TH,200)-frac(2617*TH**2,250)))
            cgr2 = 1+(frac(-541,4)+2432.95*frac(Cb*B,L)-10137.7*frac(Cb*B,L)**2)*TH**4.81
            if TH <= (-0.332*frac(T,B)+0.581):
                gamma_R_minus = cgr2 * self.gamma_R_minus 
                gamma_R_plus = cgr2 * self.gamma_R_plus
            else:
                gamma_R_minus = cgr1 * self.gamma_R_minus
                gamma_R_plus = cgr1 * self.gamma_R_plus

            return (
                X_vv_dash, X_vr_dash, X_rr_dash, X_vvvv_dash,\
                Y_v_dash, Y_r_dash, Y_vvv_dash, Y_vvr_dash, Y_vrr_dash, Y_rrr_dash,\
                N_v_dash, N_r_dash, N_vvv_dash, N_vvr_dash, N_vrr_dash, N_rrr_dash,\
                w_P0, t_P, gamma_R_minus, gamma_R_plus
                )

    def _control(self, a):
        """
        Action 'a' is an integer taking values in [0, 1, 2] for the discrete case. They correspond to:

        0 - keep rudder angle as is
        1 - increase rudder angle
        2 - decrease rudder angle

        In the continuous case, a is a float in [-1,1].
        """
        # store action for rendering
        self.action = a

        # continuous
        if self.cont_acts:
            raise NotImplementedError("Continuous action updating inside the KVLCC2-object is deprecated.")

        # discrete
        else:
            assert a in range(3), "Unknown action."

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

    def _get_u_from_nps(self, nps, V_w=0.0, beta_w=0.0, V_c=0.0, beta_c=0.0, H=None, psi=0.0, 
                        beta_wave=None, eta_wave=None, T_0_wave=None, lambda_wave=None):
        """Returns the converged u-velocity for given revolutions per second if rudder angle is 0.0."""

        def to_find_root_of(u):
            nu = np.array([u, 0.0, 0.0])
            return self._mmg_dynamics(nu=nu, rud_angle=0.0, nps=nps, V_w=V_w, beta_w=beta_w, V_c=V_c, beta_c=beta_c, H=H, psi=psi,
                                      beta_wave=beta_wave, eta_wave=eta_wave, T_0_wave=T_0_wave, lambda_wave=lambda_wave)[0]

        return newton(func=to_find_root_of, x0=5.0, maxiter=10_000)

    def _get_nps_from_u(self, u, V_w=0.0, beta_w=0.0, V_c=0.0, beta_c=0.0, H=None, psi=0.0,
                        beta_wave=None, eta_wave=None, T_0_wave=None, lambda_wave=None):
        """Returns the revolutions per second for a given u-velocity if rudder angle is 0.0."""

        def to_find_root_of(nps):
            nu = np.array([u, 0.0, 0.0])
            return self._mmg_dynamics(nu=nu, rud_angle=0.0, nps=nps, V_w=V_w, beta_w=beta_w, V_c=V_c, beta_c=beta_c, H=H, psi=psi,
                                      beta_wave=beta_wave, eta_wave=eta_wave, T_0_wave=T_0_wave, lambda_wave=lambda_wave)[0]

        return newton(func=to_find_root_of, x0=2.0, maxiter=10_000)


class Hull:
    def __init__(self, Lpp, B, N=1000, h_xs=None, h_ys=None) -> None:
        """Computes x-y-coordinates (x ist Ordinate, y ist Abszisse) based on an elliptical approximation of a ship hull.

        Args:
            Lpp (float): length of ship
            B (float): width of ship
            N (float): number of integration intervals / points in hull approximation
            h_xs (np.array, optional): x points
            h_ys (np.array, optional): y points

        Generates attributes:
            h_xs (np.array): x points
            h_ys (np.array): y points
            N_Xs (np.array): x components of normal vector
            N_Ys (np.array): y components of normal vector
            thetas (np.array): angles between hull and center line
            dls (np.array): length of normal vectors
        """
        self.Lpp = Lpp
        self.B = B
        self.N = N
        self.h_xs = h_xs
        self.h_ys = h_ys
        self._construct_hull()

    def _construct_hull(self):
        if self.h_xs is not None and self.h_ys is not None:
            assert len(self.h_xs) == len(self.h_ys), "Incorrect ship hull specification."
            self.N = len(self.h_xs)
        else:
            a = self.Lpp*0.5
            b = self.B*0.5
            degs = np.linspace(0, 2*math.pi, self.N, endpoint=False)

            lengths = np.power(np.power(np.cos(degs),2) / a + np.power(np.sin(degs),2) / b, -0.5)
            self.h_xs = lengths * np.cos(degs)
            self.h_ys = lengths * np.sin(degs)

        # reverse axis like in Faltinsen et al. (1980)
        self.h_xs *= -1.0

        # integration components
        self.N_Xs = np.zeros(self.N)
        self.N_Ys = np.zeros(self.N)
        self.x0s = np.zeros(self.N)
        self.y0s = np.zeros(self.N)
        self.thetas = np.zeros(self.N)

        for n in range(self.N):
            y1 = self.h_ys[n]
            x1 = self.h_xs[n]

            if n == self.N-1:
                y2 = self.h_ys[0]
                x2 = self.h_xs[0]
            else:
                y2 = self.h_ys[n+1]
                x2 = self.h_xs[n+1]

            self.x0s[n] = 0.5 * (x1 + x2)
            self.y0s[n] = 0.5 * (y1 + y2)

            # rectangle cases
            if x1 == x2:
                self.N_Ys[n] = 0.0
                self.N_Xs[n] = y1 - y2
            
            elif y1 == y2:
                self.N_Xs[n] = 0.0
                self.N_Ys[n] = x2 - x1

            # ellipse/corner cases
            else:
                self.N_Ys[n] = math.sqrt(((x2-x1)**2 + (y2-y1)**2) / (1 + (y2-y1)**2/(x2-x1)**2))
                self.N_Xs[n] = -(y2-y1)/(x2-x1)*self.N_Ys[n]

                if x2 - x1 < 0.0:
                    self.N_Ys[n] *= -1.0
                    self.N_Xs[n] *= -1.0

            self.thetas[n] = bng_abs(N0=x2, E0=y2, N1=x1, E1=y1)
            if self.thetas[n] <= math.pi:
                self.thetas[n] += math.pi
            else:
                self.thetas[n] -= math.pi

        self.dls = np.sqrt(self.N_Xs**2 + self.N_Ys**2)

        # some useful definitions
        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

    def get_wave_XYN(self, U, psi, T, beta_wave, eta_wave, T_0_wave, lambda_wave, rho):
        """Computes the short wave induced surge, sway forces and the yaw moment. 
        Based on numerical integration outlined in Sakamoto and Baba (1986) and Taimuri et al. (2020).

        Args:
            U (float): speed of ship in m/s
            psi (float): heading of ship in rad (usual N is 0° definition)
            T (float): ship draft in m
            beta_wave (float): incident wave angle in rad (usual N is 0° definition, e.g., 270° means waves flow from W to E)
            eta_wave (float): incident wave amplitude in m
            T_0_wave (float): modal period of waves in s
            lambda_wave (float) : wave length in m
            rho (float): water density in kg/m³

        Returns:
            surge force, sway force, yaw moment (all floats)
        """
        # parameters
        g = 9.80665                      # gravity in m/s²
        omega_0 = 2*math.pi / T_0_wave   # model frequency in rad/s
        k = 2*math.pi/ lambda_wave       # wave number in rad/m

        # wave angle from vessel perspective
        beta_w = angle_to_2pi(beta_wave - psi)

        # detect non-shadow region
        N_WX = np.cos(beta_w)
        N_WY = np.sin(beta_w)
        alphas = np.arccos(self.N_Xs*N_WX/self.dls + self.N_Ys*N_WY/self.dls)
        non_shadow = (alphas <= math.pi/2)

        # compute integration block components
        inner_terms = np.sin(self.thetas + beta_w)**2 + 2*omega_0*U/g * (np.cos(beta_w) - self.cos_thetas*np.cos(self.thetas + beta_w))
        inner_terms *= non_shadow * self.dls

        # integrate
        X_int = np.sum(inner_terms * self.sin_thetas)
        Y_int = np.sum(inner_terms * self.cos_thetas)
        M_int = np.sum(inner_terms * (self.x0s * self.cos_thetas - self.y0s * self.sin_thetas))

        # pre-factors
        norm = 0.5 * rho * g * eta_wave**2
        FX_SW = norm * X_int
        FY_SW = norm * Y_int
        MN_SW = norm * M_int

        # draft correction
        corr_d = 1 - math.exp(-2 * k * T)
        FX_SW *= corr_d
        FY_SW *= corr_d
        MN_SW *= corr_d
        
        return FX_SW, FY_SW, MN_SW
