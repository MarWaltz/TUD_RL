import math

import matplotlib.pyplot as plt
import numpy as np
from mycolorpy import colorlist as mcp

COLREG_NAMES  = {0 : "Null", 
                 1 : "Head-on", 
                 2 : "Starb. cross.", 
                 3 : "Ports. cross.", 
                 4 : "Overtaking"}
COLREG_COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(8)] + mcp.gen_color(cmap="tab20b", n=20) 


#------------------- Helper functions with angles, mainly following Benjamin (2017) ----------------
def dtr(angle):
    """Takes angle in degree and transforms it to radiant."""
    return angle * math.pi / 180

def rtd(angle):
    """Takes angle in radiant and transforms it to degree."""
    return angle * 180 / math.pi

def angle_to_2pi(angle):
    """Transforms an angle to [0, 2pi)."""
    if angle >= 0:
        return angle - math.floor(angle / (2*math.pi)) * 2*math.pi
    else:
        return angle + (math.floor(-angle / (2*math.pi)) + 1) * 2*math.pi

def angle_to_pi(angle):
    """Transforms an angle to [-pi, pi)."""
    if angle >= 0:
        return angle - math.floor((angle + math.pi) / (2*math.pi)) * 2*math.pi
    else:
        return angle + math.floor((-angle + math.pi)  / (2*math.pi)) * 2*math.pi

def head_inter(head_OS, head_TS, to_2pi=True):
    """Computes the intersection angle between headings in radiant (in [0, 2pi) if to_2pi, else [-pi, pi)).
    Corresponds to C_T in Xu et al. (2022, Neurocomputing)."""
    if to_2pi:
        return angle_to_2pi(head_TS - head_OS)
    else:
        return angle_to_pi(head_TS - head_OS)

def ED(N0, E0, N1, E1, sqrt=True):
    """Computes the euclidean distance between two points."""
    d_sq = (N0 - N1)**2 + (E0 - E1)**2

    if sqrt:
        return np.sqrt(d_sq)
    return d_sq

def polar_from_xy(x, y, with_r=True, with_angle=True):
    """Get polar coordinates (r, angle in rad in [0, 2pi)) from x,y-coordinates. Angles are defined clockwise with zero at the y-axis.
    Args:
        with_r (bool):     Whether to compute the radius.
        with_angle (bool): Whether to compute the angle.
    Returns:
        r, angle as a tuple of floats."""

    r = math.sqrt(x**2 + y**2) if with_r else None
    angle = angle_to_2pi(math.atan2(x, y)) if with_angle else None
    return r, angle

def xy_from_polar(r, angle):
    """Get x,y-coordinates from polar system, where angle is defined clockwise with zero at the y-axis."""
    return r * np.sin(angle), r * np.cos(angle)

def bng_abs(N0, E0, N1, E1):
    """Computes the absolute bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0)."""
    return polar_from_xy(x=E1-E0, y=N1-N0, with_r=False, with_angle=True)[1]

def bng_rel(N0, E0, N1, E1, head0, to_2pi=True):
    """Computes the relative bearing (in radiant in [0, 2pi) if to_2pi, else [-pi, pi)) of 
    (N1, E1) from perspective of (N0, E0) and heading head0."""
    if to_2pi:
        return angle_to_2pi(bng_abs(N0, E0, N1, E1) - head0)
    else:
        return angle_to_pi(bng_abs(N0, E0, N1, E1) - head0)

def tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes the time to closest point of approach (TCPA). Follows Lenart (1983)."""

    # easy access
    xOS = EOS
    yOS = NOS
    xTS = ETS
    yTS = NTS

    # compute velocities in x,y-coordinates
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)

    # relative velocity
    vrx = vxTS - vxOS
    vry = vyTS - vyOS

    # tcpa
    nom = (xTS - xOS)*vrx + (yTS - yOS)*vry
    den = vrx**2 + vry**2

    if den == 0:
        return 0.0
    return - nom / den

def cpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS, get_positions=False):
    """Returns DCPA and TCPA. Follows Chun et al. (2021, OE).
    If get_positions, returns DCPA, TCPA, and NOS, EOS, NTS, ETS when TCPA = 0."""

    # easy access
    xOS = EOS
    yOS = NOS
    xTS = ETS
    yTS = NTS

    # compute velocities in x,y-coordinates
    vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
    vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)

    # get TCPA
    TCPA = tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS)

    # forecast OS
    xOS_tcpa = xOS + TCPA * vxOS
    yOS_tcpa = yOS + TCPA * vyOS

    # forecast TS
    xTS_tcpa = xTS + TCPA * vxTS
    yTS_tcpa = yTS + TCPA * vyTS

    if get_positions:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA, yOS_tcpa, xOS_tcpa, yTS_tcpa, xTS_tcpa
    else:
        return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa), TCPA

def project_vector(VA, angleA, VB, angleB):
    """Projects vector A, characterized in polar coordinates VA and angleA, onto vector B (also polar coordinates).
    Angles are defined clockwise with zero at the y-axis.

    Args:
        VA (float):     norm of vector A
        angleA (float): angle of vector A (in rad)
        VB (float):     norm of vector B
        angleB (float): angle of vector B (in rad)

    Returns:
        Velocity components in x- and y-direction, respectively, not polar coordinates. Both floats.
    """

    # x,y components of vectors A and B
    vxA, vyA = xy_from_polar(r=VA, angle=angleA)
    vxB, vyB = xy_from_polar(r=VB, angle=angleB)

    # projection of A on B
    v_proj = (vxA * vxB + vyA * vyB) / VB

    # x,y components of projection
    return xy_from_polar(r=v_proj, angle=angleB)

def get_ship_domain(A, B, C, D, OS, TS, ang=None):
    """Computes a ship domain for the OS with respect to TS following Chun et al. (2021, Ocean Engineering).
    Args:
        A/B/C/D: int, domain lengths
        OS: KVLCC2
        TS: KVLCC2"""

    # relative bearing
    if ang is None:
        ang = bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], head0=OS.eta[2])

    # ellipsis
    if 0 <= rtd(ang) < 90:
        a = D
        b = A
    elif 90 <= rtd(ang) < 180:
        ang = dtr(180) - ang
        a = D
        b = C
    elif 180 <= rtd(ang) < 270:
        ang = ang - dtr(180)
        a = B
        b = C
    else:
        ang = dtr(360) - ang
        a = B
        b = A
    return ((math.sin(ang) / a)**2 + (math.cos(ang) / b)**2)**(-0.5)

def NM_to_meter(NM):
    """Convertes nautical miles in meter."""
    return NM * 1852

def meter_to_NM(meter):
    """Convertes meter in nautical miles."""
    return meter / 1852

def mps_to_knots(mps):
    return mps * 1.9438452

def knots_to_mps(knots):
    return knots / 1.9438452

# -------------- APF method following Lyu & Yin (2019) ----------------
def get_F_att(N0, E0, N_goal, E_goal, eps):
    """Computes the attractive force of an APF method.
    
    Returns: 
        F_east, F_north
    """
    return eps * (E_goal - E0), eps * (N_goal - N0)

def get_theta(p_OS, v_OS, p_TS, v_TS):
    """Computes the angle between relative positions and relative speeds."""
    # catch edge cases
    if (v_OS == v_TS).all() or (p_OS == p_TS).all():
        return 0.0

    # relative positions and speeds
    p_ot = p_TS - p_OS
    v_to = v_OS - v_TS

    # compute angle
    nom = np.dot(p_ot, v_to)
    den = norm(p_ot) * norm(v_to)
    return np.arccos(np.clip(nom / den, -1.0, 1.0))

def get_theta_m(p_OS, p_TS, dm):
    """Computes the angle of the tangent to the safety circle around the target ship."""
    return np.arctan(dm / np.sqrt(norm(p_OS-p_TS)**2 - dm**2))

def F_rd1_abs(eta_d, R_TS, dg, d, dm, rho_O, theta_m, theta, v_to_norm):
    A = 1/(d-dm) - 1/rho_O
    frac = dm/(d*np.sqrt(d**2 -dm**2))
    C = frac + np.sin(theta_m)/v_to_norm
    F_to0 = A * C

    B = frac + np.sin(theta)/v_to_norm
    return - eta_d * R_TS * dg**2 * (A * np.exp(theta_m-theta) * B + ((np.exp(theta_m-theta)-1) / (d-dm)**2) - F_to0)

def F_rd2_abs(eta_d, R_TS, dg, d, dm, rho_O, theta_m, theta, p_ot_norm, v_to_norm, v_ot_perp_norm):
    A = 1/(d-dm) - 1/rho_O
    C = 1/p_ot_norm + np.cos(theta_m)/v_to_norm
    F_to_perp = A * C

    B = 1/p_ot_norm + np.cos(theta)/v_to_norm
    return eta_d * R_TS * dg**2 * (A * np.exp(theta_m-theta) * B + (v_ot_perp_norm * (np.exp(theta_m-theta)-1) / (d*(d-dm)**2)) - F_to_perp)

def F_rd3_abs(eta_d, R_TS, dg, d, dm, rho_O, theta_m, theta):
    return eta_d * R_TS * dg * (1/(d-dm) - 1/rho_O) * (np.exp(theta_m-theta)-1)

def F_re1_abs(eta_e, R_TS, d, dg, tau, dm):
    return -2 * eta_e * R_TS * (1/(d-tau) - 1/dm) * dg**2 / (d-tau)**2

def F_re2_abs(eta_e, R_TS, d, dg, v_to_norm, theta):
    return 2 * eta_e * R_TS * dg/d * v_to_norm**2 * np.cos(theta) * np.sin(theta)

def F_re3_abs(eta_e, R_TS, d, dg, tau, dm, v_to_norm, theta):
    return 2 * eta_e * R_TS * dg * ((1/(d-tau) - 1/dm)**2 + v_to_norm**2 * np.cos(theta)**2)

def norm(vec):
    return np.sqrt(np.sum(np.square(vec)))

def apf(N0:float, E0:float, head0, vN0:float, vE0:float,   # OS info
        N_goal:float, E_goal:float,                        # goal info
        N1:list, E1:list, vN1:list, vE1:list,              # TS info
        eps       = 3000, 
        tau       = 0.3, 
        d_safe    = 0.5,
        R_OS      = 0.5,
        R_TS      = 0.3, 
        rho_O     = 5.0, 
        eta_d     = 2000, 
        eta_e     = 5000, 
        dhead_max = dtr(2.5)):
    """Computes the heading change based on the artificial potential field method outlined in Lyu & Yin (2019).
    
    Returns:
        dhead in radiant"""

    # normalize
    N0 = meter_to_NM(N0)
    E0 = meter_to_NM(E0)
    E1 = [meter_to_NM(e) for e in E1]
    N1 = [meter_to_NM(n) for n in N1]
    N_goal = meter_to_NM(N_goal)
    E_goal = meter_to_NM(E_goal)
    vN0 = mps_to_knots(vN0)
    vE0 = mps_to_knots(vE0)
    vN1 = [mps_to_knots(v) for v in vN1]
    vE1 = [mps_to_knots(v) for v in vE1]

    # compute vectorized
    p_OS = np.array([E0, N0])
    v_OS = np.array([vE0, vN0])
    p_G  = np.array([E_goal, N_goal])

    # goal distance and direction
    dg = norm(p_G-p_OS)    
    n_og = (p_G-p_OS)/dg

    # attractive force
    F = eps * dg * n_og
    F_att_E = F[0]
    F_att_N = F[1]

    # repulsive forces
    F_rep_E = 0.0
    F_rep_N = 0.0

    # some metrics
    dm = R_OS + d_safe + R_TS
    CR = dm + rho_O

    for i in range(len(N1)):
        # quick access
        p_TS = np.array([E1[i], N1[i]])
        v_TS = np.array([vE1[i], vN1[i]])

        # avoid egde case
        if (v_TS == v_OS).all():
            v_TS += np.array([1e-10, 1e-10])

        # compute some metrics
        d = norm(p_TS-p_OS)
        theta = get_theta(p_OS, v_OS, p_TS, v_TS)
        v_to_norm = norm(v_OS-v_TS)
        p_ot_norm = norm(p_TS-p_OS)
        v_ot_perp_norm = v_to_norm

        # unit vector from OS to TS (ship to obstacle)
        n_ot = (p_TS-p_OS)/d

        # perpendicular unit vector to starboard side of n_ot for collision avoidance
        # 90 degree clockwise rotation
        n_ot_perp = np.array([n_ot[1], -n_ot[0]])

        # repulsive force not defined here
        if d <= tau:
            pass

        # emergency case
        elif tau < d <= dm:

            # force from target ship to own ship
            F_re1 = F_re1_abs(eta_e=eta_e, R_TS=R_TS, d=d, dg=dg, tau=tau, dm=dm)
            E_add, N_add = F_re1 * n_ot
            F_rep_E += E_add
            F_rep_N += N_add

            # force to steer starboard or portside depending on the situation
            sign = np.sign(np.dot(n_ot_perp, v_OS-v_TS))
            F_re2 = F_re2_abs(eta_e=eta_e, R_TS=R_TS, d=d, dg=dg, v_to_norm=v_to_norm, theta=theta)
            E_add, N_add = sign * F_re2 * n_ot_perp
            F_rep_E += E_add
            F_rep_N += N_add

            # force from own ship to goal
            F_re3 = F_re3_abs(eta_e=eta_e, R_TS=R_TS, d=d, dg=dg, tau=tau, dm=dm, v_to_norm=v_to_norm, theta=theta)
            E_add, N_add = F_re3 * n_og
            F_rep_E += E_add
            F_rep_N += N_add

        else:
            theta_m = get_theta_m(p_OS, p_TS, dm=dm)

            if (dm < d <= CR) and (theta < theta_m):

                # force from target ship to own ship
                F_rd1 = F_rd1_abs(eta_d=eta_d, R_TS=R_TS, dg=dg, d=d, dm=dm, rho_O=rho_O, 
                                  theta_m=theta_m, theta=theta, v_to_norm=v_to_norm)
                E_add, N_add = F_rd1 * n_ot
                F_rep_E += E_add
                F_rep_N += N_add

                # force to steer starboard
                F_rd2 = F_rd2_abs(eta_d=eta_d, R_TS=R_TS, dg=dg, d=d, dm=dm, rho_O=rho_O, 
                                  theta_m=theta_m, theta=theta, p_ot_norm=p_ot_norm, v_to_norm=v_to_norm,
                                  v_ot_perp_norm=v_ot_perp_norm)
                E_add, N_add = F_rd2 * n_ot_perp
                F_rep_E += E_add
                F_rep_N += N_add
                
                # force from own ship to goal
                F_rd3 = F_rd3_abs(eta_d=eta_d, R_TS=R_TS, dg=dg, d=d, dm=dm, rho_O=rho_O,
                                 theta_m=theta_m, theta=theta)
                E_add, N_add = F_rd3 * n_og
                F_rep_E += E_add
                F_rep_N += N_add

    # aggregate forces
    F_E = F_att_E + F_rep_E
    F_N = F_att_N + F_rep_N

    # compute and clip heading change
    dh = math.atan2(F_E, F_N) - head0

    if dh < -math.pi:
        dh += 2*math.pi
    elif dh > math.pi:
        dh -= 2*math.pi

    return np.clip(dh, -dhead_max, dhead_max)

def r_safe_dyn(a:float, r_min:float):
    """Computes safety radius in the APF method.
    Args:
        a(float): relative bearing in [-pi, pi)
        r_min(float): minimum safety radius    
    """
    assert -math.pi <= a < math.pi, "The angle for computing the safety distance should be in [-pi, pi)."
    if abs(a) <= math.pi/4:
        f = 2 + math.cos(4*a)
    else:
        f = 1
    return r_min * f

def k_r_TS_dyn(DCPA:float, TCPA:float, a1:float=math.log(0.1)/3704, a2:float=1.5):
    """Computes factor for dynamic repulsive forces of moving obstacles.
    Args:
        DCPA(float): DCPA between OS and obstacle
        TCPA(float): TCPA between OS and obstacle
        a1(float): weighting constant
        a2(float): weighting constant"""
    if TCPA < 0.0:
        return 0.0
    else:
        return math.exp(a1*(DCPA + a2*TCPA))

def apf_DZN(N0:float, E0:float, head0:float, v0:float, chi0:float, 
            N1:list, E1:list, v1:list, chi1:list, 
            N_goal:float, E_goal:float, 
            dh_clip: float = None,
            r_min=250, 
            k_a=100, 
            k_r_TS=2.5e6):
    """Computes a local path based on the artificial potential field method, see Du, Zhang, and Nie (2019, IEEE).
    Args:
        OS(KVLCC2):     own ship containing positional and speed information
        TSs(list):      contains target ships
        G(dict):        contains keys "x" and "y", marking the position of the goal
        dh_clip(float): maximum heading change
        du_clip(float): maximum surge velocity change
        river_n(np.ndarray): north coordinates of river, considered static obstacles
        river_e(np.ndarray): east coordinates of river, considered static obstacles
        r_min(float):  safety distance for repulsive forces  

    Returns: 
        float: Δheading."""
    
    # quick access
    x_OS = E0
    y_OS = N0
    x_G = E_goal
    y_G = N_goal

    # attractive forces
    F_x = k_a * (x_G - x_OS)
    F_y = k_a * (y_G - y_OS)

    # repulsive forces due to vessel positions
    for i in range(len(N1)):

        # distance
        x_TS = E1[i]
        y_TS = N1[i]
        d = ED(N0=y_OS, E0=x_OS, N1=y_TS, E1=x_TS)

        # bearing-dependent safety radius
        r_safe = r_safe_dyn(a = bng_rel(N0=N0, E0=E0, N1=N1[i], E1=E1[i], head0=head0, to_2pi=False), 
                            r_min = r_min)
        if d <= r_safe:

            # compute CPA-measure adjustment
            DCPA, TCPA = cpa(NOS=N0, EOS=E0, NTS=N1[i], ETS=E1[i], chiOS=chi0,
                             chiTS=chi1[i], VOS=v0, VTS=v1[i])
            f = k_r_TS_dyn(DCPA=DCPA, TCPA=TCPA)

            F_x += k_r_TS * (1 + f) * (1/r_safe - 1/d) * (x_TS - x_OS) / d
            F_y += k_r_TS * (1 + f) * (1/r_safe - 1/d) * (y_TS - y_OS) / d

    # translate into Δheading
    dh = angle_to_pi(math.atan2(F_x, F_y) - head0)
    return np.clip(dh, -dh_clip, dh_clip)

def cte(N1, E1, N2, E2, NA, EA, pi_path=None):
    """Computes the cross-track error following p.350 of Fossen (2021). The waypoints are (N1, E1) and (N2, E2), while
    the agent is at (NA, EA). The angle of the path relative to the NED-frame can optionally be specified."""
    if pi_path is None:
        pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)
    return -np.sin(pi_path)*(NA - N1) + np.cos(pi_path)*(EA - E1)

def ate(N1, E1, N2, E2, NA, EA, pi_path=None):
    """Computes the along-track error following p.350 of Fossen (2021). The waypoints are (N1, E1) and (N2, E2), while
    the agent is at (NA, EA). The angle of the path relative to the NED-frame can optionally be specified."""
    if pi_path is None:
        pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)
    return np.cos(pi_path)*(NA - N1) + np.sin(pi_path)*(EA - E1)

def apf_2023(N0:float, E0:float, head0:float, vN0:float, vE0:float,
              N1:list, E1:list, vN1:list, vE1:list,
              N_goal:float, E_goal:float, N_start:float, E_start:float,
              dh_clip = dtr(3.0),
              d_star  = NM_to_meter(5.0),
              d_l     = NM_to_meter(3.0),
              k_a1    = 1e-3, 
              k_a2    = 0.1,
              k_m     = 1.5,
              t_s     = 300,
              R_UO    = NM_to_meter(3.0),
              k_r1    = 1e4,
              k_r2    = 1e2):
    """Computes an APF method based on Liu et al. (2023, Physical Communication).
    Returns: 
        float: Δheading."""
    # normalize
    #C = 1000
    #E0 = E0/C
    #N0 = N0/C
    #E_goal = E_goal/C
    #N_goal = N_goal/C
    #N1 = [n/C for n in N1]
    #E1 = [e/C for e in E1]

    # compute vectorized
    p_OS = np.array([E0, N0])
    v_OS = np.array([vE0, vN0])
    p_G  = np.array([E_goal, N_goal])

    # unit vector from OS to goal
    dg = norm(p_G-p_OS)
    n_sg = (p_G-p_OS)/dg

    # ------------- attractive forces ------------
    # attractive force to goal
    F_att_E = 0.0
    F_att_N = 0.0

    if dg < d_star:
        F = dg
    else:
        F = d_star

    #print(f"F_att1: {F}")

    E_add, N_add = F * k_a1 * n_sg
    F_att_E += E_add
    F_att_N += N_add

    # attractive force to planned path
    CTE = cte(N1=N_start, E1=E_start, N2=N_goal, E2=E_goal, NA=N0, EA=E0)
    
    if abs(CTE) > d_l:
        # compute point on planned path G
        ATE = ate(N1=N_start, E1=E_start, N2=N_goal, E2=E_goal, NA=N0, EA=E0)
        angle = bng_abs(N0=N_start, E0=E_start, N1=N_goal, E1=E_goal)
        E_add, N_add = xy_from_polar(r=ATE, angle=angle)
        G_E = E_start + E_add
        G_N = N_start + N_add
        
        # unit vector in direction of G
        G_path = np.array([G_E, G_N])
        n_SG = (G_path-p_OS)/norm(G_path-p_OS)

        # add force towards G
        F = abs(CTE)
        #print(f"F_att2: {F}")
        E_add, N_add = F * k_a2 * n_SG
        F_att_E += E_add
        F_att_N += N_add

    # ------------- repulsive forces ------------
    F_rep_E = 0.0
    F_rep_N = 0.0

    for i in range(len(N1)):
        # quick access
        p_TS = np.array([E1[i], N1[i]])
        v_TS = np.array([vE1[i], vN1[i]])

        # compute some metrics
        d = norm(p_TS-p_OS)
        V_UR = v_OS - v_TS
        V_UR_norm = norm(V_UR)

        n_so = (p_TS-p_OS)/d  # unit vector from OS to TS (ship to obstacle)
        n_os = (p_OS-p_TS)/d  # unit vector from TS to OS (obstacle to ship)

        r, angle = polar_from_xy(x=n_so[0], y=n_so[1])
        e, n = xy_from_polar(r=r, angle=angle_to_2pi(angle + dtr(90.0)))
        n_so_perp = np.array([e, n]) # perpendicular on n_so to guarantee starboard collision avoidance

        # get rho_0 based on relative velocity
        rho_0 = k_m * np.sqrt((V_UR_norm * t_s)**2 + R_UO**2)

        if (d <= rho_0) and np.dot(V_UR, n_so) > 0:

            # distance-based force from the target ship to the own ship 
            F1 = (1/d - 1/rho_0) * dg**2 / d**2
            E_add, N_add = F1 * k_r1 * n_os
            F_rep_E += E_add
            F_rep_N += N_add

            # distance-based force from the ship to the goal
            F2 = (1/d - 1/rho_0)**2 * dg
            E_add, N_add = F2 * k_r1 * n_sg
            F_rep_E += E_add
            F_rep_N += N_add

            # relative velocity-based force to steer starboard with relation to the target ship
            theta = get_theta(p_OS=p_OS, v_OS=v_OS, p_TS=p_TS, v_TS=v_TS)
            F3 = V_UR_norm * np.sin(theta)
            E_add, N_add = F3 * k_r2 * n_so_perp
            F_rep_E += E_add
            F_rep_N += N_add

            # fixed force from the target ship to the own ship
            F4 = 1.0
            E_add, N_add = F4 * k_r2 * n_os
            F_rep_E += E_add
            F_rep_N += N_add

            #print(f"F1: {F1}, F2: {F2}, F3: {F3}, F4: {F4} \n")
            #print("----------------")

    # aggregate
    F_x = F_att_E + F_rep_E
    F_y = F_att_N + F_rep_N

    # translate into Δheading
    dh = angle_to_pi(math.atan2(F_x, F_y) - head0)
    return np.clip(dh, -dh_clip, dh_clip)


class VO_Planner:
    """Defines a velocity obstacle-based planner building on Kuwata et al. (2014)."""
    def __init__(self, N_TSs) -> None:

        # parameters
        self.h            = 60
        self.dmin         = NM_to_meter(0.75)
        self.tmax         = 15 * 60 # s
        self.N_search     = 500
        self.dhead_search = dtr(90)
        self.dhead_max    = dtr(2.5)
        self.w_time       = 100.0
        self.w_v          = 1.0
        self.CR_rec_dist  = NM_to_meter(2.0)
        self.CR_al        = 0.1

        # ship domain size
        Lpp = 320.0
        B = 58.0
        self.domain_A = 3 * Lpp + 0.5 * Lpp
        self.domain_B = 1 * Lpp + 0.5 * B   
        self.domain_C = 1 * Lpp + 0.5 * Lpp
        self.domain_D = 3 * Lpp + 0.5 * B

        # setup
        self.N_TSs = N_TSs
        self.encounters = [list([0] * self.h) for _ in range(N_TSs)]
        self.static_heads = np.linspace(start=-self.dhead_search, stop=self.dhead_search, num=self.N_search)

        # counter for hysteresis
        self.h_cnt = 0

    def _ttc(self, p_OS, v_OS, p_TS, v_TS):
        A = norm(v_OS-v_TS)**2
        B = 2 * np.dot(p_OS-p_TS, v_OS-v_TS)
        C = norm(p_OS-p_TS)**2 - self.dmin**2

        pre = -B/(2*A)
        root = np.sqrt(B**2 / (4*A**2) - C/A)
        return np.min([pre + root, pre - root])

    def _tcpa(self, p_OS, v_OS, p_TS, v_TS):
        if norm(v_OS-v_TS) <= 1e-5:
            return 0.0
        else:
            return -np.dot(p_OS-p_TS, v_OS-v_TS) / norm(v_OS-v_TS)**2

    def _dcpa(self, p_OS, v_OS, p_TS, v_TS, tcpa):
        return norm(p_OS + v_OS*tcpa - p_TS - v_TS*tcpa)

    def _get_cost_for_TS(self, i:int, heads:np.ndarray,
                         p_OS:np.ndarray, v_OS:np.ndarray, v_OS_norm:float, head0:float, 
                         p_TS:np.ndarray, v_TS:np.ndarray, head1:float,
                         v_desired:np.ndarray, C_idx:int, check_COLREGs:bool):
        # init zero costs
        head_costs_TS = np.zeros(self.N_search)

        # precollision check
        TCPA = self._tcpa(p_OS=p_OS, v_OS=v_OS, p_TS=p_TS, v_TS=v_TS)
        DCPA = self._dcpa(p_OS=p_OS, v_OS=v_OS, p_TS=p_TS, v_TS=v_TS, tcpa=TCPA)

        if ((0 <= TCPA <= self.tmax) and (DCPA <= self.dmin)):
            
            # classify COLREG-situation
            s = self._get_COLREG_situation(N0=p_OS[1], E0=p_OS[0], head0=head0, v0=v_OS_norm,
                                           N1=p_TS[1], E1=p_TS[0], head1=head1, v1=norm(v_TS))
            self.encounters[i][C_idx] = s

        # not in relevant range, thus no COLREG situation
        else:
            self.encounters[i][C_idx] = 0

        # COLREG compliance
        if check_COLREGs:
            if any([s in self.encounters[i] for s in [1, 2, 4]]):
                no_left = True
            else:
                no_left = False
        
        # check every single heading
        for n in range(self.N_search):

            # Would it lead to collision?
            vE0_tmp, vN0_tmp = xy_from_polar(r=v_OS_norm, angle=heads[n])
            v0_tmp = np.array([vE0_tmp, vN0_tmp])

            TCPA_tmp = self._tcpa(p_OS=p_OS, v_OS=v0_tmp, p_TS=p_TS, v_TS=v_TS)
            DCPA_tmp = self._dcpa(p_OS=p_OS, v_OS=v0_tmp, p_TS=p_TS, v_TS=v_TS, tcpa=TCPA_tmp)

            # if yes, add infinite costs
            if (0 <= TCPA_tmp) and (DCPA_tmp <= self.dmin):
                head_costs_TS[n] = np.inf
                continue

            # COLREG compliance
            if check_COLREGs:
                if no_left:
                    
                    # not in V3 area of Kuwata et al. (2014), 
                    # so OS is not moving away from TS (which would be fine)
                    if np.dot(p_TS - p_OS, v0_tmp - v_TS) >= 0:

                        # in V1 area (bow-crossing of TS) - if we are, infinity cost
                        #d_vec_DCPA_tmp = p_OS + v0_tmp*TCPA_tmp - p_TS - v_TS*TCPA_tmp
                        #if np.dot(d_vec_DCPA_tmp, v_TS) > 0:
                        if np.cross(p_TS-p_OS, v0_tmp-v_TS) > 0:
                            head_costs_TS[n] = np.inf
                            continue

            # cost based on TCPA and difference to desired heading
            if TCPA_tmp == 0.0:
                TCPA_tmp = 1e-10
            head_costs_TS[n] += self.w_v * norm(v_desired-v0_tmp) #+ self.w_time / TCPA_tmp 

        return head_costs_TS

    def plan(self, 
             N0:float, E0:float, head0, vN0:float, vE0:float,
             N_goal:float, E_goal:float,
             N1:list, E1:list, head1:list, vN1:list, vE1:list):

        # vectorize
        p_OS  = np.array([E0, N0])
        v_OS  = np.array([vE0, vN0])
        v_OS_norm = norm(v_OS)
        
        # angle to goal
        g_angle = bng_abs(N0=N0, E0=E0, N1=N_goal, E1=E_goal)
        v_d_E, v_d_N = xy_from_polar(r=v_OS_norm, angle=g_angle)
        v_desired = np.array([v_d_E, v_d_N])

        # full heading set
        heads = (self.static_heads + head0) % (2*math.pi)

        # COLREG-encounter index
        C_idx = self.h_cnt % self.h

        # initially try under consideration of COLREGs
        check_COLREGs = True

        while True:
            
            # init total costs
            head_costs = np.zeros_like(heads)

            for i in range(len(N1)):

                # vectorize
                p_TS = np.array([E1[i], N1[i]])
                v_TS = np.array([vE1[i], vN1[i]])

                # compute costs
                head_costs += self._get_cost_for_TS(i=i, heads=heads, 
                                                    p_OS=p_OS, v_OS=v_OS, v_OS_norm=v_OS_norm, head0=head0,
                                                    p_TS=p_TS, v_TS=v_TS, head1=head1[i], v_desired=v_desired,
                                                    C_idx=C_idx, check_COLREGs=check_COLREGs)
            # all good if there are non-infinite costs
            if not np.all(head_costs == np.inf):
                break

            # otherwise neglect COLREGs
            else:
                raise Exception("Insufficient.")
                if check_COLREGs is False:
                    raise Exception("The problem is ill-posed even when neglecting traffic rules.")
                else:
                    check_COLREGs = False

        # update the COLREG counter
        self.h_cnt += 1

        # no TS has been in sight - select angle to goal
        if np.all(head_costs == 0):
            head_desired = g_angle

        # otherwise select the one with minimum costs
        else:
            idx = np.random.choice(np.flatnonzero(head_costs == head_costs.min()))
            head_desired = heads[idx]

        # translate into Δheading
        dh = head_desired - head0

        if dh < -math.pi:
            dh += 2*math.pi
        elif dh > math.pi:
            dh -= 2*math.pi

        return np.clip(dh, -self.dhead_max, self.dhead_max), head_costs, heads

    def _get_COLREG_situation(self, N0, E0, head0, v0, N1, E1, head1, v1):
        """Determines the COLREG situation from the perspective of the OS. 
        Follows Xu et al. (2020, Ocean Engineering; 2022, Neurocomputing).

        Returns:
            0  -  no conflict situation
            1  -  head-on
            2  -  starboard crossing
            3  -  portside crossing
            4  -  overtaking
        """

        # relative bearing from OS to TS
        bng_OS = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0)

        # relative bearing from TS to OS
        bng_TS = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)

        # intersection angle
        C_T = head_inter(head_OS=head0, head_TS=head1)

        # velocity component of OS in direction of TS
        V_rel_x, V_rel_y = project_vector(VA=v0, angleA=head0, VB=v1, angleB=head1)
        V_rel = polar_from_xy(x=V_rel_x, y=V_rel_y, with_r=True, with_angle=False)[0]

        # COLREG 1: Head-on
        if -5 <= rtd(angle_to_pi(bng_OS)) <= 5 and 175 <= rtd(C_T) <= 185:
            return 1
        
        # COLREG 2: Starboard crossing
        if 5 <= rtd(bng_OS) <= 112.5 and 185 <= rtd(C_T) <= 292.5:
            return 2

        # COLREG 3: Portside crossing
        if 247.5 <= rtd(bng_OS) <= 355 and 67.5 <= rtd(C_T) <= 175:
            return 3

        # COLREG 4: Overtaking
        if 112.5 <= rtd(bng_TS) <= 247.5 and -67.5 <= rtd(angle_to_pi(C_T)) <= 67.5 and V_rel > v1:
            return 4

        # COLREG 0: nothing
        return 0
