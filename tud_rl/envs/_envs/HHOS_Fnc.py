import math

import numpy as np
import utm
from tud_rl.envs._envs.VesselFnc import (ED, angle_to_2pi, angle_to_pi,
                                         bng_abs, bng_rel, rtd, dtr)


def get_utm_zone_number(lat, lon):
    """Computes the UTM zone number (not the letter) for given latitude and longitude.
    Considers the special cases for Norway and Svalbard."""
    if lat > 55 and lat < 64 and lon > 2 and lon < 6:
        return 32
    elif lat > 71 and lon >= 6 and lon < 9:
        return 31
    elif lat > 71 and lon >= 9 and lon < 12:
        return 33
    elif lat > 71 and lon >= 18 and lon < 21:
        return 33
    elif lat > 71 and lon >= 21 and lon < 24:
        return 35
    elif lat > 71 and lon >= 30 and lon < 33:
        return 35
    elif lon >= -180 and lon <= 180:
        return (math.floor((lon + 180)/6) % 60) + 1
    else:
        raise ValueError("UTM zone determination failed. Check your latitude and longitude again.")


def to_latlon(north, east, number):
    """Converts North, East, number in UTM into longitude and latitude. Assumes northern hemisphere.
    Returns: (lat, lon)"""
    return utm.to_latlon(easting=east, northing=north, zone_number=number, northern=True)


def to_utm(lat, lon):
    """Converts latitude and longitude into North, East, and zone number in UTM.
    Returns: (North, East, number)."""
    E, N, number, _ = utm.from_latlon(latitude=lat, longitude=lon)
    return (N, E, number)


def find_nearest(array, value):
    """Finds the closest entry in an array to a given value.
    Returns (entry, idx)."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], int(idx)


def find_nearest_two(array, value):
    """Finds the closest two entries in a SORTED (ascending) array with UNIQUE entries to a given value.
    Returns (entry1, idx1, entry2, idx2)."""
    array = np.asarray(array)

    # out of array
    if value <= array[0]:
        return array[0], int(0), array[0], int(0)

    if value >= array[-1]:
        idx = len(array)-1
        ent = array[-1]
        return ent, int(idx), ent, int(idx)

    # in array
    abs_diff = np.abs(array - value)
    hit_idx = np.where(abs_diff == 0.0)[0]
    if len(hit_idx) == 1:
        hit_idx = hit_idx[0]
        return array[hit_idx], int(hit_idx), array[hit_idx], int(hit_idx)

    # neighbours
    idx1, idx2 = np.sort(np.argpartition(abs_diff, kth=2)[0:2])
    return array[idx1], int(idx1), array[idx2], int(idx2)


def Z_at_latlon(Z, lat_array, lon_array, lat_q, lon_q):
    """Computes the linearly interpolated value (e.g. water depth, wind) at a (queried) longitude-latitude position.
    Args:
        Z[np.array(M, N)]:        data over grid
        lat_array[np.array(M,)]:  latitude points
        lon_array[np.array(N,)]:  longitude points
        lat_q[float]: latitude of interest
        lon_q[float]: longitude of interest
    """
    # determine four corners
    lat_low, lat_low_idx, lat_upp, lat_upp_idx = find_nearest_two(array=lat_array, value=lat_q)
    lon_low, lon_low_idx, lon_upp, lon_upp_idx = find_nearest_two(array=lon_array, value=lon_q)

    lat_low_idx = int(lat_low_idx)
    lat_upp_idx = int(lat_upp_idx)
    lon_low_idx = int(lon_low_idx)
    lon_upp_idx = int(lon_upp_idx)

    # value has been in array
    if lat_low == lat_upp and lon_low == lon_upp:
        return Z[lat_low_idx, lon_low_idx]

    # get depth of S6 and S7
    if lat_low == lat_upp:
        depth_S6 = Z[lat_low_idx, lon_low_idx]
        depth_S7 = Z[lat_low_idx, lon_upp_idx]
    else:
        # depth of S1, S2, S3, S4
        depth_S1 = Z[lat_upp_idx, lon_low_idx]
        depth_S3 = Z[lat_low_idx, lon_low_idx]

        depth_S2 = Z[lat_upp_idx, lon_upp_idx]
        depth_S4 = Z[lat_low_idx, lon_upp_idx]

        delta_lat16 = lat_upp - lat_q
        delta_lat63 = lat_q - lat_low

        depth_S6 = (delta_lat16 * depth_S3 + delta_lat63 * depth_S1) / (delta_lat16 + delta_lat63)
        depth_S7 = (delta_lat16 * depth_S4 + delta_lat63 * depth_S2) / (delta_lat16 + delta_lat63)

    # get depth of SQ
    if lon_q == lon_low:
        return depth_S6
    elif lon_q == lon_upp:
        return depth_S7
    else:
        delta_lon6Q = lon_q - lon_low
        delta_lonQ7 = lon_upp - lon_q
        return (delta_lon6Q * depth_S7 + delta_lonQ7 * depth_S6) / (delta_lon6Q + delta_lonQ7)


def mps_to_knots(mps):
    """Transform m/s in knots."""
    return mps * 1.943844


def knots_to_mps(knots):
    """Transforms knots in m/s."""
    return knots / 1.943844


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


def VFG(N1, E1, N2, E2, NA, EA, K):
    """Computes the cte, desired course, and path angle based on the vector field guidance method following pp.354-55 of Fossen (2021).
    The waypoints are (N1, E1) and (N2, E2), while the agent is at (NA, EA). K is the convergence rate of the vector field.
    
    Returns:
        cte (float), desired_course (float, in radiant), path_angle (float, in radiant)"""

    assert K > 0, "K of VFG must be larger zero."

    # get path angle
    pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)

    # get CTE
    ye = cte(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path)

    # get desired course, which is rotated back to NED
    return ye, pi_path - math.atan(ye * K), pi_path


def get_init_two_wp(lat_array, lon_array, a_n, a_e):
    """Returns for a given set of waypoints and an agent position the coordinates of the first two waypoints.

    Args:
        lon_array (np.array): array of lon-coordinates of waypoints
        lat_array (np.array): array of lat-coordinates of waypoints
        a_n (float): agent N position
        a_e (float): agent E position
    Returns:
        wp1_idx, wp1_N, wp1_E, wp2_idx, wp2_N, wp2_E
    """
    # transform everything in utm
    ne_tups = [to_utm(lat=lat_array[idx], lon=lon_array[idx])[0:2] for idx in range(len(lat_array))]

    # compute the smallest euclidean distance
    EDs = [ED(N0=a_n, E0=a_e, N1=wp_n, E1=wp_e, sqrt=False) for (wp_n, wp_e) in ne_tups]
    min_idx = np.argmin(EDs)

    # limit cases
    if min_idx == len(lat_array)-1:
        raise ValueError("The agent spawned already at the goal!")
    if min_idx == 0:
        idx1 = min_idx
        idx2 = min_idx + 1

    # the second wp is constituted by the smaller ED of the surrounding wps surrounding the min-ED wp
    if EDs[min_idx-1] < EDs[min_idx+1]:
        idx1 = min_idx - 1
        idx2 = min_idx
    else:
        idx1 = min_idx
        idx2 = min_idx + 1
    
    # return both wps
    wp1_N, wp1_E, _ = to_utm(lat_array[idx1], lon_array[idx1])
    wp2_N, wp2_E, _ = to_utm(lat_array[idx2], lon_array[idx2])

    return idx1, wp1_N, wp1_E, idx2, wp2_N, wp2_E


def switch_wp(wp1_N, wp1_E, wp2_N, wp2_E, a_N, a_E):
    """Decides whether we should move on to the next pair of waypoints. Returns a boolean, True if we should switch.

    Args:
        wp1_N (float): N of first wp
        wp1_E (float): E of first wp
        wp2_N (float): N of second wp
        wp2_E (float): E of second wp
        a_N (float): agent N
        a_E (float): agent E
    """
    # path angle
    pi_path = bng_abs(N0=wp1_N, E0=wp1_E, N1=wp2_N, E1=wp2_E)

    # check relative bearing
    bng_rel_p = angle_to_pi(bng_rel(N0=a_N, E0=a_E, N1=wp2_N, E1=wp2_E, head0=pi_path))

    if abs(rtd(bng_rel_p)) > 90:
        return True
    else:
        return False


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
            # define ellipse
            a = self.Lpp*0.5
            b = self.B*0.5
            degs = np.linspace(0, 2*math.pi, self.N, endpoint=False)
            lengths = np.power(np.power(np.cos(degs),2) / a + np.power(np.sin(degs),2) / b, -0.5)

            # construct points
            self.h_xs = lengths * np.cos(degs)
            self.h_ys = lengths * np.sin(degs)

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

        # some usefull definitions
        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)

        #import matplotlib.pyplot as plt
        #plt.plot(h_ys, h_xs, marker='o', markersize=3)
        #plt.gca().set_aspect('equal')
        #plt.show()

            #dls_tmp = np.sqrt(N_Xs[n]**2 + N_Ys[n]**2)
            #show = np.abs(angle_to_pi(np.arccos(N_Xs[n]*N_WX/dls_tmp + N_Ys[n]*N_WY/dls_tmp))) <= math.pi/2

            #if show:
            #    plt.scatter(y0s[n], x0s[n], color="red", s=3)
            #    plt.arrow(y0s[n] - N_WY, x0s[n] - N_WX, N_WY, N_WX, color="red", length_includes_head=True,
            #    head_width=0.2, head_length=0.3)
            #    plt.arrow(y0s[n] -N_Ys[n], x0s[n] - N_Xs[n], N_Ys[n], N_Xs[n], color="green", length_includes_head=True,
            #    head_width=0.2, head_length=0.3)
        #plt.show()


def get_wave_XYN(U, psi, T, C_b, alpha_WL, Lpp, beta_wave, eta_wave, T_0_wave, lambda_wave, rho, hull):
    """Computes the short wave induced surge, sway forces and the yaw moment. 
    Based on numerical integration outlined in Taimuri et al. (2020, Ocean Engineering).

    Args:
        U (float): speed of ship in m/s
        psi (float): heading of ship in rad (usual N is 0° definition)
        T (float): ship draft in m
        C_b (float): block coefficient of ship
        alpha_WL (float): sectional flare angle of the vessel (in rad, 0 for tanker is realistic)
        L_pp (float): length between perpendiculars
        beta_wave (float): incident wave angle in rad (usual N is 0° definition, e.g., 270° means waves flow from W to E)
        eta_wave (float): incident wave amplitude in m
        T_0_wave (float): modal period of waves in s
        lambda_wave (float) : wave length in m
        rho (float): water density in kg/m³
        hull (Hull): hull object with relevant integration components

    Note:
        If h_xs and h_ys are not given, the argument 'B' is required. In this case, the hull of the ship is approximated
        by an ellipse using Lpp and B.

    Returns:
        surge force, sway force, yaw moment (all floats)
    """
    # parameters
    g = 9.80665                      # gravity in m/s²
    omega_0 = 2*math.pi / T_0_wave   # model frequency in rad/s
    k = 2*math.pi/ lambda_wave       # wave number in rad/m
    Fn = U / math.sqrt(g * Lpp)      # Froude number

    # wave angle from vessel perspective
    beta_w = angle_to_2pi(beta_wave - psi - math.pi)

    # compute integration block components
    N_WX = np.cos(beta_w)
    N_WY = np.sin(beta_w)
    alphas = [angle_to_pi(a) for a in np.arccos(hull.N_Xs*N_WX/hull.dls + hull.N_Ys*N_WY/hull.dls)]

    # integration
    inner_terms = np.cos(alphas)**2 + 2*omega_0*U/g * (-N_WX - hull.cos_thetas*np.sin(alphas)) * (np.abs(alphas) <= math.pi/2)
    
    X_int = np.sum(inner_terms * hull.sin_thetas * hull.dls)
    Y_int = np.sum(inner_terms * hull.cos_thetas * hull.dls)
    M_int = np.sum(inner_terms * (hull.x0s * hull.cos_thetas - hull.y0s * hull.sin_thetas) * hull.dls)

    FX_SW = 0.5 * rho * g * eta_wave**2 * X_int
    FY_SW = 0.5 * rho * g * eta_wave**2 * Y_int
    MN_SW = 0.5 * rho * g * eta_wave**2 * M_int

    # draft correction
    corr_d = 1 - math.exp(-2 * k * T)
    FX_SW *= corr_d
    FY_SW *= corr_d
    MN_SW *= corr_d

    # speed correction
    FX_SW *= (0.87/C_b)**(1 + 4*math.sqrt(Fn)) * 1/np.cos(alpha_WL)
    
    return FX_SW, FY_SW, MN_SW

#print(get_wave_XYN(U=0.0, psi=0.0, T=20.0, C_b=0.8, alpha_WL=0.0, Lpp=64.0, \
#    beta_wave=dtr(270.0), eta_wave=3.0, T_0_wave=5.0, lambda_wave=10.0, rho=1000, B=11))
