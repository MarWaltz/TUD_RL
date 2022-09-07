import math

import numpy as np
import utm
from tud_rl.envs._envs.VesselFnc import (ED, angle_to_2pi, angle_to_pi,
                                         bng_abs, bng_rel, rtd)


def to_latlon(north, east, number):
    """Converts North, East, number in UTM into longitude and latitude. Assumes northern hemisphere.
    Returns: (lat, lon)"""
    return utm.to_latlon(easting=east, northing=north, zone_number=number, northern=True, strict=False)


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


def find_nearest_two_old(array, value):
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


def find_nearest_two(array, value):
    """Finds the closest two entries in a SORTED (ascending) array with UNIQUE entries to a given value. Based on binary search.
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
    n = len(array); i = 0; j = n
    while i < j:
        mid = (i + j) // 2

        # direct hit
        if array[mid] == value:
            return value, mid, value, mid

        # target is to the left
        if value < array[mid]:

            # target is right of one entry smaller
            if mid > 0:
                if value > array[mid-1]:
                    return array[mid-1], mid-1, array[mid], mid
            
            # repeat for left half
            j = mid

        # target is to the right
        else:
            
            # target is left of one entry larger
            if mid < n-1:
                if value < array[mid+1]:
                    return array[mid], mid, array[mid+1], mid+1

            # repeat for right half
            i = mid + 1

    # one element left
    return find_neighbor(array=array, value=value, idx=mid)


def find_neighbor(array, value, idx):
    """Checks whether left or right neighbor of a given idx in an array is closer to the value, when idx is the closest in general.
    Returns (entry1, idx1, entry2, idx2)."""

    # corner cases
    if idx == 0:
        return array[idx], idx, array[idx+1], idx+1

    elif idx == len(array)-1:
        return array[idx-1], idx-1, array[idx], idx
    
    # check neighbors
    else:
        left = array[idx-1]
        right = array[idx+1]

        if np.abs(left - value) <= np.abs(right - value):
            return left, idx-1, array[idx], idx
        else:
            return array[idx], idx, right, idx+1


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

    # value has been in array or completely out of the domain
    if lat_low == lat_upp and lon_low == lon_upp:
        return Z[lat_low_idx, lon_low_idx]

    # latitudes are equal
    elif lat_low == lat_upp:
        depth_S6 = Z[lat_low_idx, lon_low_idx]
        depth_S7 = Z[lat_low_idx, lon_upp_idx]

        delta_lon6Q = lon_q - lon_low
        delta_lonQ7 = lon_upp - lon_q
        return (delta_lon6Q * depth_S7 + delta_lonQ7 * depth_S6) / (delta_lon6Q + delta_lonQ7)

    # longitudes are equal
    elif lon_low == lon_upp:
        depth_S1 = Z[lat_upp_idx, lon_low_idx]
        depth_S3 = Z[lat_low_idx, lon_low_idx]

        delta_lat16 = lat_upp - lat_q
        delta_lat63 = lat_q - lat_low
        return (delta_lat16 * depth_S3 + delta_lat63 * depth_S1) / (delta_lat16 + delta_lat63)

    # everything is different
    else:
        depth_S1 = Z[lat_upp_idx, lon_low_idx]
        depth_S3 = Z[lat_low_idx, lon_low_idx]

        depth_S2 = Z[lat_upp_idx, lon_upp_idx]
        depth_S4 = Z[lat_low_idx, lon_upp_idx]

        delta_lat16 = lat_upp - lat_q
        delta_lat63 = lat_q - lat_low

        depth_S6 = (delta_lat16 * depth_S3 + delta_lat63 * depth_S1) / (delta_lat16 + delta_lat63)
        depth_S7 = (delta_lat16 * depth_S4 + delta_lat63 * depth_S2) / (delta_lat16 + delta_lat63)

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


def VFG(N1, E1, N2, E2, NA, EA, K, N3=None, E3=None):
    """Computes the cte, desired course, and path angle based on the vector field guidance method following pp.354-55 of Fossen (2021).
    The waypoints are (N1, E1) and (N2, E2), while the agent is at (NA, EA). K is the convergence rate of the vector field.
    
    Returns:
        cte (float), desired_course (float, in radiant), path_angle (float, in radiant)"""

    assert K > 0, "K of VFG must be larger zero."

    # get path angle between point 1 and 2
    pi_path_12 = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)

    # get CTE
    ye = cte(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path_12)

    # potentially consired point 3 for desired course
    if N3 is not None and E3 is not None:
        
        # ate between point 1 and 2
        ate_12 = ate(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path_12)

        # path angle between points 2 and 3
        pi_path_23 = bng_abs(N0=N2, E0=E2, N1=N3, E1=E3)

        # percentage of overall distance
        frac = ate_12 / ED(N0=N1, E0=E1, N1=N2, E1=E2)
        w23 = frac**15

        # construct new angle
        pi_path_up = angle_to_2pi(w23*pi_path_23 + (1-w23)*pi_path_12)
    
    else:
        pi_path_up = pi_path_12

    # get desired course, which is rotated back to NED
    return ye, angle_to_2pi(pi_path_up - math.atan(ye * K)), pi_path_12


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

    # limit case one
    if min_idx == len(lat_array)-1:
        raise ValueError("The agent spawned already at the goal!")

    # limit case two
    elif min_idx == 0:
        idx1 = min_idx
        idx2 = min_idx + 1
        wp1_N, wp1_E, _ = to_utm(lat_array[idx1], lon_array[idx1])
        wp2_N, wp2_E, _ = to_utm(lat_array[idx2], lon_array[idx2])

    # arbitrarily select prior index as current set of wps
    else:
        idx1 = min_idx - 1
        idx2 = min_idx
        wp1_N, wp1_E, _ = to_utm(lat_array[idx1], lon_array[idx1])
        wp2_N, wp2_E, _ = to_utm(lat_array[idx2], lon_array[idx2])

        # check whether waypoints should be switched
        if switch_wp(wp1_N=wp1_N, wp1_E=wp1_E, wp2_N=wp2_N, wp2_E=wp2_E, a_N=a_n, a_E=a_e):
            idx1 += 1
            idx2 += 1
            wp1_N, wp1_E, _ = to_utm(lat_array[idx1], lon_array[idx1])
            wp2_N, wp2_E, _ = to_utm(lat_array[idx2], lon_array[idx2])

    return idx1, wp1_N, wp1_E, idx2, wp2_N, wp2_E


def fill_array(Z, lat_idx1, lon_idx1, lat_idx2, lon_idx2, value):
    """Returns the array Z where indicies between [lat_idx1, lon_idx1] and [lat_idx2, lon_idx2] are filled with value.
    Example:
    Z = [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]]

    [lat_idx1, lon_idx1] = [0, 1]
    [lat_idx2, lon_idx2] = [2, 3]
    value = 5

    Output is:
    Z = [[0.0, 5.0, 5.0, 5.0, 0.0],
         [0.0, 5.0, 5.0, 5.0, 0.0],
         [0.0, 5.0, 5.0, 5.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]]

    Args:
        Z (np.array(lat_length, lon_length)): Array with data
        lat_idx1 (int): lat-idx of first point
        lon_idx1 (int): lon-idx of first point
        lat_idx2 (int): lat-idx of second point
        lon_idx2 (int): lon-idx of second point
        value (float): value used for filling 
    """
    # indices should be in array
    lat_N, lon_N = Z.shape
    lat_idx1 = int(np.clip(lat_idx1, 0, lat_N-1))
    lat_idx2 = int(np.clip(lat_idx2, 0, lat_N-1))
    lon_idx1 = int(np.clip(lon_idx1, 0, lon_N-1))
    lon_idx2 = int(np.clip(lon_idx2, 0, lon_N-1))
    
    # order points to slice over rectangle
    lat_idx1_c = min([lat_idx1, lat_idx2])
    lat_idx2_c = max([lat_idx1, lat_idx2])

    lon_idx1_c = min([lon_idx1, lon_idx2])
    lon_idx2_c = max([lon_idx1, lon_idx2])

    # filling
    Z[lat_idx1_c:lat_idx2_c+1, lon_idx1_c:lon_idx2_c+1] = value
    return Z

"""Z = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]])
lat_idx1 = 0
lon_idx1 = 1
lat_idx2 = 2
lon_idx2 = 3
value = 5
print(fill_array(Z, lat_idx1, lon_idx1, lat_idx2, lon_idx2, value))
"""

class Hull:
    def __init__(self, Lpp, B, N=1000, approx_method="ellipse", h_xs=None, h_ys=None) -> None:
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
        assert approx_method in ["rectangle", "ellipse"], "Unknown ship hull approximation."

        self.Lpp = Lpp
        self.B = B
        self.N = N
        self.approx_method = approx_method
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

            if self.approx_method == "rectangle":
                theta = np.arctan(b/a)
                self.h_xs = np.zeros(self.N)
                self.h_ys = np.zeros(self.N)

                for n, eta in enumerate(degs):

                    # I
                    if 0 <= eta < theta:
                        x = a
                        y = a * np.tan(eta)

                    # II
                    elif theta <= eta < math.pi/2:
                        eta_p = math.pi/2 - eta
                        y = b
                        x = b * np.tan(eta_p)

                    # III                    
                    elif math.pi/2 <= eta < math.pi - theta:
                        eta_p = eta - math.pi/2
                        y = b
                        x = -b * np.tan(eta_p)
                    
                    # IV
                    elif math.pi - theta <= eta < math.pi:
                        eta_p = math.pi - eta
                        x = -a
                        y = a * np.tan(eta_p)

                    # V                    
                    elif math.pi <= eta < math.pi + theta:
                        eta_p = eta - math.pi
                        x = -a
                        y = -a * np.tan(eta_p)

                    # VI
                    elif math.pi + theta <= eta < 3/2*math.pi:
                        eta_p = 3/2*math.pi - eta
                        y = -b
                        x = -b * np.tan(eta_p)

                    # VII                    
                    elif 3/2*math.pi <= eta < 2*math.pi - theta:
                        eta_p = eta - 3/2*math.pi
                        y = -b
                        x = b * np.tan(eta_p)
                    
                    # VIII
                    else:
                        eta_p = 2*math.pi - eta
                        x = a
                        y = -a * np.tan(eta_p)

                    self.h_xs[n] = x
                    self.h_ys[n] = y

            elif self.approx_method == "ellipse":
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

        #import matplotlib.pyplot as plt
        #plt.plot(self.h_ys, self.h_xs, marker='o', markersize=3)
        #plt.gca().set_aspect('equal')

        #for n in range(self.N):
        #    plt.scatter(self.y0s[n], self.x0s[n], color="red", s=3)
        #    plt.arrow(self.y0s[n] - self.N_Ys[n], self.x0s[n] - self.N_Xs[n], self.N_Ys[n], self.N_Xs[n], color="green", length_includes_head=True,
        #                  head_width=2, head_length=3)
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

    # detect non-shadow region
    N_WX = np.cos(beta_w)
    N_WY = np.sin(beta_w)
    alphas = np.arccos(hull.N_Xs*N_WX/hull.dls + hull.N_Ys*N_WY/hull.dls)
    non_shadow = (alphas <= math.pi/2)

    # compute integration block components
    inner_terms = np.sin(hull.thetas + beta_w)**2 + 2*omega_0*U/g * (np.cos(beta_w) - hull.cos_thetas*np.cos(hull.thetas + beta_w))
    inner_terms *= non_shadow * hull.dls

    # integrate
    X_int = np.sum(inner_terms * hull.sin_thetas)
    Y_int = np.sum(inner_terms * hull.cos_thetas)
    M_int = np.sum(inner_terms * (hull.x0s * hull.cos_thetas - hull.y0s * hull.sin_thetas))

    # pre-factors
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
