import math

import numpy as np
import utm
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
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
        cte (float), desired_course (float, rad), path_angle12 (float, rad), smoothed path_angle (float, rad)"""

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
        frac = np.clip(ate_12 / ED(N0=N1, E0=E1, N1=N2, E1=E2), 0.0, 1.0)
        w23 = frac**1

        # adjustment to avoid boundary issues at 2pi
        if abs(pi_path_12-pi_path_23) >= math.pi:
            if pi_path_12 >= pi_path_23:
                pi_path_12 = angle_to_pi(pi_path_12)
            else:
                pi_path_23 = angle_to_pi(pi_path_23)

        # construct new angle
        pi_path_up = angle_to_2pi(w23*pi_path_23 + (1-w23)*pi_path_12)
    
    else:
        pi_path_up = pi_path_12

    # get desired course, which is rotated back to NED
    return ye, angle_to_2pi(pi_path_up - math.atan(ye * K)), pi_path_12, pi_path_up


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
    n, e, _ = to_utm(lat=lat_array, lon=lon_array)

    # compute the smallest euclidean distance
    EDs = ED(N0=a_n, E0=a_e, N1=n, E1=e, sqrt=False)
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

def r_safe_dyn(a, r_min=100):
    assert -math.pi <= a < math.pi, "The angle for computing the safety distance should be in [-pi, pi)."
    if abs(a) <= math.pi/4:
        f = 2 + math.cos(4*a)
    else:
        f = 1
    return r_min * f


def apf(OS : KVLCC2, 
        TSs : list[KVLCC2], 
        G: dict, 
        dh_clip: float = None,
        du_clip: float = None,
        river_n=None, 
        river_e=None, 
        r_min=250, 
        k_a=0.001, 
        k_r_TS=1000, 
        k_r_river=1000):
    """Computes a local path based on the artificial potential field method, see Du, Zhang, and Nie (2019, IEEE).
    Returns: Δu, Δheading."""
    
    # quick access
    x_OS = OS.eta[1]
    y_OS = OS.eta[0]
    x_G = G["x"]
    y_G = G["y"]

    # attractive forces
    F_x = k_a * (x_G - x_OS)
    F_y = k_a * (y_G - y_OS)

    # repulsive forces due to vessel positions
    for TS in TSs:

        # distance
        x_TS = TS.eta[1]
        y_TS = TS.eta[0]
        d = ED(N0=y_OS, E0=x_OS, N1=y_TS, E1=x_TS)

        # bearing-dependent safety radius
        r_safe = r_safe_dyn(a=bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], head0=OS.eta[2], to_2pi=False), r_min=r_min)

        if d <= r_safe:
            F_x += k_r_TS * (1/r_safe - 1/d) * (x_TS - x_OS) / d
            F_y += k_r_TS * (1/r_safe - 1/d) * (y_TS - y_OS) / d

    # repulsive forces due to river bounds
    if river_n is not None and river_e is not None:
        for i, n in enumerate(river_n):

            # distance
            x_Riv = river_e[i]
            y_Riv = n
            d = ED(N0=y_OS, E0=x_OS, N1=y_Riv, E1=x_Riv)

            # bearing-dependent safety radius
            r_safe = r_safe_dyn(a=bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=y_Riv, E1=x_Riv, head0=OS.eta[2], to_2pi=False), r_min=r_min)

            if d <= r_min:
                F_x += k_r_river * (1/r_safe - 1/d) * (x_Riv - x_OS) / d
                F_y += k_r_river * (1/r_safe - 1/d) * (y_Riv - y_OS) / d

    # translate into Δu, Δheading
    du = math.sqrt(F_x**2 + F_y**2) / OS.m
    dh = angle_to_pi(math.atan2(F_x, F_y) - OS.eta[2])

    if du_clip is not None:
        du = np.clip(du, -du_clip, du_clip)
    if dh_clip is not None:
        dh = np.clip(dh, -dh_clip, dh_clip)

    return du, dh
