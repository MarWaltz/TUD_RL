import math
import pickle
from typing import List

import numpy as np
import pandas as pd
import utm

from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (ED, NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, get_theta, head_inter,
                                         meter_to_NM, norm, polar_from_xy, rtd,
                                         tcpa, xy_from_polar)


def to_latlon(north, east, number):
    """Converts North, East, number in UTM into longitude and latitude. Assumes northern hemisphere.
    Returns: (lat, lon)"""
    if isinstance(north, list):
        north = np.array(north)
    if isinstance(east, list):
        east = np.array(east)
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

def prep_angles_for_average(ang1, ang2):
    """Prepares two angles before they get averaged. Necessary since otherwise there are boundary issues at 2pi."""
    # make sure they are in [0,2pi)
    ang1 = angle_to_2pi(ang1)
    ang2 = angle_to_2pi(ang2)

    # transform if necessary
    if abs(ang1-ang2) >= math.pi:
        if ang1 >= ang2:
            ang1 = angle_to_pi(ang1)
        else:
            ang2 = angle_to_pi(ang2)
    return ang1, ang2

def Z_at_latlon(Z, lat_array, lon_array, lat_q, lon_q, angle:bool=False):
    """Computes the linearly interpolated value (e.g. water depth, wind) at a (queried) longitude-latitude position.
    Args:
        Z[np.array(M, N)]:        data over grid
        lat_array[np.array(M,)]:  latitude points
        lon_array[np.array(N,)]:  longitude points
        lat_q[float]: latitude of interest
        lon_q[float]: longitude of interest
        angle[bool]: whether data consists of angles (in radiant), need to consider special boundary issue handling
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
        Z_S6 = Z[lat_low_idx, lon_low_idx]
        Z_S7 = Z[lat_low_idx, lon_upp_idx]

        delta_lon6Q = lon_q - lon_low
        delta_lonQ7 = lon_upp - lon_q

        if angle:
            Z_S6, Z_S7 = prep_angles_for_average(Z_S6, Z_S7)
        out = (delta_lon6Q * Z_S7 + delta_lonQ7 * Z_S6) / (delta_lon6Q + delta_lonQ7)
        if angle:
            return angle_to_2pi(out)
        return out

    # longitudes are equal
    elif lon_low == lon_upp:
        Z_S1 = Z[lat_upp_idx, lon_low_idx]
        Z_S3 = Z[lat_low_idx, lon_low_idx]

        delta_lat16 = lat_upp - lat_q
        delta_lat63 = lat_q - lat_low

        if angle:
            Z_S1, Z_S3 = prep_angles_for_average(Z_S1, Z_S3)
        out = (delta_lat16 * Z_S3 + delta_lat63 * Z_S1) / (delta_lat16 + delta_lat63)
        if angle:
            return angle_to_2pi(out)
        return out

    # everything is different
    else:
        Z_S1 = Z[lat_upp_idx, lon_low_idx]
        Z_S3 = Z[lat_low_idx, lon_low_idx]

        Z_S2 = Z[lat_upp_idx, lon_upp_idx]
        Z_S4 = Z[lat_low_idx, lon_upp_idx]

        delta_lat16 = lat_upp - lat_q
        delta_lat63 = lat_q - lat_low

        if angle:
            Z_S1, Z_S3 = prep_angles_for_average(Z_S1, Z_S3)
            Z_S2, Z_S4 = prep_angles_for_average(Z_S2, Z_S4)

        Z_S6 = (delta_lat16 * Z_S3 + delta_lat63 * Z_S1) / (delta_lat16 + delta_lat63)
        Z_S7 = (delta_lat16 * Z_S4 + delta_lat63 * Z_S2) / (delta_lat16 + delta_lat63)

        if angle:
            Z_S6 = angle_to_2pi(Z_S6)
            Z_S7 = angle_to_2pi(Z_S7)

        delta_lon6Q = lon_q - lon_low
        delta_lonQ7 = lon_upp - lon_q

        if angle:
            Z_S6, Z_S7 = prep_angles_for_average(Z_S6, Z_S7)
        out = (delta_lon6Q * Z_S7 + delta_lonQ7 * Z_S6) / (delta_lon6Q + delta_lonQ7)
        if angle:
            return angle_to_2pi(out)
        return out

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
        pi_path_12, pi_path_23 = prep_angles_for_average(pi_path_12, pi_path_23)

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

def get_init_two_wp(n_array, e_array, a_n, a_e, stop_goal=False):
    """Returns for a given set of waypoints and an agent position the coordinates of the first two waypoints.

    Args:
        n_array (np.array): array of N-coordinates of waypoints
        e_array (np.array): array of E-coordinates of waypoints
        a_n (float): agent N position
        a_e (float): agent E position
        stop_goal(bool): whether to raise a ValueError if the agent spawns at the end of the path
    Returns:
        wp1_idx, wp1_N, wp1_E, wp2_idx, wp2_N, wp2_E
    """
    # compute the smallest euclidean distance
    EDs = ED(N0=a_n, E0=a_e, N1=n_array, E1=e_array, sqrt=False)
    min_idx = np.argmin(EDs)

    # limit cases
    if min_idx in [0, len(n_array)-1]:

        # start of path
        if min_idx == 0:
            idx1 = 0
            idx2 = 1
        
        # end of path
        else:
            idx1 = len(n_array)-2
            idx2 = len(n_array)-1
            if stop_goal:
                raise ValueError("The agent spawned already at the goal!")

        wp1_N = n_array[idx1]
        wp1_E = e_array[idx1]
        wp2_N = n_array[idx2]
        wp2_E = e_array[idx2]

    # arbitrarily select prior index as current set of wps
    else:
        idx1 = min_idx - 1
        idx2 = min_idx
        wp1_N = n_array[idx1]
        wp1_E = e_array[idx1]
        wp2_N = n_array[idx2]
        wp2_E = e_array[idx2]

        # check whether waypoints should be switched
        if switch_wp(wp1_N=wp1_N, wp1_E=wp1_E, wp2_N=wp2_N, wp2_E=wp2_E, a_N=a_n, a_E=a_e):
            idx1 += 1
            idx2 += 1
            wp1_N = n_array[idx1]
            wp1_E = e_array[idx1]
            wp2_N = n_array[idx2]
            wp2_E = e_array[idx2]
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

def apf_DZN(OS : KVLCC2, 
        TSs : List[KVLCC2], 
        G: dict, 
        dh_clip: float = None,
        du_clip: float = None,
        river_n: np.ndarray=None, 
        river_e: np.ndarray=None, 
        r_min=250, 
        k_a=100, 
        k_r_TS=2.5e6, 
        k_r_river=1e7):
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
        tuple(float, float): Δu, Δheading."""
    
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
        r_safe = r_safe_dyn(a = bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], head0=OS.eta[2], to_2pi=False), 
                            r_min = r_min)
        if d <= r_safe:

            # compute CPA-measure adjustment
            DCPA, TCPA = cpa(NOS=OS.eta[0], EOS=OS.eta[1], NTS=TS.eta[0], ETS=TS.eta[1], chiOS=OS._get_course(),\
                 chiTS=TS._get_course(), VOS=OS._get_V(), VTS=TS._get_V())
            f = k_r_TS_dyn(DCPA=DCPA, TCPA=TCPA)

            F_x += k_r_TS * (1 + f) * (1/r_safe - 1/d) * (x_TS - x_OS) / d
            F_y += k_r_TS * (1 + f) * (1/r_safe - 1/d) * (y_TS - y_OS) / d

    # repulsive forces due to river bounds
    if river_n is not None and river_e is not None:
        for i, n in enumerate(river_n):

            # distance
            x_Riv = river_e[i]
            y_Riv = n
            d = ED(N0=y_OS, E0=x_OS, N1=y_Riv, E1=x_Riv)

            # bearing-dependent safety radius
            r_safe = r_safe_dyn(a=bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=y_Riv, E1=x_Riv, head0=OS.eta[2], to_2pi=False), r_min=r_min)

            if d <= r_safe:
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

class HHOSPlotter:
    """Implements trajectory storing to enable validation plotting for the HHOS project."""
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, [value])

    def store(self, **kwargs):
        for key, value in kwargs.items():
            eval(f"self.{key}.append({value})")

    def dump(self, name):
        # special handling of DepthData, GlobalPath, and ReversedGlobalPath
        info = dict()
        for att in ["DepthData", "GlobalPath", "RevGlobalPath"]:
            if hasattr(self, att):
                info[att] = getattr(self, att)
                delattr(self, att)
        if len(info) > 0:
            with open(f"{name}_info.pkl", "wb") as f:
                pickle.dump(info, f)

        # df creation and storage
        df = pd.DataFrame(vars(self))
        df.replace(to_replace=[None], value=0.0, inplace=True) # clear None
        df.to_pickle(f"{name}.pkl")

class PID_controller:
    """A basic PID control function for rudder angle control of vessels."""
    def __init__(self, Kp:float, Kd:float, Ki:float) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integrator_sum = 0.0

    def control(self, rud_angle0, rud_angle_inc, rud_angle_max, course_error, r):
        # update integral
        self.integrator_sum += course_error

        # compute new rudder angle
        rud_angle_new = self.Kp * course_error + self.Kd * r + self.Ki * self.integrator_sum

        # don't deviate too much from prior one
        rud_angle_new = np.clip(rud_angle_new, rud_angle0-rud_angle_inc, rud_angle0+rud_angle_inc)

        # clip it to the allowed domain
        return np.clip(rud_angle_new, -rud_angle_max, rud_angle_max)

    def reset(self):
        self.integrator_sum = 0.0


class APF_planner_riverOut:
    """
    Specifies an APF planner for rivers. 
    The attractive & repulsive forces build on Liu et al. (2023, Physical Communication)
    and the elliptical condition for the repulsive forces on Wang et al. (2019, Energies,
    https://doi.org/10.3390/en12122342).
    """
    def __init__(self, 
                 dh_clip:float,
                 d_star:float,
                 d_l:float,
                 k_a1:float, 
                 k_a2:float,
                 ell_A:float,      # long end in m
                 ell_A_rev:float,  # long end in m
                 ell_A_sta:float,  # long end in m
                 ell_B:float,      # short end in m
                 ell_B_rev:float,  # short end in m
                 ell_B_sta:float,  # short end in m
                 rho_s:float,
                 k_r1:float,
                 k_r1_rev:float,
                 k_r1_spd:float,
                 k_r2:float,
                 k_r2_rev:float,
                 k_r2_spd:float,
                 k_s:float) -> None:
        self.dh_clip = dh_clip

        # attractive forces params
        self.d_star  = d_star
        self.d_l     = d_l
        self.k_a1    = k_a1
        self.k_a2    = k_a2

        # repulsive forces params
        self.ell_A     = ell_A
        self.ell_A_rev = ell_A_rev
        self.ell_A_sta = ell_A_sta
        self.ell_B     = ell_B
        self.ell_B_rev = ell_B_rev
        self.ell_B_sta = ell_B_sta

        self.rho_s     = rho_s
        self.k_r1      = k_r1
        self.k_r1_rev  = k_r1_rev
        self.k_r1_spd  = k_r1_spd
        self.k_r2      = k_r2
        self.k_r2_rev  = k_r2_rev
        self.k_r2_spd  = k_r2_spd
        self.k_s       = k_s

    def _d_from_ellip(self, ang, rev_dir):
        if rev_dir is None:
            A = self.ell_A_sta
            B = self.ell_B_sta
        elif rev_dir:
            A = self.ell_A_rev
            B = self.ell_B_rev
        else:
            A = self.ell_A
            B = self.ell_B
        return math.sqrt(1/(math.cos(ang)**2/A**2 + math.sin(ang)**2/B**2))

    def plan(self,
             N0:float, E0:float, head0:float, vN0:float, vE0:float,
             N_goal:float, E_goal:float, N_start:float, E_start:float,
             N1:List[float], E1:List[float], head1:List[float], vN1:List[float], vE1:List[float],
             L_dist:List[float], L_n:List[float], L_e:List[float]) -> float:

        # calculate in NM and knots
        N0 = meter_to_NM(N0)
        E0 = meter_to_NM(E0)

        N_goal  = meter_to_NM(N_goal)
        E_goal  = meter_to_NM(E_goal)
        N_start = meter_to_NM(N_start)
        E_start = meter_to_NM(E_start)

        N1 = [meter_to_NM(n) for n in N1]
        E1 = [meter_to_NM(e) for e in E1]
        L_dist = [meter_to_NM(d) for d in L_dist]
        L_n = [meter_to_NM(n) for n in L_n]
        L_e = [meter_to_NM(e) for e in L_e]

        vN0 = mps_to_knots(vN0)
        vE0 = mps_to_knots(vE0)
        v0  = math.sqrt(vN0**2 + vE0**2)
        vN1 = [mps_to_knots(v) for v in vN1]
        vE1 = [mps_to_knots(v) for v in vE1]
        v1 = [math.sqrt(vn**2 + ve**2) for (vn, ve) in zip(vN1, vE1)]

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

        if dg < self.d_star:
            F = dg
        else:
            F = self.d_star

        print(f"F_att1: {F* self.k_a1}")

        E_add, N_add = F * self.k_a1 * n_sg
        F_att_E += E_add
        F_att_N += N_add

        # attractive force to planned path
        CTE = cte(N1=N_start, E1=E_start, N2=N_goal, E2=E_goal, NA=N0, EA=E0)
        
        if abs(CTE) > self.d_l:

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
            print(f"F_att2: {F * self.k_a2}")
            
            E_add, N_add = F * self.k_a2 * n_SG
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
            n_os = -n_so          # unit vector from TS to OS (obstacle to ship)

            # ------------ dynamic obstacles ----------------
            if not np.isclose(v1[i], 0.0):

                # perpendicular vectors for starboard or portside CA
                r, angle = polar_from_xy(x=n_so[0], y=n_so[1])
                e, n = xy_from_polar(r=r, angle=angle_to_2pi(angle + dtr(90.0)))
                n_so_right = np.array([e, n])

                e, n = xy_from_polar(r=r, angle=angle_to_2pi(angle - dtr(90.0)))
                n_so_left = np.array([e, n])

                # --- get rho_0 based on ellipse ---
                # check whether TS has reversed direction
                rev_dir = (abs(head_inter(head_OS=head0, head_TS=head1[i], to_2pi=False)) >= dtr(90.0))

                # we need relative bearing from perspective of the target ship
                ang = bng_rel(N0=N1[i], E0=E1[i], N1=N0, E1=E0, head0=head1[i])
                rho_0 = meter_to_NM(self._d_from_ellip(ang=ang, rev_dir=rev_dir))

                if (d <= rho_0) and np.dot(V_UR, n_so) > 0:

                    if rev_dir:
                        k_r1 = self.k_r1_rev
                        k_r2 = self.k_r2_rev
                    elif v1[i] > v0:
                        k_r1 = self.k_r1_spd
                        k_r2 = self.k_r2_spd
                    else:
                        k_r1 = self.k_r1
                        k_r2 = self.k_r2

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

                    # relative velocity-based force to steer starboard/portside with relation to the target ship
                    theta = get_theta(p_OS=p_OS, v_OS=v_OS, p_TS=p_TS, v_TS=v_TS)
                    F3 = V_UR_norm * np.sin(theta)

                    #TCPA = tcpa(NOS=N0, EOS=E0, NTS=N1[i], ETS=E1[i], chiOS=head0, chiTS=head1[i],
                    #            VOS=v0, VTS=v1[i])
                    #if TCPA > 0:
                    #    k_r2 = self.k_r2
                    #else:
                    #    k_r2 = 0.0

                    if rev_dir:
                        n_so_perp = n_so_right
                    else:
                        n_so_perp = n_so_left

                    E_add, N_add = F3 * k_r2 * n_so_perp
                    F_rep_E += E_add
                    F_rep_N += N_add

                    # fixed force from the target ship to the own ship
                    F4 = 1.0
                    E_add, N_add = F4 * k_r2 * n_os
                    F_rep_E += E_add
                    F_rep_N += N_add

                    print(f"F1: {F1 * k_r1}, F2: {F2 * k_r1}, F3: {F3 * k_r2}, F4: {F4 * k_r2} \n")
                    print("----------------")

            # ------------ static obstacles ----------------
            else:
                # we need relative bearing from perspective of the static obstacle
                ang = bng_rel(N0=N1[i], E0=E1[i], N1=N0, E1=E0, head0=head1[i])
                rho_0 = meter_to_NM(self._d_from_ellip(ang=ang, rev_dir=None))

                # assume circle shaped obstacle
                if d <= rho_0:

                    # distance-based force from the obstacle to the own ship 
                    F1 = (1/d - 1/rho_0) / d**2
                    E_add, N_add = F1 * self.k_s * n_os
                    F_rep_E += E_add
                    F_rep_N += N_add

        # aggregate
        F_x = F_att_E + F_rep_E
        F_y = F_att_N + F_rep_N

        # translate into Δheading
        dh = angle_to_pi(math.atan2(F_x, F_y) - head0)
        return np.clip(dh, -self.dh_clip, self.dh_clip)


class APF_planner_river:
    """
    Specifies an APF planner for rivers. 
    The attractive & repulsive forces build on Liu et al. (2023, Physical Communication)
    and the elliptical condition for the repulsive forces on Wang et al. (2019, Energies,
    https://doi.org/10.3390/en12122342).
    """
    def __init__(self, 
                 dh_clip:float,
                 d_star:float,
                 d_l:float,
                 k_a1:float, 
                 k_a2:float,
                 ell_A:float,      # long end in m
                 ell_B:float,      # short end in m
                 k_r1:float,
                 k_r2:float) -> None:
        self.dh_clip = dh_clip

        # attractive forces params
        self.d_star  = d_star
        self.d_l     = d_l
        self.k_a1    = k_a1
        self.k_a2    = k_a2

        # repulsive forces params
        self.ell_A     = ell_A
        self.ell_B     = ell_B
        self.rho_obs   = max(ell_A, ell_B)
        self.k_r1      = k_r1
        self.k_r2      = k_r2

    def _d_from_ellip(self, ang):
        A = self.ell_A
        B = self.ell_B
        return math.sqrt(1/(math.cos(ang)**2/A**2 + math.sin(ang)**2/B**2))

    def plan(self,
             N0:float, E0:float, head0:float, vN0:float, vE0:float, chiOS:float,
             N_goal:float, E_goal:float, N_start:float, E_start:float,
             N1:List[float], E1:List[float], head1:List[float], vN1:List[float], vE1:List[float],
             chiTSs:List[float], L_dist:List[float], L_n:List[float], L_e:List[float]) -> float:
        # compute tcpas
        tcpas = []
        for i in range(len(N1)):
            tcpas.append(tcpa(NOS=N0, EOS=E0, NTS=N1[i], ETS=E1[i], chiOS=chiOS, chiTS=chiTSs[i],
                              VOS=math.sqrt(vN0**2 + vE0**2), VTS=math.sqrt(vN1[i]**2 + vE1[i]**2)))

        # calculate in NM and knots
        N0 = meter_to_NM(N0)
        E0 = meter_to_NM(E0)

        N_goal  = meter_to_NM(N_goal)
        E_goal  = meter_to_NM(E_goal)
        N_start = meter_to_NM(N_start)
        E_start = meter_to_NM(E_start)

        N1 = [meter_to_NM(n) for n in N1]
        E1 = [meter_to_NM(e) for e in E1]
        L_dist = [meter_to_NM(d) for d in L_dist]
        L_n = [meter_to_NM(n) for n in L_n]
        L_e = [meter_to_NM(e) for e in L_e]

        vN0 = mps_to_knots(vN0)
        vE0 = mps_to_knots(vE0)
        v0  = math.sqrt(vN0**2 + vE0**2)
        vN1 = [mps_to_knots(v) for v in vN1]
        vE1 = [mps_to_knots(v) for v in vE1]
        v1 = [math.sqrt(vn**2 + ve**2) for (vn, ve) in zip(vN1, vE1)]

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

        if dg < self.d_star:
            F = dg
        else:
            F = self.d_star

        print(f"F_att1: {F* self.k_a1}")

        E_add, N_add = F * self.k_a1 * n_sg
        F_att_E += E_add
        F_att_N += N_add

        # attractive force to planned path
        CTE = cte(N1=N_start, E1=E_start, N2=N_goal, E2=E_goal, NA=N0, EA=E0)
        
        if abs(CTE) > self.d_l:

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
            print(f"F_att2: {F * self.k_a2}")
            
            E_add, N_add = F * self.k_a2 * n_SG
            F_att_E += E_add
            F_att_N += N_add

        # ------------- repulsive forces ------------
        F_rep_E = 0.0
        F_rep_N = 0.0

        for i in range(len(N1)):

            # ignore past vessels!
            if tcpas[i] < 0:
                continue

            # quick access
            p_TS = np.array([E1[i], N1[i]])

            # compute some metrics
            d = norm(p_TS-p_OS)
            n_so = (p_TS-p_OS)/d  # unit vector from OS to TS (ship to obstacle)
            n_os = -n_so          # unit vector from TS to OS (obstacle to ship)

            # ------------ dynamic obstacles ----------------
            # perpendicular vectors for starboard or portside CA
            r, angle = polar_from_xy(x=n_so[0], y=n_so[1])
            e, n = xy_from_polar(r=r, angle=angle_to_2pi(angle + dtr(90.0)))
            n_so_right = np.array([e, n])

            e, n = xy_from_polar(r=r, angle=angle_to_2pi(angle - dtr(90.0)))
            n_so_left = np.array([e, n])

            # --- get d_ell based on ellipse ---
            # we need relative bearing from perspective of the target ship
            ang = bng_rel(N0=N1[i], E0=E1[i], N1=N0, E1=E0, head0=head1[i])
            d_ell = self._d_from_ellip(ang=ang)

            if (d <= d_ell):

                k_r1 = self.k_r1
                k_r2 = self.k_r2

                # F_rep1: distance-based force from the target ship to the own ship 
                F1 = (1/d - 1/self.rho_obs) / d**2
                E_add, N_add = F1 * k_r1 * n_os
                F_rep_E += E_add
                F_rep_N += N_add

                # F_rep2: constant force to steer starboard/portside with relation to the target ship
                F2 = 1

                # check whether TS has reversed direction
                rev_dir = (abs(head_inter(head_OS=head0, head_TS=head1[i], to_2pi=False)) >= dtr(90.0))

                if rev_dir:
                    n_so_perp = np.array([0., 0.]) # n_so_right
                else:
                    n_so_perp = n_so_left

                E_add, N_add = F2 * k_r2 * n_so_perp
                F_rep_E += E_add
                F_rep_N += N_add

                print(f"F1: {F1 * k_r1}, F2: {F2 * k_r2}\n")
                print("----------------")

        # aggregate
        F_x = F_att_E + F_rep_E
        F_y = F_att_N + F_rep_N

        # translate into Δheading
        dh = angle_to_pi(math.atan2(F_x, F_y) - head0)
        return np.clip(dh, -self.dh_clip, self.dh_clip)
