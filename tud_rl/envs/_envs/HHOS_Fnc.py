import math

import numpy as np
import utm
from tud_rl.envs._envs.VesselFnc import bng_abs

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


def desired_course_VFG(N1, E1, N2, E2, NA, EA, K):
    """Computes the desired course based on the vector field guidance method following pp.354-55 of Fossen (2021).
    The waypoints are (N1, E1) and (N2, E2), while the agent is at (NA, EA). K is the convergence rate of the vector field."""

    assert K > 0, "K of VFG must be larger zero."

    # get path angle
    pi_path = bng_abs(N0=N1, E0=E1, N1=N2, E1=E2)

    # get CTE
    ye = cte(N1=N1, E1=E1, N2=N2, E2=E2, NA=NA, EA=EA, pi_path=pi_path)

    # get desired course, which is rotated back to NED
    return pi_path - math.atan(ye * K)
