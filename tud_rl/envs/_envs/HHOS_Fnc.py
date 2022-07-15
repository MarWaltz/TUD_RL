import math

import numpy as np
import utm


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
    Returns (idx, entry)."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], int(idx)
