import math
import matplotlib.pyplot as plt
import numpy as np

COLREG_NAMES  = {0 : "Null", 
                 1 : "Head-on", 
                 2 : "Starb. cross.", 
                 3 : "Ports. cross.", 
                 4 : "Overtaking"}
COLREG_COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(5)]


#------------------- Helper functions with angles, mainly following Benjamin (2017) ----------------

def dtr(angle):
    """Takes angle in degree an transforms it to radiant."""
    return angle * np.pi / 180


def rtd(angle):
    """Takes angle in degree an transforms it to radiant."""
    return angle * 180 / np.pi


def angle_to_2pi(angle):
    """Transforms an angle to [0, 2pi)."""
    if angle >= 0:
        return angle - np.floor(angle / (2*np.pi)) * 2*np.pi
    else:
        return angle + (np.floor(-angle / (2*np.pi)) + 1) * 2*np.pi


def angle_to_pi(angle):
    """Transforms an angle to [-pi, pi)."""
    if angle >= 0:
        return angle - np.floor((angle + np.pi) / (2*np.pi)) * 2*np.pi
    else:
        return angle + np.floor((-angle + np.pi)  / (2*np.pi)) * 2*np.pi


def head_inter(head_OS, head_TS):
    """Computes the intersection angle between headings in radiant (in [0, 2pi)). Corresponds to C_T in Xu et al. (2022, Neurocomputing)."""
    return angle_to_2pi(head_TS - head_OS)


def ED(N0, E0, N1, E1, sqrt=True):
    """Computes the euclidean distance between two points."""
    d_sq = (N0 - N1)**2 + (E0 - E1)**2

    if sqrt:
        return math.sqrt(d_sq)
    return d_sq


def polar_from_xy(x, y, with_r=True, with_angle=True):
    """Get polar coordinates (r, angle in rad in [0, 2pi)) from x,y-coordinates. Angles are defined clockwise with zero at the y-axis.
    Args:
        with_r (bool):     Whether to compute the radius.
        with_angle (bool): Whether to compute the angle.
    Returns:
        r, angle as a tuple of floats."""

    #------------ radius ---------------
    r = math.sqrt(x**2 + y**2) if with_r else None
    angle = angle_to_2pi(math.atan2(x, y)) if with_angle else None
    return r, angle


def xy_from_polar(r, angle):
    """Get x,y-coordinates from polar system, where angle is defined clockwise with zero at the y-axis."""
    return r * math.sin(angle), r * math.cos(angle)


def bng_abs(N0, E0, N1, E1):
    """Computes the absolute bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0)."""
    return polar_from_xy(x=E1-E0, y=N1-N0, with_r=False, with_angle=True)[1]


def bng_rel(N0, E0, N1, E1, head0):
    """Computes the relative bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0) and heading head0."""
    return angle_to_2pi(bng_abs(N0, E0, N1, E1) - head0)


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

    return - nom / den


def cpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Returns DCPA and TCPA. Follows Chun et al. (2021, OE)."""

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


def NM_to_meter(NM):
    """Convertes nautical miles in meter."""
    return NM * 1852


def meter_to_NM(meter):
    """Convertes meter in nautical miles."""
    return meter / 1852
