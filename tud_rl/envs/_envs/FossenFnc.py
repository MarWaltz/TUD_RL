from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

COLREG_NAMES  = {0 : "Null", 
                 1 : "Head-on", 
                 2 : "Starb. cross. (small)", 
                 3 : "Starb. cross. (large)",
                 4 : "Ports. cross.", 
                 5 : "Overtaking"}
COLREG_COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(6)]


class StaticObstacle:
    """A static circle-shaped obstacle."""
    
    def __init__(self, N_init, E_init, max_radius) -> None:
        
        # spawning point
        self.N = N_init
        self.E = E_init

        # size
        self.radius = np.random.uniform(1, max_radius)
        self.radius_norm = self.radius / max_radius


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
        return np.sqrt(d_sq)
    return d_sq


def polar_from_xy(x, y, with_r=True, with_angle=True):
    """Get polar coordinates (r, angle in rad in [0, 2pi)) from x,y-coordinates. Angles are defined clockwise with zero at the y-axis.
    Args:
        with_r (bool):     Whether to compute the radius.
        with_angle (bool): Whether to compute the angle.
    Returns:
        r, angle as a tuple of floats."""

    #------------ radius ---------------
    if with_r:
        r = sqrt(x**2 + y**2)
    else:
        r = None

    #------------ angle ---------------
    if with_angle:
        # zero cases
        if x == 0 and y >= 0:
            angle = 0
        
        elif x == 0 and y < 0:
            angle = np.pi

        elif x >= 0 and y == 0:
            angle = np.pi/2
        
        elif x < 0 and y == 0:
            angle = 3/2 * np.pi

        else:
            frac = np.arctan(np.abs(x / y))

            # I. Q.
            if x > 0 and y > 0:
                angle = frac

            # II. Q.
            elif x < 0 and y > 0:
                angle = 2*np.pi - frac
            
            # III. Q.
            elif x < 0 and y < 0:
                angle = frac + np.pi
            
            # IV. Q.
            elif x > 0 and y < 0:
                angle = np.pi - frac
    else:
        angle = None
    
    return r, angle


def xy_from_polar(r, angle):
    """Get x,y-coordinates from polar system, where angle is defined clockwise with zero at the y-axis."""
    return r * np.sin(angle), r * np.cos(angle)


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


def dcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes distance of closest point of approach. Follows Chun et al. (2021, OE)."""

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

    return ED(N0=yOS_tcpa, E0=xOS_tcpa, N1=yTS_tcpa, E1=xTS_tcpa)


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


#-------------------------------------- Backup ----------------------------------------
def range_rate(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes the rate at which the range (ED) of two vehicles is changing."""

    beta = bng_rel(N0=NOS, E0=EOS, N1=NTS, E1=ETS, head0=chiOS)
    alpha = bng_rel(N0=NTS, E0=ETS, N1=NOS, E1=EOS, head0=chiTS)

    return np.cos(alpha) * VOS + np.cos(beta) * VTS


def tcpa_benjamin(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes the time to closest point of approach. If 0, the CPA has already been past. Follows Benjamin (2017)."""
    
    rdot = range_rate(NOS=NOS, EOS=EOS, NTS=NTS, ETS=ETS, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)

    if rdot >= 0:
        return 0.0
    else:
        # easy access
        xOS = EOS
        yOS = NOS
        xTS = ETS
        yTS = NTS

        k1 = 2 * np.cos(chiOS) * VOS * yOS - 2 * np.cos(chiOS) * VOS * yTS - 2 * yOS * np.cos(chiTS) * VTS \
            + 2 * np.cos(chiTS) * VTS * yTS + 2 * np.sin(chiOS) * VOS * xOS - 2 * np.sin(chiOS) * VOS * xTS \
                - 2 * xOS * np.sin(chiTS) * VTS + 2 * np.sin(chiTS) * VTS * xTS
        
        k2 = np.cos(chiOS)**2 * VOS**2 - 2 * np.cos(chiOS) * VOS * np.cos(chiTS) * VTS + np.cos(chiTS)**2 * VTS**2 \
            + np.sin(chiOS)**2 * VOS**2 - 2 * np.sin(chiOS) * VOS * np.sin(chiTS) * VTS + np.sin(chiTS)**2 * VTS**2
        
        return - k1/k2
