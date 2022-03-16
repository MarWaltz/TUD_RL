import matplotlib.pyplot as plt
import numpy as np

COLREG_NAMES  = {1 : "Head-on", 2 : "Starb. cross.", 3 : "Ports. cross.", 4 : "Overtaking", 0 : "Null"}
COLREG_COLORS = [plt.rcParams["axes.prop_cycle"].by_key()["color"][i] for i in range(5)]


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
    """Computes the intersection angle between headings in radiant. Corresponds to C_T in Xu et al. (2022, Neurocomputing)."""
    return angle_to_2pi(head_TS - head_OS)


def ED(N0, E0, N1, E1, sqrt=True):
    """Computes the euclidean distance between two points."""
    d_sq = (N0 - N1)**2 + (E0 - E1)**2

    if sqrt:
        return np.sqrt(d_sq)
    return d_sq


def bng_abs(N0, E0, N1, E1):
    """Computes the absolute bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0)."""
    
    # rename to x,y for convenience
    xOS = E0
    yOS = N0
    xCN = E1
    yCN = N1

    if xOS == xCN and yOS <= yCN:
        return 0
    
    elif xOS == xCN and yOS > yCN:
        return np.pi

    elif xOS <= xCN and yOS == yCN:
        return np.pi/2
    
    elif xOS > xCN and yOS == yCN:
        return 3/2 * np.pi
   
    elif xOS < xCN and yOS < yCN:  # I. Q.
        return np.arctan(np.abs(xOS - xCN) / np.abs(yOS - yCN))
       
    elif xOS > xCN and yOS < yCN:  # II. Q.
        return 2 * np.pi - np.arctan(np.abs(xOS - xCN) / np.abs(yOS - yCN))
    
    elif xOS > xCN and yOS > yCN:   # III. Q.
        return np.pi + np.arctan(np.abs(xOS - xCN) / np.abs(yOS - yCN))

    elif xOS < xCN and yOS > yCN:   # IV. Q.
        return np.pi - np.arctan(np.abs(xOS - xCN) / np.abs(yOS - yCN))


def bng_rel(N0, E0, N1, E1, head0):
    """Computes the relative bearing (in radiant, [0, 2pi)) of (N1, E1) from perspective of (N0, E0) and heading head0."""
    return angle_to_2pi(bng_abs(N0, E0, N1, E1) - head0)


def tcpa(NOS, EOS, NTS, ETS, chiOS, chiTS, VOS, VTS):
    """Computes the time to closest point of approach (TCPA). Follows Lenart (2017)."""

    # easy access
    xOS = EOS
    yOS = NOS
    xTS = ETS
    yTS = NTS

    # compute velocities in x,y-coordinates
    vxOS = VOS * np.sin(chiOS)
    vyOS = VOS * np.cos(chiOS)
    vxTS = VTS * np.sin(chiTS)
    vyTS = VTS * np.cos(chiTS)

    nom = - ((yTS - yOS)*(vyTS - vyOS) + (xTS - xOS)*(vxTS - vxOS))
    den = (vyTS - vyOS)**2 + (vxTS - vxOS)**2

    return nom / den


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
