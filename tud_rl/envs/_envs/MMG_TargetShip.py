import math
from typing import List, Union

import numpy as np

from tud_rl.envs._envs.HHOS_Fnc import (VFG, apf_DZN, prep_angles_for_average,
                                        to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (ED, NM_to_meter, angle_to_2pi,
                                         bng_abs, bng_rel, cpa, dtr, rtd,
                                         xy_from_polar)


class TargetShip(KVLCC2):
    """This class provides a target ship based on the dynamics of the KVLCC2 tanker."""
    def __init__(self, N_init, E_init, psi_init, u_init, v_init, r_init, nps, delta_t, N_max, E_max, full_ship, ship_domain_size) -> None:
        super().__init__(N_init, E_init, psi_init, u_init, v_init, r_init, nps, delta_t, N_max, E_max, full_ship, ship_domain_size)
        self.non_cooperative = np.random.choice([True, False], p=[0.20, 0.80])
        self.APF_moves = False

    def _is_overtaking(self, other_vessel : KVLCC2, role : str):
        """Checks whether a vessel overtakes an other one.
        Args:
            role(str): Either 'get_overtaken' or 'is_overtaking'. If 'get_overtaken', the function checks whether self gets 
                       overtaken by other_vessel, and vice verse if 'is_overtaking'.
        Returns:
            bool."""

        assert role in ["gets_overtaken", "is_overtaking"], "Unknown role in overtaking scenario."

        # vessel1 overtakes vessel2
        if role == "gets_overtaken":
            vessel1 = other_vessel
            vessel2 = self
        else:
            vessel1 = self
            vessel2 = other_vessel

        dist = ED(N0=vessel1.eta[0], E0=vessel1.eta[1], N1=vessel2.eta[0], E1=vessel2.eta[1])
        if (vessel1._get_V() > vessel2._get_V()) and (dist <= 10*max([vessel1.Lpp, vessel2.Lpp])):
            if 135 <= rtd(bng_rel(N0=vessel2.eta[0], E0=vessel2.eta[1], N1=vessel1.eta[0], E1=vessel1.eta[1], head0=vessel2.eta[2])) <= 315:
                return True
        return False

    def _is_opposing(self, other_vessel : KVLCC2):
        """Checks whether the other vessel is opposing to the target ship."""
        dist = ED(N0=other_vessel.eta[0], E0=other_vessel.eta[1], N1=self.eta[0], E1=self.eta[1])
        if (other_vessel.rev_dir == self.rev_dir) or (dist > 10*max([other_vessel.Lpp, self.Lpp])):
            return False

        DCPA, TCPA = cpa(NOS=self.eta[0], EOS=self.eta[1], NTS=other_vessel.eta[0], ETS=other_vessel.eta[1],\
             chiOS=self._get_course(), chiTS=other_vessel._get_course(), VOS=self._get_V(), VTS=other_vessel._get_V())
        
        if (TCPA > 0.0) and (DCPA < 2*max([other_vessel.B, self.B])):
            return True
        return False

    def opensea_control(self, other_vessels : List[KVLCC2]):
        """Defines target ship behavior for vessels on open sea."""
        if self.APF_moves:
            _, dh = apf(OS=self, TSs=other_vessels, G={"x":self.path.east[100], "y":self.path.north[100]},
                        dh_clip=dtr(10.0), r_min=NM_to_meter(1.0), k_r_TS=2.5e10)
            self.eta[2] = angle_to_2pi(self.eta[2] + dh)

        elif self.random_moves:
            self.eta[2] = angle_to_2pi(self.eta[2] + dtr(float(np.random.uniform(-5.0, 5.0, size=1))))

    def river_control(self, other_vessels : List[KVLCC2], VFG_K : float):
        """Defines a deterministic rule-based controller for target ships on rivers."""
        # rare case that we move from open sea back to river
        if not hasattr(self, "rev_dir") or any([not hasattr(vess, "rev_dir") for vess in other_vessels]):
            return

        # easy access
        ye, dc, _, smoothed_path_ang = VFG(N1 = self.glo_wp1_N, 
                                            E1 = self.glo_wp1_E, 
                                            N2 = self.glo_wp2_N, 
                                            E2 = self.glo_wp2_E,
                                            NA = self.eta[0], 
                                            EA = self.eta[1], 
                                            K  = VFG_K, 
                                            N3 = self.glo_wp3_N, 
                                            E3 = self.glo_wp3_E)

        # non-cooperative vessels only run VFG
        if self.non_cooperative:
            self.eta[2] = dc

        # otherwise consider others
        elif len(other_vessels) > 0:

            # consider vessel with smallest euclidean distance
            EDs = [ED(N0=self.eta[0], E0=self.eta[1], N1=other_vessel.eta[0], E1=other_vessel.eta[1]) for other_vessel in other_vessels]
            i = EDs.index(min(EDs))
            other_vessel = other_vessels[i]

            # opposing traffic is a threat
            opposing_traffic = self._is_opposing(other_vessel)
            if opposing_traffic:
                self.eta[2] = angle_to_2pi(dc + dtr(5.0))
            else:
                # vessel gets overtaken by other ship
                #gets_overtaken = self._is_overtaking(other_vessel=other_vessel, role="gets_overtaken")
                #if gets_overtaken:
                #    ang = smoothed_path_ang if ye > 0.0 else dc
                #    delta = self._control_hlp(ye=ye, x1=5*self.B, role="gets_overtaken")
                #    self.eta[2] = angle_to_2pi(ang + dtr(delta))
                #else:

                # vessel is overtaking someone else
                is_overtaking = self._is_overtaking(other_vessel=other_vessel, role="is_overtaking")
                if is_overtaking:
                    ang = dc if ye > 0.0 else smoothed_path_ang
                    delta = self._control_hlp(ye=ye, x1=5*self.B, role="is_overtaking")
                    self.eta[2] = angle_to_2pi(ang - dtr(delta))
                else:
                    # otherwise we just use basic VFG control
                    self.eta[2] = dc
        else:
            self.eta[2] = dc

    def _control_hlp(self, role, ye, x1):
        """Rule-based control helper fnc to determine the heading adjustment."""
        assert role in ["gets_overtaken", "is_overtaking"], "Unknown scenario for rule-based heading control."
        
        if role == "gets_overtaken":
            y1 = 1
            y2 = 8
            if ye < 0.0:
                return y2
            return y2 * math.exp(ye/x1 * math.log(y1/y2))
        else:
            y1 = 2
            y2 = 8
            if ye > 0.0:
                return y2
            return y2 * math.exp(ye/x1 * math.log(y2/y1))


class Path:
    """Defines a path in the HHOS-project."""
    def __init__(self, level, lat=None, lon=None, n_wps=None, north=None, east=None, vs=None, **kwargs) -> None:
        # checks
        assert level in ["global", "local"], "Path is either 'global' or 'local'."
        assert not ((lat is None and lon is None) and (north is None and east is None)), "Need some data to construct the path."

        # UTM and lat-lon
        if (lat is None) or (lon is None):
            self._store(north, "north")
            self._store(east, "east")
            self.lat, self.lon = to_latlon(north=self.north, east=self.east, number=32)
        else:
            self._store(lat, "lat")
            self._store(lon, "lon")

        # UTM coordinates
        if (north is None) or (east is None):
            self.north, self.east, _ = to_utm(lat=self.lat, lon=self.lon)
        else:
            self._store(north, "north")
            self._store(east, "east")

        # velocity
        if vs is not None:
            self._store(vs, "vs")

        # number of waypoints
        if n_wps is None:
            self.n_wps = len(self.lat)
        else:
            assert n_wps == len(self.lat), "Number of waypoints do not match given data."
            self.n_wps = n_wps

        # optionally store kwarg information
        for key, value in kwargs.items():
            self._store(value, key)

    def _store(self, data:Union[list,np.ndarray], name:str):
        """Stores data by transforming to np.ndarray."""
        if not isinstance(data, np.ndarray):
            setattr(self, name, np.array(data))
        else:
            setattr(self, name, data)

    def reverse(self, offset:float):
        """Reverses the path based on a constant offset."""
        rev_north = np.zeros_like(self.north)
        rev_east = np.zeros_like(self.east)

        flipped_north = np.flip(self.north)
        flipped_east = np.flip(self.east)

        for i in range(self.n_wps):
            n = flipped_north[i]
            e = flipped_east[i]

            if i != (self.n_wps-1):
                n_nxt = flipped_north[i+1]
                e_nxt = flipped_east[i+1]
                ang = angle_to_2pi(bng_abs(N0=n, E0=e, N1=n_nxt, E1=e_nxt) + math.pi/2)
            else:
                n_last = flipped_north[i-1]
                e_last = flipped_east[i-1]
                ang = angle_to_2pi(bng_abs(N0=n_last, E0=e_last, N1=n, E1=e) + math.pi/2)

            e_add, n_add = xy_from_polar(r=offset, angle=ang)
            rev_north[i] = n + n_add
            rev_east[i] = e + e_add

        # store
        self.north = rev_north
        self.east = rev_east

        # lat-lon
        self.lat, self.lon = to_latlon(north=self.north, east=self.east, number=32)

    def wp_dist(self, wp1_idx, wp2_idx):
        """Computes the euclidean distance between two waypoints."""
        if wp1_idx not in range(self.n_wps) or wp2_idx not in range(self.n_wps):
            raise ValueError("Your path index is out of order. Please check your sampling strategy.")

        return ED(N0=self.north[wp1_idx], E0=self.east[wp1_idx], N1=self.north[wp2_idx], E1=self.east[wp2_idx])

    def get_rev_path_wps(self, wp1_idx, wp2_idx):
        """Computes the waypoints from the reversed version of the path.
        Returns: wp1_rev_idx, wp2_rev_idx."""
        wp1_rev = self.n_wps - (wp2_idx+1)
        wp2_rev = self.n_wps - (wp1_idx+1)

        return wp1_rev, wp2_rev

    def construct_local_path(self, wp_idx:int, n_wps_loc:int, v_OS:float=None):
        """Constructs a local path based on a given wp_idx and number of waypoints.
        Args:
            wp_idx(int):       Waypoint index where the local path starts
            n_wps_loc(int):    Length of the constructed path
            v_OS(float):       velocity of OS, assumed constant over the path since we do not consider trajectory following
        Returns:
            Path"""
        # make sure local path is not longer than global one
        if wp_idx + n_wps_loc >= self.n_wps:
            n_wps_loc = self.n_wps - wp_idx

        # select
        lat   = self.lat[wp_idx:wp_idx+n_wps_loc]
        lon   = self.lon[wp_idx:wp_idx+n_wps_loc]
        north = self.north[wp_idx:wp_idx+n_wps_loc]
        east  = self.east[wp_idx:wp_idx+n_wps_loc]

        # compute heading
        heads = np.zeros_like(north)
        for i in range(n_wps_loc):
            if i != (n_wps_loc-1):
                heads[i] = bng_abs(N0=north[i], E0=east[i], N1=north[i+1], E1=east[i+1])
            else:
                heads[i] = heads[i-1]

        # speed
        if v_OS is not None:
            vs = np.ones_like(lat) * v_OS
        else:
            vs = None

        # construct while setting course to heading
        return Path(level="local", lat=lat, lon=lon, n_wps=n_wps_loc, north=north, east=east, heads=heads, chis=heads, vs=vs)

    def interpolate(self, attribute:str, n_wps_between:int, angle:bool=False):
        """Linearly interpolates a specific attribute to have additional waypoints. 
        Careful: We DO NOT ALTER the attribute self.n_wps (!)
        
        Args:
            attribute(str):     attribute of interest, e.g., north
            n_wps_between(int): number of waypoints to add in each interval.
            angle(bool):        whether the attribute of interest is an angle, e.g., the heading, 
                                since the interpolation needs to consider this
        Example: 
            When before the function call we have self.north = [0, 1, 2], and call interpolate(attribute='north', n_wps_between=3),
            we will have self.north = [0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0] afterward."""

        assert attribute != "rev_dir", "Cannot interpolate attribute 'rev_dir'; it's binary."
        
        # preparation
        att = getattr(self, attribute)
        old_len = len(att)
        new_len = old_len + (old_len-1)*n_wps_between

        # take existing values
        new_att = np.zeros(new_len)
        mods = (np.arange(new_len) % (n_wps_between+1))  # modulos
        ind = (mods == 0)     # indices of old values in new array
        new_att[ind] = att

        # interpolate others
        old_i = -1
        for i in range(len(new_att)):
            if ind[i]:
                old_i += 1
            else:
                # access
                frac = mods[i] / (n_wps_between+1)
                last_val = att[old_i]
                next_val = att[old_i+1]

                # consider boundary issues for angles
                if angle:
                    last_val, next_val = prep_angles_for_average(last_val, next_val)
 
                # average
                new_att[i] = frac * next_val + (1-frac) * last_val
                
                # [0,2pi) for angles
                if angle:
                    new_att[i] = angle_to_2pi(new_att[i])
        # store
        setattr(self, attribute, new_att)
