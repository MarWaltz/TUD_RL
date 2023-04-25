import math
import pickle
import random
from copy import copy

import gym
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from gym import spaces
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from tud_rl.envs._envs.HHOS_Fnc import (VFG, Z_at_latlon, bng_abs, cte,
                                        find_nearest, get_init_two_wp,
                                        mps_to_knots, to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.MMG_TargetShip import Path, TargetShip
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                         NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_rel, dtr,
                                         get_ship_domain, head_inter,
                                         polar_from_xy, project_vector, rtd,
                                         tcpa, xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Base_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 vessel from Hamburg to Oslo."""
    def __init__(self):
        super().__init__()

        # simulation settings
        self.delta_t = 5.0     # simulation time interval (in s)
        self.base_speed = 3.0  # default surge velocity (in m/s)

        # range definitions
        self.sight_open  = NM_to_meter(5.0)     # sight on open waters
        self.sight_river = NM_to_meter(0.5)     # sight on river

        self.open_enc_range      = NM_to_meter(5.0)     # distance when we consider encounter situations on open waters
        self.river_enc_range_min = NM_to_meter(0.25)    # lower distance when we consider encounter situations on the river
        self.river_enc_range_max = NM_to_meter(0.50)    # upper distance when we consider encounter situations on the river

        # data range
        self.lon_lims = [4.83, 14.33]
        self.lat_lims = [51.83, 60.5]
        self.lon_range = self.lon_lims[1] - self.lon_lims[0]
        self.lat_range = self.lat_lims[1] - self.lat_lims[0]

        # LiDAR
        self.lidar_range       = NM_to_meter(1.0)   # range of LiDAR sensoring in m
        self.lidar_beam_angles = [20.0, 45., 90., 135.0]
        self.lidar_beam_angles += [360. - ang for ang in self.lidar_beam_angles] + [0.0, 180.0]
        self.lidar_beam_angles = np.deg2rad(np.sort(self.lidar_beam_angles))

        self.lidar_n_beams   = len(self.lidar_beam_angles)
        self.n_dots_per_beam = 50          # number of subpoints per beam
        self.d_dots_per_beam = (np.logspace(0.01, 1, self.n_dots_per_beam, endpoint=True)-1) /9 * self.lidar_range  # distances from midship of subpoints per beam

        # initially compute ship domain for plotting
        rads  = np.linspace(0.0, 2*math.pi, 100)
        Lpp   = 64
        width = 11.6

        A = 2 * Lpp + 0.5 * Lpp
        B = 2 * width + 0.5 * width
        C = 2 * width + 0.5 * Lpp
        D = 2 * width + 0.5 * width #4 * width + 0.5 * width

        dists = [get_ship_domain(A=A, B=B, C=C, D=D, OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        # visualization
        self.plot_in_latlon = True         # if False, plots in UTM coordinates
        self.plot_depth     = True
        self.plot_path      = True
        self.plot_wind      = False
        self.plot_current   = False
        self.plot_waves     = False
        self.plot_lidar     = True
        self.plot_reward    = True
        self.default_cols   = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        self.first_init     = True

        if not self.plot_in_latlon:
            self.show_lon_lat    = np.clip(self.show_lon_lat, 0.005, 5.95)
            self.UTM_viz_range_E = abs(to_utm(lat=52.0, lon=6.0001)[1] - to_utm(lat=52.0, lon=6.0001+self.show_lon_lat/2)[1])
            self.UTM_viz_range_N = abs(to_utm(lat=50.0, lon=8.0)[0] - to_utm(lat=50.0+self.show_lon_lat/2, lon=8.0)[0])

    def _load_global_path(self):
        with open(self.global_path_file, "rb") as f:
            GlobalPath = pickle.load(f)
        self.GlobalPath = Path(level="global", **GlobalPath)

    def _load_depth_data(self):
        with open(self.depth_data_file, "rb") as f:
            self.DepthData = pickle.load(f)

        # logarithm
        Depth_tmp = copy(self.DepthData["data"])
        Depth_tmp[Depth_tmp < 1] = 1
        self.log_Depth = np.log(Depth_tmp)

        # for contour plot
        self.con_ticks = np.log([1.0, 2.0, 5.0, 15.0, 50.0, 150.0, 500.0])
        self.con_ticklabels = [int(np.round(tick, 0)) for tick in np.exp(self.con_ticks)]
        self.con_ticklabels[0] = 0
        self.clev = np.arange(0, self.log_Depth.max(), .1)

    def _load_wind_data(self):
        with open(self.wind_data_file, "rb") as f:
            self.WindData = pickle.load(f)

    def _load_current_data(self):
        with open(self.current_data_file, "rb") as f:
            self.CurrentData = pickle.load(f)

    def _load_wave_data(self):
        with open(self.wave_data_file, "rb") as f:
            self.WaveData = pickle.load(f)

    def _sample_global_path(self, n_seg_path:int, straight_wp_dist:int, straight_lmin:int, straight_lmax:int,
                            phi_min:int, phi_max:int, rad_min:int, rad_max:int, build:str):
        """Constructs a river by interchanging straight and curved segments; Fossen (2021) and Paulig, Okhrin (2023).
        The agent should follows the path always in direction of increasing indices."""

        assert build in ["straight", "right_curved", "left_curved", "random"], "Unknown river build."

        # do it until we have a path whichs stays in our simulation domain 
        while True:

            # set starting point
            n0, e0, _ = to_utm(lat=56.0, lon=9.0)
            path_n = [n0]
            path_e = [e0]

            # initialize path angle
            if build == "random":
                eps = dtr(np.random.randint(361))
                last_curves_right = np.array([None, None])
            else:
                eps = 0.0

            for _ in range(n_seg_path):

                # start with straight segment
                l = float(np.random.choice(np.arange(straight_lmin, straight_lmax + straight_wp_dist, straight_wp_dist), 1))
                cnt_l = 0
                while cnt_l < l:
                    e_add, n_add = xy_from_polar(r=straight_wp_dist, angle=eps)
                    path_n.append(path_n[-1] + n_add)
                    path_e.append(path_e[-1] + e_add)
                    cnt_l += straight_wp_dist

                if build != "straight":

                    # sample curved segment
                    if build == "random":

                        # allow for maximal two times the same curve to avoid producing a self-crossing river
                        if all(last_curves_right):
                            right_curve = False
                        elif all(last_curves_right == False):
                            right_curve = True
                        else:
                            right_curve = bool(random.getrandbits(1))

                        # update curve-history
                        last_curves_right[0] = copy(last_curves_right[1])
                        last_curves_right[1] = right_curve
                    
                    elif build == "right_curved":
                        right_curve = True

                    elif build == "left_curved":
                        right_curve = False
                    
                    phi = int(np.random.choice(np.arange(phi_min, phi_max + 1), 1))
                    r = np.random.choice(np.arange(rad_min, rad_max + 1))

                    # get circle origin
                    radius_ang = angle_to_2pi(eps + dtr(90) if right_curve else eps - dtr(90))
                    erad_add, nrad_add = xy_from_polar(r=r, angle=radius_ang)
                    n_rad = path_n[-1] + nrad_add
                    e_rad = path_e[-1] + erad_add

                    # create curve
                    ang0 = rtd(bng_abs(N0=n_rad, E0=e_rad, N1=path_n[-1], E1=path_e[-1]))

                    # careful: remove last wp from straight segment to avoid redundant points
                    del path_n[-1]
                    del path_e[-1]

                    if right_curve:
                        gen = range(0, phi + 1)
                    else:
                        gen = reversed(range(-phi, 1))
                    for ang in gen:
                        e_add, n_add = xy_from_polar(r=r, angle=angle_to_2pi(dtr(ang0 + ang)))
                        path_n.append(n_rad + n_add)
                        path_e.append(e_rad + e_add)

                    # update epsilon, rounded to nearest full degree
                    eps = bng_abs(N0=path_n[-2], E0=path_e[-2], N1=path_n[-1], E1=path_e[-1])

            # to latlon
            lat, lon = to_latlon(north=np.array(path_n), east=np.array(path_e), number=32)
 
            # check
            if all(self.lat_lims[0] <= lat) and all(self.lat_lims[1] >= lat) and all(6.1 <= lon)\
                and all(11.9 >= lon):
                break
        # store
        self.GlobalPath = Path(level="global", lat=lat, lon=lon, north=path_n, east=path_e)

        # overwrite data range
        self.off = 0.075
        self.lat_lims = [np.min(lat)-self.off, np.max(lat)+self.off]
        self.lon_lims = [np.min(lon)-self.off, np.max(lon)+self.off]

    def _sample_river_depth_data(self, offset:float, noise:bool):
        """Generates random depth data following Paulig & Okhrin (2023)."""
        path_n = np.zeros_like(self.GlobalPath.north)
        path_e = np.zeros_like(self.GlobalPath.east)

        # use an offset since the global path should not lie directly in the center of the river
        nwps = len(path_n)
        for i in range(nwps):
            n = self.GlobalPath.north[i]
            e = self.GlobalPath.east[i]

            if i != (nwps-1):
                n_nxt = self.GlobalPath.north[i+1]
                e_nxt = self.GlobalPath.east[i+1]
                ang = angle_to_2pi(bng_abs(N0=n, E0=e, N1=n_nxt, E1=e_nxt) - math.pi/2)
            else:
                n_last = self.GlobalPath.north[i-1]
                e_last = self.GlobalPath.east[i-1]
                ang = angle_to_2pi(bng_abs(N0=n_last, E0=e_last, N1=n, E1=e) - math.pi/2)

            e_add, n_add = xy_from_polar(r=offset, angle=ang)
            path_n[i] = n + n_add
            path_e[i] = e + e_add

        # river config parameters
        eps_param = 5e-10
        C_npts    = 10     # per side, so 500m total width
        C_pt_dist = 25

        if noise:
            noise = lambda: np.random.uniform(-2.0, 2.0)
            H_max = np.clip(float(np.random.exponential(scale=35, size=1)), 20, 100)
        else:
            noise = lambda: 0.0
            H_max = 35

        # prep final lists
        depth_n = []
        depth_e = []
        depth = []

        # fill each cross-section
        for i in range(len(path_n)):

            # get curvature
            if i == len(path_n)-1:
                ang = bng_abs(N0=path_n[i-1], E0=path_e[i-1], N1=path_n[i], E1=path_e[i])
            else:
                ang = bng_abs(N0=path_n[i], E0=path_e[i], N1=path_n[i+1], E1=path_e[i+1])

            # init depth along the path
            C_n = [path_n[i]]
            C_e = [path_e[i]]
            C_H = [H_max + noise()]

            # fill both sides of cross-section
            for direc in ["left", "right"]:
                phi = angle_to_2pi(ang - dtr(90.0) if direc == "left" else ang + dtr(90.0))
                for n in range(1, C_npts+1):
                    e_add, n_add = xy_from_polar(r=n*C_pt_dist, angle=phi)
                    C_n.append(path_n[i] + n_add)
                    C_e.append(path_e[i] + e_add)
                    new_H = H_max * math.exp(-eps_param * (n*C_pt_dist)**4) + noise()
                    C_H.append(max([new_H, 1.0]))
            
            # store it
            depth_n += C_n
            depth_e += C_e
            depth += C_H

        # go to lat-lon
        depth_lat, depth_lon = to_latlon(north=np.array(depth_n), east=np.array(depth_e), number=32)

        # store everything in regular grid based on linear interpolation
        self.DepthData = {}
        self.DepthData["lon"] = np.linspace(start=np.min(depth_lon), stop=np.max(depth_lon), num=200)
        self.DepthData["lat"] = np.linspace(start=np.min(depth_lat), stop=np.max(depth_lat), num=200)

        lat_grid, lon_grid = np.meshgrid(self.DepthData["lat"], self.DepthData["lon"])
        self.DepthData["data"] = griddata(points=(depth_lat, depth_lon), values=np.array(depth), 
                                          xi=(lat_grid, lon_grid), method="linear", fill_value=1.0, rescale=False).transpose()

        # water depth is 1m outside the river
        depth_n, depth_e, _ = to_utm(lat=self.DepthData["lat"], lon=self.DepthData["lon"])
        for i, e in enumerate(depth_e):
            for j, n in enumerate(depth_n):
                if np.min(ED(N0=n, E0=e, N1=path_n, E1=path_e)) > 250:
                    self.DepthData["data"][j, i] = 1.0

        # speed-up testing
        """
        ind = [(20*inter, 20*(inter+1)) for inter in range(10)]
        for row in ind:
            for col in ind:
                block = self.DepthData["data"][row[0]:row[1], col[0]:col[1]]
                if (block == 1.0).all():
                    pass
                else:
                    depth_n_block = depth_n[row[0]:row[1]]
                    depth_e_block = depth_e[col[0]:col[1]]

                    for i, e in enumerate(depth_e_block):
                        for j, n in enumerate(depth_n_block):
                            if np.min(ED(N0=n, E0=e, N1=path_n, E1=path_e)) > 250:
                                block[j, i] = 1.0
                self.DepthData["data"][row[0]:row[1], col[0]:col[1]] = block
        """
        if self.plot_depth:
            # log
            self.log_Depth = np.log(self.DepthData["data"])

            # for contour plot
            self.con_ticks = np.log([1.0, 2.0, 5.0, 15.0, 50.0, 150.0, 500.0])
            self.con_ticklabels = [int(np.round(tick, 0)) for tick in np.exp(self.con_ticks)]
            self.con_ticklabels[0] = 0
            self.clev = np.arange(0, self.log_Depth.max(), .1)

    def _depth_at_latlon(self, lat_q, lon_q):
        """Computes the water depth at a (queried) longitude-latitude position based on linear interpolation."""
        return Z_at_latlon(Z=self.DepthData["data"], lat_array=self.DepthData["lat"], lon_array=self.DepthData["lon"],
                           lat_q=lat_q, lon_q=lon_q)

    def _current_at_latlon(self, lat_q, lon_q):
        """Computes the current speed and angle at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (speed, angle)"""
        speed = Z_at_latlon(Z=self.CurrentData["speed_mps"], lat_array=self.CurrentData["lat"], lon_array=self.CurrentData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
        angle = Z_at_latlon(Z=self.CurrentData["angle"], lat_array=self.CurrentData["lat"], lon_array=self.CurrentData["lon"],
                            lat_q=lat_q, lon_q=lon_q, angle=True)
        return speed, angle

    def _wind_at_latlon(self, lat_q, lon_q):
        """Computes the wind speed and angle at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (speed, angle)"""
        speed = Z_at_latlon(Z=self.WindData["speed_mps"], lat_array=self.WindData["lat"], lon_array=self.WindData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
        angle = Z_at_latlon(Z=self.WindData["angle"], lat_array=self.WindData["lat"], lon_array=self.WindData["lon"], 
                            lat_q=lat_q, lon_q=lon_q, angle=True)
        return speed, angle

    def _wave_at_latlon(self, lat_q, lon_q):
        """Computes the wave angle, amplitude, period, and length at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (angle, amplitude, period, length)"""
        angle = Z_at_latlon(Z=self.WaveData["angle"], lat_array=self.WaveData["lat"], lon_array=self.WaveData["lon"],
                            lat_q=lat_q, lon_q=lon_q, angle=True)
        amplitude = Z_at_latlon(Z=self.WaveData["height"], lat_array=self.WaveData["lat"], lon_array=self.WaveData["lon"],
                                lat_q=lat_q, lon_q=lon_q)
        period = Z_at_latlon(Z=self.WaveData["period"], lat_array=self.WaveData["lat"], lon_array=self.WaveData["lon"],
                             lat_q=lat_q, lon_q=lon_q)
        length = Z_at_latlon(Z=self.WaveData["length"], lat_array=self.WaveData["lat"], lon_array=self.WaveData["lon"],
                             lat_q=lat_q, lon_q=lon_q)
        return angle, amplitude, period, length

    def _init_wps(self, vessel:KVLCC2, path_level:str):
        """Initializes the waypoints on the global and local path, respectively, based on the position of the vessel.
        Returns the vessel."""
        assert path_level in ["global", "local"], "Unknown path level."

        if path_level == "global":
            if hasattr(vessel, "rev_dir"):
                path = self.RevGlobalPath if vessel.rev_dir else self.GlobalPath
            else:
                path = self.GlobalPath

            vessel.glo_wp1_idx, vessel.glo_wp1_N, vessel.glo_wp1_E, vessel.glo_wp2_idx, vessel.glo_wp2_N, \
                vessel.glo_wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=vessel.eta[0], a_e=vessel.eta[1])
            try:
                vessel.glo_wp3_idx = vessel.glo_wp2_idx + 1
                vessel.glo_wp3_N = path.north[vessel.glo_wp3_idx] 
                vessel.glo_wp3_E = path.east[vessel.glo_wp3_idx]
            except:
                raise ValueError("Waypoint setting fails if vessel is not at least two waypoints away from the goal.")
        else:
            path = self.LocalPath
            vessel.loc_wp1_idx, vessel.loc_wp1_N, vessel.loc_wp1_E, vessel.loc_wp2_idx, vessel.loc_wp2_N, \
                vessel.loc_wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=vessel.eta[0], a_e=vessel.eta[1])
            try:
                vessel.loc_wp3_idx = vessel.loc_wp2_idx + 1
                vessel.loc_wp3_N = path.north[vessel.loc_wp3_idx] 
                vessel.loc_wp3_E = path.east[vessel.loc_wp3_idx]
            except:
                raise ValueError("Waypoint setting fails if vessel is not at least two waypoints away from the goal.")
        return vessel

    def _get_COLREG_situation(self, N0:float, E0:float, head0:float, v0:float, chi0:float,
                                    N1:float, E1:float, head1:float, v1:float, chi1:float):
        """Determines the COLREG situation from the perspective of the OS. 
        Follows Xu et al. (2020, Ocean Engineering; 2022, Neurocomputing).

        Args:
            N0(float):     north of OS
            E0(float):     east of OS
            head0(float):  heading of OS
            v0(float):     speed of OS
            chi0(float):   course of OS
            N1(float):     north of TS
            E1(float):     east of TS
            head1(float):  heading of TS
            v1(float):     speed of TS
            chi1(float):   course of TS

        Returns:
            1 - head-on 
            2 - starboard crossing
            3 - portside crossing
            4 - overtaking
            5 - no conflict situation
        """
        # check whether TS is too far away
        if ED(N0=N0, E0=E0, N1=N1, E1=E1) > self.open_enc_range:
            return 5

        # relative bearing from OS to TS
        bng_OS = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0)

        # relative bearing from TS to OS
        bng_TS = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)

        # intersection angle
        C_T = head_inter(head_OS=head0, head_TS=head1)

        # velocity component of OS in direction of TS
        V_rel_x, V_rel_y = project_vector(VA=v0, angleA=chi0, VB=v1, angleB=chi1)
        V_rel = polar_from_xy(x=V_rel_x, y=V_rel_y, with_r=True, with_angle=False)[0]

        # COLREG: Head-on
        if -5 <= rtd(angle_to_pi(bng_OS)) <= 5 and 175 <= rtd(C_T) <= 185:
            return 1
        
        # COLREG: Starboard crossing
        if 5 <= rtd(bng_OS) <= 112.5 and 185 <= rtd(C_T) <= 292.5:
            return 2

        # COLREG: Portside crossing
        if 247.5 <= rtd(bng_OS) <= 355 and 67.5 <= rtd(C_T) <= 175:
            return 3

        # COLREG: Overtaking
        if 112.5 <= rtd(bng_TS) <= 247.5 and -67.5 <= rtd(angle_to_pi(C_T)) <= 67.5 and V_rel > v1:
            return 4

        # no encounter situation
        return 5

    def _violates_COLREG_rules(self, N0:float, E0:float, head0:float, chi0:float, v0:float, r0:float, N1:float, E1:float,\
        head1:float, chi1:float, v1:float) -> bool:
        """Checks whether a situation violates the COLREG rules of the open sea.
        Args:
            N0(float):     north of OS
            E0(float):     east of OS
            head0(float):  heading of OS
            chi0(float):   course of OS
            v0(float):     speed of OS
            r0(float):     turning rate of OS
            N1(float):     north of TS
            E1(float):     east of TS
            head1(float):  heading of TS
            chi1(float):   course of TS
            v1(float):     speed of TS
        """
        sit = self._get_COLREG_situation(N0=N0, E0=E0, head0=head0, chi0=chi0, v0=v0, 
                                         N1=N1, E1=E1, head1=head1, chi1=chi1, v1=v1)

        # steer to the right in Head-on and starboard crossing situations
        if (sit in [1, 2]) and (r0 < 0.0):

            # evaluate only if TCPA with TS is positive
            if tcpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=chi0, chiTS=chi1, VOS=v0, VTS=v1) >= 0.0:
                return True
        return False

    def sense_LiDAR(self, N0:float, E0:float, head0:float, check_lane_river:bool = False):
        """Generates an observation via LiDAR sensoring. There are 'lidar_n_beams' equally spaced beams originating from the midship of the OS.
        The first beam is defined in direction of the heading of the OS. Each beam consists of 'n_dots_per_beam' sub-points, which are sequentially considered. 
        Returns for each beam the distance at which insufficient water depth has been detected, where the maximum range is 'lidar_range'.
        Furthermore, it returns the endpoints in lat-lon of each (truncated) beam.
        Args:
            N0(float): north position
            E0(float): east position
            head0(float): heading
            check_lane_river(bool): whether to consider the reversed path as artifical land that cannot be crossed
        
        Returns (as tuple):
            np.array(lidar_n_beams,) of dists
            list : endpoints as lat-lon-tuples
            np.array(lidar_n_beams,) of n-positions
            np.array(lidar_n_beams,) of e-positions
        """
        # setup output
        out_dists = np.ones(self.lidar_n_beams) * self.lidar_range
        out_lat_lon = []
        out_n = []
        out_e = []

        if check_lane_river:
            path = self.RevGlobalPath
        
        for out_idx, angle in enumerate(self.lidar_beam_angles):

            # current angle under consideration of the heading
            angle = angle_to_2pi(angle + head0)

            for dist in self.d_dots_per_beam:

                # compute N-E coordinates of dot
                delta_E_dot, delta_N_dot = xy_from_polar(r=dist, angle=angle)
                N_dot = N0 + delta_N_dot
                E_dot = E0 + delta_E_dot

                # transform to LatLon
                lat_dot, lon_dot = to_latlon(north=N_dot, east=E_dot, number=32)

                # check water depth at that point
                finish = False
                depth_dot = self._depth_at_latlon(lat_q=lat_dot, lon_q=lon_dot)

                if depth_dot <= self.OS.critical_depth:
                    finish = True

                elif check_lane_river:
                    # compute CTE to reversed lane from that point
                    _, wp1_N, wp1_E, _, wp2_N, wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=N_dot, a_e=E_dot)

                    # switch wps since the path is reversed
                    if cte(N1=wp2_N, E1=wp2_E, N2=wp1_N, E2=wp1_E, NA=N_dot, EA=E_dot) < 0:
                        finish = True
                
                # breaking condition fulfilled
                if finish:
                    out_dists[out_idx] = dist
                    out_lat_lon.append((lat_dot, lon_dot))
                    out_n.append(N_dot)
                    out_e.append(E_dot)
                    break

                if dist == self.lidar_range:
                    out_lat_lon.append((lat_dot, lon_dot))
                    out_n.append(N_dot)
                    out_e.append(E_dot)

        return out_dists, out_lat_lon, np.array(out_n), np.array(out_e)
    
    def _get_closeness_from_lidar(self, dists):
        """Computes the closeness from given LiDAR distance measurements."""
        return np.clip(1.0-dists/self.lidar_range, 0.0, 1.0)

    def _get_river_enc_range(self, ang:float):
        """Computes based on a relative bearing from TS perspective in [0,2pi) the assumed encounter range on the river."""
        ang = rtd(ang)
        a = self.river_enc_range_min
        b = self.river_enc_range_max

        if 0 <= ang < 90.0:
            return a + ang * (b-a)/90.0
        
        elif 90.0 <= ang < 180.0:
            return (2*b-a) + ang * (a-b)/90.0

        elif 180.0 <= ang < 270.0:
            return (3*a-2*b) + ang * (b-a)/90.0

        else:
            return (4*b-3*a) + ang * (a-b)/90.0

    def _violates_river_traffic_rules(self, N0:float, E0:float, head0:float, v0:float, N1:float, E1:float, head1:float,\
        v1:float, only_speedys:bool) -> bool:
        """Checks whether a situation violates the rules on the Elbe from Lighthouse Tinsdal to Cuxhaven.
        Args:
            N0(float):     north of OS
            E0(float):     east of OS
            head0(float):  heading of OS
            v0(float):     speed of OS
            N1(float):     north of TS
            E1(float):     east of TS
            head1(float):  heading of TS
            v1(float):     speed of TS
            only_speedys(bool): whether only faster TSs are in the proximity of the agent"""
        # preparation
        ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
        rev_dir = (abs(head_inter(head_OS=head0, head_TS=head1, to_2pi=False)) >= dtr(90.0))

        bng_rel_TS_pers = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)
        river_enc_range = self._get_river_enc_range(bng_rel_TS_pers)

        # check whether TS is too far away
        if ED_OS_TS > river_enc_range:
            return False
        else:
            # normal target ships should be overtaken on their portside
            if (not rev_dir) and (v0 > v1):
                if dtr(90.0) <= bng_rel_TS_pers <= dtr(180.0):
                    return True

            # if there are only speedys around, they should pass the OS on their starboard side
            else:
                if only_speedys:
                    if (not rev_dir) and (v0 < v1):
                        if dtr(270.0) <= bng_rel_TS_pers <= dtr(360.0):
                            return True
        return False

    def _on_river(self, N0:float, E0:float):
        """Checks whether we are on river or on open sea, depending on surroundings.
        
        Returns:
            bool, True if on river, False if on open sea"""
        dists, _, _, _ = self.sense_LiDAR(N0=N0, E0=E0, head0=0.0)
        if all(dists == self.lidar_range):
            return False
        else:
            return True

    def _set_cte(self, path_level:str, smooth_dc:bool=True):
        """Sets the cross-track error based on VFG for both the local and global path."""
        assert path_level in ["global", "local"], "Choose between the global and local path for waypoint updating."

        if path_level == "global":
            if smooth_dc:
                N3, E3 = self.OS.glo_wp3_N, self.OS.glo_wp3_E
            else:
                N3, E3 = None, None
            self.glo_ye, self.glo_desired_course, self.glo_pi_path, _ = VFG(N1 = self.OS.glo_wp1_N, 
                                                                            E1 = self.OS.glo_wp1_E, 
                                                                            N2 = self.OS.glo_wp2_N, 
                                                                            E2 = self.OS.glo_wp2_E,
                                                                            NA = self.OS.eta[0], 
                                                                            EA = self.OS.eta[1], 
                                                                            K  = self.VFG_K, 
                                                                            N3 = N3,
                                                                            E3 = E3)
        else:
            if smooth_dc:
                N3, E3 = self.OS.loc_wp3_N, self.OS.loc_wp3_E
            else:
                N3, E3 = None, None
            self.loc_ye, self.loc_desired_course, self.loc_pi_path, _ = VFG(N1 = self.OS.loc_wp1_N, 
                                                                            E1 = self.OS.loc_wp1_E, 
                                                                            N2 = self.OS.loc_wp2_N, 
                                                                            E2 = self.OS.loc_wp2_E,
                                                                            NA = self.OS.eta[0], 
                                                                            EA = self.OS.eta[1], 
                                                                            K  = self.VFG_K, 
                                                                            N3 = N3,
                                                                            E3 = E3)
    def _set_ce(self, path_level):
        """Sets the course error, which is desired course minus course."""
        assert path_level in ["global", "local"], "Choose between the global and local path for waypoint updating."

        if path_level == "global":
            self.glo_course_error = angle_to_pi(self.glo_desired_course - self.OS._get_course())
        else:
            self.loc_course_error = angle_to_pi(self.loc_desired_course - self.OS._get_course())

    def _handle_respawn(self, TS:TargetShip):
        """Checks whether the respawning condition of the target ship is fulfilled.
        Returns:
            TargetShip"""
        return TS

    def _update_disturbances(self, OS_lat=None, OS_lon=None):
        """Updates the environmental disturbances at the agent's current position."""
        if OS_lat is None and OS_lon is None:
            OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=32)

        if hasattr(self, "CurrentData"):
            self.V_c, self.beta_c = self._current_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        
        if hasattr(self, "WindData"):
            self.V_w, self.beta_w = self._wind_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        
        if hasattr(self, "DepthData"):
            self.H = self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        
        if hasattr(self, "WaveData"):
            self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = self._wave_at_latlon(lat_q=OS_lat, lon_q=OS_lon)

            # consider wave data issues
            if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
                [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
                self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None
            
            elif self.T_0_wave == 0.0:
                self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None

    def reset(self):
        raise NotImplementedError()

    def step(self, a):
        raise NotImplementedError()

    def _set_state(self):
        raise NotImplementedError()

    def _calculate_reward(self, a):
        raise NotImplementedError()

    def _done(self):
        raise NotImplementedError()

    def __str__(self, OS_lat, OS_lon) -> str:
        u, v, r = self.OS.nu
        course = self.OS._get_course()

        ste = f"Step: {self.step_cnt}"
        pos = f"Lat [°]: {OS_lat:.4f}, Lon [°]: {OS_lon:.4f}, " + r"$\psi$ [°]: " + f"{rtd(self.OS.eta[2]):.2f}"  + r", $\chi$ [°]: " + f"{rtd(course):.2f}"
        vel = f"u [m/s]: {u:.3f}, v [m/s]: {v:.3f}, r [rad/s]: {r:.3f}"
        out = ste + ", " + pos + "\n" + vel
        
        if hasattr(self, "DepthData"):
            depth = f"H [m]: {self.H:.2f}"
            out = out + "," + depth
        
        if hasattr(self, "WindData"):
            wind = r"$V_{\rm wind}$" + f" [kn]: {mps_to_knots(self.V_w):.2f}, " + r"$\psi_{\rm wind}$" + f" [°]: {rtd(self.beta_w):.2f}"
            out = out + "\n" + wind + ", "

        if hasattr(self, "CurrentData"):
            current = r"$V_{\rm current}$" + f" [m/s]: {self.V_c:.2f}, " + r"$\psi_{\rm current}$" + f" [°]: {rtd(self.beta_c):.2f}"
            out = out + current
        
        if hasattr(self, "WaveData"):
            if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
                [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
                wave = r"$\psi_{\rm wave}$" + f" [°]: -" + r", $\xi_{\rm wave}$ [m]: " + f"-" \
                    + r", $T_{\rm wave}$ [s]: " + f"-" + r", $\lambda_{\rm wave}$ [m]: " + f"-"
            else:
                wave = r"$\psi_{\rm wave}$" + f" [°]: {rtd(self.beta_wave):.2f}" + r", $\xi_{\rm wave}$ [m]: " + f"{self.eta_wave:.2f}" \
                    + r", $T_{\rm wave}$ [s]: " + f"{self.T_0_wave:.2f}" + r", $\lambda_{\rm wave}$ [m]: " + f"{self.lambda_wave:.2f}"
            out = out + "\n" + wave

        if hasattr(self, "glo_ye"):
            glo_path_info = "Global path: " + r"$y_e$" + f" [m]: {self.glo_ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.glo_desired_course):.2f}, " \
                + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.glo_course_error):.2f}"
            out = out + "\n" + glo_path_info

        if hasattr(self, "LocalPath"):
            loc_path_info = "Local path: " + r"$y_e$" + f" [m]: {self.loc_ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.loc_desired_course):.2f}, " \
                + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.loc_course_error):.2f}"
            return out + "\n" + loc_path_info
        return out

    def _render_ship(self, ax:plt.Axes, vessel:KVLCC2, color:str, plot_CR:bool=False, with_domain:bool=False):
        """Draws the ship on the axis, including ship domain. Returns the ax."""
        # quick access
        l = vessel.Lpp/2
        b = vessel.B/2
        N0, E0, head0 = self.OS.eta
        N1, E1, head1 = vessel.eta

        # overwrite color for COLREG situation
        if not "River" in type(self).__name__ and (vessel is not self.OS):
            sit = self._get_COLREG_situation(N0=N0, E0=E0, head0=head0, chi0=self.OS._get_course(), v0=vessel._get_V(),
                                             N1=N1, E1=E1, head1=head1, chi1=vessel._get_course(), v1=vessel._get_V())
            sit = 0 if sit == 5 else sit # different labeling in HHOS
            color = COLREG_COLORS[sit]

        # get rectangle/polygon end points in UTM
        A = (E1 - b, N1 + l)
        B = (E1 + b, N1 + l)
        C = (E1 - b, N1 - l)
        D = (E1 + b, N1 - l)

        # rotate them according to heading
        A = rotate_point(x=A[0], y=A[1], cx=E1, cy=N1, angle=-head1)
        B = rotate_point(x=B[0], y=B[1], cx=E1, cy=N1, angle=-head1)
        C = rotate_point(x=C[0], y=C[1], cx=E1, cy=N1, angle=-head1)
        D = rotate_point(x=D[0], y=D[1], cx=E1, cy=N1, angle=-head1)

        # ship domain
        if with_domain:
            xys = [rotate_point(E1 + x, N1 + y, cx=E1, cy=N1, angle=-head1) for x, y in zip(self.domain_xs, self.domain_ys)]

        if self.plot_in_latlon:

            # convert rectangle points to lat/lon
            A_lat, A_lon = to_latlon(north=A[1], east=A[0], number=32)
            B_lat, B_lon = to_latlon(north=B[1], east=B[0], number=32)
            C_lat, C_lon = to_latlon(north=C[1], east=C[0], number=32)
            D_lat, D_lon = to_latlon(north=D[1], east=D[0], number=32)

            # draw the polygon (A is included twice to create a closed shape)
            lons = [A_lon, B_lon, D_lon, C_lon, A_lon]
            lats = [A_lat, B_lat, D_lat, C_lat, A_lat]
            ax.plot(lons, lats, color=color, linewidth=2.0, zorder=10)

            # plot ship domain
            if with_domain:
                lat_lon_tups = [to_latlon(north=y, east=x, number=32)[:2] for x, y in xys]
                lats = [e[0] for e in lat_lon_tups]
                lons = [e[1] for e in lat_lon_tups]
                ax.plot(lons, lats, color=color, alpha=0.7)

            if plot_CR:
                CR_x = min(lons) - np.abs(min(lons) - max(lons))
                CR_y = min(lats) - 2*np.abs(min(lats) - max(lats))
                ax.text(CR_x, CR_y, f"CR: {self._get_CR_open_sea(vessel0=self.OS, vessel1=vessel,TCPA_norm=15*60, DCPA_norm=NM_to_meter(1.0), dist_norm=(NM_to_meter(0.5))**2):.2f}",\
                        fontsize=7, horizontalalignment='center', verticalalignment='center', color=color)
        else:
            # draw the polygon
            xs = [A[0], B[0], D[0], C[0], A[0]]
            ys = [A[1], B[1], D[1], C[1], A[1]]
            ax.plot(xs, ys, color=color, linewidth=2.0)

            # plot ship domain
            if with_domain:
                xs = [xy[0] for xy in xys]
                ys = [xy[1] for xy in xys]
                ax.plot(xs, ys, color=color, alpha=0.7)

            if plot_CR:
                CR_x = min(xs) - np.abs(min(xs) - max(xs))
                CR_y = min(ys) - 2*np.abs(min(ys) - max(ys))
                ax.text(CR_x, CR_y, f"CR: {self._get_CR_open_sea(vessel0=self.OS, vessel1=vessel,TCPA_norm=15*60, DCPA_norm=NM_to_meter(1.0), dist_norm=(NM_to_meter(0.5))**2):.2f}",\
                        fontsize=7, horizontalalignment='center', verticalalignment='center', color=color)
        return ax

    def _render_wps(self, ax, vessel, path_level, color):
        """Renders the current waypoints of a vessel."""
        assert path_level in ["global", "local"], "Choose between the global and local path for waypoint rendering."

        if path_level == "global":
            wp1_lat, wp1_lon = to_latlon(north=vessel.glo_wp1_N, east=vessel.glo_wp1_E, number=32)
            wp2_lat, wp2_lon = to_latlon(north=vessel.glo_wp2_N, east=vessel.glo_wp2_E, number=32)
        else:
            wp1_lat, wp1_lon = to_latlon(north=vessel.loc_wp1_N, east=vessel.loc_wp1_E, number=32)
            wp2_lat, wp2_lon = to_latlon(north=vessel.loc_wp2_N, east=vessel.loc_wp2_E, number=32)

        ax.plot([wp1_lon, wp2_lon], [wp1_lat, wp2_lat], color=color, linewidth=1.0, markersize=3)
        return ax

    def render(self, data=None):
        """Renders the current environment. Note: The 'data' argument is needed since a recent update of the 'gym' package."""
        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            if self.plot_reward:
                self.f = plt.figure(figsize=(14, 8))
                self.gs  = self.f.add_gridspec(2, 2)
                self.ax1 = self.f.add_subplot(self.gs[:, 0]) # ship
                self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward
                self.ax3 = self.f.add_subplot(self.gs[1, 1]) # action
            else:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))

            plt.ion()
            plt.show()

        if self.step_cnt % 1 == 0:
            
            # ------------------------------ reward and action plot --------------------------------
            if self.plot_reward:
                
                # storage
                if self.step_cnt == 0:

                    # reward - array init
                    self.ax2.r      = np.zeros(self._max_episode_steps)
                    self.ax2.r_ye   = np.zeros(self._max_episode_steps)
                    self.ax2.r_ce   = np.zeros(self._max_episode_steps)
                    self.ax2.r_comf = np.zeros(self._max_episode_steps)

                    if "Plan" in type(self).__name__:
                        self.ax2.r_coll = np.zeros(self._max_episode_steps)
                        self.ax2.r_rule = np.zeros(self._max_episode_steps)

                    # reward - naming
                    self.ax2.r_names = ["agg", "ye", "ce", "comf"]
                    if "Plan" in type(self).__name__:
                        self.ax2.r_names += ["coll", "rule"]

                    # action - array init
                    self.ax3.a0 = np.zeros(self._max_episode_steps)

                    # action - naming
                    self.ax3.a_names = ["steering"]

                else:
                    # reward
                    self.ax2.r[self.step_cnt]      = self.r
                    self.ax2.r_ye[self.step_cnt]   = self.r_ye
                    self.ax2.r_ce[self.step_cnt]   = self.r_ce
                    self.ax2.r_comf[self.step_cnt] = self.r_comf

                    if "Plan" in type(self).__name__:
                        self.ax2.r_coll[self.step_cnt] = self.r_coll
                        self.ax2.r_rule[self.step_cnt] = self.r_rule

                    # action
                    self.ax3.a0[self.step_cnt] = float(self.a[0])

                # periodically clear and init
                if self.step_cnt % 50 == 0:

                    self.ax2.clear()
                    self.ax3.clear()

                    # appearance
                    self.ax2.set_title(type(self).__name__.replace("_", "-"))
                    self.ax2.set_xlabel("Timestep in episode")
                    self.ax2.set_ylabel("Reward")
                    self.ax2.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
                    self.ax2.set_ylim(-5.0, 3.0)

                    self.ax3.set_xlabel("Timestep in episode")
                    self.ax3.set_ylabel("Action")
                    self.ax3.set_ylim(-1.05, 1.05)
                    self.ax3.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))

                    # ---------- animated artists: initial drawing ---------
                    # rewards
                    self.ax2.lns = []
                    for i, lab in enumerate(self.ax2.r_names):
                        self.ax2.lns.append(self.ax2.plot([], [], color=COLREG_COLORS[i], label=lab, animated=True)[0])
                    self.ax2.legend()

                    # actions                       
                    self.ax3.lns = []
                    for i, lab in enumerate(self.ax3.a_names):
                        self.ax3.lns.append(self.ax3.plot([], [], color="red" if i == 0 else "blue", label=self.ax3.a_names[i], animated=True)[0])
                    self.ax3.legend()

                    # ----------------- store background -------------------
                    self.f.canvas.draw()
                    self.ax2.bg = self.f.canvas.copy_from_bbox(self.ax2.bbox)
                    self.ax3.bg = self.f.canvas.copy_from_bbox(self.ax3.bbox)
                
                else:
                    # ------------- restore the background ---------------
                    self.f.canvas.restore_region(self.ax2.bg)
                    self.f.canvas.restore_region(self.ax3.bg)

                    # ----------- animated artists: update ---------------
                    # reward
                    for i, lab in enumerate(self.ax2.r_names):
                        if lab == "agg":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r[:self.step_cnt+1])
                        
                        elif lab == "ye":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_ye[:self.step_cnt+1])
                        
                        elif lab == "ce":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_ce[:self.step_cnt+1])
                        
                        elif lab == "comf":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_comf[:self.step_cnt+1])

                        elif lab == "coll":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_coll[:self.step_cnt+1])

                        elif lab == "rule":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_rule[:self.step_cnt+1])

                        elif lab == "speed":
                            self.ax2.lns[i].set_data(np.arange(self.step_cnt+1), self.ax2.r_speed[:self.step_cnt+1])

                        self.ax2.draw_artist(self.ax2.lns[i])
                    
                    # action
                    for i, lab in enumerate(self.ax3.a_names):
                        if lab == "steering":
                            self.ax3.lns[i].set_data(np.arange(self.step_cnt+1), self.ax3.a0[:self.step_cnt+1])
                        
                        elif lab == "speed":
                            self.ax3.lns[i].set_data(np.arange(self.step_cnt+1), self.ax3.a1[:self.step_cnt+1])
                        
                        self.ax3.draw_artist(self.ax3.lns[i])

                    # show it on screen
                    self.f.canvas.blit(self.ax2.bbox)
                    self.f.canvas.blit(self.ax3.bbox)

            # ------------------------------ ship movement --------------------------------
            # get position of OS in lat/lon
            N0, E0, head0 = self.OS.eta
            OS_lat, OS_lon = to_latlon(north=N0, east=E0, number=32)

            for ax in [self.ax1]:
                ax.clear()

                # general information
                if self.plot_in_latlon:
                    ax.text(0.125, 0.8875, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=8, transform=plt.gcf().transFigure)

                    ax.set_xlabel("Longitude [°]", fontsize=10)
                    ax.set_ylabel("Latitude [°]", fontsize=10)

                    xlims = [max([self.lon_lims[0], OS_lon - self.show_lon_lat/2]), min([self.lon_lims[1], OS_lon + self.show_lon_lat/2])]
                    ylims = [max([self.lat_lims[0], OS_lat - self.show_lon_lat/2]), min([self.lat_lims[1], OS_lat + self.show_lon_lat/2])]
                    ax.set_xlim(*xlims)
                    ax.set_ylim(*ylims)
                else:
                    ax.text(0.125, 0.8675, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=10, transform=plt.gcf().transFigure)

                    ax.set_xlabel("UTM-E [m]", fontsize=10)
                    ax.set_ylabel("UTM-N [m]", fontsize=10)

                    # reverse xaxis in UTM
                    ax.set_xlim(E0 - self.UTM_viz_range_E, E0 + self.UTM_viz_range_E)
                    ax.set_ylim(N0 - self.UTM_viz_range_N, N0 + self.UTM_viz_range_N)

                #--------------- depth plot ---------------------
                if self.plot_depth and self.plot_in_latlon and hasattr(self, "DepthData"):
                    lower_lon_idx = max([find_nearest(array=self.DepthData["lon"], value=xlims[0])[1] - 1, 0])
                    upper_lon_idx = min([find_nearest(array=self.DepthData["lon"], value=xlims[1])[1] + 1, len(self.DepthData["lon"])-1])

                    lower_lat_idx = max([find_nearest(array=self.DepthData["lat"], value=ylims[0])[1] - 1, 0])
                    upper_lat_idx = min([find_nearest(array=self.DepthData["lat"], value=ylims[1])[1] + 1, len(self.DepthData["lat"])-1])

                    # contour plot from depth data
                    con = ax.contourf(self.DepthData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                                      self.DepthData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                                      self.log_Depth[lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                                      self.clev, cmap=cm.ocean)

                    # colorbar as legend
                    if self.step_cnt == 0 and self.first_init:
                        cbar = self.f.colorbar(con, ticks=self.con_ticks, ax=ax)
                        cbar.ax.set_yticklabels(self.con_ticklabels)
                        self.first_init = False

                #--------------- wind plot ---------------------
                if self.plot_wind and self.plot_in_latlon and hasattr(self, "WindData"):
                    lower_lon_idx = max([find_nearest(array=self.WindData["lon"], value=xlims[0])[1] - 1, 0])
                    upper_lon_idx = min([find_nearest(array=self.WindData["lon"], value=xlims[1])[1] + 1, len(self.WindData["lon"])-1])

                    lower_lat_idx = max([find_nearest(array=self.WindData["lat"], value=ylims[0])[1] - 1, 0])
                    upper_lat_idx = min([find_nearest(array=self.WindData["lat"], value=ylims[1])[1] + 1, len(self.WindData["lat"])-1])

                    ax.barbs(self.WindData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                             self.WindData["lat"][lower_lat_idx:(upper_lat_idx+1)], 
                             self.WindData["eastward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                             self.WindData["northward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                             length=4, barbcolor="goldenrod")

                #------------------ set ships ------------------------
                # OS
                ax = self._render_ship(ax=ax, vessel=self.OS, color="red", plot_CR=False, with_domain=True)

                # TSs
                if hasattr(self, "TSs"):
                    for TS in self.TSs:
                        if "River" in type(self).__name__ and hasattr(TS, "rev_dir"):
                            col = "darkgoldenrod" if TS.rev_dir else "gray"
                        else:
                            col = "yellow"
                        ax = self._render_ship(ax=ax, vessel=TS, color=col, plot_CR=True if not "River" in type(self).__name__ else False)

                        #if hasattr(TS, "path"):
                        #    ax.scatter(TS.path.lon, TS.path.lat, c=col)

                    # set legend for COLREGS
                    if not "River" in type(self).__name__:
                        legend1 = ax.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], 
                                            fontsize=6, loc='upper center', ncol=6)
                        ax.add_artist(legend1)

                #--------------------- Path ------------------------
                if self.plot_path:
                    loc_path_col = "salmon"

                    if self.plot_in_latlon:
                        # global
                        ax.plot(self.GlobalPath.lon, self.GlobalPath.lat, marker='o', color="purple", linewidth=1.0, markersize=3, label="Global Path")
                        if "River" in type(self).__name__:
                            ax.plot(self.RevGlobalPath.lon, self.RevGlobalPath.lat, marker='o', color="darkgoldenrod", linewidth=1.0, 
                                    alpha=0.5, markersize=3, label="Reversed Global Path")

                        # local
                        if hasattr(self, "LocalPath"):
                            if "PathFollowing" in type(self).__name__:
                                label = "Local Path" + f" ({self.planning_method})"
                            else:
                                label = "Local Path"
                            ax.plot(self.LocalPath.lon, self.LocalPath.lat, marker='o', color=loc_path_col, linewidth=1.0, 
                                    alpha=0.5, markersize=3, label=label)
                        ax.legend(loc="lower left")

                        # wps of OS
                        self._render_wps(ax=ax, vessel=self.OS, path_level="global", color="springgreen")

                        # cross-track error
                        if self.glo_ye < 0:
                            dE, dN = xy_from_polar(r=abs(self.glo_ye), angle=angle_to_2pi(self.glo_pi_path + dtr(90.0)))
                        else:
                            dE, dN = xy_from_polar(r=self.glo_ye, angle=angle_to_2pi(self.glo_pi_path - dtr(90.0)))
                        yte_lat, yte_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=32)
                        ax.plot([OS_lon, yte_lon], [OS_lat, yte_lat], color="purple")
                    else:
                        # global
                        ax.plot(self.GlobalPath.east, self.GlobalPath.north, marker='o', color="purple", linewidth=1.0, markersize=3, label="Global Path")
                        if "River" in type(self).__name__:
                            ax.plot(self.RevGlobalPath.east, self.RevGlobalPath.north, marker='o', color="darkgoldenrod", linewidth=1.0, markersize=3, label="Reversed Global Path")

                        # local
                        if hasattr(self, "LocalPath"):
                            if "PathFollowing" in type(self).__name__:
                                label = "Local Path" + f" ({self.planning_method})"
                            else:
                                label = "Local Path"
                            ax.plot(self.LocalPath.east, self.LocalPath.north, marker='o', color=loc_path_col, linewidth=1.0, markersize=3, label=label)
                        ax.legend(loc="lower left")

                        # current global waypoints
                        ax.plot([self.OS.glo_wp1_E, self.OS.glo_wp2_E], [self.OS.glo_wp1_N, self.OS.glo_wp2_N], color="springgreen", linewidth=1.0, markersize=3)

                        # cross-track error
                        if self.glo_ye < 0:
                            dE, dN = xy_from_polar(r=abs(self.glo_ye), angle=angle_to_2pi(self.glo_pi_path + dtr(90.0)))
                        else:
                            dE, dN = xy_from_polar(r=self.glo_ye, angle=angle_to_2pi(self.glo_pi_path - dtr(90.0)))
                        ax.plot([E0, E0+dE], [N0, N0+dN], color="purple")

                #--------------------- Current data ------------------------
                if self.plot_current and self.plot_in_latlon and hasattr(self, "CurrentData"):
                    lower_lon_idx = max([find_nearest(array=self.CurrentData["lon"], value=xlims[0])[1] - 1, 0])
                    upper_lon_idx = min([find_nearest(array=self.CurrentData["lon"], value=xlims[1])[1] + 1, len(self.CurrentData["lon"])-1])

                    lower_lat_idx = max([find_nearest(array=self.CurrentData["lat"], value=ylims[0])[1] - 1, 0])
                    upper_lat_idx = min([find_nearest(array=self.CurrentData["lat"], value=ylims[1])[1] + 1, len(self.CurrentData["lat"])-1])
                    
                    ax.quiver(self.CurrentData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                              self.CurrentData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                              self.CurrentData["eastward_mps"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                              self.CurrentData["northward_mps"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                              headwidth=2.0, color="orange", scale=10)

                #--------------------- Wave data ------------------------
                if self.plot_waves and self.plot_in_latlon and hasattr(self, "WaveData"):
                    lower_lon_idx = max([find_nearest(array=self.WaveData["lon"], value=xlims[0])[1] - 1, 0])
                    upper_lon_idx = min([find_nearest(array=self.WaveData["lon"], value=xlims[1])[1] + 1, len(self.WaveData["lon"])-1])

                    lower_lat_idx = max([find_nearest(array=self.WaveData["lat"], value=ylims[0])[1] - 1, 0])
                    upper_lat_idx = min([find_nearest(array=self.WaveData["lat"], value=ylims[1])[1] + 1, len(self.WaveData["lat"])-1])
                    
                    ax.quiver(self.WaveData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                            self.WaveData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                            self.WaveData["eastward"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                            self.WaveData["northward"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                            headwidth=2.0, color="purple", scale=10)

                #--------------------- LiDAR sensing ------------------------
                if self.plot_lidar and self.plot_in_latlon and "River" in type(self).__name__ and hasattr(self, "sense_LiDAR"):
                    L_dists, lidar_lat_lon, L_n, L_e  = self.sense_LiDAR(N0=N0, E0=E0, head0=head0, check_lane_river=True)

                    for i, latlon in enumerate(lidar_lat_lon):
                        #if L_dists[i] < 150:
                        #    ax.scatter(latlon[1], latlon[0], color="black", zorder=20)
                        ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.4)

            #plt.gca().set_aspect('equal')
            plt.pause(0.001)
