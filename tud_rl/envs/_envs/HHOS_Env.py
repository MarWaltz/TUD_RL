import math
import pickle
import random
from copy import copy, deepcopy
from typing import List

import gym
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import scipy.ndimage
from gym import spaces
from matplotlib import cm
from matplotlib import pyplot as plt

from tud_rl.envs._envs.HHOS_Fnc import (VFG, Z_at_latlon, ate, fill_array,
                                        find_nearest, get_init_two_wp,
                                        mps_to_knots, to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.MMG_TargetShip import Path, TargetShip
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                         NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, get_ship_domain, head_inter,
                                         polar_from_xy, project_vector, rtd,
                                         tcpa, xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 vessel from Hamburg to Oslo."""
    def __init__(self,
                 nps_control_follower : bool,
                 data : str, 
                 scenario_based : bool, 
                 N_TSs_max : int, 
                 N_TSs_random : bool, 
                 w_ye : float, 
                 w_ce : float, 
                 w_coll : float, 
                 w_comf : float, 
                 w_speed : float):
        super().__init__()

        # simulation settings
        self.delta_t = 5.0   # simulation time interval (in s)

        # LiDAR
        self.lidar_range       = NM_to_meter(1.0)   # range of LiDAR sensoring in m
        self.lidar_beam_angles = [20.0, 45., 90., 135.0]
        self.lidar_beam_angles += [360. - ang for ang in self.lidar_beam_angles] + [0.0, 180.0]
        self.lidar_beam_angles = np.deg2rad(np.sort(self.lidar_beam_angles))

        self.lidar_n_beams   = len(self.lidar_beam_angles)
        self.n_dots_per_beam = 10          # number of subpoints per beam
        self.d_dots_per_beam = np.linspace(start=0.0, stop=self.lidar_range, 
                                           num=self.lidar_n_beams+1, endpoint=True)[1:] # distances from midship of subpoints per beam

        # vector field guidance
        self.VFG_K = 0.001

        # data range
        self.lon_lims = [4.83, 14.33]
        self.lat_lims = [51.83, 60.5]
        self.lon_range = self.lon_lims[1] - self.lon_lims[0]
        self.lat_range = self.lat_lims[1] - self.lat_lims[0]

        # setting
        assert data in ["sampled", "real"], "Unknown HHOS data. Can either be 'sampled' or 'real'."
        self.data = data

        # data loading
        if self.data == "real":
            path_to_HHOS = "C:/Users/Martin Waltz/Desktop/Forschung/RL_packages/HHOS"
            self._load_global_path(path_to_HHOS)
            self._load_depth_data(path_to_HHOS + "/DepthData")
            self._load_wind_data(path_to_HHOS + "/winds")
            self._load_current_data(path_to_HHOS + "/currents")
            self._load_wave_data(path_to_HHOS + "/waves")
        else:
            self.n_wps_glo = 100         # number of wps of the global path
            self.l_seg_path = 200        # wp distance of the global path in m

            # depth data sampling parameters
            self.river_dist_left_loc  = 300
            self.river_dist_right_loc = 100
            self.river_dist_sca = 20
            self.river_dist_noise_loc = 0
            self.river_dist_noise_sca = 2
            self.river_min = 75

        # path characteristics
        self.n_wps_loc = 7
        self.dist_des_rev_path = 200

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.05

        # visualization
        self.plot_in_latlon = True         # if False, plots in UTM coordinates
        self.plot_depth = True
        self.plot_path = True
        self.plot_wind = False
        self.plot_current = False
        self.plot_waves = False
        self.plot_lidar = True
        self.plot_reward = True
        self.default_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        self.first_init = True

        if not self.plot_in_latlon:
            self.show_lon_lat = np.clip(self.show_lon_lat, 0.005, 5.95)
            self.UTM_viz_range_E = abs(to_utm(lat=52.0, lon=6.0001)[1] - to_utm(lat=52.0, lon=6.0001+self.show_lon_lat/2)[1])
            self.UTM_viz_range_N = abs(to_utm(lat=50.0, lon=8.0)[0] - to_utm(lat=50.0+self.show_lon_lat/2, lon=8.0)[0])

        # other ships
        assert not (scenario_based is True and N_TSs_max < 4), "You need at least four TSs in scenario-based spawning."
        self.scenario_based = scenario_based           # whether to spawn agent in specific scenarios (arg True), or all random (arg False)
        self.N_TSs_max    = N_TSs_max                  # maximum number of other vessels
        self.N_TSs_random = N_TSs_random               # if true, samples a random number in [0, N_TSs] at start of each episode
                                                       # if false, always have N_TSs_max
        self.TS_spawn_dists = [NM_to_meter(0.5), NM_to_meter(0.75)]  # for river
        self.TCPA_crit = 25 * 60 # spawning time for target ships on open sea

        # trajectory or path-planning and following
        self.two_actions = nps_control_follower

        self.loc_path_upd_freq = 25
        self.desired_V = 3.0

        # action/reward setup
        self.a = [0, 0] if self.two_actions else [0]
        
        self.w_ye = w_ye
        self.w_ce = w_ce
        self.w_coll = w_coll
        self.w_comf = w_comf
       
        self.r = 0
        self.r_ye = 0
        self.r_ce = 0
        self.r_coll = 0
        self.r_comf = 0

        if self.two_actions:
            self.w_speed = w_speed
            self.r_speed = 0.0

    def _load_global_path(self, path_to_global_path):
        with open(f"{path_to_global_path}/Path_latlon.pickle", "rb") as f:
            GlobalPath = pickle.load(f)
        self.GlobalPath = Path(level="global", **GlobalPath)
        self.n_wps_glo = self.GlobalPath.n_wps

    def _load_depth_data(self, path_to_depth_data):
        with open(f"{path_to_depth_data}/DepthData.pickle", "rb") as f:
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

    def _load_wind_data(self, path_to_wind_data):
        with open(f"{path_to_wind_data}/WindData_latlon.pickle", "rb") as f:
            self.WindData = pickle.load(f)

    def _load_current_data(self, path_to_current_data):
        with open(f"{path_to_current_data}/CurrentData_latlon.pickle", "rb") as f:
            self.CurrentData = pickle.load(f)

    def _load_wave_data(self, path_to_wave_data):
        with open(f"{path_to_wave_data}/WaveData_latlon.pickle", "rb") as f:
            self.WaveData = pickle.load(f)

    def _sample_global_path(self):
        """Constructs a path with n_wps equally-spaced way points. 
        The agent should follows the path always in direction of increasing indices."""
        # do it until we have a path whichs stays in our simulation domain
        while True:

            # set starting point
            path_n = np.zeros(self.n_wps_glo)
            path_e = np.zeros(self.n_wps_glo)
            path_n[0], path_e[0], _ = to_utm(lat=56.0, lon=9.0)

            # sample other points
            ang = np.random.uniform(0, 2*math.pi)
            ang_diff = dtr(np.random.uniform(-20., 20.))
            ang_diff2 = 0.0
            for i in range(1, self.n_wps_glo):
                
                # next angle
                ang_diff2 = 0.5 * ang_diff2 + 0.5 * dtr(np.random.uniform(-5.0, 5.0))
                ang_diff = 0.5 * ang_diff + 0.5 * ang_diff2 + 0.0 * dtr(np.random.uniform(-5.0, 5.0))
                ang = angle_to_2pi(ang + ang_diff)

                # next point
                e_add, n_add = xy_from_polar(r=self.l_seg_path, angle=ang)
                path_n[i] = path_n[i-1] + n_add
                path_e[i] = path_e[i-1] + e_add

            # to latlon
            lat, lon = to_latlon(north=path_n, east=path_e, number=32)

            # check
            if all(self.lat_lims[0] <= lat) and all(self.lat_lims[1] >= lat) and \
                all(6.1 <= lon) and all(11.9 >= lon):
                break

        # store
        self.GlobalPath = Path(level="global", lat=lat, lon=lon, north=path_n, east=path_e)

        # overwrite data range
        self.off = 0.075
        self.lat_lims = [np.min(lat)-self.off, np.max(lat)+self.off]
        self.lon_lims = [np.min(lon)-self.off, np.max(lon)+self.off]

    def _sample_depth_data(self, OS_lat, OS_lon):
        """Generates random depth data."""
        self.DepthData = {}
        self.DepthData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1] + self.off, num=500)
        self.DepthData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=500)
        self.DepthData["data"] = np.ones((len(self.DepthData["lat"]), len(self.DepthData["lon"])))

        if self.plan_on_river:

            while True:
                # sample distances to waypoints
                d_left  = np.zeros(self.n_wps_glo)
                d_right = np.zeros(self.n_wps_glo)
                depth = np.zeros(self.n_wps_glo)

                for i in range(self.n_wps_glo):
                    if i == 0:
                        d_left[i]  = np.clip(np.random.normal(loc=self.river_dist_left_loc, scale=self.river_dist_sca, size=1), self.river_min, np.infty)
                        d_right[i] = np.clip(np.random.normal(loc=self.river_dist_right_loc, scale=self.river_dist_sca, size=1), self.river_min, np.infty)
                        depth[i] = np.clip(np.random.exponential(scale=15, size=1), 20, 100)
                    else:
                        d_left[i]  = np.clip(
                                        d_left[i-1] + np.random.normal(loc=self.river_dist_noise_loc, scale=self.river_dist_noise_sca, size=1),
                                    self.river_min, 3*self.river_dist_left_loc)
                        d_right[i] = np.clip(
                                        d_right[i-1] + np.random.normal(loc=self.river_dist_noise_loc, scale=self.river_dist_noise_sca, size=1),
                                    self.river_min, 3*self.river_dist_right_loc)
                        depth[i] = np.clip(depth[i-1] + np.random.normal(loc=0.0, scale=10.0), 20, 100)

                # fill close points at that distance away from the path
                lat_path = self.GlobalPath.lat
                lon_path = self.GlobalPath.lon

                for i, (lat_p, lon_p) in enumerate(zip(lat_path, lon_path)):
                    
                    if i != self.n_wps_glo-1:

                        # go to utm
                        n1, e1, _ = to_utm(lat=lat_p, lon=lon_p)
                        n2, e2, _ = to_utm(lat=lat_path[i+1], lon=lon_path[i+1])

                        # angles
                        bng_absolute = bng_abs(N0=n1, E0=e1, N1=n2, E1=e2)
                        angle_left  = angle_to_2pi(bng_absolute - math.pi/2)
                        angle_right = angle_to_2pi(bng_absolute + math.pi/2)

                        # compute resulting points
                        e_add_l, n_add_l = xy_from_polar(r=d_left[i], angle=angle_left)
                        lat_left, lon_left = to_latlon(north=n1+n_add_l, east=e1+e_add_l, number=32)

                        e_add_r, n_add_r = xy_from_polar(r=d_right[i], angle=angle_right)
                        lat_right, lon_right = to_latlon(north=n1+n_add_r, east=e1+e_add_r, number=32)

                        # find closest point in the array
                        _, lat_left_idx = find_nearest(array=self.DepthData["lat"], value=lat_left)
                        _, lon_left_idx = find_nearest(array=self.DepthData["lon"], value=lon_left)

                        _, lat_right_idx = find_nearest(array=self.DepthData["lat"], value=lat_right)
                        _, lon_right_idx = find_nearest(array=self.DepthData["lon"], value=lon_right)

                        if i != 0:
                            lat_idx1 = np.min([lat_left_idx, lat_left_idx_old, lat_right_idx, lat_right_idx_old])
                            lat_idx2 = np.max([lat_left_idx, lat_left_idx_old, lat_right_idx, lat_right_idx_old])

                            lon_idx1 = np.min([lon_left_idx, lon_left_idx_old, lon_right_idx, lon_right_idx_old])
                            lon_idx2 = np.max([lon_left_idx, lon_left_idx_old, lon_right_idx, lon_right_idx_old])

                            self.DepthData["data"] = fill_array(Z=self.DepthData["data"], lat_idx1=lat_idx1, lon_idx1=lon_idx1,
                                                                lat_idx2=lat_idx2, lon_idx2=lon_idx2,
                                                                value=depth[i])
                        lat_left_idx_old = lat_left_idx
                        lon_left_idx_old = lon_left_idx
                        lat_right_idx_old = lat_right_idx
                        lon_right_idx_old = lon_right_idx

                # smoothing things
                self.DepthData["data"] = np.clip(scipy.ndimage.gaussian_filter(self.DepthData["data"], sigma=[3, 3], mode="constant"), 1.0, np.infty)

                if self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon) >= self.OS.critical_depth:
                    break
        else:
            self.DepthData["data"] *= np.clip(np.random.exponential(scale=90, size=1), 20, 700)

        # log
        self.log_Depth = np.log(self.DepthData["data"])

        # for contour plot
        self.con_ticks = np.log([1.0, 2.0, 5.0, 15.0, 50.0, 150.0, 500.0])
        self.con_ticklabels = [int(np.round(tick, 0)) for tick in np.exp(self.con_ticks)]
        self.con_ticklabels[0] = 0
        self.clev = np.arange(0, self.log_Depth.max(), .1)

    def _sample_wind_data(self):
        """Generates random wind data."""
        self.WindData = {}
        self.WindData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1], num=10)
        self.WindData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=10)
  
        speed_mps = np.zeros((len(self.WindData["lat"]), len(self.WindData["lon"])))
        angle = np.zeros_like(speed_mps)

        # size of homogenous wind areas
        lat_n_areas = np.random.randint(5, 20)
        lon_n_areas = np.random.randint(5, 20)

        idx_freq_lat = speed_mps.shape[0] / lat_n_areas
        idx_freq_lon = speed_mps.shape[1] / lon_n_areas

        V_const = np.random.uniform(low=0.0, high=15.0, size=(lat_n_areas, lon_n_areas))
        angle_const = np.random.uniform(low=0.0, high=2*math.pi, size=(lat_n_areas, lon_n_areas))

        # sampling
        for lat_idx, _ in enumerate(self.WindData["lat"]):
            for lon_idx, _ in enumerate(self.WindData["lon"]):

                lat_area = int(lat_idx / idx_freq_lat)
                lat_area = int(lat_n_areas-1 if lat_area >= lat_n_areas else lat_area)

                lon_area = int(lon_idx / idx_freq_lon)
                lon_area = int(lon_n_areas-1 if lon_area >= lon_n_areas else lon_area)

                speed_mps[lat_idx, lon_idx] = V_const[lat_area, lon_area] + np.random.normal(0.0, 1.0)
                angle[lat_idx, lon_idx] = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.WindData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[5, 5], mode="constant"), 0.0, 15.0)
        self.WindData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")

        # overwrite other entries
        e = np.zeros_like(speed_mps)
        n = np.zeros_like(speed_mps)

        for lat_idx, _ in enumerate(self.WindData["lat"]):
            for lon_idx, _ in enumerate(self.WindData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=speed_mps[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.WindData["eastward_mps"] = e
        self.WindData["northward_mps"] = n
        self.WindData["eastward_knots"] = mps_to_knots(self.WindData["eastward_mps"])
        self.WindData["northward_knots"] = mps_to_knots(self.WindData["northward_mps"])

    def _sample_current_data(self):
        """Generates random current data."""
        self.CurrentData = {}
        self.CurrentData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1], num=10)
        self.CurrentData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=10)

        speed_mps = np.zeros((len(self.CurrentData["lat"]), len(self.CurrentData["lon"])))
        angle = np.zeros_like(speed_mps)

        # size of homogenous current areas
        lat_n_areas = np.random.randint(5, 20)
        lon_n_areas = np.random.randint(5, 20)

        idx_freq_lat = speed_mps.shape[0] / lat_n_areas
        idx_freq_lon = speed_mps.shape[1] / lon_n_areas

        V_const = np.clip(np.random.exponential(scale=0.2, size=(lat_n_areas, lon_n_areas)), 0.0, 0.5)
        angle_const = np.random.uniform(low=0.0, high=2*math.pi, size=(lat_n_areas, lon_n_areas))

        # sampling
        for lat_idx, _ in enumerate(self.CurrentData["lat"]):
            for lon_idx, _ in enumerate(self.CurrentData["lon"]):

                # no currents at land
                lat_area = int(lat_idx / idx_freq_lat)
                lat_area = int(lat_n_areas-1 if lat_area >= lat_n_areas else lat_area)

                lon_area = int(lon_idx / idx_freq_lon)
                lon_area = int(lon_n_areas-1 if lon_area >= lon_n_areas else lon_area)

                speed_mps[lat_idx, lon_idx] = V_const[lat_area, lon_area] + np.random.normal(0.0, 0.25)
                angle[lat_idx, lon_idx] = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.CurrentData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[1, 1], mode="constant"), 0.0, 0.5)
        self.CurrentData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")

        # overwrite other entries
        e = np.zeros_like(speed_mps)
        n = np.zeros_like(speed_mps)

        for lat_idx, _ in enumerate(self.CurrentData["lat"]):
            for lon_idx, _ in enumerate(self.CurrentData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=speed_mps[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.CurrentData["eastward_mps"] = e
        self.CurrentData["northward_mps"] = n

    def _sample_wave_data(self):
        """Generates random wave data."""
        self.WaveData = {}
        self.WaveData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1], num=10)
        self.WaveData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=10)

        height = np.zeros((len(self.WaveData["lat"]), len(self.WaveData["lon"])))
        length = np.zeros_like(height)
        period = np.zeros_like(height)
        angle = np.zeros_like(height)

        # size of homogenous wave areas
        lat_n_areas = np.random.randint(5, 20)
        lon_n_areas = np.random.randint(5, 20)

        idx_freq_lat = height.shape[0] / lat_n_areas
        idx_freq_lon = height.shape[1] / lon_n_areas

        height_const = np.random.exponential(scale=0.1, size=(lat_n_areas, lon_n_areas))
        length_const = np.random.exponential(scale=20., size=(lat_n_areas, lon_n_areas))
        period_const = np.random.exponential(scale=1.0, size=(lat_n_areas, lon_n_areas))
        angle_const = np.random.uniform(low=0.0, high=2*math.pi, size=(lat_n_areas, lon_n_areas))

        # sampling
        for lat_idx, _ in enumerate(self.WaveData["lat"]):
            for lon_idx, _ in enumerate(self.WaveData["lon"]):

                lat_area = int(lat_idx / idx_freq_lat)
                lat_area = int(lat_n_areas-1 if lat_area >= lat_n_areas else lat_area)

                lon_area = int(lon_idx / idx_freq_lon)
                lon_area = int(lon_n_areas-1 if lon_area >= lon_n_areas else lon_area)

                height[lat_idx, lon_idx] = height_const[lat_area, lon_area] + np.random.normal(0.0, 0.05)
                length[lat_idx, lon_idx] = length_const[lat_area, lon_area] + np.random.normal(0.0, 5.0)
                period[lat_idx, lon_idx] = period_const[lat_area, lon_area] + np.random.normal(0.0, 0.5)
                angle[lat_idx, lon_idx]  = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.WaveData["height"] = np.clip(scipy.ndimage.gaussian_filter(height, sigma=[0.01, 0.01], mode="constant"), 0.01, 2.0)
        self.WaveData["length"] = np.clip(scipy.ndimage.gaussian_filter(length, sigma=[1, 1], mode="constant"), 1.0, 100.0)
        self.WaveData["period"] = np.clip(scipy.ndimage.gaussian_filter(period, sigma=[0.1, 0.1], mode="constant"), 0.5, 7.0)
        self.WaveData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")

        # overwrite other entries
        e = np.zeros_like(height)
        n = np.zeros_like(height)

        for lat_idx, _ in enumerate(self.WaveData["lat"]):
            for lon_idx, _ in enumerate(self.WaveData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=height[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.WaveData["eastward"] = e
        self.WaveData["northward"] = n

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

    def _get_closeness_from_lidar(self, dists):
        """Computes the closeness following Heiberg et al. (2022, Neural Networks) from given LiDAR distance measurements."""
        return np.clip(1 - np.log(dists+1)/np.log(self.lidar_range+1), 0, 1)

    def _sense_LiDAR(self, N0:float, E0:float, head0:float):
        """Generates an observation via LiDAR sensoring. There are 'lidar_n_beams' equally spaced beams originating from the midship of the OS.
        The first beam is defined in direction of the heading of the OS. Each beam consists of 'n_dots_per_beam' sub-points, which are sequentially considered. 
        Returns for each beam the distance at which insufficient water depth has been detected, where the maximum range is 'lidar_range'.
        Furthermore, it returns the endpoints in lat-lon of each (truncated) beam.
        Args:
            N0(float): north position
            E0(float): east position
            head0(float): heading
        
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
                depth_dot = self._depth_at_latlon(lat_q=lat_dot, lon_q=lon_dot)

                if depth_dot <= self.OS.critical_depth:
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

    def reset(self, OS_wp_idx=20):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # sample global path if no real data is used
        if self.data == "sampled":
            self._sample_global_path()

        # init OS
        lat_init = self.GlobalPath.lat[OS_wp_idx]
        lon_init = self.GlobalPath.lon[OS_wp_idx]
        N_init = self.GlobalPath.north[OS_wp_idx]
        E_init = self.GlobalPath.east[OS_wp_idx]

        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = None,
                         u_init    = 0.0,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = np.random.uniform(0.8, 1.2) * 3.0,
                         full_ship = False,
                         ship_domain_size = 2)
        self.OS.rev_dir = False

        # real data: determine where we are
        if self.data == "real":
            self.plan_on_river = self._on_river(N0=self.OS.eta[0], E0=self.OS.eta[1])

        # on river: add reversed global path for TS spawning on rivers
        # Note: if real data is used, we might start at the open sea but later arrive at a river, so we need the RevGlobalPath as well
        if self.data == "real" or self.plan_on_river:
            self.RevGlobalPath = deepcopy(self.GlobalPath)
            self.RevGlobalPath.reverse(offset=self.dist_des_rev_path)

        # init waypoints and cte of OS for global path
        self.OS = self._init_wps(self.OS, "global")
        self._set_cte(path_level="global")

        # init local path
        self._init_local_path()

        # init waypoints and cte of OS for local path
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")

        # set heading with noise
        if type(self).__name__ != "HHOS_PathFollowing_Validation":
            self.OS.eta[2] = angle_to_2pi(self.glo_pi_path + dtr(np.random.uniform(-25.0, 25.0)))
        else:
            self.OS.eta[2] = self.glo_pi_path

        # generate random environmental data
        if self.data == "sampled":
            self._sample_depth_data(lat_init, lon_init)
            self._sample_wind_data()
            self._sample_current_data()
            self._sample_wave_data()

        # environmental effects
        self._update_disturbances(lat_init, lon_init)

        # set u-speed to near-convergence
        self.OS.nu[0] = self.OS._get_u_from_nps(nps         = self.OS.nps, 
                                                psi         = self.OS.eta[2], 
                                                V_c         = self.V_c, 
                                                beta_c      = self.beta_c, 
                                                V_w         = self.V_w, 
                                                beta_w      = self.beta_w, 
                                                H           = self.H,
                                                beta_wave   = self.beta_wave, 
                                                eta_wave    = self.eta_wave, 
                                                T_0_wave    = self.T_0_wave, 
                                                lambda_wave = self.lambda_wave)
        # set course error
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # initially compute ship domain for plotting
        rads  = np.linspace(0.0, 2*math.pi, 100)
        dists = [get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
            OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        # init other vessels
        self._init_TSs()

        # init state
        self._set_state()
        self.state_init = self.state

        # viz
        if hasattr(self, "plotter"):
            self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                    OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                        glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                        T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                                rud_angle=self.OS.rud_angle, nps=self.OS.nps)
        return self.state

    def _init_wps(self, vessel:KVLCC2, path_level:str):
        """Initializes the waypoints on the global and local path, respectively, based on the position of the vessel.
        Returns the vessel."""
        assert path_level in ["global", "local"], "Unknown path level."

        if path_level == "global":
            path = self.RevGlobalPath if vessel.rev_dir else self.GlobalPath
            vessel.glo_wp1_idx, vessel.glo_wp1_N, vessel.glo_wp1_E, vessel.glo_wp2_idx, vessel.glo_wp2_N, \
                vessel.glo_wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=vessel.eta[0], a_e=vessel.eta[1])
            try:
                vessel.glo_wp3_idx = vessel.glo_wp2_idx + 1
                vessel.glo_wp3_N = path.north[vessel.glo_wp3_idx] 
                vessel.glo_wp3_E = path.east[vessel.glo_wp3_idx]
            except:
                raise ValueError("Waypoint setting fails if vessel is not at least two waypoints away from the goal.")
        else:
            if vessel.rev_dir:
                raise ValueError("Reversed direction for a local path can never happen.")

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

    def _init_local_path(self):
        """Generates a local path based on the global one."""
        self.LocalPath = self.GlobalPath.construct_local_path(wp_idx      = self.OS.glo_wp1_idx, 
                                                              n_wps_loc   = self.n_wps_loc,
                                                              two_actions = self.two_actions, 
                                                              desired_V   = self.desired_V)
        self.planning_method = "global"

    def _update_local_path(self):
        self._init_local_path()

    def _init_TSs(self):
        if self.plan_on_river:

            # sample a scenario on river
            if self.scenario_based:
                p_scene = [0.5, 0.125, 0.125, 0.125, 0.125]
            else:
                p_scene = [1.0, 0.0, 0.0, 0.0, 0.0]
            self.scene = np.random.choice([0, 1, 2, 3, 4], p=p_scene)

            # all random
            if self.scene == 0.0:
                if self.N_TSs_random:
                    self.N_TSs = np.random.choice(self.N_TSs_max)
                else:
                    self.N_TSs = self.N_TSs_max

            # vessel train
            elif self.scene == 1:
                self.N_TSs = self.N_TSs_max
            
            # overtake the overtaker
            elif self.scene == 2:
                self.N_TSs = 2

            # overtaking under oncoming traffic
            elif self.scene == 3:
                self.N_TSs = 5
            
            # overtake the overtaker under oncoming traffic
            elif self.scene == 4:
                self.N_TSs = 3
        else:
            if self.N_TSs_random:
                self.N_TSs = np.random.choice(self.N_TSs_max)
            else:
                self.N_TSs = self.N_TSs_max

        # sample TSs
        self.TSs : List[TargetShip]= []
        for n in range(self.N_TSs):
            if self.plan_on_river:
                self.TSs.append(self._get_TS_river(scene=self.scene, n=n))
            else:
                self.TSs.append(self._get_TS_open_sea())

    def _get_TS_open_sea(self):
        """Places a target ship by sampling a 
            1) COLREG situation,
            2) TCPA (in s, or setting to 60s),
            3) relative bearing (in rad), 
            4) intersection angle (in rad),
            5) and a forward thrust (tau-u in N).
        Returns: 
            TargetShip"""
        TS = TargetShip(N_init    = 0.0,
                        E_init    = 0.0,
                        psi_init  = 0.0,
                        u_init    = 0.0,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = np.random.uniform(0.8, 1.2) * self.OS.nps,
                        full_ship = False,
                        ship_domain_size = 2)

        # predict converged speed of sampled TS
        # Note: if we don't do this, all further calculations are heavily biased
        TS.nu[0] = TS._get_u_from_nps(TS.nps, psi=TS.eta[2])

        # quick access for OS
        N0, E0, _ = self.OS.eta
        chiOS = self.OS._get_course()
        VOS   = self.OS._get_V()

        # sample COLREG situation 
        # head-on = 1, starb. cross. = 2, ports. cross. = 3, overtaking = 4, null = 5
        COLREG_s = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.2])

        # determine relative speed of OS towards goal, need absolute bearing first
        try:
            g_idx = self.OS.glo_wp3_idx + 3
            g_n = self.GlobalPath.north[g_idx]
            g_e = self.GlobalPath.east[g_idx]
        except:
            g_idx = self.OS.glo_wp3_idx
            g_n = self.GlobalPath.north[g_idx]
            g_e = self.GlobalPath.east[g_idx]
        bng_abs_goal = bng_abs(N0=N0, E0=E0, N1=g_n, E1=g_e)

        # project VOS vector on relative velocity direction
        VR_goal_x, VR_goal_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=bng_abs_goal)
        
        # sample time
        t_hit = np.random.uniform(self.TCPA_crit * 0.75, self.TCPA_crit)

        # compute hit point
        E_hit = E0 + VR_goal_x * t_hit
        N_hit = N0 + VR_goal_y * t_hit

        # Note: In the following, we only sample the intersection angle and not a relative bearing.
        #       This is possible since we construct the trajectory of the TS so that it will pass through (E_hit, N_hit), 
        #       and we need only one further information to uniquely determine the origin of its motion.

        # head-on
        if COLREG_s == 1:
            C_TS_s = dtr(np.random.uniform(175, 185))

        # starboard crossing
        elif COLREG_s == 2:
            C_TS_s = dtr(np.random.uniform(185, 292.5))

        # portside crossing
        elif COLREG_s == 3:
            C_TS_s = dtr(np.random.uniform(67.5, 175))

        # overtaking
        elif COLREG_s == 4:
            C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

        # null: TS comes from behind
        elif COLREG_s == 5:
                C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

        # determine TS heading (treating absolute bearing towards goal as heading of OS)
        head_TS_s = angle_to_2pi(C_TS_s + bng_abs_goal)   

        # no speed constraints except in overtaking
        if COLREG_s in [1, 2, 3, 5]:
            VTS = TS.nu[0]

        elif COLREG_s == 4:

            # project VOS vector on TS direction
            VR_TS_x, VR_TS_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=head_TS_s)
            V_max_TS = polar_from_xy(x=VR_TS_x, y=VR_TS_y, with_r=True, with_angle=False)[0]

            # sample TS speed
            VTS = np.random.uniform(0.3, 0.7) * V_max_TS
            TS.nu[0] = VTS

            # set nps of TS so that it will keep this velocity
            TS.nps = TS._get_nps_from_u(VTS, psi=TS.eta[2])

        # backtrace original position of TS
        E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
        N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

        # set positional values
        TS.eta = np.array([N_TS, E_TS, head_TS_s], dtype=np.float32)

        # set attribute rev_dir since we might move from open sea to river and need it
        TS.rev_dir = bool(random.getrandbits(1))
        return TS

    def _get_TS_river(self, scene, n=None):
        """Places a target ship by setting a 
            1) traveling direction,
            2) distance on the global path,
        depending on the scene. All ships spawn in front of the agent.
        Args:
            scene (int):  considered scenario
            n (int):      index of the spawned vessel
        Returns: 
            KVLCC2."""
        assert not (scene != 0 and n is None), "Need to provide index in non-random scenario-based spawning."

        #------------------ set distances, directions, offsets from path, and nps ----------------------
        # Note: An offset is some float. If it is negative (positive), the vessel is placed on the 
        #       right (left) side of the global path.

        # random
        if scene == 0:
            speedy = bool(np.random.choice([0, 1], p=[0.8, 0.2]))
            if speedy: 
                d = np.random.uniform(self.TS_spawn_dists[0]/2, self.TS_spawn_dists[0])
                rev_dir = False
                nps = np.random.uniform(1.3, 1.5) * self.OS.nps
            else:
                d = np.random.uniform(*self.TS_spawn_dists)
                rev_dir = bool(random.getrandbits(1))
                nps = np.random.uniform(0.3, 0.6) * self.OS.nps
            offset = np.random.uniform(-20.0, 50.0)

        # vessel train
        if scene == 1:
            if n == 0:
                d = NM_to_meter(0.5)
            else:
                d = NM_to_meter(0.5) + n*NM_to_meter(0.1)
            rev_dir = False
            offset = 0.0
            nps = np.random.uniform(0.4, 0.5) * self.OS.nps
            speedy = False

        # overtake the overtaker
        elif scene == 2:
            d = NM_to_meter(0.5)
            rev_dir = False
            offset = 0.0 if n == 0 else 100.0
            if n == 0:
                nps = np.random.uniform(0.4, 0.45) * self.OS.nps
            else:
                nps = np.random.uniform(0.45, 0.5) * self.OS.nps
            speedy = False

        # overtaking under oncoming traffic
        elif scene == 3:
            if n == 0:
                d = NM_to_meter(0.5)
                rev_dir = False
                offset = 0.0
                nps = np.random.uniform(0.4, 0.5) * self.OS.nps

            elif n == 1:
                d = NM_to_meter(1.5)
                rev_dir = True
                offset = 50.0
                nps = np.random.uniform(0.6, 0.7) * self.OS.nps

            elif n == 2:
                d = NM_to_meter(1.5)
                rev_dir = True
                offset = 0.0
                nps = np.random.uniform(0.6, 0.7) * self.OS.nps

            elif n == 3:
                d = NM_to_meter(1.7)
                rev_dir = True
                offset = 0.0
                nps = np.random.uniform(0.6, 0.7) * self.OS.nps

            elif n == 4:
                d = NM_to_meter(1.7)
                rev_dir = True
                offset = 50.0
                nps = np.random.uniform(0.6, 0.7) * self.OS.nps
            speedy = False

        # overtake the overtaker under oncoming traffic
        elif scene == 4:
            if n == 0:
                d = NM_to_meter(0.5)
                rev_dir = False
                offset = 0.0
                nps = np.random.uniform(0.4, 0.45) * self.OS.nps

            elif n == 1:
                d = NM_to_meter(0.5)
                rev_dir = False
                offset = 50.0
                nps = np.random.uniform(0.45, 0.5) * self.OS.nps

            elif n == 2:
                d = NM_to_meter(1.5)
                rev_dir = True
                offset = 0.0
                nps = np.random.uniform(0.6, 0.7) * self.OS.nps
            speedy = False

        # add some noise to distance
        pass

        # get wps
        if speedy:
            wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = self.RevGlobalPath.north, 
                                                                   e_array = self.RevGlobalPath.east, 
                                                                   a_n     = self.OS.eta[0], 
                                                                   a_e     = self.OS.eta[1])
            path = self.RevGlobalPath
        else:
            wp1 = self.OS.glo_wp1_idx
            wp1_N = self.OS.glo_wp1_N
            wp1_E = self.OS.glo_wp1_E

            wp2 = self.OS.glo_wp2_idx
            wp2_N = self.OS.glo_wp2_N
            wp2_E = self.OS.glo_wp2_E

            path = self.GlobalPath

        # determine starting position
        ate_init = ate(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])
        d_to_nxt_wp = path.wp_dist(wp1, wp2) - ate_init
        orig_seg = True

        while True:
            if d > d_to_nxt_wp:
                d -= d_to_nxt_wp
                wp1 += 1
                wp2 += 1
                d_to_nxt_wp = path.wp_dist(wp1, wp2)
                orig_seg = False
            else:
                break

        # path angle
        pi_path_spwn = bng_abs(N0=path.north[wp1], E0=path.east[wp1], N1=path.north[wp2], E1=path.east[wp2])

        # still in original segment
        if orig_seg:
            E_add, N_add = xy_from_polar(r=ate_init+d, angle=pi_path_spwn)
        else:
            E_add, N_add = xy_from_polar(r=d, angle=pi_path_spwn)

        # determine position
        N_TS = path.north[wp1] + N_add
        E_TS = path.east[wp1] + E_add
        
        # jump on the other path: either due to speedy or opposing traffic
        if speedy or rev_dir:
            E_add_rev, N_add_rev = xy_from_polar(r=self.dist_des_rev_path, angle=angle_to_2pi(pi_path_spwn-math.pi/2))
            N_TS += N_add_rev
            E_TS += E_add_rev

        # consider offset
        TS_head = angle_to_2pi(pi_path_spwn + math.pi) if rev_dir or speedy else pi_path_spwn

        if offset != 0.0:
            ang = TS_head - math.pi/2 if offset > 0.0 else TS_head + math.pi/2
            E_add_rev, N_add_rev = xy_from_polar(r=offset, angle=angle_to_2pi(ang))
            N_TS += N_add_rev
            E_TS += E_add_rev

        # generate TS
        TS = TargetShip(N_init    = N_TS,
                        E_init    = E_TS,
                        psi_init  = TS_head,
                        u_init    = 0.0,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = nps,
                        full_ship = False,
                        ship_domain_size = 2)

        # store waypoint information
        TS.rev_dir = rev_dir

        if speedy:
            wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = self.GlobalPath.north, 
                                                                   e_array = self.GlobalPath.east, 
                                                                   a_n     = TS.eta[0], 
                                                                   a_e     = TS.eta[1])
            TS.glo_wp1_idx = wp1
            TS.glo_wp1_N = wp1_N
            TS.glo_wp1_E = wp1_E
            TS.glo_wp2_idx = wp2
            TS.glo_wp2_N = wp2_N
            TS.glo_wp2_E = wp2_E
            TS.glo_wp3_idx = wp2 + 1
            TS.glo_wp3_N = self.GlobalPath.north[wp2 + 1]
            TS.glo_wp3_E = self.GlobalPath.east[wp2 + 1]
        else:
            if rev_dir:
                TS.glo_wp1_idx, TS.glo_wp2_idx = self.GlobalPath.get_rev_path_wps(wp1, wp2)
                TS.glo_wp3_idx = TS.glo_wp2_idx + 1
                path = self.RevGlobalPath
            else:
                TS.glo_wp1_idx, TS.glo_wp2_idx, TS.glo_wp3_idx = wp1, wp2, wp2 + 1
                path = self.GlobalPath

            TS.glo_wp1_N, TS.glo_wp1_E = path.north[TS.glo_wp1_idx], path.east[TS.glo_wp1_idx]
            TS.glo_wp2_N, TS.glo_wp2_E = path.north[TS.glo_wp2_idx], path.east[TS.glo_wp2_idx]
            TS.glo_wp3_N, TS.glo_wp3_E = path.north[TS.glo_wp3_idx], path.east[TS.glo_wp3_idx]

        # predict converged speed of sampled TS
        TS.nu[0] = TS._get_u_from_nps(TS.nps, psi=TS.eta[2])
        return TS

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
        # check whether TS is out of sight
        if ED(N0=N0, E0=E0, N1=N1, E1=E1) > self.lidar_range:
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
        if (sit in [1, 2]) and r0 < 0.0:

            # evaluate only if TCPA with TS is positive
            if tcpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=chi0, chiTS=chi1, VOS=v0, VTS=v1) >= 0.0:
                return True
        return False

    def _violates_river_traffic_rules(self, N0:float, E0:float, head0:float, v0:float, N1:float, E1:float, head1:float,\
        v1:float, Lpp:float) -> bool:
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
            Lpp(float):    length between perpendiculars of OS
            """
        # preparation
        ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
        bng_rel_TS_pers = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)
        rev_dir = (abs(head_inter(head_OS=head0, head_TS=head1, to_2pi=False)) >= dtr(90.0))

        # OS should let speedys pass on OS's portside
        if v1 > v0:
            if dtr(180.0) <= bng_rel_TS_pers <= dtr(270.0) and ED_OS_TS <= 10*Lpp:
                return True

        # OS should not pass opposing ships on their portside
        if rev_dir:
            if dtr(0.0) <= bng_rel_TS_pers <= dtr(90.0) and ED_OS_TS <= 10*Lpp:
                return True

        # normal target ships should be overtaken on their portside
        if v0 > v1:
            if dtr(90.0) <= bng_rel_TS_pers <= dtr(180.0):
                l = (35 - 2/math.pi *15* bng_rel_TS_pers)*Lpp
                if ED_OS_TS <= l:
                    return True
        return False

    def _on_river(self, N0:float, E0:float):
        """Checks whether we are on river or on open sea, depending on surroundings.
        
        Returns:
            bool, True if on river, False if on open sea"""
        dists, _, _, _ = self._sense_LiDAR(N0=N0, E0=E0, head0=0.0)
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
        N0, E0, _ = self.OS.eta
        N1, E1, _ = TS.eta

        TCPA = tcpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=self.OS._get_course(), chiTS=TS._get_course(),\
             VOS=self.OS._get_V(), VTS=TS._get_V())
        dist = ED(N0=N0, E0=E0, N1=N1, E1=E1)

        # on river
        if self.plan_on_river:
            if (TCPA < 0.0) and (dist >= NM_to_meter(0.5)):
                return self._get_TS_river(scene=0, n=None)
        # open sea
        else:
            if (TCPA < 0.0) and (dist >= self.lidar_range):
                return self._get_TS_open_sea()
        return TS

    def _update_disturbances(self, OS_lat=None, OS_lon=None):
        """Updates the environmental disturbances at the agent's current position."""
        if OS_lat is None and OS_lon is None:
            OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=32)

        self.V_c, self.beta_c = self._current_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.V_w, self.beta_w = self._wind_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.H = self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = self._wave_at_latlon(lat_q=OS_lat, lon_q=OS_lon)

        # consider wave data issues
        if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
             [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
            self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None
        
        elif self.T_0_wave == 0.0:
            self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None

    def step(self, a):
        pass

    def _set_state(self):
        self.state = 0.0
        return

    def _calculate_reward(self, a):
        self.r = 0
        return

    def _done(self):
        return False

    def __str__(self, OS_lat, OS_lon) -> str:
        u, v, r = self.OS.nu
        course = self.OS._get_course()

        ste = f"Step: {self.step_cnt}"
        pos = f"Lat [°]: {OS_lat:.4f}, Lon [°]: {OS_lon:.4f}, " + r"$\psi$ [°]: " + f"{rtd(self.OS.eta[2]):.2f}"  + r", $\chi$ [°]: " + f"{rtd(course):.2f}"
        vel = f"u [m/s]: {u:.3f}, v [m/s]: {v:.3f}, r [rad/s]: {r:.3f}"
        
        depth = f"H [m]: {self.H:.2f}"
        wind = r"$V_{\rm wind}$" + f" [kn]: {mps_to_knots(self.V_w):.2f}, " + r"$\psi_{\rm wind}$" + f" [°]: {rtd(self.beta_w):.2f}"
        current = r"$V_{\rm current}$" + f" [m/s]: {self.V_c:.2f}, " + r"$\psi_{\rm current}$" + f" [°]: {rtd(self.beta_c):.2f}"
        
        if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
             [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
            wave = r"$\psi_{\rm wave}$" + f" [°]: -" + r", $\xi_{\rm wave}$ [m]: " + f"-" \
                + r", $T_{\rm wave}$ [s]: " + f"-" + r", $\lambda_{\rm wave}$ [m]: " + f"-"
        else:
            wave = r"$\psi_{\rm wave}$" + f" [°]: {rtd(self.beta_wave):.2f}" + r", $\xi_{\rm wave}$ [m]: " + f"{self.eta_wave:.2f}" \
                + r", $T_{\rm wave}$ [s]: " + f"{self.T_0_wave:.2f}" + r", $\lambda_{\rm wave}$ [m]: " + f"{self.lambda_wave:.2f}"

        glo_path_info = "Global path: " + r"$y_e$" + f" [m]: {self.glo_ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.glo_desired_course):.2f}, " \
            + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.glo_course_error):.2f}"
        
        if hasattr(self, "LocalPath"):
            loc_path_info = "Local path: " + r"$y_e$" + f" [m]: {self.loc_ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.loc_desired_course):.2f}, " \
                + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.loc_course_error):.2f}"
        out = ste + ", " + pos + "\n" + vel + ", " + depth + "\n" + wind + ", " + current + "\n" + wave + "\n" + glo_path_info

        if hasattr(self, "LocalPath"):
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
        if not self.plan_on_river and (vessel is not self.OS):
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

        # collision risk and metrics
        if plot_CR:
            DCPA, TCPA, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=self.OS._get_course(),\
                chiTS=vessel._get_course(), VOS=self.OS._get_V(), VTS=vessel._get_V(), get_positions=True)

            # substract ship domain at TCPA = 0 from DCPA
            bng_rel_tcpa_from_OS_pers = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
            domain_tcpa = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C,\
                D=self.OS.ship_domain_D, OS=None, TS=None, ang=bng_rel_tcpa_from_OS_pers)
            DCPA = max([0.0, DCPA-domain_tcpa])

        if self.plot_in_latlon:

            # convert rectangle points to lat/lon
            A_lat, A_lon = to_latlon(north=A[1], east=A[0], number=32)
            B_lat, B_lon = to_latlon(north=B[1], east=B[0], number=32)
            C_lat, C_lon = to_latlon(north=C[1], east=C[0], number=32)
            D_lat, D_lon = to_latlon(north=D[1], east=D[0], number=32)

            # draw the polygon (A is included twice to create a closed shape)
            lons = [A_lon, B_lon, D_lon, C_lon, A_lon]
            lats = [A_lat, B_lat, D_lat, C_lat, A_lat]
            ax.plot(lons, lats, color=color, linewidth=2.0)

            # plot ship domain
            if with_domain:
                lat_lon_tups = [to_latlon(north=y, east=x, number=32)[:2] for x, y in xys]
                lats = [e[0] for e in lat_lon_tups]
                lons = [e[1] for e in lat_lon_tups]
                ax.plot(lons, lats, color=color, alpha=0.7)

            if plot_CR:
                CR_x = min(lons) - np.abs(min(lons) - max(lons))
                CR_y = min(lats) - 2*np.abs(min(lats) - max(lats))
                ax.text(CR_x, CR_y, f"DCPA: {DCPA:.2f}" + "\n" + f"TCPA: {TCPA:.2f}",\
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
                ax.text(CR_x, CR_y, f"DCPA: {DCPA:.2f}" + "\n" + f"TCPA: {TCPA:.2f}",\
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
            # ------------------------------ ship movement --------------------------------
            # get position of OS in lat/lon
            N0, E0, head0 = self.OS.eta
            OS_lat, OS_lon = to_latlon(north=N0, east=E0, number=32)

            for ax in [self.ax1]:
                ax.clear()

                # general information
                if self.plot_in_latlon:
                    ax.text(0.125, 0.8875, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=9, transform=plt.gcf().transFigure)

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
                if self.plot_depth and self.plot_in_latlon:
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
                if self.plot_wind and self.plot_in_latlon:
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
                for TS in self.TSs:
                    if self.plan_on_river:
                        col = "darkgoldenrod" if TS.rev_dir else "purple"
                    else:
                        col = None
                    ax = self._render_ship(ax=ax, vessel=TS, color=col, plot_CR=False)

                # set legend for COLREGS
                if not self.plan_on_river:
                    legend1 = ax.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], 
                                        fontsize=8, loc='upper center', ncol=6)
                    ax.add_artist(legend1)

                #--------------------- Path ------------------------
                if self.plot_path:
                    loc_path_col = "salmon" if self.data == "sampled" else "lightsteelblue"

                    if self.plot_in_latlon:
                        # global
                        ax.plot(self.GlobalPath.lon, self.GlobalPath.lat, marker='o', color="purple", linewidth=1.0, markersize=3, label="Global Path")
                        if type(self).__name__ != "HHOS_PathFollowing_Validation" and self.plan_on_river:
                            ax.plot(self.RevGlobalPath.lon, self.RevGlobalPath.lat, marker='o', color="darkgoldenrod", linewidth=1.0, markersize=3, label="Reversed Global Path")

                        # local
                        if hasattr(self, "LocalPath"):
                            if "PathFollowing" in type(self).__name__:
                                label = "Local Path" + f" ({self.planning_method})"
                            else:
                                label = "Local Path"
                            ax.plot(self.LocalPath.lon, self.LocalPath.lat, marker='o', color=loc_path_col, linewidth=1.0, markersize=3, label=label)
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
                        if type(self).__name__ != "HHOS_PathFollowing_Validation" and self.plan_on_river:
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
                if self.plot_current and self.plot_in_latlon:
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
                if self.plot_waves and self.plot_in_latlon:
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
                if self.plot_lidar and self.plot_in_latlon:
                    lidar_lat_lon = self._sense_LiDAR(N0=N0, E0=E0, head0=head0)[1]

                    for _, latlon in enumerate(lidar_lat_lon):
                        ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.4)#, alpha=(idx+1)/len(lidar_lat_lon))

            # ------------------------------ reward and action plot --------------------------------
            if self.plot_reward:
                if self.step_cnt == 0:
                    # reward
                    self.ax2.clear()
                    self.ax2.old_time = 0
                    self.ax2.old_r_ye = 0
                    self.ax2.old_r_ce = 0
                    self.ax2.old_r_coll = 0
                    self.ax2.old_r_comf = 0
                    if self.two_actions:
                        self.ax2.old_r_speed = 0
                    self.ax2.r = 0
                    self.ax2.set_title(type(self).__name__.replace("_", "-"))

                    # action
                    self.ax3.clear()
                    self.ax3.old_time = 0
                    self.ax3.old_a0 = 0
                    if self.two_actions:
                        self.ax3.old_a1 = 0

                # reward
                self.ax2.set_xlabel("Timestep in episode")
                self.ax2.set_ylabel("Reward")

                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ye, self.r_ye], color = "black", label="Cross-track error")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ce, self.r_ce], color = "red", label="Course error")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_comf, self.r_comf], color = "blue", label="Comfort")

                if type(self).__name__ == "HHOS_PathPlanning_Env":
                    self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_coll, self.r_coll], color = "green", label="Collision")
                if self.two_actions:
                    self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_speed, self.r_speed], color = "salmon", label="Speed")

                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.r, self.r], color = "darkgoldenrod", label="Aggregated")
                
                if self.step_cnt == 0:
                    self.ax2.legend()

                self.ax2.old_time = self.step_cnt
                self.ax2.old_r_ye = self.r_ye
                self.ax2.old_r_ce = self.r_ce
                self.ax2.old_r_coll = self.r_coll
                self.ax2.old_r_comf = self.r_comf
                if self.two_actions:
                    self.ax2.old_r_speed = self.r_speed
                self.ax2.r = self.r

                # action
                self.ax3.set_xlabel("Timestep in episode")
                self.ax3.set_ylabel("Action")
                self.ax3.set_ylim(-1.05, 1.05)

                self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_a0, float(self.a[0])], color="blue", label="Steering")
                if self.two_actions:
                    self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_a1, float(self.a[1])], color="red", label="Speed")

                if self.step_cnt == 0:
                    self.ax3.legend()

                self.ax3.old_time = self.step_cnt
                self.ax3.old_a0 = float(self.a[0])
                if self.two_actions:
                    self.ax3.old_a1 = float(self.a[1])

            #plt.gca().set_aspect('equal')
            plt.pause(0.001)
