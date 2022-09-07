import copy
import math
import pickle
import random

import gym
import numpy as np
import pandas as pd
import scipy.ndimage
from gym import spaces
from matplotlib import cm
from matplotlib import pyplot as plt
from tud_rl.envs._envs.HHOS_Fnc import (VFG, Z_at_latlon, ate, fill_array,
                                        find_nearest, get_init_two_wp,
                                        mps_to_knots, switch_wp, to_latlon,
                                        to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (ED, NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, get_ship_domain, head_inter, rtd,
                                         tcpa, xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 vessel from Hamburg to Oslo."""

    def __init__(self, mode="train", N_TSs_max=3, N_TSs_random=True):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0   # simulation time interval (in s)

        # LiDAR
        self.lidar_range       = NM_to_meter(1.0)                                                # range of LiDAR sensoring in m
        #self.lidar_n_beams     = 25                                                              # number of beams
        #self.lidar_beam_angles = np.linspace(0.0, 2*math.pi, self.lidar_n_beams, endpoint=False) # beam angles

        self.lidar_beam_angles = [20.0, 45., 90., 135.0]
        self.lidar_beam_angles += [360. - ang for ang in self.lidar_beam_angles] + [0.0, 180.0]
        self.lidar_beam_angles = np.deg2rad(np.sort(self.lidar_beam_angles))

        self.lidar_n_beams   = len(self.lidar_beam_angles)
        self.n_dots_per_beam = 10                                                              # number of subpoints per beam
        self.d_dots_per_beam = np.linspace(start=0.0, stop=self.lidar_range, num=self.lidar_n_beams+1, endpoint=True)[1:] # distances from midship of subpoints per beam

        # vector field guidance
        self.VFG_K = 0.001

        # data range
        self.lon_lims = [4.83, 14.33]
        self.lat_lims = [51.83, 60.5]
        self.lon_range = self.lon_lims[1] - self.lon_lims[0]
        self.lat_range = self.lat_lims[1] - self.lat_lims[0]

        # data loading
        path_to_HHOS = "C:/Users/localadm/Desktop/Forschung/RL_packages/HHOS"

        self._load_desired_path(path_to_HHOS)
        self._load_depth_data(path_to_HHOS + "/DepthData")
        self._load_wind_data(path_to_HHOS + "/winds")
        self._load_current_data(path_to_HHOS + "/currents")
        self._load_wave_data(path_to_HHOS + "/waves")

        # setting
        assert mode in ["train", "validate"], "Unknown HHOS mode. Can either train or validate."
        self.mode = mode

        if self.mode == "train":
            self.n_wps = 250
            self.river_dist_loc = 70 # 10_000
            self.river_dist_sca = 20  # 2_000
            self.river_dist_noise_loc = 5 # 100
            self.river_dist_noise_sca = 2 # 50
            self.river_min = 5

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.7

        # visualization
        self.plot_in_latlon = True         # if false, plots in UTM coordinates
        self.plot_depth = True
        self.plot_path = True
        self.plot_wind = True
        self.plot_current = True
        self.plot_waves = True
        self.plot_lidar = True
        self.plot_reward = False
        self.default_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]

        if not self.plot_in_latlon:
            self.show_lon_lat = np.clip(self.show_lon_lat, 0.005, 5.95)
            self.UTM_viz_range_E = abs(to_utm(lat=52.0, lon=6.0001)[1] - to_utm(lat=52.0, lon=6.0001+self.show_lon_lat/2)[1])
            self.UTM_viz_range_N = abs(to_utm(lat=50.0, lon=8.0)[0] - to_utm(lat=50.0+self.show_lon_lat/2, lon=8.0)[0])

        # other ships
        self.N_TSs_max    = N_TSs_max                  # maximum number of other vessels
        self.N_TSs_random = N_TSs_random               # if true, samples a random number in [0, N_TSs] at start of each episode
                                                       # if false, always have N_TSs_max
        self.TCPA_respawn = 120                        # (negative) TCPA in seconds considered as respawning condition
        self.TS_spawn_dists = [NM_to_meter(0.1), NM_to_meter(0.75)]

        # CR calculation
        self.CR_rec_dist = 300                   # collision risk distance [m]
        self.CR_al = 0.1                         # collision risk metric normalization

        # gym inherits
        path_info_size = 16
        TS_info_size = 5
        obs_size = path_info_size + self.lidar_n_beams + TS_info_size * self.N_TSs_max

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        self.r = 0
        self.r_ye = 0
        self.r_ce = 0
        self.r_coll = 0
        self.r_comf = 0
        self._max_episode_steps = 1500


    def _load_desired_path(self, path_to_desired_path):
        with open(f"{path_to_desired_path}/Path_latlon.pickle", "rb") as f:
            self.DesiredPath = pickle.load(f)
        
        # store number of waypoints
        self.DesiredPath["n_wps"] = len(self.DesiredPath["lat"])
        self.n_wps = self.DesiredPath["n_wps"]

        # add utm coordinates
        path_n = np.zeros_like(self.DesiredPath["lat"])
        path_e = np.zeros_like(self.DesiredPath["lon"])

        for idx in range(len(path_n)):
            path_n[idx], path_e[idx], _ = to_utm(lat=self.DesiredPath["lat"][idx], lon=self.DesiredPath["lon"][idx])
        
        self.DesiredPath["north"] = path_n
        self.DesiredPath["east"] = path_e


    def _load_depth_data(self, path_to_depth_data):
        with open(f"{path_to_depth_data}/DepthData.pickle", "rb") as f:
            self.DepthData = pickle.load(f)

        # logarithm
        Depth_tmp = copy.copy(self.DepthData["data"])
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


    def _sample_desired_path(self, l=0.001):
        """Constructs a path with n_wps way points, each being of length l apart from its neighbor in the lat-lon-system.
        The agent should follows the path always in direction of increasing indices."""
        self.DesiredPath = {"n_wps" : self.n_wps}

        # do it until we have a path whichs stays in our simulation domain
        while True:

            # sample starting point
            lat = np.zeros(self.n_wps)
            lon = np.zeros(self.n_wps)
            lat[0] = 56.0
            lon[0] = 9.0

            # sample other points
            ang = np.random.uniform(0, 2*math.pi)
            ang_diff = dtr(np.random.uniform(-20., 20.))
            ang_diff2 = 0.0
            for n in range(1, self.n_wps):
                
                # next angle
                ang_diff2 = 0.5 * ang_diff2 + 0.5 * dtr(np.random.uniform(-5.0, 5.0))
                ang_diff = 0.5 * ang_diff + 0.5 * ang_diff2 + 0.0 * dtr(np.random.uniform(-5.0, 5.0))
                ang = angle_to_2pi(ang + ang_diff)

                # next point
                lon_diff, lat_diff = xy_from_polar(r=l, angle=ang)
                lat[n] = lat[n-1] + lat_diff
                lon[n] = lon[n-1] + lon_diff

            if all(self.lat_lims[0] <= lat) and all(self.lat_lims[1] >= lat) and \
                all(6.1 <= lon) and all(11.9 >= lon):
                break

        self.DesiredPath["lat"] = lat
        self.DesiredPath["lon"] = lon

        # add utm coordinates
        path_n = np.zeros_like(self.DesiredPath["lat"])
        path_e = np.zeros_like(self.DesiredPath["lon"])

        for idx in range(len(path_n)):
            path_n[idx], path_e[idx], _ = to_utm(lat=self.DesiredPath["lat"][idx], lon=self.DesiredPath["lon"][idx])
        
        self.DesiredPath["north"] = path_n
        self.DesiredPath["east"] = path_e

        # overwrite data range
        self.off = 15*l
        self.lat_lims = [np.min(lat)-self.off, np.max(lat)+self.off]
        self.lon_lims = [np.min(lon)-self.off, np.max(lon)+self.off]


    def _sample_depth_data(self, OS_lat, OS_lon):
        """Generates random depth data by overwriting the real data information."""
        # clear real data
        self.DepthData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1] + self.off, num=500)
        self.DepthData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=500)
        self.DepthData["data"] = np.ones((len(self.DepthData["lat"]), len(self.DepthData["lon"])))

        while True:
            # sample distances to waypoints
            d_left  = np.zeros(self.n_wps)
            d_right = np.zeros(self.n_wps)
            depth = np.zeros(self.n_wps)

            for i in range(self.n_wps):
                if i == 0:
                    d_left[i]  = np.clip(np.random.normal(loc=self.river_dist_loc, scale=self.river_dist_sca, size=1), self.river_min, np.infty)
                    d_right[i] = np.clip(np.random.normal(loc=self.river_dist_loc, scale=self.river_dist_sca, size=1), self.river_min, np.infty)
                    depth[i] = np.clip(np.random.exponential(scale=15, size=1), 20, 100)
                else:
                    d_left[i]  = np.clip(
                                    d_left[i-1] + np.random.normal(loc=self.river_dist_noise_loc, scale=self.river_dist_noise_sca, size=1),
                                self.river_min, 3*self.river_dist_loc)
                    d_right[i] = np.clip(
                                    d_right[i-1] + np.random.normal(loc=self.river_dist_noise_loc, scale=self.river_dist_noise_sca, size=1),
                                self.river_min, 3*self.river_dist_loc)
                    depth[i] = np.clip(depth[i-1] + np.random.normal(loc=0.0, scale=10.0), 20, 100)

            # fill close points at that distance away from the path
            lat_path = self.DesiredPath["lat"]
            lon_path = self.DesiredPath["lon"]

            for i, (lat_p, lon_p) in enumerate(zip(lat_path, lon_path)):
                
                if i != self.n_wps-1:

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
            #self.DepthData["data"] = np.clip(self.DepthData["data"], 1.0, np.infty)

            if self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon) >= self.OS.critical_depth:
                break

        # log
        self.log_Depth = np.log(self.DepthData["data"])

        # for contour plot
        self.con_ticks = np.log([1.0, 2.0, 5.0, 15.0, 50.0, 150.0, 500.0])
        self.con_ticklabels = [int(np.round(tick, 0)) for tick in np.exp(self.con_ticks)]
        self.con_ticklabels[0] = 0
        self.clev = np.arange(0, self.log_Depth.max(), .1)


    def _sample_wind_data(self):
        """Generates random wind data by overwriting the real data information."""
        # clear real data
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

                speed_mps[lat_idx, lon_idx] = max([0.0, V_const[lat_area, lon_area] + np.random.normal(0.0, 1.0)])
                angle[lat_idx, lon_idx] = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.WindData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[5, 5], mode="constant"), 0.0, np.infty)
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
        """Generates random current data by overwriting the real data information."""
        # clear real data
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

                speed_mps[lat_idx, lon_idx] = np.clip(V_const[lat_area, lon_area] + np.random.normal(0.0, 0.25), 0.0, 0.5)
                angle[lat_idx, lon_idx] = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.CurrentData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[1, 1], mode="constant"), 0.0, np.infty)
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
        """Generates random wave data by overwriting the real data information."""
        # clear real data
        self.WaveData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1], num=10)
        self.WaveData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=10)

        speed_mps = np.zeros((len(self.CurrentData["lat"]), len(self.CurrentData["lon"])))
        angle = np.zeros_like(speed_mps)

        height = np.zeros((len(self.WaveData["lat"]), len(self.WaveData["lon"])))
        length = np.zeros_like(height)
        period = np.zeros_like(height)
        angle = np.zeros_like(height)

        # size of homogenous current areas
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

                height[lat_idx, lon_idx] = np.clip(height_const[lat_area, lon_area] + np.random.normal(0.0, 0.05), 0.0, 0.5)
                length[lat_idx, lon_idx] = np.clip(length_const[lat_area, lon_area] + np.random.normal(0.0, 5.0), 0.0, 100.0)
                period[lat_idx, lon_idx] = np.clip(period_const[lat_area, lon_area] + np.random.normal(0.0, 0.5), 0.0, 10.0)
                angle[lat_idx, lon_idx]  = angle_to_2pi(angle_const[lat_area, lon_area] + dtr(np.random.normal(0.0, 5.0)))

        # smoothing things
        self.WaveData["height"] = np.clip(scipy.ndimage.gaussian_filter(height, sigma=[0.01, 0.01], mode="constant"), 0.0001, np.infty)
        self.WaveData["length"] = np.clip(scipy.ndimage.gaussian_filter(length, sigma=[1, 1], mode="constant"), 0.0001, np.infty)
        self.WaveData["period"] = np.clip(scipy.ndimage.gaussian_filter(period, sigma=[0.1, 0.1], mode="constant"), 0.0001, np.infty)
        self.WaveData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")

        # overwrite other entries
        e = np.zeros_like(height)
        n = np.zeros_like(height)

        for lat_idx, _ in enumerate(self.WaveData["lat"]):
            for lon_idx, _ in enumerate(self.WaveData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=height[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.WaveData["eastward"] = e
        self.WaveData["northward"] = n


    def _add_rev_path(self):
        """Sets the reversed version of the path."""
        self.RevPath = {"n_wps" : self.DesiredPath["n_wps"]}
        self.RevPath["lat"] = np.flip(self.DesiredPath["lat"])
        self.RevPath["lon"] = np.flip(self.DesiredPath["lon"])
        self.RevPath["north"] = np.flip(self.DesiredPath["north"])
        self.RevPath["east"]  = np.flip(self.DesiredPath["east"])


    def _depth_at_latlon(self, lat_q, lon_q):
        """Computes the water depth at a (queried) longitude-latitude position based on linear interpolation."""
        return Z_at_latlon(Z=self.DepthData["data"], lat_array=self.DepthData["lat"], lon_array=self.DepthData["lon"],
                           lat_q=lat_q, lon_q=lon_q)


    def _current_at_latlon(self, lat_q, lon_q):
        """Computes the current speed and angle at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (speed, angle)"""
        speed = Z_at_latlon(Z=self.CurrentData["speed_mps"], lat_array=self.CurrentData["lat"], lon_array=self.CurrentData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
        angle = angle_to_2pi(Z_at_latlon(Z=self.CurrentData["angle"], lat_array=self.CurrentData["lat"], 
                                         lon_array=self.CurrentData["lon"], lat_q=lat_q, lon_q=lon_q))
        return speed, angle


    def _wind_at_latlon(self, lat_q, lon_q):
        """Computes the wind speed and angle at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (speed, angle)"""
        speed = Z_at_latlon(Z=self.WindData["speed_mps"], lat_array=self.WindData["lat"], lon_array=self.WindData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
        angle = angle_to_2pi(Z_at_latlon(Z=self.WindData["angle"], lat_array=self.WindData["lat"], 
                                         lon_array=self.WindData["lon"], lat_q=lat_q, lon_q=lon_q))
        return speed, angle


    def _wave_at_latlon(self, lat_q, lon_q):
        """Computes the wave angle, amplitude, period, and length at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (angle, amplitude, period, length)"""
        angle = Z_at_latlon(Z=self.WaveData["angle"], lat_array=self.WaveData["lat"], lon_array=self.WaveData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
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


    def _sense_LiDAR(self):
        """Generates an observation via LiDAR sensoring. There are 'lidar_n_beams' equally spaced beams originating from the midship of the OS.
        The first beam is defined in direction of the heading of the OS. Each beam consists of 'n_dots_per_beam' sub-points, which are sequentially considered. 
        Returns for each beam the distance at which insufficient water depth has been detected, where the maximum range is 'lidar_range'.
        Furthermore, it returns the endpoints in lat-lon of each (truncated) beam.
        Returns (as tuple):
            dists as a np.array(lidar_n_beams,)
            endpoints in lat-lon as list of lat-lon-tuples
        """
        # UTM coordinates of OS
        N0, E0, head0 = self.OS.eta

        # setup output
        out_dists = np.ones(self.lidar_n_beams) * self.lidar_range
        out_lat_lon = []
        
        for out_idx, angle in enumerate(self.lidar_beam_angles):

            # current angle under consideration of the heading
            angle = angle_to_2pi(angle + head0)
            
            for dist in self.d_dots_per_beam:

                # compute N-E coordinates of dot
                delta_E_dot, delta_N_dot = xy_from_polar(r=dist, angle=angle)
                N_dot = N0 + delta_N_dot
                E_dot = E0 + delta_E_dot

                # transform to LatLon
                lat_dot, lon_dot = to_latlon(north=N_dot, east=E_dot, number=self.OS.utm_number)

                # check water depth at that point
                depth_dot = self._depth_at_latlon(lat_q=lat_dot, lon_q=lon_dot)

                if depth_dot <= self.OS.critical_depth:
                    out_dists[out_idx] = dist
                    out_lat_lon.append((lat_dot, lon_dot))
                    break
                if dist == self.lidar_range:
                    out_lat_lon.append((lat_dot, lon_dot))

        return out_dists, out_lat_lon


    def reset(self):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # sample path in training
        if self.mode == "train":
            self._sample_desired_path()
            self._add_rev_path()

        # init OS
        wp_idx = 20 #np.random.uniform(low=int(self.n_wps*0.25), high=int(self.n_wps*0.75), size=(1,)).astype(int)[0]
        lat_init = self.DesiredPath["lat"][wp_idx]# if self.mode == "train" else 56.635
        lon_init = self.DesiredPath["lon"][wp_idx]# if self.mode == "train" else 7.421
        N_init, E_init, number = to_utm(lat=lat_init, lon=lon_init)

        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = None,
                         u_init    = 0.0,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = 3.0,
                         full_ship = False,
                         cont_acts = True)
        # init waypoints of OS
        self._init_OS_wps()

        # init cross-track error
        self._set_cte()

        # set heading with noise
        self.OS.eta[2] = angle_to_2pi(self.pi_path + dtr(np.random.uniform(-45.0, 45.0)))

        # Critical point: We do not update the UTM number (!) since our simulation primarily takes place in 32U and 32V.
        self.OS.utm_number = number

        # generate random environmental data in training
        if self.mode == "train":
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
        self._set_ce()

        # initially compute ship domain for plotting
        rads  = np.linspace(0.0, 2*math.pi, 100)
        dists = [get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
            OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        # init other vessels
        if self.N_TSs_random:
            self.N_TSs = np.random.choice(self.N_TSs_max)
        else:
            self.N_TSs = self.N_TSs_max

        # no list comprehension since we need access to previously spawned TS
        self.TSs = []
        for _ in range(self.N_TSs):
            self.TSs.append(self._get_TS())

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state


    def _init_OS_wps(self):
        """Initializes the two waypoints based on the initial position of the agent."""
        self.OS.rev_dir = False
        self.OS.wp1_idx, self.OS.wp1_N, self.OS.wp1_E, self.OS.wp2_idx, self.OS.wp2_N, self.OS.wp2_E = get_init_two_wp(lat_array=self.DesiredPath["lat"], \
            lon_array=self.DesiredPath["lon"], a_n=self.OS.eta[0], a_e=self.OS.eta[1])
        try:
            self.OS.wp3_idx = self.OS.wp2_idx + 1
            self.OS.wp3_N, self.OS.wp3_E, _ = to_utm(self.DesiredPath["lat"][self.OS.wp3_idx], self.DesiredPath["lon"][self.OS.wp3_idx])
        except:
            raise ValueError("The agent should spawn at least two waypoints away from the goal.")


    def _wp_dist(self, wp1_idx, wp2_idx, use_rev_path=False):
        """Computes the euclidean distance between two waypoints of the desired path."""
        if wp1_idx not in range(self.n_wps) or wp2_idx not in range(self.n_wps):
            raise ValueError("Your path index is out of order. Please check your sampling strategy.")

        if use_rev_path:
            return ED(N0=self.RevPath["north"][wp1_idx], E0=self.RevPath["east"][wp1_idx], \
                N1=self.RevPath["north"][wp2_idx], E1=self.RevPath["east"][wp2_idx])
        else:
            return ED(N0=self.DesiredPath["north"][wp1_idx], E0=self.DesiredPath["east"][wp1_idx], \
                N1=self.DesiredPath["north"][wp2_idx], E1=self.DesiredPath["east"][wp2_idx])


    def _get_rev_path_wps(self, wp1_idx, wp2_idx):
        """Computes the waypoints from the reversed version of the desired path based on indices from the real path.
        Returns: wp1_rev_idx, wp1_rev_N, wp1_rev_E, wp2_rev_idx, wp2_rev_N, wp2_rev_E."""

        wp1_rev = self.n_wps - (wp2_idx+1)
        wp2_rev = self.n_wps - (wp1_idx+1)

        return wp1_rev, wp2_rev


    def _get_TS(self):
        """Places a target ship by sampling a 
            1) traveling direction,
            2) distance on the path.
        All ships spawn in front of the agent.
        Returns: 
            KVLCC2."""
        # sample distance
        d = np.random.uniform(*self.TS_spawn_dists)

        # sample direction
        rev_dir = bool(random.getrandbits(1))

        # get wps
        wp1 = self.OS.wp1_idx
        wp1_N = self.OS.wp1_N
        wp1_E = self.OS.wp1_E

        wp2 = self.OS.wp2_idx
        wp2_N = self.OS.wp2_N
        wp2_E = self.OS.wp2_E

        # determine starting position
        ate_init = ate(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])
        d_to_nxt_wp = self._wp_dist(wp1, wp2) - ate_init
        orig_seg = True

        while True:
            if d > d_to_nxt_wp:
                d -= d_to_nxt_wp
                wp1 += 1
                wp2 += 1
                d_to_nxt_wp = self._wp_dist(wp1, wp2)
                orig_seg = False
            else:
                break

        # path angle
        pi_path_spwn = bng_abs(N0=self.DesiredPath["north"][wp1], E0=self.DesiredPath["east"][wp1], \
            N1=self.DesiredPath["north"][wp2], E1=self.DesiredPath["east"][wp2])

        # still in original segment
        if orig_seg:
            E_add, N_add = xy_from_polar(r=ate_init+d, angle=pi_path_spwn)
        else:
            E_add, N_add = xy_from_polar(r=d, angle=pi_path_spwn)

        # include random disturbances
        N_TS = self.DesiredPath["north"][wp1] + N_add + np.random.normal(0.0, 50)
        E_TS = self.DesiredPath["east"][wp1] + E_add + np.random.normal(0.0, 50)

        TS = KVLCC2(N_init    = N_TS,
                    E_init    = E_TS,
                    psi_init  = angle_to_2pi(pi_path_spwn + math.pi) if rev_dir else pi_path_spwn,
                    u_init    = 0.0,
                    v_init    = 0.0,
                    r_init    = 0.0,
                    delta_t   = self.delta_t,
                    N_max     = np.infty,
                    E_max     = np.infty,
                    nps       = np.random.uniform(0.3, 0.8) * self.OS.nps,
                    full_ship = False)
        TS.utm_number = 32

        # store waypoint information
        TS.rev_dir = rev_dir

        if rev_dir:
            TS.wp1_idx, TS.wp2_idx = self._get_rev_path_wps(wp1, wp2)
            TS.wp3_idx = TS.wp2_idx + 1
            path = self.RevPath
        else:
            TS.wp1_idx, TS.wp2_idx, TS.wp3_idx = wp1, wp2, wp2 + 1
            path = self.DesiredPath
    
        TS.wp1_N, TS.wp1_E = path["north"][TS.wp1_idx], path["east"][TS.wp1_idx]
        TS.wp2_N, TS.wp2_E = path["north"][TS.wp2_idx], path["east"][TS.wp2_idx]
        TS.wp3_N, TS.wp3_E = path["north"][TS.wp3_idx], path["east"][TS.wp3_idx]

        # predict converged speed of sampled TS
        TS.nu[0] = TS._get_u_from_nps(TS.nps, psi=TS.eta[2])
        return TS


    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([7.0, 0.7, 0.004])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

        # ------------------------- path information ---------------------------
        state_path = np.array([self.ye/self.OS.Lpp, self.course_error/math.pi])

        # -------------------- environmental disturbances ----------------------
        if any([pd.isnull(ele) or np.isinf(ele) or ele is None for ele in\
             [self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave]]):
            beta_wave = 0.0
            eta_wave = 0.0
            T_0_wave = 0.0
            lambda_wave = 0.0
        else:
            beta_wave = self.beta_wave
            eta_wave = self.eta_wave
            T_0_wave = self.T_0_wave
            lambda_wave = self.lambda_wave

        state_env = np.array([self.V_c/0.5,  self.beta_c/(2*math.pi),      # currents
                              self.V_w/15.0, self.beta_w/(2*math.pi),      # winds
                              beta_wave/(2*math.pi), eta_wave/0.5, T_0_wave/7.0, lambda_wave/60.0,    # waves
                              self.H/100.0])    # depth

        # ----------------------- LiDAR for depth -----------------------------
        state_LiDAR = self._get_closeness_from_lidar(self._sense_LiDAR()[0])

        # ----------------------- TS information ------------------------------
        N0, E0, head0 = self.OS.eta
        state_TSs = []

        for TS in self.TSs:
            N, E, headTS = TS.eta

            # euclidean distance
            D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
                 OS=self.OS, TS=TS)
            ED_OS_TS = (ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True) - D) / (20*self.OS.Lpp)
            ED_OS_TS_norm = np.clip(1-ED_OS_TS, 0.0, 1.0)

            # relative bearing
            bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0, to_2pi=False) / (math.pi)

            # heading intersection angle
            C_TS = head_inter(head_OS=head0, head_TS=headTS, to_2pi=False) / (math.pi)

            # speed
            V_TS = TS._get_V() / 7.0

            # collision risk
            CR = self._get_CR(OS=self.OS, TS=TS)

            # store it
            state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, CR])

        # need always same state size, thus we pad ghost ships
        while len(state_TSs) != self.N_TSs_max:
            # ED
            ED_ghost = 0.0

            # relative bearing
            bng_rel_ghost = -1.0

            # heading intersection angle
            C_ghost = -1.0

            # speed
            V_ghost = 0.0

            # collision risk
            CR_ghost = 0.0

            state_TSs.append([ED_ghost, bng_rel_ghost, C_ghost, V_ghost, CR_ghost])

        # sort according to collision risk (ascending, larger CR is more dangerous)
        state_TSs = np.array(sorted(state_TSs, key=lambda x: x[-1])).flatten()

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_env, state_LiDAR, state_TSs])


    def _set_cte(self, smooth_dc=False):
        """Sets the cross-track error based on VFG."""
        if smooth_dc:
            N3, E3 = self.OS.wp3_N, self.OS.wp3_E
        else:
            N3, E3 = None, None

        self.ye, self.desired_course, self.pi_path = VFG(N1 = self.OS.wp1_N, 
                                                         E1 = self.OS.wp1_E, 
                                                         N2 = self.OS.wp2_N, 
                                                         E2 = self.OS.wp2_E,
                                                         NA = self.OS.eta[0], 
                                                         EA = self.OS.eta[1], 
                                                         K  = self.VFG_K, 
                                                         N3 = N3,
                                                         E3 = E3)
    def _set_ce(self):
        """Sets the course error, which is desired course minus course."""
        self.course_error = angle_to_pi(self.desired_course - self.OS._get_course())


    def _update_wps(self, vessel):
        """Updates the waypoints for following the (potentially reversed) path."""
        # check whether we need to switch wps
        switch = switch_wp(wp1_N = vessel.wp1_N, 
                           wp1_E = vessel.wp1_E, 
                           wp2_N = vessel.wp2_N, 
                           wp2_E = vessel.wp2_E, 
                           a_N   = vessel.eta[0], 
                           a_E   = vessel.eta[1])
        path = self.RevPath if vessel.rev_dir else self.DesiredPath

        if switch and (vessel.wp3_idx != (path["n_wps"]-1)):
            # update waypoint 1
            vessel.wp1_idx += 1
            vessel.wp1_N = vessel.wp2_N
            vessel.wp1_E = vessel.wp2_E

            # update waypoint 2
            vessel.wp2_idx += 1
            vessel.wp2_N = vessel.wp3_N
            vessel.wp2_E = vessel.wp3_E

            # update waypoint 3
            vessel.wp3_idx += 1
            vessel.wp3_N = path["north"][vessel.wp3_idx]
            vessel.wp3_E = path["east"][vessel.wp3_idx]
        return vessel


    def _heading_control(self, vessel):
        """Controls the heading of a to smoothly follow the (potentially reversed) path."""
        # angles of the two segments
        pi_path_12 = bng_abs(N0=vessel.wp1_N, E0=vessel.wp1_E, N1=vessel.wp2_N, E1=vessel.wp2_E)
        pi_path_23 = bng_abs(N0=vessel.wp2_N, E0=vessel.wp2_E, N1=vessel.wp3_N, E1=vessel.wp3_E)

        ate_TS = ate(N1=vessel.wp1_N, E1=vessel.wp1_E, N2=vessel.wp2_N, E2=vessel.wp2_E, NA=vessel.eta[0], EA=vessel.eta[1], pi_path=pi_path_12)
        dist_12 = self._wp_dist(vessel.wp1_idx, vessel.wp2_idx, vessel.rev_dir)

        # weighting
        frac = np.clip(ate_TS/dist_12, 0.0, 1.0)
        w23 = frac**15

        # adjustment to avoid boundary issues at 2pi
        if abs(pi_path_12-pi_path_23) >= math.pi:
            if pi_path_12 >= pi_path_23:
                pi_path_12 = angle_to_pi(pi_path_12)
            else:
                pi_path_23 = angle_to_pi(pi_path_23)

        # heading construction
        vessel.eta[2] = angle_to_2pi(w23*pi_path_23 + (1-w23)*pi_path_12)
        return vessel


    def _handle_respawn(self, TS):
        """Checks whether the respawning condition of the target ship is fulfilled."""
        N0, E0, _ = self.OS.eta
        N1, E1, _ = TS.eta
        TCPA = tcpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=self.OS._get_course(), chiTS=TS._get_course(),\
             VOS=self.OS._get_V(), VTS=TS._get_V())

        if TCPA < 0.0 and abs(TCPA) > self.TCPA_respawn and ED(N0=N0, E0=E0, N1=N1, E1=E1) >= 7*self.OS.Lpp:
            return self._get_TS()
        else:
            return TS


    def _update_disturbances(self, OS_lat=None, OS_lon=None):
        """Updates the environmental disturbances at the agent's current position."""
        if OS_lat is None and OS_lon is None:
            OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=self.OS.utm_number)

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
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        a = float(a)
        self.OS._control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # environmental effects
        self._update_disturbances()

        # update waypoints of path
        self.OS = self._update_wps(self.OS)

        # compute new cross-track error and course error
        self._set_cte()
        self._set_ce()

        # HEADING CONTROL OS
        #self.OS = self._heading_control(self.OS)

        # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
        [TS._upd_dynamics() for TS in self.TSs]

        # check respawn
        self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

        # update waypoints for other vessels
        self.TSs = [self._update_wps(TS) for TS in self.TSs]

        # simple heading control of target ships
        self.TSs = [self._heading_control(TS) for TS in self.TSs]

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}


    def _get_CR(self, OS, TS):
        """Computes the collision risk metric similar to Chun et al. (2021)."""
        N0, E0, head0 = OS.eta
        N1, E1, _ = TS.eta
        D = get_ship_domain(A=OS.ship_domain_A, B=OS.ship_domain_B, C=OS.ship_domain_C, D=OS.ship_domain_D, OS=OS, TS=TS)

        # check if already in ship domain
        ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
        if ED_OS_TS <= D:
            return 1.0

        # compute speeds and courses
        VOS = OS._get_V()
        VTS = TS._get_V()
        chiOS = OS._get_course()
        chiTS = TS._get_course()

        # compute CPA measures
        DCPA, TCPA, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, \
            chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS, get_positions=True)

        # substract ship domain at TCPA = 0 from DCPA
        bng_rel_tcpa_from_OS_pers = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
        domain_tcpa = get_ship_domain(A=OS.ship_domain_A, B=OS.ship_domain_B, C=OS.ship_domain_C, D=OS.ship_domain_D, OS=None, TS=None,\
            ang=bng_rel_tcpa_from_OS_pers)
        DCPA = max([0.0, DCPA-domain_tcpa])

        if TCPA >= 0.0:
            cr_cpa = math.exp((DCPA + 0.75 * TCPA) * math.log(self.CR_al) / self.CR_rec_dist)
        else:
            cr_cpa = math.exp((DCPA + 20.0 * abs(TCPA)) * math.log(self.CR_al) / self.CR_rec_dist)

        # CR based on euclidean distance to ship domain
        cr_ed = math.exp(-(ED_OS_TS-D)/200)

        return min([1.0, max([cr_cpa, cr_ed])])


    def _calculate_reward(self, a):

        # ----------------------- Path-following reward --------------------
        # cross-track error
        k_ye = 0.05
        self.r_ye = math.exp(-k_ye * abs(self.ye))

        # course error
        k_ce = 5.0
        self.r_ce = math.exp(-k_ce * abs(self.course_error))

        # ---------------------- Collision Avoidance reward -----------------
        self.r_coll = 0

        for TS in self.TSs:
            CR = self._get_CR(OS=self.OS, TS=TS)
            if CR == 1.0:
                self.r_coll -= 10.0
            else:
                self.r_coll -= CR

        if self.H <= self.OS.critical_depth:
            self.r_coll -= 10.0

        # -------------------------- Comfort reward -------------------------
        self.r_comf = -a**4

        # ---------------------------- Aggregation --------------------------
        weights = np.array([0.5, 0.5, 0.5, 0.02])
        rews = np.array([self.r_ye, self.r_ce, self.r_coll, self.r_comf])
        self.r = np.sum(weights * rews) / np.sum(weights)


    def _done(self):
        """Returns boolean flag whether episode is over."""
        # OS is too far away from path
        if self.ye > 400:
            return True

        # OS hit land
        elif self.H <= self.OS.critical_depth:
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True

        else:
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

        path_info = r"$y_e$" + f" [m]: {self.ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.desired_course):.2f}, " \
            + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.course_error):.2f}"
        return ste + ", " + pos + "\n" + vel + ", " + depth + "\n" + wind + "\n" + current + "\n" + wave + "\n" + path_info


    def _render_ship(self, ax, vessel, color, plot_CR=False):
        """Draws the ship on the axis. Vessel should by of type KVLCC2. Returns the ax."""
        # quick access
        l = vessel.Lpp/2
        b = vessel.B/2
        N, E, head = vessel.eta
        N0, E0, head0 = self.OS.eta

        # get rectangle/polygon end points in UTM
        A = (E - b, N + l)
        B = (E + b, N + l)
        C = (E - b, N - l)
        D = (E + b, N - l)

        # rotate them according to heading
        A = rotate_point(x=A[0], y=A[1], cx=E, cy=N, angle=-head)
        B = rotate_point(x=B[0], y=B[1], cx=E, cy=N, angle=-head)
        C = rotate_point(x=C[0], y=C[1], cx=E, cy=N, angle=-head)
        D = rotate_point(x=D[0], y=D[1], cx=E, cy=N, angle=-head)

        # collision risk and metrics
        if plot_CR:
            CR = self._get_CR(OS=self.OS, TS=vessel)

            DCPA, TCPA, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=self.OS._get_course(),\
                chiTS=vessel._get_course(), VOS=self.OS._get_V(), VTS=vessel._get_V(), get_positions=True)

            # substract ship domain at TCPA = 0 from DCPA
            bng_rel_tcpa_from_OS_pers = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
            domain_tcpa = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C,\
                D=self.OS.ship_domain_D, OS=None, TS=None, ang=bng_rel_tcpa_from_OS_pers)
            DCPA = max([0.0, DCPA-domain_tcpa])

        if self.plot_in_latlon:

            # convert them to lat/lon
            A_lat, A_lon = to_latlon(north=A[1], east=A[0], number=vessel.utm_number)
            B_lat, B_lon = to_latlon(north=B[1], east=B[0], number=vessel.utm_number)
            C_lat, C_lon = to_latlon(north=C[1], east=C[0], number=vessel.utm_number)
            D_lat, D_lon = to_latlon(north=D[1], east=D[0], number=vessel.utm_number)

            # draw the polygon (A is included twice to create a closed shape)
            lons = [A_lon, B_lon, D_lon, C_lon, A_lon]
            lats = [A_lat, B_lat, D_lat, C_lat, A_lat]
            ax.plot(lons, lats, color=color, linewidth=2.0)

            if plot_CR:
                CR_x = min(lons) - np.abs(min(lons) - max(lons))
                CR_y = min(lats)
                ax.text(CR_x, CR_y, f"CR: {CR:.2f}" + "\n" + f"DCPA: {DCPA:.2f}" + "\n" + f"TCPA: {TCPA:.2f}",\
                     fontsize=7, horizontalalignment='center', verticalalignment='center', color=color)
        else:
            xs = [A[0], B[0], D[0], C[0], A[0]]
            ys = [A[1], B[1], D[1], C[1], A[1]]
            ax.plot(xs, ys, color=color, linewidth=2.0)

            if plot_CR:
                CR_x = min(xs) - np.abs(min(xs) - max(xs))
                CR_y = min(ys)
                ax.text(CR_x, CR_y, f"CR: {CR:.2f}" + "\n" + f"DCPA: {DCPA:.2f}" + "\n" + f"TCPA: {TCPA:.2f}",\
                     fontsize=7, horizontalalignment='center', verticalalignment='center', color=color)
        return ax


    def _render_wps(self, ax, vessel, color):
        """Renders the current waypoints of a vessel."""
        wp1_lat, wp1_lon = to_latlon(north=vessel.wp1_N, east=vessel.wp1_E, number=vessel.utm_number)
        wp2_lat, wp2_lon = to_latlon(north=vessel.wp2_N, east=vessel.wp2_E, number=vessel.utm_number)
        ax.plot([wp1_lon, wp2_lon], [wp1_lat, wp2_lat], color=color, linewidth=1.0, markersize=3)
        return ax


    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            if self.plot_reward:
                self.f, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 8))
            else:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(10, 10))

            plt.ion()
            plt.show()

        if self.step_cnt % 1 == 0:
            # ------------------------------ ship movement --------------------------------
            # get position of OS in lat/lon
            N0, E0, head0 = self.OS.eta
            OS_lat, OS_lon = to_latlon(north=N0, east=E0, number=self.OS.utm_number)

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
                    if self.step_cnt == 0:
                        cbar = self.f.colorbar(con, ticks=self.con_ticks, ax=ax)
                        cbar.ax.set_yticklabels(self.con_ticklabels)

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
                ax = self._render_ship(ax=ax, vessel=self.OS, color="red", plot_CR=False)

                # ship domain
                xys = [rotate_point(E0 + x, N0 + y, cx=E0, cy=N0, angle=-head0) for x, y in zip(self.domain_xs, self.domain_ys)]

                if self.plot_in_latlon:
                    lat_lon_tups = [to_latlon(north=y, east=x, number=self.OS.utm_number)[:2] for x, y in xys]
                    lats = [e[0] for e in lat_lon_tups]
                    lons = [e[1] for e in lat_lon_tups]
                    ax.plot(lons, lats, color="red", alpha=0.7)
                else:
                    xs = [xy[0] for xy in xys]
                    ys = [xy[1] for xy in xys]
                    ax.plot(xs, ys, color="red", alpha=0.7)

                # TSs
                for i, TS in enumerate(self.TSs):
                    col = self.default_cols[i] if i <= (len(self.default_cols)-1) else "black"
                    ax = self._render_ship(ax=ax, vessel=TS, color=col, plot_CR=True)

                #--------------------- Desired path ------------------------
                if self.plot_path:

                    if self.plot_in_latlon:
                        ax.plot(self.DesiredPath["lon"], self.DesiredPath["lat"], marker='o', color="salmon", linewidth=1.0, markersize=3)

                        # wps of OS
                        self._render_wps(ax=ax, vessel=self.OS, color="black")

                        # wps of TSs
                        for i, TS in enumerate(self.TSs):
                            col = self.default_cols[i] if i <= (len(self.default_cols)-1) else "black"
                            ax = self._render_wps(ax=ax, vessel=TS, color=col)

                        # cross-track error
                        if self.ye < 0:
                            dE, dN = xy_from_polar(r=abs(self.ye), angle=angle_to_2pi(self.pi_path + dtr(90.0)))
                        else:
                            dE, dN = xy_from_polar(r=self.ye, angle=angle_to_2pi(self.pi_path - dtr(90.0)))
                        yte_lat, yte_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                        ax.plot([OS_lon, yte_lon], [OS_lat, yte_lat], color="salmon")

                    else:
                        ax.plot(self.DesiredPath["east"], self.DesiredPath["north"], marker='o', color="salmon", linewidth=1.0, markersize=3)

                        # current waypoints
                        ax.plot([self.OS.wp1_E, self.OS.wp2_E], [self.OS.wp1_N, self.OS.wp2_N], color="springgreen", linewidth=1.0, markersize=3)

                        # cross-track error
                        if self.ye < 0:
                            dE, dN = xy_from_polar(r=abs(self.ye), angle=angle_to_2pi(self.pi_path + dtr(90.0)))
                        else:
                            dE, dN = xy_from_polar(r=self.ye, angle=angle_to_2pi(self.pi_path - dtr(90.0)))
                        ax.plot([E0, E0+dE], [N0, N0+dN], color="salmon")

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
                            headwidth=2.0, color="whitesmoke", scale=10)

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
                            headwidth=2.0, color="salmon", scale=10)

                #--------------------- LiDAR sensing ------------------------
                if self.plot_lidar and self.plot_in_latlon:
                    _, lidar_lat_lon = self._sense_LiDAR()

                    for _, latlon in enumerate(lidar_lat_lon):
                        ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.4)#, alpha=(idx+1)/len(lidar_lat_lon))

            # ------------------------------ reward plot --------------------------------
            if self.plot_reward:
                if self.step_cnt == 0:
                    self.ax2.clear()
                    self.ax2.old_time = 0
                    self.ax2.old_r_ye = 0
                    self.ax2.old_r_ce = 0
                    self.ax2.old_r_coll = 0
                    self.ax2.old_r_comf = 0
                    self.ax2.r = 0

                self.ax2.set_xlim(0, self._max_episode_steps)
                #self.ax2.set_ylim(0, 1)
                self.ax2.set_xlabel("Timestep in episode")
                self.ax2.set_ylabel("Reward")

                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ye, self.r_ye], color = "black", label="Cross-track error")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ce, self.r_ce], color = "red", label="Course error")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_coll, self.r_coll], color = "green", label="Collision")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_comf, self.r_comf], color = "blue", label="Comfort")
                self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.r, self.r], color = "orange", label="Aggregated")
                
                if self.step_cnt == 0:
                    self.ax2.legend()

                self.ax2.old_time = self.step_cnt
                self.ax2.old_r_ye = self.r_ye
                self.ax2.old_r_ce = self.r_ce
                self.ax2.old_r_coll = self.r_coll
                self.ax2.old_r_comf = self.r_comf
                self.ax2.r = self.r

            #plt.gca().set_aspect('equal')
            plt.pause(0.001)
