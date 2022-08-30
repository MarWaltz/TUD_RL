import copy
import math
import pickle

import gym
import numpy as np
import scipy.ndimage
from gym import spaces
from matplotlib import cm
from matplotlib import pyplot as plt
from tud_rl.envs._envs.HHOS_Fnc import (VFG, Z_at_latlon, fill_array,
                                        find_nearest, find_nearest_two,
                                        get_init_two_wp, mps_to_knots,
                                        switch_wp, to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, dtr, rtd,
                                         xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 from Hamburg to Oslo."""

    def __init__(self, mode="train"):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0   # simulation time interval (in s)

        # LiDAR
        self.lidar_range       = NM_to_meter(1.0)                                                # range of LiDAR sensoring in m
        self.lidar_n_beams     = 0 # 25                                                              # number of beams
        self.lidar_beam_angles = np.linspace(0.0, 2*math.pi, self.lidar_n_beams, endpoint=False) # beam angles
        self.n_dots_per_beam   = 10                                                              # number of subpoints per beam
        self.d_dots_per_beam   = np.linspace(start=0.0, stop=self.lidar_range, num=self.lidar_n_beams+1, endpoint=True)[1:] # distances from midship of subpoints per beam
        self.sense_depth       = 15  # water depth at which LiDAR recognizes an obstacle

        # vector field guidance
        self.VFG_K = 0.001

        # data range
        self.lon_lims = [4.83, 14.33]
        self.lat_lims = [51.83, 60.5]
        self.lon_range = self.lon_lims[1] - self.lon_lims[0]
        self.lat_range = self.lat_lims[1] - self.lat_lims[0]

        # data loading
        self._load_desired_path(path_to_desired_path="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS")
        self._load_depth_data(path_to_depth_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/DepthData")
        self._load_wind_data(path_to_wind_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/winds")
        self._load_current_data(path_to_current_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/currents")
        self._load_wave_data(path_to_wave_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/waves")

        # setting
        assert mode in ["train", "validate"], "Unknown HHOS mode. Can either train or validate."
        self.mode = mode

        # In training, we are not using real data. Thus, we only stick to the format and overwrite them by sampled values.
        if mode == "train":
            self.n_wps = 500
            self._sample_desired_path()
            self._sample_depth_data()
            self._sample_wind_data()
            self._sample_current_data()
            self._sample_wave_data()

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.05
        self.half_num_depth_idx   = math.ceil((self.show_lon_lat / 2.0) / self.DepthData["metaData"]["cellsize"]) + 1
        self.half_num_wind_idx    = math.ceil((self.show_lon_lat / 2.0) / self.WindData["metaData"]["cellsize"]) + 1
        self.half_num_current_idx = math.ceil((self.show_lon_lat / 2.0) / np.mean(np.diff(self.CurrentData["lat"]))) + 1
        self.half_num_wave_idx    = math.ceil((self.show_lon_lat / 2.0) / self.WaveData["metaData"]["cellsize"]) + 1

        # visualization
        self.plot_in_latlon = True         # if false, plots in UTM coordinates
        self.plot_depth = True
        self.plot_path = True
        self.plot_wind = False
        self.plot_current = False
        self.plot_waves = True
        self.plot_lidar = False
        self.plot_reward = True

        if not self.plot_in_latlon:
            self.show_lon_lat = np.clip(self.show_lon_lat, 0.005, 5.95)
            self.UTM_viz_range_E = abs(to_utm(lat=52.0, lon=6.0001)[1] - to_utm(lat=52.0, lon=6.0001+self.show_lon_lat/2)[1])
            self.UTM_viz_range_N = abs(to_utm(lat=50.0, lon=8.0)[0] - to_utm(lat=50.0+self.show_lon_lat/2, lon=8.0)[0])

        # gym inherits
        obs_size = 16 + self.lidar_n_beams
        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        self.r = 0
        self.r_ye = 0
        self.r_ce = 0
        self._max_episode_steps = 1_000


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


    def _sample_desired_path(self, l=0.01):
        """Constructs a path with n_wps way points, each being of length l apart from its neighbor in the lat-lon-system."""
        self.DesiredPath = {"n_wps" : self.n_wps}

        # do it until we have a path whichs stays in our simulation domain
        while True:

            # sample starting point
            lat = np.zeros(self.n_wps)
            lon = np.zeros(self.n_wps)

            #lat[0] = np.random.uniform(low  = self.lat_lims[0] + 0.25 * self.lat_range, 
            #                           high = self.lat_lims[1] - 0.25 * self.lat_range)
            #lon[0] = np.random.uniform(low  = self.lon_lims[0] + 0.25 * self.lon_range, 
            #                           high = self.lon_lims[1] - 0.25 * self.lon_range)
            lat[0] = 60.0
            lon[0] = 9.0

            # sample other points
            ang = np.random.uniform(math.pi/2, 3/2*math.pi)
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


    def _sample_depth_data(self):
        """Generates random depth data by overwriting the real data information."""
        # clear real data
        lat_depth = self.DepthData["lat"]
        lon_depth = self.DepthData["lon"]
        self.DepthData["data"] *= 0.0
        self.DepthData["data"] += 1.0

        # sample distances to waypoints
        d_left  = np.zeros(self.n_wps)
        d_right = np.zeros(self.n_wps)

        for i in range(self.n_wps):
            if i == 0:
                d_left[i]  = np.clip(np.random.normal(loc=10_000, scale=2_000, size=1), 500, np.infty)
                d_right[i] = np.clip(np.random.normal(loc=10_000, scale=2_000, size=1), 500, np.infty)
            else:
                d_left[i]  = np.clip(d_left[i-1] + np.random.normal(loc=100, scale=50, size=1), 500, 30_000)
                d_right[i] = np.clip(d_right[i-1] + np.random.normal(loc=100, scale=50, size=1), 500, 30_000)

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

                # find closest point in the array and set depth
                _, lat_left_idx = find_nearest(array=lat_depth, value=lat_left)
                _, lon_left_idx = find_nearest(array=lon_depth, value=lon_left)

                _, lat_right_idx = find_nearest(array=lat_depth, value=lat_right)
                _, lon_right_idx = find_nearest(array=lon_depth, value=lon_right)

                self.DepthData["data"] = fill_array(Z=self.DepthData["data"], lat_idx1=lat_left_idx, lon_idx1=lon_left_idx,
                                                    lat_idx2=lat_right_idx, lon_idx2=lon_right_idx, 
                                                    value=np.clip(np.random.exponential(scale=75, size=1), 1, 1000))

        # smoothing things
        self.DepthData["data"] = np.clip(scipy.ndimage.gaussian_filter(self.DepthData["data"], sigma=[20, 20], mode="constant"), 1, np.infty)

        # log
        self.log_Depth = np.log(self.DepthData["data"])

        # for contour plot
        self.con_ticks = np.log([1.0, 2.0, 5.0, 15.0, 50.0, 150.0, 500.0])
        self.con_ticklabels = [int(np.round(tick, 0)) for tick in np.exp(self.con_ticks)]
        self.con_ticklabels[0] = 0
        self.clev = np.arange(0, self.log_Depth.max(), .1)


    def _sample_wind_data(self):
        """Generates random wind data by overwriting the real data information."""
        # zero out real data
        speed_mps = np.zeros_like(self.WindData["speed_mps"])
        angle = np.zeros_like(self.WindData["angle"])

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
        e = self.WindData["eastward_mps"] * 0.0
        n = self.WindData["northward_mps"] * 0.0

        for lat_idx, _ in enumerate(self.WindData["lat"]):
            for lon_idx, _ in enumerate(self.WindData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=speed_mps[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.WindData["eastward_mps"] = e
        self.WindData["northward_mps"] = n
        self.WindData["eastward_knots"] = mps_to_knots(self.WindData["eastward_mps"])
        self.WindData["northward_knots"] = mps_to_knots(self.WindData["northward_mps"])


    def _sample_current_data(self):
        """Generates random current data by overwriting the real data information."""
        speed_mps = np.zeros_like(self.CurrentData["speed_mps"])
        angle = np.zeros_like(self.CurrentData["angle"])

        # size of homogenous current areas
        lat_n_areas = np.random.randint(5, 20)
        lon_n_areas = np.random.randint(5, 20)

        idx_freq_lat = speed_mps.shape[0] / lat_n_areas
        idx_freq_lon = speed_mps.shape[1] / lon_n_areas

        V_const = np.clip(np.random.exponential(scale=0.2, size=(lat_n_areas, lon_n_areas)), 0.0, 0.5)
        angle_const = np.random.uniform(low=0.0, high=2*math.pi, size=(lat_n_areas, lon_n_areas))

        # sampling
        for lat_idx, lat in enumerate(self.CurrentData["lat"]):
            for lon_idx, lon in enumerate(self.CurrentData["lon"]):

                # no currents at land
                if self._depth_at_latlon(lat_q=lat, lon_q=lon) > 1.0:

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
        e = self.CurrentData["eastward_mps"] * 0.0
        n = self.CurrentData["northward_mps"] * 0.0

        for lat_idx, _ in enumerate(self.CurrentData["lat"]):
            for lon_idx, _ in enumerate(self.CurrentData["lon"]):
                e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=speed_mps[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

        self.CurrentData["eastward_mps"] = e
        self.CurrentData["northward_mps"] = n


    def _sample_wave_data(self):
        """Generates random wave data by overwriting the real data information."""
        height = np.zeros_like(self.WaveData["height"])
        length = np.zeros_like(self.WaveData["length"])
        period = np.zeros_like(self.WaveData["period"])
        angle = np.zeros_like(self.WaveData["angle"])

        # size of homogenous current areas
        lat_n_areas = np.random.randint(5, 20)
        lon_n_areas = np.random.randint(5, 20)

        idx_freq_lat = height.shape[0] / lat_n_areas
        idx_freq_lon = height.shape[1] / lon_n_areas

        height_const = np.random.exponential(scale=0.1, size=(lat_n_areas, lon_n_areas))
        length_const = np.random.exponential(scale=10., size=(lat_n_areas, lon_n_areas))
        period_const = np.random.exponential(scale=1.0, size=(lat_n_areas, lon_n_areas))
        angle_const = np.random.uniform(low=0.0, high=2*math.pi, size=(lat_n_areas, lon_n_areas))

        # sampling
        for lat_idx, lat in enumerate(self.WaveData["lat"]):
            for lon_idx, lon in enumerate(self.WaveData["lon"]):

                # no waves at land
                if self._depth_at_latlon(lat_q=lat, lon_q=lon) > 1.0:

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
        self.WaveData["length"] = np.clip(scipy.ndimage.gaussian_filter(length, sigma=[5, 5], mode="constant"), 0.0001, np.infty)
        self.WaveData["period"] = np.clip(scipy.ndimage.gaussian_filter(period, sigma=[0.1, 0.1], mode="constant"), 0.0001, np.infty)
        self.WaveData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")

        # overwrite other entries
        e = self.WaveData["eastward"] * 0.0
        n = self.WaveData["northward"] * 0.0

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

                if depth_dot <= self.sense_depth:
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

        # init OS
        wp_idx = np.random.uniform(low=int(self.n_wps*0.25), high=int(self.n_wps*0.75), size=(1,)).astype(int)[0]
        lat_init = self.DesiredPath["lat"][wp_idx] if self.mode == "train" else 56.635
        lon_init = self.DesiredPath["lon"][wp_idx] if self.mode == "train" else 7.421
        N_init, E_init, number = to_utm(lat=lat_init, lon=lon_init)

        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = dtr(180.0),
                         u_init    = 0.0,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = 3.0,
                         full_ship = False,
                         cont_acts = True)

        # Critical point: We do not update the UTM number (!) since our simulation primarily takes place in 32U and 32V.
        self.OS.utm_number = number

        # set u-speed to near-convergence
        self.V_c, self.beta_c = self._current_at_latlon(lat_q=lat_init, lon_q=lon_init)
        self.V_w, self.beta_w = self._wind_at_latlon(lat_q=lat_init, lon_q=lon_init)
        self.H = self._depth_at_latlon(lat_q=lat_init, lon_q=lon_init)
        self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = self._wave_at_latlon(lat_q=lat_init, lon_q=lon_init)
        
        if self.T_0_wave == 0.0:
            self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None

        self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps, psi=self.OS.eta[2], V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w, H=self.H,
                                                beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # initialize waypoints
        self.wp1_idx, self.wp1_N, self.wp1_E, self.wp2_idx, self.wp2_N, self.wp2_E = get_init_two_wp(lat_array=self.DesiredPath["lat"], \
            lon_array=self.DesiredPath["lon"], a_n=N_init, a_e=E_init)

        # init cross-track error and desired course
        self.ye, self.desired_course, _ = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K=self.VFG_K)
        self.course_error = angle_to_pi(self.desired_course - self.OS._get_course())

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state


    def _set_state(self):
        # OS information
        cmp1 = self.OS.nu / np.array([7.0, 0.7, 0.004])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

        # path information
        state_path = np.array([self.ye/self.OS.Lpp, self.course_error/math.pi])

        # environmental disturbances
        if self.T_0_wave is None:
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

        # LiDAR
        #state_LiDAR = self._get_closeness_from_lidar(self._sense_LiDAR()[0])

        self.state = np.concatenate([state_OS, state_path, state_env])


    def _update_wps(self):
        """Updates the waypoints for following the desired path."""
        # check whether we need to switch wps
        switch = switch_wp(wp1_N=self.wp1_N, wp1_E=self.wp1_E, wp2_N=self.wp2_N, wp2_E=self.wp2_E, a_N=self.OS.eta[0], a_E=self.OS.eta[1])

        if switch and (self.wp2_idx != (self.DesiredPath["n_wps"]-1)):
            # update waypoint 1
            self.wp1_idx += 1
            self.wp1_N = self.wp2_N
            self.wp1_E = self.wp2_E

            # update waypoint 2
            self.wp2_idx += 1
            self.wp2_N = self.DesiredPath["north"][self.wp2_idx]
            self.wp2_E = self.DesiredPath["east"][self.wp2_idx]


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        a = a.item()
        self.OS._control(a)

        # update agent dynamics
        OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=self.OS.utm_number)

        self.V_c, self.beta_c = self._current_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.V_w, self.beta_w = self._wind_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.H = self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = self._wave_at_latlon(lat_q=OS_lat, lon_q=OS_lon)

        if self.T_0_wave == 0.0:
            self.beta_wave, self.eta_wave, self.T_0_wave, self.lambda_wave = None, None, None, None

        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)
        
        # update waypoints of path
        self._update_wps()

        # compute new cross-track error and desired course
        self.ye, self.desired_course, _ = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K=self.VFG_K)
        self.course_error = angle_to_pi(self.desired_course - self.OS._get_course())

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, self.r, d, {}


    def _calculate_reward(self):
        # cross-track error
        k_ye = 0.05
        self.r_ye = math.exp(-k_ye * abs(self.ye))

        # course error
        k_ce = 5.0
        self.r_ce = math.exp(-k_ce * abs(self.course_error))

        weights = np.array([0.5, 0.5])
        self.r = np.sum(weights * np.array([self.r_ye, self.r_ce])) / np.sum(weights)


    def _done(self):
        """Returns boolean flag whether episode is over."""
        # end episode if OS is too far away from path
        if self.ye > 400:
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
        pos = f"Lat [°]: {OS_lat:.4f}, Lon [°]: {OS_lon:.4f}, " + r"$\psi$ [°]: " + f"{rtd(self.OS.eta[2]):.2f}"  + r", $\chi$ [°]: " + f"{course:.2f}"
        vel = f"u [m/s]: {u:.3f}, v [m/s]: {v:.3f}, r [rad/s]: {r:.3f}"
        
        depth = f"H [m]: {self.H:.2f}"
        wind = r"$V_{\rm wind}$" + f" [kn]: {mps_to_knots(self.V_w):.2f}, " + r"$\psi_{\rm wind}$" + f" [°]: {rtd(self.beta_w):.2f}"
        current = r"$V_{\rm current}$" + f" [m/s]: {self.V_c:.2f}, " + r"$\psi_{\rm current}$" + f" [°]: {rtd(self.beta_c):.2f}"
        wave = r"$\psi_{\rm wave}$" + f" [°]: {rtd(self.beta_wave):.2f}" + r", $\xi_{\rm wave}$ [m]: " + f"{self.eta_wave:.2f}" \
            + r", $T_{\rm wave}$ [s]: " + f"{self.T_0_wave:.2f}" + r", $\lambda_{\rm wave}$ [m]: " + f"{self.lambda_wave:.2f}"

        path_info = r"$y_e$" + f" [m]: {self.ye:.2f}, " + r"$\chi_{\rm desired}$" + f" [°]: {rtd(self.desired_course):.2f}, " \
            + r"$\chi_{\rm error}$" + f" [°]: {rtd(self.course_error):.2f}"
        return ste + ", " + pos + "\n" + vel + ", " + depth + "\n" + wind + "\n" + current + "\n" + wave + "\n" + path_info


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

                ax.set_xlim(max([self.lon_lims[0], OS_lon - self.show_lon_lat/2]), min([self.lon_lims[1], OS_lon + self.show_lon_lat/2]))
                ax.set_ylim(max([self.lat_lims[0], OS_lat - self.show_lon_lat/2]), min([self.lat_lims[1], OS_lat + self.show_lon_lat/2]))
            else:
                ax.text(0.125, 0.8675, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=10, transform=plt.gcf().transFigure)

                ax.set_xlabel("UTM-E [m]", fontsize=10)
                ax.set_ylabel("UTM-N [m]", fontsize=10)

                # reverse xaxis in UTM
                ax.set_xlim(E0 - self.UTM_viz_range_E, E0 + self.UTM_viz_range_E)
                ax.set_ylim(N0 - self.UTM_viz_range_N, N0 + self.UTM_viz_range_N)

            #--------------- depth plot ---------------------
            if self.plot_depth and self.plot_in_latlon:
                cnt_lat, cnt_lat_idx = find_nearest(array=self.DepthData["lat"], value=OS_lat)
                cnt_lon, cnt_lon_idx = find_nearest(array=self.DepthData["lon"], value=OS_lon)

                lower_lat_idx = int(max([cnt_lat_idx - self.half_num_depth_idx, 0]))
                upper_lat_idx = int(min([cnt_lat_idx + self.half_num_depth_idx, len(self.DepthData["lat"]) - 1]))

                lower_lon_idx = int(max([cnt_lon_idx - self.half_num_depth_idx, 0]))
                upper_lon_idx = int(min([cnt_lon_idx + self.half_num_depth_idx, len(self.DepthData["lon"]) - 1]))
                
                #ax.set_xlim(max([self.DepthData["lon"][0],  cnt_lon - self.show_lon_lat/2]), 
                #            min([self.DepthData["lon"][-1], cnt_lon + self.show_lon_lat/2]))
                #ax.set_ylim(max([self.DepthData["lat"][0],  cnt_lat - self.show_lon_lat/2]), 
                #            min([self.DepthData["lat"][-1], cnt_lat + self.show_lon_lat/2]))

                # contour plot from depth data
                con = ax.contourf(self.DepthData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                                self.DepthData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                                self.log_Depth[lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                                self.clev, cmap=cm.ocean)

                # colorbar as legend
                if self.step_cnt == 0:
                    cbar = self.f.colorbar(con, ticks=self.con_ticks)
                    cbar.ax.set_yticklabels(self.con_ticklabels)

            #--------------- wind plot ---------------------
            if self.plot_wind and self.plot_in_latlon:

                # no barb plot if there is no wind data
                if any([OS_lat < min(self.WindData["lat"]),
                        OS_lat > max(self.WindData["lat"]),
                        OS_lon < min(self.WindData["lon"]),
                        OS_lon > max(self.WindData["lon"])]):
                    pass
                else:
                    _, cnt_lat_idx = find_nearest(array=self.WindData["lat"], value=OS_lat)
                    _, cnt_lon_idx = find_nearest(array=self.WindData["lon"], value=OS_lon)

                    lower_lat_idx = int(max([cnt_lat_idx - self.half_num_wind_idx, 0]))
                    upper_lat_idx = int(min([cnt_lat_idx + self.half_num_wind_idx, len(self.WindData["lat"]) - 1]))

                    lower_lon_idx = int(max([cnt_lon_idx - self.half_num_wind_idx, 0]))
                    upper_lon_idx = int(min([cnt_lon_idx + self.half_num_wind_idx, len(self.WindData["lon"]) - 1]))

                    ax.barbs(self.WindData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                            self.WindData["lat"][lower_lat_idx:(upper_lat_idx+1)], 
                            self.WindData["eastward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                            self.WindData["northward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                            length=4, barbcolor="goldenrod")

            #------------------ set OS ------------------------
            # midship
            #ax.plot(OS_lon, OS_lat, marker="o", color="red")
            
            # quick access
            l = self.OS.Lpp/2
            b = self.OS.B/2

            # get rectangle/polygon end points in UTM
            A = (E0 - b, N0 + l)
            B = (E0 + b, N0 + l)
            C = (E0 - b, N0 - l)
            D = (E0 + b, N0 - l)

            # rotate them according to heading
            A = rotate_point(x=A[0], y=A[1], cx=E0, cy=N0, angle=-head0)
            B = rotate_point(x=B[0], y=B[1], cx=E0, cy=N0, angle=-head0)
            C = rotate_point(x=C[0], y=C[1], cx=E0, cy=N0, angle=-head0)
            D = rotate_point(x=D[0], y=D[1], cx=E0, cy=N0, angle=-head0)

            if self.plot_in_latlon:

                # convert them to lat/lon
                A_lat, A_lon = to_latlon(north=A[1], east=A[0], number=self.OS.utm_number)
                B_lat, B_lon = to_latlon(north=B[1], east=B[0], number=self.OS.utm_number)
                C_lat, C_lon = to_latlon(north=C[1], east=C[0], number=self.OS.utm_number)
                D_lat, D_lon = to_latlon(north=D[1], east=D[0], number=self.OS.utm_number)

                # draw the polygon (A is included twice to create a closed shape)
                lons = [A_lon, B_lon, D_lon, C_lon, A_lon]
                lats = [A_lat, B_lat, D_lat, C_lat, A_lat]
                ax.plot(lons, lats, color="red", linewidth=2.0)
            else:
                ax.plot([A[0], B[0], D[0], C[0], A[0]], [A[1], B[1], D[1], C[1], A[1]], color="red", linewidth=2.0)

            #--------------------- Desired path ------------------------
            if self.plot_path:

                if self.plot_in_latlon:
                    ax.plot(self.DesiredPath["lon"], self.DesiredPath["lat"], marker='o', color="salmon", linewidth=1.0, markersize=3)

                    # current waypoints
                    wp1_lat, wp1_lon = to_latlon(north=self.wp1_N, east=self.wp1_E, number=self.OS.utm_number)
                    wp2_lat, wp2_lon = to_latlon(north=self.wp2_N, east=self.wp2_E, number=self.OS.utm_number)
                    ax.plot([wp1_lon, wp2_lon], [wp1_lat, wp2_lat], color="springgreen", linewidth=1.0, markersize=3)

                    # wp switching line
                    #pi_path = bng_abs(N0=self.wp1_N, E0=self.wp1_E, N1=self.wp2_N, E1=self.wp2_E)
                    #pi_lot = angle_to_2pi(pi_path + dtr(90.0))
                    #delta_E, delta_N = xy_from_polar(r=100000, angle=pi_lot)
                    #end_lat, end_lon = to_latlon(north=self.wp2_N + delta_N, east=self.wp2_E + delta_E, number=self.OS.utm_number)
                    #ax.plot([wp2_lon, end_lon], [wp2_lat, end_lat])

                    # desired course
                    ye, dc, pi_path = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K= self.VFG_K)
                    #dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=dc)
                    #dc_lat, dc_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    #ax.arrow(x=OS_lon, y=OS_lat, dx=dc_lon-OS_lon, dy=dc_lat-OS_lat, length_includes_head=True,
                    #        width=0.0004, head_width=0.002, head_length=0.003, color="salmon")

                    # actual course
                    #dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=self.OS._get_course())
                    #ac_lat, ac_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    #ax.arrow(x=OS_lon, y=OS_lat, dx=ac_lon-OS_lon, dy=ac_lat-OS_lat, length_includes_head=True,
                    #        width=0.0004, head_width=0.002, head_length=0.003, color="rosybrown")

                    # cross-track error
                    if ye < 0:
                        dE, dN = xy_from_polar(r=abs(ye), angle=angle_to_2pi(pi_path + dtr(90.0)))
                    else:
                        dE, dN = xy_from_polar(r=ye, angle=angle_to_2pi(pi_path - dtr(90.0)))
                    yte_lat, yte_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    ax.plot([OS_lon, yte_lon], [OS_lat, yte_lat], color="salmon")

                else:
                    ax.plot(self.DesiredPath["east"], self.DesiredPath["north"], marker='o', color="salmon", linewidth=1.0, markersize=3)

                    # current waypoints
                    ax.plot([self.wp1_E, self.wp2_E], [self.wp1_N, self.wp2_N], color="springgreen", linewidth=1.0, markersize=3)

                    # desired course
                    #ye, dc, pi_path = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K= self.VFG_K)
                    #dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=dc)
                    #ax.arrow(x=E0, y=N0, dx=dE, dy=dN, length_includes_head=True,
                    #        width=0.0004, head_width=0.002, head_length=0.003, color="salmon")

                    # actual course
                    #dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=self.OS._get_course())
                    #ax.arrow(x=E0, y=N0, dx=dE, dy=dN, length_includes_head=True,
                    #        width=0.0004, head_width=0.002, head_length=0.003, color="rosybrown")

                    # cross-track error
                    if ye < 0:
                        dE, dN = xy_from_polar(r=abs(ye), angle=angle_to_2pi(pi_path + dtr(90.0)))
                    else:
                        dE, dN = xy_from_polar(r=ye, angle=angle_to_2pi(pi_path - dtr(90.0)))
                    ax.plot([E0, E0+dE], [N0, N0+dN], color="salmon")

            #--------------------- Current data ------------------------
            if self.plot_current and self.plot_in_latlon:

                _, cnt_lat_idx = find_nearest(array=self.CurrentData["lat"], value=OS_lat)
                _, cnt_lon_idx = find_nearest(array=self.CurrentData["lon"], value=OS_lon)

                lower_lat_idx = int(max([cnt_lat_idx - self.half_num_current_idx, 0]))
                upper_lat_idx = int(min([cnt_lat_idx + self.half_num_current_idx, len(self.CurrentData["lat"]) - 1]))

                lower_lon_idx = int(max([cnt_lon_idx - self.half_num_current_idx, 0]))
                upper_lon_idx = int(min([cnt_lon_idx + self.half_num_current_idx, len(self.CurrentData["lon"]) - 1]))
                
                ax.quiver(self.CurrentData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                          self.CurrentData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                          self.CurrentData["eastward_mps"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                          self.CurrentData["northward_mps"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                          headwidth=2.0, color="whitesmoke", scale=10)

            #--------------------- Wave data ------------------------
            if self.plot_waves and self.plot_in_latlon:

                _, cnt_lat_idx = find_nearest(array=self.WaveData["lat"], value=OS_lat)
                _, cnt_lon_idx = find_nearest(array=self.WaveData["lon"], value=OS_lon)

                lower_lat_idx = int(max([cnt_lat_idx - self.half_num_wave_idx, 0]))
                upper_lat_idx = int(min([cnt_lat_idx + self.half_num_wave_idx, len(self.WaveData["lat"]) - 1]))

                lower_lon_idx = int(max([cnt_lon_idx - self.half_num_wave_idx, 0]))
                upper_lon_idx = int(min([cnt_lon_idx + self.half_num_wave_idx, len(self.WaveData["lon"]) - 1]))
                
                ax.quiver(self.WaveData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                          self.WaveData["lat"][lower_lat_idx:(upper_lat_idx+1)],
                          self.WaveData["eastward"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)], 
                          self.WaveData["northward"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                          headwidth=2.0, color="salmon", scale=10)

            #--------------------- LiDAR sensing ------------------------
            if self.plot_lidar and self.plot_in_latlon:
                _, lidar_lat_lon = self._sense_LiDAR()

                for _, latlon in enumerate(lidar_lat_lon):
                    ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.75)#, alpha=(idx+1)/len(lidar_lat_lon))
        
        # ------------------------------ reward plot --------------------------------
        if self.plot_reward:
            if self.step_cnt == 0:
                self.ax2.clear()
                self.ax2.old_time = 0
                self.ax2.old_r_ye = 0
                self.ax2.old_r_ce = 0

            self.ax2.set_xlim(0, self._max_episode_steps)
            self.ax2.set_ylim(0, 1)
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward")

            self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ye, self.r_ye], color = "black", label="Cross-track error")
            self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_r_ce, self.r_ce], color = "grey", label="Course error")
            
            if self.step_cnt == 0:
                self.ax2.legend()

            self.ax2.old_time = self.step_cnt
            self.ax2.old_r_ye = self.r_ye
            self.ax2.old_r_ce = self.r_ce

        #plt.gca().set_aspect('equal')
        plt.pause(0.001)
