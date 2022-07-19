import copy
import math
import pickle

import gym
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from tud_rl.envs._envs.HHOS_Fnc import (Z_at_latlon, find_nearest,
                                        mps_to_knots, to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (NM_to_meter, angle_to_2pi, dtr, rtd,
                                         xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 from Hamburg to Oslo."""

    def __init__(self):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0   # simulation time interval (in s)

        # LiDAR
        self.lidar_range       = NM_to_meter(1.0)                                                # range of LiDAR sensoring in m
        self.lidar_n_beams     = 25                                                              # number of beams
        self.lidar_beam_angles = np.linspace(0.0, 2*math.pi, self.lidar_n_beams, endpoint=False) # beam angles
        self.n_dots_per_beam   = 20                                                              # number of subpoints per beam
        self.d_dots_per_beam   = np.linspace(start=0.0, stop=self.lidar_range, num=self.lidar_n_beams+1, endpoint=True)[1:] # distances from midship of subpoints per beam
        self.sense_depth       = 15  # water depth at which LiDAR recognizes an obstacle

        # data loading
        self._load_desired_path(path_to_desired_path="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS")
        self._load_depth_data(path_to_depth_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/DepthData")
        self._load_wind_data(path_to_wind_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/winds")

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.5
        self.half_num_depth_idx = int((self.show_lon_lat / 2.0) / self.DepthData["metaData"]["cellsize"])
        self.half_num_wind_idx  = int((self.show_lon_lat / 2.0) / self.WindData["metaData"]["cellsize"])

        # visualization
        self.plot_wind  = False
        self.plot_path  = True
        self.plot_lidar = False

        # custom inits
        self.r = 0
        self._max_episode_steps = np.infty


    def _load_desired_path(self, path_to_desired_path):
        with open(f"{path_to_desired_path}/Path_latlon.pickle", "rb") as f:
            self.DesiredPath = pickle.load(f)
        
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


    def _depth_at_latlon(self, lat_q, lon_q):
        """Computes the water depth at a (queried) longitude-latitude position based on linear interpolation."""
        return Z_at_latlon(Z=self.DepthData["data"], lat_array=self.DepthData["lat"], lon_array=self.DepthData["lon"],
                           lat_q=lat_q, lon_q=lon_q)


    def _wind_at_latlon(self, lat_q, lon_q):
        """Computes the wind speed and angle at a (queried) longitude-latitude position based on linear interpolation.
        Returns: (speed, angle)"""
        speed = Z_at_latlon(Z=self.WindData["speed_mps"], lat_array=self.WindData["lat"], lon_array=self.WindData["lon"],
                            lat_q=lat_q, lon_q=lon_q)
        angle = angle_to_2pi(Z_at_latlon(Z=self.WindData["angle"], lat_array=self.WindData["lat"], 
                                         lon_array=self.WindData["lon"], lat_q=lat_q, lon_q=lon_q))
        return speed, angle


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
        lat_init = 54.125
        lon_init = 7.88
        N_init, E_init, number = to_utm(lat=lat_init, lon=lon_init)

        self.OS = KVLCC2(N_init   = N_init, 
                         E_init   = E_init, 
                         psi_init = dtr(355),
                         u_init   = 0.0,
                         v_init   = 0.0,
                         r_init   = 0.0,
                         delta_t  = self.delta_t,
                         N_max    = np.infty,
                         E_max    = np.infty,
                         nps      = 1.8)

        # Critical point: We do not update the UTM number (!) since our simulation primarily takes place in 32U and 32V.
        self.OS.utm_number = number

        # set u-speed to near-convergence
        # Note: if we don't do this, the TCPA calculation for spawning other vessels is heavily biased
        self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps, psi=self.OS.eta[2])

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _set_state(self):
        self.state = None

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        self.OS._control(a)

        # update agent dynamics
        self.OS._upd_dynamics()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
        return self.state, self.r, d, {}

    def __str__(self, OS_lat, OS_lon) -> str:
        u, v, r = self.OS.nu

        ste = f"Step: {self.step_cnt}"
        pos = f"Lat [°]: {OS_lat:.4f}, Lon [°]: {OS_lon:.4f}, " + r"$\psi$ [°]: " + f"{rtd(self.OS.eta[2]):.2f}"
        vel = f"u [m/s]: {u:.2f}, v [m/s]: {v:.2f}, r [m/s]: {r:.2f}"
        
        depth = f"Water depth [m]: {self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon):.2f}"
        wind_speed, wind_angle = self._wind_at_latlon(lat_q=OS_lat, lon_q=OS_lon)

        wind = f"Wind speed [kn]: {mps_to_knots(wind_speed):.2f}, Wind direction [°]: {rtd(wind_angle):.2f}"
        return ste + "\n" + pos + "\n" + vel + "\n" + depth + ", " + wind

    def _calculate_reward(self):
        return 0.0

    def _done(self):
        """Returns boolean flag whether episode is over."""
        d = False
        return d

    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # check whether figure has been initialized
        if len(plt.get_fignums()) == 0:
            self.f, self.ax = plt.subplots(1, 1, figsize=(10, 10))

            plt.ion()
            plt.show()

        # ------------------------------ ship movement --------------------------------
        # get position of OS in lat/lon
        OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=self.OS.utm_number)

        for ax in [self.ax]:
            ax.clear()

            #--------------- depth plot ---------------------
            cnt_lat, cnt_lat_idx = find_nearest(array=self.DepthData["lat"], value=OS_lat)
            cnt_lon, cnt_lon_idx = find_nearest(array=self.DepthData["lon"], value=OS_lon)

            lower_lat_idx = int(max([cnt_lat_idx - self.half_num_depth_idx, 0]))
            upper_lat_idx = int(min([cnt_lat_idx + self.half_num_depth_idx, len(self.DepthData["lat"]) - 1]))

            lower_lon_idx = int(max([cnt_lon_idx - self.half_num_depth_idx, 0]))
            upper_lon_idx = int(min([cnt_lon_idx + self.half_num_depth_idx, len(self.DepthData["lon"]) - 1]))
            
            ax.set_xlim(max([self.DepthData["lon"][0],  cnt_lon - self.show_lon_lat/2]), 
                        min([self.DepthData["lon"][-1], cnt_lon + self.show_lon_lat/2]))
            ax.set_ylim(max([self.DepthData["lat"][0],  cnt_lat - self.show_lon_lat/2]), 
                        min([self.DepthData["lat"][-1], cnt_lat + self.show_lon_lat/2]))

            ax.set_xlabel("Longitude [°]", fontsize=10)
            ax.set_ylabel("Latitude [°]", fontsize=10)

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
            if self.plot_wind:
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

                    # swapaxes necessary in barbs-plots
                    ax.barbs(self.WindData["lon"][lower_lon_idx:(upper_lon_idx+1)], 
                            self.WindData["lat"][lower_lat_idx:(upper_lat_idx+1)], 
                            self.WindData["eastward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                            self.WindData["northward_knots"][lower_lat_idx:(upper_lat_idx+1), lower_lon_idx:(upper_lon_idx+1)],
                            length=4, barbcolor="goldenrod")

            #------------------ set OS ------------------------
            # midship
            #ax.plot(OS_lon, OS_lat, marker="o", color="red")
            ax.text(0.125, 0.9, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=10, transform=plt.gcf().transFigure)
            
            # quick access
            N0, E0, head0 = self.OS.eta
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

            # convert them to lat/lon
            A_lat, A_lon = to_latlon(north=A[1], east=A[0], number=self.OS.utm_number)
            B_lat, B_lon = to_latlon(north=B[1], east=B[0], number=self.OS.utm_number)
            C_lat, C_lon = to_latlon(north=C[1], east=C[0], number=self.OS.utm_number)
            D_lat, D_lon = to_latlon(north=D[1], east=D[0], number=self.OS.utm_number)

            # draw the polygon (A is included twice to create a closed shape)
            lons = [A_lon, B_lon, D_lon, C_lon, A_lon]
            lats = [A_lat, B_lat, D_lat, C_lat, A_lat]
            ax.plot(lons, lats, color="red", linewidth=2.0)

            #--------------------- Desired path ------------------------
            if self.plot_path:
                ax.plot(self.DesiredPath["lon"], self.DesiredPath["lat"], color="salmon", linewidth=1.0)

            #--------------------- LiDAR sensing ------------------------
            if self.plot_lidar:
                _, lidar_lat_lon = self._sense_LiDAR()

                for _, latlon in enumerate(lidar_lat_lon):
                    ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.75)#, alpha=(idx+1)/len(lidar_lat_lon))

        plt.pause(0.001)
