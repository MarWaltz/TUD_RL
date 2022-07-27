import copy
import math
import pickle

import gym
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from tud_rl.envs._envs.HHOS_Fnc import (VFG, Z_at_latlon, cte, find_nearest,
                                        get_init_two_wp, mps_to_knots,
                                        switch_wp, to_latlon, to_utm)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (NM_to_meter, angle_to_2pi, bng_abs,
                                         dtr, rtd, xy_from_polar)
from tud_rl.envs._envs.VesselPlots import rotate_point


class HHOS_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2 from Hamburg to Oslo."""

    def __init__(self):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0   # simulation time interval (in s)

        # LiDAR
        self.lidar_range       = NM_to_meter(3.0)                                                # range of LiDAR sensoring in m
        self.lidar_n_beams     = 25                                                              # number of beams
        self.lidar_beam_angles = np.linspace(0.0, 2*math.pi, self.lidar_n_beams, endpoint=False) # beam angles
        self.n_dots_per_beam   = 20                                                              # number of subpoints per beam
        self.d_dots_per_beam   = np.linspace(start=0.0, stop=self.lidar_range, num=self.lidar_n_beams+1, endpoint=True)[1:] # distances from midship of subpoints per beam
        self.sense_depth       = 15  # water depth at which LiDAR recognizes an obstacle

        # vector field guidance
        self.VFG_K = 0.001

        # data loading
        self._load_desired_path(path_to_desired_path="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS")
        self._load_depth_data(path_to_depth_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/DepthData")
        self._load_wind_data(path_to_wind_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/winds")
        self._load_current_data(path_to_current_data="C:/Users/MWaltz/Desktop/Forschung/RL_packages/HHOS/currents")

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.10
        self.half_num_depth_idx = math.ceil((self.show_lon_lat / 2.0) / self.DepthData["metaData"]["cellsize"]) + 1
        self.half_num_wind_idx = math.ceil((self.show_lon_lat / 2.0) / self.WindData["metaData"]["cellsize"]) + 1
        self.half_num_current_idx = math.ceil((self.show_lon_lat / 2.0) / np.mean(np.diff(self.CurrentData["lat"]))) + 1

        # visualization
        self.plot_in_latlon = False         # if false, plots in UTM coordinates
        self.plot_depth = True
        self.plot_path = True
        self.plot_wind = True
        self.plot_current = True
        self.plot_lidar = False

        # data range
        self.lon_lims = [4.83, 14.33]
        self.lat_lims = [51.83, 60.5]

        if not self.plot_in_latlon:
            self.show_lon_lat = np.clip(self.show_lon_lat, 0.005, 5.95)
            self.UTM_viz_range_E = abs(to_utm(lat=52.0, lon=6.0001)[1] - to_utm(lat=52.0, lon=6.0001+self.show_lon_lat/2)[1])
            self.UTM_viz_range_N = abs(to_utm(lat=50.0, lon=8.0)[0] - to_utm(lat=50.0+self.show_lon_lat/2, lon=8.0)[0])

        # custom inits
        self.r = 0
        self._max_episode_steps = np.infty


    def _load_desired_path(self, path_to_desired_path):
        with open(f"{path_to_desired_path}/Path_latlon.pickle", "rb") as f:
            self.DesiredPath = pickle.load(f)
        
        # store number of waypoints
        self.DesiredPath["n_wps"] = len(self.DesiredPath["lat"])

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
        lat_init = 56.635
        lon_init = 7.421
        N_init, E_init, number = to_utm(lat=lat_init, lon=lon_init)

        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = dtr(0.0),
                         u_init    = 0.0,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = 1.8,
                         full_ship = True)

        # Critical point: We do not update the UTM number (!) since our simulation primarily takes place in 32U and 32V.
        self.OS.utm_number = number

        # set u-speed to near-convergence
        V_c, beta_c = self._current_at_latlon(lat_q=lat_init, lon_q=lon_init)
        self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps, psi=self.OS.eta[2], V_c=0.0, beta_c=0.0)

        # initialize waypoints
        self.wp1_idx, self.wp1_N, self.wp1_E, self.wp2_idx, self.wp2_N, self.wp2_E = get_init_two_wp(lat_array=self.DesiredPath["lat"], \
            lon_array=self.DesiredPath["lon"], a_n=N_init, a_e=E_init)

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state


    def _set_state(self):
        self.state = None


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
        self.OS._control(a)

        # update agent dynamics
        OS_lat, OS_lon = to_latlon(north=self.OS.eta[0], east=self.OS.eta[1], number=self.OS.utm_number)
        V_c, beta_c = self._current_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        self.OS._upd_dynamics(V_c=0.5, beta_c=dtr(0.0))

        # update waypoints of path
        self._update_wps()

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
        vel = f"u [m/s]: {u:.3f}, v [m/s]: {v:.3f}, r [rad/s]: {r:.3f}"
        
        depth = f"Water depth [m]: {self._depth_at_latlon(lat_q=OS_lat, lon_q=OS_lon):.2f}"

        wind_speed, wind_angle = self._wind_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        wind = f"Wind speed [kn]: {mps_to_knots(wind_speed):.2f}, Wind direction [°]: {rtd(wind_angle):.2f}"

        current_speed, current_angle = self._current_at_latlon(lat_q=OS_lat, lon_q=OS_lon)
        current = f"Current speed [m/s]: {current_speed:.2f}, Current direction [°]: {rtd(current_angle):.2f}"

        ye, desired_course, _ = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K=self.VFG_K)
        path_info = f"CTE [m]: {ye:.2f}, Desired course [°]: {rtd(desired_course):.2f}, Course error [°]: {rtd(desired_course - self.OS._get_course()):.2f}"
        return ste + "\n" + pos + ", " + vel + "\n" + depth + ", " + wind + "\n" + current + "\n" + path_info

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
        N0, E0, head0 = self.OS.eta
        OS_lat, OS_lon = to_latlon(north=N0, east=E0, number=self.OS.utm_number)

        for ax in [self.ax]:
            ax.clear()

            # general information
            ax.text(0.125, 0.8675, self.__str__(OS_lat=OS_lat, OS_lon=OS_lon), fontsize=10, transform=plt.gcf().transFigure)

            if self.plot_in_latlon:
                ax.set_xlabel("Longitude [°]", fontsize=10)
                ax.set_ylabel("Latitude [°]", fontsize=10)

                ax.set_xlim(max([self.lon_lims[0], OS_lon - self.show_lon_lat/2]), min([self.lon_lims[1], OS_lon + self.show_lon_lat/2]))
                ax.set_ylim(max([self.lat_lims[0], OS_lat - self.show_lon_lat/2]), min([self.lat_lims[1], OS_lat + self.show_lon_lat/2]))
            else:
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
                    dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=dc)
                    dc_lat, dc_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    plt.arrow(x=OS_lon, y=OS_lat, dx=dc_lon-OS_lon, dy=dc_lat-OS_lat, length_includes_head=True,
                            width=0.0004, head_width=0.002, head_length=0.003, color="salmon")

                    # actual course
                    dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=self.OS._get_course())
                    ac_lat, ac_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    plt.arrow(x=OS_lon, y=OS_lat, dx=ac_lon-OS_lon, dy=ac_lat-OS_lat, length_includes_head=True,
                            width=0.0004, head_width=0.002, head_length=0.003, color="rosybrown")

                    # cross-track error
                    if ye < 0:
                        dE, dN = xy_from_polar(r=abs(ye), angle=angle_to_2pi(pi_path + dtr(90.0)))
                    else:
                        dE, dN = xy_from_polar(r=ye, angle=angle_to_2pi(pi_path - dtr(90.0)))
                    yte_lat, yte_lon = to_latlon(north=self.OS.eta[0]+dN, east=self.OS.eta[1]+dE, number=self.OS.utm_number)
                    plt.plot([OS_lon, yte_lon], [OS_lat, yte_lat], color="salmon")

                else:
                    ax.plot(self.DesiredPath["east"], self.DesiredPath["north"], marker='o', color="salmon", linewidth=1.0, markersize=3)

                    # current waypoints
                    ax.plot([self.wp1_E, self.wp2_E], [self.wp1_N, self.wp2_N], color="springgreen", linewidth=1.0, markersize=3)

                    # desired course
                    ye, dc, pi_path = VFG(N1=self.wp1_N, E1=self.wp1_E, N2=self.wp2_N, E2=self.wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1], K= self.VFG_K)
                    dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=dc)
                    plt.arrow(x=E0, y=N0, dx=dE, dy=dN, length_includes_head=True,
                            width=0.0004, head_width=0.002, head_length=0.003, color="salmon")

                    # actual course
                    dE, dN = xy_from_polar(r=3*self.OS.Lpp, angle=self.OS._get_course())
                    plt.arrow(x=E0, y=N0, dx=dE, dy=dN, length_includes_head=True,
                            width=0.0004, head_width=0.002, head_length=0.003, color="rosybrown")

                    # cross-track error
                    if ye < 0:
                        dE, dN = xy_from_polar(r=abs(ye), angle=angle_to_2pi(pi_path + dtr(90.0)))
                    else:
                        dE, dN = xy_from_polar(r=ye, angle=angle_to_2pi(pi_path - dtr(90.0)))
                    plt.plot([E0, E0+dE], [N0, N0+dN], color="salmon")

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
                          headwidth=2.0, color="whitesmoke", scale=5)

            #--------------------- LiDAR sensing ------------------------
            if self.plot_lidar and self.plot_in_latlon:
                _, lidar_lat_lon = self._sense_LiDAR()

                for _, latlon in enumerate(lidar_lat_lon):
                    ax.plot([OS_lon, latlon[1]], [OS_lat, latlon[0]], color="goldenrod", alpha=0.75)#, alpha=(idx+1)/len(lidar_lat_lon))
        
        plt.gca().set_aspect('equal')
        plt.pause(0.001)
