from tud_rl.envs._envs.HHOS_Env import *
from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter


class HHOS_PathFollowing_Validation(HHOS_Env):
    def __init__(self, nps_control_follower:bool, data:str, val_disturbance:str):
        super().__init__(nps_control_follower=nps_control_follower, thrust_control_planner=None, data=data, \
            scenario_based=None, N_TSs_max=0, N_TSs_random=False,w_ye=None, w_ce=None, w_coll=None, w_comf=None, w_speed=None)

        assert val_disturbance in [None, "currents", "winds", "waves"],\
            "Unknown environmental disturbance to validate. Should be 'currents', 'winds', or 'waves'."
        self.val_disturbance = val_disturbance

        if val_disturbance == "currents":
            self.plot_current = True
        elif val_disturbance == "winds":
            self.plot_wind = True
        elif val_disturbance == "waves":
            self.plot_waves = True

        # gym inherits
        OS_infos = 7 if self.nps_control_follower else 5
        path_infos = 2
        env_infos = 9
        obs_size = OS_infos + path_infos + env_infos
        act_size = 2 if self.nps_control_follower else 1

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(act_size, -1, dtype=np.float32), 
                                       high = np.full(act_size,  1, dtype=np.float32))
        if self.val_disturbance is None:
            self._max_episode_steps = np.inf
        else:
            self._max_episode_steps = 600

        # vessel config
        self.desired_V = 3.0
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)
        self.nps_inc = 0.25
        self.nps_min = 0.5
        self.nps_max = 5.0

        # viz
        self.plotter = HHOSPlotter()

    def reset(self):
        return super().reset(OS_wp_idx=0)

    def _sample_depth_data(self, OS_lat, OS_lon):
        """Generates random depth data."""
        self.DepthData = {}
        self.DepthData["lat"] = np.linspace(self.lat_lims[0], self.lat_lims[1] + self.off, num=500)
        self.DepthData["lon"] = np.linspace(self.lon_lims[0], self.lon_lims[1], num=500)
        self.DepthData["data"] = np.ones((len(self.DepthData["lat"]), len(self.DepthData["lon"])))

        while True:
            # sample distances to waypoints
            d_left  = np.ones(self.n_wps_glo) * 150
            d_right = np.ones(self.n_wps_glo) * 150
            depth = np.ones(self.n_wps_glo) * 50

            # fill close points at that distance away from the path
            lat_path = self.GlobalPath["lat"]
            lon_path = self.GlobalPath["lon"]

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

    def _sample_current_data(self):
        """Generates random current data."""
        if self.val_disturbance == "currents":
            self.CurrentData = {}
            self.CurrentData["lat"] = copy(self.DepthData["lat"])
            self.CurrentData["lon"] = copy(self.DepthData["lon"])

            speed_mps = np.zeros((len(self.CurrentData["lat"]), len(self.CurrentData["lon"])))
            angle = np.zeros_like(speed_mps)

            V_const = 1.0

            for lat_idx, lat in enumerate(self.CurrentData["lat"]):
                for lon_idx, lon in enumerate(self.CurrentData["lon"]):

                    # only on river
                    if self.DepthData["data"][lat_idx, lon_idx] > 1.0:

                        if (56.06 <= lat <= 56.08):
                            speed_mps[lat_idx, lon_idx] = V_const
                            angle[lat_idx, lon_idx] = 3/2*np.pi

                        elif (56.10 <= lat <= 56.12):
                            speed_mps[lat_idx, lon_idx] = V_const
                            angle[lat_idx, lon_idx] = 1/2*np.pi

            # smoothing things
            #self.CurrentData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[1, 1], mode="constant"), 0.0, 0.5)
            #self.CurrentData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")
            self.CurrentData["speed_mps"] = speed_mps
            self.CurrentData["angle"] = angle

            # overwrite other entries
            e = np.zeros_like(speed_mps)
            n = np.zeros_like(speed_mps)

            for lat_idx, _ in enumerate(self.CurrentData["lat"]):
                for lon_idx, _ in enumerate(self.CurrentData["lon"]):
                    e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=speed_mps[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

            self.CurrentData["eastward_mps"] = e
            self.CurrentData["northward_mps"] = n
        else:
            # clear wave impact
            super()._sample_current_data()
            for key in self.CurrentData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.CurrentData[key] *= 0.0

    def _sample_wave_data(self):
        """Generates random wave data by overwriting the real data information."""
        if self.val_disturbance == "waves":
            self.WaveData = {}
            self.WaveData["lat"] = copy(self.DepthData["lat"])
            self.WaveData["lon"] = copy(self.DepthData["lon"])

            height = np.zeros((len(self.WaveData["lat"]), len(self.WaveData["lon"])))
            length = np.zeros_like(height)
            period = np.zeros_like(height)
            angle = np.zeros_like(height)

            height_const = 1.0
            length_const = 20
            period_const = 3

            # sampling
            for lat_idx, lat in enumerate(self.WaveData["lat"]):
                for lon_idx, lon in enumerate(self.WaveData["lon"]):

                    # only on river
                    if self.DepthData["data"][lat_idx, lon_idx] > 1.0:

                        if (56.06 <= lat <= 56.08):
                            height[lat_idx, lon_idx] = height_const
                            length[lat_idx, lon_idx] = length_const
                            period[lat_idx, lon_idx] = period_const
                            angle[lat_idx, lon_idx] = 3/2*np.pi

                        elif (56.10 <= lat <= 56.12):
                            height[lat_idx, lon_idx] = height_const
                            length[lat_idx, lon_idx] = length_const
                            period[lat_idx, lon_idx] = period_const
                            angle[lat_idx, lon_idx] = 1/2*np.pi

            # smoothing things
            self.WaveData["height"] = height
            self.WaveData["length"] = length
            self.WaveData["period"] = period
            self.WaveData["angle"] = angle

            # overwrite other entries
            e = np.zeros_like(height)
            n = np.zeros_like(height)

            for lat_idx, _ in enumerate(self.WaveData["lat"]):
                for lon_idx, _ in enumerate(self.WaveData["lon"]):
                    e[lat_idx, lon_idx], n[lat_idx, lon_idx] = xy_from_polar(r=height[lat_idx, lon_idx], angle=angle_to_2pi(angle[lat_idx, lon_idx] - math.pi))

            self.WaveData["eastward"] = e
            self.WaveData["northward"] = n
        else:
            # clear wave impact
            super()._sample_wave_data()
            for key in self.WaveData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.WaveData[key] *= 0.0

    def _sample_global_path(self):
        """Constructs a straight path with n_wps way points, each being of length l apart from its neighbor in the lat-lon-system.
        The agent should follows the path always in direction of increasing indices."""
        self.GlobalPath = {"n_wps" : self.n_wps_glo}

        # do it until we have a path whichs stays in our simulation domain
        while True:

            # sample starting point
            lat = np.zeros(self.n_wps_glo)
            lon = np.zeros(self.n_wps_glo)
            lat[0] = 56.0
            lon[0] = 9.0

            # sample other points
            ang = 0 # np.random.uniform(0, 2*math.pi)
            #ang_diff = dtr(np.random.uniform(-20., 20.))
            #ang_diff2 = 0.0
            for n in range(1, self.n_wps_glo):
                
                # next angle
                #ang_diff2 = 0.5 * ang_diff2 + 0.5 * dtr(np.random.uniform(-5.0, 5.0))
                #ang_diff = 0.5 * ang_diff + 0.5 * ang_diff2 + 0.0 * dtr(np.random.uniform(-5.0, 5.0))
                #ang = angle_to_2pi(ang + ang_diff)

                # next point
                lon_diff, lat_diff = xy_from_polar(r=self.l_seg_path, angle=ang)
                lat[n] = lat[n-1] + lat_diff
                lon[n] = lon[n-1] + lon_diff

            if all(self.lat_lims[0] <= lat) and all(self.lat_lims[1] >= lat) and \
                all(6.1 <= lon) and all(11.9 >= lon):
                break

        self.GlobalPath["lat"] = lat
        self.GlobalPath["lon"] = lon

        # add utm coordinates
        path_n = np.zeros_like(self.GlobalPath["lat"])
        path_e = np.zeros_like(self.GlobalPath["lon"])

        for idx in range(len(path_n)):
            path_n[idx], path_e[idx], _ = to_utm(lat=self.GlobalPath["lat"][idx], lon=self.GlobalPath["lon"][idx])
        
        self.GlobalPath["north"] = path_n
        self.GlobalPath["east"] = path_e

        # overwrite data range
        self.off = 15*self.l_seg_path
        self.lat_lims = [np.min(lat)-self.off, np.max(lat)+self.off]
        self.lon_lims = [np.min(lon)-self.off, np.max(lon)+self.off]

    def _sample_wind_data(self):
        """Generates random wind data."""
        if self.val_disturbance == "winds":

            self.WindData = {}
            self.WindData["lat"] = copy(self.DepthData["lat"])
            self.WindData["lon"] = copy(self.DepthData["lon"])

            speed_mps = np.zeros((len(self.WindData["lat"]), len(self.WindData["lon"])))
            angle = np.zeros_like(speed_mps)
            V_const = 15.0

            for lat_idx, lat in enumerate(self.WindData["lat"]):
                for lon_idx, lon in enumerate(self.WindData["lon"]):

                    # only on river
                    if self.DepthData["data"][lat_idx, lon_idx] > 1.0:

                        if (56.06 <= lat <= 56.08):
                            speed_mps[lat_idx, lon_idx] = V_const
                            angle[lat_idx, lon_idx] = 3/2*np.pi

                        elif (56.10 <= lat <= 56.12):
                            speed_mps[lat_idx, lon_idx] = V_const
                            angle[lat_idx, lon_idx] = 1/2*np.pi

            # smoothing things
            #self.CurrentData["speed_mps"] = np.clip(scipy.ndimage.gaussian_filter(speed_mps, sigma=[1, 1], mode="constant"), 0.0, 0.5)
            #self.CurrentData["angle"] = scipy.ndimage.gaussian_filter(angle, sigma=[0.2, 0.2], mode="constant")
            self.WindData["speed_mps"] = speed_mps
            self.WindData["angle"] = angle

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
        else:
            # clear wind impact
            super()._sample_wind_data()
            for key in self.WindData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.WindData[key] *= 0.0

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        a = a.flatten()
        self.a = a
        self._OS_control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # environmental effects
        self._update_disturbances()

        # set the local path
        if self.step_cnt % self.loc_path_upd_freq == 0:
            self._update_local_path()

        # update OS waypoints of global and local path
        self.OS = self._init_wps(self.OS, "global")
        self.OS = self._init_wps(self.OS, "local")

        # compute new cross-track error and course error (for local and global path)
        self._set_cte(path_level="global")
        self._set_cte(path_level="local")
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # increase step cnt and overall simulation time
        self.sim_t += self.delta_t
        self.step_cnt += 1
        
        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()

        # viz
        self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                    glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                    T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                         rud_angle=self.OS.rud_angle, nps=self.OS.nps)
        return self.state, self.r, d, {}

    def _OS_control(self, a):
        """Performs the control action for the own ship."""
        # rudder control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.rud_angle = np.clip(self.OS.rud_angle + float(a[0])*self.rud_angle_inc, -self.rud_angle_max, self.rud_angle_max)

        # nps control
        if self.nps_control_follower:
            assert -1 <= float(a[1]) <= 1, "Unknown action."
            self.OS.nps = np.clip(self.OS.nps + float(a[1])*self.nps_inc, self.nps_min, self.nps_max)

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([3.0, 0.2, 0.002])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

        if self.nps_control_follower:
            state_OS = np.append(state_OS, [self._get_v_desired() - self.OS._get_V(),
                                            self.OS.nps / 3.0])

        # ------------------------- local path information ---------------------------
        state_path = np.array([self.loc_ye/self.OS.Lpp, self.loc_course_error/math.pi])

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

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_env], dtype=np.float32)

    def _get_v_desired(self):
        return self.desired_V

    def _calculate_reward(self, a):
        self.r = 0.0

    def _done(self):
        """Returns boolean flag whether episode is over."""
        d = False

        # OS hit land
        if self.H <= self.OS.critical_depth:
            d = True

        # OS reaches end of global waypoints
        if any([i >= self.n_wps_glo-1 for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            d = True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            d = True
        
        # viz
        if d:
            self.plotter.dump(val_disturbance=self.val_disturbance)
        return d
