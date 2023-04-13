from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter
from tud_rl.envs._envs.HHOS_Following_Env import *


class HHOS_Following_Validation(HHOS_Following_Env):
    def __init__(self, 
                 real_data:bool, 
                 val_disturbance:str,
                 global_path_file:str,
                 depth_data_file:str,
                 current_data_file:str,
                 wind_data_file:str,
                 wave_data_file:str):
        super().__init__(w_ye=0.0, w_ce=0.0, w_comf=0.0)

        assert not(real_data and val_disturbance is not None), "Do not set a validation disturbance if the real data check goes."
        assert val_disturbance in [None, "currents", "winds", "waves"],\
            "Unknown environmental disturbance to validate. Should be 'currents', 'winds', or 'waves'."

        self.real_data         = real_data
        self.val_disturbance   = val_disturbance
        self.global_path_file  = global_path_file
        self.depth_data_file   = depth_data_file
        self.current_data_file = current_data_file
        self.wind_data_file    = wind_data_file
        self.wave_data_file    = wave_data_file

        if real_data:
            self.plot_current = True
            self.plot_wind    = True
            self.plot_waves   = True
        else:
            # specify a straight path for the agent
            self.path_config = {"n_seg_path" : 10, "straight_wp_dist" : 50, "straight_lmin" : 2000, "straight_lmax" : 2000, 
                                "phi_min" : None, "phi_max" : None, "rad_min" : None, "rad_max" : None, "build" : "straight"}
 
            # depth configuration
            self.depth_config = {"validation" : True}

            if val_disturbance == "currents":
                self.plot_current = True
                self.plot_wind    = False
                self.plot_waves   = False
            
            elif val_disturbance == "winds":
                self.plot_current = False
                self.plot_wind    = True
                self.plot_waves   = False

            elif val_disturbance == "waves":
                self.plot_current = False
                self.plot_wind    = False
                self.plot_waves   = True

        if real_data:
            self._max_episode_steps = int(1e7)
        else:
            self._max_episode_steps = 600

    def reset(self):
        s = super().reset(OS_wp_idx=0 if self.real_data else 10, real_data=self.real_data)

        # viz
        self.plotter = HHOSPlotter(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
            OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                    T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                        rud_angle=self.OS.rud_angle, nps=self.OS.nps)
        return s

    def _sample_current_data_follower(self):
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

            # overwrite other entries
            e, n = xy_from_polar(r=speed_mps, angle=(angle-np.pi) % (2*np.pi))

            self.CurrentData["speed_mps"] = speed_mps
            self.CurrentData["angle"] = angle
            self.CurrentData["eastward_mps"] = e
            self.CurrentData["northward_mps"] = n
        else:
            # clear current impact
            super()._sample_current_data_follower()
            for key in self.CurrentData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.CurrentData[key] *= 0.0

    def _sample_wave_data_follower(self):
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
            e, n = xy_from_polar(r=height, angle=(angle-np.pi) % (2*np.pi))
            self.WaveData["eastward"]  = e
            self.WaveData["northward"] = n
        else:
            # clear wave impact
            super()._sample_wave_data_follower()
            for key in self.WaveData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.WaveData[key] *= 0.0

    def _sample_wind_data_follower(self):
        """Generates random wind data."""
        if self.val_disturbance == "winds":

            self.WindData = {}
            self.WindData["lat"] = copy(self.DepthData["lat"])
            self.WindData["lon"] = copy(self.DepthData["lon"])

            speed_mps = np.zeros((len(self.WindData["lat"]), len(self.WindData["lon"])))
            angle = np.zeros_like(speed_mps)
            V_const = 10.0

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

            # overwrite other entries
            e, n = xy_from_polar(r=speed_mps, angle=(angle-np.pi) % (2*np.pi))

            self.WindData["speed_mps"] = speed_mps
            self.WindData["angle"]     = angle
            self.WindData["eastward_mps"]    = e
            self.WindData["northward_mps"]   = n
            self.WindData["eastward_knots"]  = mps_to_knots(self.WindData["eastward_mps"])
            self.WindData["northward_knots"] = mps_to_knots(self.WindData["northward_mps"])
        else:
            # clear wind impact
            super()._sample_wind_data_follower()
            for key in self.WindData.keys():
                if key not in ["lat", "lon", "metaData"]:
                    self.WindData[key] *= 0.0

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        s, r, d, info = super().step(a)

        # viz
        self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                    glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                    T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                         rud_angle=self.OS.rud_angle, nps=self.OS.nps)
        return s, r, d, info

    def _done(self):
        """Returns boolean flag whether episode is over."""
        d = False

        # OS hit land
        if self.H <= self.OS.critical_depth:
            d = True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            d = True
        
        # viz
        if d:
            if self.real_data:
                self.plotter.dump(name="Follow_Real_Data")
            else:
                self.plotter.dump(name="Follow_" + self.val_disturbance)
        return d
