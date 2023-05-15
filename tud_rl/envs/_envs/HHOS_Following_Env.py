from tud_rl.envs._envs.HHOS_Base_Env import *


class HHOS_Following_Env(HHOS_Base_Env):
    def __init__(self, 
                 w_ye : float, 
                 w_ce : float, 
                 w_comf : float):
        super().__init__()

        # sample new depth data only every 5 episodes since this is computationally demanding
        self.n_resets = 0

        # weights
        self.w_ye   = w_ye
        self.w_ce   = w_ce
        self.w_comf = w_comf

        # gym inherits
        OS_infos   = 5
        path_infos = 2
        env_infos  = 9
        obs_size   = OS_infos + path_infos + env_infos
        act_size   = 1

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                       high = np.full(act_size,  1.0, dtype=np.float32))
        # vessel config
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)

        # vector field guidance
        self.VFG_K = 0.01

        # global path characteristics
        self.path_config = {"n_seg_path" : 5, "straight_wp_dist" : 50, "straight_lmin" :400, "straight_lmax" :2000, 
                            "phi_min" : 60, "phi_max" : 100, "rad_min" : 1000, "rad_max" : 5000, "build" : "random"}

        # local path characteristics
        self.n_wps_loc = 50
        self.loc_path_upd_freq = 24 # results in a new local path every 2mins with delta t being 5s

        # depth configuration
        self.depth_config = {"offset" : 100, "noise" : True}

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.05

        # episode length
        self._max_episode_steps = 500

    def _sample_wind_data_follower(self):
        """Generates random wind data."""
        self.WindData = {}
        self.WindData["lat"] = copy(self.DepthData["lat"])
        self.WindData["lon"] = copy(self.DepthData["lon"])

        # sample constants
        speed_mps = np.ones_like(self.DepthData["data"]) * np.random.uniform(low=0.0, high=15.0)
        angle = np.ones_like(self.DepthData["data"]) * np.random.uniform(low=0.0, high=2*math.pi)

        # add noise
        speed_mps += np.random.normal(loc=0.0, scale=1.0, size=speed_mps.shape)
        angle += dtr(np.random.normal(loc=0.0, scale=5.0, size=angle.shape))

        # make sure to stay in right domain
        speed_mps = np.clip(speed_mps, a_min=0.0, a_max=15.0)
        angle = angle % (2*np.pi)

        # overwrite other entries
        e, n = xy_from_polar(r=speed_mps, angle=(angle-np.pi) % (2*np.pi))

        self.WindData["speed_mps"]       = speed_mps
        self.WindData["angle"]           = angle
        self.WindData["eastward_mps"]    = e
        self.WindData["northward_mps"]   = n
        self.WindData["eastward_knots"]  = mps_to_knots(self.WindData["eastward_mps"])
        self.WindData["northward_knots"] = mps_to_knots(self.WindData["northward_mps"])

    def _sample_current_data_follower(self):
        """Generates random current data."""
        self.CurrentData = {}
        self.CurrentData["lat"] = copy(self.DepthData["lat"])
        self.CurrentData["lon"] = copy(self.DepthData["lon"])

        # sample constants
        speed_mps = np.ones_like(self.DepthData["data"]) * np.random.exponential(scale=0.2)
        angle = np.ones_like(self.DepthData["data"]) * np.random.uniform(low=0.0, high=2*math.pi)

        # add noise
        speed_mps += np.random.normal(loc=0.0, scale=0.05, size=speed_mps.shape)
        angle += dtr(np.random.normal(loc=0.0, scale=5.0, size=angle.shape))

        # make sure to stay in right domain
        speed_mps = np.clip(speed_mps, a_min=0.0, a_max=0.5)
        angle = angle % (2*np.pi)

        # overwrite other entries
        e, n = xy_from_polar(r=speed_mps, angle=(angle-np.pi) % (2*np.pi))

        self.CurrentData["speed_mps"]     = speed_mps
        self.CurrentData["angle"]         = angle
        self.CurrentData["eastward_mps"]  = e
        self.CurrentData["northward_mps"] = n

    def _sample_wave_data_follower(self):
        """Generates random wave data."""
        self.WaveData = {}
        self.WaveData["lat"] = copy(self.DepthData["lat"])
        self.WaveData["lon"] = copy(self.DepthData["lon"])

        # sample constants
        height = np.ones_like(self.DepthData["data"]) * np.random.exponential(scale=0.1)
        length = np.ones_like(self.DepthData["data"]) * np.random.exponential(scale=20.0)
        period = np.ones_like(self.DepthData["data"]) * np.random.exponential(scale=1.0)
        angle  = np.ones_like(self.DepthData["data"]) * np.random.uniform(low=0.0, high=2*math.pi)

        # add noise
        height += np.random.normal(loc=0.0, scale=0.05, size=height.shape)
        length += np.random.normal(loc=0.0, scale=1.0, size=length.shape)
        period += np.random.normal(loc=0.0, scale=0.5, size=period.shape)
        angle += dtr(np.random.normal(loc=0.0, scale=5.0, size=angle.shape))

        # make sure to stay in right domain
        height = np.clip(height, a_min=0.01, a_max=2.0)
        length = np.clip(length, a_min=1.0,  a_max=100.0)
        period = np.clip(period, a_min=0.5,  a_max=7.0)
        angle = angle % (2*np.pi)

        # overwrite other entries
        e, n = xy_from_polar(r=height, angle=(angle-np.pi) % (2*np.pi))

        self.WaveData["height"]    = height
        self.WaveData["length"]    = length
        self.WaveData["period"]    = period
        self.WaveData["angle"]     = angle
        self.WaveData["eastward"]  = e
        self.WaveData["northward"] = n

    def _init_local_path(self):
        """Generates a local path based on the global one."""
        self.LocalPath = self.GlobalPath.construct_local_path(wp_idx = self.OS.glo_wp1_idx, n_wps_loc = self.n_wps_loc)
        self.planning_method = "global"

    def reset(self, OS_wp_idx=20, real_data=False):
        # the local path equals the first couple of entries of the global path after the super().reset() call
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # generate global path
        if real_data:
            self._load_global_path()
        else:
            if self.n_resets % 5 == 0:
                self._sample_global_path(**self.path_config)

        # init OS
        lat_init = self.GlobalPath.lat[OS_wp_idx]
        lon_init = self.GlobalPath.lon[OS_wp_idx]
        N_init = self.GlobalPath.north[OS_wp_idx]
        E_init = self.GlobalPath.east[OS_wp_idx]

        # consider different speeds in training
        if "Validation" in type(self).__name__:
            spd = self.base_speed
        else:
            spd = float(np.random.uniform(0.8, 1.2)) * self.base_speed

        self.OS = KVLCC2(N_init    = N_init, 
                         E_init    = E_init, 
                         psi_init  = None,
                         u_init    = spd,
                         v_init    = 0.0,
                         r_init    = 0.0,
                         delta_t   = self.delta_t,
                         N_max     = np.infty,
                         E_max     = np.infty,
                         nps       = None,
                         full_ship = False,
                         ship_domain_size = 2)

        # init waypoints and cte of OS for global path
        self.OS = self._init_wps(self.OS, "global")
        self._set_cte(path_level="global")

        # init local path
        self._init_local_path()

        # init waypoints and cte of OS for local path
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")

        # set heading with noise in training
        if "Validation" in type(self).__name__:
            self.OS.eta[2] = self.loc_pi_path
        else:
            self.OS.eta[2] = angle_to_2pi(self.loc_pi_path + dtr(np.random.uniform(-25.0, 25.0)))

        # generate environmental data
        finish = False
        while not finish:
            if real_data:
                self._load_depth_data()
                self._load_wind_data()
                self._load_current_data()
                self._load_wave_data()
            else:
                if self.n_resets % 5 == 0:
                    self._sample_river_depth_data(**self.depth_config)

                self._sample_wind_data_follower()
                self._sample_current_data_follower()
                self._sample_wave_data_follower()

            # environmental effects
            self._update_disturbances(lat_init, lon_init)

            # set nps to near-convergence
            try:
                self.OS.nps = self.OS._get_nps_from_u(u           = self.OS.nu[0], 
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
                finish = True
            except:
                finish = False

        # update environmental disturbances sampling counter
        if not real_data:
            self.n_resets += 1

        # set course error
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # init state
        self._set_state()
        return self.state

    def step(self, a:np.ndarray):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        # perform control action
        self._OS_control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # environmental effects
        self._update_disturbances()

        # set the local path
        if self.step_cnt % self.loc_path_upd_freq == 0:
            self._init_local_path()

        # update OS waypoints of global and local path
        self.OS:KVLCC2 = self._init_wps(self.OS, "global")
        self.OS:KVLCC2 = self._init_wps(self.OS, "local")

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
        return self.state, self.r, d, {}

    def _OS_control(self, a:np.ndarray):
        """Performs the control action for the own ship."""

        # store for viz
        a = a.flatten()
        self.a = a

        # make sure array has correct size
        assert len(a) == 1, "There needs to be one action for the follower without nps-control."

        # rudder control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.rud_angle = np.clip(self.OS.rud_angle + float(a[0])*self.rud_angle_inc, -self.rud_angle_max, self.rud_angle_max)

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([3.0, 0.2, 0.002])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

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

        head0 = self.OS.eta[2]
        state_env = np.array([self.V_c/0.5,  angle_to_pi(self.beta_c-head0)/(math.pi),      # currents
                              self.V_w/15.0, angle_to_pi(self.beta_w-head0)/(math.pi),      # winds
                              angle_to_pi(beta_wave-head0)/(math.pi), eta_wave/2.0, T_0_wave/7.0, lambda_wave/100.0,    # waves
                              self.H/100.0])    # depth

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_env]).astype(np.float32)

    def _calculate_reward(self, a):
        # ----------------------- LocalPath-following reward --------------------
        # cross-track error
        k_ye = 0.05
        self.r_ye = math.exp(-k_ye * abs(self.loc_ye))

        # course error
        k_ce = 5.0
        if abs(rtd(self.loc_course_error)) >= 90.0:
            self.r_ce = -10.0
        else:
            self.r_ce = math.exp(-k_ce * abs(self.loc_course_error))

        # -------------------------- Comfort reward -------------------------
        # steering-based
        self.r_comf = -float(a[0])**2

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_comf])
        rews = np.array([self.r_ye, self.r_ce, self.r_comf])
        self.r = float(np.sum(weights * rews) / np.sum(weights)) if np.sum(weights) != 0.0 else 0.0

    def _done(self):
        """Returns boolean flag whether episode is over."""
        # OS is too far away from local path
        if abs(self.loc_ye) > 400:
            return True

        # OS hit land
        elif self.H <= self.OS.critical_depth:
            return True

        # OS reaches end of global waypoints
        if any([i >= int(0.8*self.GlobalPath.n_wps) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
