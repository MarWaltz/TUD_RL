from tud_rl.envs._envs.HHOS_Env import *


class HHOS_PathFollowing_Env(HHOS_Env):
    def __init__(self, data="sampled", w_ye=0.5, w_ce=0.5, w_comf=0.05):
        super().__init__(data=data, w_ye=w_ye, w_ce=w_ce, w_comf=w_comf, N_TSs_max=0, N_TSs_random=False, w_coll=0.0)

        # gym inherits
        path_info_size = 16
        obs_size = path_info_size

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        self._max_episode_steps = 5_000
    
    def _add_local_path(self):
        self.LocalPath = copy.deepcopy(self.GlobalPath)

    def _update_local_path(self):
        pass

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([7.0, 0.7, 0.004])                # u, v, r
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

        state_env = np.array([self.V_c/0.5,  self.beta_c/(2*math.pi),      # currents
                              self.V_w/15.0, self.beta_w/(2*math.pi),      # winds
                              beta_wave/(2*math.pi), eta_wave/0.5, T_0_wave/7.0, lambda_wave/60.0,    # waves
                              self.H/100.0])    # depth

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_env], dtype=np.float32)


    def _calculate_reward(self, a):
        # ----------------------- LocalPath-following reward --------------------
        # cross-track error
        k_ye = 0.05
        self.r_ye = math.exp(-k_ye * abs(self.loc_ye))

        # course error
        k_ce = 5.0
        if abs(rtd(self.loc_course_error)) >= 90.0:
            self.r_ce = -10
        else:
            self.r_ce = math.exp(-k_ce * abs(self.loc_course_error))

        # -------------------------- Comfort reward -------------------------
        # steering-based
        self.r_comf = -a**4

        # drift-based
        self.r_comf -= abs(self.OS.nu[1])

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_comf])
        rews = np.array([self.r_ye, self.r_ce, self.r_comf])
        self.r = np.sum(weights * rews) / np.sum(weights)

    def _done(self):
        """Returns boolean flag whether episode is over."""
        # OS is too far away from local path
        if abs(self.loc_ye) > 400:
            return True

        # OS hit land
        elif self.H <= self.OS.critical_depth:
            return True

        # OS reaches end of global waypoints
        if any([i >= int(0.9*self.n_wps) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
