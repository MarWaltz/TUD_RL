from tud_rl.envs._envs.HHOS_Env import *


class HHOS_PathPlanning_Env(HHOS_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self, data="sampled", scenario_based=True, state_design="conventional", N_TSs_max=1, N_TSs_random=False, w_ye=1.0, w_ce=1.0, w_coll=1.0):
        super().__init__(scenario_based=scenario_based, data=data, w_ye=w_ye, w_ce=w_ce, w_coll=w_coll, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, w_comf=0.0)

        assert state_design in ["recursive", "conventional"], "Unknown state design for the HHOS-planner. Should be 'recursive' or 'conventional'."
        self.state_design = state_design

        # forward run
        self.n_loops = int(60.0/self.delta_t)

        # gym inherits
        OS_path_info_size = 3
        self.num_obs_TS = 5
        obs_size = OS_path_info_size + self.lidar_n_beams + self.num_obs_TS * self.N_TSs_max

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        self.d_head_scale = dtr(10.0)
        self._max_episode_steps = 5_000

    def reset(self):
        s = super().reset()

        # we can delete the local path and its characteritics
        del [self.LocalPath, self.loc_ye, self.loc_desired_course, self.loc_pi_path][:]
        return s

    def _update_local_path(self):
        pass

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # control action is heading adjustment
        a = float(a)
        self.OS = self._manual_heading_control(vessel=self.OS, a=a)

        # update agent dynamics (independent of environmental disturbances in this module)
        [self.OS._upd_dynamics() for _ in range(self.n_loops)]

        # environmental effects
        self._update_disturbances()

        # update OS waypoints of global path
        self.OS = self._update_wps(self.OS, path_level="global")

        # compute new cross-track error and course error for global path
        self._set_cte(path_level="global")
        self._set_ce(path_level="global")

        for _ in range(self.n_loops):
            # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
            [TS._upd_dynamics() for TS in self.TSs]

            # check respawn
            self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

            # update waypoints for other vessels
            self.TSs = [self._update_wps(TS, path_level="global") for TS in self.TSs]

            # simple heading control of target ships
            self.TSs = [self._rule_based_control(TS) for TS in self.TSs]

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}

    def _manual_heading_control(self, vessel, a):
        """Adjust the heading of a vessel."""
        assert -1 <= a <= 1, "Unknown action."

        vessel.eta[2] = angle_to_2pi(vessel.eta[2] + a*self.d_head_scale)
        return vessel

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        # speed, heading relative to global path
        state_OS = np.array([self.OS.nu[0]/7.0, angle_to_pi(self.OS.eta[2] - self.glo_pi_path)/math.pi])

        # ------------------------- path information ---------------------------
        state_path = np.array([self.glo_ye/self.OS.Lpp])

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

            # heading intersection angle with path
            C_TS_path = angle_to_pi(headTS - self.glo_pi_path) / math.pi

            # speed
            V_TS = TS._get_V() / 7.0

            # direction
            TS_dir = -1.0 if TS.rev_dir else 1.0

            # store it
            state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS_path, V_TS, TS_dir])          
        
        if self.state_design == "recursive":
            # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
            if len(state_TSs) == 0:
                state_TSs.append([0.0, -1.0, -1.0, 0.0, -1.0])

            # sort according to euclidean distance (descending euclidean distance, smaller is more dangerous)
            state_TSs = np.array(sorted(state_TSs, key=lambda x: x[0], reverse=True)).flatten()

            # at least one since there is always the ghost ship
            desired_length = self.num_obs_TS * max([self.N_TSs_max, 1])  

            state_TSs = np.pad(state_TSs, (0, desired_length - len(state_TSs)), \
                'constant', constant_values=np.nan).astype(np.float32)
        
        else:
            # pad ghost ships
            while len(state_TSs) != self.N_TSs_max:
                state_TSs.append([0.0, -1.0, -1.0, 0.0, -1.0])

            # sort according to euclidean distance (descending euclidean distance, smaller is more dangerous)
            state_TSs = np.array(sorted(state_TSs, key=lambda x: x[0], reverse=True)).flatten().astype(np.float32)

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_LiDAR, state_TSs], dtype=np.float32)


    def _calculate_reward(self, a):
        # ----------------------- GlobalPath-following reward --------------------
        # cross-track error
        k_ye = 0.05
        self.r_ye = math.exp(-k_ye * abs(self.glo_ye))

        # course violation
        if abs(angle_to_pi(self.OS.eta[2] - self.glo_pi_path)) >= math.pi/2:
            self.r_ce = -10.0
        else:
            self.r_ce = 0.0

        # ---------------------- Collision Avoidance reward -----------------
        self.r_coll = 0

        # other vessels
        for TS in self.TSs:
            # compute ship domain
            N0, E0, _ = self.OS.eta
            N1, E1, head1 = TS.eta
            D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D, OS=self.OS, TS=TS)
        
            # check if collision
            ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
            if ED_OS_TS <= D:
                self.r_coll -= 10.0
            else:
                self.r_coll -= math.exp(-(ED_OS_TS-D)/200)

            # violating traffic rules is considered a collision
            bng_rel_TS_pers = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)
            if dtr(90.0) <= bng_rel_TS_pers <= dtr(180.0) and ED_OS_TS <= 5*self.OS.Lpp:
                self.r_coll -= 10.0

        # hit ground
        if self.H <= self.OS.critical_depth:
            self.r_coll -= 10.0

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_coll])
        rews = np.array([self.r_ye, self.r_ce, self.r_coll])
        self.r = np.sum(weights * rews) / np.sum(weights) if np.sum(weights) != 0.0 else 0.0


    def _done(self):
        """Returns boolean flag whether episode is over."""
        # OS is too far away from path
        if abs(self.glo_ye) > 1000:
            return True

        # OS reached final waypoint
        elif any([i >= int(0.9*self.n_wps_glo) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
