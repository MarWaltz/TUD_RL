from tud_rl.envs._envs.HHOS_Env import *


class HHOS_PathPlanning_Env(HHOS_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self, 
                 plan_on_river : bool,
                 state_design : str, 
                 data : str, 
                 scenario_based : bool, 
                 N_TSs_max : int, 
                 N_TSs_random : bool, 
                 w_ye : float, 
                 w_ce : float, 
                 w_coll : float, 
                 w_comf : float, 
                 w_speed : float):
        super().__init__(nps_control_follower=None, data=data, scenario_based=scenario_based, w_ye=w_ye, w_ce=w_ce, \
            w_coll=w_coll, w_comf=w_comf, w_speed=w_speed, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random)

        assert state_design in ["recursive", "conventional"], "Unknown state design for the HHOS-planner. Should be 'recursive' or 'conventional'."
        self.state_design = state_design

        # forward run
        self.n_loops = int(60.0/self.delta_t)

        # type of planner
        self.plan_on_river = plan_on_river

        # gym inherits
        if plan_on_river is not None:
            obs_size = 3 + 5 * self.N_TSs_max
            if self.plan_on_river:
                obs_size += self.lidar_n_beams

            self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                                high = np.full(obs_size,  np.inf, dtype=np.float32))
            self.action_space = spaces.Box(low  = np.full(1, -1.0, dtype=np.float32), 
                                        high = np.full(1,  1.0, dtype=np.float32))
        # control scales
        self.surge_scale = 0.5
        self.surge_min = 0.1
        self.surge_max = 5.0
        self.d_head_scale = dtr(10.0)

        self._max_episode_steps = 200

    def reset(self):
        s = super().reset()

        # we can delete the local path and its characteritics
        del self.LocalPath
        del self.loc_ye 
        del self.loc_desired_course
        del self.loc_course_error
        del self.loc_pi_path
        return s

    def _update_local_path(self):
        raise NotImplementedError("Updating the local path should not be called for the path planner.")

    def step(self, a, control_TS=True):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        # control action
        self._manual_control(a)

        # update agent dynamics (independent of environmental disturbances in this module)
        [self.OS._upd_dynamics() for _ in range(self.n_loops)]

        # real data: check whether we are on river or open sea
        if self.data == "real":
            self.plan_on_river = self._on_river(N0=self.OS.eta[0], E0=self.OS.eta[1])

        # environmental effects
        self._update_disturbances()

        # update OS waypoints of global path
        self.OS:KVLCC2= self._init_wps(self.OS, "global")

        # compute new cross-track error and course error for global path
        self._set_cte(path_level="global")
        self._set_ce(path_level="global")

        for _ in range(self.n_loops):
            # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
            [TS._upd_dynamics() for TS in self.TSs]

            # check respawn
            self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

            # on river: update waypoints for other vessels
            if self.plan_on_river:
                self.TSs = [self._init_wps(TS, "global") for TS in self.TSs]

            # on river: simple heading control of target ships
            if control_TS:
                if self.plan_on_river:
                    for TS in self.TSs:
                        other_vessels = [self.OS] + [ele for ele in self.TSs if ele is not TS]
                        TS.river_control(other_vessels, VFG_K=self.VFG_K)
                else:
                    [TS.opensea_control() for TS in self.TSs]

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.n_loops * self.delta_t

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(self.a)
        d = self._done()
        return self.state, self.r, d, {}

    def _manual_control(self, a:np.ndarray):
        """Manually controls heading and surge of the own ship."""
        a = a.flatten()
        self.a = a

        # make sure array has correct size
        assert len(a) == 1, "There needs to be one action for the planner."

        # heading control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] + float(a[0])*self.d_head_scale)

    def _set_state(self):
        #--------------------------- OS information ----------------------------
        # speed, heading relative to global path
        state_OS = np.array([self.OS.nu[0]/3.0, angle_to_pi(self.OS.eta[2] - self.glo_pi_path)/math.pi])

        # ------------------------- path information ---------------------------
        state_path = np.array([self.glo_ye/self.OS.Lpp])

        # ----------------------- TS information ------------------------------
        N0, E0, head0 = self.OS.eta
        v0 = self.OS._get_V()
        state_TSs = []

        for TS in self.TSs:
            N1, E1, head1 = TS.eta
            v1 = TS._get_V()

            # closeness
            D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                OS=self.OS, TS=TS)
            ED_OS_TS = (ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True) - D) / (20*self.OS.Lpp)
            closeness = np.clip(1-ED_OS_TS, 0.0, 1.0)

            # relative bearing
            bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0, to_2pi=False) / (math.pi)

            # heading intersection angle with path
            C_TS_path = angle_to_pi(head1 - self.glo_pi_path) / math.pi

            # speed
            v_rel = v1-v0

            # encounter situation
            if self.plan_on_river:
                TS_encounter = -1.0 if (abs(head_inter(head_OS=head0, head_TS=head1, to_2pi=False)) >= 90.0) else 1.0
            else:
                TS_encounter = self._get_COLREG_situation(N0=N0, E0=E0, head0=head0, v0=v0, chi0=self.OS._get_course(), 
                                                          N1=N1, E1=E1, head1=head1, v1=v1, chi1=TS._get_course())
            # store it
            state_TSs.append([closeness, bng_rel_TS, C_TS_path, v_rel, TS_encounter])

        if self.state_design == "recursive":
            raise NotImplementedError("Recursive state definition not implemented yet.")

            # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
            if len(state_TSs) == 0:
                state_TSs.append([0.0, -1.0, -1.0, 0.0, -1.0, -1.0])

            # sort according to closeness (ascending, larger closeness is more dangerous)
            state_TSs = np.array(sorted(state_TSs, key=lambda x: x[0])).flatten()

            # at least one since there is always the ghost ship
            desired_length = self.num_obs_TS * max([self.N_TSs_max, 1])  

            state_TSs = np.pad(state_TSs, (0, desired_length - len(state_TSs)), \
                'constant', constant_values=np.nan).astype(np.float32)
        else:
            # pad ghost ships
            while len(state_TSs) != self.N_TSs_max:
                enc_pad = -2.0 if self.plan_on_river else 3.0
                state_TSs.append([0.0, -1.0, -1.0, 0.0, enc_pad])

            # sort according to closeness (ascending, larger closeness is more dangerous)
            state_TSs = np.hstack(sorted(state_TSs, key=lambda x: x[0])).astype(np.float32)

        # ----------------------- LiDAR for depth -----------------------------
        if self.plan_on_river:
            N0, E0, head0 = self.OS.eta
            state_LiDAR = self._get_closeness_from_lidar(self._sense_LiDAR(N0=N0, E0=E0, head0=head0)[0])
        else:
            state_LiDAR = np.array([])

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_TSs, state_LiDAR], dtype=np.float32)

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
        # hit ground
        if self.H <= self.OS.critical_depth:
            self.r_coll = -10.0
        else:
            self.r_coll = 0

        # other vessels
        for TS in self.TSs:

            # compute ship domain
            N0, E0, head0 = self.OS.eta
            N1, E1, head1 = TS.eta
            D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D, OS=self.OS, TS=TS)
        
            # check if collision
            ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
            if ED_OS_TS <= D:
                self.r_coll -= 10.0
            else:
                self.r_coll -= math.exp(-(ED_OS_TS-D)/200)

            # violating traffic rules is considered a collision
            if self.plan_on_river:
                if self._violates_river_traffic_rules(N0=N0, E0=E0, head0=head0, v0=self.OS._get_V(), N1=N1, E1=E1, head1=head1,\
                    v1=TS._get_V(), Lpp=self.OS.Lpp):
                    self.r_coll -= 10.0
            else:
                if self._violates_COLREG_rules(N0=N0, E0=E0, head0=head0, chi0=self.OS._get_course(), v0=self.OS._get_V(),\
                    r0=self.OS.nu[2], N1=N1, E1=E1, head1=head1, chi1=TS._get_course(), v1=TS._get_V()):
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

        # OS approaches end of global path
        elif any([i >= int(0.9*self.n_wps_glo) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
