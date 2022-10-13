import torch
from tud_rl.common.nets import MLP
from tud_rl.envs._envs.HHOS_Env import *
from tud_rl.envs._envs.HHOS_PathPlanning_Env import HHOS_PathPlanning_Env


class HHOS_PathFollowing_Env(HHOS_Env):
    def __init__(self, planner_weights, time, scenario_based, data, N_TSs_max, N_TSs_random, w_ye, w_ce, w_coll, w_comf, w_time):
        super().__init__(time=time, data=data, scenario_based=scenario_based, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, \
            w_ye=w_ye, w_ce=w_ce, w_coll=w_coll, w_comf=w_comf, w_time=w_time)

        if planner_weights is not None:
            plan_in_size = 3 + self.lidar_n_beams + 5 * self.N_TSs_max
            self.planner = MLP(in_size=plan_in_size, out_size=1, net_struc=[[128, "relu"], [128, "relu"], "tanh"])
            self.planner.load_state_dict(torch.load(planner_weights))
            self.planning_env = HHOS_PathPlanning_Env(state_design="conventional", time=time, data=data, scenario_based=scenario_based,\
                N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, w_ye=None, w_ce=None, w_coll=None, w_comf=None, w_time=None)
        else:
            self.planner = None

        # gym inherits
        OS_infos = 7 if self.time else 5
        path_infos = 2
        env_infos = 9
        obs_size = OS_infos + path_infos + env_infos

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        
        # vessel config
        self.desired_V = 3.0
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)
        self.nps_inc = 0.25
        self.nps_min = 0.5
        self.nps_max = 5.0
        
        self._max_episode_steps = 100
    

    def reset(self):
        # the local path equals the first couple of entries of the global path after the super().reset() call
        s = super().reset()

        if self.planner is None:
            return s
        
        # prepare planning env
        self.setup_planning_env(initial=True)
        self.setup_planning_env(initial=False)
        self.planning_env.step_cnt = 0
        self.planning_env.sim_t = 0.0

        # override the local path since our planning agent should work from the beginning
        self._update_local_path()

        # update wps and error
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")
        self._set_ce(path_level="local")

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state


    def setup_planning_env(self, initial : bool):
        """Sets the relevant characteristics of the planning env to the status of self, the PathFollowing Env. The planner will execute its actions there, and a 
        set of waypoints constituting the local path is returned. The follower's object is to follow this local path.
        If initial is True, the global path is also set."""

        if initial:      
            # number of TS and scene
            self.planning_env.N_TSs = self.N_TSs
            self.planning_env.scene = self.scene

            # global path
            self.planning_env.GlobalPath = deepcopy(self.GlobalPath)
            self.planning_env.RevGlobalPath = deepcopy(self.RevGlobalPath)

            # environmental disturbances (although not needed for dynamics, only for depth-checking)
            self.planning_env.DepthData = deepcopy(self.DepthData)
            self.planning_env.log_Depth = deepcopy(self.log_Depth)
            self.planning_env.WindData = deepcopy(self.WindData)
            self.planning_env.CurrentData = deepcopy(self.CurrentData)
            self.planning_env.WaveData = deepcopy(self.WaveData)

            # visualization specs
            self.planning_env.domain_xs = self.domain_xs
            self.planning_env.domain_ys = self.domain_ys
            self.planning_env.con_ticks = self.con_ticks
            self.planning_env.con_ticklabels = self.con_ticklabels
            self.planning_env.clev = self.clev
        else:
            # clear time in trajectory tracking
            if self.time:
                self.planning_env.sim_t = 0.0

            # OS and TSs
            self.planning_env.OS = deepcopy(self.OS)
            self.planning_env.TSs = deepcopy(self.TSs)
            self.planning_env.H = copy(self.H)

            # guarantee OS moves linearly
            self.planning_env.OS.nu[2] = 0.0
            self.planning_env.OS.rud_angle = 0.0

            # global path error
            self.planning_env.glo_ye = self.glo_ye
            self.planning_env.glo_desired_course = self.glo_desired_course
            self.planning_env.glo_course_error = self.glo_course_error
            self.planning_env.glo_pi_path = self.glo_pi_path

            # set state in planning env and return it
            self.planning_env._set_state()
            return self.planning_env.state

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        self._OS_control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # environmental effects
        self._update_disturbances()

        # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
        [TS._upd_dynamics() for TS in self.TSs]

        # check respawn
        self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

        # set the local path
        if self.step_cnt % self.loc_path_upd_freq == 0:
            self.sim_t = 0.0
            self._update_local_path()

        # update OS waypoints of global and local path
        self.OS = self._init_wps(self.OS, "global")
        self.OS = self._init_wps(self.OS, "local")

        # compute new cross-track error and course error (for local and global path)
        self._set_cte(path_level="global")
        self._set_cte(path_level="local")
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # update waypoints for other vessels
        self.TSs = [self._init_wps(TS, path_level="global") for TS in self.TSs]

        # control of target ships
        self.TSs = [self._rule_based_control(TS) for TS in self.TSs]

        # increase step cnt and overall simulation time
        if self.step_cnt % self.loc_path_upd_freq != 0:
            self.sim_t += self.delta_t
        self.step_cnt += 1
        
        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}


    def _OS_control(self, a):
        """Performs the control action for the own ship."""
        # rudder control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.rud_angle = np.clip(self.OS.rud_angle + float(a[0])*self.rud_angle_inc, -self.rud_angle_max, self.rud_angle_max)

        # nps control
        if self.time:
            assert -1 <= float(a[1]) <= 1, "Unknown action."
            self.OS.nps = np.clip(self.OS.nps + float(a[1])*self.nps_inc, self.nps_min, self.nps_max)


    def _update_local_path(self):
        if self.planner is not None:
            # prep planning env
            s = self.setup_planning_env(initial=False)

            # setup wps and potentially time
            ns, es = [self.planning_env.OS.eta[0]], [self.planning_env.OS.eta[1]]
            if self.time:
                ts = [self.planning_env.sim_t]

            # planning loop
            for _ in range(self.n_wps_loc-1):
                # planner's move
                s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                a = self.planner(s)

                # planning env's reaction
                s, _, d, _ = self.planning_env.step(a)

                # store wps and potentially time
                ns.append(self.planning_env.OS.eta[0])
                es.append(self.planning_env.OS.eta[1])
                if self.time:
                    ts.append(self.planning_env.sim_t)

                if d:
                    break
            
            # set it as local path
            if self.time:
                self.LocalPath["t"] = ts
            self.LocalPath["north"] = np.array(ns)
            self.LocalPath["east"] = np.array(es)
            self.LocalPath["lat"] = np.zeros_like(self.LocalPath["north"])
            self.LocalPath["lon"] = np.zeros_like(self.LocalPath["north"])

            for i in range(len(ns)):
                self.LocalPath["lat"][i], self.LocalPath["lon"][i] = to_latlon(north=ns[i], east=es[i], number=32)
        else:
            super()._update_local_path()


    def _set_state(self):
        #--------------------------- OS information ----------------------------
        cmp1 = self.OS.nu / np.array([3.0, 0.2, 0.002])                # u, v, r
        cmp2 = np.array([self.OS.nu_dot[2] / (8e-5), self.OS.rud_angle / self.OS.rud_angle_max])   # r_dot, rudder angle
        state_OS = np.concatenate([cmp1, cmp2])

        if self.time:
            state_OS = np.append(state_OS, [(self._get_t_desired() - self.sim_t)/self.delta_t,
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


    def _get_t_desired(self):
        """Computes the desired simulation time at the position of the agent."""
        t1 = self.LocalPath["t"][self.OS.loc_wp1_idx]
        t2 = self.LocalPath["t"][self.OS.loc_wp2_idx]

        ate_12 = ate(N1 = self.OS.loc_wp1_N, 
                     E1 = self.OS.loc_wp1_E, 
                     N2 = self.OS.loc_wp2_N, 
                     E2 = self.OS.loc_wp2_E,
                     NA = self.OS.eta[0], 
                     EA = self.OS.eta[1])
        d = self._wp_dist(wp1_idx=self.OS.loc_wp1_idx, wp2_idx=self.OS.loc_wp2_idx, path=self.LocalPath)
        frac = np.clip(ate_12/d, 0.0, 1.0)
        return frac*t2 + (1-frac)*t1


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
        self.r_comf = -float(a[0])**4

        # drift-based
        self.r_comf -= abs(self.OS.nu[1])

        # nps-based
        if self.time:
            self.r_comf -= 2*float(a[1])**4

        # ---------------------------- Time reward -------------------------
        if self.time:
            self.t_des = self._get_t_desired()
            self.r_time = -abs(self.t_des-self.sim_t)/self.delta_t

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_comf])
        rews = np.array([self.r_ye, self.r_ce, self.r_comf])

        if self.time:
            weights = np.append(weights, [self.w_time])
            rews = np.append(rews, [self.r_time])

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
        if any([i >= int(0.9*self.n_wps_glo) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
