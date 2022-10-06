import torch
from tud_rl.common.nets import MLP
from tud_rl.envs._envs.HHOS_Env import *
from tud_rl.envs._envs.HHOS_PathPlanning_Env import HHOS_PathPlanning_Env


class HHOS_PathFollowing_Env(HHOS_Env):
    def __init__(self, planner_weights=None, data="sampled", N_TSs_max=0, N_TSs_random=False, w_ye=0.5, w_ce=0.5, w_comf=0.05):
        super().__init__(data=data, w_ye=w_ye, w_ce=w_ce, w_comf=w_comf, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, w_coll=0.0)

        if planner_weights is not None:
            plan_in_size = 3 + self.lidar_n_beams + 5 * self.N_TSs_max
            self.planner = MLP(in_size=plan_in_size, out_size=1, net_struc=[[128, "relu"], [128, "relu"], "tanh"])
            self.planner.load_state_dict(torch.load(planner_weights))
            self.planning_env = HHOS_PathPlanning_Env(data=data, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, w_ye=0.0, w_ce=0.0, w_coll=0.0)
        else:
            self.planner = None

        # gym inherits
        path_info_size = 16
        obs_size = path_info_size

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), 
                                       high=np.array([1], dtype=np.float32))
        self._max_episode_steps = 5_000
    

    def reset(self):
        # the local path equals the first couple of entries of the global paths after the super().reset() call
        s = super().reset()

        if self.planner is None:
            return s
        
        # we override the local path since our planning agent should work from the beginning
        self.planning_env.reset()
        self.setup_planning_env(initial=True)
        self._update_local_path()

        # update wps and error
        self._init_OS_wps(path_level="local")
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
            # global path and its characteristics
            self.planning_env.GlobalPath = copy.deepcopy(self.GlobalPath)
            self.planning_env.RevGlobalPath = copy.deepcopy(self.RevGlobalPath)
            self.planning_env.glo_ye = self.glo_ye
            self.planning_env.glo_desired_course = self.glo_desired_course
            self.planning_env.glo_pi_path = self.glo_pi_path

            # environmental disturbances (although not need for dynamics, only for depth-checking)
            self.planning_env.DepthData = copy.deepcopy(self.DepthData)
            self.planning_env.WindData = copy.deepcopy(self.WindData)
            self.planning_env.CurrentData = copy.deepcopy(self.CurrentData)
            self.planning_env.WaveData = copy.deepcopy(self.WaveData)
        else:
            # OS and TSs
            self.planning_env.OS = copy.deepcopy(self.OS)
            self.planning_env.TSs = copy.deepcopy(self.TSs)
            self.planning_env.H = copy.copy(self.H)

            # set state in planning env and return it
            self.planning_env._set_state()
            return self.planning_env.state


    def _update_local_path(self):
        if self.planner is not None:
            if self.step_cnt % 25 == 0:

                # prep planning env
                s = self.setup_planning_env(initial=False)

                # setup wps
                ns, es = [self.OS.eta[0]], [self.OS.eta[1]]

                # planning loop
                for _ in range(self.n_wps_loc-1):
                    # planner's move
                    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    a = self.planner(s)

                    # planning env's reaction
                    s, _, d, _ = self.planning_env.step(a.item())

                    # store wps
                    ns.append(self.planning_env.OS.eta[0])
                    es.append(self.planning_env.OS.eta[1])

                    if d:
                        break
                
                # set it as local path
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
        if any([i >= int(0.9*self.n_wps_glo) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True
        return False
