from copy import copy, deepcopy
from typing import List, Union

import torch

from tud_rl.common.nets import MLP, LSTMRecActor
from tud_rl.envs._envs.HHOS_Env import *
from tud_rl.envs._envs.HHOS_Fnc import apf
from tud_rl.envs._envs.HHOS_PathPlanning_Env import HHOS_PathPlanning_Env


class HHOS_PathFollowing_Env(HHOS_Env):
    def __init__(self, 
                 plan_on_river : bool,
                 planner_river_weights : str,
                 planner_opensea_weights : str, 
                 planner_state_design : str,
                 planner_safety_net : bool, 
                 nps_control_follower : bool, 
                 scenario_based : bool, 
                 data : str, 
                 N_TSs_max : int, 
                 N_TSs_random : bool, 
                 w_ye : float, 
                 w_ce : float, 
                 w_coll : float, 
                 w_comf : float,
                 w_speed : float):
        super().__init__(nps_control_follower=nps_control_follower, data=data, scenario_based=scenario_based,\
             N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random,w_ye=w_ye, w_ce=w_ce, w_coll=w_coll, w_comf=w_comf, w_speed=w_speed)

        # whether nps are controlled
        self.nps_control_follower = nps_control_follower

        # checks
        if data == "real":
            assert plan_on_river is None, "In real data, we check whether we are on river or open sea."
            assert planner_river_weights is not None and planner_opensea_weights is not None, \
                "Need a river-planner and an opensea-planner on the real data."
        else:
            assert isinstance(plan_on_river, bool), "Specify whether we are on river or not in artificial scenarios."
            assert not(planner_river_weights is not None and planner_opensea_weights is not None), \
                "Specify a river-planner or an opensea-planner, not both."
        assert planner_state_design in ["conventional", "recursive"], "Unknown state design."
        self.planner_state_design = planner_state_design

        # situation
        self.plan_on_river = plan_on_river

        # construct planner network(s)
        self.planner = dict()

        if planner_river_weights is not None:
            plan_in_size = 3 + 5 * self.N_TSs_max + self.lidar_n_beams

            # init
            if planner_state_design == "recursive":
                self.planner["river"] = LSTMRecActor(action_dim=1, num_obs_OS=3+self.lidar_n_beams)
            else:
                self.planner["river"] = MLP(in_size=plan_in_size, out_size=1, net_struc=[[128, "relu"], [128, "relu"], "tanh"])
            
            # weight loading
            self.planner["river"].load_state_dict(torch.load(planner_river_weights))

        if planner_opensea_weights is not None:
            plan_in_size = 3 + 5 * self.N_TSs_max

            # init
            if planner_state_design == "recursive":
                self.planner["opensea"] = LSTMRecActor(action_dim=1, num_obs_OS=3)
            else:
                self.planner["opensea"] = MLP(in_size=plan_in_size, out_size=1, net_struc=[[128, "relu"], [128, "relu"], "tanh"])
            
            # weight loading
            self.planner["opensea"].load_state_dict(torch.load(planner_opensea_weights))

        # construct planning env
        if len(self.planner.keys()) > 0:
            self.planning_env = HHOS_PathPlanning_Env(plan_on_river=plan_on_river, state_design=planner_state_design, data=data,\
                scenario_based=scenario_based, N_TSs_max=N_TSs_max, N_TSs_random=N_TSs_random, w_ye=0.0, w_ce=0.0, w_coll=0.0,\
                     w_comf=0.0, w_speed=0.0)

            # whether to use a safety-net as backup to guarantee safe plans
            self.planner_safety_net = planner_safety_net  

        # gym inherits
        OS_infos = 7 if self.nps_control_follower else 5
        path_infos = 2
        env_infos = 9
        obs_size = OS_infos + path_infos + env_infos
        act_size = 2 if self.nps_control_follower else 1

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(act_size, -1.0, dtype=np.float32), 
                                       high = np.full(act_size,  1.0, dtype=np.float32))
        # vessel config
        self.desired_V = 3.0
        self.rud_angle_max = dtr(20.0)
        self.rud_angle_inc = dtr(5.0)
        self.nps_inc = 0.25
        self.nps_min = 0.5
        self.nps_max = 5.0
        
        self._max_episode_steps = 500

    def reset(self):
        # the local path equals the first couple of entries of the global path after the super().reset() call
        s = super().reset()

        if len(self.planner.keys()) == 0:
            return s
        
        # prepare planning env
        self.planning_env = self.setup_planning_env(env=self.planning_env, initial=True)
        self.planning_env, _ = self.setup_planning_env(env=self.planning_env, initial=False)
        self.planning_env.step_cnt = 0
        self.planning_env.sim_t = 0.0

        # override the local path to start the control-system from the beginning
        self._update_local_path()

        # update wps and error
        self.OS = self._init_wps(self.OS, "local")
        self._set_cte(path_level="local")
        self._set_ce(path_level="local")

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def setup_planning_env(self, env:HHOS_PathPlanning_Env, initial : bool):
        """Prepares the planning env depending on the current status of the following env. Returns the env if initial, else
        returns env, state."""

        if initial:      
            # number of TS and scene
            env.N_TSs = self.N_TSs
            env.plan_on_river = self.plan_on_river
            if hasattr(self, "scene"):
                env.scene = self.scene

            # global path
            env.GlobalPath = deepcopy(self.GlobalPath)
            if hasattr(self, "RevGlobalPath"):
                env.RevGlobalPath = deepcopy(self.RevGlobalPath)

            # environmental disturbances (although not needed for dynamics, only for depth-checking)
            env.DepthData = deepcopy(self.DepthData)
            env.log_Depth = deepcopy(self.log_Depth)
            env.WindData = deepcopy(self.WindData)
            env.CurrentData = deepcopy(self.CurrentData)
            env.WaveData = deepcopy(self.WaveData)

            # visualization specs
            env.domain_xs = self.domain_xs
            env.domain_ys = self.domain_ys
            env.con_ticks = self.con_ticks
            env.con_ticklabels = self.con_ticklabels
            env.clev = self.clev
            return env
        else:
            # step count
            env.step_cnt = 0

            # OS and TSs
            env.OS = deepcopy(self.OS)
            env.TSs = deepcopy(self.TSs)
            env.H = copy(self.H)

            # guarantee OS moves linearly
            env.OS.nu[1:] = 0.0
            env.OS.rud_angle = 0.0

            # global path error
            env.glo_ye = self.glo_ye
            env.glo_desired_course = self.glo_desired_course
            env.glo_course_error = self.glo_course_error
            env.glo_pi_path = self.glo_pi_path

            # set state in planning env and return it
            env._set_state()
            return env, env.state

    def step(self, a:np.ndarray):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""

        # perform control action
        self._OS_control(a)

        # update agent dynamics
        self.OS._upd_dynamics(V_w=self.V_w, beta_w=self.beta_w, V_c=self.V_c, beta_c=self.beta_c, H=self.H, 
                              beta_wave=self.beta_wave, eta_wave=self.eta_wave, T_0_wave=self.T_0_wave, lambda_wave=self.lambda_wave)

        # real data: check whether we are on river or open sea
        if self.data == "real":
            self.plan_on_river = self._on_river(N0=self.OS.eta[0], E0=self.OS.eta[1])

        # environmental effects
        self._update_disturbances()

        # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
        [TS._upd_dynamics() for TS in self.TSs]

        # check respawn
        self.TSs : List[TargetShip] = [self._handle_respawn(TS) for TS in self.TSs]

        # set the local path
        if self.step_cnt % self.loc_path_upd_freq == 0:
            self._update_local_path()

        # update OS waypoints of global and local path
        self.OS:KVLCC2 = self._init_wps(self.OS, "global")
        self.OS:KVLCC2 = self._init_wps(self.OS, "local")

        # compute new cross-track error and course error (for local and global path)
        self._set_cte(path_level="global")
        self._set_cte(path_level="local")
        self._set_ce(path_level="global")
        self._set_ce(path_level="local")

        # on river: update waypoints for other vessels
        if self.plan_on_river:
            self.TSs = [self._init_wps(TS, path_level="global") for TS in self.TSs]

        # on river: control of target ships
        if self.plan_on_river:
            for TS in self.TSs:
                other_vessels = [self.OS] + [ele for ele in self.TSs if ele is not TS]
                TS.river_control(other_vessels, VFG_K=self.VFG_K)
        else:
            [TS.opensea_control() for TS in self.TSs]

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
        if self.nps_control_follower:
            assert len(a) == 2, "There need to be two actions for the follower with nps-control."
        else:
            assert len(a) == 1, "There needs to be one action for the follower without nps-control."

        # rudder control
        assert -1 <= float(a[0]) <= 1, "Unknown action."
        self.OS.rud_angle = np.clip(self.OS.rud_angle + float(a[0])*self.rud_angle_inc, -self.rud_angle_max, self.rud_angle_max)

        # nps control
        if self.nps_control_follower:
            assert -1 <= float(a[1]) <= 1, "Unknown action."
            self.OS.nps = np.clip(self.OS.nps + float(a[1])*self.nps_inc, self.nps_min, self.nps_max)

    def _hasConflict(self, localPath:Path, TSnavData : Union[List[Path], None], path_from:str):
        """Checks whether a given local path yields a conflict.
        
        Args:
            localPath(Path): path of the OS
            TSnavData(list): contains Path-objects, one for each TS. If None, the planning_env is used to generate this data.
            path_from(str):  either 'global' or 'RL'. If 'global', checks for conflicts in the sense of collisions AND traffic rules. 
                             If 'RL', checks only for collisions.
        Returns:
            bool, conflict (True) or no conflict (False)"""
        assert path_from in ["global", "RL"], "Use either 'global' or 'RL' for conflict checking."
        
        # always check for collisions with land
        for i in range(len(localPath.lat)):
            if self._depth_at_latlon(lat_q=localPath.lat[i], lon_q=localPath.lon[i]) <= self.OS.critical_depth:
                return True

        # interpolate path of OS
        atts_to_interpolate = ["north", "east", "heads", "chis", "vs"]

        for att in atts_to_interpolate:
            angle = True if att in ["heads", "chis"] else False
            localPath.interpolate(attribute=att, n_wps_between=5, angle=angle)

        # create TSnavData if not existent
        if TSnavData is None:
            TSnavData = self._update_local_path_safe(method=None)
        
        # interpolate paths of TS to make sure to detect collisions
        for path in TSnavData:
            for att in atts_to_interpolate:
                angle = True if att in ["heads", "chis"] else False
                path.interpolate(attribute=att, n_wps_between=5, angle=angle)

        #----- check for target ship collisions in both methods -----
        n_TS  = len(TSnavData)
        n_wps = len(TSnavData[0].north)

        for t in range(n_wps):
            for i in range(n_TS):

                # quick access
                N0, E0, head0 = localPath.north[t], localPath.east[t], localPath.heads[t]
                N1, E1, head1 = TSnavData[i].north[t], TSnavData[i].east[t], TSnavData[i].heads[t]

                # compute ship domain
                ang = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0)
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                    ang=ang, OS=None, TS=None)
            
                # check if collision
                ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
                if ED_OS_TS <= D:
                    return True

        #----- check for traffic rules only when the path comes from the global path -----
        if path_from == "global":
            for t in range(n_wps):
                for i in range(n_TS):

                    # quick access
                    N0, E0, head0 = localPath.north[t], localPath.east[t], localPath.heads[t]
                    chi0, v0 = localPath.chis[t], localPath.vs[t]
                    
                    N1, E1, head1 = TSnavData[i].north[t], TSnavData[i].east[t], TSnavData[i].heads[t]
                    chi1, v1 = TSnavData[i].chis[t], TSnavData[i].vs[t]

                    # on river: check whether the global path violates some rules
                    if self.plan_on_river:
                        if self._violates_river_traffic_rules(N0=N0, E0=E0, head0=head0, v0=v0, N1=N1, E1=E1, head1=head1, v1=v1,
                                                              Lpp=self.OS.Lpp):
                            return True

                    # on open sea: activate RL planner as soon as there is an encounter situation
                    else:
                        if self._get_COLREG_situation(N0=N0, E0=E0, head0=head0, chi0=chi0, v0=v0, 
                                                      N1=N1, E1=E1, head1=head1, chi1=chi1, v1=v1) != 5:
                            return True
        return False

    def _update_local_path(self):
        """Updates the local path either by simply using the global path, using the RL-based path planner, or the APF safety net."""
        # default: set part of global path as local path
        super()._update_local_path()

        if len(self.planner.keys()) > 0:

            # check local plan for conflicts (collisions AND traffic rules)
            conflict = self._hasConflict(localPath=copy(self.LocalPath), TSnavData=None, path_from="global")
            if conflict:
                TSnavData:List[Path] = self._update_local_path_safe(method="RL")
                self.planning_method = "RL"

                # if we use safety net, check only for collisions
                if self.planner_safety_net:
                    conflict = self._hasConflict(localPath=copy(self.LocalPath), TSnavData=TSnavData, path_from="RL")
                    if conflict:
                        self._update_local_path_safe(method="APF")
                        self.planning_method = "APF"

    def _update_local_path_safe(self, method:Union[str, None]):
        """Updates the local path based on the RL or the APF planner. Can alternatively be used to solely generate TSNavData.
        
        Args:
            method(str): 'RL', 'APF', or None
        Returns:
            List[Path]: TSNavData"""
        assert method in [None, "RL", "APF"], "Use either 'RL' or 'APF' to safely update the local path. \
            Alternatively, use None to generate TSNavData."

        # prep planning env
        self.planning_env, s = self.setup_planning_env(self.planning_env, initial=False)

        # setup OS wps and speed in RL or APF case
        if method in ["RL", "APF"]:
            ns, es, heads = [self.planning_env.OS.eta[0]], [self.planning_env.OS.eta[1]], [self.planning_env.OS.eta[2]]
            chis, vs = [self.planning_env.OS._get_course()], [self.planning_env.OS._get_V()]

        # TSNavData in RL or None mode
        if method in [None, "RL"]:
            TS_ns      = [[TS.eta[0]] for TS in self.planning_env.TSs]
            TS_es      = [[TS.eta[1]] for TS in self.planning_env.TSs]
            TS_heads   = [[TS.eta[2]] for TS in self.planning_env.TSs]
            TS_chis    = [[TS._get_course()] for TS in self.planning_env.TSs]
            TS_vs      = [[TS._get_V()] for TS in self.planning_env.TSs]

        # river or open sea
        assert isinstance(self.planning_env.plan_on_river, bool), "Check whether we are on river or open sea."

        # setup history if needed
        if self.planner_state_design == "recursive":
            if self.planning_env.plan_on_river:
                state_shape = 3 + 5 * self.N_TSs_max + self.lidar_n_beams
            else:
                state_shape = 3 + 5 * self.N_TSs_max

            s_hist = np.zeros((2, state_shape))  # history length 2
            hist_len = 0

        # planning loop
        for _ in range(self.n_wps_loc-1):

            # planner's move
            if method == "RL":
                
                # recursive state design needs history
                if self.planner_state_design == "recursive":
                    s_tens = torch.tensor(s, dtype=torch.float32).view(1, state_shape)
                    s_hist_tens = torch.tensor(s_hist, dtype=torch.float32).view(1, 2, state_shape) # batch size, history length, state shape
                    hist_len_tens = torch.tensor(hist_len)

                    if self.planning_env.plan_on_river:
                        a = self.planner["river"](s=s_tens, s_hist=s_hist_tens, a_hist=None, hist_len=hist_len_tens)[0]
                    else:
                        a = self.planner["opensea"](s=s_tens, s_hist=s_hist_tens, a_hist=None, hist_len=hist_len_tens)[0]

                # conventional state design based only on current observation
                else:
                    s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    if self.planning_env.plan_on_river:
                        a = self.planner["river"](s)
                    else:
                        a = self.planner["opensea"](s)

            # get APF move, always two components
            elif method == "APF":
                du, dh = self._get_APF_move(self.planning_env)

                # apply heading change
                self.planning_env.OS.eta[2] = angle_to_2pi(self.planning_env.OS.eta[2] + dh)

                # apply longitudinal speed change
                self.planning_env.OS.nu[0] = np.clip(self.planning_env.OS.nu[0] + du, \
                    self.planning_env.surge_min, self.planning_env.surge_max)

                # set nps accordingly
                self.planning_env.OS.nps = self.planning_env.OS._get_nps_from_u(self.planning_env.OS.nu[0])

                # need to define dummy zero-actions since we control the OS outside the planning env
                a = np.array([0.0])

            # TSnavData generation - use dummy zero-actions (although it does not matter anyways, we only extract TS info)
            elif method is None:
                a = np.array([0.0])

            # planning env's reaction
            s2, _, d, _ = self.planning_env.step(a, control_TS=False)

            # update history
            if self.planner_state_design == "recursive":
                if hist_len == 2:
                    s_hist = np.roll(s_hist, shift=-1, axis=0)
                    s_hist[1, :] = s
                else:
                    s_hist[hist_len] = s
                    hist_len += 1

            # s becomes s2
            s = s2

            # store wps and speed
            if method in ["RL", "APF"]:
                ns.append(self.planning_env.OS.eta[0])
                es.append(self.planning_env.OS.eta[1])
                heads.append(self.planning_env.OS.eta[2])
                chis.append(self.planning_env.OS._get_course())
                vs.append(self.planning_env.OS._get_V())

            # TSNavData in RL or None mode
            if method in ["RL", None]:
                for i, TS in enumerate(self.planning_env.TSs):
                    TS_ns[i].append(TS.eta[0])
                    TS_es[i].append(TS.eta[1])
                    TS_heads[i].append(TS.eta[2])
                    TS_chis[i].append(TS._get_course())
                    TS_vs[i].append(TS._get_V())
            if d:
                break

        # update the path in RL or APF mode
        if method in ["RL", "APF"]:
            self.LocalPath = Path(level="local", north=ns, east=es, heads=heads, vs=vs, chis=chis)

        # generate TSnavData object for conflict detection
        if method in ["RL", None]:
            TSnavData = []
            for i in range(len(TS_ns)):
                TSnavData.append(Path(level="local", north=TS_ns[i], east=TS_es[i], heads=TS_heads[i], vs=TS_vs[i], chis=TS_chis[i]))
            return TSnavData

    def _get_APF_move(self, env : HHOS_PathPlanning_Env):
        """Takes an env as input and returns the du, dh suggested by the APF-method."""
        # define goal for attractive forces of APF
        try:
            g_idx = env.OS.glo_wp3_idx + 3
            g_n = env.GlobalPath.north[g_idx]
            g_e = env.GlobalPath.east[g_idx]
        except:
            g_idx = env.OS.glo_wp3_idx
            g_n = env.GlobalPath.north[g_idx]
            g_e = env.GlobalPath.east[g_idx]

        # consider river geometry
        N0, E0, head0 = env.OS.eta
        dists, _, river_n, river_e = env._sense_LiDAR(N0=N0, E0=E0, head0=head0)
        i = np.where(dists != env.lidar_range)

        # computation
        du, dh = apf(OS=env.OS, TSs=env.TSs, G={"x" : g_e, "y" : g_n}, 
                     river_n=river_n[i], river_e=river_e[i],
                     du_clip=env.surge_scale, dh_clip=env.d_head_scale)
        return du, dh

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
        """Computes the desired velocity time at the position of the agent."""
        v1 = self.LocalPath.vs[self.OS.loc_wp1_idx]
        v2 = self.LocalPath.vs[self.OS.loc_wp2_idx]

        ate_12 = ate(N1 = self.OS.loc_wp1_N, 
                     E1 = self.OS.loc_wp1_E, 
                     N2 = self.OS.loc_wp2_N, 
                     E2 = self.OS.loc_wp2_E,
                     NA = self.OS.eta[0], 
                     EA = self.OS.eta[1])
        d = self.LocalPath.wp_dist(wp1_idx=self.OS.loc_wp1_idx, wp2_idx=self.OS.loc_wp2_idx)
        frac = np.clip(ate_12/d, 0.0, 1.0)
        return frac*v2 + (1-frac)*v1

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
        #self.r_comf = -float(a[0])**2
        self.r_comf = 0.0

        # drift-based
        #self.r_comf -= abs(self.OS.nu[1])

        # nps-based
        if self.nps_control_follower:
            self.r_comf -= float(a[1])**2

        # ---------------------------- Speed reward -------------------------
        if self.nps_control_follower:
            self.v_des = self._get_v_desired()
            self.r_speed = max([-(self.v_des-self.OS._get_V())**2, -1.0])

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_comf])
        rews = np.array([self.r_ye, self.r_ce, self.r_comf])

        if self.nps_control_follower:
            weights = np.append(weights, [self.w_speed])
            rews = np.append(rews, [self.r_speed])

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
