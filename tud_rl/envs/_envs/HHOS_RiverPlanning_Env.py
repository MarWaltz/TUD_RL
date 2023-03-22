import random
from copy import deepcopy
from typing import List

from tud_rl.envs._envs.HHOS_Base_Env import *
from tud_rl.envs._envs.HHOS_Fnc import ate, cte
from tud_rl.envs._envs.VesselFnc import bng_abs, cpa


class HHOS_RiverPlanning_Env(HHOS_Base_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self,
                 N_TSs_max : int, 
                 N_TSs_random : bool, 
                 w_ye : float, 
                 w_ce : float, 
                 w_coll : float, 
                 w_rule : float,
                 w_comf : float):
        super().__init__()
        
        # sample new depth data only every 5 episodes since this is computationally demanding
        self.n_resets = 0

        # time horizon
        self.act_every = 20.0 # [s], every how many seconds the planner can make a move
        self.n_loops   = int(self.act_every / self.delta_t)

        # other ships
        self.N_TSs_max    = N_TSs_max       # maximum number of other vessels
        self.N_TSs_random = N_TSs_random    # if true, samples a random number in [0, N_TSs] at start of each episode
                                            # if false, always have N_TSs_max
        # vector field guidance
        self.VFG_K    = 0.01
        self.VFG_K_TS = 0.001

        # path characteristics
        self.path_config = {"n_seg_path" : 4, "straight_wp_dist" : 50, "straight_lmin" :400, "straight_lmax" :2000, 
                            "phi_min" : 60, "phi_max" : 100, "rad_min" : 1000, "rad_max" : 5000, "build" : "random"}
        self.dist_des_rev_path = 200

        # depth configuration
        self.depth_config = {"validation" : False}

        # weights
        self.w_ye   = w_ye
        self.w_ce   = w_ce
        self.w_coll = w_coll
        self.w_rule = w_rule
        self.w_comf = w_comf

        # gym inherits
        self.num_obs_TS = 7
        self.obs_size = 4 + self.num_obs_TS * max([self.N_TSs_max, 1]) + self.lidar_n_beams

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(1, -1.0, dtype=np.float32), 
                                        high = np.full(1,  1.0, dtype=np.float32))

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.05

        # control scale and episode length
        self.d_head_scale = dtr(10.0)
        self._max_episode_steps = 150

    def reset(self, OS_wp_idx=30, set_state=True):
        """Resets environment to initial state."""
        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # generate global path
        if self.n_resets % 5 == 0:
            self._sample_global_path(**self.path_config)

        # init OS
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
        self.OS.rev_dir = False

        # add reversed global path for TS spawning on rivers
        self.RevGlobalPath = deepcopy(self.GlobalPath)
        self.RevGlobalPath.reverse(offset=self.dist_des_rev_path)

        # init waypoints and cte of OS for global path
        self.OS = self._init_wps(self.OS, "global")
        self._set_cte(path_level="global")

        # set heading with noise in training
        if "Validation" in type(self).__name__:
            self.OS.eta[2] = self.glo_pi_path
        else:
            self.OS.eta[2] = angle_to_2pi(self.glo_pi_path + dtr(np.random.uniform(-25.0, 25.0)))

        # generate random environmental data
        if self.n_resets % 5 == 0:
            self._sample_river_depth_data(**self.depth_config)

        # depth updating
        self._update_disturbances()

        # set nps to near-convergence
        self.OS.nps = self.OS._get_nps_from_u(u = self.OS.nu[0], psi = self.OS.eta[2])

        # set course error
        self._set_ce(path_level="global")

        # init other vessels
        self._init_TSs()

        # init state
        if set_state:
            self._set_state()
        else:
            self.state = None

        # viz
        if hasattr(self, "plotter"):
            self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                    OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], loc_ye=self.loc_ye, glo_ye=self.glo_ye, loc_course_error=self.loc_course_error,\
                        glo_course_error=self.glo_course_error, V_c=self.V_c, beta_c=self.beta_c, V_w=self.V_w, beta_w=self.beta_w,\
                        T_0_wave=self.T_0_wave, eta_wave=self.eta_wave, beta_wave=self.beta_wave, lambda_wave=self.lambda_wave,\
                                rud_angle=self.OS.rud_angle, nps=self.OS.nps)
        return self.state

    def step(self, a, control_TS=True):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        # control action
        self._manual_control(a)

        # update agent dynamics (independent of environmental disturbances in this module)
        [self.OS._upd_dynamics() for _ in range(self.n_loops)]

        # update environmental effects since we check in the reward function whether water depth is insufficient
        self._update_disturbances()

        # update OS waypoints of global path
        self.OS:KVLCC2= self._init_wps(self.OS, "global")

        # compute new cross-track error and course error for global path
        self._set_cte(path_level="global")
        self._set_ce(path_level="global")

        for _ in range(self.n_loops):
            # behavior of target ships
            if control_TS:
                for i, TS in enumerate(self.TSs):
                    # update waypoints
                    try:
                        self.TSs[i] = self._init_wps(TS, "global")
                        cnt = True
                    except:
                        cnt = False

                    # simple heading control
                    if cnt:
                        other_vessels = [ele for ele in self.TSs if ele is not TS] #+ [self.OS]
                        TS.river_control(other_vessels, VFG_K=self.VFG_K_TS)

            # update TS dynamics (independent of environmental disturbances since they move linear and deterministic)
            [TS._upd_dynamics() for TS in self.TSs]

            # check respawn
            self.TSs = [self._handle_respawn(TS) for TS in self.TSs]

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += (self.n_loops * self.delta_t)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(self.a)
        d = self._done()
        return self.state, self.r, d, {}

    def _init_TSs(self):
        # scenario = 0 means all TS random, no manual configuration
        if self.N_TSs_random:
            assert self.N_TSs_max == 3, "Go for maximum 3 TSs in HHOS planning."
            self.N_TSs = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.3, 0.3])
        else:
            self.N_TSs = self.N_TSs_max

        # sample TSs
        self.TSs : List[TargetShip]= []
        for n in range(self.N_TSs):
            TS = None
            while TS is None:
                try:
                    TS = self._get_TS_river(scenario=0, n=n)
                except:
                    pass
            self.TSs.append(TS)

    def _get_TS_river(self, scenario, n=None):
        """Places a target ship by setting a 
            1) traveling direction,
            2) distance on the global path,
        depending on the scenario. All ships spawn in front of the agent.
        Args:
            scenario (int):  considered scenario
            n (int):      index of the spawned vessel
        Returns: 
            KVLCC2."""
        assert not (scenario != 0 and n is None), "Need to provide index in non-random scenario-based spawning."

        #------------------ set distances, directions, offsets from path, and nps ----------------------
        # Note: An offset is some float. If it is negative (positive), the vessel is placed on the 
        #       right (left) side of the global path.

        # random
        if scenario == 0:
            speedy = bool(np.random.choice([0, 1], p=[0.8, 0.2]))
            d      = self.river_enc_range_max + np.random.normal(loc=0.0, scale=20.0)

            if speedy: 
                rev_dir = False
                spd     = np.random.uniform(1.3, 1.5) * self.base_speed
            else:
                rev_dir = bool(random.getrandbits(1))
                spd     = np.random.uniform(0.4, 0.8) * self.base_speed
            offset = np.random.uniform(-20.0, 50.0)

        # vessel train
        if scenario == 1:
            if n == 0:
                d = NM_to_meter(0.5)
            else:
                d = NM_to_meter(0.5) + n*NM_to_meter(0.1)
            rev_dir = False
            offset = 0.0
            spd    = 0.5 * self.base_speed
            speedy = False

        # overtake the overtaker
        elif scenario == 2:
            rev_dir = False
            if n == 0:
                offset = 0.0
                spd = 0.35 * self.base_speed
                d = NM_to_meter(0.5)
            else:
                offset = 0.0
                spd = 0.55 * self.base_speed
                d = NM_to_meter(0.3)
            speedy = False

        # overtaking under oncoming traffic
        elif scenario == 3:
            if n == 0:
                d = NM_to_meter(0.5)
                rev_dir = False
                offset = 0.0
                spd = 0.4 * self.base_speed

            elif n == 1:
                d = NM_to_meter(1.5)
                rev_dir = True
                offset = 0.0
                spd = 0.7 * self.base_speed

            elif n == 2:
                d = NM_to_meter(1.3)
                rev_dir = True
                offset = 0.0
                spd = 0.4 * self.base_speed

            speedy = False

        # overtake the overtaker under oncoming traffic
        elif scenario == 4:
            if n == 0:
                d = NM_to_meter(0.5)
                rev_dir = False
                offset = 0.0
                spd = 0.4 * self.base_speed

            elif n == 1:
                d = NM_to_meter(1.5)
                rev_dir = True
                offset = 0.0
                spd = 0.7 * self.base_speed

            elif n == 2:
                d = NM_to_meter(1.3)
                rev_dir = True
                offset = 0.0
                spd = 0.4 * self.base_speed

            elif n == 3:
                offset = 0.0
                rev_dir = False
                spd = 0.55 * self.base_speed
                d = NM_to_meter(0.3)
            speedy = False

        # get wps
        if speedy:
            wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = self.RevGlobalPath.north, 
                                                                   e_array = self.RevGlobalPath.east, 
                                                                   a_n     = self.OS.eta[0], 
                                                                   a_e     = self.OS.eta[1])
            path = self.RevGlobalPath
        else:
            wp1 = self.OS.glo_wp1_idx
            wp1_N = self.OS.glo_wp1_N
            wp1_E = self.OS.glo_wp1_E

            wp2 = self.OS.glo_wp2_idx
            wp2_N = self.OS.glo_wp2_N
            wp2_E = self.OS.glo_wp2_E

            path = self.GlobalPath

        # determine starting position
        ate_init = ate(N1=wp1_N, E1=wp1_E, N2=wp2_N, E2=wp2_E, NA=self.OS.eta[0], EA=self.OS.eta[1])
        d_to_nxt_wp = path.wp_dist(wp1, wp2) - ate_init
        orig_seg = True

        while True:
            if d > d_to_nxt_wp:
                d -= d_to_nxt_wp
                wp1 += 1
                wp2 += 1
                d_to_nxt_wp = path.wp_dist(wp1, wp2)
                orig_seg = False
            else:
                break

        # path angle
        pi_path_spwn = bng_abs(N0=path.north[wp1], E0=path.east[wp1], N1=path.north[wp2], E1=path.east[wp2])

        # still in original segment
        if orig_seg:
            E_add, N_add = xy_from_polar(r=ate_init+d, angle=pi_path_spwn)
        else:
            E_add, N_add = xy_from_polar(r=d, angle=pi_path_spwn)

        # determine position
        N_TS = path.north[wp1] + N_add
        E_TS = path.east[wp1] + E_add
        
        # jump on the other path: either due to speedy or opposing traffic
        if speedy or rev_dir:
            E_add_rev, N_add_rev = xy_from_polar(r=self.dist_des_rev_path, angle=angle_to_2pi(pi_path_spwn-math.pi/2))
            N_TS += N_add_rev
            E_TS += E_add_rev

        # consider offset
        TS_head = angle_to_2pi(pi_path_spwn + math.pi) if rev_dir or speedy else pi_path_spwn

        if offset != 0.0:
            ang = TS_head - math.pi/2 if offset > 0.0 else TS_head + math.pi/2
            E_add_rev, N_add_rev = xy_from_polar(r=abs(offset), angle=angle_to_2pi(ang))
            N_TS += N_add_rev
            E_TS += E_add_rev

        # generate TS
        TS = TargetShip(N_init    = N_TS,
                        E_init    = E_TS,
                        psi_init  = TS_head,
                        u_init    = spd,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = None,
                        full_ship = False,
                        ship_domain_size = 2)

        # store waypoint information
        TS.rev_dir = rev_dir

        if speedy:
            wp1, wp1_N, wp1_E, wp2, wp2_N, wp2_E = get_init_two_wp(n_array = self.GlobalPath.north, 
                                                                   e_array = self.GlobalPath.east, 
                                                                   a_n     = TS.eta[0], 
                                                                   a_e     = TS.eta[1])
            TS.glo_wp1_idx = wp1
            TS.glo_wp1_N = wp1_N
            TS.glo_wp1_E = wp1_E
            TS.glo_wp2_idx = wp2
            TS.glo_wp2_N = wp2_N
            TS.glo_wp2_E = wp2_E
            TS.glo_wp3_idx = wp2 + 1
            TS.glo_wp3_N = self.GlobalPath.north[wp2 + 1]
            TS.glo_wp3_E = self.GlobalPath.east[wp2 + 1]
        else:
            if rev_dir:
                TS.glo_wp1_idx, TS.glo_wp2_idx = self.GlobalPath.get_rev_path_wps(wp1, wp2)
                TS.glo_wp3_idx = TS.glo_wp2_idx + 1
                path = self.RevGlobalPath
            else:
                TS.glo_wp1_idx, TS.glo_wp2_idx, TS.glo_wp3_idx = wp1, wp2, wp2 + 1
                path = self.GlobalPath

            TS.glo_wp1_N, TS.glo_wp1_E = path.north[TS.glo_wp1_idx], path.east[TS.glo_wp1_idx]
            TS.glo_wp2_N, TS.glo_wp2_E = path.north[TS.glo_wp2_idx], path.east[TS.glo_wp2_idx]
            TS.glo_wp3_N, TS.glo_wp3_E = path.north[TS.glo_wp3_idx], path.east[TS.glo_wp3_idx]

        # predict converged speed of sampled TS
        TS.nps = TS._get_nps_from_u(TS.nu[0], psi=TS.eta[2])
        return TS

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
        state_OS = np.array([self.OS.nu[0]/3.0, 
                             angle_to_pi(self.OS.eta[2] - self.glo_pi_path)/math.pi,
                             angle_to_pi(self.glo_course_error)/math.pi])

        # ------------------------- path information ---------------------------
        state_path = np.array([self.glo_ye/self.OS.Lpp])

        # ----------------------- TS information ------------------------------
        # parametrization
        sight     = self.sight_river         # [m]
        tcpa_norm = 5 * 60                   # [s]
        dcpa_norm = self.river_enc_range_min # [m]
        v_norm    = 3                        # [m/s]

        N0, E0, head0 = self.OS.eta
        v0 = self.OS._get_V()
        chi0 = self.OS._get_course()
        state_TSs = []

        for TS in self.TSs:
            N1, E1, head1 = TS.eta
            v1 = TS._get_V()
            chi1 = TS._get_course()

            # check whether TS is in sight
            dist = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
            if dist <= sight:

                # distance
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,
                                    OS=self.OS, TS=TS)
                dist = (dist - D)/sight

                # relative bearing
                bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N1, E1=E1, head0=head0, to_2pi=False) / (math.pi)

                # heading intersection angle with path
                C_TS_path = angle_to_pi(head1 - self.glo_pi_path) / math.pi

                # speed
                v_rel = (v1-v0)/v_norm

                # encounter situation
                TS_encounter = -1.0 if (abs(head_inter(head_OS=head0, head_TS=head1, to_2pi=False)) >= 90.0) else 1.0

                # collision risk metrics
                d_cpa, t_cpa, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, chiOS=chi0,
                                                                        chiTS=chi1, VOS=v0, VTS=v1, get_positions=True)
                ang = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
                domain_tcpa = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C,
                                            D=self.OS.ship_domain_D, OS=None, TS=None, ang=ang)
                d_cpa = max([0.0, d_cpa-domain_tcpa])

                t_cpa = t_cpa/tcpa_norm
                d_cpa = d_cpa/dcpa_norm
                
                # store it
                state_TSs.append([dist, bng_rel_TS, C_TS_path, v_rel, TS_encounter, t_cpa, d_cpa])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
        if len(state_TSs) == 0:
            state_TSs.append([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        # sort according to distance (descending, smaller distance is more dangerous)
        state_TSs = np.array(sorted(state_TSs, key=lambda x: x[0], reverse=True)).flatten()

        # at least one since there is always the ghost ship
        desired_length = self.num_obs_TS * max([self.N_TSs_max, 1])

        state_TSs = np.pad(state_TSs, (0, desired_length - len(state_TSs)), \
            'constant', constant_values=np.nan).astype(np.float32)

        # ----------------------- LiDAR for depth -----------------------------
        state_LiDAR = self._get_closeness_from_lidar(self._sense_LiDAR(N0=N0, E0=E0, head0=head0, check_lane_river=True)[0])

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_LiDAR, state_TSs]).astype(np.float32)

    def _calculate_reward(self, a):
        # parametrization
        sight             =  self.sight_river
        k_ye              =  2.0
        k_ce              =  4.0
        ye_norm           =  2*self.OS.Lpp
        pen_coll_depth    = -10.0
        pen_coll_TS       = -10.0
        pen_traffic_rules = -2.0
        dx_norm           =  (3*self.OS.B)**2
        dy_norm           =  (1*self.OS.Lpp)**2

        # --------------- Collision Avoidance & Traffic rule reward -------------
        self.r_coll = 0
        self.r_rule = 0.0

        # hit ground or cross lane on river
        if self.H <= self.OS.critical_depth:
            self.r_coll += pen_coll_depth
        
        # compute CTE to reversed lane
        path = self.RevGlobalPath
        NA, EA, _ = self.OS.eta
        _, wp1_N, wp1_E, _, wp2_N, wp2_E = get_init_two_wp(n_array=path.north, e_array=path.east, a_n=NA, a_e=EA)

        # switch wps since the path is reversed
        if cte(N1=wp2_N, E1=wp2_E, N2=wp1_N, E2=wp1_E, NA=NA, EA=EA) < 0:
            self.r_coll += pen_coll_depth
        
        # maximum of Gaussian collision rewards
        r_Gaussian_colls = [0.0]

        for TS in self.TSs:

            # quick access
            N0, E0, head0 = self.OS.eta
            N1, E1, head1 = TS.eta
            
            # check whether TS is in sight
            dist = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
            if dist <= sight:

                # compute ship domain
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D, OS=self.OS, TS=TS)
                dist -= D

                # check if collision
                if dist <= 0.0:
                    self.r_coll += pen_coll_TS
                else:
                    # relative bng from TS perspective
                    bng_rel_TS = bng_rel(N0=N1, E0=E1, N1=N0, E1=E0, head0=head1)
                    dx, dy = xy_from_polar(r=dist, angle=bng_rel_TS)
                    r_Gaussian_colls.append(-math.exp(-(dx)**2/dx_norm) * math.exp(-(dy)**2/dy_norm))

                # violating traffic rules
                if self._violates_river_traffic_rules(N0=N0, E0=E0, head0=head0, v0=self.OS._get_V(), N1=N1, E1=E1, \
                    head1=head1, v1=TS._get_V()):
                    self.r_rule += pen_traffic_rules
        self.r_coll += min(r_Gaussian_colls)

        # ----------------------- GlobalPath-following reward --------------------
        # cross-track error
        self.r_ye = math.exp(-k_ye * abs(self.glo_ye)/ye_norm)

        # course violation
        self.r_ce = math.exp(-k_ce * abs(angle_to_pi(self.glo_course_error)))

        # ---------------------- Comfort reward -----------------
        self.r_comf = -(float(a)**2)

        # ---------------------------- Aggregation --------------------------
        weights = np.array([self.w_ye, self.w_ce, self.w_coll, self.w_rule, self.w_comf])
        rews    = np.array([self.r_ye, self.r_ce, self.r_coll, self.r_rule, self.r_comf])
        self.r  = float(np.sum(weights * rews) / np.sum(weights)) if np.sum(weights) != 0.0 else 0.0

    def _done(self):
        """Returns boolean flag whether episode is over."""
        # OS approaches end of global path
        if any([i >= int(0.9*self.GlobalPath.n_wps) for i in (self.OS.glo_wp1_idx, self.OS.glo_wp2_idx, self.OS.glo_wp3_idx)]):
            return True

        # artificial done signal
        elif self.step_cnt >= self._max_episode_steps:
            return True

        # don't go too far away from path
        if abs(self.glo_ye) >= NM_to_meter(0.5):
            return True
        return False
