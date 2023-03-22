from typing import List

from tud_rl.envs._envs.HHOS_Base_Env import *
from tud_rl.envs._envs.HHOS_Fnc import ate
from tud_rl.envs._envs.VesselFnc import bng_abs, cpa


class HHOS_OpenPlanning_Env(HHOS_Base_Env):
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

        # time horizon
        self.act_every = 20.0 # [s], every how many seconds the planner can make a move
        self.n_loops   = int(self.act_every / self.delta_t)

        # other ships
        self.N_TSs_max    = N_TSs_max       # maximum number of other vessels
        self.N_TSs_random = N_TSs_random    # if true, samples a random number in [0, N_TSs] at start of each episode
                                            # if false, always have N_TSs_max

        # vector field guidance
        self.VFG_K = 0.0005

        # global path characteristics
        self.path_config = {"n_seg_path" : 5, "straight_wp_dist" : 50, "straight_lmin" : 2000, "straight_lmax" :5000, 
                            "phi_min" : 5, "phi_max" : 30, "rad_min" : 3000, "rad_max" : 5000, "build" : "random"}

        # spawn distances target ships
        self.TS_dist_low  = NM_to_meter(1.0)
        self.TS_dist_high = NM_to_meter(3.0)

        # weights
        self.w_ye   = w_ye
        self.w_ce   = w_ce
        self.w_coll = w_coll
        self.w_rule = w_rule
        self.w_comf = w_comf

        # gym inherits
        self.num_obs_TS = 6
        self.obs_size = 4 + self.num_obs_TS * max([self.N_TSs_max, 1])

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(1, -1.0, dtype=np.float32), 
                                        high = np.full(1,  1.0, dtype=np.float32))

        # how many longitude/latitude degrees to show for the visualization
        self.show_lon_lat = 0.20

        # control scale and episode length
        self.d_head_scale = dtr(10.0)
        self._max_episode_steps = 125

    def reset(self, OS_wp_idx=30, set_state=True):
        """Resets environment to initial state."""
        self.step_cnt = 0     # simulation step counter
        self.sim_t    = 0     # overall passed simulation time (in s)

        # generate global path
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

        # init waypoints and cte of OS for global path
        self.OS = self._init_wps(self.OS, "global")
        self._set_cte(path_level="global")
        self.glo_ye_old = self.glo_ye

        # set heading with noise in training
        if "Validation" in type(self).__name__:
            self.OS.eta[2] = self.glo_pi_path
        else:
            self.OS.eta[2] = angle_to_2pi(self.glo_pi_path + dtr(np.random.uniform(-25.0, 25.0)))

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

    def _init_TSs(self):
        if self.N_TSs_random:
            assert self.N_TSs_max == 3, "Go for maximum 3 TSs in HHOS open planning."
            self.N_TSs = np.random.choice([1, 2, 3])
        else:
            self.N_TSs = self.N_TSs_max

        # sample TSs
        self.TSs : List[TargetShip]= []
        for _ in range(self.N_TSs):
            self.TSs.append(self._get_TS_open_sea())

    def _get_CR_open_sea(self, vessel0:KVLCC2, vessel1:KVLCC2, DCPA_norm:float, TCPA_norm:float, dist_norm:float, dist:float=None):
        """Computes the collision risk with vessel 1 from perspective of vessel 0. Inspired by Waltz & Okhrin (2022)."""
        N0, E0, head0 = vessel0.eta
        N1, E1, head1 = vessel1.eta

        # compute distance under consideration of ship domain
        if dist is None:
            dist = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
            D = get_ship_domain(A=vessel0.ship_domain_A, B=vessel0.ship_domain_B, C=vessel0.ship_domain_C,\
                D=vessel0.ship_domain_D, OS=vessel0, TS=vessel1)
            dist -= D

        if dist <= 0.0:
            return 1.0
        
        # compute CPA measures
        DCPA, TCPA, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, 
                                                                 chiOS=vessel0._get_course(), chiTS=vessel1._get_course(),
                                                                 VOS=vessel0._get_V(), VTS=vessel1._get_V(), get_positions=True)
        # substract ship domain at TCPA = 0 from DCPA
        bng_rel_tcpa_from_OS_pers = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
        domain_tcpa = get_ship_domain(A=vessel0.ship_domain_A, B=vessel0.ship_domain_B, C=vessel0.ship_domain_C,\
            D=vessel0.ship_domain_D, OS=None, TS=None, ang=bng_rel_tcpa_from_OS_pers)
        DCPA = max([0.0, DCPA-domain_tcpa])

        # check whether OS will be in front of TS when TCPA = 0
        bng_rel_tcpa_from_TS_pers = abs(bng_rel(N0=NTS_tcpa, E0=ETS_tcpa, N1=NOS_tcpa, E1=EOS_tcpa, head0=head1, to_2pi=False))

        if TCPA >= 0.0 and bng_rel_tcpa_from_TS_pers <= dtr(30.0):
            DCPA = DCPA * (1.2-math.exp(-math.log(5.0)/dtr(30.0)*bng_rel_tcpa_from_TS_pers))

        # weight positive and negative TCPA differently
        f = 5 if TCPA < 0 else 1
        CR_cpa = math.exp(-DCPA/DCPA_norm) * math.exp(-f * abs(TCPA)/TCPA_norm)

        # euclidean distance
        CR_ed = math.exp(-(dist)**2/dist_norm)
        return np.clip(max([CR_cpa, CR_ed]), 0.0, 1.0)

    def _get_TS_open_sea(self) -> TargetShip:
        """Samples a target ship.
        Returns: 
            TargetShip"""
        TS = TargetShip(N_init    = 0.0,
                        E_init    = 0.0,
                        psi_init  = 0.0,
                        u_init    = float(np.random.uniform(0.8, 1.2)) * self.base_speed,
                        v_init    = 0.0,
                        r_init    = 0.0,
                        delta_t   = self.delta_t,
                        N_max     = np.infty,
                        E_max     = np.infty,
                        nps       = None,
                        full_ship = False,
                        ship_domain_size = 2)

        # predict converged speed of sampled TS
        # Note: if we don't do this, all further calculations are heavily biased
        TS.nps = TS._get_nps_from_u(TS.nu[0], psi=TS.eta[2])

        # quick access for OS
        chiOS = self.OS._get_course()
        VOS   = self.OS._get_V()

        # sample COLREG situation 
        # head-on = 1, starb. cross. = 2, ports. cross. = 3, overtaking = 4, null = 5
        COLREG_s = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.2])

        try:
            # distance to hit point
            d_hit = np.random.uniform(low=self.TS_dist_low, high=self.TS_dist_high)

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

            d = copy(d_hit)
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
            N_hit = path.north[wp1] + N_add
            E_hit = path.east[wp1] + E_add
        except:
            raise Exception()

        bng_abs_hit = pi_path_spwn

        # Note: In the following, we only sample the intersection angle and not a relative bearing.
        #       This is possible since we construct the trajectory of the TS so that it will pass through (E_hit, N_hit), 
        #       and we need only one further information to uniquely determine the origin of its motion.

        # head-on
        if COLREG_s == 1:
            C_TS_s = dtr(np.random.uniform(175, 185))

        # starboard crossing
        elif COLREG_s == 2:
            C_TS_s = dtr(np.random.uniform(185, 292.5))

        # portside crossing
        elif COLREG_s == 3:
            C_TS_s = dtr(np.random.uniform(67.5, 175))

        # overtaking
        elif COLREG_s == 4:
            C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

        # null: TS comes from behind
        elif COLREG_s == 5:
                C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

        # determine TS heading (treating absolute bearing towards goal as heading of OS)
        head_TS_s = angle_to_2pi(C_TS_s + bng_abs_hit)   

        # no speed constraints except in overtaking
        if COLREG_s in [1, 2, 3, 5]:
            VTS = TS.nu[0]

        elif COLREG_s == 4:

            # project VOS vector on TS direction
            VR_TS_x, VR_TS_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=head_TS_s)
            V_max_TS = polar_from_xy(x=VR_TS_x, y=VR_TS_y, with_r=True, with_angle=False)[0]

            # sample TS speed
            VTS = np.random.uniform(0.3, 0.7) * V_max_TS
            TS.nu[0] = VTS

            # set nps of TS so that it will keep this velocity
            TS.nps = TS._get_nps_from_u(VTS, psi=TS.eta[2])

        # backtrace original position of TS
        t_hit = d_hit/self.OS._get_V()
        E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
        N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

        # set positional values
        TS.eta = np.array([N_TS, E_TS, head_TS_s], dtype=np.float32)
        return TS

    def step(self, a, control_TS=True):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        # control action
        self._manual_control(a)

        # update agent dynamics (independent of environmental disturbances in this module)
        [self.OS._upd_dynamics() for _ in range(self.n_loops)]

        # update OS waypoints of global path
        self.OS:KVLCC2= self._init_wps(self.OS, "global")

        # compute new cross-track error and course error for global path
        self._set_cte(path_level="global")
        self._set_ce(path_level="global")

        for _ in range(self.n_loops):
            # behavior of target ships
            if control_TS:
                for TS in self.TSs:
                    other_vessels = [ele for ele in self.TSs if ele is not TS] + [self.OS]
                    TS.opensea_control(other_vessels)

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
        # speed, heading relative to global path, global course error
        state_OS = np.array([self.OS.nu[0]/3.0, 
                             angle_to_pi(self.OS.eta[2] - self.glo_pi_path)/math.pi,
                             angle_to_pi(self.glo_course_error)/math.pi])

        # ------------------------- path information ---------------------------
        state_path = np.array([self.glo_ye/self.OS.Lpp])

        # ----------------------- TS information ------------------------------
        sight     = self.sight_open       # [m]
        tcpa_norm = 15 * 60               # [s]
        dcpa_norm = NM_to_meter(1.0)      # [m]
        dist_norm = (NM_to_meter(0.5))**2 # [m²]
        v_norm    = 3                     # [m/s]

        N0, E0, head0 = self.OS.eta
        v0 = self.OS._get_V()
        state_TSs = []

        for TS in self.TSs:
            N1, E1, head1 = TS.eta
            v1 = TS._get_V()

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
                TS_encounter = self._get_COLREG_situation(N0=N0, E0=E0, head0=head0, v0=v0, chi0=self.OS._get_course(), 
                                                          N1=N1, E1=E1, head1=head1, v1=v1, chi1=TS._get_course())
    
                # collision risk metric
                CR = self._get_CR_open_sea(vessel0=self.OS, vessel1=TS, DCPA_norm=dcpa_norm, TCPA_norm=tcpa_norm, 
                                           dist=dist, dist_norm=dist_norm)
                # store it
                state_TSs.append([dist, bng_rel_TS, C_TS_path, v_rel, TS_encounter, CR])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
        if len(state_TSs) == 0:
            state_TSs.append([1.0, -1.0, 1.0, -1.0, 5.0, 0.0])

        # sort according to CR (ascending, larger d_cpa is more dangerous)
        state_TSs = np.array(sorted(state_TSs, key=lambda x: x[-1])).flatten()

        # at least one since there is always the ghost ship
        desired_length = self.num_obs_TS * max([self.N_TSs_max, 1])

        state_TSs = np.pad(state_TSs, (0, desired_length - len(state_TSs)), \
            'constant', constant_values=np.nan).astype(np.float32)

        # ------------------------- aggregate information ------------------------
        self.state = np.concatenate([state_OS, state_path, state_TSs]).astype(np.float32)

    def _calculate_reward(self, a):
        # parametrization
        sight             =  self.sight_open
        pen_coll_TS       = -10.0
        pen_traffic_rules = -2.0
        delta_ye_norm     =  self.OS.Lpp
        dist_norm         =  (NM_to_meter(0.5))**2
        tcpa_norm = 15 * 60             # [s]
        dcpa_norm = NM_to_meter(1.0)    # [m]

        # --------------- Collision Avoidance & Traffic rule reward -------------
        self.r_coll = 0
        self.r_rule = 0.0

        # being too far away from path on open sea
        if abs(self.glo_ye) >= NM_to_meter(4.0):
            self.r_coll += pen_coll_TS
        
        CR_max = 0.0

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

                # check if collision, use CR metric
                if dist <= 0.0:
                    self.r_coll += pen_coll_TS
                else:
                    CR = self._get_CR_open_sea(vessel0=self.OS, vessel1=TS, DCPA_norm=dcpa_norm, TCPA_norm=tcpa_norm, 
                                               dist=dist, dist_norm=dist_norm)
                    #self.r_coll += -math.sqrt(CR)
                    CR_max = max([CR_max, CR])

                # violating traffic rules, we consider the current action for evaluating COLREG-compliance
                if self._violates_COLREG_rules(N0=N0, E0=E0, head0=head0, chi0=self.OS._get_course(), v0=self.OS._get_V(),
                                               r0=a, N1=N1, E1=E1, head1=head1, chi1=TS._get_course(), v1=TS._get_V()):
                    self.r_rule += pen_traffic_rules

        self.r_coll += -math.sqrt(CR_max)

        # ----------------------- GlobalPath-following reward --------------------
        # cross-track error
        if abs(self.glo_ye) <= 2*self.OS.Lpp:
            self.r_ye = 1.0
        else:
            self.r_ye = np.clip((abs(self.glo_ye_old) - abs(self.glo_ye))/delta_ye_norm, -1.0, 1.0)
        self.glo_ye_old = self.glo_ye

        # course violation
        self.r_ce = 1.0 - abs(angle_to_pi(self.glo_course_error))/math.pi

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
        if abs(self.glo_ye) >= NM_to_meter(6.0):
            return True
        return False
