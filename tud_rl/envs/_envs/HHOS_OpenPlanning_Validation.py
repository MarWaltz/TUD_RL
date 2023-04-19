from copy import deepcopy

from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter
from tud_rl.envs._envs.HHOS_OpenPlanning_Env import *


class HHOS_OpenPlanning_Validation(HHOS_OpenPlanning_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self,
                 scenario : int,
                 star_formation : bool = False,
                 star_N_TSs : int = 0,
                 clock_formation : bool = False,
                 clock_scenario : int = 1,
                 full_RL : bool = False,
                 APF_TS : bool = False):
        self.scenario        = scenario
        self.star_formation  = star_formation
        self.star_N_TSs      = star_N_TSs
        self.clock_formation = clock_formation
        self.clock_scenario  = clock_scenario
        self.full_RL         = full_RL
        self.APF_TS          = APF_TS

        if self.full_RL:
            self.history_length = 2
        
        assert not (full_RL and APF_TS), "Either RL- or APF-control for target ships, not both."
        self.TCPA_gap = 25 * 60 # [s]

        # Star
        if self.star_formation:
            self.N_TSs = self.star_N_TSs

        # Around the Clock
        elif self.clock_formation:
            self.N_TSs = 1

        # Imazu
        else:
            assert self.scenario in range(1, 23), "Unknown validation scenario for open sea."
            if self.scenario in range(1, 5):
                self.N_TSs = 1
            
            elif self.scenario in range(5, 12):
                self.N_TSs = 2

            elif self.scenario in range(12, 23):
                self.N_TSs = 3
            
            elif self.scenario == 23:
                self.N_TSs = 0

        super().__init__(N_TSs_max=self.N_TSs, N_TSs_random=False, 
                         w_ye=.0, w_ce=.0, w_coll=.0, w_rule=.0, w_comf=.0)
        
        # always specify a straight path for the agent
        self.path_config = {"n_seg_path" : 13, "straight_wp_dist" : 50, "straight_lmin" : 2000, "straight_lmax" : 2000, 
                            "phi_min" : None, "phi_max" : None, "rad_min" : None, "rad_max" : None, "build" : "straight"}

        self._max_episode_steps = 200

    def reset(self):
        s = super().reset(set_state=False if self.full_RL else True)

        # create straight paths for TSs in all-RL Imazu cases or when APF-control is used
        if self.full_RL or self.APF_TS:
            for TS in self.TSs:
                n, e = [TS.eta[0]], [TS.eta[1]]
                for _ in range(1, 500):
                    e_add, n_add = xy_from_polar(r=100, angle=TS.eta[2])
                    n.append(n[-1] + n_add)
                    e.append(e[-1] + e_add)
                TS.path = Path(level="global", north=n, east=e)

            # now set the state in full-RL mode
            if self.full_RL:
                self._set_state()
                s = self.state

        # viz
        TS_info = {}
        for i, TS in enumerate(self.TSs):
            TS_info[f"TS{str(i)}_N"] = TS.eta[0]
            TS_info[f"TS{str(i)}_E"] = TS.eta[1]
            TS_info[f"TS{str(i)}_head"] = TS.eta[2]
            TS_info[f"TS{str(i)}_V"] = TS._get_V()

        self.plotter = HHOSPlotter(sim_t=self.sim_t, a=0.0, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_V=self.OS._get_V(), OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s

    def _set_state(self):
        if self.full_RL:
            
            # since we use LSTMRecTD3, we need history from the perspective of each TS as well
            if self.step_cnt == 0:
                self.TS_s_hist = np.zeros((self.history_length, self.N_TSs, self.obs_size))
                self.TS_a_hist = np.zeros((self.history_length, self.N_TSs, 1))
                self.hist_len  = 0
                #self.TS_state  = np.zeros((self.N_TSs, self.obs_size))
            else:
                # update history, where most recent state component is the old state from last step
                if self.hist_len == self.history_length:
                    self.TS_s_hist = np.roll(self.TS_s_hist, shift=-1, axis=0)
                    self.TS_s_hist[self.history_length - 1] = self.TS_state
                else:
                    self.TS_s_hist[self.hist_len] = self.TS_state
                    self.hist_len += 1

            # overwrite old state
            self.TS_state = self._get_TS_state()
        
        # always usual state computation
        super()._set_state()

    def _get_TS_state(self):
        """Computes the state from the perspective of each TS. Returns np.array([N_TSs, obs_size])."""
        # copy things
        OS_orig         = deepcopy(self.OS)
        TSs_orig        = deepcopy(self.TSs)
        GlobalPath_orig = deepcopy(self.GlobalPath)

        # prep output
        s_out = np.zeros((self.N_TSs, self.obs_size), dtype=np.float32)

        for i, TS in enumerate(self.TSs):

            # make TS the OS and vice versa
            OS_cpy = deepcopy(OS_orig)
            TS_cpy = deepcopy(TS)

            self.OS     = TS_cpy
            self.TSs[i] = OS_cpy

            # update global path and cte
            self.GlobalPath = TS_cpy.path

            # update new OS waypoints of global path
            self.OS:KVLCC2= self._init_wps(self.OS, "global")

            # compute new cross-track error and course error for global path
            self._set_cte(path_level="global")
            self._set_ce(path_level="global")

            # compute state form TS perspective
            super()._set_state()
            s_out[i] = self.state

            # make sure everything is as before
            self.OS         = deepcopy(OS_orig)
            self.TSs        = deepcopy(TSs_orig)
            self.GlobalPath = deepcopy(GlobalPath_orig)
            self.OS = self._init_wps(self.OS, "global")
            self._set_cte(path_level="global")
            self._set_ce(path_level="global")
        return s_out

    def step(self, a):
        """If we are in full_RL scenario, a is a list of [OS_action, RL_agent], otherwise just the OS_action in form of a np.array."""
        if self.full_RL:
            # unpack
            a, agent = a[0], a[1]

            # TS control
            for i, TS in enumerate(self.TSs):
                a_TS = agent.select_action(s        = self.TS_state[i], 
                                           s_hist   = self.TS_s_hist[:, i, :], 
                                           a_hist   = self.TS_a_hist[:, i, :], 
                                           hist_len = self.hist_len)
                TS.eta[2] = angle_to_2pi(TS.eta[2] + a_TS*self.d_head_scale)

        s, r, d, info = super().step(a, control_TS=True)

        # viz
        if not d:
            TS_info = {}
            for i, TS in enumerate(self.TSs):
                TS_info[f"TS{str(i)}_N"] = TS.eta[0]
                TS_info[f"TS{str(i)}_E"] = TS.eta[1]
                TS_info[f"TS{str(i)}_head"] = TS.eta[2]
                TS_info[f"TS{str(i)}_V"] = TS._get_V()

            self.plotter.store(sim_t=self.sim_t, a=float(self.a), OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_V=self.OS._get_V(), OS_u=self.OS.nu[0],\
                    OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s, r, d, info

    def _handle_respawn(self, TS: TargetShip):
        return TS

    def _init_TSs(self):
        # determine spawning origin
        self.CPA_N = self.OS.eta[0] + self.OS._get_V() * np.cos(self.OS.eta[2]) * self.TCPA_gap
        self.CPA_E = self.OS.eta[1] + self.OS._get_V() * np.sin(self.OS.eta[2]) * self.TCPA_gap

        # create the TSs
        self._spawn_TS(CPA_N=self.CPA_N, CPA_E=self.CPA_E, TCPA=self.TCPA_gap)
        
        # deterministic behavior in evaluation
        for TS in self.TSs:
            TS.non_cooperative = False

            if self.APF_TS:
                TS.APF_moves = True

    def _spawn_TS(self, CPA_N, CPA_E, TCPA):
        """TS should be after 'TCPA' at point (CPA_N, CPA_E).
        Since we have speed and heading, we can uniquely determine origin of the motion."""

        # construct ship with dummy N, E, heading
        TS1 = TargetShip(N_init   = 0.0, 
                         E_init   = 0.0, 
                         psi_init = 0.0,
                         u_init   = 3.0,
                         v_init   = 0.0,
                         r_init   = 0.0,
                         delta_t  = self.delta_t,
                         N_max    = np.infty,
                         E_max    = np.infty,
                         nps      = None,
                         full_ship = False,
                         ship_domain_size = 2)

        if self.star_formation:
            # predict converged speed of first TS
            TS1.nps = TS1._get_nps_from_u(TS1.nu[0])

            # create other TSs
            self.TSs = [deepcopy(TS1) for _ in range(self.N_TSs)]

            for i, TS in enumerate(self.TSs):
                # set heading
                TS.eta[2] = angle_to_2pi(((i+1) * 2*math.pi/(self.N_TSs+1)))

                # backtrace motion
                TS.eta[0] = CPA_N - TS._get_V() * np.cos(TS.eta[2]) * TCPA
                TS.eta[1] = CPA_E - TS._get_V() * np.sin(TS.eta[2]) * TCPA
        
        elif self.clock_formation:
            # predict converged speed of first TS
            TS1.nps = TS1._get_nps_from_u(TS1.nu[0])
            
            # init TS-list
            self.TSs = [TS1]

            # set heading
            TS1.eta[2] = np.linspace(0, 2*math.pi, num=25, endpoint=False)[1:25][self.clock_scenario]

            # backtrace motion
            TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
            TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

        else:
            if self.scenario in range(1, 5):

                # lower speed for overtaking situations
                if self.scenario == 3:
                    TS1.nu[0] = 1.5

                # predict converged speed of TS
                TS1.nps = TS1._get_nps_from_u(TS1.nu[0])

                # heading according to situation
                if self.scenario == 1:
                    headTS1 = dtr(180)
                
                elif self.scenario == 2:
                    headTS1 = dtr(270)

                elif self.scenario == 3:
                    headTS1 = 0.0
                
                elif self.scenario == 4:
                    headTS1 = dtr(45)

                # backtrace to motion
                TS1.eta[2] = headTS1
                TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
                TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

                # setup
                self.TSs = [TS1]

            elif self.scenario in range(5, 12):

                # set TS2
                TS2 = deepcopy(TS1)

                # lower speed for overtaking situations
                if self.scenario == 7:
                    TS1.nu[0] = 1.5

                # predict converged speed of TS
                TS1.nps = TS1._get_nps_from_u(TS1.nu[0])
                TS2.nps = TS2._get_nps_from_u(TS2.nu[0])

                # heading according to situation
                if self.scenario == 5:
                    headTS1 = dtr(180)
                    headTS2 = angle_to_2pi(dtr(-90))
                
                elif self.scenario == 6:
                    headTS1 = angle_to_2pi(dtr(-10))
                    headTS2 = angle_to_2pi(dtr(-45))

                elif self.scenario == 7:
                    headTS1 = 0.0
                    headTS2 = angle_to_2pi(dtr(-45))
                
                elif self.scenario == 8:
                    headTS1 = dtr(180)
                    headTS2 = angle_to_2pi(dtr(-90))

                elif self.scenario == 9:
                    headTS1 = angle_to_2pi(dtr(-30))
                    headTS2 = angle_to_2pi(dtr(-90))

                elif self.scenario == 10:
                    headTS1 = angle_to_2pi(dtr(-90))
                    headTS2 = dtr(15)

                elif self.scenario == 11:
                    headTS1 = dtr(90)
                    headTS2 = angle_to_2pi(dtr(-30))

                # backtrace to motion
                TS1.eta[2] = headTS1
                TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
                TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

                TS2.eta[2] = headTS2
                TS2.eta[0] = CPA_N - TS2._get_V() * np.cos(TS2.eta[2]) * TCPA
                TS2.eta[1] = CPA_E - TS2._get_V() * np.sin(TS2.eta[2]) * TCPA

                # setup
                self.TSs = [TS1, TS2]

            elif self.scenario in range(12, 23):

                # set TS2, TS3
                TS2 = deepcopy(TS1)
                TS3 = deepcopy(TS1)

                # lower speed for overtaking situations
                if self.scenario in [15, 17, 20, 22]:
                    TS1.nu[0] = 1.5

                # predict converged speed of TS
                TS1.nps = TS1._get_nps_from_u(TS1.nu[0])
                TS2.nps = TS2._get_nps_from_u(TS2.nu[0])
                TS3.nps = TS3._get_nps_from_u(TS3.nu[0])

                # heading according to situation
                if self.scenario == 12:
                    headTS1 = dtr(180)
                    headTS2 = angle_to_2pi(dtr(-45))
                    headTS3 = angle_to_2pi(dtr(-10))

                elif self.scenario == 13:
                    headTS1 = dtr(180)
                    headTS2 = dtr(10)
                    headTS3 = dtr(45)
                
                elif self.scenario == 14:
                    headTS1 = angle_to_2pi(dtr(-10))
                    headTS2 = angle_to_2pi(dtr(-45))
                    headTS3 = angle_to_2pi(dtr(-90))

                elif self.scenario == 15:
                    headTS1 = 0.0
                    headTS2 = angle_to_2pi(dtr(-45))
                    headTS3 = angle_to_2pi(dtr(-90))
                
                elif self.scenario == 16:
                    headTS1 = dtr(45)
                    headTS2 = dtr(90)
                    headTS3 = angle_to_2pi(dtr(-90))

                elif self.scenario == 17:
                    headTS1 = 0.0
                    headTS2 = dtr(10)
                    headTS3 = angle_to_2pi(dtr(-45))

                elif self.scenario == 18:
                    headTS1 = angle_to_2pi(dtr(-135))
                    headTS2 = angle_to_2pi(dtr(-15))
                    headTS3 = angle_to_2pi(dtr(-30))

                elif self.scenario == 19:
                    headTS1 = dtr(15)
                    headTS2 = angle_to_2pi(dtr(-15))
                    headTS3 = angle_to_2pi(dtr(-135))

                elif self.scenario == 20:
                    headTS1 = 0.0
                    headTS2 = angle_to_2pi(dtr(-15))
                    headTS3 = angle_to_2pi(dtr(-90))

                elif self.scenario == 21:
                    headTS1 = angle_to_2pi(dtr(-15))
                    headTS2 = dtr(15)
                    headTS3 = angle_to_2pi(dtr(-90))

                elif self.scenario == 22:
                    headTS1 = 0.0
                    headTS2 = angle_to_2pi(dtr(-45))
                    headTS3 = angle_to_2pi(dtr(-90))

                # backtrace to motion origin
                TS1.eta[2] = headTS1
                TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
                TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

                TS2.eta[2] = headTS2
                TS2.eta[0] = CPA_N - TS2._get_V() * np.cos(TS2.eta[2]) * TCPA
                TS2.eta[1] = CPA_E - TS2._get_V() * np.sin(TS2.eta[2]) * TCPA

                TS3.eta[2] = headTS3
                TS3.eta[0] = CPA_N - TS3._get_V() * np.cos(TS3.eta[2]) * TCPA
                TS3.eta[1] = CPA_E - TS3._get_V() * np.sin(TS3.eta[2]) * TCPA

                # setup
                self.TSs = [TS1, TS2, TS3]

            elif self.scenario == 23:
                self.TSs = []

    def _done(self):
        d = super()._done()

        # viz
        if d:
            if self.star_formation:
                self.plotter.dump(name="Plan_Star_" + str(self.star_N_TSs))
            
            elif self.clock_formation:
                self.plotter.dump(name="Plan_Clock_" + str(self.clock_scenario))
            
            else:
                self.plotter.dump(name="Plan_Imazu_" + str(self.scenario))
        return d

    #def render(self, data=None):
    #    pass
