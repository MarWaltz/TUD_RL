from tud_rl.envs._envs.HHOS_Fnc import HHOSPlotter
from tud_rl.envs._envs.HHOS_PathPlanning_Env import *


class HHOS_PathPlan_Validation(HHOS_PathPlanning_Env):
    """Does not consider any environmental disturbances since this is considered by the local-path following unit."""
    def __init__(self, 
                 plan_on_river : bool,
                 state_design : str, 
                 data : str, 
                 scenario : bool):
        self.scenario = scenario
        assert data == "sampled", "Planning validation should be on simulated data."

        # scenarios on the river
        if plan_on_river:
            assert self.scenario in range(1, 5), "Unknown validation scenario for the river."

            # vessel train
            if self.scenario == 1:
                self.N_TSs = 4
            
            # overtake the overtaker
            elif self.scenario == 2:
                self.N_TSs = 2

            # overtaking under oncoming traffic
            elif self.scenario == 3:
                self.N_TSs = 5
            
            # overtake the overtaker under oncoming traffic
            elif self.scenario == 4:
                self.N_TSs = 3
        
        # Imazu problems for open sea
        else:
            self.TCPA_gap = 25 * 60 # [s]

            assert self.scenario in range(1, 23), "Unknown validation scenario for open sea."
            if self.scenario in range(1, 5):
                self.N_TSs = 1
            
            elif self.scenario in range(5, 12):
                self.N_TSs = 2

            elif self.scenario in range(12, 23):
                self.N_TSs = 3
            
            elif self.scenario == 23:
                self.N_TSs = 0

        super().__init__(plan_on_river=plan_on_river, state_design=state_design, data=data, N_TSs_max=self.N_TSs,\
            N_TSs_random=False, w_ye=.0, w_ce=.0, w_coll=.0, w_comf=.0, w_speed=.0)
        self._max_episode_steps = 50

    def reset(self):
        s = super().reset()

        # overwrite OS nps since its computation considered environmental disturbances
        self.OS.nps = self.OS._get_nps_from_u(u=self.desired_V)
        
        # viz
        TS_info = {}
        for i, TS in enumerate(self.TSs):
            TS_info[f"TS{str(i)}_N"] = TS.eta[0]
            TS_info[f"TS{str(i)}_E"] = TS.eta[1]
            TS_info[f"TS{str(i)}_head"] = TS.eta[2]

        self.plotter = HHOSPlotter(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s

    def step(self, a):
        # transition
        s, r, d, info = super().step(a, control_TS=True)
        
        # viz
        if not d:
            TS_info = {}
            for i, TS in enumerate(self.TSs):
                TS_info[f"TS{str(i)}_N"] = TS.eta[0]
                TS_info[f"TS{str(i)}_E"] = TS.eta[1]
                TS_info[f"TS{str(i)}_head"] = TS.eta[2]

            self.plotter.store(sim_t=self.sim_t, OS_N=self.OS.eta[0], OS_E=self.OS.eta[1], OS_head=self.OS.eta[2], OS_u=self.OS.nu[0],\
                    OS_v=self.OS.nu[1], OS_r=self.OS.nu[2], glo_ye=self.glo_ye, glo_course_error=self.glo_course_error, **TS_info)
        return s, r, d, info

    def _sample_global_path(self):
        """Constructs a straight path with n_wps way points, each being of length l apart from its neighbor in the lat-lon-system.
        The agent should follows the path always in direction of increasing indices."""
        # set starting point
        path_n = np.zeros(self.n_wps_glo)
        path_e = np.zeros(self.n_wps_glo)
        path_n[0], path_e[0], _ = to_utm(lat=56.0, lon=9.0)

        # sample other points
        for i in range(1, self.n_wps_glo):
            # next point
            e_add, n_add = xy_from_polar(r=self.l_seg_path, angle=0)
            path_n[i] = path_n[i-1] + n_add
            path_e[i] = path_e[i-1] + e_add

        # to latlon
        lat, lon = to_latlon(north=path_n, east=path_e, number=32)

        # store
        self.GlobalPath = Path(level="global", lat=lat, lon=lon, north=path_n, east=path_e)

        # overwrite data range
        self.off = 0.075
        self.lat_lims = [np.min(lat)-self.off, np.max(lat)+self.off]
        self.lon_lims = [np.min(lon)-self.off, np.max(lon)+self.off]

    def _handle_respawn(self, TS: TargetShip):
        return TS

    def _init_TSs(self):
        if self.plan_on_river:
            self.TSs : List[TargetShip]= []
            for n in range(self.N_TSs):
                self.TSs.append(self._get_TS_river(scenario=self.scenario, n=n))
        else:
            # determine spawning origin
            self.CPA_N = self.OS.eta[0] + self.OS._get_V() * np.cos(self.OS.eta[2]) * self.TCPA_gap
            self.CPA_E = self.OS.eta[1] + self.OS._get_V() * np.sin(self.OS.eta[2]) * self.TCPA_gap

            # create the TSs
            self._spawn_TS(CPA_N=self.CPA_N, CPA_E=self.CPA_E, TCPA=self.TCPA_gap)

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

        # deterministic behavior in evaluation
        TS1.random_moves = False

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
            if self.plan_on_river:
                self.plotter.DepthData = self.DepthData
                self.plotter.dump(name="Plan_river_" + str(self.scenario))
            else:
                self.plotter.dump(name="Plan_Imazu_" + str(self.scenario))
        return d

    def render(self, data=None):
        pass
