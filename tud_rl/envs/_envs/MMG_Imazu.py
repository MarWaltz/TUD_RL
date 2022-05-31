from tud_rl.envs._envs.MMG_Env import *
from tud_rl.envs._envs.VesselFnc import dtr


class MMG_Imazu(MMG_Env):
    """Implements the 22 ship encounter situations of Imazu (1987) as detailed in Sawada et al. (2021, JMST)."""

    def __init__(self, plot_traj, situation, state_design):
        
        if situation in range(5):
            N_TSs = 1
        
        elif situation in range(5, 12):
            N_TSs = 2

        elif situation in range(12, 23):
            N_TSs = 3
        
        elif situation == 23:
            N_TSs = 0

        super().__init__(N_TSs_max=N_TSs, plot_traj=plot_traj, N_TSs_random=False, N_TSs_increasing=False, state_design=state_design)

        self.situation = situation
        self.N_TSs = N_TSs

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)
        TCPA = 25 * 60              # time to CPA [s]

        CPA_N = NM_to_meter(7.0)
        CPA_E = NM_to_meter(7.0)

        #--------------------------- OS spawn --------------------------------
        head = 0.0

        # init agent (OS for 'Own Ship') with dummy N, E
        self.OS = KVLCC2(N_init   = 0.0, 
                         E_init   = 0.0, 
                         psi_init = head,
                         u_init   = 0.0,
                         v_init   = 0.0,
                         r_init   = 0.0,
                         delta_t  = self.delta_t,
                         N_max    = self.N_max,
                         E_max    = self.E_max,
                         nps      = 1.8)

        # set longitudinal speed to near-convergence
        self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps)

        # backtrace motion
        self.OS.eta[0] = CPA_N - self.OS._get_V() * np.cos(head) * TCPA
        self.OS.eta[1] = CPA_E - self.OS._get_V() * np.sin(head) * TCPA

        # initially compute ship domain for plotting
        rads  = np.linspace(0.0, 2*math.pi, 100)
        dists = [self._get_ship_domain(OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_plot_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_plot_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        #--------------------------- Goal spawn --------------------------------
        self.goal = {"N" : CPA_N + abs(CPA_N - self.OS.eta[0]), "E" : self.OS.eta[1]}
        self.OS_goal_init = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"])
        self.OS_goal_old  = self.OS_goal_init

        #--------------------------- TS spawn --------------------------------
        
        # TS should be after 'TCPA' at point (0,0)
        # since we have speed and heading, we can uniquely determine origin of the motion

        # construct ship with dummy N, E, heading
        TS1 = KVLCC2(N_init   = 0.0, 
                     E_init   = 0.0, 
                     psi_init = 0.0,
                     u_init   = 0.0,
                     v_init   = 0.0,
                     r_init   = 0.0,
                     delta_t  = self.delta_t,
                     N_max    = self.N_max,
                     E_max    = self.E_max,
                     nps      = 1.8)

        if self.situation in range(5):

            # lower speed for overtaking situations
            if self.situation == 3:
                TS1.nps = 0.7

            # predict converged speed of TS
            TS1.nu[0] = TS1._get_u_from_nps(TS1.nps)

            # heading according to situation
            if self.situation == 1:
                headTS1 = dtr(180)
            
            elif self.situation == 2:
                headTS1 = dtr(270)

            elif self.situation == 3:
                headTS1 = 0.0
            
            elif self.situation == 4:
                headTS1 = dtr(45)

            # backtrace to motion
            TS1.eta[2] = headTS1
            TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
            TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

            # setup
            self.TSs = [TS1]

        elif self.situation in range(5, 12):

            # set TS2
            TS2 = copy.deepcopy(TS1)

            # lower speed for overtaking situations
            if self.situation == 7:
                TS1.nps = 0.7

            # predict converged speed of TS
            TS1.nu[0] = TS1._get_u_from_nps(TS1.nps)
            TS2.nu[0] = TS2._get_u_from_nps(TS2.nps)

            # heading according to situation
            if self.situation == 5:
                headTS1 = dtr(180)
                headTS2 = angle_to_2pi(dtr(-90))
            
            elif self.situation == 6:
                headTS1 = angle_to_2pi(dtr(-10))
                headTS2 = angle_to_2pi(dtr(-45))

            elif self.situation == 7:
                headTS1 = 0.0
                headTS2 = angle_to_2pi(dtr(-45))
            
            elif self.situation == 8:
                headTS1 = dtr(180)
                headTS2 = angle_to_2pi(dtr(-90))

            elif self.situation == 9:
                headTS1 = angle_to_2pi(dtr(-30))
                headTS2 = angle_to_2pi(dtr(-90))

            elif self.situation == 10:
                headTS1 = angle_to_2pi(dtr(-90))
                headTS2 = dtr(15)

            elif self.situation == 11:
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

        elif self.situation in range(12, 23):

            # set TS2, TS3
            TS2 = copy.deepcopy(TS1)
            TS3 = copy.deepcopy(TS1)

            # lower speed for overtaking situations
            if self.situation in [15, 17, 20, 22]:
                TS1.nps = 0.7

            # predict converged speed of TS
            TS1.nu[0] = TS1._get_u_from_nps(TS1.nps)
            TS2.nu[0] = TS2._get_u_from_nps(TS2.nps)
            TS3.nu[0] = TS3._get_u_from_nps(TS3.nps)

            # heading according to situation
            if self.situation == 12:
                headTS1 = dtr(180)
                headTS2 = angle_to_2pi(dtr(-45))
                headTS3 = angle_to_2pi(dtr(-10))

            elif self.situation == 13:
                headTS1 = dtr(180)
                headTS2 = dtr(10)
                headTS3 = dtr(45)
            
            elif self.situation == 14:
                headTS1 = angle_to_2pi(dtr(-10))
                headTS2 = angle_to_2pi(dtr(-45))
                headTS3 = angle_to_2pi(dtr(-90))

            elif self.situation == 15:
                headTS1 = 0.0
                headTS2 = angle_to_2pi(dtr(-45))
                headTS3 = angle_to_2pi(dtr(-90))
            
            elif self.situation == 16:
                headTS1 = dtr(45)
                headTS2 = dtr(90)
                headTS3 = angle_to_2pi(dtr(-90))

            elif self.situation == 17:
                headTS1 = 0.0
                headTS2 = dtr(10)
                headTS3 = angle_to_2pi(dtr(-45))

            elif self.situation == 18:
                headTS1 = angle_to_2pi(dtr(-135))
                headTS2 = angle_to_2pi(dtr(-15))
                headTS3 = angle_to_2pi(dtr(-30))

            elif self.situation == 19:
                headTS1 = dtr(15)
                headTS2 = angle_to_2pi(dtr(-15))
                headTS3 = angle_to_2pi(dtr(-135))

            elif self.situation == 20:
                headTS1 = 0.0
                headTS2 = angle_to_2pi(dtr(-15))
                headTS3 = angle_to_2pi(dtr(-90))

            elif self.situation == 21:
                headTS1 = angle_to_2pi(dtr(-15))
                headTS2 = dtr(15)
                headTS3 = angle_to_2pi(dtr(-90))

            elif self.situation == 22:
                headTS1 = 0.0
                headTS2 = angle_to_2pi(dtr(-45))
                headTS3 = angle_to_2pi(dtr(-90))

            # backtrace to motion
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

        elif self.situation == 23:
            self.TSs = []

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs_max
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        # trajectory storing
        if self.plot_traj:
            self.OS_traj_rud_angle = [self.OS.rud_angle]

            self.OS_traj_N = [self.OS.eta[0]]
            self.OS_traj_E = [self.OS.eta[1]]
            self.OS_traj_h = [self.OS.eta[2]]

            self.OS_col_N = []
            self.OS_col_E = []

            self.TS_traj_N = [[] for _ in range(self.N_TSs)]
            self.TS_traj_E = [[] for _ in range(self.N_TSs)]
            self.TS_traj_h = [[] for _ in range(self.N_TSs)]

            self.TS_spawn_steps = [[self.step_cnt] for _ in range(self.N_TSs)]
 
            for TS_idx, TS in enumerate(self.TSs):             
                self.TS_traj_N[TS_idx].append(TS.eta[0])
                self.TS_traj_E[TS_idx].append(TS.eta[1])
                self.TS_traj_h[TS_idx].append(TS.eta[2])

        return self.state

    def _handle_respawn(self, TS):
        return TS, False
