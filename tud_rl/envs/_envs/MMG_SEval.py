from tud_rl.envs._envs.MMG_Imazu import *
from tud_rl.envs._envs.VesselFnc import dtr


class MMG_SEval(MMG_Env):
    """Implements 24 systematic one-ship evaluation scenarios."""

    def __init__(self, pdf_traj, eval_situation, state_design, w_dist, w_head, w_coll, w_COLREG, w_comf):

        N_TSs = 1

        super().__init__(N_TSs_max=N_TSs, pdf_traj=pdf_traj, N_TSs_random=False, N_TSs_increasing=False, state_design=state_design,\
             w_dist=w_dist, w_head=w_head, w_coll=w_coll, w_COLREG=w_COLREG, w_comf=w_comf)

        assert eval_situation in range(24), "We consider 24 eval situations."
        self.TS_heading = np.linspace(0, 2*math.pi, num=25, endpoint=False)[1:25][eval_situation]
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
        dists = [get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
            OS=None, TS=None, ang=rad) for rad in rads]

        self.domain_plot_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_plot_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        self.outer_domain_plot_xs = [(dist + self.CR_rec_dist) * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.outer_domain_plot_ys = [(dist + self.CR_rec_dist) * math.cos(rad) for dist, rad in zip(dists, rads)]

        # goal spawn
        self.goal = {"N" : CPA_N + abs(CPA_N - self.OS.eta[0]), "E" : self.OS.eta[1]}
        self.OS_goal_init = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"])
        self.OS_goal_old  = self.OS_goal_init

        # TS spawn
        self._spawn_TS(CPA_N=CPA_N, CPA_E=CPA_E, TCPA=TCPA)

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs_max
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        # trajectory storing
        self.TrajPlotter.reset(OS=self.OS, TSs=self.TSs, N_TSs=self.N_TSs)

        return self.state

    def _handle_respawn(self, TS):
        return TS, False

    def _spawn_TS(self, CPA_N, CPA_E, TCPA):
        """TS should be after 'TCPA' at point (CPA_N, CPA_E).
        Since we have speed and heading, we can uniquely determine origin of the motion."""

        # construct ship with dummy N, E
        TS1 = KVLCC2(N_init   = 0.0, 
                     E_init   = 0.0, 
                     psi_init = self.TS_heading,
                     u_init   = 0.0,
                     v_init   = 0.0,
                     r_init   = 0.0,
                     delta_t  = self.delta_t,
                     N_max    = self.N_max,
                     E_max    = self.E_max,
                     nps      = 1.8)

        # predict converged speed of TS
        TS1.nu[0] = TS1._get_u_from_nps(TS1.nps)

        # backtrace to motion
        TS1.eta[0] = CPA_N - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
        TS1.eta[1] = CPA_E - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

        # setup
        self.TSs = [TS1]
