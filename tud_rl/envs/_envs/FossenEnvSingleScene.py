from .FossenEnv import *
from .FossenFnc import dtr


class FossenEnvSingleScene(FossenEnv):
    """This environment contains four agents, each steering a CyberShip II. They spawn in a N-E-S-W positions and should all turn right."""

    def __init__(self, scene):
        
        if scene == 0:
            N_TSs = 0
        
        elif scene in [1, 2, 4]:
            N_TSs = 1
        
        elif scene in [5]:
            N_TSs = 2

        super().__init__(N_TSs=N_TSs, N_TSs_random=False, cnt_approach="tau", state_pad=np.nan)
        self.scene = scene

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        self.goal = {"N" : 0.8 * self.N_max, "E" : 0.8 * self.E_max}
        N_init = 0.2 * self.N_max
        E_init = 0.2 * self.E_max
        head   = np.random.uniform(dtr(40.), dtr(50.))

        # init agent (OS for 'Own Ship')
        self.OS = CyberShipII(N_init       = N_init, 
                              E_init       = E_init, 
                              psi_init     = head,
                              u_init       = 0.0,
                              v_init       = 0.0,
                              r_init       = 0.0,
                              delta_t      = self.delta_t,
                              N_max        = self.N_max,
                              E_max        = self.E_max,
                              cnt_approach = self.cnt_approach,
                              tau_u        = 3.0)

        # set longitudinal speed to near-convergence
        # Note: if we don't do this, the TCPA calculation for spawning other vessels is heavily biased
        self.OS.nu[0] = self.OS._u_from_tau_u(self.OS.tau_u)

        # TS init depending on scenario
        if self.scene == 0:
            self.TSs = []

        elif self.scene in [1, 2, 4]:
            # head-on
            if self.scene == 1:
                N_init_TS = 100.0
                E_init_TS = 100.0
                head_TS   = dtr(225.0)
                tau_u_TS  = 3.0

            # starboard crossing        
            elif self.scene == 2:
                N_init_TS = 40
                E_init_TS = 150
                head_TS   = dtr(315.0)
                tau_u_TS  = 3.0

            # overtaking
            elif self.scene == 4:
                N_init_TS = 60.0
                E_init_TS = 60.0
                head_TS   = dtr(45.0)
                tau_u_TS  = 0.5

            # construct ship
            TS = CyberShipII(N_init      = N_init_TS, 
                            E_init       = E_init_TS, 
                            psi_init     = head_TS,
                            u_init       = 0.0,
                            v_init       = 0.0,
                            r_init       = 0.0,
                            delta_t      = self.delta_t,
                            N_max        = self.N_max,
                            E_max        = self.E_max,
                            cnt_approach = self.cnt_approach,
                            tau_u        = tau_u_TS)

            # predict converged speed of sampled TS
            # Note: if we don't do this, all further calculations are heavily biased
            TS.nu[0] = TS._u_from_tau_u(TS.tau_u)

            # setup
            self.TSs = [TS]
        
        elif self.scene in [5]:

            # one head-on
            TS1 = CyberShipII(N_init       = 100.0, 
                              E_init       = 100.0, 
                              psi_init     = dtr(225.0),
                              u_init       = 0.0,
                              v_init       = 0.0,
                              r_init       = 0.0,
                              delta_t      = self.delta_t,
                              N_max        = self.N_max,
                              E_max        = self.E_max,
                              cnt_approach = self.cnt_approach,
                              tau_u        = 3.0)

            # one starboard crosser
            TS2 = CyberShipII(N_init       = 40.0, 
                              E_init       = 150.0, 
                              psi_init     = dtr(315.0),
                              u_init       = 0.0,
                              v_init       = 0.0,
                              r_init       = 0.0,
                              delta_t      = self.delta_t,
                              N_max        = self.N_max,
                              E_max        = self.E_max,
                              cnt_approach = self.cnt_approach,
                              tau_u        = 3.0)
            self.TSs = [TS1, TS2]

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        return self.state

    def _handle_respawn(self, TS, respawn=True, mirrow=False, clip=False):
        return TS, False
