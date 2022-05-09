from .FossenEnv import *
from .FossenFnc import dtr


class Imazu(FossenEnv):
    """Implements the 22 ship encounter situations of Imazu (1987) as detailed in Sawada et al. (2021, JMST)."""

    def __init__(self, plot_traj, situation, state_design):
        
        if situation in range(5):
            N_TSs = 1
        
        elif situation in range(5, 13):
            N_TSs = 2
        
        elif situation in range(13, 23):
            N_TSs = 3

        super().__init__(N_TSs_max=N_TSs, plot_traj=plot_traj, N_TSs_random=False, N_TSs_increasing=False, cnt_approach="tau", state_design=state_design)

        self.situation = situation
        self.N_TSs = N_TSs

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)
        TCPA = 80
        
        #--------------------------- OS spawn --------------------------------
        head = 0.0

        # init agent (OS for 'Own Ship') with dummy N, E
        self.OS = CyberShipII(N_init       = 0.0, 
                              E_init       = 0.0, 
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
        self.OS.nu[0] = self.OS._u_from_tau_u(self.OS.tau_u)

        # backtrace to motion
        self.OS.eta[0] = 100 - self.OS._get_V() * np.cos(head) * TCPA
        self.OS.eta[1] = 100 - self.OS._get_V() * np.sin(head) * TCPA
        
        #--------------------------- Goal spawn --------------------------------
        self.goal = {"N" : 100 + abs(100 - self.OS.eta[0]), "E" : self.OS.eta[1]}

        #--------------------------- TS spawn --------------------------------
        
        # TS should be after 'TCPA' at point (0,0)
        # since we have speed and heading, we can uniquely determine origin of the motion

        # construct ship with dummy N, E, heading
        TS1 = CyberShipII(N_init       = 0, 
                          E_init       = 0, 
                          psi_init     = 0,
                          u_init       = 0.0,
                          v_init       = 0.0,
                          r_init       = 0.0,
                          delta_t      = self.delta_t,
                          N_max        = self.N_max,
                          E_max        = self.E_max,
                          cnt_approach = self.cnt_approach,
                          tau_u        = 3.0)

        if self.situation in range(5):

            # lower speed for overtaking situations
            if self.situation == 3:
                TS1.tau_u = 1.0
                TS1._set_tau()

            # predict converged speed of TS
            TS1.nu[0] = TS1._u_from_tau_u(TS1.tau_u)

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
            TS1.eta[0] = 100 - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
            TS1.eta[1] = 100 - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

            # setup
            self.TSs = [TS1]
        

        elif self.situation in range(5, 13):

            # set TS2
            TS2 = copy.deepcopy(TS1)

            # lower speed for overtaking situations
            if self.situation == 7:
                TS1.tau_u = 1.0
                TS1._set_tau()

            # predict converged speed of TS
            TS1.nu[0] = TS1._u_from_tau_u(TS1.tau_u)
            TS2.nu[0] = TS2._u_from_tau_u(TS2.tau_u)

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

            elif self.situation == 12:
                headTS1 = angle_to_2pi(dtr(-45))
                headTS2 = angle_to_2pi(dtr(-10))

            # backtrace to motion
            TS1.eta[2] = headTS1
            TS1.eta[0] = 100 - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
            TS1.eta[1] = 100 - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

            TS2.eta[2] = headTS2
            TS2.eta[0] = 100 - TS2._get_V() * np.cos(TS2.eta[2]) * TCPA
            TS2.eta[1] = 100 - TS2._get_V() * np.sin(TS2.eta[2]) * TCPA

            # setup
            self.TSs = [TS1, TS2]
        
        elif self.situation in range(13, 23):

            # set TS2, TS3
            TS2 = copy.deepcopy(TS1)
            TS3 = copy.deepcopy(TS1)

            # lower speed for overtaking situations
            if self.situation in [15, 17, 20, 22]:
                TS1.tau_u = 1.0
                TS1._set_tau()

            # predict converged speed of TS
            TS1.nu[0] = TS1._u_from_tau_u(TS1.tau_u)
            TS2.nu[0] = TS2._u_from_tau_u(TS2.tau_u)
            TS3.nu[0] = TS3._u_from_tau_u(TS3.tau_u)

            # heading according to situation
            if self.situation == 13:
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
            TS1.eta[0] = 100 - TS1._get_V() * np.cos(TS1.eta[2]) * TCPA
            TS1.eta[1] = 100 - TS1._get_V() * np.sin(TS1.eta[2]) * TCPA

            TS2.eta[2] = headTS2
            TS2.eta[0] = 100 - TS2._get_V() * np.cos(TS2.eta[2]) * TCPA
            TS2.eta[1] = 100 - TS2._get_V() * np.sin(TS2.eta[2]) * TCPA

            TS3.eta[2] = headTS3
            TS3.eta[0] = 100 - TS3._get_V() * np.cos(TS3.eta[2]) * TCPA
            TS3.eta[1] = 100 - TS3._get_V() * np.sin(TS3.eta[2]) * TCPA

            # setup
            self.TSs = [TS1, TS2, TS3]

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs_max
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        # trajectory storing
        if self.plot_traj:
            self.OS_traj_N = [self.OS.eta[0]]
            self.OS_traj_E = [self.OS.eta[1]]

            self.TS_traj_N = [[] for _ in range(self.N_TSs)]
            self.TS_traj_E = [[] for _ in range(self.N_TSs)]
            self.TS_ptr = [i for i in range(self.N_TSs)]

            for TS_idx, TS in enumerate(self.TSs):
                ptr = self.TS_ptr[TS_idx]                
                self.TS_traj_N[ptr].append(TS.eta[0])
                self.TS_traj_E[ptr].append(TS.eta[1])

        return self.state

    def _nm_up(self, x):
        """Takes a value in nautic miles and transforms it to the customized simulation scale."""
        return (x + 10) * 10

    def _handle_respawn(self, TS, respawn=True, mirrow=False, clip=False):
        return TS, False

    def _calculate_reward(self):
        return 0.0
