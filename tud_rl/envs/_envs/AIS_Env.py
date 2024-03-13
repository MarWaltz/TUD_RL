import copy
import math
import pickle

import gym
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from matplotlib import pyplot as plt
from tud_rl.envs._envs.AIS_Helper import AIS_Ship, CPANet
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                         NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, get_ship_domain, head_inter,
                                         meter_to_NM, polar_from_xy,
                                         project_vector, rtd, tcpa,
                                         xy_from_polar)
from tud_rl.envs._envs.VesselPlots import (TrajPlotter, get_rect, plot_jet,
                                           rotate_point)


class AIS_Env(gym.Env):
    """This environment contains an agent which gets thrown into an environment with vessels following real AIS trajectories.."""

    def __init__(self, 
                 AIS_path : str = None,
                 supervised_path : str = None,
                 N_TSs_max        = 3, 
                 N_TSs_random     = True, 
                 pdf_traj         = False,
                 w_dist           = 0.05,
                 w_head           = 2.0,
                 w_coll           = 1.8,
                 w_COLREG         = 2.0,
                 w_comf           = 0.3,
                 ada_r_comf       = True,
                 nonlinear_r_coll = True,
                 spawn_mode       = "line"):
        super().__init__()

        # Simulation settings
        self.delta_t = 5.0                           # simulation time interval (in s)
        self.N_max   = NM_to_meter(12.0)             # maximum N-coordinate (in m)
        self.E_max   = NM_to_meter(12.0)             # maximum E-coordinate (in m)
        self.CPA_N   = self.N_max / 2                # center point (N)
        self.CPA_E   = self.E_max / 2                # center point (E)

        self.N_TSs_max    = N_TSs_max                  # maximum number of other vessels
        self.N_TSs_random = N_TSs_random               # if true, samples a random number in [0, N_TSs] at start of each episode
                                                       # if false, always have N_TSs_max

        self.sight = NM_to_meter(20.0)     # sight of the agent (in m)

        # Read AIS data
        with open(AIS_path,"rb") as f:
            self.AIS_data = pickle.load(f)

        # Supervised learning estimator for DCPA / TCPA
        self.supervised_path = supervised_path
        if supervised_path is not None:
            self.forecast_window = 200 # forecast window
            self.n_obs_est   = 20      # number of points used for iterative 'n_forecasts'-step ahead forecasts
            self.n_forecasts = 5       # number of points to predict per forward pass

            assert self.forecast_window % self.n_forecasts == 0, "The number of points you want to estimate is not a multiple of the single-step forecasts."

            self.CPA_net = CPANet(n_forecasts=self.n_forecasts)
            self.CPA_net.load_state_dict(torch.load(supervised_path))

        # CR calculation
        self.CR_rec_dist = NM_to_meter(2.0)      # collision risk distance
        self.CR_al = 0.1                         # collision risk metric when TS is at CR_dist of agent

        # Spawning
        self.TCPA_crit         = 25 * 60               # critical TCPA (in s), relevant for state and spawning of TSs
        self.min_dist_spawn_TS = 5 * 320               # minimum distance of a spawning vessel to other TSs (in m)

        assert spawn_mode in ["center", "line", "line_v2"], "Unknown TS spawning mode."
        self.spawn_mode = spawn_mode

        self.goal_reach_dist = 3 * 320                    # euclidean distance (in m) at which goal is considered as reached
        self.stop_spawn_dist = NM_to_meter(7.0)           # euclidean distance (in m) under which vessels do not spawn anymore

        self.num_obs_OS = 3                               # number of observations for the OS
        
        if supervised_path is not None:
            self.num_obs_TS = 7                               # number of observations per TS
        else:
            self.num_obs_TS = 5

        # Plotting
        self.pdf_traj = pdf_traj   # whether to plot trajectory after termination
        self.TrajPlotter = TrajPlotter(plot_every=5 * 60, delta_t=self.delta_t)

        # Gym definitions
        obs_size = self.num_obs_OS + self.N_TSs_max * self.num_obs_TS

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Box(low  = np.full(1, -1.0, dtype=np.float32), 
                                       high = np.full(1,  1.0, dtype=np.float32))
        self.dhead = dtr(2.0)

        # Reward weights
        self.w_dist   = w_dist
        self.w_head   = w_head
        self.w_coll   = w_coll
        self.w_COLREG = w_COLREG
        self.w_comf   = w_comf
        self.ada_r_comf = ada_r_comf  # if True, comfort reward is off if there is collision risk
        self.nonlinear_r_coll = nonlinear_r_coll

        # Custom inits
        self._max_episode_steps = 1500
        self.r = 0
        self.r_dist   = 0
        self.r_head   = 0
        self.r_coll   = 0
        self.r_COLREG = 0
        self.r_comf   = 0

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # sample setup
        #self.sit_init = np.random.choice([0, 1, 2, 3])
        self.sit_init = 0

        # init agent heading
        head = [0.0, 1/2 * math.pi, math.pi, 3/2 * math.pi][self.sit_init]

        # init agent (OS for 'Own Ship'), so that CPA_N, CPA_E will be reached in 25 [min]
        self.OS = KVLCC2(N_init   = 0.0, 
                         E_init   = 0.0, 
                         psi_init = head,
                         u_init   = 0.0,
                         v_init   = 0.0,
                         r_init   = 0.0,
                         delta_t  = self.delta_t,
                         N_max    = self.N_max,
                         E_max    = self.E_max,
                         nps      = 1.21312) # gives 5 m/s speed

        # set longitudinal speed to near-convergence
        # Note: if we don't do this, the TCPA calculation for spawning other vessels is heavily biased
        self.OS.nu[0] = self.OS._get_u_from_nps(self.OS.nps, psi=self.OS.eta[2])

        # backtrace motion
        self.OS.eta[0] = self.CPA_N - self.OS._get_V() * np.cos(head) * self.TCPA_crit
        self.OS.eta[1] = self.CPA_E - self.OS._get_V() * np.sin(head) * self.TCPA_crit

        # init goal
        if self.sit_init == 0:
            self.goal = {"N" : self.CPA_N + abs(self.CPA_N - self.OS.eta[0]), "E" : self.OS.eta[1]}
        
        elif self.sit_init == 1:
            self.goal = {"N" : self.OS.eta[0], "E" : self.CPA_E + abs(self.CPA_E - self.OS.eta[1])}

        elif self.sit_init == 2:
            self.goal = {"N" : self.CPA_N - abs(self.CPA_N - self.OS.eta[0]), "E" : self.OS.eta[1]}

        elif self.sit_init == 3:
            self.goal = {"N" : self.OS.eta[0], "E" : self.CPA_E - abs(self.CPA_E - self.OS.eta[1])}

        # disturb heading for generalization
        self.OS.eta[2] += dtr(np.random.uniform(-5, 5))

        # initial distance to goal
        self.OS_goal_init = ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"])
        self.OS_goal_old  = self.OS_goal_init

        # initially compute ship domain for plotting
        rads  = np.linspace(0.0, 2*math.pi, 25)
        dists = [get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
            OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        # init other vessels
        if self.N_TSs_random:
            self.N_TSs = np.random.choice(a=[0, 1, 2, 3], p=[0.1, 0.3, 0.3, 0.3])
            #self.N_TSs = 3
        else:
            self.N_TSs = self.N_TSs_max

        # no list comprehension since we need access to previously spawned TS
        self.TSs = []
        for _ in range(self.N_TSs):
            self.TSs.append(self._get_TS())
        self.respawn_flags = [True for _ in self.TSs]

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        # trajectory storing
        self.TrajPlotter.reset(OS=self.OS, TSs=self.TSs, N_TSs=self.N_TSs)
        return self.state

    def _get_TS(self) -> AIS_Ship:
        """Places a target ship by sampling a 
            1) COLREG situation,
            2) TCPA (in s, or setting to 60s),
            3) relative bearing (in rad), 
            4) intersection angle (in rad),
            5) and a forward thrust (tau-u in N).
        """

        while True:
            if self.spawn_mode == "line":
                
                # Constructor for target ship
                TS = AIS_Ship(data=self.AIS_data)
                VTS = TS.v

                # quick access for OS
                N0, E0, _ = self.OS.eta
                chiOS = self.OS._get_course()
                VOS   = self.OS._get_V()

                # sample COLREG situation 
                # null = 0, head-on = 1, starb. cross. = 2, ports. cross. = 3, overtaking = 4
                if VTS <= 0.7 * VOS:
                    COLREG_s = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
                else:
                    COLREG_s = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])

                # determine relative speed of OS towards goal, need absolute bearing first
                bng_abs_goal = bng_abs(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])

                # project VOS vector on relative velocity direction
                VR_goal_x, VR_goal_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=bng_abs_goal)
                
                # sample time
                t_hit = np.random.uniform(self.TCPA_crit * 0.75, self.TCPA_crit)

                # compute hit point
                E_hit = E0 + VR_goal_x * t_hit
                N_hit = N0 + VR_goal_y * t_hit

                # Note: In the following, we only sample the intersection angle and not a relative bearing.
                #       This is possible since we construct the trajectory of the TS so that it will pass through (E_hit, N_hit), 
                #       and we need only one further information to uniquely determine the origin of its motion.

                # null: TS comes from behind
                if COLREG_s == 0:
                     C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

                # head-on
                elif COLREG_s == 1:
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

                # determine TS heading (treating absolute bearing towards goal as heading of OS)
                head_TS_s = angle_to_2pi(C_TS_s + bng_abs_goal)   

                # backtrace original position of TS
                E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
                N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

                # Set positional values
                TS.place_on_map(E_init=E_TS, N_init=N_TS, head_init=head_TS_s)
                
                # no TS yet there
                if len(self.TSs) == 0:
                    break

                # TS shouldn't spawn too close to other TS
                if min([ED(N0=N_TS, E0=E_TS, N1=TS_there.n, E1=TS_there.e, sqrt=True) for TS_there in self.TSs])\
                    >= self.min_dist_spawn_TS:
                    break
            else:
                raise NotImplementedError("Only 'line' spawning available.")
        return TS

    def _set_COLREGs(self):
        """Computes for each target ship the current COLREG situation and stores it internally."""

        # overwrite old situations
        self.TS_COLREGs_old = copy.copy(self.TS_COLREGs)

        # compute new ones
        self.TS_COLREGs = []

        for TS in self.TSs:
            self.TS_COLREGs.append(self._get_COLREG_situation(OS=self.OS, TS=TS))

    def _get_ghost_ship_list(self) -> list:
        """Returns a list of states characterizing a zero risk 'ghost' ship."""
        # ED
        ED_ghost = 1.0

        # relative bearing
        bng_rel_ghost = -1.0

        # heading intersection angle
        C_ghost = -1.0

        # speed
        V_ghost = 0.0

        # COLREG mode
        sigma_ghost = 0

        if self.supervised_path is not None:
            dcpa = 1.0
            tcpa = -1.0
            return [ED_ghost, bng_rel_ghost, C_ghost, V_ghost, sigma_ghost, dcpa, tcpa]
        else:
            return [ED_ghost, bng_rel_ghost, C_ghost, V_ghost, sigma_ghost]

    def _iterative_forecasts(self, n_for_est : np.ndarray, e_for_est : np.ndarray) -> tuple:
        """Performs iterative 'n_forecasts'-step ahead forecasts using the given input points.
        Returns:
            np.ndarray(self.forecast_windows,) : predicted n_points
            np.ndarray(self.forecast_windows,) : predicted e_points
        """
        assert (len(n_for_est) == self.n_obs_est) and (len(e_for_est) == self.n_obs_est), "Size mismatch for estimation data."
        
        # Rotate estimation data
        #rot_angle = 0.0 - bng_abs(N0=n_for_est[0], E0=e_for_est[0], N1=n_for_est[1], E1=e_for_est[1])
        #e_for_est, n_for_est = rotate_point(x=e_for_est, y=n_for_est, cx=e_for_est[0], cy=n_for_est[0], angle=-rot_angle)

        # Output setup
        n_steps = int(self.forecast_window / self.n_forecasts)
        n_predicted = np.array([])
        e_predicted = np.array([])

        for _ in range(n_steps):        

            # Take last 'n_obs_est' elements for prediction
            n_diff = torch.tensor(np.diff(n_for_est[-self.n_obs_est:]), dtype=torch.float32)
            e_diff = torch.tensor(np.diff(e_for_est[-self.n_obs_est:]), dtype=torch.float32)

            n_diff = torch.reshape(n_diff, (1, self.n_obs_est-1, 1))
            e_diff = torch.reshape(e_diff, (1, self.n_obs_est-1, 1))

            # Forward pass
            est = self.CPA_net(e_diff, n_diff)
            e_add = est[0][:self.n_forecasts].detach().numpy()
            n_add = est[0][self.n_forecasts:].detach().numpy()
            
            new_e = e_for_est[-1] + np.cumsum(e_add)
            new_n = n_for_est[-1] + np.cumsum(n_add)

            # Append estimates
            n_predicted = np.concatenate((n_predicted, new_n))
            e_predicted = np.concatenate((e_predicted, new_e))

            # Update estimation data
            n_for_est = np.concatenate((n_for_est, new_n))
            e_for_est = np.concatenate((e_for_est, new_e))

        # Rotate back
        #e_predicted, n_predicted = rotate_point(x=np.array(e_predicted), y=np.array(n_predicted), cx=e_for_est[0], cy=n_for_est[0], #angle=rot_angle)
        return n_predicted, e_predicted

    def _update_dcpa_tcpa_supervised(self, TS: AIS_Ship) -> None:
        """Updates the DCPA and TCPA using the corresponding neural network, 
        which has been trained separately."""
        # Only every 10th step for computation time
        if not hasattr(TS, "tcpa") or self.step_cnt % 10 == 0:

            ptr = TS.ptr

            # Take last 'n_obs_est' observations, including the current one
            n_base = TS.n_traj[ptr-self.n_obs_est+1:ptr+1]
            e_base = TS.e_traj[ptr-self.n_obs_est+1:ptr+1]

            # Predict future
            n_future, e_future = self._iterative_forecasts(n_for_est=n_base.copy(), e_for_est=e_base.copy())

            # 'Predict' past
            n_past, e_past = self._iterative_forecasts(n_for_est=np.flip(n_base.copy()), e_for_est=np.flip(e_base.copy()))
            n_past = np.flip(n_past)
            e_past = np.flip(e_past)

            # Stack it together
            n_TS = np.concatenate((n_past, n_base, n_future))
            e_TS = np.concatenate((e_past, e_base, e_future))
            TS.n_traj_pred = n_TS
            TS.e_traj_pred = e_TS

            # Agent behavior
            n_OS, e_OS, head_OS = self.OS.eta
            vOS_e, vOS_n = xy_from_polar(r=self.OS._get_V(), angle=head_OS)

            #n_OS_future = np.arange(1, self.forecast_window+1)
            #n_OS_base   = np.arange(-self.n_obs_est+1,1)
            #n_OS_past   = np.arange(-self.forecast_window - self.n_obs_est+1, -self.n_obs_est+1)
            n_OS_full = n_OS + vOS_n * np.arange(-self.forecast_window-self.n_obs_est+1, self.forecast_window+1) * self.delta_t
            e_OS_full = e_OS + vOS_e * np.arange(-self.forecast_window-self.n_obs_est+1, self.forecast_window+1) * self.delta_t

            # Compute dcpa and tcpa
            dist = np.sqrt((n_OS_full - n_TS)**2 + (e_OS_full - e_TS)**2)

            # Smoothing dist curve by using moving average (np.convolve)
            #kernel_size = 21
            #kernel = np.ones(kernel_size) / kernel_size
            #smoothed_dist = np.convolve(dist, kernel, mode='same')

            TS.dcpa = max([0, np.min(dist)])
            TS.tcpa = self.delta_t * (np.argmin(dist) - self.forecast_window - self.n_obs_est)
        else: 
            TS.tcpa -= self.delta_t

    def _set_state(self):
        """State consists of (all from agent's perspective): 
        
        OS:
            speed
        Goal:
            relative bearing
            ED_goal
        
        Dynamic obstacle:
            ED_TS
            relative bearing
            heading intersection angle C_T
            speed (V)
            COLREG mode TS (sigma_TS)
        """

        # quick access for OS
        N0, E0, head0 = self.OS.eta

        #-------------------------------- OS related ---------------------------------
        state_OS = np.array([self.OS._get_V() / 5.0])

        #------------------------------ goal related ---------------------------------
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        state_goal = np.array([bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0, to_2pi=False) / (math.pi), 
                               OS_goal_ED / self.E_max])

        #--------------------------- dynamic obstacle related -------------------------
        state_TSs = []

        for TS_idx, TS in enumerate(self.TSs):

            N = TS.n
            E = TS.e
            headTS = TS.head

            # consider TS if it is in sight
            ED_OS_TS = ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True)

            if ED_OS_TS <= self.sight:

                # euclidean distance
                D = get_ship_domain(A=self.OS.ship_domain_A, B=self.OS.ship_domain_B, C=self.OS.ship_domain_C, D=self.OS.ship_domain_D,\
                     OS=self.OS, TS=TS)
                ED_OS_TS_norm = (ED_OS_TS-D) / self.E_max

                # relative bearing
                bng_rel_TS = bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0, to_2pi=False) / (math.pi)

                # heading intersection angle
                C_TS = head_inter(head_OS=head0, head_TS=headTS, to_2pi=False) / (math.pi)

                # speed
                V_TS = TS.v / 5.0

                # COLREG mode
                sigma_TS = self.TS_COLREGs[TS_idx]

                # Interface to supervised learning module
                if self.supervised_path is not None:
                    self._update_dcpa_tcpa_supervised(TS)

                    dcpa = np.min([TS.dcpa / 2000.0, 1.0])
                    tcpa = np.clip(TS.tcpa / 900.0, -1.0, 1.0)

                    # store it
                    state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, sigma_TS, dcpa, tcpa])
                else:
                    # store it
                    state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, sigma_TS])

        # pad 'ghost ships'
        while len(state_TSs) != self.N_TSs_max:
            state_TSs.append(self._get_ghost_ship_list())

        # sort according to Euclidean distance
        state_TSs = sorted(state_TSs, key=lambda x: x[0], reverse=False)
        state_TSs = np.array(state_TSs).flatten()

        # combine state
        self.state = np.concatenate([state_OS, state_goal, state_TSs], dtype=np.float32)

    def _heading_control(self, a : float):
        """Controls the heading of the vessel."""
        assert (-1 <= a) and (a <= 1), "Unknown action."
        self.OS.eta[2] = angle_to_2pi(self.OS.eta[2] + a * self.dhead)

    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns new_state, r, done, {}."""
        #self.render()

        # perform control action
        a = float(a)
        self._heading_control(a)

        # update agent dynamics
        self.OS._upd_dynamics()

        # update environmental dynamics, e.g., other vessels
        [TS.update_dynamics() for TS in self.TSs]

        # handle respawning of other vessels
        if self.N_TSs > 0:
            self.TSs, self.respawn_flags = list(zip(*[self._handle_respawn(TS) for TS in self.TSs]))

        # update COLREG scenarios
        self._set_COLREGs()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # trajectory plotting
        self.TrajPlotter.step(OS=self.OS, TSs=self.TSs, respawn_flags=self.respawn_flags if self.N_TSs > 0 else None, \
            step_cnt=self.step_cnt)

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward(a)
        d = self._done()
        return self.state, self.r, d, {}

    def _handle_respawn(self, TS):
        """Handles respawning of a vessel due to being too far away from the agent.

        Args:
            TS (KVLCC2): Vessel of interest.
        Returns
            KVLCC2, respawn_flag (bool)
        """
        return TS, False

    def _get_CR(self, OS, TS):
        """Computes the collision risk metric similar to Chun et al. (2021)."""
        N0, E0, head0 = OS.eta
        N1, E1, head1 = TS.n, TS.e, TS.head
        D = get_ship_domain(A=OS.ship_domain_A, B=OS.ship_domain_B, C=OS.ship_domain_C, D=OS.ship_domain_D, OS=OS, TS=TS)

        # check if already in ship domain
        ED_OS_TS = ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True)
        if ED_OS_TS <= D:
            return 1.0

        # compute speeds and courses
        VOS = OS._get_V()
        VTS = TS.v
        chiOS = OS._get_course()
        chiTS = TS.head

        # compute CPA measures
        DCPA, TCPA, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N1, ETS=E1, \
            chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS, get_positions=True)

        # substract ship domain at TCPA = 0 from DCPA
        bng_rel_tcpa_from_OS_pers = bng_rel(N0=NOS_tcpa, E0=EOS_tcpa, N1=NTS_tcpa, E1=ETS_tcpa, head0=head0)
        domain_tcpa = get_ship_domain(A=OS.ship_domain_A, B=OS.ship_domain_B, C=OS.ship_domain_C, D=OS.ship_domain_D, OS=None, TS=None,\
            ang=bng_rel_tcpa_from_OS_pers)
        DCPA = max([0.0, DCPA-domain_tcpa])

        # check whether OS will be in front of TS when TCPA = 0
        bng_rel_tcpa_from_TS_pers = abs(bng_rel(N0=NTS_tcpa, E0=ETS_tcpa, N1=NOS_tcpa, E1=EOS_tcpa, head0=head1, to_2pi=False))

        if TCPA >= 0.0 and bng_rel_tcpa_from_TS_pers <= dtr(30.0):
            DCPA = DCPA * (1.2-math.exp(-math.log(5.0)/dtr(30.0)*bng_rel_tcpa_from_TS_pers))
        
        if TCPA >= 0.0:
            cr_cpa = math.exp((DCPA + 1.5 * TCPA) * math.log(self.CR_al) / self.CR_rec_dist)
        else:
            cr_cpa = math.exp((DCPA + 20.0 * abs(TCPA)) * math.log(self.CR_al) / self.CR_rec_dist)
        #self.DCPA = DCPA
        #self.TCPA = TCPA

        # CR based on euclidean distance to ship domain
        cr_ed = math.exp(-(ED_OS_TS-D)/(self.CR_rec_dist*0.3))

        #self.cr_ed_old = self.cr_ed
        #self.cr_cpa_old = self.cr_cpa

        #self.cr_ed = cr_ed
        #self.cr_cpa = cr_cpa

        return min([1.0, max([cr_cpa, cr_ed])])

    def _get_COLREG_situation(self, OS, TS):
        """Determines the COLREG situation from the perspective of the OS. 
        Follows Xu et al. (2020, Ocean Engineering; 2022, Neurocomputing).

        Args:
            OS (CyberShip):    own vessel with attributes eta, nu
            TS (CyberShip):    target vessel with attributes eta, nu

        Returns:
            0  -  no conflict situation
            1  -  head-on
            2  -  starboard crossing (small)
            3  -  starboard crossing (large)
            4  -  portside crossing
            5  -  overtaking
        """

        # quick access
        NOS, EOS, psi_OS = OS.eta
        NTS, ETS, psi_TS = TS.n, TS.e, TS.head
        V_OS  = OS._get_V()
        V_TS  = TS.v

        chiOS = OS._get_course()
        chiTS = TS.head

        # check whether TS is out of sight
        if ED(N0=NOS, E0=EOS, N1=NTS, E1=ETS) > self.sight:
            return 0

        # relative bearing from OS to TS
        bng_OS = bng_rel(N0=NOS, E0=EOS, N1=NTS, E1=ETS, head0=psi_OS)

        # relative bearing from TS to OS
        bng_TS = bng_rel(N0=NTS, E0=ETS, N1=NOS, E1=EOS, head0=psi_TS)

        # intersection angle
        C_T = head_inter(head_OS=psi_OS, head_TS=psi_TS)

        # velocity component of OS in direction of TS
        V_rel_x, V_rel_y = project_vector(VA=V_OS, angleA=chiOS, VB=V_TS, angleB=chiTS)
        V_rel = polar_from_xy(x=V_rel_x, y=V_rel_y, with_r=True, with_angle=False)[0]

        #-------------------------------------------------------------------------------------------------------
        # Note: For Head-on, starboard crossing, and portside crossing, we do not care about the sideslip angle.
        #       The latter comes only into play for checking the overall speed of USVs in overtaking.
        #-------------------------------------------------------------------------------------------------------

        # COLREG 1: Head-on
        if -5 <= rtd(angle_to_pi(bng_OS)) <= 5 and 175 <= rtd(C_T) <= 185:
            return 1
        
        # COLREG 2: Starboard crossing
        if 5 <= rtd(bng_OS) <= 112.5 and 185 <= rtd(C_T) <= 292.5:
            return 2

        # COLREG 3: Portside crossing
        if 247.5 <= rtd(bng_OS) <= 355 and 67.5 <= rtd(C_T) <= 175:
            return 3

        # COLREG 4: Overtaking
        if 112.5 <= rtd(bng_TS) <= 247.5 and -67.5 <= rtd(angle_to_pi(C_T)) <= 67.5 and V_rel > V_TS:
            return 4

        # COLREG 0: nothing
        return 0

    def _calculate_reward(self, a):
        """Returns reward of the current state."""

        N0, E0, head0 = self.OS.eta
        safe_sit = all([self._get_CR(OS=self.OS, TS=TS) <= 0.2 for TS in self.TSs])

        # --------------- Path planning reward (Xu et al. 2022 in Neurocomputing, Ocean Eng.) -----------
        # Distance reward
        OS_goal_ED       = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        r_dist           = (self.OS_goal_old - OS_goal_ED) / (20.0) - 1.0
        self.OS_goal_old = OS_goal_ED

        # Heading reward
        r_head = -abs(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0, to_2pi=False)) / math.pi

        # --------------------------------- 3./4. Collision/COLREG reward --------------------------------
        r_coll = 0
        r_COLREG = 0

        for TS_idx, TS in enumerate(self.TSs):

            # get ED
            ED_TS = ED(N0=N0, E0=E0, N1=TS.n, E1=TS.e)

            # reward based on collision risk
            CR = self._get_CR(OS=self.OS, TS=TS)
            if CR == 1.0:
                r_coll -= 10.0
            else:
                if self.nonlinear_r_coll:
                    r_coll -= math.sqrt(CR)
                else:
                    r_coll -= CR

            # COLREG: if vessel just spawned, don't assess COLREG reward
            if not self.respawn_flags[TS_idx]:

                # evaluate TS if in sight and has positive TCPA
                if ED_TS <= self.sight and tcpa(NOS=N0, EOS=E0, NTS=TS.n, ETS=TS.e,\
                     chiOS=self.OS._get_course(), chiTS=TS.head, VOS=self.OS._get_V(), VTS=TS.v) >= 0.0:

                    # steer to the right in Head-on and starboard crossing situations
                    if self.TS_COLREGs_old[TS_idx] in [1, 2] and a <= 0.0:
                        r_COLREG -= 1.0

        # --------------------------------- 5. Comfort penalty --------------------------------
        if self.ada_r_comf:
            if safe_sit:
                r_comf = -(a**2)
            else:
                r_comf = 0.0
        else:
            r_comf = -(a**2)

        # -------------------------------------- Overall reward --------------------------------------------
        w_sum = self.w_dist + self.w_head + self.w_coll + self.w_COLREG + self.w_comf

        self.r_dist   = r_dist * self.w_dist / w_sum
        self.r_head   = r_head * self.w_head / w_sum
        self.r_coll   = r_coll * self.w_coll / w_sum
        self.r_COLREG = r_COLREG * self.w_COLREG / w_sum
        self.r_comf   = r_comf * self.w_comf / w_sum
        self.r = self.r_dist + self.r_head + self.r_coll + self.r_COLREG + self.r_comf

    def _done(self):
        """Returns boolean flag whether episode is over."""

        d = False
        N0 = self.OS.eta[0]
        E0 = self.OS.eta[1]

        # goal reached
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        if OS_goal_ED <= self.goal_reach_dist:
            d = True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            d = True

        # create pdf trajectory
        if d and self.pdf_traj:
            self.TrajPlotter.plot_traj_fnc(E_max=self.E_max, N_max=self.N_max, goal=self.goal,\
                goal_reach_dist=self.goal_reach_dist, Lpp=self.OS.Lpp, step_cnt=self.step_cnt)
        return d

    def __str__(self) -> str:
        N0, E0, head0 = self.OS.eta
        u, v, r = np.round(self.OS.nu,3)

        ste = f"Step: {self.step_cnt}"
        pos = f"N: {meter_to_NM(N0) - 7:.2f}, E: {meter_to_NM(E0) - 7:.2f}, " + r"$\psi$: " + f"{rtd(head0):.2f}°"
        vel = f"u: {u:.4f}, v: {v:.4f}, r: {r:.4f}"
        return ste + "\n" + pos + "\n" + vel

    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep (except we only want trajectory)
        if not self.pdf_traj:

            if self.step_cnt % 1 == 0: 

                # check whether figure has been initialized
                if len(plt.get_fignums()) == 0:
                    #self.fig = plt.figure(figsize=(10, 7))
                    #self.gs  = self.fig.add_gridspec(2, 2)
                    #self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                    #self.ax1 = self.fig.add_subplot(self.gs[0, 1]) # reward
                    #self.ax2 = self.fig.add_subplot(self.gs[1, 0]) # state
                    #self.ax3 = self.fig.add_subplot(self.gs[1, 1]) # action

                    self.fig2 = plt.figure(figsize=(10,7))
                    self.fig2_ax = self.fig2.add_subplot(111)

                    plt.ion()
                    plt.show()
                
                # ------------------------------ ship movement --------------------------------
                for ax in [self.fig2_ax]:
                    # clear prior axes, set limits and add labels and title
                    ax.clear()
                    ax.set_xlim(0, self.E_max)
                    ax.set_ylim(0, self.N_max)

                    # E-axis
                    ax.set_xlim(0, self.E_max)
                    ax.set_xticks([NM_to_meter(nm) for nm in range(15) if nm % 2 == 1])
                    ax.set_xticklabels([nm - 7 for nm in range(15) if nm % 2 == 1])
                    ax.set_xlabel("East [NM]", fontsize=8)

                    # N-axis
                    ax.set_ylim(0, self.N_max)
                    ax.set_yticks([NM_to_meter(nm) for nm in range(15) if nm % 2 == 1])
                    ax.set_yticklabels([nm - 7 for nm in range(15) if nm % 2 == 1])
                    ax.set_ylabel("North [NM]", fontsize=8)

                    # access
                    N0, E0, head0 = self.OS.eta
                    chiOS = self.OS._get_course()
                    VOS = self.OS._get_V()

                    # set OS
                    rect = get_rect(E = E0, N = N0, width = self.OS.B, length = self.OS.Lpp, heading = head0,
                                    linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
                    
                    # step information
                    ax.text(0.05 * self.E_max, 0.9 * self.N_max, self.__str__(), fontsize=8)

                    # ship domain
                    xys = [rotate_point(E0 + x, N0 + y, cx=E0, cy=N0, angle=-head0) for x, y in zip(self.domain_xs, self.domain_ys)]
                    xs = [xy[0] for xy in xys]
                    ys = [xy[1] for xy in xys]
                    ax.plot(xs, ys, color="black", alpha=0.7)

                    # add jets according to COLREGS
                    for COLREG_deg in [5, 355]:
                        ax = plot_jet(axis = ax, E=E0, N=N0, l = self.sight, 
                                      angle = head0 + dtr(COLREG_deg), color='black', alpha=0.3)

                    # set goal (stored as NE)
                    ax.scatter(self.goal["E"], self.goal["N"], color="blue")
                    ax.text(self.goal["E"], self.goal["N"] + 2,
                                r"$\psi_g$" + f": {np.round(rtd(bng_rel(N0=N0, E0=E0, N1=self.goal['N'], E1=self.goal['E'], head0=head0)),3)}°",
                                horizontalalignment='center', verticalalignment='center', color='blue')
                    circ = patches.Circle((self.goal["E"], self.goal["N"]), radius=self.goal_reach_dist, edgecolor='blue', facecolor='none', alpha=0.3)
                    ax.add_patch(circ)

                    # set other vessels
                    for TS in self.TSs:

                        # access
                        N, E, headTS = TS.n, TS.e, TS.head
                        chiTS = TS.head
                        VTS = TS.v

                        # determine color according to COLREG scenario
                        COLREG = self._get_COLREG_situation(OS=self.OS, TS=TS)
                        col = COLREG_COLORS[COLREG]

                        # place TS
                        rect = get_rect(E = E, N = N, width = self.OS.B, length = self.OS.Lpp, heading = headTS,
                                        linewidth=1, edgecolor=col, facecolor='none', label=COLREG_NAMES[COLREG])
                        ax.add_patch(rect)

                        # add two jets according to COLREGS
                        #for COLREG_deg in [5, 355]:
                        #    ax= plot_jet(axis = ax, E=E, N=N, l = self.sight, 
                        #                 angle = headTS + dtr(COLREG_deg), color=col, alpha=0.3)

                        # collision risk                        
                        #CR = self._get_CR(OS=self.OS, TS=TS)
                        #ax.text(E + 800, N-600, f"CR: {np.round(CR, 4)}", fontsize=7,
                        #            horizontalalignment='center', verticalalignment='center', color=col)

                        # compute CPA measures
                        DCPA_TS, TCPA_TS, NOS_tcpa, EOS_tcpa, NTS_tcpa, ETS_tcpa = cpa(NOS=N0, EOS=E0, NTS=N, ETS=E, \
                            chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS, get_positions=True)

                        # check whether OS will be in front of TS when TCPA = 0
                        #bng_rel_tcpa_TS = abs(bng_rel(N0=NTS_tcpa, E0=ETS_tcpa, N1=NOS_tcpa, E1=EOS_tcpa, head0=chiTS, to_2pi=False))
                        #col = "salmon" if bng_rel_tcpa_TS <= dtr(30.0) else col
                        #ax.scatter(ETS_tcpa, NTS_tcpa, color=col, s=10)
                        #ax.scatter(EOS_tcpa, NOS_tcpa, color="black", s=10)

                        ax.text(E + 1000, N + 200, f"TCPA: {np.round(TCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        ax.text(E + 1000, N-200, f"DCPA: {np.round(DCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        
                        try:
                            ax.text(E + 1000, N - 700, f"TCPA-SL: {np.round(TS.tcpa, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                            ax.text(E + 1000, N - 1100, f"DCPA-SL: {np.round(TS.dcpa, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        except:
                            pass

                        # Plot true trajectory
                        ax.scatter(TS.e_traj[TS.ptr:TS.ptr+self.forecast_window], TS.n_traj[TS.ptr:TS.ptr+self.forecast_window],
                                   s=12, color="black")

                        # Plot estimated trajectory
                        if hasattr(TS, "n_traj_pred"):
                            ax.scatter(TS.e_traj_pred, TS.n_traj_pred, s=12, color="red")

                    # set legend for COLREGS
                    ax.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], fontsize=8,
                                    loc='lower center', bbox_to_anchor=(0.52, 1.0), fancybox=False, shadow=False, ncol=6).get_frame().set_linewidth(0.0)


                # ------------------------------ reward plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax1.clear()
                        self.ax1.old_time = 0
                        self.ax1.old_r_dist = 0
                        self.ax1.old_r_head = 0
                        self.ax1.old_r_coll = 0
                        self.ax1.old_r_COLREG = 0
                        self.ax1.old_r_comf = 0

                    self.ax1.set_xlim(0, self._max_episode_steps)
                    #self.ax1.set_ylim(-1.25, 0.1)
                    self.ax1.set_xlabel("Timestep in episode")
                    self.ax1.set_ylabel("Reward")

                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_dist, self.r_dist], color = "black", label="Distance")
                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_head, self.r_head], color = "grey", label="Heading")
                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_coll, self.r_coll], color = "red", label="Collision")
                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_COLREG, self.r_COLREG], color = "darkorange", label="COLREG")
                    self.ax1.plot([self.ax1.old_time, self.step_cnt], [self.ax1.old_r_comf, self.r_comf], color = "darkcyan", label="Comfort")
                    
                    if self.step_cnt == 0:
                        self.ax1.legend()

                    self.ax1.old_time = self.step_cnt
                    self.ax1.old_r_dist = self.r_dist
                    self.ax1.old_r_head = self.r_head
                    self.ax1.old_r_coll = self.r_coll
                    self.ax1.old_r_COLREG = self.r_COLREG
                    self.ax1.old_r_comf = self.r_comf


                # ------------------------------ state plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax2.clear()
                        self.ax2.old_time = 0
                        self.ax2.old_state = self.state_init

                    self.ax2.set_xlim(0, self._max_episode_steps)
                    self.ax2.set_xlabel("Timestep in episode")
                    self.ax2.set_ylabel("State information")

                    #for i, obs in enumerate(self.state):
                    #    if i > 6:
                    #        self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_state[i], obs], 
                    #                    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i-7], 
                    #                    label=self.state_names[i])
                    #self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.cr_cpa_old, self.cr_cpa], color="red", label="CR_CPA")
                    #self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.cr_ed_old, self.cr_ed], color="blue", label="CR_ED")
                    #if self.step_cnt == 0:
                    #    self.ax2.legend()

                    self.ax2.old_time = self.step_cnt
                    self.ax2.old_state = self.state

                # ------------------------------ action plot --------------------------------
                if False:
                    if self.step_cnt == 0:
                        self.ax3.clear()
                        self.ax3_twin = self.ax3.twinx()
                        #self.ax3_twin.clear()
                        self.ax3.old_time = 0
                        self.ax3.old_action = 0
                        self.ax3.old_rud_angle = 0
                        self.ax3.old_tau_cnt_r = 0

                    self.ax3.set_xlim(0, self._max_episode_steps)
                    self.ax3.set_ylim(-0.1, self.action_space.n - 1 + 0.1)
                    self.ax3.set_yticks(range(self.action_space.n))
                    self.ax3.set_yticklabels(range(self.action_space.n))
                    self.ax3.set_xlabel("Timestep in episode")
                    self.ax3.set_ylabel("Action (discrete)")

                    self.ax3.plot([self.ax3.old_time, self.step_cnt], [self.ax3.old_action, self.OS.action], color="black", alpha=0.5)
                    
                    self.ax3.old_time = self.step_cnt
                    self.ax3.old_action = self.OS.action

                plt.pause(0.001)
