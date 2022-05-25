import copy
import math

import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                         NM_to_meter, angle_to_2pi,
                                         angle_to_pi, bng_abs, bng_rel, cpa,
                                         dtr, head_inter, meter_to_NM,
                                         polar_from_xy, project_vector, rtd,
                                         tcpa, xy_from_polar)


class MMG_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2."""

    def __init__(self, 
                 N_TSs_max        = 3, 
                 N_TSs_random     = False, 
                 N_TSs_increasing = False,
                 state_design     = "RecDQN", 
                 plot_traj        = True,
                 CR_zero_negTCPA  = True,
                 w_dist           = 1.0,
                 w_head           = 1.0,
                 w_coll           = 1.0,
                 w_COLREG         = 1.0,
                 w_comf           = 1.0,
                 spawn_mode       = "line"):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0                           # simulation time interval (in s)
        self.N_max   = NM_to_meter(14.0)             # maximum N-coordinate (in m)
        self.E_max   = NM_to_meter(14.0)             # maximum E-coordinate (in m)
        self.CPA_N   = self.N_max / 2                # center point (N)
        self.CPA_E   = self.E_max / 2                # center point (E)

        self.N_TSs_max    = N_TSs_max                  # maximum number of other vessels
        self.N_TSs_random = N_TSs_random               # if true, samples a random number in [0, N_TSs] at start of each episode
                                                       # if false, always have N_TSs_max
        self.N_TSs_increasing = N_TSs_increasing       # if true, have schedule for number TSs
                                                       # if false, always have N_TSs_max

        assert sum([N_TSs_random, N_TSs_increasing]) <= 1, "Either random number of TS or schedule, not both."
        if self.N_TSs_increasing:
            self.outer_step_cnt = 0

        self.sight             = NM_to_meter(20.0)     # sight of the agent (in m)

        # CR calculation
        self.CR_dist_multiple  = 4                     # collision risk distance = multiple * ship_domain (in m)
        self.CR_al             = 0.1                   # collision risk metric when TS is at CR_dist of agent
        self.CR_zero_negTCPA   = CR_zero_negTCPA       # if True: CR is 0 if TCPA is negative, otherwise uses scaled absolute value

        # spawning
        self.TCPA_crit         = 25 * 60               # critical TCPA (in s), relevant for state and spawning of TSs
        self.min_dist_spawn_TS = 5 * 320               # minimum distance of a spawning vessel to other TSs (in m)

        assert state_design in ["maxRisk", "RecDQN"], "Unknown state design for FossenEnv. Should be 'maxRisk' or 'RecDQN'."
        self.state_design = state_design

        self.goal_reach_dist = 3 * 320                    # euclidean distance (in m) at which goal is considered as reached
        self.stop_spawn_dist = NM_to_meter(7.0)           # euclidean distance (in m) under which vessels do not spawn anymore

        self.num_obs_OS = 7                               # number of observations for the OS
        self.num_obs_TS = 6                               # number of observations per TS

        self.plot_traj = plot_traj                             # whether to plot trajectory after termination
        self.plot_every = 5 * 60                                # seconds between markers in trajectory plotting
        self.plot_every_step = self.plot_every / self.delta_t  # number of timesteps between markers in trajectory plotting

        assert spawn_mode in ["center", "line", "line_v2"], "Unknown TS spawning mode."
        self.spawn_mode = spawn_mode

        # gym definitions
        if state_design == "RecDQN":
            obs_size = self.num_obs_OS + max([1, self.N_TSs_max]) * self.num_obs_TS

        elif state_design == "maxRisk":
            obs_size = self.num_obs_OS + self.num_obs_TS

        self.observation_space = spaces.Box(low  = np.full(obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(obs_size,  np.inf, dtype=np.float32))
        self.action_space = spaces.Discrete(3)

        # reward weights
        self.w_dist   = w_dist
        self.w_head   = w_head
        self.w_coll   = w_coll
        self.w_COLREG = w_COLREG
        self.w_comf   = w_comf

        # custom inits
        self._max_episode_steps = 1500
        self.r = 0
        self.r_dist   = 0
        self.r_head   = 0
        self.r_coll   = 0
        self.r_COLREG = 0
        self.r_comf   = 0
        self.state_names = ["u", "v", "r", r"$\Psi$", r"$\dot{r}$", r"$\tau_r$", r"$\beta_{G}$", r"$ED_{G}$"]


    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # sample setup
        self.sit_init = np.random.choice([0, 1, 2, 3])

        # init agent heading
        if self.sit_init == 0:
            head = 0.0
        
        elif self.sit_init == 1:
            head = 1/2 * math.pi

        elif self.sit_init == 2:
            head = math.pi

        elif self.sit_init == 3:
            head = 3/2 * math.pi

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
                         nps      = 1.8)

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
        rads  = np.linspace(0.0, 2*math.pi, 100)
        dists = [self._get_ship_domain(OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_plot_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_plot_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        # init other vessels
        if self.N_TSs_random:
            self.N_TSs = np.random.choice(a=[0, 1, 2, 3], p=[0.1, 0.5, 0.3, 0.1])

        elif self.N_TSs_increasing:
            raise NotImplementedError()
        else:
            self.N_TSs = self.N_TSs_max

        # no list comprehension since we need access to previously spawned TS
        self.TSs = []
        for _ in range(self.N_TSs):
            self.TSs.append(self._get_TS())

        # determine current COLREG situations
        self.TS_COLREGs = [0] * self.N_TSs
        self._set_COLREGs()

        # init state
        self._set_state()
        self.state_init = self.state

        # trajectory storing
        if self.plot_traj:
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


    def _get_TS(self):
        """Places a target ship by sampling a 
            1) COLREG situation,
            2) TCPA (in s, or setting to 60s),
            3) relative bearing (in rad), 
            4) intersection angle (in rad),
            5) and a forward thrust (tau-u in N).
        Returns: 
            KVLCC2."""

        while True:
            TS = KVLCC2(N_init   = np.random.uniform(self.N_max / 5, self.N_max), 
                        E_init   = np.random.uniform(self.E_max / 5, self.E_max), 
                        psi_init = np.random.uniform(0, 2*math.pi),
                        u_init   = 0.0,
                        v_init   = 0.0,
                        r_init   = 0.0,
                        delta_t  = self.delta_t,
                        N_max    = self.N_max,
                        E_max    = self.E_max,
                        nps      = np.random.uniform(0.9, 1.1) * self.OS.nps)

            # predict converged speed of sampled TS
            # Note: if we don't do this, all further calculations are heavily biased
            TS.nu[0] = TS._get_u_from_nps(TS.nps, psi=TS.eta[2])

            #--------------------------------------- line mode --------------------------------------
            if self.spawn_mode == "line":

                # quick access for OS
                N0, E0, _ = self.OS.eta
                chiOS = self.OS._get_course()
                VOS   = self.OS._get_V()

                # sample COLREG situation 
                # head-on = 1, starb. cross. = 2, ports. cross. = 3, overtaking = 4
                COLREG_s = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.225, 0.225, 0.225, 0.225])
                if COLREG_s == 0:
                    return TS

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

                # determine TS heading (treating absolute bearing towards goal as heading of OS)
                head_TS_s = angle_to_2pi(C_TS_s + bng_abs_goal)   

                # no speed constraints except in overtaking
                if COLREG_s in [1, 2, 3]:
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
                E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
                N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

                # set positional values
                TS.eta = np.array([N_TS, E_TS, head_TS_s], dtype=np.float32)
                
                # no TS yet there
                if len(self.TSs) == 0:
                    break

                # TS shouldn't spawn too close to other TS
                if min([ED(N0=N_TS, E0=E_TS, N1=TS_there.eta[0], E1=TS_there.eta[1], sqrt=True) for TS_there in self.TSs])\
                    >= self.min_dist_spawn_TS:
                    break

            elif self.spawn_mode == "line_v2":

                # quick access for OS
                N0, E0, _ = self.OS.eta
                chiOS = self.OS._get_course()
                VOS   = self.OS._get_V()

                # sample situation
                setting = np.random.choice(["overtaker", "head-on", "random"], p=[0.1, 0.1, 0.8])

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

                if setting == "overtaker":

                    # sample intersection angle
                    C_TS_s = angle_to_2pi(dtr(np.random.uniform(-67.5, 67.5)))

                    # determine TS heading (treating absolute bearing towards goal as heading of OS)
                    head_TS_s = angle_to_2pi(C_TS_s + bng_abs_goal)   

                    # project VOS vector on TS direction
                    VR_TS_x, VR_TS_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=head_TS_s)
                    V_max_TS = polar_from_xy(x=VR_TS_x, y=VR_TS_y, with_r=True, with_angle=False)[0]

                    # sample TS speed
                    VTS = np.random.uniform(0.3, 0.7) * V_max_TS
                    TS.nu[0] = VTS

                    # set nps of TS so that it will keep this velocity
                    TS.nps = TS._get_nps_from_u(VTS, psi=head_TS_s)

                else:

                    # sample intersection angle
                    if setting == "head-on":
                        C_TS_s = dtr(np.random.uniform(175, 185))
                    else:
                        C_TS_s = dtr(np.random.uniform(0, 360))

                    # determine TS heading (treating absolute bearing towards goal as heading of OS)
                    head_TS_s = angle_to_2pi(C_TS_s + bng_abs_goal)

                    # no speed constraints here
                    VTS = TS.nu[0]

                # backtrace original position of TS
                E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
                N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

                # set positional values
                TS.eta = np.array([N_TS, E_TS, head_TS_s], dtype=np.float32)

                # no TS yet there
                if len(self.TSs) == 0:
                    break

                # TS shouldn't spawn too close to other TS
                if min([ED(N0=N_TS, E0=E_TS, N1=TS_there.eta[0], E1=TS_there.eta[1], sqrt=True) for TS_there in self.TSs])\
                    >= self.min_dist_spawn_TS:
                    break

            #--------------------------------------- center mode --------------------------------------
            elif self.spawn_mode == "center":

                # sample either overtaker, head-on, or random angle
                setting = np.random.choice(["overtaker", "head-on", "random"], p=[0.15, 0.15, 0.7])

                if setting == "overtaker":
                    
                    # set heading
                    TS.eta[2] = angle_to_2pi(self.OS.eta[2] + dtr(np.random.uniform(-5, 5)))

                    # reduce speed
                    TS.nps = 0.7
                    TS.nu[0] = TS._get_u_from_nps(TS.nps, psi=TS.eta[2])
                
                elif setting == "head-on":

                    # set heading
                    TS.eta[2] = angle_to_2pi(self.OS.eta[2] + dtr(np.random.uniform(175, 185)))

                else:
                    # sample heading
                    TS.eta[2] = np.random.uniform(0, 2*math.pi)
                    
                # backtrace motion
                TS.eta[0] = self.CPA_N - TS._get_V() * np.cos(TS.eta[2]) * self.TCPA_crit
                TS.eta[1] = self.CPA_E - TS._get_V() * np.sin(TS.eta[2]) * self.TCPA_crit

                break

        return TS


    def _set_COLREGs(self):
        """Computes for each target ship the current COLREG situation and stores it internally."""

        # overwrite old situations
        self.TS_COLREGs_old = copy.copy(self.TS_COLREGs)

        # compute new ones
        self.TS_COLREGs = []

        for TS in self.TSs:
            self.TS_COLREGs.append(self._get_COLREG_situation(OS=self.OS, TS=TS))


    def _set_state(self):
        """State consists of (all from agent's perspective): 
        
        OS:
            u, v, r
            r_dot
            rudder_angle

        Goal:
            relative bearing
            ED_goal
        
        Dynamic obstacle:
            ED_TS
            relative bearing
            heading intersection angle C_T
            speed (V)
            COLREG mode TS (sigma_TS)
            CR
        """

        # quick access for OS
        N0, E0, head0 = self.OS.eta

        #-------------------------------- OS related ---------------------------------
        cmp1 = self.OS.nu / np.array([5.0, 0.5, 0.002])
        cmp2 = np.array([#angle_to_pi(head0) / (math.pi),               # heading
                         self.OS.nu_dot[2],                            # r_dot
                         self.OS.rud_angle / self.OS.rud_angle_max])   # rudder angle
        state_OS = np.concatenate([cmp1, cmp2])


        #------------------------------ goal related ---------------------------------
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        state_goal = np.array([angle_to_pi(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0)) / (math.pi), 
                               OS_goal_ED / self.E_max])


        #--------------------------- dynamic obstacle related -------------------------
        state_TSs = []

        for TS_idx, TS in enumerate(self.TSs):

            N, E, headTS = TS.eta

            # consider TS if it is in sight
            ED_OS_TS = ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True)

            if ED_OS_TS <= self.sight:

                # euclidean distance
                ED_OS_TS_norm = ED_OS_TS / self.E_max

                # relative bearing
                bng_rel_TS = angle_to_pi(bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0)) / (math.pi)

                # heading intersection angle
                C_TS = angle_to_pi(head_inter(head_OS=head0, head_TS=headTS)) / (math.pi)

                # speed
                V_TS = TS._get_V()

                # COLREG mode
                sigma_TS = self.TS_COLREGs[TS_idx]

                # collision risk
                CR = self._get_CR(OS=self.OS, TS=TS)

                # store it
                state_TSs.append([ED_OS_TS_norm, bng_rel_TS, C_TS, V_TS, sigma_TS, CR])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agents
        if len(state_TSs) == 0:

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

            # collision risk
            CR_ghost = 0.0

            state_TSs.append([ED_ghost, bng_rel_ghost, C_ghost, V_ghost, sigma_ghost, CR_ghost])

        # sort according to collision risk (ascending, larger CR is more dangerous)
        state_TSs = sorted(state_TSs, key=lambda x: x[-1])

        #order = np.argsort(risk_ratios)[::-1]
        #state_TSs = [state_TSs[idx] for idx in order]

        if self.state_design == "RecDQN":

            # keep everything, pad nans at the right side to guarantee state size is always identical
            state_TSs = np.array(state_TSs).flatten(order="C")

            # at least one since there is always the ghost ship
            desired_length = self.num_obs_TS * max([self.N_TSs_max, 1])  

            state_TSs = np.pad(state_TSs, (0, desired_length - len(state_TSs)), \
                'constant', constant_values=np.nan).astype(np.float32)

        elif self.state_design == "maxRisk":

            # select only highest risk TS
            state_TSs = np.array(state_TSs[-1])

        #------------------------------- combine state ------------------------------
        self.state = np.concatenate([state_OS, state_goal, state_TSs])


    def step(self, a):
        """Takes an action and performs one step in the environment.
        Returns reward, new_state, done, {}."""

        # perform control action
        self.OS._control(a)

        # update agent dynamics
        self.OS._upd_dynamics()

        # update environmental dynamics, e.g., other vessels
        [TS._upd_dynamics() for TS in self.TSs]

        # handle respawning of other vessels
        if self.N_TSs > 0:
            self.TSs, self.respawn_flags = list(zip(*[self._handle_respawn(TS) for TS in self.TSs]))

        # update COLREG scenarios
        self._set_COLREGs()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        if self.N_TSs_increasing:
            self.outer_step_cnt += 1

        # trajectory plotting
        if self.plot_traj:

            # agent update
            self.OS_traj_N.append(self.OS.eta[0])
            self.OS_traj_E.append(self.OS.eta[1])
            self.OS_traj_h.append(self.OS.eta[2])

            if self.N_TSs > 0:

                # check whether we had a collision
                for TS in self.TSs:
                    D = self._get_ship_domain(self.OS, TS)
                    if ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], sqrt=True) <= D:
                        self.OS_col_N.append(self.OS.eta[0])
                        self.OS_col_E.append(self.OS.eta[1])
                        break

                # check TS respawning
                for TS_idx, flag in enumerate(self.respawn_flags):
                    if flag:
                        self.TS_spawn_steps[TS_idx].append(self.step_cnt)

                # TS update
                for TS_idx, TS in enumerate(self.TSs):
                    self.TS_traj_N[TS_idx].append(TS.eta[0])
                    self.TS_traj_E[TS_idx].append(TS.eta[1])
                    self.TS_traj_h[TS_idx].append(TS.eta[2])

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
       
        return self.state, self.r, d, {}


    def _handle_respawn(self, TS):
        """Handles respawning of a vessel due to being too far away from the agent.

        Args:
            TS (KVLCC2): Vessel of interest.
        Returns
            KVLCC2, respawn_flag (bool)
        """
        if self.spawn_mode == "center":
            return TS, False

        # check whether spawning is still considered
        if ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"], sqrt=True) > self.stop_spawn_dist:

            TCPA_TS = tcpa(NOS=self.OS.eta[0], EOS=self.OS.eta[1], NTS=TS.eta[0], ETS=TS.eta[1],
                        chiOS=self.OS._get_course(), chiTS=TS._get_course(), VOS=self.OS._get_V(), VTS=TS._get_V())
            
            if TCPA_TS < -0.1*self.TCPA_crit or TCPA_TS > 1.25*self.TCPA_crit:
                return self._get_TS(), True
        return TS, False


    def _calculate_reward(self):
        """Returns reward of the current state."""

        N0, E0, head0 = self.OS.eta

        # --------------- Path planning reward (Xu et al. 2022 in Neurocomputing, Ocean Eng.) -----------
        # Distance reward
        OS_goal_ED       = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        r_dist           = (self.OS_goal_old - OS_goal_ED) / 20.0
        self.OS_goal_old = OS_goal_ED

        # Heading reward
        r_head = -abs(angle_to_pi(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0))) / math.pi

        # --------------------------------- 3./4. Collision/COLREG reward --------------------------------
        r_coll = 0
        r_COLREG = 0

        for TS_idx, TS in enumerate(self.TSs):

            # get ED
            ED_TS = ED(N0=N0, E0=E0, N1=TS.eta[0], E1=TS.eta[1])

            # reward based on collision risk
            CR = self._get_CR(OS=self.OS, TS=TS)
            if CR == 1.0:
                r_coll -= 10.0
            else:
                r_coll -= CR

            # COLREG: if vessel just spawned, don't assess COLREG reward
            if not self.respawn_flags[TS_idx]:

                # evaluate TS if in sight and has positive TCPA
                if ED_TS <= self.sight and tcpa(NOS=N0, EOS=E0, NTS=TS.eta[0], ETS=TS.eta[1],\
                     chiOS=self.OS._get_course(), chiTS=TS._get_course(), VOS=self.OS._get_V(), VTS=TS._get_V()) >= 0.0:

                    # steer to the right in Head-on and starboard crossing situations
                    if self.TS_COLREGs_old[TS_idx] in [1, 2] and self.OS.nu[2] < 0.0:
                        r_COLREG -= 1.0

        # --------------------------------- 5. Comfort penalty --------------------------------
        r_comf = -(self.OS.nu[2]/0.1)**2

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
        
        # plot trajectory
        if self.plot_traj and d:

            # show the plot for MMG training env, only return axis in Imazu problems
            if "MMG_Env" == self.__class__.__name__:
                self.plot_traj_fnc()
        return d


    def plot_traj_fnc(self, ax=None, sit=None, r_dist=None, r_head=None, r_coll=None, r_COLREG=None, star=False):
        """Creates the final trajectory plot."""
        if ax is None:
            _, ax = plt.subplots()
            show = True
        else:
            show = False

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

        ax.scatter(self.goal["E"], self.goal["N"])
        
        # OS trajectory
        ax.plot(self.OS_traj_E, self.OS_traj_N, color='black')

        # triangle at beginning
        rec = self._get_triangle(E = self.OS_traj_E[0], N = self.OS_traj_N[0], l=self.OS.Lpp, heading = self.OS_traj_h[0],\
                facecolor="white", edgecolor="black", linewidth=1.5, zorder=10)
        ax.add_patch(rec)

        # OS markers
        OS_traj_E_m = [ele for idx, ele in enumerate(self.OS_traj_E) if idx % self.plot_every_step == 0]
        OS_traj_N_m = [ele for idx, ele in enumerate(self.OS_traj_N) if idx % self.plot_every_step == 0]
        ax.scatter(OS_traj_E_m, OS_traj_N_m, color="black", s=5)

        # OS collision
        if len(self.OS_col_N) > 0:
            ax.scatter(self.OS_col_E, self.OS_col_N, color="red", s=5)

        # TS
        for TS_idx in range(self.N_TSs):

            col = COLREG_COLORS[TS_idx]

            # add final step cnt since we finish here
            spawn_steps = self.TS_spawn_steps[TS_idx]
            if spawn_steps[-1] != self.step_cnt:
                spawn_steps.append(self.step_cnt)

            # trajectories
            if len(spawn_steps) == 1:
                ax.plot(self.TS_traj_E[TS_idx], self.TS_traj_N[TS_idx])
            else:
                for step_idx in range(len(spawn_steps)-1):
                    start = spawn_steps[step_idx]
                    end   = spawn_steps[step_idx+1]

                    E_traj = self.TS_traj_E[TS_idx][start:end]
                    N_traj = self.TS_traj_N[TS_idx][start:end]

                    # triangle at beginning
                    rec = self._get_triangle(E = E_traj[0], N = N_traj[0], l=self.OS.Lpp, heading = self.TS_traj_h[TS_idx][start],\
                         facecolor="white", edgecolor=col, linewidth=1.5, zorder=10)
                    ax.add_patch(rec)

                    # trajectory
                    ax.plot(E_traj, N_traj, color=col)

            # markers
            TS_traj_E_m = [ele for idx, ele in enumerate(self.TS_traj_E[TS_idx]) if idx % self.plot_every_step == 0]
            TS_traj_N_m = [ele for idx, ele in enumerate(self.TS_traj_N[TS_idx]) if idx % self.plot_every_step == 0]
            ax.scatter(TS_traj_E_m, TS_traj_N_m, color=col, s=5)

        # goal
        if not star:
            circ = patches.Circle((self.goal["E"], self.goal["N"]), radius=self.goal_reach_dist, edgecolor='blue', facecolor='none', alpha=0.3)
            ax.add_patch(circ)
        else:
            for i in range(len(self.agents)):
                if i == 0:
                    col = "black"
                else:
                    col = COLREG_COLORS[i-1]
                circ = patches.Circle((self.goals[i]["E"], self.goals[i]["N"]), radius=self.goal_reach_dist, edgecolor=col, facecolor='none', alpha=0.4)
                ax.add_patch(circ)

        if show:
            plt.show()
        else:
            ax.grid(linewidth=1.0, alpha=0.425)

            if not star:

                if all([ele is not None for ele in [r_dist, r_head, r_coll, r_COLREG]]):
                    ax.text(NM_to_meter(0.5), NM_to_meter(11.5), r"$r_{\rm dist}$: " + format(r_dist, '.2f'), fontdict={"fontsize" : 7})
                    ax.text(NM_to_meter(0.5), NM_to_meter(10.5), r"$r_{\rm head}$: " + format(r_head, '.2f'), fontdict={"fontsize" : 7})
                    ax.text(NM_to_meter(0.5), NM_to_meter(9.5), r"$r_{\rm coll}$: " + format(r_coll, '.2f'), fontdict={"fontsize" : 7})
                    ax.text(NM_to_meter(0.5), NM_to_meter(8.5),  r"$r_{\rm COLR}$: " + format(r_COLREG, '.2f'), fontdict={"fontsize" : 7})
                    ax.text(NM_to_meter(0.5), NM_to_meter(7.2),  r"$\sum r$: " + format(r_dist + r_head + r_coll + r_COLREG, '.2f'), fontdict={"fontsize" : 7})
                
                ax.text(NM_to_meter(0.5), NM_to_meter(12.5), f"Case: {sit}", fontdict={"fontsize" : 7})

                if sit not in [18, 19, 20, 21, 22, 23]:
                    ax.tick_params(axis='x', labelsize=8, which='both', bottom=False, top=False, labelbottom=False)
                    ax.set_xlabel("")
                else:
                    ax.tick_params(axis='x', labelsize=8)

                if sit not in [1, 7, 13, 19]:
                    ax.tick_params(axis='y', labelsize=8, which='both', left=False, right=False, labelleft=False)
                    ax.set_ylabel("")
                else:
                    ax.tick_params(axis='y', labelsize=8)
            
            return ax


    def _get_CR(self, OS, TS):
        """Computes the collision risk metric similar to Chun et al. (2021)."""
        # quick access
        N0, E0, _ = OS.eta
        N1, E1, _ = TS.eta
        D = self._get_ship_domain(OS, TS)
        CR_dist = self.CR_dist_multiple * D

        # check if already in ship domain
        if ED(N0=N0, E0=E0, N1=N1, E1=E1, sqrt=True) <= D:
            return 1.0

        # compute speeds and courses
        VOS = OS._get_V()
        VTS = TS._get_V()
        chiOS = OS._get_course()
        chiTS = TS._get_course()

        # compute relative speed
        vxOS, vyOS = xy_from_polar(r=VOS, angle=chiOS)
        vxTS, vyTS = xy_from_polar(r=VTS, angle=chiTS)
        VR = math.sqrt((vyTS - vyOS)**2 + (vxTS - vxOS)**2)

        # compute CPA measures under the assumption that agent is at ship domain border in the direction of the TS
        bng_absolute = bng_abs(N0=N0, E0=E0, N1=N1, E1=E1)
        E_add, N_add = xy_from_polar(r=D, angle=bng_absolute)

        DCPA, TCPA = cpa(NOS=N0+N_add, EOS=E0+E_add, NTS=N1, ETS=E1, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
        
        if self.CR_zero_negTCPA:
            if TCPA < 0:
                return 0.0

            return min([1.0, math.exp((DCPA + VR * TCPA) * math.log(self.CR_al) / CR_dist)])

        else:
            if TCPA >= 0:
                cr = math.exp((DCPA + VR * TCPA) * math.log(self.CR_al) / CR_dist)
            else:
                cr = math.exp((DCPA + VR * 5 * abs(TCPA)) * math.log(self.CR_al) / CR_dist)
            return min([1.0, cr])


    def _get_ship_domain(self, OS, TS, ang=None):
        """Computes a ship domain for the OS with respect to TS following Chun et al. (2021, Ocean Engineering).
        Args:
            OS: KVLCC2
            TS: KVLCC2"""

        # relative bearing
        if ang is None:
            ang = bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], head0=OS.eta[2])

        # ellipsis
        if 0 <= rtd(ang) < 90:
            a = self.OS.ship_domain_D
            b = self.OS.ship_domain_A

        elif 90 <= rtd(ang) < 180:
            ang = dtr(180) - ang
            a = self.OS.ship_domain_D
            b = self.OS.ship_domain_C

        elif 180 <= rtd(ang) < 270:
            ang = ang - dtr(180)
            a = self.OS.ship_domain_B
            b = self.OS.ship_domain_C

        else:
            ang = dtr(360) - ang
            a = self.OS.ship_domain_B
            b = self.OS.ship_domain_A

        return ((math.sin(ang) / a)**2 + (math.cos(ang) / b)**2)**(-0.5)


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
        NTS, ETS, psi_TS = TS.eta
        V_OS  = OS._get_V()
        V_TS  = TS._get_V()

        chiOS = OS._get_course()
        chiTS = TS._get_course()

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


    def __str__(self) -> str:
        N0, E0, head0 = self.OS.eta
        u, v, r = np.round(self.OS.nu,3)

        ste = f"Step: {self.step_cnt}"
        pos = f"N: {np.round(meter_to_NM(N0),3) - 7}, E: {np.round(meter_to_NM(E0),3) - 7}, " + r"$\psi$: " + f"{np.round(rtd(head0), 3)}°"
        vel = f"u: {u}, v: {v}, r: {r}"
        return ste + "\n" + pos + "\n" + vel


    def _rotate_point(self, x, y, cx, cy, angle):
        """Rotates a point (x,y) around origin (cx,cy) by an angle (defined counter-clockwise with zero at y-axis)."""

        # translate point to origin
        tempX = x - cx
        tempY = y - cy

        # apply rotation
        rotatedX = tempX * math.cos(angle) - tempY * math.sin(angle)
        rotatedY = tempX * math.sin(angle) + tempY * math.cos(angle)

        # translate back
        return rotatedX + cx, rotatedY + cy

    def _get_rect(self, E, N, width, length, heading, **kwargs):
        """Returns a patches.rectangle object. heading in rad."""

        # quick access
        x = E - width/2
        y = N - length/2, 
        cx = E
        cy = N
        heading = -heading   # negate since our heading is defined clockwise, contrary to plt rotations

        E0, N0 = self._rotate_point(x=x, y=y, cx=cx, cy=cy, angle=heading)

        # create rect
        return patches.Rectangle((E0, N0), width, length, rtd(heading), **kwargs)


    def _get_triangle(self, E, N, l, heading, **kwargs):
        """Returns a patches.polygon object. heading in rad."""

        # quick access 
        cx = E
        cy = N
        heading = -heading   # negate since our heading is defined clockwise, contrary to plt rotations

        topx = cx
        topy = cy + 2*l

        rightx = cx + l
        righty = cy - l

        leftx = cx - l
        lefty = cy - l

        topE, topN     = self._rotate_point(x=topx, y=topy, cx=cx, cy=cy, angle=heading)
        rightE, rightN = self._rotate_point(x=rightx, y=righty, cx=cx, cy=cy, angle=heading)
        leftE, leftN   = self._rotate_point(x=leftx, y=lefty, cx=cx, cy=cy, angle=heading)

        # create rect
        return patches.Polygon(xy=np.array([[topE, topN], [rightE, rightN], [leftE, leftN]]), **kwargs)


    def _plot_jet(self, axis, E, N, l, angle, **kwargs):
        """Adds a line to an axis (plt-object) originating at (E,N), having a given length l, 
           and following the angle (in rad). Returns the new axis."""

        # transform angle in [0, 2pi)
        angle = angle_to_2pi(angle)

        # 1. Quadrant
        if angle <= math.pi/2:
            E1 = E + math.sin(angle) * l
            N1 = N + math.cos(angle) * l
        
        # 2. Quadrant
        elif 3/2 *math.pi < angle <= 2*math.pi:
            angle = 2*math.pi - angle

            E1 = E - math.sin(angle) * l
            N1 = N + math.cos(angle) * l

        # 3. Quadrant
        elif math.pi < angle <= 3/2*math.pi:
            angle -= math.pi

            E1 = E - math.sin(angle) * l
            N1 = N - math.cos(angle) * l

        # 4. Quadrant
        elif math.pi/2 < angle <= math.pi:
            angle = math.pi - angle

            E1 = E + math.sin(angle) * l
            N1 = N - math.cos(angle) * l
        
        # draw on axis
        axis.plot([E, E1], [N, N1], **kwargs)
        return axis


    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep (except we only want trajectory)
        if not self.plot_traj:

            if self.step_cnt % 2 == 0: 

                # check whether figure has been initialized
                if len(plt.get_fignums()) == 0:
                    self.fig = plt.figure(figsize=(10, 7))
                    self.gs  = self.fig.add_gridspec(2, 2)
                    self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                    self.ax1 = self.fig.add_subplot(self.gs[0, 1]) # reward
                    self.ax2 = self.fig.add_subplot(self.gs[1, 0]) # state
                    self.ax3 = self.fig.add_subplot(self.gs[1, 1]) # action

                    self.fig2 = plt.figure(figsize=(10,7))
                    self.fig2_ax = self.fig2.add_subplot(111)

                    plt.ion()
                    plt.show()
                
                # ------------------------------ ship movement --------------------------------
                for ax in [self.ax0, self.fig2_ax]:
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
                    rect = self._get_rect(E = E0, N = N0, width = self.OS.B, length = self.OS.Lpp, heading = head0,
                                        linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
                    
                    # step information
                    ax.text(0.05 * self.E_max, 0.9 * self.N_max, self.__str__(), fontsize=8)

                    # ship domain
                    xys = [self._rotate_point(E0 + x, N0 + y, cx=E0, cy=N0, angle=-head0) for x, y in zip(self.domain_plot_xs, self.domain_plot_ys)]
                    xs = [xy[0] for xy in xys]
                    ys = [xy[1] for xy in xys]
                    ax.plot(xs, ys, color="black", alpha=0.7)

                    # collision risk distance
                    xys = [self._rotate_point(E0 + (1 + self.CR_dist_multiple) * x, N0 + (1 + self.CR_dist_multiple) * y, cx=E0, cy=N0, angle=-head0) for x, y in zip(self.domain_plot_xs, self.domain_plot_ys)]
                    xs = [xy[0] for xy in xys]
                    ys = [xy[1] for xy in xys]
                    ax.plot(xs, ys, color="black", alpha=0.3)

                    # add jets according to COLREGS
                    for COLREG_deg in [5, 355]:
                        ax = self._plot_jet(axis = ax, E=E0, N=N0, l = self.sight, 
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
                        N, E, headTS = TS.eta
                        chiTS = TS._get_course()
                        VTS = TS._get_V()

                        # determine color according to COLREG scenario
                        COLREG = self._get_COLREG_situation(OS=self.OS, TS=TS)
                        col = COLREG_COLORS[COLREG]

                        # place TS
                        rect = self._get_rect(E = E, N = N, width = TS.B, length = TS.Lpp, heading = headTS,
                                            linewidth=1, edgecolor=col, facecolor='none', label=COLREG_NAMES[COLREG])
                        ax.add_patch(rect)

                        # add two jets according to COLREGS
                        for COLREG_deg in [5, 355]:
                            ax= self._plot_jet(axis = ax, E=E, N=N, l = self.sight, 
                                            angle = headTS + dtr(COLREG_deg), color=col, alpha=0.3)

                        # collision risk                        
                        CR = self._get_CR(OS=self.OS, TS=TS)
                        ax.text(E + 800, N-600, f"CR: {np.round(CR, 4)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        
                        D = self._get_ship_domain(OS=self.OS, TS=TS)
                        ax.text(E + 800, N-1000, f"D: {np.round(D, 4)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)

                        bng_absolute = bng_abs(N0=N0, E0=E0, N1=TS.eta[0], E1=TS.eta[1])
                        E_add, N_add = xy_from_polar(r=D, angle=bng_absolute)

                        DCPA_TS, TCPA_TS = cpa(NOS=N0+N_add, EOS=E0+E_add, NTS=TS.eta[0], ETS=TS.eta[1], \
                            chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
                        ax.scatter(E0+E_add, N0+N_add, color=col, s=10)

                        ax.text(E + 800, N + 200, f"TCPA: {np.round(TCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        ax.text(E + 800, N-200, f"DCPA: {np.round(DCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)

                    # set legend for COLREGS
                    ax.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], fontsize=8,
                                    loc='lower center', bbox_to_anchor=(0.52, 1.0), fancybox=False, shadow=False, ncol=6).get_frame().set_linewidth(0.0)


                # ------------------------------ reward plot --------------------------------
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

                    for i in range(8):
                        self.ax2.plot([self.ax2.old_time, self.step_cnt], [self.ax2.old_state[i], self.state[i]], 
                                    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i], 
                                    label=self.state_names[i])    
                    if self.step_cnt == 0:
                        self.ax2.legend()

                    self.ax2.old_time = self.step_cnt
                    self.ax2.old_state = self.state

                # ------------------------------ action plot --------------------------------
                plt.pause(0.001)
