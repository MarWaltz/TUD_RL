import copy
import math
import random

import gym
import matplotlib.patches as patches
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from tud_rl.envs._envs.VesselFnc import (COLREG_COLORS, COLREG_NAMES, ED,
                                         angle_to_2pi, angle_to_pi, bng_abs,
                                         bng_rel, dcpa, dtr, head_inter,
                                         polar_from_xy, project_vector, rtd,
                                         tcpa, xy_from_polar)
from tud_rl.envs._envs.MMG_KVLCC2 import KVLCC2


class MMG_Env(gym.Env):
    """This environment contains an agent steering a KVLCC2."""

    def __init__(self, 
                 N_TSs_max        = 3, 
                 N_TSs_random     = False, 
                 N_TSs_increasing = False,
                 state_design     = "RecDQN", 
                 plot_traj        = True,
                 w_dist           = 1.0,
                 w_head           = 1.0,
                 w_coll           = 1.0,
                 w_COLREG         = 1.0,
                 w_comf           = 1.0,
                 COLREG_def       = "line"):
        super().__init__()

        # simulation settings
        self.delta_t = 3.0                           # simulation time interval (in s)
        self.N_max   = 15_000                        # maximum N-coordinate (in m)
        self.E_max   = 15_000                        # maximum E-coordinate (in m)

        self.N_TSs_max    = N_TSs_max                  # maximum number of other vessels
        self.N_TSs_random = N_TSs_random               # if true, samples a random number in [0, N_TSs] at start of each episode
                                                       # if false, always have N_TSs_max
        self.N_TSs_increasing = N_TSs_increasing       # if true, have schedule for number TSs
                                                       # if false, always have N_TSs_max

        assert sum([N_TSs_random, N_TSs_increasing]) <= 1, "Either random number of TS or schedule, not both."
        if self.N_TSs_increasing:
            self.outer_step_cnt = 0

        self.sight             = 5_000                 # sight of the agent (in m)
        self.coll_dist         = 320                   # collision distance (in m, five times ship length)
        self.CR_al             = 0.1                   # collision risk metric when TS enters sight of agent
        self.TCPA_crit         = 15 * 60               # critical TCPA (in s), relevant for state and spawning of TSs
        self.min_dist_spawn_TS = 5 * 320               # minimum distance of a spawning vessel to other TSs (in m)

        assert state_design in ["maxRisk", "RecDQN"], "Unknown state design for FossenEnv. Should be 'maxRisk' or 'RecDQN'."
        self.state_design = state_design

        self.goal_reach_dist = 320                        # euclidean distance (in m) at which goal is considered as reached
        self.stop_spawn_dist = 2_500                      # euclidean distance (in m) under which vessels do not spawn anymore

        self.num_obs_OS = 8                               # number of observations for the OS
        self.num_obs_TS = 6                               # number of observations per TS

        self.plot_traj = plot_traj       # whether to plot trajectory after termination

        assert COLREG_def in ["correct", "line"], "Unknown COLREG definition."
        self.COLREG_def = COLREG_def

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
        self._max_episode_steps = 1000
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

        # sample goal and agent positions
        sit_init = np.random.choice([0, 1, 2, 3])

        # init goal
        if sit_init == 0:
            self.goal = {"N" : 0.9 * self.N_max, "E" : 0.5 * self.E_max}
            N_init = 0.1 * self.N_max
            E_init = 0.5 * self.E_max
            head   = angle_to_2pi(dtr(np.random.uniform(-10, 10)))
        
        elif sit_init == 1:
            self.goal = {"N" : 0.5 * self.N_max, "E" : 0.9 * self.E_max}
            N_init = 0.5 * self.N_max
            E_init = 0.1 * self.E_max
            head   = dtr(np.random.uniform(35, 55))

        elif sit_init == 2:
            self.goal = {"N" : 0.1 * self.N_max, "E" : 0.5 * self.E_max}
            N_init = 0.9 * self.N_max
            E_init = 0.5 * self.E_max
            head   = dtr(np.random.uniform(170, 190))

        elif sit_init == 3:
            self.goal = {"N" : 0.5 * self.N_max, "E" : 0.1 * self.E_max}
            N_init = 0.5 * self.N_max
            E_init = 0.9 * self.E_max
            head   = dtr(np.random.uniform(260, 280))

        # init agent (OS for 'Own Ship')
        self.OS = KVLCC2(N_init   = N_init, 
                         E_init   = E_init, 
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
            self.N_TSs = np.random.choice(self.N_TSs_max + 1)

        elif self.N_TSs_increasing:

            if self.outer_step_cnt <= 1e6:
                self.N_TSs = 0
            elif self.outer_step_cnt <= 2e6:
                self.N_TSs = 1
            elif self.outer_step_cnt <= 3e6:
                self.N_TSs = 2
            else:
                self.N_TSs = 3
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

            self.TS_traj_N = [[] for _ in range(self.N_TSs)]
            self.TS_traj_E = [[] for _ in range(self.N_TSs)]
            self.TS_ptr = [i for i in range(self.N_TSs)]

            for TS_idx, TS in enumerate(self.TSs):
                ptr = self.TS_ptr[TS_idx]                
                self.TS_traj_N[ptr].append(TS.eta[0])
                self.TS_traj_E[ptr].append(TS.eta[1])

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
                        psi_init = np.random.uniform(0, 2*np.pi),
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

            # quick access for OS
            N0, E0, _ = self.OS.eta
            chiOS = self.OS._get_course()
            VOS   = self.OS._get_V()

            # sample COLREG situation 
            # head-on = 1, starb. cross. = 2, ports. cross. = 3, overtaking = 4
            COLREG_s = random.choice([1, 2, 3, 4])

            #--------------------------------------- line mode --------------------------------------

            # determine relative speed of OS towards goal, need absolute bearing first
            bng_abs_goal = bng_abs(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])

            # project VOS vector on relative velocity direction
            VR_goal_x, VR_goal_y = project_vector(VA=VOS, angleA=chiOS, VB=1, angleB=bng_abs_goal)
            
            # sample time
            t_hit = np.random.uniform(self.TCPA_crit * 0.5, self.TCPA_crit)

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
                VTS = np.random.uniform(0, V_max_TS)
                TS.nu[0] = VTS

                # set nps of TS so that it will keep this velocity
                TS.nps = TS._get_nps_from_u(VTS, psi=TS.eta[2])

            # backtrace original position of TS
            E_TS = E_hit - VTS * math.sin(head_TS_s) * t_hit
            N_TS = N_hit - VTS * math.cos(head_TS_s) * t_hit

            # set positional values
            TS.eta = np.array([N_TS, E_TS, head_TS_s], dtype=np.float32)
            
            # TS should spawn outside the sight of the agent
            #if ED(N0=N0, E0=E0, N1=N_TS, E1=E_TS, sqrt=True) > self.sight:

            # no TS yet there
            if len(self.TSs) == 0:
                break

            # TS shouldn't spawn too close to other TS
            if min([ED(N0=N_TS, E0=E_TS, N1=TS_there.eta[0], E1=TS_there.eta[1], sqrt=True) for TS_there in self.TSs])\
                >= self.min_dist_spawn_TS:
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
            heading
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
        cmp1 = self.OS.nu
        cmp2 = np.array([angle_to_pi(head0) / (np.pi),                 # heading
                         self.OS.nu_dot[2],                            # r_dot
                         self.OS.rud_angle / self.OS.rud_angle_max])   # tau component
        state_OS = np.concatenate([cmp1, cmp2])


        #------------------------------ goal related ---------------------------------
        OS_goal_ED = ED(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"])
        state_goal = np.array([angle_to_pi(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0)) / (np.pi), 
                               OS_goal_ED / self.E_max])


        #--------------------------- dynamic obstacle related -------------------------
        state_TSs = []

        for TS_idx, TS in enumerate(self.TSs):

            N, E, headTS = TS.eta

            # consider TS if it is in sight
            ED_OS_TS = ED(N0=N0, E0=E0, N1=N, E1=E, sqrt=True)

            if ED_OS_TS <= self.sight:

                # euclidean distance
                ED_OS_TS_norm = ED_OS_TS / self.sight

                # relative bearing
                bng_rel_TS = angle_to_pi(bng_rel(N0=N0, E0=E0, N1=N, E1=E, head0=head0)) / (np.pi)

                # heading intersection angle
                C_TS = angle_to_pi(head_inter(head_OS=head0, head_TS=headTS)) / (np.pi)

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
            CR_ghost = 0.1

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
            self.TSs, self.respawn_flags = list(zip(*[self._handle_respawn(TS, respawn=True) for TS in self.TSs]))

        # trajectory plotting
        if self.plot_traj:

            # agent update
            self.OS_traj_N.append(self.OS.eta[0])
            self.OS_traj_E.append(self.OS.eta[1])

            # check TS respawning
            if self.N_TSs > 0:
                for TS_idx, flag in enumerate(self.respawn_flags):
                    if flag:
                        # add new trajectory slot
                        self.TS_traj_N.append([])
                        self.TS_traj_E.append([])

                        # update pointer
                        self.TS_ptr[TS_idx] = max(self.TS_ptr) + 1

                # TS update
                for TS_idx, TS in enumerate(self.TSs):
                    ptr = self.TS_ptr[TS_idx]
                    self.TS_traj_N[ptr].append(TS.eta[0])
                    self.TS_traj_E[ptr].append(TS.eta[1])

        # update COLREG scenarios
        self._set_COLREGs()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        if self.N_TSs_increasing:
            self.outer_step_cnt += 1

        # compute state, reward, done        
        self._set_state()
        self._calculate_reward()
        d = self._done()
       
        return self.state, self.r, d, {}


    def _handle_respawn(self, TS, respawn=True, mirrow=False, clip=False):
        """Handles respawning of a vessel. Considers two situations:
        1) Respawn due to leaving of the simulation map.
        2) Respawn due to being too far away from the agent.

        Args:
            TS (KVLCC2): Vessel of interest.
            respawn (bool):   For 1) Whether the vessel should respawn somewhere else.
            mirrow (bool):    For 1) Whether the vessel should by mirrowed if it hits the boundary of the simulation area. 
                                     Inspired by Xu et al. (2022, Neurocomputing).
            clip (bool):      For 1) Whether to artificially keep vessel on the map by clipping. Thus, it will stay on boarder.
        Returns
            KVLCC2, respawn_flag (bool)
        """
        assert sum([respawn, mirrow, clip]) <= 1, "Can choose either 'respawn', 'mirrow', or 'clip', not a combination."

        # check whether spawning is still considered
        if ED(N0=self.OS.eta[0], E0=self.OS.eta[1], N1=self.goal["N"], E1=self.goal["E"], sqrt=True) > self.stop_spawn_dist:

            # 1) leaving of simulation area
            if TS._is_off_map():
                
                if respawn:
                    return self._get_TS(), True
                
                elif mirrow:
                    # quick access
                    psi = TS.eta[2]

                    # right or left bound (E-axis)
                    if TS.eta[1] <= 0 or TS.eta[1] >= TS.E_max:
                        TS.eta[2] = 2*np.pi - psi
                    
                    # upper and lower bound (N-axis)
                    else:
                        TS.eta[2] = np.pi - psi
                
                elif clip:
                    TS.eta[0] = np.clip(TS.eta[0], 0, TS.N_max)
                    TS.eta[1] = np.clip(TS.eta[1], 0, TS.E_max)

                return TS, False
            
            # 2) too far away from agent
            else:
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
        r_head = -np.abs(angle_to_pi(bng_rel(N0=N0, E0=E0, N1=self.goal["N"], E1=self.goal["E"], head0=head0))) / np.pi

        # --------------------------------- 3./4. Collision/COLREG reward --------------------------------
        r_coll = 0
        r_COLREG = 0

        for TS_idx, TS in enumerate(self.TSs):

            # get ED
            ED_TS = ED(N0=N0, E0=E0, N1=TS.eta[0], E1=TS.eta[1])

            # reward based on collision risk
            CR = self._get_CR(OS=self.OS, TS=TS)
            if CR == 1.0:
                r_coll -= 10
            elif CR > 0.1:
                r_coll -= 100/81 * (CR - 0.1)**2

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

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            
            ax.plot(self.OS_traj_E, self.OS_traj_N, color='black')

            for idx in range(len(self.TS_traj_E)):
                ax.plot(self.TS_traj_E[idx], self.TS_traj_N[idx])

            circ = patches.Circle((self.goal["E"], self.goal["N"]), radius=self.goal_reach_dist, edgecolor='blue', facecolor='none', alpha=0.3)
            ax.add_patch(circ)

            ax.set_xlim(0, self.E_max)
            ax.set_ylim(0, self.N_max)
            ax.scatter(self.goal["E"], self.goal["N"])
            plt.show()

        return d


    def _get_CR(self, OS, TS):
        """Computes the collision risk metric similar to Chun et al. (2021)."""

        S = self.sight
        D = self._get_ship_domain(OS, TS)

        # check if already in ship domain
        if ED(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], sqrt=True) <= D:
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

        # CPA measures
        TCPA = tcpa(NOS=OS.eta[0], EOS=OS.eta[1], NTS=TS.eta[0], ETS=TS.eta[1], chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
        DCPA = dcpa(NOS=OS.eta[0], EOS=OS.eta[1], NTS=TS.eta[0], ETS=TS.eta[1], chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)

        frac = math.log(self.CR_al) / (S-D)
        return min([1.0, math.exp(frac * (DCPA + VR * abs(TCPA) - D))])


    def _get_ship_domain(self, OS, TS, ang=None):
        """Computes a ship domain for the OS with respect to TS following Chun et al. (2021, Ocean Engineering).
        Args:
            OS: KVLCC2
            TS: KVLCC2"""

        # relative bearing
        if ang is None:
            ang = bng_rel(N0=OS.eta[0], E0=OS.eta[1], N1=TS.eta[0], E1=TS.eta[1], head0=OS.eta[2])

        # circle parts
        if 0 <= rtd(ang) < 90:
            return self.OS.ship_domain_AB
        elif 180 <= rtd(ang) < 270:
            return self.OS.ship_domain_CD

        # ellipsis
        elif 90 <= rtd(ang) < 180:
            ang -= dtr(90)
            a = self.OS.ship_domain_AB
            b = self.OS.ship_domain_CD
        else:
            ang -= dtr(270)
            a = self.OS.ship_domain_CD
            b = self.OS.ship_domain_AB
        return ((math.cos(ang) / a)**2 + (math.sin(ang) / b)**2)**(-0.5)
    

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
        ste = f"Step: {self.step_cnt}"
        pos = f"N: {np.round(self.OS.eta[0], 3)}, E: {np.round(self.OS.eta[1], 3)}, " + r"$\psi$: " + f"{np.round(rtd(self.OS.eta[2]), 3)}°"
        vel = f"u: {np.round(self.OS.nu[0], 3)}, v: {np.round(self.OS.nu[1], 3)}, r: {np.round(self.OS.nu[2], 3)}"
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


    def _plot_jet(self, axis, E, N, l, angle, **kwargs):
        """Adds a line to an axis (plt-object) originating at (E,N), having a given length l, 
           and following the angle (in rad). Returns the new axis."""

        # transform angle in [0, 2pi)
        angle = angle_to_2pi(angle)

        # 1. Quadrant
        if angle <= np.pi/2:
            E1 = E + math.sin(angle) * l
            N1 = N + math.cos(angle) * l
        
        # 2. Quadrant
        elif 3/2 *np.pi < angle <= 2*np.pi:
            angle = 2*np.pi - angle

            E1 = E - math.sin(angle) * l
            N1 = N + math.cos(angle) * l

        # 3. Quadrant
        elif np.pi < angle <= 3/2*np.pi:
            angle -= np.pi

            E1 = E - math.sin(angle) * l
            N1 = N - math.cos(angle) * l

        # 4. Quadrant
        elif np.pi/2 < angle <= np.pi:
            angle = np.pi - angle

            E1 = E + math.sin(angle) * l
            N1 = N - math.cos(angle) * l
        
        # draw on axis
        axis.plot([E, E1], [N, N1], **kwargs)
        return axis


    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep (except we only want trajectory)
        if not self.plot_traj:

            if self.step_cnt % 1 == 0: 

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
                    ax.set_xlim(-5, self.E_max + 5)
                    ax.set_ylim(-5, self.N_max + 5)
                    ax.set_xlabel("East")
                    ax.set_ylabel("North")

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
                        TCPA_TS = tcpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
                        ax.text(E + 800, N + 200, f"TCPA: {np.round(TCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)

                        DCPA_TS = dcpa(NOS=N0, EOS=E0, NTS=N, ETS=E, chiOS=chiOS, chiTS=chiTS, VOS=VOS, VTS=VTS)
                        ax.text(E + 800, N-200, f"DCPA: {np.round(DCPA_TS, 2)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        
                        CR = self._get_CR(OS=self.OS, TS=TS)
                        ax.text(E + 800, N-600, f"CR: {np.round(CR, 4)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)
                        
                        D = self._get_ship_domain(OS=self.OS, TS=TS)
                        ax.text(E + 800, N-1000, f"D: {np.round(D, 4)}", fontsize=7,
                                    horizontalalignment='center', verticalalignment='center', color=col)

                    # set legend for COLREGS
                    ax.legend(handles=[patches.Patch(color=COLREG_COLORS[i], label=COLREG_NAMES[i]) for i in range(5)], fontsize=8,
                                    loc='lower center', bbox_to_anchor=(0.6, 1.0), fancybox=False, shadow=False, ncol=6).get_frame().set_linewidth(0.0)


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