from tud_rl.envs._envs.MMG_Env import *
from tud_rl.envs._envs.VesselPlots import get_rect

class MMG_Star(MMG_Env):
    """This environment contains four agents, each steering a KVLCC2."""

    def __init__(self, N_TSs_max=3, state_design="RecDQN"):
        super().__init__(N_TSs_max=N_TSs_max, state_design=state_design, pdf_traj=True, N_TSs_increasing=False, N_TSs_random=False)
        assert N_TSs_max in [3, 7, 15], "Consider either 4, 8, or 16 ships in total."
        self.N_TSs = self.N_TSs_max

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init four agents
        self.agents = []
        for i in range(self.N_TSs_max + 1):
            head = i * 2*math.pi/(self.N_TSs_max+1)
            self.agents.append(KVLCC2(N_init   = 0.0, 
                                      E_init   = 0.0, 
                                      psi_init = head,
                                      u_init   = 0.0,
                                      v_init   = 0.0,
                                      r_init   = 0.0,
                                      delta_t  = self.delta_t,
                                      N_max    = self.N_max,
                                      E_max    = self.E_max,
                                      nps      = 1.8))

        # set longitudinal speed to near-convergence
        # Note: if we don't do this, the TCPA calculation for spawning other vessels is heavily biased
        for agent in self.agents:
            agent.nu[0] = agent._get_u_from_nps(agent.nps, psi=agent.eta[2])

        # backtrace motion
        for agent in self.agents:
            agent.eta[0] = self.CPA_N - agent._get_V() * np.cos(agent.eta[2]) * self.TCPA_crit
            agent.eta[1] = self.CPA_E - agent._get_V() * np.sin(agent.eta[2]) * self.TCPA_crit

        # init four goals
        self.goals = []
        for _, agent in enumerate(self.agents):
            a_N = agent.eta[0]
            a_E = agent.eta[1]

            bng_abs_CPA = bng_abs(N0=a_N, E0=a_E, N1=self.CPA_N, E1=self.CPA_E)
            ED_CPA = ED(N0=a_N, E0=a_E, N1=self.CPA_N, E1=self.CPA_E)

            # project point
            E_add, N_add = xy_from_polar(r=2*ED_CPA, angle=bng_abs_CPA)
            g = {"N" : a_N + N_add, "E" : a_E + E_add}
            self.goals.append(g)

        # determine current COLREG situations
        # Note: We have four agents in this scenario. Thus, treating each as a OS, all have three TSs from their perspective
        #       We need to update all those COLREG modes.
        self.TS_COLREGs_all = [[0] * self.N_TSs] * (self.N_TSs + 1)
        self._set_COLREGs()

        # init aggregated state
        self._set_aggregated_state()
        self.state_init = self.state_agg

        # episode finishing condition
        self.finished = [False] * len(self.agents)

        # we arbitrarily consider the first ship as the OS for plotting
        self.TrajPlotter.reset(OS=self.agents[0], TSs=self.agents[1:], N_TSs=self.N_TSs)

        rads  = np.linspace(0.0, 2*math.pi, 25)
        dists = [get_ship_domain(A=self.agents[0].ship_domain_A, B=self.agents[0].ship_domain_B, C=self.agents[0].ship_domain_C, \
            D=self.agents[0].ship_domain_D,\
            OS=None, TS=None, ang=rad) for rad in rads]
        self.domain_xs = [dist * math.sin(rad) for dist, rad in zip(dists, rads)]
        self.domain_ys = [dist * math.cos(rad) for dist, rad in zip(dists, rads)]

        return self.state_agg


    def _set_aggregated_state(self):
        """Sets the overall state in this scenario."""
        self.state_agg = []

        for idx, agent in enumerate(self.agents):

            # agent acts as OS
            self.OS = agent
            self.TSs = [TS_agent for TS_idx, TS_agent in enumerate(self.agents) if idx != TS_idx]
            self.TS_COLREGs = self.TS_COLREGs_all[idx]
            self.goal = self.goals[idx]
            
            # compute state from this perspective
            self._set_state()
            self.state_agg.append(self.state)


    def _set_COLREGs(self):
        """Computes for each target ship the current COLREG situation and stores it internally.
        Again: Each agent acts as OS and has three TS, respectively, leading to overall 4 * 3 COLREG modes."""

        # overwrite old situations
        self.TS_COLREGs_all_old = copy.deepcopy(self.TS_COLREGs_all)

        # compute new ones
        self.TS_COLREGs_all = []

        for idx, agent in enumerate(self.agents):

            TS_COLREGs_agent = []

            for TS_idx, TS_agent in enumerate(self.agents):

                if idx == TS_idx:
                    continue
                TS_COLREGs_agent.append(self._get_COLREG_situation(OS=agent, TS=TS_agent))

            self.TS_COLREGs_all.append(TS_COLREGs_agent)


    def step(self, a):
        """The action a is a list of four integers, each representing an action for an agent."""

        # perform control action
        [self.agents[idx]._control(act) for idx, act in enumerate(a) if not self.finished[idx]]

        # update dynamics
        [agent._upd_dynamics() for idx, agent in enumerate(self.agents) if not self.finished[idx]]

        # update COLREG scenarios
        self._set_COLREGs()

        # compute state, reward, done        
        self._set_aggregated_state()
        self._calculate_reward()
        d = self._done()

        # increase step cnt and overall simulation time
        self.step_cnt += 1
        self.sim_t += self.delta_t

        # trajectory plotting
        self.TrajPlotter.step(OS=self.agents[0], TSs=self.agents[1:], respawn_flags=[False for _ in range(self.N_TSs)], step_cnt=self.step_cnt)
        
        return self.state_agg, self.r, d, {}


    def _calculate_reward(self):
        """We don't train anymore, just eyetest. No reward needed."""
        self.r = 0


    def _done(self):
        """Returns boolean flag whether episode is over."""

        # check goal reaching
        for idx, agent in enumerate(self.agents):
            
            # already finished
            if self.finished[idx]:
                continue

            # ED to goal
            else:
                OS_goal_ED = ED(N0=agent.eta[0], E0=agent.eta[1], N1=self.goals[idx]["N"], E1=self.goals[idx]["E"])
                if OS_goal_ED <= self.goal_reach_dist:
                    self.finished[idx] = True

        # everyone reached goal
        if all(self.finished):
            return True

        # artificial done signal
        if self.step_cnt >= self._max_episode_steps:
            return True
        return False


    def render(self, mode=None):
        """Renders the current environment. Note: The 'mode' argument is needed since a recent update of the 'gym' package."""

        # plot every nth timestep
        if self.step_cnt % 2 == 0: 

            # check whether figure has been initialized
            if len(plt.get_fignums()) == 0:
                self.fig = plt.figure(figsize=(10, 7))
                self.gs  = self.fig.add_gridspec(1, 1)
                self.ax0 = self.fig.add_subplot(self.gs[0, 0]) # ship
                plt.ion()
                plt.show()
            
            # ------------------------------ ship movement --------------------------------
            # clear prior axes, set limits and add labels and title
            self.ax0.clear()
            self.ax0.set_xlim(-5, self.E_max + 5)
            self.ax0.set_ylim(-5, self.N_max + 5)
            self.ax0.set_xlabel("East")
            self.ax0.set_ylabel("North")

            # set agents and goals
            for idx, agent in enumerate(self.agents):
                
                col = plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
                N0, E0, head0 = agent.eta
               
                rect = get_rect(E = E0, N = N0, width = agent.B, length = agent.Lpp, heading = head0,
                                      linewidth=1, edgecolor=col, facecolor='none')
                self.ax0.add_patch(rect)

                xys = [rotate_point(E0 + x, N0 + y, cx=E0, cy=N0, angle=-head0) for x, y in zip(self.domain_xs, self.domain_ys)]
                xs = [xy[0] for xy in xys]
                ys = [xy[1] for xy in xys]
                self.ax0.plot(xs, ys, color="black", alpha=0.7)

                # add jets according to COLREGS
                #for COLREG_deg in [5, 355]:
                #    self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.sight, 
                #                              angle = head0 + dtr(COLREG_deg), color=col, alpha=0.3)

                self.ax0.scatter(self.goals[idx]["E"], self.goals[idx]["N"], color=col)
                circ = patches.Circle((self.goals[idx]["E"], self.goals[idx]["N"]), radius=self.goal_reach_dist, edgecolor=col, facecolor='none', alpha=0.3)
                self.ax0.add_patch(circ)

            plt.pause(0.001)
