from tud_rl.envs._envs.MMG_Env import *


class MMG_Star(MMG_Env):
    """This environment contains four agents, each steering a KVLCC2."""

    def __init__(self, N_TSs_max=3, state_design="RecDQN"):
        super().__init__(N_TSs_max=N_TSs_max, state_design=state_design, plot_traj=True, N_TSs_increasing=False, N_TSs_random=False)
        self.N_TSs = self.N_TSs_max

    def reset(self):
        """Resets environment to initial state."""

        self.step_cnt = 0           # simulation step counter
        self.sim_t    = 0           # overall passed simulation time (in s)

        # init four agents
        self.agents = []
        for i in range(self.N_TSs_max + 1):
            
            if i == 0:
                head = 0.0

            elif i == 1:
                head = 1/2 * math.pi

            elif i == 2:
                head = math.pi
            
            elif i == 3:
                head = 3/2 * math.pi
            
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
        for a_id, agent in enumerate(self.agents):
            if a_id == 0:
                g = {"N" : self.CPA_N + abs(self.CPA_N - agent.eta[0]), "E" : agent.eta[1]}
            
            elif a_id == 1:
                g = {"N" : agent.eta[0], "E" : self.CPA_E + abs(self.CPA_E - agent.eta[1])}

            elif a_id == 2:
                g = {"N" : self.CPA_N - abs(self.CPA_N - agent.eta[0]), "E" : agent.eta[1]}

            elif a_id == 3:
                g = {"N" : agent.eta[0], "E" : self.CPA_E - abs(self.CPA_E - agent.eta[1])}

            self.goals.append(g)

        # determine current COLREG situations
        # Note: We have four agents in this scenario. Thus, treating each as a OS, all have three TSs from their perspective
        #       We need to update all those COLREG modes.
        self.TS_COLREGs_all = [[0] * 3] * 4
        self._set_COLREGs()

        # init aggregated state
        self._set_aggregated_state()
        self.state_init = self.state_agg

        # episode finishing condition
        self.finished = [False] * len(self.agents)

        # we arbitrarily consider the first ship as the OS for plotting
        if self.plot_traj:
            self.OS_traj_N = [self.agents[0].eta[0]]
            self.OS_traj_E = [self.agents[0].eta[1]]
            self.OS_traj_h = [self.agents[0].eta[2]]

            self.OS_col_N = []

            self.TS_traj_N = [[] for _ in range(self.N_TSs)]
            self.TS_traj_E = [[] for _ in range(self.N_TSs)]
            self.TS_traj_h = [[] for _ in range(self.N_TSs)]

            self.TS_spawn_steps = [[self.step_cnt] for _ in range(self.N_TSs)]
 
            for TS_idx, TS in enumerate(self.agents[1:]):             
                self.TS_traj_N[TS_idx].append(TS.eta[0])
                self.TS_traj_E[TS_idx].append(TS.eta[1])
                self.TS_traj_h[TS_idx].append(TS.eta[2])

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
        if self.plot_traj:

            # agent update
            self.OS_traj_N.append(self.agents[0].eta[0])
            self.OS_traj_E.append(self.agents[0].eta[1])
            self.OS_traj_h.append(self.agents[0].eta[2])

            if self.N_TSs > 0:

                # TS update
                for TS_idx, TS in enumerate(self.agents[1:]):
                    self.TS_traj_N[TS_idx].append(TS.eta[0])
                    self.TS_traj_E[TS_idx].append(TS.eta[1])
                    self.TS_traj_h[TS_idx].append(TS.eta[2])
        
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
        if self.step_cnt % 1 == 0: 

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
               
                rect = self._get_rect(E = E0, N = N0, width = agent.width, length = agent.length, heading = head0,
                                      linewidth=1, edgecolor=col, facecolor='none')
                self.ax0.add_patch(rect)

                # add jets according to COLREGS
                for COLREG_deg in [5, 355]:
                    self.ax0 = self._plot_jet(axis = self.ax0, E=E0, N=N0, l = self.sight, 
                                              angle = head0 + dtr(COLREG_deg), color=col, alpha=0.3)

                self.ax0.scatter(self.goals[idx]["E"], self.goals[idx]["N"], color=col)
                circ = patches.Circle((self.goals[idx]["E"], self.goals[idx]["N"]), radius=self.goal_reach_dist, edgecolor=col, facecolor='none', alpha=0.3)
                self.ax0.add_patch(circ)

            plt.pause(0.001)
